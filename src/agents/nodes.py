"""
Implements the five agent nodes that make up the LangGraph pipeline.
Each function takes the shared RAGState and returns a dict of updated fields.
LangGraph merges those updates back into the state automatically.
"""

import json
import logging
import re
from typing import Any

from src.agents.llm import generate
from src.agents.state import RAGState

logger = logging.getLogger(__name__)


# Prompt helpers

def _format_chunks_as_context(chunks: list[dict]) -> str:
    """Convert retrieved chunks into a numbered context block for prompts."""
    if not chunks:
        return "No context available."
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = f"[{chunk.get('doc_name', 'unknown')}, page {chunk.get('page_number', '?')}]"
        lines.append(f"[{i}] {source}\n{chunk.get('text', '').strip()}")
    return "\n\n".join(lines)


def _extract_json_from_response(text: str) -> dict:
    """
    Parse a JSON block from LLM output.
    LLMs sometimes wrap JSON in ```json ... ``` — strip fences before parsing.
    Falls back to an empty dict on failure.
    """
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract anything that looks like a JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# Node 1: Query Understanding Agent

def query_understanding_node(state: RAGState) -> dict:
    """
    Parse the user's question to extract structured financial metadata.
    This helps the retrieval agent narrow its search to the right company and year.

    Extracts: company name, time period, financial concept (intent), cleaned question.
    """
    question = state["question"]

    prompt = f"""You are a financial analyst assistant. Extract structured metadata from this question.
Respond ONLY with valid JSON — no explanation, no markdown fences.

Question: {question}

Return exactly this JSON structure:
{{
  "company": "<company name or empty string>",
  "doc_period": "<4-digit year or quarter like 2022 or Q3-2022 or empty string>",
  "intent": "<financial concept being asked about, e.g. operating income, risk factors>",
  "parsed_question": "<cleaned version of the question>"
}}"""

    response = generate(prompt, max_tokens=256)
    parsed = _extract_json_from_response(response)

    logger.info(f"Query understanding: {parsed}")
    return {
        "company": parsed.get("company", ""),
        "doc_period": parsed.get("doc_period", ""),
        "intent": parsed.get("intent", ""),
        "parsed_question": parsed.get("parsed_question", question),
    }


# Node 2: Retrieval Agent 

def retrieval_node(state: RAGState, retriever_fn) -> dict:
    """
    Retrieve the most relevant chunks from the vector store.

    Uses the company filter from query understanding to reduce noise.
    retriever_fn is injected at graph-build time (dependency injection pattern)
    so we can swap dense / BM25 / hybrid without changing this node.

    Args:
        state: shared RAGState
        retriever_fn: callable(query, company_filter) → list[dict]
    """
    query = state.get("parsed_question") or state["question"]
    company = state.get("company") or None

    # On retry, expand the search by removing the company filter
    retry_count = state.get("retry_count", 0)
    if retry_count and retry_count > 0:
        logger.info("Retry detected — removing company filter to broaden search")
        company = None

    chunks = retriever_fn(query=query, company_filter=company)

    logger.info(f"Retrieved {len(chunks)} chunks")
    return {"retrieved_chunks": chunks}


# Node 3: Evidence Verification Agent

def evidence_verification_node(state: RAGState) -> dict:
    """
    Filter and score the retrieved chunks.
    Removes chunks that are clearly irrelevant to the question.
    This reduces hallucination by keeping only well-supported evidence.
    """
    question = state.get("parsed_question") or state["question"]
    chunks = state.get("retrieved_chunks") or []

    if not chunks:
        return {"verified_chunks": []}

    context = _format_chunks_as_context(chunks[:8])  # send max 8 chunks to LLM

    prompt = f"""You are a financial evidence verifier. A question was asked and some document chunks were retrieved.
Your job: identify which chunks actually contain evidence that helps answer the question.

Question: {question}

Retrieved chunks:
{context}

Respond ONLY with JSON like this:
{{
  "relevant_indices": [1, 3, 4],
  "reasoning": "chunk 1 mentions operating income, chunk 3 gives the 2022 figure"
}}
Use 1-indexed positions matching the chunk numbers above. Only include clearly relevant chunks."""

    response = generate(prompt, max_tokens=256)
    parsed = _extract_json_from_response(response)

    relevant_indices = parsed.get("relevant_indices", list(range(1, len(chunks) + 1)))

    # Convert to 0-indexed and select only relevant chunks
    verified = [chunks[i - 1] for i in relevant_indices if 1 <= i <= len(chunks)]

    # Fallback: if LLM returned nothing, keep all chunks
    if not verified:
        verified = chunks

    logger.info(f"Evidence verification: {len(chunks)} → {len(verified)} chunks")
    return {"verified_chunks": verified}


# Node 4: Answer Writer Agent

def answer_writing_node(state: RAGState) -> dict:
    """
    Generate a structured, evidence-grounded answer from verified chunks.

    The prompt enforces: answer only from evidence, cite sources, admit uncertainty.
    This is the core anti-hallucination instruction.
    """
    question = state.get("parsed_question") or state["question"]
    chunks = state.get("verified_chunks") or state.get("retrieved_chunks") or []

    context = _format_chunks_as_context(chunks)

    prompt = f"""You are a precise financial analyst. Answer the question ONLY using the provided evidence.
If the evidence does not fully support an answer, say "Insufficient evidence in the retrieved documents."

Question: {question}

Evidence:
{context}

Respond with JSON only:
{{
  "answer": "<direct factual answer>",
  "explanation": "<1-2 sentences explaining which evidence supports the answer>",
  "confidence": "<High|Medium|Low>",
  "sources": [
    {{"doc_name": "DOC_NAME", "page_number": N}},
    ...
  ]
}}"""

    response = generate(prompt, max_tokens=512)
    parsed = _extract_json_from_response(response)

    # Defensive defaults if LLM fails to produce valid JSON
    return {
        "answer": parsed.get("answer", "Could not generate answer."),
        "explanation": parsed.get("explanation", ""),
        "confidence": parsed.get("confidence", "Low"),
        "sources": parsed.get("sources", []),
    }


# Node 5: Critic Agent

def critic_node(state: RAGState) -> dict:
    """
    Review the generated answer and decide if it needs improvement.

    If the answer is weak (not grounded, incomplete, or contradicts evidence),
    sets needs_retry=True to loop back to retrieval with a broader search.
    Caps retries at 2 to prevent infinite loops.
    """
    MAX_RETRIES = 2

    # Safety check: never loop more than MAX_RETRIES times
    retry_count = state.get("retry_count", 0) or 0
    if retry_count >= MAX_RETRIES:
        logger.warning("Max retries reached — accepting current answer")
        return {"needs_retry": False, "critique": "Max retries reached.", "final_output": _build_final_output(state)}

    question = state.get("parsed_question") or state["question"]
    answer = state.get("answer", "")
    explanation = state.get("explanation", "")
    chunks = state.get("verified_chunks") or state.get("retrieved_chunks") or []
    context = _format_chunks_as_context(chunks[:4])  # brief context for critic

    prompt = f"""You are a financial QA critic. Evaluate this answer strictly.

Question: {question}
Answer: {answer}
Explanation: {explanation}
Supporting evidence (excerpt):
{context}

Respond with JSON only:
{{
  "is_grounded": <true if the answer is directly supported by the evidence above, else false>,
  "is_complete": <true if the answer actually addresses the question>,
  "issues": "<describe any problems or empty string if none>",
  "verdict": "<ACCEPT or RETRY>"
}}"""

    response = generate(prompt, max_tokens=256)
    parsed = _extract_json_from_response(response)

    verdict = parsed.get("verdict", "ACCEPT").upper()
    needs_retry = (verdict == "RETRY")

    logger.info(f"Critic verdict: {verdict} | issues: {parsed.get('issues', '')}")

    result = {
        "critique": parsed.get("issues", ""),
        "needs_retry": needs_retry,
        "retry_count": retry_count + (1 if needs_retry else 0),
    }

    # If accepted, assemble the final structured output
    if not needs_retry:
        result["final_output"] = _build_final_output(state)

    return result


# Final output assembler

def _build_final_output(state: RAGState) -> dict:
    """
    Assemble the structured output dict returned to the UI.
    Deduplicate sources and format everything cleanly.
    """
    # Deduplicate sources by (doc_name, page_number)
    seen = set()
    unique_sources = []
    for src in (state.get("sources") or []):
        key = (src.get("doc_name", ""), src.get("page_number", 0))
        if key not in seen:
            seen.add(key)
            unique_sources.append(src)

    # Build evidence snippets list
    evidence_snippets = [
        {
            "text": c.get("text", "")[:300] + "…",
            "doc_name": c.get("doc_name", ""),
            "page_number": c.get("page_number", ""),
        }
        for c in (state.get("verified_chunks") or [])[:3]
    ]

    return {
        "question": state.get("question", ""),
        "company": state.get("company", ""),
        "answer": state.get("answer", ""),
        "explanation": state.get("explanation", ""),
        "confidence": state.get("confidence", "Low"),
        "sources": unique_sources,
        "evidence_snippets": evidence_snippets,
        "critique": state.get("critique", ""),
    }
