"""
Simple single-chain RAG pipeline (no agents, no graph).
Used as Experiment 1 baseline to compare against the multi-agent system.

Flow: question → retrieve → generate answer
Everything happens in one function call — fast but no verification or self-correction.
"""

import logging
from typing import Callable

logger = logging.getLogger(__name__)


ANSWER_PROMPT_TEMPLATE = """You are a financial analyst assistant. Answer the question strictly based on the context below.
If the context does not contain enough information, say "Insufficient evidence."

Question: {question}

Context:
{context}

Respond with:
Answer: <your answer>
Source: <document name and page number>
Confidence: <High / Medium / Low>"""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        src = f"[{chunk.get('doc_name','?')}, page {chunk.get('page_number','?')}]"
        lines.append(f"[{i}] {src}\n{chunk.get('text','').strip()}")
    return "\n\n".join(lines)


def baseline_answer(question: str, retriever_fn: Callable) -> dict:
    """
    Single-chain RAG: retrieve → generate. No agents, no critique.

    Args:
        question:     user's question
        retriever_fn: callable(query, company_filter=None) → list[dict]

    Returns:
        dict with answer, sources, confidence
    """
    from src.agents.llm import generate  # local import to keep startup fast

    # Step 1: retrieve
    chunks = retriever_fn(query=question, company_filter=None)

    if not chunks:
        return {
            "question": question,
            "answer": "No relevant documents found.",
            "sources": [],
            "confidence": "Low",
            "method": "baseline",
        }

    # Step 2: build prompt and generate
    context = _format_context(chunks[:5])
    prompt = ANSWER_PROMPT_TEMPLATE.format(question=question, context=context)

    raw_response = generate(prompt, max_tokens=400)

    # Step 3: parse simple text response (no JSON for baseline simplicity)
    sources = [
        {"doc_name": c.get("doc_name", ""), "page_number": c.get("page_number", "")}
        for c in chunks[:3]
    ]

    return {
        "question": question,
        "answer": raw_response,
        "sources": sources,
        "confidence": "Medium",
        "method": "baseline",
    }
