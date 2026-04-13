"""
Three evaluation tiers matching the dissertation plan:
    1. retrieval_eval  — did we find the right source pages?
    2. answer_eval     — does the generated answer match the gold answer?
    3. grounding_eval  — is the answer actually supported by retrieved evidence?

Uses only open-source metrics (ROUGE, exact match) 
"""

import logging
import re
from typing import Optional

from rouge_score import rouge_scorer 

logger = logging.getLogger(__name__)


# 1. Retrieval Evaluation

def retrieval_eval(
    retrieved_chunks: list[dict],
    gold_doc_name: str,
    gold_page_number: Optional[int] = None,
    top_k: int = 5,
) -> dict:
    """
    Check whether the correct document (and ideally page) was retrieved.

    Args:
        retrieved_chunks:  list of chunk dicts from the retriever
        gold_doc_name:     expected source document name (from FinanceBench)
        gold_page_number:  expected page number (from FinanceBench evidence_page_num)
        top_k:             how many top results to check

    Returns:
        dict with doc_match, page_match, rank_of_match metrics
    """
    results = retrieved_chunks[:top_k]

    doc_names = [r.get("doc_name", "").lower() for r in results]
    page_numbers = [r.get("page_number", -1) for r in results]
    gold_name_lower = gold_doc_name.lower()

    # Check if gold document appears anywhere in top-k results
    doc_match = any(gold_name_lower in name for name in doc_names)

    # Check if gold page also appears (stricter criterion)
    page_match = False
    if gold_page_number is not None:
        page_match = any(
            gold_name_lower in doc_names[i] and abs(page_numbers[i] - gold_page_number) <= 1
            for i in range(len(results))
        )

    # Rank of first correct document (1-indexed; 0 if not found)
    rank = 0
    for i, name in enumerate(doc_names, start=1):
        if gold_name_lower in name:
            rank = i
            break

    # MRR (Mean Reciprocal Rank) component for this single query
    mrr = 1 / rank if rank > 0 else 0.0

    return {
        "doc_match": doc_match,
        "page_match": page_match,
        "rank": rank,
        "mrr": mrr,
        "top_k": top_k,
    }


# 2. Answer Evaluation

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation — used for exact match comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)      # collapse whitespace
    return text


def answer_eval(generated_answer: str, gold_answer: str) -> dict:
    """
    Compare generated answer to gold answer using exact match and ROUGE-L.

    Args:
        generated_answer: answer produced by the RAG pipeline
        gold_answer:      human-annotated correct answer from FinanceBench

    Returns:
        dict with exact_match (bool) and rouge_l (float 0–1)
    """
    gen_norm = _normalise(generated_answer)
    gold_norm = _normalise(gold_answer)

    # Exact match: does the generated answer contain the gold answer?
    exact_match = gold_norm in gen_norm or gen_norm == gold_norm

    # ROUGE-L: longest common subsequence F1-score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(gold_norm, gen_norm)
    rouge_l = scores["rougeL"].fmeasure  # F1 score

    return {
        "exact_match": exact_match,
        "rouge_l": round(rouge_l, 4),
    }


# 3. Grounding Evaluation

def grounding_eval(answer: str, evidence_chunks: list[dict]) -> dict:
    """
    Check whether the answer is grounded in the retrieved evidence.
    A simple heuristic: does the answer share key financial terms with the evidence?

    For a more rigorous grounding check, use an LLM judge (see llm_judge below).

    Returns:
        dict with grounded (bool) and overlap_score (float 0–1)
    """
    if not evidence_chunks or not answer:
        return {"grounded": False, "overlap_score": 0.0}

    # Extract meaningful words from the answer (skip stop words)
    stop_words = {"the", "a", "an", "is", "was", "in", "of", "and", "or", "for", "to", "that"}
    answer_words = set(
        w.lower() for w in re.findall(r"\b\w+\b", answer) if w.lower() not in stop_words
    )

    # Combine all evidence text into one pool
    evidence_text = " ".join(c.get("text", "") for c in evidence_chunks)
    evidence_words = set(
        w.lower() for w in re.findall(r"\b\w+\b", evidence_text) if w.lower() not in stop_words
    )

    if not answer_words:
        return {"grounded": False, "overlap_score": 0.0}

    overlap = len(answer_words & evidence_words) / len(answer_words)
    # Threshold: if >40% of answer words appear in evidence → grounded
    grounded = overlap > 0.4

    return {"grounded": grounded, "overlap_score": round(overlap, 4)}


# Batch evaluation

def evaluate_batch(results: list[dict]) -> dict:
    """
    Aggregate metrics over a list of individual evaluation result dicts.

    Args:
        results: list of dicts, each with keys:
                 doc_match, page_match, mrr, exact_match, rouge_l, grounded

    Returns:
        dict of mean metrics across the batch
    """
    if not results:
        return {}

    def mean(key: str) -> float:
        values = [r[key] for r in results if key in r]
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "n": len(results),
        "doc_match_rate": mean("doc_match"),
        "page_match_rate": mean("page_match"),
        "mrr": mean("mrr"),
        "exact_match_rate": mean("exact_match"),
        "rouge_l_mean": mean("rouge_l"),
        "grounding_rate": mean("grounded"),
    }
