"""
scripts/run_evaluation.py
--------------------------
Run the full evaluation suite on the 150 FinanceBench annotated questions.
Compares baseline RAG vs multi-agent RAG across retrieval, answer, and grounding metrics.

Usage:
    python scripts/run_evaluation.py                    # evaluate all 150 questions
    python scripts/run_evaluation.py --limit 20         # quick run on 20 questions
    python scripts/run_evaluation.py --mode baseline    # baseline only
    python scripts/run_evaluation.py --retrieval bm25   # change retrieval strategy

Results are saved to results/evaluation_TIMESTAMP.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion.dataset_loader import build_merged_df
from src.retrieval.embedder import get_embedding_model
from src.retrieval.vector_store import load_vector_store, get_all_documents
from src.retrieval.retriever import dense_retrieve, BM25Index, hybrid_retrieve
from src.evaluation.metrics import retrieval_eval, answer_eval, grounding_eval, evaluate_batch
from src.app.baseline_rag import baseline_answer
from src.agents.graph import build_graph, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_evaluation")


def build_retriever(vector_store, bm25_index, mode: str):
    """Factory: returns a retriever function for the given mode."""
    def fn(query: str, company_filter=None):
        if mode == "dense":
            return dense_retrieve(vector_store, query, company_filter=company_filter)
        elif mode == "bm25":
            return bm25_index.search(query, company_filter=company_filter)
        else:
            return hybrid_retrieve(vector_store, bm25_index, query, company_filter=company_filter)
    return fn


def evaluate_single(
    row: dict,
    retriever_fn,
    graph_app,
    pipeline_mode: str,
) -> dict:
    """
    Run the pipeline on one question and compute all metrics.

    Args:
        row:           merged DataFrame row (question + gold answer + metadata)
        retriever_fn:  retrieval function
        graph_app:     compiled LangGraph (None for baseline mode)
        pipeline_mode: "baseline" or "multiagent"

    Returns:
        flat dict of all metric values for this question
    """
    question = row.get("question", "")
    gold_answer = str(row.get("answer", ""))
    gold_doc_name = str(row.get("doc_name", ""))
    gold_page = row.get("evidence_page_num", None)
    if gold_page is not None:
        try:
            gold_page = int(gold_page)
        except (ValueError, TypeError):
            gold_page = None

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        if pipeline_mode == "baseline":
            result = baseline_answer(question, retriever_fn)
            generated_answer = result.get("answer", "")
            retrieved_chunks = result.get("retrieved_chunks", [])
            evidence_chunks = retrieved_chunks[:5]
        else:
            result = run_pipeline(graph_app, question)
            generated_answer = result.get("answer", "")
            # Use full (non-truncated) chunks exposed by the pipeline:
            # retrieved_chunks → retrieval_eval (was the correct doc fetched?)
            # verified_chunks  → grounding_eval (is the answer supported?)
            retrieved_chunks = result.get("retrieved_chunks", [])
            evidence_chunks = result.get("verified_chunks", [])
    except Exception as exc:
        logger.warning(f"Pipeline error for '{question[:60]}': {exc}")
        return {"question": question, "error": str(exc)}

    # ── Compute metrics ───────────────────────────────────────────────────────
    r_metrics = retrieval_eval(retrieved_chunks, gold_doc_name, gold_page)
    a_metrics = answer_eval(generated_answer, gold_answer)
    g_metrics = grounding_eval(generated_answer, evidence_chunks)

    return {
        "question": question[:100],
        "gold_answer": gold_answer[:200],
        "generated_answer": generated_answer[:200],
        "doc_name": gold_doc_name,
        # retrieval metrics
        "doc_match": r_metrics["doc_match"],
        "page_match": r_metrics["page_match"],
        "mrr": r_metrics["mrr"],
        # answer metrics
        "exact_match": a_metrics["exact_match"],
        "rouge_l": a_metrics["rouge_l"],
        # grounding metrics
        "grounded": g_metrics["grounded"],
        "overlap_score": g_metrics["overlap_score"],
    }


def main(limit: int, pipeline_mode: str, retrieval_mode: str):
    logger.info("=" * 60)
    logger.info(f"Evaluation | pipeline={pipeline_mode} | retrieval={retrieval_mode} | limit={limit}")
    logger.info("=" * 60)

    # ── Load dataset ──────────────────────────────────────────────────────────
    if not settings.QUESTIONS_FILE.exists():
        logger.error(f"Questions file not found: {settings.QUESTIONS_FILE}")
        sys.exit(1)

    df = build_merged_df()
    if limit:
        df = df.head(limit)
    logger.info(f"Evaluating on {len(df)} questions")

    # ── Load retrieval components ─────────────────────────────────────────────
    embeddings = get_embedding_model()
    vector_store = load_vector_store(embeddings)

    # Build BM25 corpus from FAISS docstore via the public helper
    bm25_index = BM25Index(get_all_documents(vector_store))
    retriever_fn = build_retriever(vector_store, bm25_index, retrieval_mode)

    # ── Build graph (only for multi-agent mode) ───────────────────────────────
    graph_app = None
    if pipeline_mode == "multiagent":
        graph_app = build_graph(retriever_fn)

    # ── Run evaluation loop ───────────────────────────────────────────────────
    all_results = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        logger.info(f"[{i}/{len(df)}] {row.get('question', '')[:70]}")
        result = evaluate_single(row.to_dict(), retriever_fn, graph_app, pipeline_mode)
        all_results.append(result)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    # Filter out error rows before aggregation
    valid_results = [r for r in all_results if "error" not in r]
    aggregated = evaluate_batch(valid_results)

    logger.info("\n" + "=" * 40)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 40)
    for key, value in aggregated.items():
        logger.info(f"  {key:<25} {value}")
    logger.info("=" * 40)

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"eval_{pipeline_mode}_{retrieval_mode}_{timestamp}.json"

    output = {
        "config": {
            "pipeline_mode": pipeline_mode,
            "retrieval_mode": retrieval_mode,
            "n_questions": len(df),
            "timestamp": timestamp,
        },
        "aggregated_metrics": aggregated,
        "per_question_results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n✅ Results saved to {output_path}")
    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Financial RAG pipeline")
    parser.add_argument("--limit", type=int, default=0, help="Max questions to evaluate (0 = all)")
    parser.add_argument("--mode", choices=["baseline", "multiagent"], default="multiagent")
    parser.add_argument("--retrieval", choices=["dense", "bm25", "hybrid"], default="hybrid")
    args = parser.parse_args()

    main(
        limit=args.limit or None,
        pipeline_mode=args.mode,
        retrieval_mode=args.retrieval,
    )
