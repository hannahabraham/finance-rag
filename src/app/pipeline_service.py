"""
Streaming wrapper around the LangGraph pipeline.
Yields an event per agent step so the React frontend can render the
pipeline flow in real time instead of a single blocking spinner.

Event schema (all JSON-serialisable):
    {"type": "step_complete", "step": "retrieval", "iteration": 1,
     "data": {...}, "duration_ms": 1234}
    {"type": "iteration_start", "iteration": 2}
    {"type": "done", "result": {...final_output...}}
    {"type": "error", "message": "..."}
"""

import logging
import time
from typing import Iterator

logger = logging.getLogger(__name__)


# Module-level cache (same pattern as the old Gradio app)
_vector_store = None
_bm25_index = None
_embeddings = None
_retrievers_by_mode: dict = {}
_graphs_by_mode: dict = {}


def initialise_pipeline(retrieval_mode: str = "hybrid"):
    """Load heavy components once; cache retriever + graph per mode."""
    global _vector_store, _bm25_index, _embeddings

    from src.retrieval.embedder import get_embedding_model
    from src.retrieval.vector_store import (
        load_vector_store,
        vector_store_exists,
        get_all_documents,
    )
    from src.retrieval.retriever import dense_retrieve, BM25Index, hybrid_retrieve
    from src.agents.graph import build_graph

    if _vector_store is None:
        if not vector_store_exists():
            raise RuntimeError(
                "Vector store not found. Run:\n  python scripts/build_index.py"
            )
        _embeddings = get_embedding_model()
        _vector_store = load_vector_store(_embeddings)
        _bm25_index = BM25Index(get_all_documents(_vector_store))
        logger.info("Heavy pipeline components loaded (one-time)")

    if retrieval_mode not in _retrievers_by_mode:
        vs = _vector_store
        bm25 = _bm25_index

        def make_retriever(mode: str):
            def retriever_fn(query: str, company_filter=None):
                if mode == "dense":
                    return dense_retrieve(vs, query, company_filter=company_filter)
                if mode == "bm25":
                    return bm25.search(query, company_filter=company_filter)
                return hybrid_retrieve(vs, bm25, query, company_filter=company_filter)
            return retriever_fn

        fn = make_retriever(retrieval_mode)
        _retrievers_by_mode[retrieval_mode] = fn
        _graphs_by_mode[retrieval_mode] = build_graph(fn)
        logger.info(f"Pipeline cached for mode={retrieval_mode}")

    return _retrievers_by_mode[retrieval_mode], _graphs_by_mode[retrieval_mode]


def _sanitize(obj):
    """Make nested values JSON-serialisable (numpy floats, etc.)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _summarise_node_output(node_name: str, output: dict) -> dict:
    """
    Shape each node's raw output into a UI-friendly summary.
    Heavy payloads (full chunk lists) are trimmed to what's needed for display.
    """
    if node_name == "query_understanding":
        return {
            "company": output.get("company", ""),
            "doc_period": output.get("doc_period", ""),
            "intent": output.get("intent", ""),
            "parsed_question": output.get("parsed_question", ""),
        }
    if node_name == "retrieval":
        chunks = output.get("retrieved_chunks") or []
        return {
            "chunk_count": len(chunks),
            "top_chunks": _sanitize([
                {
                    "doc_name": c.get("doc_name", ""),
                    "page_number": c.get("page_number", ""),
                    "company": c.get("company", ""),
                    "score": c.get("score", 0),
                    "text": (c.get("text", "")[:220] + "…") if len(c.get("text", "")) > 220 else c.get("text", ""),
                }
                for c in chunks[:5]
            ]),
        }
    if node_name == "evidence_verification":
        chunks = output.get("verified_chunks") or []
        return {
            "verified_count": len(chunks),
            "kept_chunks": _sanitize([
                {
                    "doc_name": c.get("doc_name", ""),
                    "page_number": c.get("page_number", ""),
                    "text": (c.get("text", "")[:220] + "…") if len(c.get("text", "")) > 220 else c.get("text", ""),
                }
                for c in chunks[:5]
            ]),
        }
    if node_name == "answer_writing":
        return {
            "answer": output.get("answer", ""),
            "explanation": output.get("explanation", ""),
            "confidence": output.get("confidence", ""),
            "sources": _sanitize(output.get("sources", [])),
        }
    if node_name == "critic":
        return {
            "critique": output.get("critique", ""),
            "needs_retry": bool(output.get("needs_retry", False)),
            "retry_count": output.get("retry_count", 0),
        }
    return _sanitize(output)


def stream_multiagent(question: str, retrieval_mode: str) -> Iterator[dict]:
    """Run the multi-agent graph, yielding one event per node completion."""
    retriever_fn, graph_app = initialise_pipeline(retrieval_mode)

    initial_state = {
        "question": question,
        "company": None,
        "doc_period": None,
        "intent": None,
        "parsed_question": None,
        "retrieved_chunks": None,
        "verified_chunks": None,
        "answer": None,
        "explanation": None,
        "sources": None,
        "confidence": None,
        "critique": None,
        "needs_retry": False,
        "retry_count": 0,
        "final_output": None,
    }

    yield {"type": "iteration_start", "iteration": 1}

    seen_in_iteration: set[str] = set()
    iteration = 1
    last_emit_time = time.time()
    merged_state = dict(initial_state)

    # graph_app.stream yields {node_name: partial_update_dict} per node
    for update in graph_app.stream(initial_state):
        for node_name, node_output in update.items():
            if node_name in seen_in_iteration:
                iteration += 1
                seen_in_iteration = set()
                yield {"type": "iteration_start", "iteration": iteration}
            seen_in_iteration.add(node_name)

            now = time.time()
            duration_ms = int((now - last_emit_time) * 1000)
            last_emit_time = now

            merged_state.update(node_output)

            yield {
                "type": "step_complete",
                "step": node_name,
                "iteration": iteration,
                "duration_ms": duration_ms,
                "data": _summarise_node_output(node_name, node_output),
            }

    final_output = merged_state.get("final_output") or {
        "question": question,
        "answer": "Pipeline did not produce an answer.",
        "confidence": "Low",
        "sources": [],
        "evidence_snippets": [],
    }
    # The stream's final_output already embeds full chunks; the frontend
    # only needs the display-ready projection.
    display_final = {
        "question": final_output.get("question", ""),
        "company": final_output.get("company", ""),
        "answer": final_output.get("answer", ""),
        "explanation": final_output.get("explanation", ""),
        "confidence": final_output.get("confidence", "Low"),
        "sources": _sanitize(final_output.get("sources", [])),
        "evidence_snippets": _sanitize(final_output.get("evidence_snippets", [])),
        "critique": final_output.get("critique", ""),
        "iterations": iteration,
    }
    yield {"type": "done", "result": display_final}


def stream_baseline(question: str, retrieval_mode: str) -> Iterator[dict]:
    """Simple single-chain retrieval → generation, streamed in two steps."""
    from src.app.baseline_rag import _format_context, ANSWER_PROMPT_TEMPLATE
    from src.agents.llm import generate

    retriever_fn, _ = initialise_pipeline(retrieval_mode)

    t0 = time.time()
    chunks = retriever_fn(query=question, company_filter=None)
    yield {
        "type": "step_complete",
        "step": "retrieval",
        "iteration": 1,
        "duration_ms": int((time.time() - t0) * 1000),
        "data": {
            "chunk_count": len(chunks),
            "top_chunks": _sanitize([
                {
                    "doc_name": c.get("doc_name", ""),
                    "page_number": c.get("page_number", ""),
                    "company": c.get("company", ""),
                    "score": c.get("score", 0),
                    "text": (c.get("text", "")[:220] + "…") if len(c.get("text", "")) > 220 else c.get("text", ""),
                }
                for c in chunks[:5]
            ]),
        },
    }

    if not chunks:
        yield {
            "type": "done",
            "result": {
                "question": question,
                "answer": "No relevant documents found.",
                "confidence": "Low",
                "sources": [],
                "evidence_snippets": [],
            },
        }
        return

    t1 = time.time()
    context = _format_context(chunks[:5])
    prompt = ANSWER_PROMPT_TEMPLATE.format(question=question, context=context)
    answer_text = generate(prompt, max_tokens=400)
    duration = int((time.time() - t1) * 1000)

    sources = [
        {"doc_name": c.get("doc_name", ""), "page_number": c.get("page_number", "")}
        for c in chunks[:3]
    ]
    evidence = [
        {
            "text": (c.get("text", "")[:300] + "…") if len(c.get("text", "")) > 300 else c.get("text", ""),
            "doc_name": c.get("doc_name", ""),
            "page_number": c.get("page_number", ""),
        }
        for c in chunks[:3]
    ]

    yield {
        "type": "step_complete",
        "step": "answer_writing",
        "iteration": 1,
        "duration_ms": duration,
        "data": {"answer": answer_text, "confidence": "Medium"},
    }
    yield {
        "type": "done",
        "result": {
            "question": question,
            "answer": answer_text,
            "explanation": "",
            "confidence": "Medium",
            "sources": _sanitize(sources),
            "evidence_snippets": _sanitize(evidence),
            "iterations": 1,
        },
    }
