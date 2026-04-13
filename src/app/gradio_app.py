"""
Gradio web interface for the Financial Research Assistant.

Gradio is ideal here because:
    - Zero-boilerplate UI (built for ML demos)
    - One-click deploy to HuggingFace Spaces (free hosting)
    - Runs locally on Mac with `python -m src.app.gradio_app`

The UI offers:
    - Question input
    - Mode selector (multi-agent vs baseline)
    - Retrieval strategy selector (dense / BM25 / hybrid)
    - Structured output display with sources and evidence
"""

import logging
import os
from pathlib import Path

import gradio as gr  # UI framework — pip install gradio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Log configuration status
from src.config import settings
logger.info(f"LangSmith tracing enabled: {settings.LANGSMITH_TRACING}")
logger.info(f"Log level: {settings.LOG_LEVEL}")


# Pipeline initialisation 

def _initialise_pipeline(retrieval_mode: str = "hybrid"):
    """
    Load vector store, embedding model, BM25 index, and LangGraph on first call.
    Cached globally so subsequent calls are instant.
    """
    global _retriever_fn, _graph_app, _bm25_index

    from src.config import settings
    from src.retrieval.embedder import get_embedding_model
    from src.retrieval.vector_store import load_vector_store, vector_store_exists
    from src.retrieval.retriever import dense_retrieve, BM25Index, hybrid_retrieve
    from src.agents.graph import build_graph

    if not vector_store_exists():
        raise RuntimeError(
            "Vector store not found. Please run:\n  python scripts/build_index.py"
        )

    embeddings = get_embedding_model()
    vector_store = load_vector_store(embeddings)

    # Reconstruct a lightweight BM25 index from the FAISS docstore
    # (we pull all stored documents out of FAISS to build BM25 corpus)
    all_docs = []
    for doc_id, doc in vector_store.docstore._dict.items():
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        all_docs.append({"text": doc.page_content, **meta})

    _bm25_index = BM25Index(all_docs)

    # Build the retriever function based on selected mode
    def make_retriever(mode: str):
        def retriever_fn(query: str, company_filter=None):
            if mode == "dense":
                return dense_retrieve(vector_store, query, company_filter=company_filter)
            elif mode == "bm25":
                return _bm25_index.search(query, company_filter=company_filter)
            else:  # hybrid (default)
                return hybrid_retrieve(vector_store, _bm25_index, query, company_filter=company_filter)
        return retriever_fn

    _retriever_fn = make_retriever(retrieval_mode)
    _graph_app = build_graph(_retriever_fn)

    logger.info(f"Pipeline initialised with {retrieval_mode} retrieval")
    return _retriever_fn, _graph_app


# Main answer function

def answer_question(
    question: str,
    mode: str,
    retrieval_strategy: str,
    progress=gr.Progress(),
) -> tuple[str, str, str, str, str]:
    """
    Called by Gradio when the user clicks Submit.

    Args:
        question:           user's natural language question
        mode:               "Multi-Agent RAG" or "Baseline RAG"
        retrieval_strategy: "hybrid" | "dense" | "bm25"

    Returns:
        Tuple of (answer, sources_text, evidence_text, confidence, metadata)
        — one value per Gradio output component
    """
    if not question.strip():
        return "Please enter a question.", "", "", "", ""

    try:
        progress(0.1, desc="Initialising pipeline …")
        retriever_fn, graph_app = _initialise_pipeline(retrieval_strategy)

        if mode == "Baseline RAG":
            progress(0.5, desc="Generating answer (baseline) …")
            from src.app.baseline_rag import baseline_answer
            result = baseline_answer(question, retriever_fn)
        else:
            progress(0.3, desc="Understanding question …")
            from src.agents.graph import run_pipeline
            progress(0.6, desc="Retrieving and verifying evidence …")
            result = run_pipeline(graph_app, question)

        progress(0.9, desc="Formatting output …")

        # ── Format answer ────────────────────────────────────────────────────
        answer_text = str(result.get("answer", "No answer generated."))
        if result.get("explanation"):
            answer_text += f"\n\n**Explanation:** {result['explanation']}"

        # ── Format sources ───────────────────────────────────────────────────
        sources = result.get("sources", [])
        if sources:
            sources_lines = [f"• {s.get('doc_name', '?')}, page {s.get('page_number', '?')}" for s in sources]
            sources_text = "\n".join(sources_lines)
        else:
            sources_text = "No sources available."

        # ── Format evidence snippets ──────────────────────────────────────────
        snippets = result.get("evidence_snippets", [])
        if snippets:
            ev_lines = []
            for i, s in enumerate(snippets, 1):
                ev_lines.append(f"**[{i}] {s.get('doc_name','?')}, p.{s.get('page_number','?')}**\n{s.get('text','')}")
            evidence_text = "\n\n---\n\n".join(ev_lines)
        else:
            evidence_text = "No evidence snippets available."

        # ── Confidence ───────────────────────────────────────────────────────
        confidence = result.get("confidence", "—")

        # ── Metadata ─────────────────────────────────────────────────────────
        meta_parts = []
        if result.get("company"):
            meta_parts.append(f"**Company detected:** {result['company']}")
        if result.get("critique"):
            meta_parts.append(f"**Critic notes:** {result['critique']}")
        metadata = "\n".join(meta_parts) if meta_parts else "—"

        progress(1.0, desc="Done!")
        return answer_text, sources_text, evidence_text, confidence, metadata

    except FileNotFoundError as exc:
        return str(exc), "", "", "", ""
    except Exception as exc:
        logger.exception("Pipeline error")
        return f"Error: {exc}", "", "", "", ""


# Gradio UI layout

def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks UI."""

    with gr.Blocks(
        title="Financial Research Assistant",
        theme=gr.themes.Soft(),
        css="""
        .answer-box { font-size: 1.05rem; }
        .confidence-badge { font-weight: bold; font-size: 1.1rem; }
        """,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 📊 Financial Research Assistant
            **Multi-Agent RAG over FinanceBench Documents**
            Ask questions about public company filings (10-K, 10-Q). The system retrieves
            relevant evidence from SEC reports and generates grounded, cited answers.
            """,
        )

        # ── Input row ────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. What was Amazon's operating income in 2022?",
                    lines=2,
                )
            with gr.Column(scale=1):
                mode_selector = gr.Radio(
                    choices=["Multi-Agent RAG", "Baseline RAG"],
                    value="Multi-Agent RAG",
                    label="Pipeline Mode",
                )
                retrieval_selector = gr.Dropdown(
                    choices=["hybrid", "dense", "bm25"],
                    value="hybrid",
                    label="Retrieval Strategy",
                )

        submit_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")

        # ── Output section ────────────────────────────────────────────────────
        with gr.Row():
            confidence_output = gr.Textbox(label="Confidence", elem_classes=["confidence-badge"])
            metadata_output = gr.Markdown(label="Pipeline Metadata")

        answer_output = gr.Markdown(label="Answer & Explanation", elem_classes=["answer-box"])

        with gr.Accordion("📚 Sources", open=True):
            sources_output = gr.Markdown()

        with gr.Accordion("🔍 Retrieved Evidence Snippets", open=False):
            evidence_output = gr.Markdown()

        # ── Example questions ─────────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["What was Amazon's operating income in 2022?", "Multi-Agent RAG", "hybrid"],
                ["What risk factors did Adobe disclose in their latest 10-K?", "Multi-Agent RAG", "hybrid"],
                ["How did 3M describe litigation exposure?", "Baseline RAG", "dense"],
            ],
            inputs=[question_input, mode_selector, retrieval_selector],
        )

        # ── Event binding ─────────────────────────────────────────────────────
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input, mode_selector, retrieval_selector],
            outputs=[answer_output, sources_output, evidence_output, confidence_output, metadata_output],
        )

        question_input.submit(
            fn=answer_question,
            inputs=[question_input, mode_selector, retrieval_selector],
            outputs=[answer_output, sources_output, evidence_output, confidence_output, metadata_output],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import settings

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",       # listen on all interfaces (needed for Docker/HF Spaces)
        server_port=settings.APP_PORT,
        share=False,                  # set True to get a public gradio.live URL
        show_error=True,
    )
