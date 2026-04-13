"""
scripts/build_index.py
----------------------
Run this ONCE after cloning the FinanceBench repository.
It extracts text from all PDFs, chunks it, embeds it, and saves the FAISS index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --strategy page    # experiment with page chunks
    python scripts/build_index.py --strategy fixed   # experiment with fixed chunks
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/build_index.py` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion.dataset_loader import build_doc_info_map
from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_pages
from src.retrieval.embedder import get_embedding_model
from src.retrieval.vector_store import build_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


def main(strategy: str = "recursive"):
    logger.info("=" * 60)
    logger.info("FinanceBench Index Builder")
    logger.info("=" * 60)

    # ── Step 1: Load document metadata ───────────────────────────────────────
    logger.info("Step 1/4 — Loading document metadata …")
    if settings.DOC_INFO_FILE.exists():
        doc_info_map = build_doc_info_map()
    else:
        logger.warning(f"Doc info file not found at {settings.DOC_INFO_FILE} — using filename inference")
        doc_info_map = {}

    # ── Step 2: Extract PDF text ──────────────────────────────────────────────
    logger.info(f"Step 2/4 — Extracting text from PDFs in {settings.PDF_DIR} …")
    if not settings.PDF_DIR.exists():
        logger.error(f"PDF directory not found: {settings.PDF_DIR}")
        logger.error("Create it and add PDF files, or update PDF_DIR in .env")
        sys.exit(1)

    pages = load_all_pdfs(pdf_dir=settings.PDF_DIR, doc_info_map=doc_info_map)
    if not pages:
        logger.error("No pages extracted. Check that PDFs are in the correct directory.")
        sys.exit(1)

    logger.info(f"Extracted {len(pages)} pages from {len(set(p.doc_name for p in pages))} documents")

    # ── Step 3: Chunk text ────────────────────────────────────────────────────
    logger.info(f"Step 3/4 — Chunking pages (strategy={strategy}) …")
    chunks = chunk_pages(pages, strategy=strategy)
    logger.info(f"Created {len(chunks)} chunks")

    # ── Step 4: Embed and save vector store ───────────────────────────────────
    logger.info("Step 4/4 — Embedding chunks and building FAISS index …")
    logger.info("This may take 5–15 minutes on first run (model download + embedding)")

    embeddings = get_embedding_model()
    build_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        save_path=settings.VECTOR_STORE_PATH,
    )

    logger.info("=" * 60)
    logger.info(f"✅ Index built and saved to {settings.VECTOR_STORE_PATH}")
    logger.info("You can now start the app: python -m src.app.gradio_app")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from FinanceBench PDFs")
    parser.add_argument(
        "--strategy",
        choices=["recursive", "fixed", "page"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    args = parser.parse_args()
    main(strategy=args.strategy)
