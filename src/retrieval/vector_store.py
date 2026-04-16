"""
Builds, saves, and loads a FAISS vector store from embedded text chunks.
"""

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from src.config import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


def build_vector_store(
    chunks: list[Chunk],
    embeddings: Embeddings,
    save_path: Path | None = None,
) -> FAISS:
    """
    Embed all chunks and build a FAISS index.

    Args:
        chunks:     list of Chunk objects (text + metadata)
        embeddings: any LangChain-compatible embedding model
        save_path:  if provided, persist the index to disk

    Returns:
        FAISS vector store ready for similarity search
    """
    if not chunks:
        raise ValueError("No chunks provided — run PDF ingestion first.")

    # Separate texts and metadata for the LangChain API
    texts = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]

    logger.info(f"Building FAISS index for {len(texts)} chunks …")

    # from_texts embeds every string and inserts into the FAISS index
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        # save_local serialises both the FAISS index and the docstore (metadata)
        vector_store.save_local(str(save_path))
        logger.info(f"Vector store saved to {save_path}")

    return vector_store


def load_vector_store(
    embeddings: Embeddings,
    load_path: Path | None = None,
) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Args:
        embeddings: must be the SAME model used during build_vector_store
        load_path:  directory where the index was saved

    Returns:
        Loaded FAISS vector store
    """
    load_path = Path(load_path or settings.VECTOR_STORE_PATH)

    if not load_path.exists():
        raise FileNotFoundError(
            f"No vector store found at {load_path}. "
            "Run `python scripts/build_index.py` first."
        )

    # allow_dangerous_deserialization=True is required by recent LangChain versions
    # It is safe here because we created the index ourselves locally
    vector_store = FAISS.load_local(
        folder_path=str(load_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    logger.info(f"Vector store loaded from {load_path}")
    return vector_store


def vector_store_exists(path: Path | None = None) -> bool:
    """Return True if a saved FAISS index exists on disk."""
    path = Path(path or settings.VECTOR_STORE_PATH)
    # LangChain saves index.faiss and index.pkl
    return (path / "index.faiss").exists()


def get_all_documents(vector_store: FAISS) -> list[dict]:
    """
    Extract every stored document (text + metadata) from a FAISS vector store.
    Used to build an auxiliary BM25 index over the same corpus.

    LangChain's FAISS wrapper does not expose a public iteration API, so we
    rely on the internal docstore dict. Centralising the access here means
    only one call-site to update if LangChain changes the internal layout.
    """
    return [
        {"text": doc.page_content, **(doc.metadata or {})}
        for doc in vector_store.docstore._dict.values()
    ]
