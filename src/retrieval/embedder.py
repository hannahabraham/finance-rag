"""
Wraps a HuggingFace sentence-transformer model to produce text embeddings.
All embedding work is done locally.
"""

import logging

from langchain_huggingface import HuggingFaceEmbeddings 

from src.config import settings

logger = logging.getLogger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and return the embedding model.

    The model is cached locally after first download (~90 MB for MiniLM).
    encode_kwargs: normalise=True makes cosine similarity == dot product,
    which FAISS uses internally — improves retrieval accuracy.

    Returns:
        HuggingFaceEmbeddings instance (LangChain-compatible)
    """
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")

    model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        # model_kwargs: passed directly to the underlying SentenceTransformer
        model_kwargs={"device": _get_device()},
        # encode_kwargs: controls how text → vector conversion behaves
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model loaded successfully")
    return model


def _get_device() -> str:
    """
    Auto-detect the best available compute device.
        mps  → Apple Silicon GPU (M-series Mac)
        cuda → NVIDIA GPU
        cpu  → fallback
    """
    try:
        import torch  
        if torch.backends.mps.is_available():
            logger.info("Using Apple MPS (Metal) backend for embeddings")
            return "mps"
        if torch.cuda.is_available():
            logger.info("Using CUDA backend for embeddings")
            return "cuda"
    except ImportError:
        pass
    logger.info("Using CPU backend for embeddings")
    return "cpu"
