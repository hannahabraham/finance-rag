"""
Loads a local GGUF quantised LLM using llama-cpp-python.
GGUF models run entirely on your Mac — no API key, no internet, no cost.

llama-cpp-python is the actively maintained binding with first-class Metal
(Apple Silicon) support. ctransformers — the previous backend — is no longer
maintained and has flaky Metal support for Mistral.

Model: Mistral-7B-Instruct-v0.2.Q4_K_M.gguf (~4 GB)
Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
"""

import logging
from functools import lru_cache
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm():
    """
    Load the local GGUF model once and cache it.
    lru_cache ensures we don't reload the model on every agent call.

    Returns:
        llama_cpp.Llama instance
    """
    from llama_cpp import Llama  # imported lazily to keep startup fast

    model_path = Path(settings.LLM_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"\n\nLLM model not found at: {model_path}\n"
            "Download it with:\n"
            "  pip install huggingface_hub\n"
            "  python scripts/download_model.py\n"
            "Or manually from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        )

    logger.info(f"Loading LLM from {model_path} …")

    llm = Llama(
        model_path=str(model_path),
        # n_ctx: context window size (max input+output tokens)
        n_ctx=4096,
        # n_gpu_layers: how many transformer layers to offload to Metal/GPU
        # 50 is a good default for M-series with 16 GB RAM
        # set to 0 for pure CPU (slower but uses less RAM)
        # set to -1 to offload every layer (fastest if memory allows)
        n_gpu_layers=50,
        # n_threads: None → llama.cpp picks a sensible default
        n_threads=None,
        verbose=False,
    )

    logger.info("LLM loaded successfully")
    return llm


def generate(prompt: str, max_tokens: int = 512) -> str:
    """
    Run inference with the local LLM.

    Args:
        prompt:     the full prompt string (system + context + question)
        max_tokens: max tokens to generate

    Returns:
        Generated text string
    """
    llm = get_llm()
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,   # low temperature → factual, less creative
        top_p=0.9,
    )
    return output["choices"][0]["text"].strip()
