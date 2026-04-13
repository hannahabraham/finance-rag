"""
Loads a local GGUF quantised LLM using ctransformers.
GGUF models run entirely on your Mac — no API key, no internet, no cost.

Recommended model: Mistral-7B-Instruct-v0.2.Q4_K_M.gguf (~4 GB)
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
        ctransformers AutoModelForCausalLM instance
    """
    from ctransformers import AutoModelForCausalLM  # imported lazily to keep startup fast

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

    llm = AutoModelForCausalLM.from_pretrained(
        str(model_path.parent),         # directory containing the GGUF file
        model_file=model_path.name,     # the specific .gguf filename
        model_type=settings.LLM_MODEL_TYPE,
        # gpu_layers: how many transformer layers to run on Metal/GPU
        # 50 is a good default for M-series with 16 GB RAM
        # set to 0 for pure CPU (slower but uses less RAM)
        gpu_layers=50,
        # context_length: max tokens the model sees at once
        # 4096 is enough for most RAG prompts
        context_length=4096,
        # max_new_tokens: max length of the generated answer
        max_new_tokens=512,
        temperature=0.1,   # low temperature → factual, less creative
        top_p=0.9,
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
    response = llm(prompt, max_new_tokens=max_tokens)
    return response.strip()
