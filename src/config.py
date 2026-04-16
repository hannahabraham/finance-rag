"""
Central configuration loader.
Reads .env (or environment variables) and exposes a single `settings` object
used across the entire project so nothing is hard-coded anywhere else.
"""

import os
from pathlib import Path
from dotenv import load_dotenv  # reads KEY=VALUE pairs from a .env file

# Load .env from the project root (two levels up from src/)
load_dotenv(Path(__file__).parent.parent / ".env")


# Enable LangSmith tracing if configured
if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
    # Set LangSmith env vars
    os.environ["LANGSMITH_TRACING"] = "true"
    if os.getenv("LANGSMITH_ENDPOINT"):
        os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    if os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
    
    # Also set LangChain env vars for tracing integration
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "finance-rag")
    
    print("LangSmith tracing is enabled for this session.")


class Settings:
    """All tuneable knobs in one place."""

    # Embedding model
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Local GGUF LLM
    LLM_MODEL_PATH: str = os.getenv("LLM_MODEL_PATH", "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    LLM_MODEL_TYPE: str = os.getenv("LLM_MODEL_TYPE", "mistral")

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    PDF_DIR: Path = Path(os.getenv("PDF_DIR", "./data/pdfs"))
    VECTOR_STORE_PATH: Path = Path(os.getenv("VECTOR_STORE_PATH", "./data/vector_store"))
    QUESTIONS_FILE: Path = Path(os.getenv("QUESTIONS_FILE", "./data/financebench_open_source.jsonl"))
    DOC_INFO_FILE: Path = Path(os.getenv("DOC_INFO_FILE", "./data/financebench_document_information.jsonl"))

    # App
    APP_PORT: int = int(os.getenv("APP_PORT", "7860"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # LangSmith Tracing
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "finance-rag")

settings = Settings()
