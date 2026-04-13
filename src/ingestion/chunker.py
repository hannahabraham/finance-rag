"""
Splits raw page text into smaller overlapping chunks.
Smaller chunks → better retrieval precision (embeddings capture focused meaning).
Overlap → prevents an answer from being split across chunk boundaries.

Supports three strategies so the we can compare them in experiments:
    - "recursive" (default): respects paragraph / sentence boundaries (best quality)
    - "fixed":               dumb character split (fastest, baseline)
    - "page":                one chunk per page (preserves layout context)

"""

import logging
from enum import Enum
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter  

from src.config import settings
from src.ingestion.pdf_loader import PageDocument

logger = logging.getLogger(__name__)


# Chunk data structure

class Chunk:
    """A text chunk with its origin metadata preserved."""

    def __init__(
        self,
        text: str,
        chunk_index: int,
        doc_name: str,
        page_number: int,
        company: str = "",
        doc_type: str = "",
        doc_period: str = "",
    ):
        self.text = text
        self.chunk_index = chunk_index      # sequential index within the document
        self.doc_name = doc_name
        self.page_number = page_number
        self.company = company
        self.doc_type = doc_type
        self.doc_period = doc_period

    def to_metadata(self) -> dict:
        """Flat dict stored alongside the embedding in the vector store."""
        return {
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "company": self.company,
            "doc_type": self.doc_type,
            "doc_period": self.doc_period,
            "chunk_index": self.chunk_index,
        }

    def __repr__(self) -> str:
        return f"<Chunk doc={self.doc_name} page={self.page_number} idx={self.chunk_index} len={len(self.text)}>"


# Chunking strategies

ChunkStrategy = Literal["recursive", "fixed", "page"]


def chunk_pages(
    pages: list[PageDocument],
    strategy: ChunkStrategy = "recursive",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Convert a list of PageDocuments into a flat list of Chunks.

    Args:
        pages:          output of pdf_loader.load_all_pdfs()
        strategy:       "recursive" | "fixed" | "page"
        chunk_size:     override settings.CHUNK_SIZE
        chunk_overlap:  override settings.CHUNK_OVERLAP

    Returns:
        list of Chunk objects ready for embedding
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    if strategy == "page":
        return _chunk_by_page(pages)
    elif strategy == "fixed":
        return _chunk_fixed(pages, chunk_size, chunk_overlap)
    else:
        return _chunk_recursive(pages, chunk_size, chunk_overlap)


def _chunk_recursive(
    pages: list[PageDocument], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """
    Use LangChain's RecursiveCharacterTextSplitter.
    Tries to split on paragraph breaks, then sentences, then words.
    Best balance of chunk quality and speed — recommended default.
    """
    # Separators tried in order: paragraph → sentence → word → character
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []
    chunk_global_index = 0

    for page in pages:
        if not page.text.strip():
            continue

        # split_text returns a list of string fragments
        fragments = splitter.split_text(page.text)

        for frag in fragments:
            if len(frag.strip()) < 30:  # skip tiny noise fragments
                continue
            chunks.append(
                Chunk(
                    text=frag.strip(),
                    chunk_index=chunk_global_index,
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    company=page.company,
                    doc_type=page.doc_type,
                    doc_period=page.doc_period,
                )
            )
            chunk_global_index += 1

    logger.info(f"Recursive chunking → {len(chunks)} chunks from {len(pages)} pages")
    return chunks


def _chunk_fixed(
    pages: list[PageDocument], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """
    Naive fixed-size character split with overlap.
    Useful as a baseline to compare against recursive.
    """
    chunks: list[Chunk] = []
    chunk_global_index = 0

    for page in pages:
        text = page.text.strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            fragment = text[start:end].strip()
            if len(fragment) >= 30:
                chunks.append(
                    Chunk(
                        text=fragment,
                        chunk_index=chunk_global_index,
                        doc_name=page.doc_name,
                        page_number=page.page_number,
                        company=page.company,
                        doc_type=page.doc_type,
                        doc_period=page.doc_period,
                    )
                )
                chunk_global_index += 1
            start += chunk_size - chunk_overlap  # slide window forward with overlap

    logger.info(f"Fixed chunking → {len(chunks)} chunks from {len(pages)} pages")
    return chunks


def _chunk_by_page(pages: list[PageDocument]) -> list[Chunk]:
    """
    One chunk = one full page. No splitting.
    Preserves page-level context; lower retrieval precision for long pages.
    """
    chunks = [
        Chunk(
            text=page.text.strip(),
            chunk_index=i,
            doc_name=page.doc_name,
            page_number=page.page_number,
            company=page.company,
            doc_type=page.doc_type,
            doc_period=page.doc_period,
        )
        for i, page in enumerate(pages)
        if len(page.text.strip()) >= 30
    ]
    logger.info(f"Page chunking → {len(chunks)} chunks from {len(pages)} pages")
    return chunks
