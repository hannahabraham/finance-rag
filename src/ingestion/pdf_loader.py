"""
Extracts text from every PDF in the configured directory, page by page.
Preserves rich metadata (company, document name, page number, doc type, year)
so retrieval can later filter by company or time period.
"""

import logging
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF — opens and reads PDF pages
from tqdm import tqdm  

from src.config import settings

logger = logging.getLogger(__name__)

# Data structure 

class PageDocument:
    """
    Represents a single extracted page from a PDF.
    Keeps all metadata attached so it travels with the text through the pipeline.
    """

    def __init__(
        self,
        text: str,
        page_number: int,
        doc_name: str,
        company: str = "",
        doc_type: str = "",
        doc_period: str = "",
        source_path: str = "",
    ):
        self.text = text                  # raw page text
        self.page_number = page_number    # 1-indexed page number
        self.doc_name = doc_name          # e.g. "AMAZON_2022_10K"
        self.company = company            # e.g. "Amazon"
        self.doc_type = doc_type          # e.g. "10-K", "10-Q"
        self.doc_period = doc_period      # e.g. "2022"
        self.source_path = source_path    # full file path (for debugging)

    def to_dict(self) -> dict:
        """Serialise to a plain dict — used when saving to the vector store."""
        return {
            "text": self.text,
            "page_number": self.page_number,
            "doc_name": self.doc_name,
            "company": self.company,
            "doc_type": self.doc_type,
            "doc_period": self.doc_period,
            "source_path": self.source_path,
        }

    def __repr__(self) -> str:
        return f"<PageDocument doc={self.doc_name} page={self.page_number} chars={len(self.text)}>"


# Core functions

def extract_pages_from_pdf(pdf_path: Path, doc_metadata: dict) -> list[PageDocument]:
    """
    Open one PDF and extract every page as a PageDocument.

    Args:
        pdf_path:      path to the .pdf file on disk
        doc_metadata:  dict with keys: company, doc_type, doc_period

    Returns:
        list of PageDocument, one per non-empty page
    """
    pages: list[PageDocument] = []
    doc_name = pdf_path.stem  # filename without .pdf extension

    try:
        # fitz.open() loads the PDF into memory — supports encrypted-but-unlocked PDFs
        pdf = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning(f"Could not open {pdf_path}: {exc}")
        return pages

    for page_index in range(len(pdf)):
        # get_text("text") returns plain unicode text from a PDF page
        page_text = pdf[page_index].get_text("text").strip()

        # Skip pages that are essentially blank (scanned images without OCR text)
        if len(page_text) < 50:
            continue

        pages.append(
            PageDocument(
                text=page_text,
                page_number=page_index + 1,  # convert 0-indexed to 1-indexed
                doc_name=doc_name,
                company=doc_metadata.get("company", ""),
                doc_type=doc_metadata.get("doc_type", ""),
                doc_period=doc_metadata.get("doc_period", ""),
                source_path=str(pdf_path),
            )
        )

    pdf.close()
    logger.debug(f"Extracted {len(pages)} pages from {doc_name}")
    return pages


def load_all_pdfs(
    pdf_dir: Path | None = None,
    doc_info_map: dict | None = None,
) -> list[PageDocument]:
    """
    Walk the PDF directory and extract all pages from all PDFs.

    Args:
        pdf_dir:      directory containing .pdf files (defaults to settings.PDF_DIR)
        doc_info_map: {doc_name: {company, doc_type, doc_period}} from financebench metadata
                      If None, metadata is inferred from the filename.

    Returns:
        Flat list of all PageDocuments across every PDF.
    """
    pdf_dir = pdf_dir or settings.PDF_DIR
    doc_info_map = doc_info_map or {}

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return []

    all_pages: list[PageDocument] = []

    # tqdm wraps the list to display a progress bar
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
        doc_name = pdf_path.stem
        metadata = doc_info_map.get(doc_name, _infer_metadata_from_filename(doc_name))
        pages = extract_pages_from_pdf(pdf_path, metadata)
        all_pages.extend(pages)

    logger.info(f"Total pages extracted: {len(all_pages)} from {len(pdf_files)} PDFs")
    return all_pages


def _infer_metadata_from_filename(filename: str) -> dict:
    """
    Best-effort metadata extraction when no JSONL metadata is available.
    Example: "AMAZON_2022_10K" → company=Amazon, doc_period=2022, doc_type=10-K
    """
    parts = filename.upper().split("_")
    company = parts[0] if parts else filename
    # look for a 4-digit year in the filename parts
    period = next((p for p in parts if p.isdigit() and len(p) == 4), "")
    # look for doc type like 10K, 10Q
    doc_type = next(
        (p.replace("10K", "10-K").replace("10Q", "10-Q") for p in parts if "10" in p), ""
    )
    return {"company": company.title(), "doc_type": doc_type, "doc_period": period}
