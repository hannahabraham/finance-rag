"""
Loads the two FinanceBench JSONL files and merges them into a single DataFrame.
Also builds a doc_info_map used by the PDF loader to attach rich metadata to pages.
"""

import json
import logging
from pathlib import Path
import pandas as pd
from src.config import settings

logger = logging.getLogger(__name__)

def load_jsonl(filepath:Path) -> list[dict]:
    """
    Read a .jsonl file (one JSON object per line) into a list of dicts.
    Skips malformed lines with a warning instead of crashing."""

    records = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line_number,line in enumerate(fh,start=1):
            line = line.strip()
            if not line:
                continue
            try: 
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping malformed JSON on line {line_number} :{exc}")
    return records


def load_questions_df(filepath: Path | None = None) -> pd.DataFrame:
    """
    Load the question-level JSONL into a DataFrame.
    Columns include: question, answer, evidence, company, doc_name, question_type, etc.
    """
    filepath = filepath or settings.QUESTIONS_FILE
    records = load_jsonl(filepath)
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} questions from {filepath.name}")
    return df


def load_doc_info_df(filepath: Path | None = None) -> pd.DataFrame:
    """
    Load document-level metadata (doc_type, doc_period, doc_link, company).
    """
    filepath = filepath or settings.DOC_INFO_FILE
    records = load_jsonl(filepath)
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} document records from {filepath.name}")
    return df


def build_merged_df() -> pd.DataFrame:
    """
    Merge question data with document metadata on the shared 'doc_name' key.
    This gives one row per question with all document context attached.
    """
    questions_df = load_questions_df()
    doc_info_df = load_doc_info_df()

    # Left join — keep all questions and attach document metadata where available
    merged = questions_df.merge(doc_info_df, on="doc_name", how="left", suffixes=("","_doc"))

    logger.info(f"Merged DataFrame: {len(merged)} rows, {len(merged.columns)} columns")
    return merged

def build_doc_info_map() -> dict:
    """
    Build a {doc_name: {company, doc_type, doc_period}} lookup dict
    used by the PDF loader to attach metadata to each extracted page.
    """
    doc_info_df = load_doc_info_df()
    doc_info_map: dict = {}

    for _, row in doc_info_df.iterrows():
        doc_name = row.get("doc_name", "")
        if not doc_name:
            continue
        doc_info_map[doc_name] = {
            "company": row.get("company", ""),
            "doc_type": row.get("doc_type", ""),
            "doc_period": str(row.get("doc_period", "")),
        }

    logger.info(f"Built doc_info_map with {len(doc_info_map)} entries")
    return doc_info_map 
