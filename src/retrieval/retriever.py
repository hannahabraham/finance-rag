"""
Three retrieval strategies for experiments:
    1. dense   — pure semantic (vector) search via FAISS
    2. bm25    — keyword search (BM25 algorithm, great for financial terms / ticker symbols)
    3. hybrid  — combine both scores (best overall quality)

Hybrid retrieval is the recommended default: dense catches semantically related
passages; BM25 catches exact financial figures and company names.
"""

import logging
from typing import Literal

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi 

from src.config import settings

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "bm25", "hybrid"]


# Dense retrieval

def dense_retrieve(
    vector_store: FAISS,
    query: str,
    top_k: int | None = None,
    company_filter: str | None = None,
) -> list[dict]:
    """
    Semantic retrieval using cosine similarity between query and chunk embeddings.

    Args:
        vector_store:    loaded FAISS index
        query:           natural language question
        top_k:           number of results (defaults to settings.TOP_K)
        company_filter:  if set, only return chunks from this company

    Returns:
        list of dicts with keys: text, score, doc_name, page_number, company, …
    """
    top_k = top_k or settings.TOP_K

    # similarity_search_with_score returns (Document, float_score) tuples
    results = vector_store.similarity_search_with_score(query, k=top_k * 2)

    chunks = []
    for doc, score in results:
        meta = doc.metadata
        # Optional metadata filter — narrows results to one company
        if company_filter and company_filter.lower() not in meta.get("company", "").lower():
            continue
        chunks.append(
            {
                "text": doc.page_content,
                "score": float(score),
                "doc_name": meta.get("doc_name", ""),
                "page_number": meta.get("page_number", 0),
                "company": meta.get("company", ""),
                "doc_type": meta.get("doc_type", ""),
                "doc_period": meta.get("doc_period", ""),
            }
        )
        if len(chunks) >= top_k:
            break

    logger.debug(f"Dense: retrieved {len(chunks)} chunks for query '{query[:60]}…'")
    return chunks


# BM25 retrieval

class BM25Index:
    """
    Thin wrapper around rank_bm25.BM25Okapi.
    Tokenises corpus once at init; search is then fast O(vocab) per query.
    """

    def __init__(self, corpus: list[dict]):
        """
        Args:
            corpus: list of dicts with at least {"text": str, ...metadata...}
        """
        self._corpus = corpus
        # Tokenise by whitespace — simple but effective for financial text
        tokenised = [doc["text"].lower().split() for doc in corpus]
        # BM25Okapi is the standard BM25 variant with term-frequency saturation
        self._bm25 = BM25Okapi(tokenised)
        logger.info(f"BM25 index built with {len(corpus)} documents")

    def search(
        self,
        query: str,
        top_k: int | None = None,
        company_filter: str | None = None,
    ) -> list[dict]:
        """
        Keyword search.

        Returns:
            list of result dicts (same schema as dense_retrieve)
        """
        top_k = top_k or settings.TOP_K
        query_tokens = query.lower().split()

        # get_scores returns a float score for every document in the corpus
        scores = self._bm25.get_scores(query_tokens)

        # Pair each document with its BM25 score and sort descending
        scored = sorted(
            zip(self._corpus, scores), key=lambda x: x[1], reverse=True
        )

        results = []
        for doc, score in scored:
            if company_filter and company_filter.lower() not in doc.get("company", "").lower():
                continue
            results.append({**doc, "score": float(score)})
            if len(results) >= top_k:
                break

        logger.debug(f"BM25: retrieved {len(results)} chunks for '{query[:60]}…'")
        return results


# Hybrid retrieval (Reciprocal Rank Fusion) 

def hybrid_retrieve(
    vector_store: FAISS,
    bm25_index: BM25Index,
    query: str,
    top_k: int | None = None,
    company_filter: str | None = None,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict]:
    """
    Combine dense and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF is simple and robust: each result gets a score of 1/(rank + k) from both
    lists; scores are summed; the highest combined score wins. We apply per-list
    weights (dense_weight, bm25_weight) as multipliers on the reciprocal rank —
    a common variant of vanilla RRF, not the original formulation.
    This avoids raw score normalisation issues between different retrieval methods.

    Args:
        dense_weight:  how much to favour semantic results (0–1)
        bm25_weight:   how much to favour keyword results (0–1)
    """
    top_k = top_k or settings.TOP_K
    k_rrf = 60  # standard RRF constant (reduces influence of very low-ranked docs)

    # Get a larger candidate pool from each retriever
    candidate_k = top_k * 3
    dense_results = dense_retrieve(vector_store, query, top_k=candidate_k, company_filter=company_filter)
    bm25_results = bm25_index.search(query, top_k=candidate_k, company_filter=company_filter)

    # Build a deduplicated map: unique_id → {result_dict, rrf_score}
    fusion: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        uid = f"{result['doc_name']}_{result['page_number']}_{result.get('chunk_index', rank)}"
        rrf = dense_weight / (rank + k_rrf)
        if uid not in fusion:
            fusion[uid] = {**result, "rrf_score": 0.0}
        fusion[uid]["rrf_score"] += rrf

    for rank, result in enumerate(bm25_results):
        uid = f"{result['doc_name']}_{result['page_number']}_{result.get('chunk_index', rank)}"
        rrf = bm25_weight / (rank + k_rrf)
        if uid not in fusion:
            fusion[uid] = {**result, "rrf_score": 0.0}
        fusion[uid]["rrf_score"] += rrf

    # Sort by combined RRF score and return top-k
    merged = sorted(fusion.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]

    # Rename rrf_score → score so callers see a consistent interface
    for item in merged:
        item["score"] = item.pop("rrf_score")

    logger.debug(f"Hybrid: {len(merged)} chunks for '{query[:60]}…'")
    return merged
