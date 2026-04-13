"""
Defines the shared state object that flows through every node in the LangGraph.
"""

from typing import TypedDict, Optional


class RAGState(TypedDict):
    """
    Shared state passed between all agent nodes in the LangGraph pipeline.
    Fields are populated incrementally as the graph executes.
    """

    # Input
    question: str                       # original user question (never mutated)

    # Query understanding agent output 
    company: Optional[str]              # extracted company name (e.g. "Amazon")
    doc_period: Optional[str]           # extracted year / quarter (e.g. "2022")
    intent: Optional[str]               # financial concept (e.g. "operating income")
    parsed_question: Optional[str]      # cleaned / rephrased question

    # Retrieval agent output
    retrieved_chunks: Optional[list]    # list of dicts from retriever

    # Evidence verification agent output
    verified_chunks: Optional[list]     # filtered / ranked subset of retrieved_chunks

    # Answer writer agent output
    answer: Optional[str]              # generated answer text
    explanation: Optional[str]         # why this is the answer (evidence summary)
    sources: Optional[list]            # list of {doc_name, page_number} citations
    confidence: Optional[str]          # "High" | "Medium" | "Low"

    # Critic agent output
    critique: Optional[str]            # feedback from critic
    needs_retry: Optional[bool]        # True → loop back to retrieval
    retry_count: Optional[int]         # guard against infinite loops (max 2)

    # Final structured output
    final_output: Optional[dict]       # assembled structured response for the UI
