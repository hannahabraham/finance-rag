"""
Assembles the five agent nodes into a LangGraph StateGraph.

LangGraph is a directed graph where:
    - Nodes   = agent functions (process and update shared state)
    - Edges   = control flow (which node runs next)
    - State   = RAGState TypedDict (passed through every node)

The graph supports a conditional edge from the critic:
    ACCEPT → END
    RETRY  → retrieval (with broader search)

This creates the self-correcting loop that makes multi-agent RAG better
than a single-chain baseline.
"""

import logging
from functools import partial
from typing import Callable

from langgraph.graph import StateGraph, END 

from src.agents.state import RAGState
from src.agents.nodes import (
    query_understanding_node,
    retrieval_node,
    evidence_verification_node,
    answer_writing_node,
    critic_node,
)

logger = logging.getLogger(__name__)

# Node names
QUERY_NODE = "query_understanding"
RETRIEVE_NODE = "retrieval"
VERIFY_NODE = "evidence_verification"
WRITE_NODE = "answer_writing"
CRITIC_NODE = "critic"


def build_graph(retriever_fn: Callable) -> StateGraph:
    """
    Build and compile the multi-agent LangGraph.

    Args:
        retriever_fn: callable(query, company_filter) → list[dict]
                      Injected so we can swap retrieval strategies without
                      changing the graph structure.

    Returns:
        Compiled LangGraph app — call app.invoke(state) to run.
    """

    # 1. Create a graph with our shared state schema
    graph = StateGraph(RAGState)

    # 2. Register node functions 
    # Each node is a plain Python function: (state) → dict
    # partial() injects retriever_fn into retrieval_node without changing its signature
    graph.add_node(QUERY_NODE, query_understanding_node)
    graph.add_node(RETRIEVE_NODE, partial(retrieval_node, retriever_fn=retriever_fn))
    graph.add_node(VERIFY_NODE, evidence_verification_node)
    graph.add_node(WRITE_NODE, answer_writing_node)
    graph.add_node(CRITIC_NODE, critic_node)

    # 3. Add deterministic edges (always executed in this order) 
    graph.set_entry_point(QUERY_NODE)          # start here
    graph.add_edge(QUERY_NODE, RETRIEVE_NODE)
    graph.add_edge(RETRIEVE_NODE, VERIFY_NODE)
    graph.add_edge(VERIFY_NODE, WRITE_NODE)
    graph.add_edge(WRITE_NODE, CRITIC_NODE)

    # 4. Conditional edge from critic 
    # The critic decides: accept the answer (→ END) or retry (→ retrieval)
    graph.add_conditional_edges(
        CRITIC_NODE,
        _critic_router,
        {
            "retry": RETRIEVE_NODE,   # loop back to broaden search
            "accept": END,            # done — return final_output
        },
    )

    #  5. Compile
    # compile() validates the graph structure and returns a runnable app
    compiled = graph.compile()
    logger.info("LangGraph compiled successfully")
    return compiled


def _critic_router(state: RAGState) -> str:
    """
    Routing function for the conditional edge after the critic node.
    Returns "retry" or "accept" based on the critic's verdict.
    """
    if state.get("needs_retry", False):
        logger.info(f"Critic: RETRY (attempt {state.get('retry_count', 1)})")
        return "retry"
    logger.info("Critic: ACCEPT")
    return "accept"


def run_pipeline(graph_app, question: str) -> dict:
    """
    Run the full multi-agent pipeline for a single question.

    Args:
        graph_app:  compiled LangGraph app (from build_graph)
        question:   user's natural language question

    Returns:
        final_output dict with answer, sources, confidence, etc.
    """
    # Initialise state with just the question; all other fields default to None
    initial_state: RAGState = {
        "question": question,
        "company": None,
        "doc_period": None,
        "intent": None,
        "parsed_question": None,
        "retrieved_chunks": None,
        "verified_chunks": None,
        "answer": None,
        "explanation": None,
        "sources": None,
        "confidence": None,
        "critique": None,
        "needs_retry": False,
        "retry_count": 0,
        "final_output": None,
    }

    # invoke() runs the graph synchronously and returns the final state
    final_state = graph_app.invoke(initial_state)

    return final_state.get("final_output") or {
        "question": question,
        "answer": "Pipeline did not produce an answer.",
        "confidence": "Low",
        "sources": [],
        "evidence_snippets": [],
    }
