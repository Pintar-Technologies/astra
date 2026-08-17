from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    broaden_search,
    cite,
    generate,
    generate_fallback,
    grade_relevance,
    retrieve,
)
from app.graph.state import RAGState


def build_graph() -> StateGraph:
    """Assemble the RAG retrieval graph."""
    builder = StateGraph(RAGState)

    # Add nodes
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_relevance", grade_relevance)
    builder.add_node("broaden_search", broaden_search)
    builder.add_node("generate", generate)
    builder.add_node("generate_fallback", generate_fallback)
    builder.add_node("cite", cite)

    # Edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_relevance")

    # Conditional: route from grade_relevance
    def route_after_grade(state: RAGState) -> str:
        graded_docs = state.get("graded_docs", [])
        needs_broaden = state.get("needs_broaden", False)

        if graded_docs:
            return "generate"
        if needs_broaden:
            # Already tried broadening — no relevant docs found
            return "generate_fallback"
        # First pass — no relevant docs, try broadening
        return "broaden_search"

    builder.add_conditional_edges(
        "grade_relevance",
        route_after_grade,
        {
            "generate": "generate",
            "generate_fallback": "generate_fallback",
            "broaden_search": "broaden_search",
        },
    )

    # broaden_search loops back to retrieve
    builder.add_edge("broaden_search", "retrieve")

    # After generate, run cite
    builder.add_edge("generate", "cite")
    builder.add_edge("cite", END)

    # generate_fallback goes directly to END
    builder.add_edge("generate_fallback", END)

    return builder.compile()
