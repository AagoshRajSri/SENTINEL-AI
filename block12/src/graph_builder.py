"""
graph_builder.py
----------------
Assembles the Sentinel-AI LangGraph DAG.

Routing logic
─────────────
                     ┌──────────┐
         ┌──────────▶│ escalator│──▶ END
         │           └──────────┘
  START ─▶ router ──┤
         │           ┌──────────────┐    ┌─────────┐
         └──────────▶│ rag_retriever│───▶│ drafter │──▶ END
                     └──────────────┘    └─────────┘

Compile once at module load; import `app` wherever you need the graph.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END

from src.graph_state import TriageState
from src.graph_nodes import router_node, escalation_node, rag_node, drafter_node


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge function
# ─────────────────────────────────────────────────────────────────────────────

def route_ticket(state: TriageState) -> Literal["escalator", "rag_retriever"]:
    """Read the router's decision and return the name of the next node."""
    if state.get("should_escalate", False):
        return "escalator"
    return "rag_retriever"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and return the compiled Sentinel-AI triage StateGraph."""

    workflow = StateGraph(TriageState)

    # Register nodes
    workflow.add_node("router",        router_node)
    workflow.add_node("escalator",     escalation_node)
    workflow.add_node("rag_retriever", rag_node)
    workflow.add_node("drafter",       drafter_node)

    # Entry point
    workflow.set_entry_point("router")

    # Conditional branch after router
    workflow.add_conditional_edges(
        "router",
        route_ticket,
        {
            "escalator":     "escalator",
            "rag_retriever": "rag_retriever",
        },
    )

    # Linear edges for the non-escalation path
    workflow.add_edge("rag_retriever", "drafter")

    # Terminal edges
    workflow.add_edge("escalator", END)
    workflow.add_edge("drafter",   END)

    return workflow.compile()


# Module-level compiled graph — import this in scripts / tests
app = build_graph()

