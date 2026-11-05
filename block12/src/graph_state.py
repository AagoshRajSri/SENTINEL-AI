"""
graph_state.py
--------------
Defines the shared state TypedDict that flows between every node
in the Sentinel-AI LangGraph DAG.  Every field is Optional so nodes
can progressively populate the state without mutation errors.
"""

from typing import TypedDict, Optional, List


class TriageState(TypedDict):
    """Immutable-ish payload that travels through the triage DAG."""

    # ── Input ──────────────────────────────────────────────────────────────
    ticket_id: str                      # Unique ID for tracing / logging
    user_query: str                     # Raw customer text

    # ── Router outputs ─────────────────────────────────────────────────────
    intent: Optional[str]               # IntentCategory name (e.g. "ORDER_STATUS")
    risk_score: float                   # 0.0 → 1.0 continuous risk estimate
    should_escalate: bool               # Hard gate: True → human handoff

    # ── RAG outputs ────────────────────────────────────────────────────────
    retrieved_context: List[str]        # Top-k policy snippets from FAISS

    # ── Drafter outputs ────────────────────────────────────────────────────
    draft_reply: Optional[str]          # LLM-generated customer reply

    # ── Audit ──────────────────────────────────────────────────────────────
    routing_path: List[str]             # Ordered list of nodes visited

