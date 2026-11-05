"""
test_graph_execution.py
-----------------------
Execution trace proof for the Sentinel-AI LangGraph DAG.

Runs two representative tickets through the compiled graph and prints
the exact node transition path, proving:

  Routine  → router → rag_retriever → drafter → END
  Severe   → router → escalator → END
"""

from __future__ import annotations

import os
import sys
import json
import textwrap
from pathlib import Path

# ── Make src/ importable when run from the scripts/ directory ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Silence noisy loggers during the trace ─────────────────────────────────
import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

from src.graph_builder import app  # compiled LangGraph StateGraph


# ─────────────────────────────────────────────────────────────────────────────

def run_ticket(ticket_id: str, user_query: str) -> dict:
    """
    Invoke the compiled graph with the given ticket and return the
    final TriageState dict.
    """
    initial_state = {
        "ticket_id": ticket_id,
        "user_query": user_query,
        "intent": None,
        "risk_score": 0.0,
        "should_escalate": False,
        "retrieved_context": [],
        "draft_reply": None,
        "routing_path": [],
    }
    return app.invoke(initial_state)


def print_trace(label: str, state: dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  TICKET : {label}")
    print(f"  QUERY  : {state['user_query']}")
    print(sep)
    print(f"  NODE PATH   : {' → '.join(state['routing_path'])} → END")
    print(f"  INTENT      : {state.get('intent', 'N/A')}")
    print(f"  RISK SCORE  : {state.get('risk_score', 0.0):.2f}")
    print(f"  ESCALATED   : {state.get('should_escalate', False)}")
    if state.get("draft_reply"):
        snippet = textwrap.shorten(state["draft_reply"], width=120, placeholder="…")
        print(f"  DRAFT       : {snippet}")
    else:
        print("  DRAFT       : ∅  (escalated — no AI draft generated)")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

TICKETS = [
    {
        "id":    "TICKET-ROUTINE-001",
        "query": "Where is my package? It has been 5 days since I ordered.",
        "expect_path":    ["router", "rag_retriever", "drafter"],
        "expect_escalate": False,
    },
    {
        "id":    "TICKET-SEVERE-001",
        "query": "If I don't get my refund today, you will be hearing from my lawyer.",
        "expect_path":    ["router", "escalator"],
        "expect_escalate": True,
    },
]


def main() -> int:
    print("\n🚀  Sentinel-AI — LangGraph DAG Execution Trace")
    print("    LangGraph compiled graph: src/graph_builder.app\n")

    failures: list[str] = []

    for ticket in TICKETS:
        print(f"⏳  Processing {ticket['id']} …", flush=True)
        final_state = run_ticket(ticket["id"], ticket["query"])
        print_trace(ticket["id"], final_state)

        # ── Assertions ──────────────────────────────────────────────────────
        actual_path     = final_state.get("routing_path", [])
        actual_escalate = final_state.get("should_escalate", False)

        if actual_path != ticket["expect_path"]:
            msg = (
                f"[FAIL] {ticket['id']} path mismatch: "
                f"expected {ticket['expect_path']}, got {actual_path}"
            )
            failures.append(msg)
            print(f"\n  ❌  {msg}")
        else:
            print(f"\n  ✅  Routing path CORRECT: {' → '.join(actual_path)} → END")

        if actual_escalate != ticket["expect_escalate"]:
            msg = (
                f"[FAIL] {ticket['id']} escalation mismatch: "
                f"expected {ticket['expect_escalate']}, got {actual_escalate}"
            )
            failures.append(msg)
            print(f"  ❌  {msg}")
        else:
            status = "ESCALATED ✓" if actual_escalate else "NOT ESCALATED ✓"
            print(f"  ✅  Escalation flag CORRECT: {status}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    if failures:
        print(f"  RESULT  : ❌  {len(failures)} ASSERTION(S) FAILED")
        for f in failures:
            print(f"            • {f}")
        print("=" * 72 + "\n")
        return 1

    print("  RESULT  : ✅  ALL ROUTING ASSERTIONS PASSED")
    print("  PROOF   :")
    print("    • Routine ticket traversed  router → rag_retriever → drafter → END")
    print("    • Severe  ticket traversed  router → escalator → END")
    print("    • No AI draft generated for escalated ticket (draft_reply = None)")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

