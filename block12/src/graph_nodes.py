"""
graph_nodes.py
--------------
Pure-function node implementations for the Sentinel-AI LangGraph DAG.
Each node accepts a TriageState dict and returns a *partial* dict of
fields it wants to update (LangGraph merges them automatically).
"""

from __future__ import annotations

import logging
from typing import Any

from src.graph_state import TriageState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 – Router
# Classifies intent, evaluates risk, and sets the escalation gate.
# ─────────────────────────────────────────────────────────────────────────────

def router_node(state: TriageState) -> dict[str, Any]:
    """Classify intent and evaluate risk score for the incoming ticket."""

    from src.classifier import classify
    from src.escalation import evaluate_risk, rule_based_escalation

    user_query = state["user_query"]

    # 1. LLM-based intent classification
    classification = classify(user_query)
    intent = classification.intent.value           # e.g. "ORDER_STATUS"
    conf   = getattr(classification, "confidence", 1.0)

    # 2. Rule-based hard-stop (legal / safety keywords + PII)
    #    signature: rule_based_escalation(text, confidence) -> (bool, str)
    hard_stop, _rule_reason = rule_based_escalation(user_query, conf)

    # 3. Probabilistic risk score from evaluate_risk
    #    signature: evaluate_risk(target, threshold) -> (float, bool)
    risk_score, risk_flag = evaluate_risk(classification)

    # Escalate if either the hard rule OR the probabilistic threshold triggers
    should_escalate = hard_stop or risk_flag

    path = list(state.get("routing_path", []))
    path.append("router")

    logger.info(
        "[router] ticket=%s intent=%s risk=%.2f escalate=%s",
        state.get("ticket_id", "?"), intent, risk_score, should_escalate,
    )

    return {
        "intent": intent,
        "risk_score": risk_score,
        "should_escalate": should_escalate,
        "routing_path": path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 – Escalator
# Hard-stops the pipeline and flags the ticket for human review.
# ─────────────────────────────────────────────────────────────────────────────

def escalation_node(state: TriageState) -> dict[str, Any]:
    """Flag ticket for immediate human review; set draft_reply to None."""

    path = list(state.get("routing_path", []))
    path.append("escalator")

    logger.warning(
        "[escalator] ticket=%s ESCALATED — routed to human operator. intent=%s risk=%.2f",
        state.get("ticket_id", "?"), state.get("intent"), state.get("risk_score", 0.0),
    )

    return {
        "draft_reply": None,       # Explicit: no AI draft for escalated tickets
        "retrieved_context": [],   # No context injection needed
        "routing_path": path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 – RAG Retriever
# Fetches top-k policy snippets from the FAISS semantic index.
# ─────────────────────────────────────────────────────────────────────────────

def rag_node(state: TriageState) -> dict[str, Any]:
    """Retrieve relevant policy context from the FAISS vector store."""

    from src.retriever import retrieve

    user_query = state["user_query"]
    docs = retrieve(user_query, k=3)

    path = list(state.get("routing_path", []))
    path.append("rag_retriever")

    logger.info(
        "[rag_retriever] ticket=%s retrieved %d docs",
        state.get("ticket_id", "?"), len(docs),
    )

    return {
        "retrieved_context": docs,
        "routing_path": path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 – Drafter
# Generates a grounded, context-aware reply using the retrieved snippets.
# ─────────────────────────────────────────────────────────────────────────────

def drafter_node(state: TriageState) -> dict[str, Any]:
    """Draft a customer reply grounded in the retrieved FAISS context."""

    from src.reply_drafter import draft_reply

    user_query = state["user_query"]
    context_snippets = state.get("retrieved_context", [])

    # Inject retrieved context into the drafter prompt
    context_block = "\n\n".join(context_snippets) if context_snippets else ""
    enriched_query = (
        f"{user_query}\n\n[Context]\n{context_block}"
        if context_block
        else user_query
    )

    result = draft_reply(enriched_query)
    reply_text = result.reply if hasattr(result, "reply") else str(result)

    path = list(state.get("routing_path", []))
    path.append("drafter")

    logger.info(
        "[drafter] ticket=%s draft generated (%d chars)",
        state.get("ticket_id", "?"), len(reply_text),
    )

    return {
        "draft_reply": reply_text,
        "routing_path": path,
    }
