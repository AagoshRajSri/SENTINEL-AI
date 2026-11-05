# src/pipeline.py
"""
Unified End-to-End Inference Pipeline for Sentinel-AI.
Combines Pass 1 (Classification + Escalation Gate) and Pass 2 (FAISS RAG + Safeguarded Reply Drafting).
"""
import hashlib
import os
from typing import Optional, List
from pydantic import BaseModel

from src.classifier import classify, ClassificationResult
from src.escalation import gatekeeper, evaluate_risk, _get_cache as get_escalation_cache
from src.reply_drafter import draft_reply, ReplyResult, apply_grounding_safeguards
from src.retriever import retrieve


class PipelineResponse(BaseModel):
    intent: str
    confidence: float
    reasoning: str
    escalation_score: float
    risk_score: Optional[float] = None
    should_escalate: bool
    escalation_reason: str
    draft: Optional[str] = None
    draft_reply: Optional[str] = None
    policy_adherence: Optional[str] = None
    auto_draft_generated: bool = False
    retrieved_context_count: int = 0
    routing_path: Optional[List[str]] = None


def run_full_pipeline(text: str, threshold: float = 0.40) -> PipelineResponse:
    """Executes the full two-pass triage pipeline on an inbound customer query."""
    if not text or not isinstance(text, str) or not text.strip():
        return PipelineResponse(
            intent="Order Issues",
            confidence=0.0,
            reasoning="Empty input text",
            escalation_score=0.0,
            risk_score=0.0,
            should_escalate=False,
            escalation_reason="Passed rule-based checks.",
            draft=None,
            draft_reply=None,
            policy_adherence="No action for empty query.",
            auto_draft_generated=False,
            retrieved_context_count=0,
            routing_path=["router", "escalator"]
        )

    # Pass 0: Deterministic Rule-Based Safety Pre-Filter (PII, Legal, Hazards)
    from src.escalation import rule_based_escalation
    rule_fails, rule_reason = rule_based_escalation(text, 1.0)
    if rule_fails:
        return PipelineResponse(
            intent="Severe Escalation",
            confidence=1.0,
            reasoning=rule_reason,
            escalation_score=1.0,
            risk_score=1.0,
            should_escalate=True,
            escalation_reason=rule_reason,
            draft=None,
            draft_reply=None,
            policy_adherence="Safety Gate Triggered: High-risk query routed to human Tier-2 queue.",
            auto_draft_generated=False,
            retrieved_context_count=0,
            routing_path=["router", "escalator"]
        )

    # Pass 1A: Classification
    intent_res = classify(text)
    
    # Pass 1B: Escalation Risk Evaluation
    risk_score, should_esc = evaluate_risk(intent_res, threshold=threshold)
    esc_decision = gatekeeper(text, intent_res.intent.value, intent_res.confidence, intent_res.reasoning)
    
    # If rule-based escalation or risk score >= threshold
    if should_esc or esc_decision.should_escalate:
        # Strict Human Escalation: Automated drafting is suppressed
        final_esc_score = max(risk_score, 1.0 if esc_decision.should_escalate else 0.0)
        return PipelineResponse(
            intent=intent_res.intent.value,
            confidence=intent_res.confidence,
            reasoning=intent_res.reasoning,
            escalation_score=final_esc_score,
            risk_score=final_esc_score,
            should_escalate=True,
            escalation_reason=esc_decision.reason,
            draft=None,
            draft_reply=None,
            policy_adherence="Safety Gate Triggered: High-risk query routed to human Tier-2 queue.",
            auto_draft_generated=False,
            retrieved_context_count=0,
            routing_path=["router", "escalator"]
        )
    
    # Pass 2: RAG Retrieval & Safe Drafting
    context = retrieve(text, k=3, exclude_exact_match=True)
    reply_res = draft_reply(text, with_rag=True, use_cache=True, exclude_exact_match=True)
    safe_draft = apply_grounding_safeguards(reply_res.draft_reply)

    return PipelineResponse(
        intent=intent_res.intent.value,
        confidence=intent_res.confidence,
        reasoning=intent_res.reasoning,
        escalation_score=risk_score,
        risk_score=risk_score,
        should_escalate=False,
        escalation_reason="Passed escalation checks.",
        draft=safe_draft,
        draft_reply=safe_draft,
        policy_adherence=reply_res.policy_adherence_check,
        auto_draft_generated=True,
        retrieved_context_count=len(context),
        routing_path=["router", "rag_retriever", "drafter"]
    )

