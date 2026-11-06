import hashlib
import json
import os
import re
import time

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class EscalationDecision(BaseModel):
    should_escalate: bool
    reason: str

class EscalationLLMResult(BaseModel):
    escalation_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

CACHE_FILE = "cache/two_pass_continuous_calibration.json"

_CACHE = None

def _get_cache():
    global _CACHE
    if _CACHE is None:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                _CACHE = json.load(f)
        else:
            _CACHE = {}
    return _CACHE

def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def rule_based_escalation(text: str, confidence: float) -> tuple[bool, str]:
    """Fast, free deterministic rules to catch obvious escalations."""
    if not text or not isinstance(text, str) or not text.strip():
        return False, "Passed rule-based checks."

    # 1. PII (Personally Identifiable Information) Regex Check
    if re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text):
        return True, "Escalated: Potential phone number detected in public tweet."
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        return True, "Escalated: Potential email address detected in public tweet."
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
        return True, "Escalated: Potential Social Security Number (SSN) detected."
    if re.search(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', text):
        return True, "Escalated: Potential credit/debit card number detected."

    # 2. Severe Safety, Legal, Medical & Misconduct Check
    keywords = [
        "sue", "lawyer", "attorney", "legal", "police",
        "hospital", "ambulance", "emergency", "injury", "injured",
        "cut my hand", "bleeding", "threatened", "verbally abusive", "abusive", "assault"
    ]
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            return True, f"Escalated: Severe keyword detected: '{kw}'."

    return False, "Passed rule-based checks."

def gatekeeper(text: str, intent: str, intent_confidence: float, intent_reasoning: str) -> EscalationDecision:
    """Decides if a human must intervene, favoring recall over precision."""
    if not text or not isinstance(text, str) or not text.strip():
        return EscalationDecision(should_escalate=False, reason="Empty text input.")
    
    # Layer 1: Check free deterministic rules first (Safety backstop)
    rule_fails, rule_reason = rule_based_escalation(text, intent_confidence)
    if rule_fails:
        return EscalationDecision(should_escalate=True, reason=rule_reason)

    # Layer 2: LLM continuous score evaluated against configurable threshold
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    cache = _get_cache()
    
    if text_hash in cache:
        res = EscalationLLMResult(**cache[text_hash])
    else:
        if os.environ.get("SENTINEL_OFFLINE_EVAL") == "1":
            print(f"[ERROR] Cache miss during offline evaluation in gatekeeper for text: {text[:30]}...")
            import sys
            sys.exit(1)

        prompt = f"""
You are an escalation gatekeeper for @AmazonHelp.

A customer issue has been initially classified:
- Intent: {intent}
- Confidence: {intent_confidence:.2f}
- Reasoning: {intent_reasoning}

Customer text:
"{text}"

Analyze if this issue requires urgent human intervention. Focus on:
1. Threat of legal action or PR risk
2. Severe distress or repeated failures
3. Intent types that inherently require a human (e.g. Account Locked, Refund Failed)

Output a JSON object with:
- "escalation_score": float between 0.0 and 1.0
- "reasoning": string explaining why it should/should not be escalated
"""

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("CRITICAL: GROQ_API_KEY environment variable is missing. Please check your .env file.")

        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(groq_url, headers=headers, json=payload)
                response.raise_for_status()
                json_content = response.json()["choices"][0]["message"]["content"]
                res = EscalationLLMResult.model_validate_json(json_content)
                cache[text_hash] = res.model_dump()
                _save_cache(cache)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    res = EscalationLLMResult(escalation_score=1.0, reasoning=f"API Error: {e!s}")
                    break
                time.sleep(4 * (2 ** attempt))

    # Explicit calibrated threshold: 0.40 is the recall-biased operating point (FNR < 35%).
    # The legacy uncalibrated code defaulted to 0.70, which caused severe false negative rates (~47.6%).
    DEFAULT_ESCALATION_THRESHOLD = 0.40
    try:
        threshold = float(os.getenv("ESCALATION_THRESHOLD", str(DEFAULT_ESCALATION_THRESHOLD)))
    except ValueError:
        threshold = DEFAULT_ESCALATION_THRESHOLD
    
    should_escalate = res.escalation_score >= threshold
    
    return EscalationDecision(should_escalate=should_escalate, reason=res.reasoning)


def evaluate_risk(target, threshold: float = 0.40) -> tuple[float, bool]:
    """Evaluates risk score and escalation decision for a message or classification result.
    
    Args:
        target: Customer text string OR ClassificationResult object.
        threshold: Decision boundary for escalation (defaults to calibrated 0.40).
        
    Returns:
        tuple[float, bool]: (escalation_risk_score, should_escalate)
    """
    if isinstance(target, str):
        query_text = target
        from src.classifier import classify
        res = classify(query_text)
        intent = res.intent.value if hasattr(res.intent, "value") else str(res.intent)
        conf = res.confidence
        reason = res.reasoning
    else:
        query_text = getattr(target, "text", "") or ""
        intent = target.intent.value if hasattr(target.intent, "value") else str(target.intent)
        conf = getattr(target, "confidence", 1.0)
        reason = getattr(target, "reasoning", "")
    
    # 1. Check deterministic safety rules (PII, legal, medical, misconduct)
    rule_fails, rule_reason = rule_based_escalation(query_text, conf)
    if rule_fails:
        return 1.0, True

    # 2. If already categorized as Severe Escalation
    if intent == "Severe Escalation":
        return 1.0, True

    # 3. Two-pass gatekeeper evaluation
    decision = gatekeeper(query_text, intent, conf, reason)
    
    # Extract continuous escalation score
    h = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
    cache = _get_cache()
    if h in cache:
        score = float(cache[h].get("escalation_score", 1.0 if decision.should_escalate else 0.0))
    else:
        score = 1.0 if decision.should_escalate else 0.0

    return score, decision.should_escalate


if __name__ == "__main__":  # pragma: no cover
    # Test queries
    test_cases = [
        ("Where is my package?", 0.95),
        ("I am going to sue you, this is a scam!", 0.90),
        ("My phone number is 555-123-4567 call me.", 0.99), 
        ("I don't understand why my card was charged twice.", 0.60), 
        ("Your driver threw my TV over the fence, I'm literally shaking.", 0.95) 
    ]
    for text, conf in test_cases:
        res = gatekeeper(text, "Unknown", conf, "LLM fallback test reason")
        print(f"Query: '{text}'\nEscalate: {res.should_escalate} | Reason: {res.reason}\n")
