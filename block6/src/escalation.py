# src/escalation.py
import os
import re
import json
import hashlib
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()

# Enforce output structure for the LLM fallback
class EscalationDecision(BaseModel):
    should_escalate: bool
    reason: str

CACHE_FILE = "cache/escalation_cache.json"

def _get_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def rule_based_escalation(text: str, confidence: float) -> tuple[bool, str]:
    """Fast, free deterministic rules to catch obvious escalations."""
    # 1. Low Classifier Confidence Check
    if confidence < 0.75:
        return True, "Escalated: Low AI intent confidence (< 0.75)."

    # 2. PII (Personally Identifiable Information) Regex Check
    if re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text):
        return True, "Escalated: Potential phone number detected in public tweet."
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        return True, "Escalated: Potential email address detected in public tweet."

    # 3. Severe Negative Sentiment / Legal Keywords Check
    keywords = ["sue", "lawyer", "attorney", "legal", "furious", "ruined", "police", "scam"]
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            return True, f"Escalated: Severe keyword detected: '{kw}'."

    return False, "Passed rule-based checks."

def gatekeeper(text: str, intent_confidence: float) -> EscalationDecision:
    """Decides if a human must intervene, favoring recall over precision."""
    
    # Layer 1: Check free deterministic rules first
    rule_fails, rule_reason = rule_based_escalation(text, intent_confidence)
    if rule_fails:
        return EscalationDecision(should_escalate=True, reason=rule_reason)

    # Layer 2: LLM Safety Check for nuance
    cache = _get_cache()
    text_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    if text_hash in cache:
        return EscalationDecision(**cache[text_hash])

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = f"""You are a safety gatekeeper for @AmazonHelp.
Review the customer message. Should a human agent intervene?

ESCALATE IF:
- The customer is highly emotional, abusive, or distressed.
- The customer implies a complex, multi-step problem the AI can't handle.
- The customer hints at account security breaches.

BIAS TOWARD ESCALATION: If you are unsure, you MUST escalate.

Customer Message: "{text}"
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=EscalationDecision,
            ),
        )
        result = EscalationDecision.model_validate_json(response.text)
        
        # Cache the valid call to save money and time
        cache[text_hash] = result.model_dump()
        _save_cache(cache)
        return result
        
    except Exception as e:
        # FAIL-SAFE ARCHITECTURE: If the API crashes, default to human escalation.
        return EscalationDecision(should_escalate=True, reason=f"API Error fallback: {str(e)}")

if __name__ == "__main__":
    # Test queries
    test_cases = [
        ("Where is my package?", 0.95), 
        ("I am going to sue you, this is a scam!", 0.90), 
        ("My phone number is 555-123-4567 call me.", 0.99), 
        ("I don't understand why my card was charged twice.", 0.60), 
        ("Your driver threw my TV over the fence, I'm literally shaking.", 0.95) 
    ]
    for text, conf in test_cases:
        res = gatekeeper(text, conf)
        print(f"Query: '{text}'\nEscalate: {res.should_escalate} | Reason: {res.reason}\n")