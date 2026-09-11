import os
import re
import json
import hashlib
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests

load_dotenv()

class EscalationDecision(BaseModel):
    should_escalate: bool
    reason: str

class EscalationLLMResult(BaseModel):
    escalation_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

CACHE_FILE = "cache/two_pass_continuous_calibration.json"

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
    # 1. PII (Personally Identifiable Information) Regex Check
    if re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text):
        return True, "Escalated: Potential phone number detected in public tweet."
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        return True, "Escalated: Potential email address detected in public tweet."

    # 2. Severe Negative Sentiment / Legal Keywords Check
    # Removed generic words like 'furious', 'ruined', 'scam' per experimental criteria
    keywords = ["sue", "lawyer", "attorney", "legal", "police"]
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            return True, f"Escalated: Severe keyword detected: '{kw}'."

    return False, "Passed rule-based checks."

def gatekeeper(text: str, intent: str, intent_confidence: float, intent_reasoning: str) -> EscalationDecision:
    """Decides if a human must intervene, favoring recall over precision."""
    
    # Layer 1: Check free deterministic rules first (Safety backstop)
    # Uses current, narrowed deterministic rules (no broad keywords, no low-confidence trigger).
    rule_fails, rule_reason = rule_based_escalation(text, intent_confidence)
    if rule_fails:
        return EscalationDecision(should_escalate=True, reason=rule_reason)

    # Layer 2: LLM continuous score evaluated against configurable threshold
    
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    cache = _get_cache()
    if text_hash in cache:
        res = EscalationLLMResult(**cache[text_hash])
    else:
        prompt = f"""
You are an escalation gatekeeper for @AmazonHelp.

The customer sent this message:
"{text}"

The first-pass classifier determined:
Intent: {intent}
Confidence: {intent_confidence}
Reasoning: {intent_reasoning}

Evaluate whether this message requires human escalation. 
Provide a continuous escalation_score from 0.0 to 1.0.
0.0 = clearly no escalation
1.0 = clearly requires escalation

Respond with a JSON object containing exactly two fields:
"escalation_score" (float between 0.0 and 1.0) and "reasoning" (string).
"""
        api_key = os.getenv("GROQ_API_KEY")
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_object"
            }
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(groq_url, headers=headers, json=payload)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                res = EscalationLLMResult.model_validate_json(content)
                cache[text_hash] = res.model_dump()
                _save_cache(cache)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    res = EscalationLLMResult(escalation_score=1.0, reasoning=f"API Error: {str(e)}")
                    break
                time.sleep(4 * (2 ** attempt))

    try:
        threshold = float(os.getenv("ESCALATION_THRESHOLD", "0.70"))
    except ValueError:
        threshold = 0.70
        
    should_escalate = res.escalation_score >= threshold
    
    return EscalationDecision(should_escalate=should_escalate, reason=res.reasoning)

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
        res = gatekeeper(text, conf, 0.50, "LLM fallback test reason")
        print(f"Query: '{text}'\nEscalate: {res.should_escalate} | Reason: {res.reason}\n")