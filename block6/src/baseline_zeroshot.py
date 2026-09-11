# src/baseline_zeroshot.py

"""
Baseline 2: Zero-Shot LLM Classifier
- Gemini zero-shot
- no RAG
- no Pydantic schema
- no confidence gating
- no historical examples
- intentionally weaker than the real Sentinel pipeline

This baseline uses Gemini to directly output JSON without schema enforcement.
It caches results, but ignores stale cache entries containing the old taxonomy.
"""

import os
import json
import hashlib

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.classifier import IntentCategory

# Load variables from .env
load_dotenv()

CACHE_FILE = "cache/zeroshot_cache.json"

# Current valid categories based on the 7-category taxonomy
VALID_CATEGORIES = {
    IntentCategory.DELIVERY_ISSUES.value,
    IntentCategory.ORDER_ISSUES.value,
    IntentCategory.PRODUCT_ISSUES.value,
    IntentCategory.PAYMENT_REFUNDS.value,
    IntentCategory.ACCOUNT_ACCESS.value,
    IntentCategory.SUBSCRIPTION_DIGITAL_SERVICES.value,
    IntentCategory.SEVERE_ESCALATION.value
}

def _get_cache():
    """Load the existing cache from disk."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def _save_cache(cache):
    """Save classification results to disk."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def baseline_zeroshot(text: str) -> dict:
    # 1. Check cache
    cache = _get_cache()

    text_hash = hashlib.sha256(
        text.strip().encode("utf-8")
    ).hexdigest()

    # 2. Robust cache check (ignore stale entries with old taxonomy)
    if text_hash in cache:
        cached_entry = cache[text_hash]
        if isinstance(cached_entry, dict):
            # Check if all required keys exist
            if all(k in cached_entry for k in ["intent", "should_escalate", "draft_reply"]):
                # Verify that the intent belongs to the CURRENT 7-category taxonomy
                if cached_entry["intent"] in VALID_CATEGORIES:
                    return cached_entry

    # 3. Create Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")

    client = genai.Client(api_key=api_key)

    # 4. Simple Zero-Shot prompt with exactly the 7 current categories
    prompt = f"""
You are an AI customer support assistant for @AmazonHelp.

Read the customer message and produce a JSON object with
exactly these three keys:

- "intent": Choose one of these categories:
  - "Delivery Issues"
  - "Order Issues"
  - "Product Issues"
  - "Payment & Refunds"
  - "Account & Access"
  - "Subscription & Digital Services"
  - "Severe Escalation"

- "should_escalate": true or false

- "draft_reply": A short customer support reply.

Customer Message:
"{text}"
"""

    # 5. Call Gemini
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        # 6. Parse the JSON manually
        result = json.loads(response.text)
        
        # Ensure fallback safety in case Gemini hallucinates an intent not in taxonomy
        if result.get("intent") not in VALID_CATEGORIES:
            result["intent"] = IntentCategory.ORDER_ISSUES.value
            
        # Ensure other fields exist
        if "should_escalate" not in result:
            result["should_escalate"] = False
        if "draft_reply" not in result:
            result["draft_reply"] = "We are looking into this."

        # 7. Save successful result to cache
        cache[text_hash] = result
        _save_cache(cache)

        return result

    except Exception as e:
        # 8. Simple failure fallback using the CURRENT taxonomy
        return {
            "intent": IntentCategory.ORDER_ISSUES.value,
            "should_escalate": True,
            "draft_reply": f"System Error: {str(e)}",
        }

if __name__ == "__main__":
    test_q = "My package was stolen!"
    result = baseline_zeroshot(test_q)
    print(f"Zero-Shot Output: {result}")
