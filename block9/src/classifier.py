# src/classifier.py

# ============================================================
# IMPORTS
# ============================================================

import os
import json
import time
import hashlib
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

# Loads variables from the .env file.
# We use this to access GEMINI_API_KEY without hardcoding
# the API key directly into the Python source code.
load_dotenv()


# ============================================================
# 1. INTENT TAXONOMY
# ============================================================

# This enum mirrors the exact 7-category taxonomy created
# and manually labeled in Block 1.
#
# Using an Enum means Gemini is constrained to return
# exactly one of these valid categories.

class IntentCategory(str, Enum):

    DELIVERY_ISSUES = "Delivery Issues"

    ORDER_ISSUES = "Order Issues"

    PRODUCT_ISSUES = "Product Issues"

    PAYMENT_REFUNDS = "Payment & Refunds"

    ACCOUNT_ACCESS = "Account & Access"

    SUBSCRIPTION_DIGITAL_SERVICES = "Subscription & Digital Services"

    SEVERE_ESCALATION = "Severe Escalation"


# ============================================================
# 2. STRUCTURED OUTPUT SCHEMA
# ============================================================

# This Pydantic model defines exactly what we expect Gemini
# to return.
#
# Expected structure:
#
# {
#     "intent": "Delivery Issues",
#     "confidence": 0.98,
#     "reasoning": "..."
# }
#
# Pydantic validates that the response actually follows
# this structure.

class ClassificationResult(BaseModel):
    # Must be one of the seven IntentCategory values above.
    intent: IntentCategory

    # Confidence must be between 0.0 and 1.0.
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Short explanation for why the category was selected.
    reasoning: str

# ============================================================
# 3. CACHE CONFIGURATION
# ============================================================

# Classifications are stored locally so that we don't repeatedly
# send the same tweet to Gemini.
#
# This saves:
#   - API calls
#   - money
#   - time
#
# The cache is stored inside the block2/cache directory.

CACHE_FILE = "cache/classifier_cache_groq.json"


# ============================================================
# 4. LOAD CACHE
# ============================================================

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


# ============================================================
# 5. SAVE CACHE
# ============================================================

def _save_cache(cache):
    """
    Save the classification cache to disk.

    os.makedirs() ensures that the cache directory exists
    before attempting to write the JSON file.
    """

    os.makedirs(
        os.path.dirname(CACHE_FILE),
        exist_ok=True
    )

    with open(CACHE_FILE, "w") as f:
        json.dump(
            cache,
            f,
            indent=2
        )


# ============================================================
# 6. CLASSIFICATION FUNCTION
# ============================================================

def classify(text: str) -> ClassificationResult:
    """
    Classify a customer tweet into one of the seven intent
    categories.

    The function uses:

    1. Local disk caching
    2. SHA-256 hashing
    3. Gemini
    4. Pydantic structured output validation
    5. Retry logic with exponential backoff
    """

    # --------------------------------------------------------
    # Load existing cache
    # --------------------------------------------------------

    cache = _get_cache()


    # --------------------------------------------------------
    # Create deterministic hash for the input text
    # --------------------------------------------------------

    # strip() removes unnecessary whitespace at the beginning
    # and end of the message.
    #
    # SHA-256 converts the text into a deterministic identifier.
    #
    # The same tweet will always produce the same hash.
    #
    # Example:
    #
    # "Where is my package?"
    #
    # -> abc123...hash...
    #

    text_hash = hashlib.sha256(
        text.strip().encode("utf-8")
    ).hexdigest()


    # --------------------------------------------------------
    # Check whether this message was already classified
    # --------------------------------------------------------

    # If the hash exists in our cache, we don't need to call
    # Gemini again.
    #
    # This is important when evaluating hundreds or thousands
    # of tweets.

    if text_hash in cache:

        return ClassificationResult(
            **cache[text_hash]
        )

    if os.environ.get("SENTINEL_OFFLINE_EVAL") == "1":
        print(f"[ERROR] Cache miss during offline evaluation in classifier for text: {text[:30]}...")
        import sys
        sys.exit(1)

    if os.environ.get("SENTINEL_OFFLINE_EVAL") == "1":
        print(f"[ERROR] Cache miss during offline evaluation in classifier for text: {text[:30]}...")
        import sys
        sys.exit(1)

    # ========================================================
    # 7. GROQ API CONFIGURATION
    # ========================================================

    # Retrieve the API key from the .env file.

    api_key = os.getenv("GROQ_API_KEY")


    # Fail early if the API key wasn't configured.

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file."
        )

    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }


    # ========================================================
    # 8. CLASSIFICATION PROMPT
    # ========================================================

    # The Enum constrains the output technically.
    #
    # The prompt provides the model with the semantic definitions
    # of the categories so it knows how to distinguish them.

    prompt = f"""
You are an inbound customer support triage agent for @AmazonHelp.

Classify the customer message into EXACTLY ONE of the following intent categories:

1. Delivery Issues (late, missing, not delivered)
2. Order Issues (placing, changing, cancelling)
3. Product Issues (damaged, defective, incorrect)
4. Payment & Refunds (charges, failed transactions, refunds)
5. Account & Access (logging in, passwords, restrictions)
6. Subscription & Digital Services (Prime, Kindle, digital content)
7. Severe Escalation (serious safety, security, criminal, highly urgent)

Customer Message:
"{text}"

Respond with a JSON object containing exactly three fields: 
"intent" (string), "confidence" (float between 0.0 and 1.0), and "reasoning" (string).
"""


    # ========================================================
    # 9. RETRY CONFIGURATION
    # ========================================================

    # If the API request temporarily fails, we'll retry it.
    #
    # Example:
    #
    # Attempt 1 -> fail
    # Wait 1 second
    #
    # Attempt 2 -> fail
    # Wait 2 seconds
    #
    # Attempt 3 -> fail
    # Raise the error
    #

    max_retries = 3


    # ========================================================
    # 10. CALL GROQ
    # ========================================================

    for attempt in range(max_retries):

        try:

            payload = {
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful customer support classification assistant."
                    },
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

            response = requests.post(
                groq_url,
                headers=headers,
                json=payload
            )
            
            # Raise an exception for bad HTTP status codes (e.g. 401, 500)
            response.raise_for_status()
            
            response_data = response.json()


            # =================================================
            # 11. VALIDATE GROQ RESPONSE
            # =================================================

            content = response_data["choices"][0]["message"]["content"]
            
            # Pydantic parses and validates it against:
            #
            # ClassificationResult
            #
            # This ensures:
            #
            # intent       -> valid enum value
            # confidence   -> 0.0 to 1.0
            # reasoning    -> string

            result = ClassificationResult.model_validate_json(content)


            # =================================================
            # 12. SAVE VALID RESULT TO CACHE
            # =================================================

            # Convert the Pydantic object into a normal
            # dictionary and store it under the SHA-256 hash.

            cache[text_hash] = result.model_dump()

            _save_cache(cache)


            # Return the validated classification.

            return result


        # =====================================================
        # 13. ERROR HANDLING + RETRY
        # =====================================================

        except Exception as e:

            # If this was the final attempt, don't silently
            # swallow the error. Raise it so we know exactly
            # what went wrong.

            if attempt == max_retries - 1:
                # FAIL-SAFE ARCHITECTURE: If the API crashes repeatedly, default to human escalation.
                # Use SEVERE_ESCALATION as a safe fallback intent.
                return ClassificationResult(
                    intent=IntentCategory.SEVERE_ESCALATION,
                    confidence=0.0,
                    reasoning=f"API Error fallback: {str(e)}"
                )


            # Exponential backoff.
            # attempt = 0 -> 4 seconds
            # attempt = 1 -> 8 seconds
            # attempt = 2 -> 16 seconds

            time.sleep(4 * (2 ** attempt))


# ============================================================
# 14. LOCAL TEST
# ============================================================

# This section only runs when you execute:
#
# python src/classifier.py
#
# It does NOT run if another Python file imports classify().

if __name__ == "__main__":

    # Three simple test cases covering different categories.

    test_queries = [

        "Where is my package? Tracking has not updated in 4 days.",

        "Your driver damaged my gate and hurled abuses at me!",

        "Can I get a refund for this broken blender?"
    ]


    # Run each test message through the classifier.

    for q in test_queries:

        res = classify(q)


        # Print the classification in a readable format.

        print(
            f"\nQuery: {q}"
            f"\nIntent: {res.intent.value}"
            f" | Confidence: {res.confidence}"
            f"\nReason: {res.reasoning}"
        )