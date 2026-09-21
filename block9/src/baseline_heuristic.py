# src/baseline_heuristic.py

"""
Baseline 1: Heuristic Classifier
- deterministic
- keyword-based
- free
- fast
- intentionally weak

This is a deliberately naive baseline to compare against Sentinel-AI.
It uses simple keyword matching and basic heuristics (ALL CAPS, "!") for escalation.
It does not use embeddings, RAG, Gemini, Pydantic, or sophisticated logic.
"""

from src.classifier import IntentCategory

def baseline_heuristic(text: str) -> dict:
    text_lower = text.lower()
    
    # 1. Simple heuristic for intent
    intent = IntentCategory.ORDER_ISSUES.value # Default
    
    if any(kw in text_lower for kw in ["delivery", "late", "missing", "tracking"]):
        intent = IntentCategory.DELIVERY_ISSUES.value
    elif any(kw in text_lower for kw in ["damage", "broken", "defective", "wrong"]):
        intent = IntentCategory.PRODUCT_ISSUES.value
    elif any(kw in text_lower for kw in ["refund", "charge", "payment", "card", "bill"]):
        intent = IntentCategory.PAYMENT_REFUNDS.value
    elif any(kw in text_lower for kw in ["login", "password", "account", "locked"]):
        intent = IntentCategory.ACCOUNT_ACCESS.value
    elif any(kw in text_lower for kw in ["prime", "subscription", "kindle", "digital"]):
        intent = IntentCategory.SUBSCRIPTION_DIGITAL_SERVICES.value
    elif any(kw in text_lower for kw in ["sue", "lawyer", "police", "stolen", "scam"]):
        intent = IntentCategory.SEVERE_ESCALATION.value
    elif any(kw in text_lower for kw in ["cancel", "order", "change"]):
        intent = IntentCategory.ORDER_ISSUES.value

    # 2. Simple heuristic for escalation
    should_escalate = False
    if "!" in text or text.isupper() or intent == IntentCategory.SEVERE_ESCALATION.value:
        should_escalate = True
        
    # 3. Hardcoded canned replies
    canned_replies = {
        IntentCategory.DELIVERY_ISSUES.value: "We are checking with the carrier regarding your delivery.",
        IntentCategory.ORDER_ISSUES.value: "We can help you manage your order.",
        IntentCategory.PRODUCT_ISSUES.value: "We apologize for the product issue. Let's get that replaced.",
        IntentCategory.PAYMENT_REFUNDS.value: "We are looking into your payment/refund request.",
        IntentCategory.ACCOUNT_ACCESS.value: "Let's help you regain access to your account.",
        IntentCategory.SUBSCRIPTION_DIGITAL_SERVICES.value: "We can assist with your digital subscription.",
        IntentCategory.SEVERE_ESCALATION.value: "This has been escalated to a senior specialist immediately."
    }
    
    draft_reply = canned_replies.get(intent, "We are looking into your issue.")
    
    return {
        "intent": intent,
        "should_escalate": should_escalate,
        "draft_reply": draft_reply
    }

if __name__ == "__main__":
    test_q = "MY PACKAGE WAS STOLEN!"
    result = baseline_heuristic(test_q)
    print(f"Heuristic Output: {result}")