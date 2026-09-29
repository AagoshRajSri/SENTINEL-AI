# src/pii_sanitizer.py
"""
PII Sanitization Middleware for Sentinel-AI.

Masks sensitive customer data BEFORE it reaches the Drafter LLM so the LLM
never needs to act as a privacy enforcer. This eliminates the robotic
"Please don't share your details publicly" scolding pattern that causes
customer dissatisfaction and LLM/human judge disagreement.

Usage:
    from src.pii_sanitizer import sanitize
    sanitized_text, detected_types = sanitize(raw_customer_text)
"""

import re
from typing import NamedTuple


class SanitizeResult(NamedTuple):
    sanitized_text: str
    detected_types: list  # e.g. ["[ORDER_ID]", "[PHONE_NUMBER]"]


# ---------------------------------------------------------------------------
# PII detection patterns — ordered from most-specific to least-specific to
# avoid partial matches (e.g. order IDs before generic long digit strings).
# ---------------------------------------------------------------------------
_PII_PATTERNS = [
    # Amazon Order IDs: 3-letter/digit prefix + 7 digits + 7 digits
    # e.g. 406-3330805-5660343, D01-1234567-8901234
    (r'\b[A-Z\d]{3}-\d{7}-\d{7}\b', '[ORDER_ID]'),

    # Amazon/carrier Tracking Numbers — several common formats:
    #   All-uppercase prefix  : TBA123456789000, 1Z999AA10123456784
    #   Mixed-case prefix     : Is119106082923 (Indian carriers)
    #   Pure digits (>=10)    : 119106082923, 913115081942
    (r'\b(?:[A-Za-z]{1,4}\d{9,18}|\d{10,22})\b', '[TRACKING_ID]'),

    # International phone numbers with optional country code
    # e.g. +44 7911 123456, 555-123-4567, 1800-300-9009, (555) 123 4567
    (r'\b(?:\+?[\d][\d\s\-().]{8,}\d)\b', '[PHONE_NUMBER]'),

    # Standard email addresses
    (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', '[EMAIL]'),
]

# Pre-compile all patterns for performance (called on every request)
_COMPILED_PATTERNS = [
    (re.compile(pattern), placeholder)
    for pattern, placeholder in _PII_PATTERNS
]


def sanitize(text):
    """
    Mask all detected PII in the customer text.

    Returns:
        SanitizeResult(sanitized_text, detected_types)
        - sanitized_text: Customer text with PII replaced by placeholders.
        - detected_types: List of placeholder strings for each PII type found
          (may contain duplicates if the same type appeared multiple times).
    """
    detected = []
    sanitized = text

    for pattern, placeholder in _COMPILED_PATTERNS:
        matches = pattern.findall(sanitized)
        if matches:
            detected.extend([placeholder] * len(matches))
            sanitized = pattern.sub(placeholder, sanitized)

    return SanitizeResult(sanitized_text=sanitized, detected_types=detected)


def build_pii_note(detected_types):
    """
    Build a natural, non-scolding instruction for the Drafter prompt when PII
    was detected and masked. Returns empty string if no PII was detected.
    """
    if not detected_types:
        return ""

    unique_types = sorted(set(detected_types))
    type_labels = {
        "[ORDER_ID]":      "order ID",
        "[TRACKING_ID]":   "tracking number",
        "[PHONE_NUMBER]":  "phone number",
        "[EMAIL]":         "email address",
    }
    human_labels = [type_labels.get(t, t.strip("[]").replace("_", " ").lower())
                    for t in unique_types]

    if len(human_labels) == 1:
        pii_summary = human_labels[0]
    elif len(human_labels) == 2:
        pii_summary = f"{human_labels[0]} and {human_labels[1]}"
    else:
        pii_summary = ", ".join(human_labels[:-1]) + f", and {human_labels[-1]}"

    return (
        f"\n\nPII CONTEXT NOTE: The customer's message contained sensitive data "
        f"({pii_summary}) that has been masked for their protection. "
        f"If your reply needs to acknowledge this, do so warmly and naturally "
        f"(e.g. 'For your security, please share those details with us privately "
        f"here: [link]') WITHOUT scolding or lecturing the customer. "
        f"Always address their actual issue FIRST, then mention the private channel."
    )


if __name__ == "__main__":
    test_cases = [
        "My order 406-3330805-5660343 hasn't arrived",
        "Tracking no. Is119106082923 shows delivered but I got nothing",
        "Call me on 555-123-4567 about my missing parcel",
        "I emailed support@amazon.com three times with no reply",
        "Where is my package? Tracking has not updated in 4 days.",
        "Order D01-1234567-8901234, tracking TBA123456789000",
    ]
    for text in test_cases:
        result = sanitize(text)
        note = build_pii_note(result.detected_types)
        print(f"Original:  {text}")
        print(f"Sanitized: {result.sanitized_text}")
        print(f"Detected:  {result.detected_types}")
        print(f"Note:      {note.strip() if note else '(none)'}")
        print()
