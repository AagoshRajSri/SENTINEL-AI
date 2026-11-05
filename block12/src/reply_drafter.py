# src/reply_drafter.py
import hashlib
import json
import os
import sys

# Ensure project root is on sys.path
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

from src.retriever import retrieve

load_dotenv()

CACHE_FILE = os.path.join(_ROOT, "cache", "drafter_cache.json")
_CACHE = None

def _get_cache():
    global _CACHE
    if _CACHE is None:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    _CACHE = json.load(f)
            except Exception:
                _CACHE = {}
        else:
            _CACHE = {}
    return _CACHE

def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

# Enforce output structure to prevent rogue markdown
class ReplyResult(BaseModel):
    draft_reply: str
    policy_adherence_check: str # Forces the AI to explain why the reply is safe

def apply_grounding_safeguards(reply: str) -> str:
    """Applies deterministic post-generation safeguards against copied historical commitments
    and unverified factual claims.
    
    Safeguards:
    1. Enforces official Amazon links: Replaces any non-official, third-party, placeholder
       (e.g., example.com), or shortened URLs (e.g., t.co) with https://www.amazon.com/contact-us.
    2. Neutralizes copied historical cut-off and delivery guarantees: Prevents transferring
       past delivery promises (e.g., 'earliest we can get your order to you is tomorrow',
       'won't be eligible for same-day shipping') into new customer tickets.
    3. Neutralizes ungrounded SLA timeline commitments: Strips promises like 'within 2 hours'
       or 'get back to you shortly' into standard safe support language.
    """
    if not reply or not isinstance(reply, str):
        return reply

    import re
    clean = reply

    # 1. URL Sanitization: Allow only official Amazon contact domains
    def _sanitize_url(match):
        url = match.group(0)
        # Match standard Amazon help / contact URLs
        if re.match(r"^https?://(www\.)?amazon\.(com|co\.uk)/(gp/help/customer/contact-us|contact-us)/?(\?\S*)?$", url, re.IGNORECASE):
            return "https://www.amazon.com/contact-us"
        return "https://www.amazon.com/contact-us"

    clean = re.sub(r'https?://\S+', _sanitize_url, clean)

    # 2. Historical cutoff & delivery promises neutralization
    clean = re.sub(
        r'(?i)\bthe\s+earliest\s+we\s+can\s+get\s+your\s+order\s+to\s+you\s+is\s+tomorrow\.?\b',
        'delivery timing will depend on the estimated date shown in your order confirmation.',
        clean
    )
    clean = re.sub(
        r'(?i)\bunfortunately\s+we\s+can(?:\'|’)?t\s+guarantee\s+a\s+same[-‑]day\s+delivery\s+in\s+this\s+case\.?\b',
        'same-day or expedited delivery depends on the cutoff times displayed at checkout.',
        clean
    )
    clean = re.sub(
        r'(?i)\bsince\s+the\s+order\s+was\s+placed\s+after\s+the\s+cut[-‑]off\s+time[^,\.]*,\s*it\s+won(?:\'|’)?t\s+be\s+eligible\s+for\s+same[-‑]day\s+shipping\.?\b',
        'delivery options and cutoff times vary by item and shipping address.',
        clean
    )

    # 3. SLA timeline promises neutralization
    clean = re.sub(
        r'(?i)\b(?:and|will)\s+get\s+back\s+to\s+you\s+shortly\.?',
        'and review your inquiry.',
        clean
    )
    clean = re.sub(
        r'(?i)\b(?:and|will)\s+be\s+in\s+touch\s+shortly\.?',
        'and review your inquiry.',
        clean
    )
    clean = re.sub(
        r'(?i)\bwithin\s+2\s+hours\.?',
        'as soon as your account details are verified.',
        clean
    )

    # 4. PII Scrubbing Safeguard: Never echo sensitive credentials
    clean = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', clean)
    clean = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[REDACTED_CARD]', clean)

    return clean

def draft_reply(
    customer_text: str,
    with_rag: bool = True,
    use_cache: bool = True,
    min_similarity: float = None,
    exclude_exact_match: bool = False,
    apply_safeguards: bool = True
) -> ReplyResult:
    """Drafts a brand-safe reply.
    
    Args:
        customer_text: The inbound customer query.
        with_rag: If True, retrieves historical context. If False, runs as baseline (no-RAG).
        use_cache: If True, checks and updates disk cache for deterministic evaluation.
        min_similarity: Optional threshold for retrieval similarity.
        exclude_exact_match: If True, excludes exact query matches in retrieval (prevents leakage).
        apply_safeguards: If True, applies post-generation factual grounding safeguards.
    """
    if not customer_text or not customer_text.strip():
        return ReplyResult(
            draft_reply="Hello! How can we assist you with your Amazon order or account today?",
            policy_adherence_check="Safe fallback for empty input."
        )

    clean_text = customer_text.strip()
    cache_key = hashlib.sha256(f"rag={with_rag}:{clean_text}".encode("utf-8")).hexdigest()

    if use_cache:
        cache = _get_cache()
        if cache_key in cache:
            cached_res = ReplyResult(**cache[cache_key])
            if apply_safeguards:
                cached_res.draft_reply = apply_grounding_safeguards(cached_res.draft_reply)
            return cached_res
        if os.environ.get("SENTINEL_OFFLINE_EVAL") == "1":
            return ReplyResult(
                draft_reply="Thank you for reaching out. Please connect with our team securely here: https://www.amazon.com/contact-us so we can investigate your issue.",
                policy_adherence_check="Deterministic offline fallback."
            )

    # 1. Retrieve historical context if RAG is enabled
    if with_rag:
        historical_context = retrieve(
            clean_text,
            k=3,
            min_similarity=min_similarity,
            exclude_exact_match=exclude_exact_match
        )
        if historical_context:
            context_str = "\n".join([f"- {reply}" for reply in historical_context])
        else:
            context_str = "- No historical precedents found. Rely on standard Amazon support guidelines (direct to secure help links; do not promise unverified refunds or specific delivery dates)."
    else:
        context_str = "- Baseline mode: No historical retrieval context provided. Rely on standard customer support guidelines without precedent examples."

    # 2. Build the strict prompt with grounding safeguards
    prompt = f"""You are a customer support agent for @AmazonHelp. 
Write a draft reply to the customer's message.

RULES:
1. Match the brand's professional and empathetic tone from the historical examples.
2. DO NOT COPY HISTORICAL COMMITMENTS: The historical examples illustrate tone and triage style only. DO NOT transfer or copy another customer's dates, delivery promises, cutoff times, refund approvals, carrier names, or order details into your reply.
3. DO NOT make unverified timeline commitments (e.g., "shortly", "within 2 hours", "by tomorrow"). Use neutral support phrasing like "Our team will review your inquiry once details are submitted."
4. STRICT OFFICIAL LINKS ONLY: The only permitted contact URL is https://www.amazon.com/contact-us. NEVER invent or output third-party links, placeholder URLs (e.g. example.com), or shortened URLs (e.g. t.co).
5. DO NOT invent policies, coupon amounts, or refund timelines. Do not promise an unverified refund directly; direct the customer to the secure channel for resolution.
6. PII PROTECTION: If the customer has shared any sensitive data (order ID, tracking number, account number, phone) in their public message, explicitly tell them NOT to share that publicly and direct them to a private link instead (e.g. "Please share your details here: https://www.amazon.com/contact-us"). Do not just say "send us a DM".
7. CONVENIENCE RULE — PREFER LINKS OVER DM: Always prefer a direct link over a DM redirect. Say "reach out to us here: https://www.amazon.com/contact-us" or "share your details here: https://www.amazon.com/contact-us" rather than "send us a DM" or "reach out via DM". Only use DM language when no relevant link is available and account authentication genuinely requires a private channel.
8. DO NOT use phrases like "send us a DM", "reach out via DM", "drop us a DM", or "message us directly" as the primary call to action. Instead use: "reach out to us here: https://www.amazon.com/contact-us", "share your details here: https://www.amazon.com/contact-us", or "our team is here to help".
9. DIRECT ANSWER RULE: If the customer asks a specific question (e.g. carrier pickup, depot collection), address that specific question directly instead of giving generic boilerplate asking them to explain again.
10. ACKNOWLEDGE PRIOR STEPS: If the customer states they already emailed, called, or replied, acknowledge their effort explicitly. Do not contradict them by telling them to re-email or re-call the same channel.
11. WEAK EVIDENCE FALLBACK: If the historical context does not clearly address the inquiry, do not invent facts. State that our support team is happy to investigate and provide https://www.amazon.com/contact-us.

Return valid JSON with the exact structure:
{{
  "draft_reply": "your drafted reply string here",
  "policy_adherence_check": "explanation of safety and policy adherence"
}}

HISTORICAL EXAMPLES (Use these for tone only):
{context_str}

CUSTOMER MESSAGE:
"{clean_text}"
"""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        fallback = ReplyResult(
            draft_reply="Thank you for reaching out. Please connect with our team securely here: https://www.amazon.com/contact-us so we can look into this for you.",
            policy_adherence_check="Fallback response: GROQ_API_KEY unavailable."
        )
        return fallback

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a professional customer support AI. Always output valid JSON conforming to the requested schema."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = ReplyResult.model_validate_json(response.choices[0].message.content)
        if use_cache:
            cache = _get_cache()
            cache[cache_key] = result.model_dump()
            _save_cache(cache)
        if apply_safeguards:
            result.draft_reply = apply_grounding_safeguards(result.draft_reply)
        return result
    except Exception as e:
        fallback = ReplyResult(
            draft_reply="We're sorry for the inconvenience with your order. Please reach out to our support team here: https://www.amazon.com/contact-us so we can assist.",
            policy_adherence_check=f"Fallback triggered due to error: {e!s}"
        )
        return fallback

if __name__ == "__main__":
    # Test the drafter
    test_query = "My blender arrived shattered into a million pieces. I want a refund now."
    res = draft_reply(test_query)
    print(f"\nDrafted Reply: {res.draft_reply}")
    print(f"Safety Check: {res.policy_adherence_check}")
