# src/reply_drafter.py
import os

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

from src.retriever import retrieve
from src.pii_sanitizer import sanitize, build_pii_note

load_dotenv()

# Enforce output structure to prevent rogue markdown
class ReplyResult(BaseModel):
    draft_reply: str
    policy_adherence_check: str # Forces the AI to explain why the reply is safe

def draft_reply(customer_text: str, severity: str = "MEDIUM") -> ReplyResult:
    """Drafts a brand-safe reply grounded ONLY in historical retrieved context."""
    # 0. Sanitize PII before it reaches the LLM or the retriever.
    #    This eliminates the scolding anti-pattern at the source.
    sanitized_text, detected_pii = sanitize(customer_text)
    pii_note = build_pii_note(detected_pii)

    # 1. Retrieve top 3 similar past resolutions using the sanitized text
    #    so PII tokens don't distort the embedding similarity.
    historical_context = retrieve(sanitized_text, k=3, severity=severity)
    context_str = "\n".join([f"- {reply}" for reply in historical_context])

    # 2. Build the strict prompt
    prompt = f"""You are a customer support agent for @AmazonHelp.
Write a draft reply to the customer's message.

RULES:
1. Match the brand's tone from the historical examples.
2. DO NOT invent policies, coupon amounts, or refund timelines.
3. PII PROTECTION: The customer's message has already been sanitized — any sensitive
   data (order ID, tracking number, phone) has been replaced with a placeholder.
   If a PII placeholder is present AND you need to route the customer to a private
   channel, do so WARMLY and naturally. ALWAYS address their core issue first with
   empathy, THEN mention the private link. NEVER open with a scolding instruction
   like "Please don't share..." or "We cannot process public order details...".
   Correct pattern: "We're sorry to hear about [issue]. For your security, please
   share your details here: [link] and we'll look into it right away."
4. CONVENIENCE RULE — PREFER LINKS OVER DM: Always prefer a direct link over a DM
   redirect. Say "reach out to us here: [link]" rather than "send us a DM".
   Only use DM language when no relevant link is available.
5. DO NOT use phrases like "send us a DM", "reach out via DM", or "drop us a DM"
   as the primary call to action. Use: "reach out to us here: [link]" instead.
6. DIRECT ANSWER RULE: If the customer asks a specific question, address it directly
   instead of giving generic boilerplate asking them to explain again.
7. ACKNOWLEDGE PRIOR STEPS: If the customer states they already emailed, called, or
   replied, acknowledge their effort explicitly. Do not contradict them by telling
   them to re-email or re-call the same channel.{pii_note}

Return valid JSON with the exact structure:
{{
  "draft_reply": "your drafted reply string here",
  "policy_adherence_check": "explanation of safety and policy adherence"
}}

HISTORICAL EXAMPLES (Use these to ground your policy):
{context_str}

CUSTOMER MESSAGE:
"{sanitized_text}"
"""

    # --- GROQ API IMPLEMENTATION ---
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a professional customer support AI. Always output valid JSON conforming to the requested schema."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return ReplyResult.model_validate_json(response.choices[0].message.content)

    # --- PREVIOUS GEMINI IMPLEMENTATION (COMMENTED OUT) ---
    #     model="gemini-3.6-flash",
    #     contents=prompt,
    #     config=types.GenerateContentConfig(
    #         temperature=0.0, # Determinism is non-negotiable
    #         response_mime_type="application/json",
    #         response_schema=ReplyResult,
    #     ),
    # )
    # return ReplyResult.model_validate_json(response.text)

if __name__ == "__main__":
    # Test the drafter
    test_query = "My blender arrived shattered into a million pieces. I want a refund now."
    res = draft_reply(test_query)
    print(f"\nDrafted Reply: {res.draft_reply}")
    print(f"Safety Check: {res.policy_adherence_check}")
