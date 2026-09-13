# src/reply_drafter.py
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types
from src.retriever import retrieve

load_dotenv()

# Enforce output structure to prevent rogue markdown
class ReplyResult(BaseModel):
    draft_reply: str
    policy_adherence_check: str # Forces the AI to explain why the reply is safe

def draft_reply(customer_text: str) -> ReplyResult:
    """Drafts a brand-safe reply grounded ONLY in historical retrieved context."""
    # 1. Retrieve the top 3 similar past resolutions
    historical_context = retrieve(customer_text, k=3)
    context_str = "\n".join([f"- {reply}" for reply in historical_context])
    
    # 2. Build the strict prompt
    prompt = f"""You are a customer support agent for @AmazonHelp. 
Write a draft reply to the customer's message.

RULES:
1. Match the brand's tone from the historical examples.
2. DO NOT invent policies, coupon amounts, or refund timelines.
3. PII PROTECTION: If the customer has shared any sensitive data (order ID, tracking number, account number, phone) in their public message, explicitly tell them NOT to share that publicly and direct them to a private link instead (e.g. "Please share your details here: [link]"). Do not just say "send us a DM".
4. CONVENIENCE RULE — PREFER LINKS OVER DM: Always prefer a direct link over a DM redirect. Say "reach out to us here: [link]" or "share your details here: [link]" rather than "send us a DM" or "reach out via DM". Only use DM language when no relevant link is available and account authentication genuinely requires a private channel.
5. DO NOT use phrases like "send us a DM", "reach out via DM", "drop us a DM", or "message us directly" as the primary call to action. Instead use: "reach out to us here: [link]", "share your details here: [link]", "we're looking into this and will get back to you shortly", or "tag us and we'll take it from there".
6. DIRECT ANSWER RULE: If the customer asks a specific question (e.g. carrier pickup, depot collection), address that specific question directly instead of giving generic boilerplate asking them to explain again.
7. ACKNOWLEDGE PRIOR STEPS: If the customer states they already emailed, called, or replied, acknowledge their effort explicitly. Do not contradict them by telling them to re-email or re-call the same channel.


HISTORICAL EXAMPLES (Use these to ground your policy):
{context_str}

CUSTOMER MESSAGE:
"{customer_text}"
"""

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0, # Determinism is non-negotiable
            response_mime_type="application/json",
            response_schema=ReplyResult,
        ),
    )
    
    return ReplyResult.model_validate_json(response.text)

if __name__ == "__main__":
    # Test the drafter
    test_query = "My blender arrived shattered into a million pieces. I want a refund now."
    res = draft_reply(test_query)
    print(f"\nDrafted Reply: {res.draft_reply}")
    print(f"Safety Check: {res.policy_adherence_check}")