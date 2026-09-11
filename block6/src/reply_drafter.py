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
3. DO NOT ask the customer to publicly share sensitive info (passwords, card numbers). Instruct them to DM (Direct Message) if account details are needed.

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