# eval/judge.py
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

from sklearn.metrics import cohen_kappa_score

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import your drafter to get the AI's reply
from src.reply_drafter import draft_reply

load_dotenv()


# Enforce the 1-5 grading rubric
class JudgeScore(BaseModel):
    brand_fidelity: int = Field(..., ge=1, le=5, description="Brand tone match")
    groundedness: int = Field(..., ge=1, le=5, description="No hallucinations")
    actionability_safety: int = Field(..., ge=1, le=5, description="No PII leaks")
    overall_score: int = Field(..., ge=1, le=5, description="Final score 1-5")


def call_with_retry(func, *args, **kwargs):
    """Executes a function with backoff on rate-limit or transient network errors."""
    max_retries = 10
    # Keywords that indicate a transient error worth retrying
    TRANSIENT_SIGNALS = (
        "429", "RESOURCE_EXHAUSTED", "Quota",       # API rate limits
        "ConnectError", "getaddrinfo", "RemoteDisconnected",  # Network blips
        "Connection reset", "timed out", "timeout",
    )
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(type(e).__name__) + " " + str(e)
            if any(sig in err_str for sig in TRANSIENT_SIGNALS):
                wait_secs = 30 * (attempt + 1)  # Progressive backoff: 30s, 60s, 90s…
                print(
                    f"  [WARNING] Transient error (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}. Waiting {wait_secs}s before retry..."
                )
                time.sleep(wait_secs)
            else:
                raise e
    return func(*args, **kwargs)


def run_llm_judge(customer_text: str, drafted_reply: str) -> JudgeScore:
    """Uses Gemini to act as a strict QA auditor for our generated replies."""

    prompt = f"""You are a strict, senior QA auditor evaluating AI-drafted customer-support replies for @AmazonHelp.

Your job is to rigorously score how well the reply actually helps the customer based on substance, actionability, safety, and operational rules.

═══════════════════════════════════════════════
STEP 1: UNDERSTAND THE CUSTOMER'S CONTEXT & SEVERITY
═══════════════════════════════════════════════
Classify severity:
  LOW      : Positive feedback, general enquiry, casual message, situation already resolved by the customer
  MEDIUM   : Delayed delivery, missing item, billing question
  HIGH     : Wrong/damaged item received, repeated failed contacts, account issue, time-sensitive (birthday/event)
  CRITICAL : Suspected theft, criminal act by driver, safety threat, missing high-value item (£200+), customer mentions police

═══════════════════════════════════════════════
STEP 2: APPLY THE ACTION-TIER FRAMEWORK
═══════════════════════════════════════════════
  TIER 1 — DIRECT LINK     : Reply provides a clickable URL for the customer to act immediately. Best for MEDIUM/HIGH/CRITICAL.
  TIER 2 — DM REDIRECT     : Reply asks customer to send a private DM with details. Acceptable only when no direct link is available.
  TIER 3 — PASSIVE ADVICE  : Reply only suggests checking something (e.g. "check your orders"). Weakest action.
  TIER 0 — NO ACTION NEEDED: Situation resolved, positive feedback, or genuinely ambiguous message. Warm acknowledgement is the COMPLETE correct response.

═══════════════════════════════════════════════
STEP 3: APPLY MICRO-CALIBRATION RULES
═══════════════════════════════════════════════
MICRO-RULE 1 — PII PROTECTION:
  If a customer shared sensitive data (order ID, account number, phone) publicly and the reply
  (a) warns them not to share it and (b) provides a private link → BOOST score. Do NOT penalise this as deflection.
  HOWEVER: If reply scolds the customer harshly or refuses to acknowledge their issue until they resubmit via private link,
  this is unhelpful. Even when protecting PII, the reply MUST be extremely polite and assure the customer that their issue is being looked into (e.g. "check your updates on this link..."). If it acts like a strict robotic scolding, score 1 or 2.

MICRO-RULE 2 — EXHAUSTED CHANNEL HARD CAP (most important rule):
  If the customer EXPLICITLY states they already tried a specific channel (phone, app, email, DM) and the reply
  sends them to ANY standard support channel (DM, CS link, same phone number, app) without offering something NEW
  and CONCRETE (e.g. direct human escalation, callback from a named team, out-of-band contact) → MAXIMUM SCORE IS 2. It brings the customer back to square one without new actions.
  A polished tone CANNOT rescue this. A link to the general CS page still counts as sending them to an exhausted channel.

MICRO-RULE 3 — RESOLVED SITUATIONS (no action needed):
  If the customer's message indicates the situation is ALREADY resolved or they are expressing satisfaction,
  the correct and complete response is a warm acknowledgement. Do NOT penalise it for lacking a link or action step.
  Replies to resolved situations that are warm and graceful → score 4 or 5.

MICRO-RULE 4 — DIRECT FACTUAL ANSWERS:
  If the customer asks a specific factual question and the reply answers it directly and accurately,
  this is HIGH QUALITY. Do NOT penalise it for not including an additional link if the question is fully answered.
  A direct, correct, on-brand factual answer → score 4 or 5.

MICRO-RULE 5 — NON-ACTIONABLE OR AMBIGUOUS MESSAGES:
  If the customer's message has no usable information (e.g. just a URL, an emoji, or a vague tag),
  asking for clarification is the ONLY correct response. Score 3 (not 1) for reasonable clarification requests.
  If the message is just a link/image/tag with no text, a polite holding reply is acceptable → score 3-4.

═══════════════════════════════════════════════
SCORING RUBRIC
═══════════════════════════════════════════════
  5 = EXCELLENT  : Fully resolves or correctly triages the issue. Correct tier for severity. All rules followed.
  4 = GOOD       : Mostly correct. Minor gap (e.g. DM instead of direct link, or slightly generic phrasing).
  3 = ACCEPTABLE : Correct intent but wrong tier, or misses specificity. Customer needs one more step.
  2 = POOR       : Misses key context, repeats failed channel, or provides weak generic action for high severity.
  1 = FAILING    : Unsafe, misleading, automated response to criminal/emergency case, or sending customer back to exhausted channel.

CRITICAL RULES (these override everything else):
  - Emergency/criminal case + generic automated response = maximum score of 1.
  - Customer stated they already tried a specific channel + reply sends them back to that or any equivalent channel = maximum score of 2.
  - Polite tone alone CANNOT push a score above 3 if the substance fails.
  - PII SCOLDING (refusing to help until customer re-submits through private channel) = score 1 or 2, NOT 4 or 5.
  - Resolved situation + warm acknowledgement = score 4 or 5, NOT 2 or 3.
  - Direct factual answer to a direct question = score 4 or 5, NOT 2 or 3.

═══════════════════════════════════════════════
FEW-SHOT CALIBRATION EXAMPLES
═══════════════════════════════════════════════

EXAMPLE A — CRITICAL + EXHAUSTED CHANNEL → Score 2
Customer: "Are you able to get in contact with DPD? I had a notification my parcel won't be delivered and have tried to contact them multiple times with no answer."
Reply: "We're sorry. We recommend checking your tracking info here: [link] for any updates or direct carrier contact options."
Score: 2
Reason: Customer explicitly said DPD doesn't answer. Reply sends them to a tracking link — no new concrete action, redirects to same carrier. Polite but brings them back to square one. MICRO-RULE 2 hard cap applies (Score 2).

EXAMPLE B — CRIMINAL EMERGENCY, DM NOT ENOUGH → Score 1
Customer: "your delivery guy in Lincoln park NJ took my friends puppy. Need help now!!! Police next call..."
Reply: "We are so concerned. Please reach out to us via DM immediately with your details so we can look into this."
Score: 1
Reason: Criminal emergency. A DM redirect is what Amazon sends for a delayed parcel. This requires explicit acknowledgment of severity and immediate human escalation — not a copy-paste DM prompt.

EXAMPLE C — EXHAUSTED CHANNEL, LINK STILL COUNTS AS EXHAUSTED → Score 2
Customer: "WTF! IT'S NOT POSSIBLE TO CONNECT TO YOUR CS TEAM, EITHER ON 1800-XXX OR THROUGH THE APP"
Reply: "I'm sorry you're having trouble. Please share your details here: [CS link] so we can take a closer look."
Score: 2
Reason: Customer explicitly said phone AND app do not work. A general CS link routes them to the same exhausted system. This is MICRO-RULE 2 (Score 2) — no new, concrete escalation path is offered, bringing them back to square one.

EXAMPLE D — PII SCOLDING INSTEAD OF HELPING → Score 1
Customer: "@AmazonHelp my order id is 403-XXXXXXX, I haven't received my cashback"
Reply: "Please don't provide your order details publicly. Our page is visible to the public. Please share your details privately here: [link]"
Score: 1
Reason: The reply refuses to acknowledge or act on the customer's stated issue (missing cashback). Instead it lectures them and forces them through another hoop. This is PII scolding (MICRO-RULE 1 violation) — it lacks the polite assurance that their issue is being looked into.

EXAMPLE E — MISSING HIGH-VALUE ITEM, DM ONLY → Score 3
Customer: "My laptop was not delivered even though tracking says it was delivered."
Reply: "We're sorry to hear this! Please send us a DM with your order details so we can investigate."
Score: 3
Reason: Takes action but asks for DM (TIER 2) when a direct link where the customer can submit order details immediately would be far more convenient. HIGH severity — a TIER 1 direct link was expected.

EXAMPLE F — MISSING HIGH-VALUE ITEM, DIRECT LINK → Score 5
Customer: "My laptop was not delivered even though tracking says it was delivered."
Reply: "We're truly sorry about this. Please share your order details directly here: [link] and our team will investigate right away."
Score: 5
Reason: Acknowledges the problem, offers a direct link (TIER 1). No extra navigation required. Grounded, actionable, appropriate to severity.

EXAMPLE G — RESOLVED SITUATION, WARM ACKNOWLEDGEMENT → Score 5
Customer: "It's okay. I don't need a call. I made him deliver the order. But the behavior was extremely unprofessional."
Reply: "I'm sorry to hear about the unprofessional behavior. I'll make sure to forward your feedback internally for review."
Score: 5
Reason: MICRO-RULE 3 — customer resolved the situation themselves. They are not asking for further help, just venting. A warm acknowledgement that takes note of the feedback is the COMPLETE and correct response. Do NOT penalise for lacking a link or action step.

EXAMPLE H — DIRECT FACTUAL ANSWER → Score 5
Customer: "I clicked next day delivery just after the cut-off time, is there any chance it comes today?"
Reply: "Because the order was placed after the cut-off time, it will arrive on the estimated delivery date shown at checkout."
Score: 5
Reason: MICRO-RULE 4 — customer asked a specific factual question. Reply answers it directly, accurately, and on-brand. No further action needed. Do NOT penalise for lacking a link.

EXAMPLE I — UNEXPECTED DELIVERY, WARM HELPFUL REPLY → Score 5
Customer: "We have something arriving tomorrow but we didn't order anything!! 😱"
Reply: "Hi there! Surprises can be fun, but we understand the concern! You can check your recent orders or reach out to us here so we can take a closer look: [link]"
Score: 5
Reason: LOW/MEDIUM concern. Reply reassures, offers a link to investigate, and stays on-brand. Warm and actionable.

EXAMPLE J — NON-ACTIONABLE MESSAGE, POLITE HOLDING REPLY → Score 4
Customer: "@AmazonHelp [link only, no text]"
Reply: "Could you please share more details about the issue you are experiencing through the link provided earlier so we can assist you further?"
Score: 4
Reason: MICRO-RULE 5 — message has no usable information. Asking politely for clarification while providing a contact link is a reasonable and helpful response.

EXAMPLE K — AMBIGUOUS MESSAGE, DM REDIRECT → Score 3
Customer: "@AmazonHelp [URL only, no text]"
Reply: "Could you please share more details about the issue via DM so we can assist you further?"
Score: 3
Reason: No usable information. Asking for clarification is correct and should not be penalised. Scores 3 not 4 because a direct link is more convenient than a DM redirect for clarification.

EXAMPLE L — PUBLIC PII, PROACTIVELY PROTECTED → Score 5
Customer: "@AmazonHelp delivery attempt at 11:54 pm? Looks like courier fraud. Please check Tracking #913115081942"
Reply: "That was unexpected. Please don't provide your tracking number here as it is personal information. Kindly share your details privately here: [link] and we'll look into this delivery attempt right away."
Score: 5
Reason: Customer shared a sensitive tracking number publicly. Reply (a) warns them, (b) provides a private link, (c) acknowledges the concern. This is MICRO-RULE 1 in action — a protective, high-convenience response.

EXAMPLE M — ROBOTIC BRUSH-OFF, IGNORES SPECIFICS → Score 2
Customer: "I have ordered Maharaja Whiteline Juicer... and this is what comes out of box..... [picture of stone]"
Reply: "We're so sorry to see this! We'd like to look into this for you. Please share your details here: [link] so we can help."
Score: 2
Reason: Customer stated a highly specific, severe issue (received a stone instead of product). Reply gives a fully generic response without naming what happened or asking for the Order ID specifically. Robotically dismissive.

EXAMPLE N — CUSTOMER ASKS FOR PHONE NUMBER, GETS CALLBACK INSTEAD → Score 3
Customer: "@AmazonHelp Can u atleast provide your customer care number so that I can contact them?"
Reply: "We don't have a direct incoming phone number, but you can request a call back or chat directly with our Customer Support team here: [link]"
Score: 3
Reason: Customer asked for a phone number. Reply correctly explains there isn't one and offers an alternative (callback/chat). This is acceptable but not perfect — it doesn't fully resolve the frustration and is a one-step-removed solution.


EXAMPLE O — PII SCOLDING WITHOUT ASSURANCE → Score 2
Customer: "@AmazonHelp Can u please check the order. Trackin no. Is119106082923..."
Reply: "Please don't provide your tracking numbers or order details publicly, as our page is visible to everyone. We'd like to take a closer look at this for you. Please share your details securely here: [link]"
Score: 2
Reason: While the bot correctly identifies PII, it opens with a scolding tone ("Please don't provide...") instead of addressing the issue politely and assuring the customer first. It feels robotic and dismissive.

═══════════════════════════════════════════════
NOW EVALUATE
═══════════════════════════════════════════════
CUSTOMER MESSAGE: "{customer_text}"
DRAFTED REPLY: "{drafted_reply}"
"""


    # --- GROQ API IMPLEMENTATION ---
    # Model: qwen/qwen3.8-27b — fast, high rate limits, reliable JSON output
    # Fallback: groq/compound-mini if qwen hits issues
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    groq_prompt = prompt + """
Format your response as a valid JSON object with the following integer keys (1 to 5):
{
  "brand_fidelity": <int 1-5>,
  "groundedness": <int 1-5>,
  "actionability_safety": <int 1-5>,
  "overall_score": <int 1-5>
}
"""
    for model_name in ["qwen/qwen3.8-27b", "groq/compound-mini"]:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a strict customer support quality judge. Always output valid JSON."},
                    {"role": "user", "content": groq_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return JudgeScore.model_validate_json(response.choices[0].message.content)
        except Exception as e:
            if "RateLimit" in type(e).__name__ or "429" in str(e):
                print(f"  [INFO] {model_name} rate limited, trying fallback...")
                time.sleep(5)
                continue
            raise e
    raise RuntimeError("All judge models failed.")

    # --- PREVIOUS GEMINI IMPLEMENTATION (COMMENTED OUT) ---
    #     api_key=os.getenv("GEMINI_API_KEY")
    # )
    #     model="gemini-3.5-flash",
    #     contents=prompt,
    #     config=types.GenerateContentConfig(
    #         temperature=0.0,
    #         response_mime_type="application/json",
    #         response_schema=JudgeScore,
    #     ),
    # )
    # return JudgeScore.model_validate_json(response.text)


def main():
    csv_path = "eval/grading_sheet.csv"

    # ---------------------------------------------------------
    # PHASE 2: Human scores already exist -> calculate Kappa
    # ---------------------------------------------------------
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Only run Phase 2 if all 45 rows are drafted
        if len(df) == 45:
            # Check if human scores are missing
            if "human_overall_score" not in df.columns or df["human_overall_score"].isnull().any():
                filled_count = df["human_overall_score"].notnull().sum() if "human_overall_score" in df.columns else 0
                print(
                    f"\n[INFO] {csv_path} already exists ({filled_count}/{len(df)} human scores filled).\n"
                    f"YOUR TASK: Open {csv_path}, enter your 1-5 rating in the 'human_overall_score' column "
                    f"for all rows, save the file, and run 'python eval/judge.py' again to compute Cohen's Kappa!"
                )
                return

            # Calculate Linear Weighted Cohen's Kappa (appropriate for ordinal 1-5 scales)
            kappa = cohen_kappa_score(
                df["llm_overall_score"],
                df["human_overall_score"],
                weights="linear"
            )

            exact_matches = (df["llm_overall_score"] == df["human_overall_score"]).sum()
            near_matches = (abs(df["llm_overall_score"] - df["human_overall_score"]) <= 1).sum()
            total = len(df)

            print("\n=== LLM-AS-JUDGE AGREEMENT ===")
            print(f"Weighted Kappa Score (linear): {round(kappa, 3)}")
            print(f"Exact Match Rate:              {exact_matches}/{total} ({round(100*exact_matches/total, 1)}%)")
            print(f"Within-1-Point Rate:           {near_matches}/{total} ({round(100*near_matches/total, 1)}%)")
            print(
                "\nKappa scale: < 0 useless | 0.01-0.20 slight | 0.21-0.40 fair | "
                "0.41-0.60 moderate | 0.61-0.80 substantial | > 0.80 almost perfect"
            )
            return

    # ---------------------------------------------------------
    # PHASE 1: Generate drafts + LLM Judge scores
    # ---------------------------------------------------------
    print(
        "Drafting replies and generating Judge scores. "
        "Pacing requests to respect Gemini API rate limits..."
    )

    with open("eval/golden_test.json", "r") as f:
        golden_set = json.load(f)[:45]

    records = []
    start_index = 0
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        if "llm_overall_score" in df_existing.columns and len(df_existing) < 45:
            records = df_existing.to_dict("records")
            start_index = len(records)
            print(f"Resuming from row {start_index + 1}...")

    for i in range(start_index, len(golden_set)):
        row = golden_set[i]
        print(f"Processing {i+1}/45...")

        cust_text = row["text_customer"]

        # 1. Draft the reply using our RAG drafter
        draft_res = call_with_retry(draft_reply, cust_text)
        reply = draft_res.draft_reply

        time.sleep(3)  # Flash-lite pacing

        # 2. Ask Gemini to judge the reply
        judge_res = call_with_retry(run_llm_judge, cust_text, reply)

        time.sleep(3)  # Flash-lite pacing

        records.append({
            "customer_text": cust_text,
            "drafted_reply": reply,

            "llm_fidelity": judge_res.brand_fidelity,
            "llm_groundedness": judge_res.groundedness,
            "llm_safety": judge_res.actionability_safety,
            "llm_overall_score": judge_res.overall_score,

            # You will manually fill this later
            "human_overall_score": None
        })

        # Save progress incrementally so work isn't lost if interrupted
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)

    print(f"\nSuccess! Saved 45 evaluated cases to {csv_path}.")
    print(
        "YOUR TASK: Open the CSV. Hide or ignore the LLM "
        "columns (to avoid anchoring). Type your own 1-5 "
        "score in the 'human_overall_score' column. "
        "Then run this script again."
    )


if __name__ == "__main__":
    main()