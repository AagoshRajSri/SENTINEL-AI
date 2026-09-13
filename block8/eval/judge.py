# eval/judge.py
import os
import json
import sys
import time
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv

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
  LOW      : Positive feedback, general enquiry, casual message
  MEDIUM   : Delayed delivery, missing item, billing question
  HIGH     : Wrong/damaged item received, repeated failed contacts, account issue, time-sensitive (birthday/event)
  CRITICAL : Suspected theft, criminal act by driver, safety threat, missing high-value item (£200+), customer mentions police

═══════════════════════════════════════════════
STEP 2: EVALUATE ON FIVE CORE DIMENSIONS
═══════════════════════════════════════════════

[A] POLITENESS AS BASELINE, NOT A SCORER
  - Politeness, professional tone, and empathetic openers are expected baselines (0 points added). They do NOT elevate a score.
  - A reply that sounds warm but lacks substantive problem-solving or progression must score 2-3 at most.

[B] NOVELTY OF ACTION
  - Does the action introduce something new and useful, or does it send the customer back to an exhausted channel/action they already tried?
  - Repeating a failed channel without a meaningful escalation scores 1.

[C] ACTION CONVENIENCE TIER
  - Evaluate the helpfulness and convenience of the provided next step using this strict hierarchy:
    1. Direct Link / Self-Service Tool (Highest convenience: direct order link, tracking page, specific form).
    2. Direct Message (DM) Redirect / Secure Link (Moderate convenience: appropriate when private account details or authentication are required).
    3. Vague / Generic "Contact Us" (Lowest convenience: unhelpful general advice).
  - For HIGH/CRITICAL cases, vague responses or low-tier actions are heavily penalized.

[D] EMERGENCY / CRIMINAL SITUATIONS
  - CRITICAL / Criminal / Safety threats (e.g. theft, police, driver misconduct, safety threats): Standard automated or generic canned responses (like standard "Please DM us") always score 1. They require immediate escalation language or specialist intervention.

[E] PII PROTECTION & PRIVACY RULE
  - Never allow or encourage customers to share sensitive personal identifiable information (PII), passwords, or payment info publicly in tweets/replies.
  - The reply MUST guide customers to share sensitive account details privately via secure link or DM, never publicly. Directing to DM for account verification is correct behavior and should not be penalized.

[F] MICRO-RULE 1 — PUBLIC PII PROTECTION BONUS & TRACKING NUMBER EXEMPTION
  - If the customer has already shared highly sensitive data (account number, phone number, full address, passwords) in a public tweet/message, and the reply (a) explicitly warns the customer not to share such data publicly AND (b) immediately provides a private/secure link for them to share it safely instead — reward this as a high-quality, convenient action.
  - EXCEPTION: Tracking numbers and order numbers are NOT highly sensitive PII for social media support. If a drafted reply scolds a customer for sharing a tracking number or refuses to look it up, penalize the reply (score 2 or 3 at most).

[G] MICRO-RULE 2 — ALL CHANNELS EXHAUSTED / WAIT LOOP HARD CAP
  - If the customer explicitly states they have already tried support channels, received no resolution, or explicitly says "no response yet" or implies they are currently waiting on the team — and the drafted reply responds by directing them to "reach out again" via a generic link or DM without offering a new concrete escalation path — cap the score at 2.
  - A score of 2 is permitted only if the reply at least acknowledges the previous failed attempts. A score of 1 applies if the reply completely ignores the exhausted context and gives a generic redirect.

[H] MICRO-RULE 3 — THE ROBOTIC BRUSH-OFF PENALTY
  - If the customer describes a severe or highly specific problem (e.g., received a stone instead of a laptop, missing release day game, wrong item), a polite but generic "Sorry to hear this, please share your details here: [link]" is a brush-off. 
  - To score a 4 or 5, the reply MUST explicitly acknowledge the specific problem and ask for the *specific* missing piece of data (like Order ID) needed to investigate. If it relies on a generic "share your details", cap the score at 3.

═══════════════════════════════════════════════
STEP 3: ASSIGN SCORES
═══════════════════════════════════════════════

SUB-SCORES (brand_fidelity, groundedness, actionability_safety): Rate 1-5 on each criterion independently.

OVERALL SCORE RUBRIC:
  5 = EXCELLENT  : Directly addresses situation, uses high convenience tier (direct link/tool), novel action, grounded, safe.
  4 = GOOD       : Mostly correct and useful; minor omission or moderate convenience tier (DM when appropriate).
  3 = ACCEPTABLE : Relevant and safe, but vague or partial. Appropriate for low-severity triage.
  2 = POOR       : Misses key context, repeats failed channel, or provides weak generic action for high severity.
  1 = FAILING    : Unsafe, misleading, automated response to criminal/emergency case, or sending customer back to exhausted channel.

CRITICAL RULES (these override everything else):
  - Emergency/criminal case + generic automated response = maximum score of 1.
  - Customer stated they already tried a specific channel + reply sends them back to that channel = maximum score of 1.
  - Polite tone alone CANNOT push a score above 3 if the substance fails.
  - PII protection rule: sensitive info must be guided to private channels/links, never shared publicly.

═══════════════════════════════════════════════
FEW-SHOT CALIBRATION EXAMPLES
═══════════════════════════════════════════════

EXAMPLE A — CRITICAL + EXHAUSTED CHANNEL → Score 1
Customer: "Are you able to get in contact with DPD? I had a notification my parcel won't be delivered and have tried to contact them multiple times with no answer."
Reply: "We're sorry. We recommend checking your tracking info here: [link] for any updates or direct carrier contact options."
Score: 1
Reason: Customer asked Amazon to contact DPD on their behalf. They already said DPD doesn't answer. Reply sends them to a tracking link — no new action, sends back to an exhausted channel. Polite but useless.

EXAMPLE B — CRIMINAL EMERGENCY, DM NOT ENOUGH → Score 1
Customer: "your delivery guy in Lincoln park NJ took my friends puppy. Need help now!!! Police next call..."
Reply: "We are so concerned. Please reach out to us via DM immediately with your details so we can look into this."
Score: 1
Reason: Criminal emergency. A DM redirect is what Amazon sends for a delayed parcel. This requires explicit acknowledgment of severity and immediate human escalation — not a copy-paste DM prompt.

EXAMPLE C — MISSING HIGH-VALUE ITEM, DM ONLY → Score 3
Customer: "My laptop was not delivered even though tracking says it was delivered."
Reply: "We're sorry to hear this! Please send us a DM with your order details so we can investigate."
Score: 3
Reason: Takes action but asks for DM (TIER 2) when a direct link where the customer can submit order details immediately would be far more convenient. HIGH severity — a TIER 1 direct link was expected.

EXAMPLE D — MISSING HIGH-VALUE ITEM, DIRECT LINK → Score 5
Customer: "My laptop was not delivered even though tracking says it was delivered."
Reply: "We're truly sorry about this. Please share your order details directly here: [link] and our team will investigate right away."
Score: 5
Reason: Acknowledges the problem, offers a direct link (TIER 1). No extra navigation required. Grounded, actionable, appropriate to severity.

EXAMPLE E — SCOLDING FOR TRACKING NUMBER → Score 2
Customer: "@AmazonHelp Can u please check the order. Tracking no. Is119106082923"
Reply: "Please don't provide your tracking numbers or order details publicly... Please share your details securely here: [link]"
Score: 2
Reason: Tracking numbers are not highly sensitive PII. The bot scolded the customer and refused to acknowledge the tracking number, forcing them to jump through another hoop.

EXAMPLE F — ROBOTIC BRUSH-OFF → Score 2
Customer: "I have ordered Maharaja Whiteline Juicer... and this is what comes out of box..... [picture of stone]"
Reply: "We're so sorry to see this! We'd like to look into this for you. Please share your details here: [link] so we can help."
Score: 2
Reason: Customer stated a highly specific, severe issue (received a stone). The reply just says "share your details" generically, which feels like a robotic brush-off. It should ask for the Order ID specifically.

EXAMPLE F — PACKAGE CLAIM DISMISSED WITH QUESTION → Score 2
Customer: "my package never came. I'm on Amazon prime too.. Smh"
Reply: "Oh no! Have we missed the provided delivery date located in Your Orders: [link]"
Score: 2
Reason: Customer made a factual claim. Reply responds with a question that implies the customer may be confused rather than acknowledging and acting on their claim. Also TIER 3 — no direct action, just a suggestion to check something.

EXAMPLE G — AMBIGUOUS MESSAGE, TRIAGE CORRECT → Score 3
Customer: "@AmazonHelp [URL only, no text]"
Reply: "Could you please share more details about the issue via DM so we can assist you further?"
Score: 3
Reason: No usable information. Asking for clarification is correct and should not be penalised. Scores 3 not 1 because clarification is the appropriate and only possible action here.

EXAMPLE H — POSITIVE FEEDBACK → Score 5
Customer: "Thanks for leaving my parcels in a safe place."
Reply: "We're so glad to hear your parcels arrived safely! Thanks for letting us know!"
Score: 5
Reason: LOW severity, positive message. A warm acknowledgement is the complete and correct response.

EXAMPLE I — PUBLIC PII SHARED, PROACTIVELY PROTECTED → Score 5
Customer: "@AmazonHelp delivery attempt at 11:54 pm? Looks like courier fraud. Please check Tracking #913115081942"
Reply: "That was unexpected. Please don't provide your tracking number here as it is personal information. Kindly share your details privately here: [link] and we'll look into this delivery attempt right away."
Score: 5
Reason: Customer shared a sensitive tracking number publicly. Reply (a) warns them explicitly not to share it publicly, (b) provides a direct private link immediately. This is MICRO-RULE 1 in action — a protective, high-convenience response. Do NOT score this as generic deflection.

EXAMPLE J — ALL CHANNELS EXHAUSTED, SAME CHANNEL SUGGESTED → Score 1
Customer: "WTF! It's not possible to connect to your CS team, either on 1800-XXX or through the app!"
Reply: "I'm sorry for the frustration! Please drop us a DM with your details so we can help you."
Score: 1
Reason: Customer explicitly said they cannot reach CS on phone or app. Reply suggests a DM — another standard support channel requiring the customer to navigate yet another platform. No escalation, no new concrete path. MICRO-RULE 2 hard cap applies — this scores 1.

═══════════════════════════════════════════════
NOW EVALUATE
═══════════════════════════════════════════════
CUSTOMER MESSAGE: "{customer_text}"
DRAFTED REPLY: "{drafted_reply}"
"""

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=JudgeScore,
        ),
    )

    return JudgeScore.model_validate_json(response.text)


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