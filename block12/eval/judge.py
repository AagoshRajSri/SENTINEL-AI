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
    reasoning: str = Field(..., description="Chain-of-thought analysis before scoring")
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

MICRO-RULE 6 — TONE-DEAF PENALTY (NEW):
  If the customer's message is clearly distressing, emotionally charged, or severe (loss, anger, urgency,
  trauma, criminal act) AND the reply's OPENING LINE is a procedural instruction or rule
  (e.g. "Please don't share...", "We recommend checking...", "We need your details first...",
  "Please provide your order ID...") with NO empathy acknowledgement before it —
  apply a mandatory -1 penalty to overall_score.
  A technically correct action does NOT redeem an emotionally dismissive or procedurally robotic opening.
  This rule applies EVEN if the rest of the reply is polite.
  EXCEPTION: If the message severity is LOW (positive feedback, casual question), this rule does not apply.

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
Each example shows the EXACT reasoning structure you must follow before scoring.
Write your reasoning field first, then output the four integer scores.

EXAMPLE A — CRITICAL + EXHAUSTED CHANNEL → Score 2
Customer: "Are you able to get in contact with DPD? I had a notification my parcel won't be delivered and have tried to contact them multiple times with no answer."
Reply: "We're sorry. We recommend checking your tracking info here: [link] for any updates or direct carrier contact options."
{{
  "reasoning": "SEVERITY: HIGH — undelivered parcel, carrier unresponsive. EXHAUSTED CHANNEL CHECK: Customer explicitly said DPD doesn't answer calls. Reply sends them to a tracking link that includes 'direct carrier contact options' — this is the same exhausted channel restated. MICRO-RULE 2 hard cap: max score is 2. TONE: Polite, but politeness cannot rescue the substance failure. CONCLUSION: Score capped at 2.",
  "brand_fidelity": 3,
  "groundedness": 4,
  "actionability_safety": 2,
  "overall_score": 2
}}

EXAMPLE B — CRIMINAL EMERGENCY, DM NOT ENOUGH → Score 1
Customer: "your delivery guy in Lincoln park NJ took my friends puppy. Need help now!!! Police next call..."
Reply: "We are so concerned. Please reach out to us via DM immediately with your details so we can look into this."
{{
  "reasoning": "SEVERITY: CRITICAL — stolen pet, police mentioned, criminal act by driver. TIER ASSESSMENT: DM redirect (TIER 2) is what Amazon sends for a delayed parcel. This is a criminal emergency requiring explicit acknowledgment of severity and immediate human escalation. MICRO-RULE 6 TONE-DEAF PENALTY: Reply opens with 'We are so concerned' but then immediately routes to a standard DM — the same response a bot gives for any query. No named team escalation, no hotline, no acknowledgment of the criminal nature. CONCLUSION: Reply is dangerously inadequate. Score 1.",
  "brand_fidelity": 1,
  "groundedness": 4,
  "actionability_safety": 1,
  "overall_score": 1
}}

EXAMPLE C — EXHAUSTED CHANNEL, LINK STILL COUNTS AS EXHAUSTED → Score 2
Customer: "WTF! IT'S NOT POSSIBLE TO CONNECT TO YOUR CS TEAM, EITHER ON 1800-XXX OR THROUGH THE APP"
Reply: "I'm sorry you're having trouble. Please share your details here: [CS link] so we can take a closer look."
{{
  "reasoning": "SEVERITY: HIGH — customer explicitly exhausted phone AND app. EXHAUSTED CHANNEL CHECK: A general CS link routes to the same digital system the customer said is broken. MICRO-RULE 2 hard cap: max 2. TONE: Empathetic opening, but empathy cannot overcome the substance failure. No new concrete escalation offered. CONCLUSION: Score capped at 2.",
  "brand_fidelity": 3,
  "groundedness": 4,
  "actionability_safety": 2,
  "overall_score": 2
}}

EXAMPLE D — PII SCOLDING INSTEAD OF HELPING → Score 1
Customer: "@AmazonHelp my order id is 403-XXXXXXX, I haven't received my cashback"
Reply: "Please don't provide your order details publicly. Our page is visible to the public. Please share your details privately here: [link]"
{{
  "reasoning": "SEVERITY: MEDIUM — missing cashback. MICRO-RULE 1 (PII): Customer shared an order ID. The reply (a) warns them, BUT (b) completely ignores the stated issue of missing cashback. No acknowledgment of 'we are looking into your cashback issue.' MICRO-RULE 6 TONE-DEAF PENALTY: Reply opens with a scolding instruction ('Please don't provide...') — this is procedural, not empathetic. The customer feels lectured, not helped. This is PII scolding — it lacks assurance that their issue is being looked into. Score 1.",
  "brand_fidelity": 2,
  "groundedness": 4,
  "actionability_safety": 1,
  "overall_score": 1
}}

EXAMPLE E — MISSING HIGH-VALUE ITEM, DM ONLY → Score 3
Customer: "My laptop was not delivered even though tracking says it was delivered."
Reply: "We're sorry to hear this! Please send us a DM with your order details so we can investigate."
{{
  "reasoning": "SEVERITY: HIGH — missing laptop (high-value item), tracking mismatch. TIER ASSESSMENT: DM redirect is TIER 2. For HIGH severity with a clear, specific issue, a TIER 1 direct link was expected. The customer has to send a DM, wait for a reply, then provide details — extra friction. MICRO-RULE 6: Opening is appropriately empathetic ('We're sorry to hear this!'). No tone-deaf penalty. ACTION: Correct intent, wrong tier. CONCLUSION: Acceptable but below expectation for severity. Score 3.",
  "brand_fidelity": 4,
  "groundedness": 5,
  "actionability_safety": 3,
  "overall_score": 3
}}

EXAMPLE F — MISSING HIGH-VALUE ITEM, DIRECT LINK → Score 5
Customer: "My laptop was not delivered even though tracking says it was delivered."
Reply: "We're truly sorry about this. Please share your order details directly here: [link] and our team will investigate right away."
{{
  "reasoning": "SEVERITY: HIGH — missing laptop. TIER ASSESSMENT: Direct link (TIER 1). Customer can submit order details immediately without a back-and-forth DM exchange. MICRO-RULE 6: 'We're truly sorry' is warm and empathetic — appropriate opener. Substance matches severity. CONCLUSION: Fully correct response. Score 5.",
  "brand_fidelity": 5,
  "groundedness": 5,
  "actionability_safety": 5,
  "overall_score": 5
}}

EXAMPLE G — RESOLVED SITUATION, WARM ACKNOWLEDGEMENT → Score 5
Customer: "It's okay. I don't need a call. I made him deliver the order. But the behavior was extremely unprofessional."
Reply: "I'm sorry to hear about the unprofessional behavior. I'll make sure to forward your feedback internally for review."
{{
  "reasoning": "SEVERITY: LOW — situation self-resolved. MICRO-RULE 3: Customer is not asking for further help; they are venting and sharing feedback. The correct and complete response is a warm acknowledgment that takes their feedback on board. A link or action step is NOT needed and NOT missing. Reply correctly acknowledges the professionalism concern and commits to internal forwarding. CONCLUSION: Score 5. Do NOT penalise for lacking a link.",
  "brand_fidelity": 5,
  "groundedness": 5,
  "actionability_safety": 5,
  "overall_score": 5
}}

EXAMPLE H — DIRECT FACTUAL ANSWER → Score 5
Customer: "I clicked next day delivery just after the cut-off time, is there any chance it comes today?"
Reply: "Because the order was placed after the cut-off time, it will arrive on the estimated delivery date shown at checkout."
{{
  "reasoning": "SEVERITY: LOW — general enquiry, no frustration. MICRO-RULE 4: Customer asked a specific, answerable factual question. Reply answers it directly and accurately. No further action needed. A link would add noise, not value. CONCLUSION: Score 5. Do NOT penalise for lacking a link.",
  "brand_fidelity": 5,
  "groundedness": 5,
  "actionability_safety": 5,
  "overall_score": 5
}}

EXAMPLE I — UNEXPECTED DELIVERY, WARM HELPFUL REPLY → Score 5
Customer: "We have something arriving tomorrow but we didn't order anything!! 😱"
Reply: "Hi there! Surprises can be fun, but we understand the concern! You can check your recent orders or reach out to us here so we can take a closer look: [link]"
{{
  "reasoning": "SEVERITY: LOW/MEDIUM — unexpected delivery, mild concern. TIER: Direct link (TIER 1). MICRO-RULE 6: 'Surprises can be fun, but we understand the concern!' — warm, human, appropriate opener. Reply is reassuring, investigative, and actionable. CONCLUSION: Score 5.",
  "brand_fidelity": 5,
  "groundedness": 5,
  "actionability_safety": 5,
  "overall_score": 5
}}

EXAMPLE J — NON-ACTIONABLE MESSAGE, POLITE HOLDING REPLY → Score 4
Customer: "@AmazonHelp [link only, no text]"
Reply: "Could you please share more details about the issue you are experiencing through the link provided earlier so we can assist you further?"
{{
  "reasoning": "SEVERITY: UNKNOWN — no usable information. MICRO-RULE 5: Asking for clarification is the only correct response. Reply does this politely while providing a contact link. Scores 4 not 5 because clarification is a one-step-removed solution, but it is the BEST possible response given zero input. CONCLUSION: Score 4.",
  "brand_fidelity": 4,
  "groundedness": 5,
  "actionability_safety": 4,
  "overall_score": 4
}}

EXAMPLE K — AMBIGUOUS MESSAGE, DM REDIRECT → Score 3
Customer: "@AmazonHelp [URL only, no text]"
Reply: "Could you please share more details about the issue via DM so we can assist you further?"
{{
  "reasoning": "SEVERITY: UNKNOWN. MICRO-RULE 5: No usable information — asking for clarification is correct and should not be penalised. However, reply asks for a DM (TIER 2) when a direct link is more convenient. Compare to Example J (TIER 1 link) which scores 4. Scores 3 because DM adds friction vs a direct link. CONCLUSION: Score 3.",
  "brand_fidelity": 4,
  "groundedness": 5,
  "actionability_safety": 3,
  "overall_score": 3
}}

EXAMPLE L — PUBLIC PII, PROACTIVELY PROTECTED → Score 5
Customer: "@AmazonHelp delivery attempt at 11:54 pm? Looks like courier fraud. Please check Tracking #913115081942"
Reply: "That was unexpected. Please don't provide your tracking number here as it is personal information. Kindly share your details privately here: [link] and we'll look into this delivery attempt right away."
{{
  "reasoning": "SEVERITY: HIGH — suspected courier fraud, late-night delivery. MICRO-RULE 1 (PII): Reply (a) warns the customer about the tracking number, (b) provides a private link, AND (c) crucially adds 'we'll look into this delivery attempt right away' — assuring the customer their issue IS being addressed. This is MICRO-RULE 1 done correctly. MICRO-RULE 6: 'That was unexpected' is warm, not robotic. CONCLUSION: Score 5.",
  "brand_fidelity": 5,
  "groundedness": 5,
  "actionability_safety": 5,
  "overall_score": 5
}}

EXAMPLE M — ROBOTIC BRUSH-OFF, IGNORES SPECIFICS → Score 2
Customer: "I have ordered Maharaja Whiteline Juicer... and this is what comes out of box..... [picture of stone]"
Reply: "We're so sorry to see this! We'd like to look into this for you. Please share your details here: [link] so we can help."
{{
  "reasoning": "SEVERITY: HIGH — received a stone instead of a product, photo evidence mentioned. SPECIFICITY CHECK: Reply is completely generic — could apply to any complaint. It does not name what happened ('received wrong item'), does not reference the evidence ('video/photo'), does not ask for the Order ID specifically. MICRO-RULE 6: 'We're so sorry to see this!' is warm, but warmth alone cannot rescue the robotic substance. CONCLUSION: Robotically dismissive of a specific, severe situation. Score 2.",
  "brand_fidelity": 3,
  "groundedness": 3,
  "actionability_safety": 2,
  "overall_score": 2
}}

EXAMPLE N — CUSTOMER ASKS FOR PHONE NUMBER, GETS CALLBACK INSTEAD → Score 3
Customer: "@AmazonHelp Can u atleast provide your customer care number so that I can contact them?"
Reply: "We don't have a direct incoming phone number, but you can request a call back or chat directly with our Customer Support team here: [link]"
{{
  "reasoning": "SEVERITY: MEDIUM — customer frustrated, wants phone contact. DIRECT ANSWER: Reply correctly explains there is no direct number and offers an alternative (callback/chat link). This is the best technically possible answer. However, it is one step removed from what the customer asked for. CONCLUSION: Acceptable, not perfect — customer needs one more step. Score 3.",
  "brand_fidelity": 4,
  "groundedness": 5,
  "actionability_safety": 3,
  "overall_score": 3
}}

EXAMPLE O — PII SCOLDING WITHOUT ASSURANCE → Score 2
Customer: "@AmazonHelp Can u please check the order. Trackin no. Is119106082923..."
Reply: "Please don't provide your tracking numbers or order details publicly, as our page is visible to everyone. We'd like to take a closer look at this for you. Please share your details securely here: [link]"
{{
  "reasoning": "SEVERITY: MEDIUM — order tracking issue. MICRO-RULE 1 (PII): Reply warns about PII and provides a private link. HOWEVER, MICRO-RULE 6 TONE-DEAF PENALTY APPLIES: The reply OPENS with a scolding instruction ('Please don't provide...') as the very first sentence. Even though 'We'd like to take a closer look' follows, the opening creates a reprimanded, not-helped feeling. Compare to Example L (score 5) where the issue acknowledgment is woven into the PII warning. CONCLUSION: Score 2.",
  "brand_fidelity": 2,
  "groundedness": 4,
  "actionability_safety": 2,
  "overall_score": 2
}}

EXAMPLE P — HIGH SEVERITY, DM REDIRECT vs DIRECT LINK (3 vs 4 boundary)
Customer: "My laptop was not delivered. Tracking says it was delivered 3 days ago."
Reply: "We're really sorry about this stressful situation. Please DM us your order details so our team can investigate urgently."
{{
  "reasoning": "SEVERITY: HIGH — missing laptop, 3 days elapsed. MICRO-RULE 6: 'Really sorry about this stressful situation' — empathetic opener, no tone-deaf penalty. TIER: DM redirect (TIER 2). For HIGH severity (high-value item, multi-day delay), TIER 1 direct link was expected. DM adds friction: customer must wait for agent reply before submitting details. Compared to Example F (score 5, same scenario with direct link), this scores lower due to the DM tier. CONCLUSION: Correct intent, suboptimal tier. Score 3.",
  "brand_fidelity": 4,
  "groundedness": 5,
  "actionability_safety": 3,
  "overall_score": 3
}}

EXAMPLE Q — VENTING CUSTOMER, WARM REPLY, NO LINK (4 vs 5 boundary)
Customer: "Finally sorted out. After 6 calls it got resolved. Never shopping here again probably."
Reply: "We're really sorry it took 6 calls to get this sorted — that's not the experience we want you to have. Your feedback has been noted, and we truly hope to serve you better in the future."
{{
  "reasoning": "SEVERITY: LOW — situation self-resolved. MICRO-RULE 3: Customer is not asking for help; they are venting and expressing disappointment. Warm acknowledgment is the correct and complete response. Reply explicitly acknowledges the 6-call effort ('that's not the experience we want you to have') which shows it is not generic. No link or action step is needed. MICRO-RULE 6: No tone-deaf opening. CONCLUSION: The specificity and warmth are excellent. Score 5. Do NOT penalise for lacking a link.",
  "brand_fidelity": 5,
  "groundedness": 5,
  "actionability_safety": 5,
  "overall_score": 5
}}

EXAMPLE R — EXHAUSTED CHANNEL: SECOND LINK NOT THE SAME SYSTEM (2 vs 3 boundary)
Customer: "I've tried the app and the website CS chat already. Both say the same thing. Need human help."
Reply: "We're sorry for the frustration. As an alternative, you can request a direct callback from our specialist team here: [callback-specific link] — a human agent will call you within 24 hours."
{{
  "reasoning": "SEVERITY: HIGH — customer explicitly exhausted app AND website chat. MICRO-RULE 2 ANALYSIS: The reply offers a 'callback from a specialist team' via a specific callback link — this is a NEW, CONCRETE escalation path not mentioned or tried before. It is not a generic CS link. MICRO-RULE 6: 'Sorry for the frustration' is an appropriate, warm opener. CONCLUSION: Reply breaks the loop by offering a genuinely new channel (phone callback vs digital chat). MICRO-RULE 2 hard cap does NOT apply here because a new channel IS offered. Score 3 — not a perfect 5 because 24-hour wait is a significant gap for a high-severity issue, and the reply doesn't urgently acknowledge that.",
  "brand_fidelity": 4,
  "groundedness": 5,
  "actionability_safety": 3,
  "overall_score": 3
}}

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
Format your response as a valid JSON object. You MUST write the reasoning field FIRST,
then the four integer scores. This forces you to think before scoring.
{
  "reasoning": "<your step-by-step analysis: severity, tier, rule checks, tone assessment, conclusion>",
  "brand_fidelity": <int 1-5>,
  "groundedness": <int 1-5>,
  "actionability_safety": <int 1-5>,
  "overall_score": <int 1-5>
}
The reasoning field is REQUIRED. Do not skip it.
"""
    for model_name in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]:
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
                # Let call_with_retry's outer backoff handle rate limits on primary model.
                # Only fall through to the next model if it's still rate-limited.
                print(f"  [INFO] {model_name} rate limited, waiting 30s then trying fallback...")
                time.sleep(30)
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

            "llm_reasoning": judge_res.reasoning,
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