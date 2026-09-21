# Sentinel-AI: Evaluation & Architecture Report

## Section 1: Problem Framing

**The Problem:** Large-scale enterprise customer support systems face an immense volume of inbound messages. While many are routine queries (e.g., "Where is my package?"), a critical minority represent severe brand risks, legal threats, safety issues, or deeply frustrated customers. Handling these requires precise triage.

**What "Good" Looks Like:** An effective support triage and drafting system must balance speed, safety, and customer experience. It needs to:
- Identify the correct intent accurately.
- Escalate high-risk, sensitive, or complex issues to human agents immediately without friction.
- Draft empathetic, contextually appropriate responses for routine issues.
- Protect Personally Identifiable Information (PII) by not confirming or repeating it publicly.

**Why Escalation Detection Matters:** Failing to escalate a severe issue (e.g., a customer threatening legal action, reporting a stolen item, or in severe distress) can lead to PR disasters, legal liabilities, and permanent loss of customer trust. The cost of a False Negative (missing a severe issue) is exponentially higher than a False Positive (unnecessarily escalating a routine issue).

**Scope & Limitations:** Sentinel-AI is designed strictly as a triage and drafting assistant. It intentionally **does not** perform irreversible actions such as processing autonomous refunds, changing database records, initiating payments, or modifying account statuses. Its role is to classify, escalate, and draft safe replies for human review.

## Section 2: Results vs. Baselines

The following table compares the Sentinel-AI pipeline against two baseline approaches on our 250-case golden dataset.

| System | Macro-F1 (Intent) | FNR (Escalation) | Latency (sec) | Cohen's Kappa |
|---|---:|---:|---:|---:|
| Baseline-Heuristic | 0.353 | 0.810 | ~0.001 | N/A |
| Baseline-ZeroShot | 0.606 | 0.391 | ~1.500 | N/A |
| Sentinel-AI (Thresh: 0.4) | 0.583 | 0.349 | ~0.001 | 0.558 |

**Analysis of Results:**
- **Best Performers:** Baseline-ZeroShot achieves the highest Macro-F1 for intent classification, leveraging the raw semantic understanding of the LLM. However, Sentinel-AI performs best on Escalation FNR (False Negative Rate) and Latency, while maintaining a competitive Macro-F1.
- **Why Macro-F1 matters:** Our dataset is highly imbalanced (e.g., 94 `Delivery Issues` intent cases vs. only 2 `Severe Escalation` intent cases). Note that these label counts refer to specific intent categories, which is distinct from the 63 overall `gold_escalate=True` edge cases spanning multiple categories. Raw accuracy would misleadingly reward a model that simply guesses the majority class. Macro-F1 ensures performance is measured equally across all rare and common intent categories.
- **Why Escalation FNR matters:** A lower FNR means fewer high-risk issues slip through undetected. Sentinel-AI's continuous calibration score allows us to tune this threshold to 0.4, successfully catching more escalations than the baselines.
- **Latency/Cost Tradeoff:** Baseline-ZeroShot requires an expensive, slow (~1.5s) API call for every query. Sentinel-AI utilizes local FAISS vector retrieval and efficient in-memory caching, reducing operational latency to ~1ms per query in production workflows.
- **Conclusion:** Sentinel-AI does not achieve the highest overall intent Macro-F1; the ZeroShot baseline scores higher (0.606 vs. 0.583). However, Sentinel-AI achieves a lower escalation FNR and substantially lower latency, demonstrating a tradeoff between raw classification performance and operational efficiency/safety. Note that Cohen's Kappa of 0.558 indicates moderate agreement between the LLM judge and human evaluator (it measures grading alignment, NOT classifier intent accuracy).

## Section 3: Top 5 Failure Modes

Analysis of the `golden_test.json` evaluation and LLM-as-Judge `grading_sheet.csv` reveals several persistent failure modes:

### 1. Inappropriate PII Scolding
- **Real example:** `"@115851 @115821 your delivery guy in Lincoln park NJ took my friends puppy. Need help now!!! Police next call... Thank you!"`
- **What happened:** The system drafted: `"Please don't provide your details as we consider them to be personal information... We want to look into..."`
- **Hypothesis:** The prompt strictly enforces PII protection. When the customer mentioned a location, the drafter likely prioritized the PII rule over the severe context of the message.
- **Impact:** Scolding a customer who just reported a stolen pet creates a disastrous, robotic, and insensitive customer experience.

### 2. Robotic Brush-Offs for Severe Product Issues
- **Real example:** `"I have ordered Maharaja Whiteline Juicer... and this is what comes out of box..... [picture of stone]"`
- **What happened:** The system replied: `"We're so sorry to see this! We'd like to look into this for you. Please share your details here: [link]"`
- **Hypothesis:** Retrieval appears to favor generic damaged-product examples rather than examples reflecting the unusual severity of the complaint.
- **Impact:** The response appears dismissive of a highly unusual and frustrating situation, failing to escalate the tone appropriately.

### 3. Vague Redirects for Unreachable Support
- **Real example:** `"WTF! IT'S NOT POSSIBLE TO CONNECT TO YOUR CS TEAM, EITHER ON 180030009009 OR THROUGH THE APP"`
- **What happened:** The system drafted: `"I'm sorry you're having trouble connecting with our customer service team. Please share your details here: [link]"`
- **Hypothesis:** The drafter lacks the contextual awareness to realize that redirecting a customer to another digital link when they are explicitly complaining about digital/phone unreachability exacerbates the frustration.
- **Impact:** Increases customer churn and frustration by trapping them in a loop of unhelpful automated responses.

### 4. Multilingual Misinterpretation and Scolding
- **Real example:** `"o? est mon colis svp ? 171-9260850-5134725"`
- **What happened:** The system replied in French: `"Bonjour, pour votre s?curit?, merci de ne pas partager vos informations personnelles..."`
- **Hypothesis:** While the model correctly identified the language and the PII (Order ID), it led with a scolding warning about privacy rather than addressing the core "where is my package" question.
- **Impact:** Customers feel reprimanded rather than assisted, degrading the perceived helpfulness of the support channel.

### 5. Contradictory Logic on Frozen Accounts
- **Real example:** `"hello. I sent a message via this link over 24 hours ago and no response yet. Can you help please? Also I would like to cancel one of the 2 orders I placed before my account was frozen"`
- **What happened:** The system replied: `"Since your account is currently frozen, we're unable to access your order..."`
- **Hypothesis:** The model logically deduced it cannot help with a frozen account, but the drafted reply is a dead-end that offers no escalation path or solution for the customer's stuck funds/orders.
- **Impact:** Leaves the customer completely stranded with no recourse, which is unacceptable for enterprise support.

## Section 4: The Misleading Number

When evaluating machine learning systems, aggregate accuracy often masks critical vulnerabilities in edge cases.

In our evaluation:
- We have **250 total golden examples**.
- Out of these, **63 are edge cases** requiring escalation (`gold_escalate == True`).
- **Sentinel-AI achieves 73.02% accuracy on the 63 edge cases.**

While an overall system accuracy across all 250 cases might appear high (often >85% because routine "Where is my order?" queries are easy to classify), the model struggles precisely where it matters most. 

**What 73.02% means operationally:**
73.02% means that Sentinel-AI correctly classified approximately 73% of the edge-case examples according to the golden intent labels, while approximately 27% were incorrectly classified. It is crucial to distinguish aggregate accuracy (performance across all 250 cases), edge-case accuracy (performance specifically on the 63 high-risk cases), intent Macro-F1 (balanced intent classification), and escalation FNR (missed escalations).

This result would not be sufficient evidence to justify fully autonomous handling of these edge cases. Additional safeguards, human review, and stronger recall on high-risk cases would be required before deployment at that level. Slicing the edge cases is important because aggregate metrics can easily hide failures on rare but high-risk examples.

## Section 5: What You'd Do With One More Week

Given the observed limitations in the current Sentinel-AI implementation, here are 4 technically realistic improvements to prioritize:

**1. PII Sanitization Middleware Before LLM Generation**
- **What would change:** Implement a lightweight NLP/Regex sanitization layer that masks PII (e.g., replacing tracking numbers with `[TRACKING_ID]`) *before* sending the prompt to the Drafter LLM.
- **Why it addresses weakness:** This removes the burden of PII-scolding from the LLM, preventing the robotic, insensitive replies seen in Failure Modes 1 and 4.
- **Benefit/Tradeoff:** Drastically improves empathy and brand tone. Tradeoff: Requires maintaining robust, multi-region regex/NER pipelines.

**2. Improve Escalation Recall via Multi-Thresholding**
- **What would change:** Implement distinct escalation thresholds per intent category rather than a global 0.4 threshold. For example, `Severe Escalation` intents trigger at 0.2, while `Delivery Issues` trigger at 0.6.
- **Why it addresses weakness:** It reduces the 27% edge-case misclassification rate by acknowledging that certain categories inherently carry higher baseline risk.
- **Benefit/Tradeoff:** Lowers FNR for critical issues. Tradeoff: Slightly increases complexity in threshold calibration and might increase overall False Positives (manual review volume).

**3. Context-Aware RAG with Sentiment/Severity Filtering**
- **What would change:** Augment the FAISS retrieval index to include sentiment and severity metadata. When querying, filter or weight historical examples so that an "angry/severe" incoming query only retrieves "angry/severe" historical resolutions.
- **Why it addresses weakness:** Solves Failure Mode 2 by ensuring the drafter doesn't use a generic, happy template for a severe or bizarre product failure.
- **Benefit/Tradeoff:** Produces more contextually appropriate drafts. Tradeoff: Requires re-indexing the historical dataset and a slightly slower two-pass retrieval step.

**4. Enhanced LLM-as-Judge Calibration (Few-Shot Prompting)**
- **What would change:** Update `eval/judge.py` to include 3-5 explicitly graded few-shot examples (with Chain-of-Thought reasoning) demonstrating the difference between a "3" and a "4" on the grading scale.
- **Why it addresses weakness:** Our current Cohen's Kappa is 0.558 (Moderate). Few-shot prompting grounds the LLM judge in specific human grading criteria, reducing scoring variance.
- **Benefit/Tradeoff:** Yields a more reliable automated evaluation pipeline that correlates tighter with human preference. Tradeoff: Increases prompt token size and evaluation cost.
