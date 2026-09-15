# Sentinel-AI 🛡️

A retrieval-augmented triage and auto-reply system for enterprise customer support. 

Sentinel-AI acts as a first line of defense for inbound customer messages. It classifies customer intent, detects high-risk severe escalations (legal threats, safety issues, highly frustrated customers), and drafts empathetic, context-aware replies for routine issues. 

> **Note:** This repository contains the final iteration of the project in the `block12/` directory.

---

## 🎯 What it does

When a customer message arrives, the pipeline executes a two-pass architecture:

1. **Classification & Escalation (Pass 1)**
   - Categorizes the message into one of 7 internal taxonomies (e.g., `Delivery Issues`, `Severe Escalation`, `Payment & Refunds`).
   - Uses an LLM to determine if the issue requires immediate human intervention (`should_escalate`), utilizing a continuous calibration threshold (currently `0.4`).

2. **Reply Drafting (Pass 2)**
   - If the issue is routine, the system queries a local FAISS vector store containing thousands of historical, human-resolved support tickets.
   - It retrieves the top `k=3` most semantically similar past resolutions.
   - A drafting LLM (Gemini) uses these past resolutions as grounded context to write a highly accurate, brand-aligned reply.
   - **PII Protection:** The drafter is strictly instructed never to confirm or repeat Personally Identifiable Information publicly.

## 🏗️ Architecture

- **Embeddings:** `all-MiniLM-L6-v2` (via SentenceTransformers)
- **Vector Search:** FAISS (Facebook AI Similarity Search)
- **LLM Provider:** Groq (Llama-3 / OSS variants) and Google Gemini (for drafting/judging)
- **Data:** Cleaned and pre-processed Twitter/X customer support dataset.

## 🚀 Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AagoshRajSri/SENTINEL-AI.git
   cd SENTINEL-AI/block12
   ```

2. **Set up the environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   Copy the example environment file and add your keys.
   ```bash
   cp .env.example .env
   # Edit .env to add GROQ_API_KEY and GEMINI_API_KEY
   ```

4. **Run the Evaluation Pipeline:**
   The `run.sh` script installs dependencies and runs the deterministic evaluation on the 250-case golden test set.
   ```bash
   ./run.sh
   ```

## 📊 Evaluation & Metrics

The system is evaluated deterministically against a manually annotated golden test set (`eval/golden_test.json`). 

Instead of relying solely on standard accuracy (which is easily skewed by highly imbalanced routine queries), we evaluate using:
- **Macro-F1 (Intent):** 0.583
- **Escalation False Negative Rate (FNR):** 34.9% 
- **LLM-as-a-Judge (Cohen's Kappa):** 0.558 (Moderate to Good agreement with human graders on reply quality: Fidelity, Groundedness, and Safety).

For a deep dive into the evaluation methodologies, baseline comparisons, and edge-case failure modes (like PII-scolding), see the final [REPORT.md](./block12/REPORT.md).

## ⚠️ Limitations & Scope

Sentinel-AI is strictly a **triage and drafting assistant**. It does not execute irreversible actions like autonomous refunds, database writes, or account modifications. It is designed to empower human agents by filtering out noise and teeing up draft responses, not to replace them entirely.
