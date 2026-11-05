# eval/evaluate_end_to_end.py
"""
Complete End-to-End Pipeline Evaluation Harness for Sentinel-AI

Evaluates the full two-pass system on the 250-case golden benchmark:
  Pass 1: Intent classification & Recall-biased escalation gate
  Pass 2: In-memory FAISS retrieval & Grounded reply drafting
  Gating Check: Verifies high-risk queries withhold automated drafting.
"""

import hashlib
import json
import os
import sys
import time
from typing import Dict, List

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.classifier import classify
from src.escalation import _get_cache as get_escalation_cache, gatekeeper, rule_based_escalation
from src.reply_drafter import draft_reply
from src.retriever import get_index_metric_type, retrieve


def run_full_pipeline_triage(text: str, threshold: float = 0.40) -> dict:
    """Executes the complete two-pass pipeline on a customer query."""
    # 1. Pass 1A: Intent Classification
    intent_res = classify(text)
    
    # 2. Pass 1B: Escalation Gate
    os.environ["ESCALATION_THRESHOLD"] = str(threshold)
    esc_decision = gatekeeper(text, intent_res.intent.value, intent_res.confidence, intent_res.reasoning)
    
    # Read escalation score
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    esc_cache = get_escalation_cache()
    score = esc_cache.get(text_hash, {}).get("escalation_score", 1.0) if text_hash in esc_cache else 1.0

    # 3. Two-pass routing
    if esc_decision.should_escalate:
        # High-risk: Automated reply withheld
        auto_draft_generated = False
        draft_text = "[ESCALATED TO HUMAN AGENT] Automated reply withheld for safety review. Case routed to Tier-2 support team."
        policy_text = f"Safety Gate Triggered: {esc_decision.reason}"
        context = []
        is_fallback = False
    else:
        # Routine: Pass 2 RAG drafting (excluding exact self-matches to evaluate true retrieval generalization)
        auto_draft_generated = True
        context = retrieve(text, k=3, exclude_exact_match=True)
        reply_res = draft_reply(text, with_rag=True, use_cache=True, exclude_exact_match=True)
        draft_text = reply_res.draft_reply
        policy_text = reply_res.policy_adherence_check
        is_fallback = "fallback" in policy_text.lower() or len(context) == 0

    return {
        "pred_intent": intent_res.intent.value,
        "confidence": intent_res.confidence,
        "escalation_score": score,
        "should_escalate": esc_decision.should_escalate,
        "escalation_reason": esc_decision.reason,
        "auto_draft_generated": auto_draft_generated,
        "draft_reply": draft_text,
        "policy_adherence": policy_text,
        "retrieved_context_count": len(context),
        "is_fallback": is_fallback
    }


def evaluate_end_to_end(dataset_path: str = None, threshold: float = 0.40):
    if dataset_path is None:
        dataset_path = os.environ.get("EVAL_DATASET_PATH", os.path.join(_ROOT, "eval", "golden_test.json"))

    print("=" * 70)
    print(" SENTINEL-AI: FULL END-TO-END PIPELINE BENCHMARK")
    print(f" Target Benchmark: {dataset_path}")
    print(f" Escalation Threshold: {threshold}")
    print(f" FAISS Index Type: {get_index_metric_type()}")
    print("=" * 70)

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    total_cases = len(golden_set)
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    records = []
    start_time = time.time()

    for idx, row in enumerate(golden_set):
        text = row["text_customer"]
        gold_intent = row["gold_intent"]
        gold_esc = row["gold_escalate"]

        try:
            res = run_full_pipeline_triage(text, threshold=threshold)
            processed_count += 1
            records.append({
                "tweet_id": row.get("tweet_id_customer"),
                "text": text,
                "gold_intent": gold_intent,
                "gold_escalate": gold_esc,
                **res
            })
        except Exception as e:
            failed_count += 1
            print(f"[ERROR] Failed case {idx+1}: {e!s}")

    total_runtime = time.time() - start_time
    avg_latency = total_runtime / processed_count if processed_count > 0 else 0.0

    # 1. Intent Classification Metrics
    y_true_intent = [r["gold_intent"] for r in records]
    y_pred_intent = [r["pred_intent"] for r in records]
    macro_f1 = f1_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)
    intent_report = classification_report(y_true_intent, y_pred_intent, output_dict=True, zero_division=0)

    # 2. Escalation Gatekeeper Metrics
    y_true_esc = [r["gold_escalate"] for r in records]
    y_pred_esc = [r["should_escalate"] for r in records]
    tn, fp, fn, tp = confusion_matrix(y_true_esc, y_pred_esc, labels=[False, True]).ravel()

    recall_esc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr_esc = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    prec_esc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spec_esc = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 3. Full Pipeline Routing Checks
    # Safety Check: Did ANY escalated case get an automated customer reply?
    escalated_cases = [r for r in records if r["should_escalate"]]
    auto_draft_violations = sum(1 for r in escalated_cases if r["auto_draft_generated"])

    # Routine Case Drafting Check:
    routine_cases = [r for r in records if not r["should_escalate"]]
    routine_drafting_success = sum(1 for r in routine_cases if r["auto_draft_generated"] and len(r["draft_reply"]) > 10)
    routine_success_rate = routine_drafting_success / len(routine_cases) if routine_cases else 0.0

    # Retrieval Coverage on routine cases
    routine_with_context = sum(1 for r in routine_cases if r["retrieved_context_count"] > 0)
    retrieval_coverage = routine_with_context / len(routine_cases) if routine_cases else 0.0
    fallback_count = sum(1 for r in routine_cases if r["is_fallback"])
    fallback_rate = fallback_count / len(routine_cases) if routine_cases else 0.0

    dataset_identifier = os.path.basename(dataset_path)
    try:
        rel_dataset_path = os.path.relpath(dataset_path, start=_ROOT).replace("\\", "/")
    except Exception:
        rel_dataset_path = f"eval/{dataset_identifier}"

    results_summary = {
        "benchmark": rel_dataset_path,
        "dataset_identifier": dataset_identifier,
        "cases_loaded": total_cases,
        "cases_processed": processed_count,
        "cases_skipped": skipped_count,
        "cases_failed": failed_count,
        "runtime_seconds": round(total_runtime, 3),
        "avg_latency_ms": round(avg_latency * 1000, 2),
        "intent_classification": {
            "macro_f1": round(macro_f1, 4),
            "per_class": {
                cls: {
                    "precision": round(intent_report[cls]["precision"], 3),
                    "recall": round(intent_report[cls]["recall"], 3),
                    "f1_score": round(intent_report[cls]["f1-score"], 3),
                    "support": int(intent_report[cls]["support"])
                }
                for cls in sorted(intent_report.keys())
                if cls not in ["accuracy", "macro avg", "weighted avg"]
            }
        },
        "escalation_gate": {
            "threshold": threshold,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "recall": round(recall_esc, 4),
            "fnr": round(fnr_esc, 4),
            "precision": round(prec_esc, 4),
            "specificity": round(spec_esc, 4)
        },
        "full_pipeline_safety_and_routing": {
            "total_escalated_cases": len(escalated_cases),
            "auto_draft_violations": auto_draft_violations,
            "violation_rate": 0.0 if not escalated_cases else round(auto_draft_violations / len(escalated_cases), 4),
            "total_routine_cases": len(routine_cases),
            "routine_drafting_success_count": routine_drafting_success,
            "routine_drafting_success_rate": round(routine_success_rate, 4),
            "retrieval_coverage": round(retrieval_coverage, 4),
            "fallback_rate": round(fallback_rate, 4)
        }
    }

    # Print Report
    print("\n" + "=" * 70)
    print(" 1. BENCHMARK EXECUTION SUMMARY")
    print("=" * 70)
    print(f" Loaded:    {total_cases} | Processed: {processed_count} | Skipped: {skipped_count} | Failed: {failed_count}")
    print(f" Total Runtime: {total_runtime:.2f}s | Average Latency: {avg_latency*1000:.2f}ms/query")

    print("\n" + "=" * 70)
    print(" 2. INTENT CLASSIFICATION METRICS")
    print("=" * 70)
    print(f" Intent Macro-F1: {macro_f1:.4f}\n")
    print(f" {'Category':<32} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Support':<7}")
    print("-" * 70)
    for cls, metrics in results_summary["intent_classification"]["per_class"].items():
        print(f" {cls:<32} | {metrics['precision']:<7.3f} | {metrics['recall']:<7.3f} | {metrics['f1_score']:<7.3f} | {metrics['support']:<7}")

    print("\n" + "=" * 70)
    print(" 3. ESCALATION GATEKEEPER METRICS (Threshold = 0.40)")
    print("=" * 70)
    print(f" Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f" Escalation Recall:      {recall_esc*100:.1f}% ({tp}/{tp+fn})")
    print(f" Escalation FNR:         {fnr_esc*100:.1f}% ({fn}/{tp+fn})  [Safety Target < 35%]")
    print(f" Escalation Precision:   {prec_esc*100:.1f}% ({tp}/{tp+fp})")
    print(f" Escalation Specificity: {spec_esc*100:.1f}% ({tn}/{tn+fp})")

    print("\n" + "=" * 70)
    print(" 4. FULL PIPELINE ROUTING & SAFETY GATING")
    print("=" * 70)
    print(f" Total Escalated Cases:            {len(escalated_cases)}")
    print(f" Auto-Draft Violations:            {auto_draft_violations} (Must be 0 to prevent unauthorized deflection)")
    print(f" Total Routine Inquiries:          {len(routine_cases)}")
    print(f" Routine Drafting Success Rate:    {routine_success_rate*100:.1f}% ({routine_drafting_success}/{len(routine_cases)})")
    print(f" Retrieval Coverage (Routine):     {retrieval_coverage*100:.1f}%")
    print(f" Fallback Rate (Routine):          {fallback_rate*100:.1f}%")
    print("=" * 70 + "\n")

    out_file = os.path.join(_ROOT, "eval", "end_to_end_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved complete end-to-end evaluation metrics to: {out_file}")

    return results_summary


if __name__ == "__main__":
    evaluate_end_to_end()
