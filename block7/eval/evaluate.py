# eval/evaluate.py
import os
import json
import time
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Import our three systems
from src.classifier import classify
from src.escalation import gatekeeper
from src.baseline_heuristic import baseline_heuristic
from src.baseline_zeroshot import baseline_zeroshot

def run_sentinel_ai(text: str) -> dict:
    """Wrapper to run the full Sentinel-AI pipeline (Intent + Escalation)."""
    intent_res = classify(text)
    
    # Run gatekeeper to ensure Pass 2 API is called and cached
    gatekeeper(text, intent_res.intent.value, intent_res.confidence, intent_res.reasoning)
    
    # Read score directly from the new Pass 2 cache
    import hashlib, json, os
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    cache_path = "cache/two_pass_continuous_calibration.json"
    score = 1.0 # fail-safe
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
            if text_hash in cache:
                score = cache[text_hash].get("escalation_score", 1.0)
    
    return {
        "intent": intent_res.intent.value,
        "confidence": intent_res.confidence,
        "escalation_score": score,
        "escalation_reason": "Pass 2 Calibration"
    }

def evaluate_system(dataset_path: str = None):
    if dataset_path is None:
        import sys
        if len(sys.argv) > 1:
            dataset_path = sys.argv[1]
        elif os.path.exists("eval/golden_test_30.json"):
            dataset_path = "eval/golden_test_30.json"
        else:
            dataset_path = "eval/golden_test_small.json"

    print(f"Loading Golden Set from {dataset_path}...")
    try:
        with open(dataset_path, "r") as f:
            golden_set = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {dataset_path}.")
        return

    # Ensure all predictions are cached first
    print("Collecting predictions (this will use cache if available)...")
    golden_predictions = []
    
    start_time = time.time()
    for row in golden_set:
        text = row["text_customer"]
        pred = run_sentinel_ai(text)
        golden_predictions.append({
            "text": text,
            "gold_intent": row["gold_intent"],
            "gold_escalate": row["gold_escalate"],
            "pred_intent": pred["intent"],
            "confidence": pred["confidence"],
            "escalation_score": pred["escalation_score"],
            "escalation_reason": pred["escalation_reason"]
        })
    end_time = time.time()
    avg_latency = (end_time - start_time) / len(golden_set)
    
    print("\n=== THRESHOLD SWEEP ===")
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    sweep_results = []
    
    for th in thresholds:
        os.environ["ESCALATION_THRESHOLD"] = str(th)
        
        y_true_intent, y_pred_intent = [], []
        y_true_esc, y_pred_esc = [], []
        
        for p in golden_predictions:
            y_true_intent.append(p["gold_intent"])
            y_pred_intent.append(p["pred_intent"])
            
            y_true_esc.append(p["gold_escalate"])
            
            # Apply threshold manually for the sweep (bypassing gatekeeper so we don't re-run deterministic rules here unless needed. Wait, deterministic rules should still apply!)
            from src.escalation import rule_based_escalation
            rule_fails, _ = rule_based_escalation(p["text"], p["confidence"])
            if rule_fails:
                y_pred_esc.append(True)
            else:
                y_pred_esc.append(p["escalation_score"] >= th)
            
        precision = precision_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)
        f1 = f1_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true_esc, y_pred_esc, labels=[False, True]).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        prec_esc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        sweep_results.append({
            "Threshold": th,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "FNR": round(fnr, 3),
            "Precision": round(prec_esc, 3),
            "Macro-F1": round(f1, 3),
            "Latency": round(avg_latency, 3)
        })
        
    df_sweep = pd.DataFrame(sweep_results)
    print(df_sweep.to_string(index=False))
    df_sweep.to_csv("eval/sweep_results.csv", index=False)
    print("\nDetailed sweep results saved to eval/sweep_results.csv")

if __name__ == "__main__":
    evaluate_system()