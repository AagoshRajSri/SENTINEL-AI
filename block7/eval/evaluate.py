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
    esc_res = gatekeeper(text, intent_res.confidence)
    return {
        "intent": intent_res.intent.value,
        "should_escalate": esc_res.should_escalate
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

    # The 3 competitors in our arena
    systems = {
        "Baseline-Heuristic": baseline_heuristic,
        "Baseline-ZeroShot": baseline_zeroshot,
        "Sentinel-AI": run_sentinel_ai
    }
    
    metrics_summary = []
    
    for sys_name, sys_func in systems.items():
        print(f"Evaluating {sys_name}...")
        
        y_true_intent, y_pred_intent = [], []
        y_true_esc, y_pred_esc = [], []
        
        start_time = time.time()
        
        for row in golden_set:
            text = row["text_customer"]
            
            # 1. Store the Human 'Answer Key' (Ground Truth)
            y_true_intent.append(row["gold_intent"])
            y_true_esc.append(row["gold_escalate"])
            
            # 2. Get the AI's Prediction
            pred = sys_func(text)
            y_pred_intent.append(pred["intent"])
            y_pred_esc.append(pred["should_escalate"])
            
        end_time = time.time()
        avg_latency = (end_time - start_time) / len(golden_set)
        
        # 3. Calculate Math & Metrics
        # Intent Metrics (Macro-average accounts for category imbalance)
        precision = precision_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)
        recall = recall_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)
        f1 = f1_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)
        
        # Escalation Metrics (True = Escalate to Human, False = Auto-reply)
        tn, fp, fn, tp = confusion_matrix(y_true_esc, y_pred_esc, labels=[False, True]).ravel()
        
        # False Negative Rate (FNR) = Missed Escalations / Total Actual Escalations
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        metrics_summary.append({
            "System": sys_name,
            "Macro-F1": round(f1, 3),
            "FNR (Missed Escalations)": round(fnr, 3),
            "False Alarms (FP)": fp,
            "Latency (sec)": round(avg_latency, 3)
        })
        
    # 4. Save and Print Results
    df_results = pd.DataFrame(metrics_summary)
    df_results.to_csv("eval/results.csv", index=False)
    
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(df_results.to_string(index=False))
    print("\nDetailed results saved to eval/results.csv")

if __name__ == "__main__":
    evaluate_system()