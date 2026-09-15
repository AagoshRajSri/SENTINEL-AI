# eval/evaluate.py
import hashlib
import json
import os
import sys
import time

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our three systems
from src.classifier import classify
from src.escalation import _get_cache, gatekeeper


def run_sentinel_ai(text: str) -> dict:
    intent_res = classify(text)
    
    # Run gatekeeper to ensure Pass 2 API is called and cached
    gatekeeper(text, intent_res.intent.value, intent_res.confidence, intent_res.reasoning)
    
    # Read score directly from in-memory Pass 2 cache
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    cache = _get_cache()
    score = 1.0 # fail-safe
    if text_hash in cache:
        score = cache[text_hash].get('escalation_score', 1.0)
    
    return {
        'intent': intent_res.intent.value,
        'confidence': intent_res.confidence,
        'escalation_score': score,
        'escalation_reason': 'Pass 2 Calibration'
    }

def evaluate_system(dataset_path: str = None):
    # Priority: explicit arg > EVAL_DATASET_PATH env var > CLI arg > default
    if dataset_path is None:
        dataset_path = os.environ.get('EVAL_DATASET_PATH')
    if dataset_path is None:
        if len(sys.argv) > 1:
            dataset_path = sys.argv[1]
        else:
            dataset_path = 'eval/golden_test.json'

    print(f'Loading Golden Set from {dataset_path}...')
    try:
        with open(dataset_path, 'r') as f:
            golden_set = json.load(f)
    except FileNotFoundError:
        print(f'[ERROR] Could not find {dataset_path}.')
        sys.exit(1)

    print('Collecting predictions (this will use cache if available)...')
    t0_load = time.time()
    
    golden_predictions = []
    
    start_time = time.time()
    for row in golden_set:
        text = row['text_customer']
        pred = run_sentinel_ai(text)
        golden_predictions.append({
            'text': text,
            'gold_intent': row['gold_intent'],
            'gold_escalate': row['gold_escalate'],
            'pred_intent': pred['intent'],
            'confidence': pred['confidence'],
            'escalation_score': pred['escalation_score'],
            'escalation_reason': pred['escalation_reason']
        })
    end_time = time.time()
    avg_latency = (end_time - start_time) / len(golden_set)
    
    print()
    print('=== THRESHOLD SWEEP ===')
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    sweep_results = []
    
    for th in thresholds:
        os.environ['ESCALATION_THRESHOLD'] = str(th)
        
        y_true_intent, y_pred_intent = [], []
        y_true_esc, y_pred_esc = [], []
        
        for p in golden_predictions:
            y_true_intent.append(p['gold_intent'])
            y_pred_intent.append(p['pred_intent'])
            
            y_true_esc.append(p['gold_escalate'])
            
            # Apply threshold manually for the sweep
            from src.escalation import rule_based_escalation
            rule_fails, _ = rule_based_escalation(p['text'], p['confidence'])
            if rule_fails:
                y_pred_esc.append(True)
            else:
                y_pred_esc.append(p['escalation_score'] >= th)
            
        f1 = f1_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true_esc, y_pred_esc, labels=[False, True]).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        prec_esc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        sweep_results.append({
            'Threshold': th,
            'TP': tp,
            'FN': fn,
            'FP': fp,
            'TN': tn,
            'FNR': round(fnr, 3),
            'Precision': round(prec_esc, 3),
            'Macro-F1': round(f1, 3),
            'Latency': round(avg_latency, 3)
        })
        
    df_sweep = pd.DataFrame(sweep_results)
    print(df_sweep.to_string(index=False))
    os.makedirs('eval', exist_ok=True)
    df_sweep.to_csv('eval/results.csv', index=False)
    print()
    print('Headline metrics saved to: eval/results.csv')

if __name__ == '__main__':
    evaluate_system()
