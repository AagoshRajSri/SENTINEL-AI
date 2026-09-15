# src/misleading_number.py
import json
from sklearn.metrics import accuracy_score
from src.classifier import classify

def calculate_edge_case_accuracy():
    print("Loading Golden Set...")
    with open("eval/golden_test.json", "r") as f:
        golden_set = json.load(f)

    # Filter for only the hard/angry/edge-case tweets
    edge_cases = [row for row in golden_set if row["gold_escalate"] == True]
    
    print(f"Found {len(edge_cases)} edge cases out of {len(golden_set)} total tweets.")
    
    y_true, y_pred = [], []
    
    for row in edge_cases:
        text = row["text_customer"]
        y_true.append(row["gold_intent"])
        
        # Predict using cached classifier (this will be instant)
        pred = classify(text)
        y_pred.append(pred.intent.value)
        
    acc = accuracy_score(y_true, y_pred)
    print(f"---")
    print(f"Headline accuracy might be high, but accuracy strictly on EDGE CASES is: {acc * 100:.2f}%")
    print("Use this computed number in Section 4 of your report!")

if __name__ == "__main__":
    calculate_edge_case_accuracy()