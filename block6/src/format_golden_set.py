# src/format_golden_set.py
import pandas as pd
import json

def convert_to_json(csv_path="eval/labeled_sample.csv", json_path="eval/golden_test.json"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Could not find {csv_path}. Did you save your manual work?")
        return

    # Ensure no rows have empty required fields
    df = df.dropna(subset=['gold_intent', 'gold_escalate'])
    
    # Convert 'TRUE' / 'FALSE' strings to actual Python booleans
    df['gold_escalate'] = df['gold_escalate'].astype(str).str.strip().str.upper() == 'TRUE'
    
    # Convert to a list of dictionaries
    records = df.to_dict(orient='records')
    
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=4)
        
    print(f"Successfully converted {len(records)} rows to {json_path}")

if __name__ == "__main__":
    convert_to_json()   