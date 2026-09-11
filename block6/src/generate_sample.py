# src/generate_sample.py
import pandas as pd

def generate_blank_golden_set(input_path="data/clean_pairs.csv", output_path="eval/unlabeled_sample.csv", sample_size=250):
    print("Loading clean pairs...")
    df = pd.read_csv(input_path)
    
    # Pull a random sample with a fixed seed for determinism
    sample_df = df.sample(n=sample_size, random_state=42).copy()
    
    # Add the blank columns you will fill in by hand
    sample_df['gold_intent'] = ""
    sample_df['gold_escalate'] = ""         # Type TRUE or FALSE
    sample_df['gold_escalate_reason'] = ""  # e.g., "Customer threatened legal action"
    sample_df['gold_reference_reply'] = sample_df['text_brand'] 
    
    # Reorder for easy reading
    final_df = sample_df[['tweet_id_customer', 'text_customer', 'gold_intent', 'gold_escalate', 'gold_escalate_reason', 'gold_reference_reply']]
    
    final_df.to_csv(output_path, index=False)
    print(f"Blank answer key generated at {output_path}. Open this in Excel/Google Sheets to begin labeling.")

if __name__ == "__main__":
    generate_blank_golden_set()