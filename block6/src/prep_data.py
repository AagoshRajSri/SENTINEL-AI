# src/prep_data.py
import pandas as pd

def prepare_dataset(input_path="data/twcs.csv", output_path="data/clean_pairs.csv", sample_size=15000):
    print("Loading raw Twitter dataset...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}. Please download it from Kaggle and place it in the /data folder.")
        return

    # 1. Isolate the brand's outgoing replies (We use AmazonHelp as recommended)
    brand_replies = df[df['author_id'] == 'AmazonHelp'].copy()
    
    # 2. Isolate all inbound customer tweets
    inbound_tweets = df[df['inbound'] == True].copy()
    
    # 3. Join customer tweets to the brand's replies
    # This matches the customer's question directly to Amazon's answer using the tweet IDs
    print("Joining customer threads...")
    merged = pd.merge(
        inbound_tweets, 
        brand_replies, 
        left_on='tweet_id', 
        right_on='in_response_to_tweet_id', 
        suffixes=('_customer', '_brand')
    )
    
    # 4. Clean the data (Drop blank texts, drop duplicate customer texts)
    merged = merged.dropna(subset=['text_customer', 'text_brand'])
    merged = merged.drop_duplicates(subset=['text_customer'])
    
    # 5. Keep only the necessary columns to keep the file lightweight
    final_df = merged[['tweet_id_customer', 'text_customer', 'text_brand']]
    
    # 6. Subsample to a manageable size with a fixed seed for determinism (Rule 1)
    if len(final_df) > sample_size:
        final_df = final_df.sample(n=sample_size, random_state=42)
        
    print(f"Extracted {len(final_df)} clean customer-brand interaction pairs.")
    
    # 7. Save to disk
    final_df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}")

if __name__ == "__main__":
    prepare_dataset()