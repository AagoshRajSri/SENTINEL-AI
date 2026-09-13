import os
import time
import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv

import google.genai as genai
from google.genai import types

load_dotenv()

# We will import the JudgeScore schema and prompt from judge.py
import sys
sys.path.append(os.getcwd())
from eval.judge import run_llm_judge, call_with_retry

def main():
    csv_path = "eval/grading_sheet.csv"
    df = pd.read_csv(csv_path)

    print("Re-running LLM Judge on existing drafted replies...")
    new_llm_scores = []
    
    for idx, row in df.iterrows():
        print(f"Judging {idx+1}/{len(df)}...")
        
        cust_text = row["customer_text"]
        reply = row["drafted_reply"]
        
        # Ask Gemini to judge the reply using the newly updated prompt
        judge_res = call_with_retry(run_llm_judge, cust_text, reply)
        new_llm_scores.append(judge_res.overall_score)
        
        # Sleep to avoid rate limits
        time.sleep(1)

    # Update the dataframe
    df["llm_overall_score"] = new_llm_scores
    df.to_csv(csv_path, index=False)
    
    # Recalculate Kappa
    kappa = cohen_kappa_score(
        df["llm_overall_score"],
        df["human_overall_score"],
        weights="linear"
    )

    exact_matches = (df["llm_overall_score"] == df["human_overall_score"]).sum()
    near_matches = (abs(df["llm_overall_score"] - df["human_overall_score"]) <= 1).sum()
    total = len(df)

    print("\n=== NEW LLM-AS-JUDGE AGREEMENT ===")
    print(f"Weighted Kappa Score (linear): {round(kappa, 3)}")
    print(f"Exact Match Rate:              {exact_matches}/{total} ({round(100*exact_matches/total, 1)}%)")
    print(f"Within-1-Point Rate:           {near_matches}/{total} ({round(100*near_matches/total, 1)}%)")
    print(
        "\nKappa scale: < 0 useless | 0.01-0.20 slight | 0.21-0.40 fair | "
        "0.41-0.60 moderate | 0.61-0.80 substantial | > 0.80 almost perfect"
    )

if __name__ == "__main__":
    main()
