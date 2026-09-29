# eval/rejudge.py
import os
import time

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

# # Gemini implementation (commented out)

load_dotenv()

# We import the JudgeScore schema and prompt from judge.py (which now uses Groq)
import sys

sys.path.append(os.getcwd())
from eval.judge import call_with_retry, run_llm_judge


def main():
    # Use absolute path so the script works regardless of working directory
    _HERE = os.path.abspath(os.path.dirname(__file__))
    csv_path = os.path.join(_HERE, "grading_sheet.csv")
    df = pd.read_csv(csv_path)

    print("Re-running LLM Judge on existing drafted replies via Groq...")
    new_llm_scores = []
    new_llm_reasoning = []

    for idx, row in df.iterrows():
        print(f"Judging {idx+1}/{len(df)}...")

        cust_text = row["customer_text"]
        reply = row["drafted_reply"]

        # Ask Groq judge to evaluate the reply
        judge_res = call_with_retry(run_llm_judge, cust_text, reply)
        new_llm_scores.append(judge_res.overall_score)
        new_llm_reasoning.append(judge_res.reasoning)

        time.sleep(3)  # 3s between calls to stay under qwen TPM limits

    # Final save — Create a fresh DataFrame to ensure the write succeeds cleanly
    df_out = df.copy()
    df_out["llm_overall_score"] = new_llm_scores
    df_out["llm_reasoning"] = new_llm_reasoning
    df_out.to_csv(csv_path, index=False)
    
    # Recalculate Kappa using the guaranteed-new scores
    kappa = cohen_kappa_score(
        new_llm_scores,
        df["human_overall_score"].tolist(),
        weights="linear"
    )

    exact_matches = sum(1 for a, b in zip(new_llm_scores, df["human_overall_score"].tolist()) if a == b)
    near_matches = sum(1 for a, b in zip(new_llm_scores, df["human_overall_score"].tolist()) if abs(a - b) <= 1)
    total = len(df)

    print("\n=== NEW LLM-AS-JUDGE AGREEMENT (GROQ) ===")
    print(f"Weighted Kappa Score (linear): {round(kappa, 3)}")
    print(f"Exact Match Rate:              {exact_matches}/{total} ({round(100*exact_matches/total, 1)}%)")
    print(f"Within-1-Point Rate:           {near_matches}/{total} ({round(100*near_matches/total, 1)}%)")
    print(
        "\nKappa scale: < 0 useless | 0.01-0.20 slight | 0.21-0.40 fair | "
        "0.41-0.60 moderate | 0.61-0.80 substantial | > 0.80 almost perfect"
    )

if __name__ == "__main__":
    main()
