# run_eval.py
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run Sentinel-AI Evaluation Pipeline")
    parser.add_argument(
        "--subset", 
        type=str, 
        default="eval/golden_test.json", 
        help="Path to evaluation dataset"
    )
    args = parser.parse_args()

    # Fail-fast validation
    if not os.path.exists(args.subset):
        print(f"[ERROR] Target dataset not found: {args.subset}")
        sys.exit(1)

    print("=" * 60)
    print(f" SENTINEL-AI EVALUATION HARNESS")
    print(f" Target Dataset: {args.subset}")
    print("=" * 60)

    try:
        # Execute the core evaluation suite built in Block 7
        import time
        t0 = time.time()
        
        env = os.environ.copy()
        env["EVAL_DATASET_PATH"] = args.subset
        env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))
        env["SENTINEL_OFFLINE_EVAL"] = "1"  # Prevent expensive API calls
        
        result = subprocess.run(
            [sys.executable, "eval/evaluate.py"], 
            check=True,
            env=env
        )
        print("\n" + "=" * 60)
        print(" [SUCCESS] Evaluation finished cleanly.")
        print(" Headline metrics saved to: eval/results.csv")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Evaluation execution failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()