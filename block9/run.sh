#!/bin/bash
# run.sh - Automated install and evaluation script
set -e # Exit immediately if any command encounters an error

echo "=== Step 1: Validating and Installing Dependencies ==="
pip install -r requirements.txt

echo -e "\n=== Step 2: Running Deterministic Evaluation ==="
python run_eval.py --subset eval/golden_test.json