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
    
    # Run gatekeeper to ensure Pass 2 API is called and cached
    gatekeeper(text, intent_res.intent.value, intent_res.confidence, intent_res.reasoning)
    
    # Read score directly from the new Pass 2 cache
    import hashlib, json, os
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    cache_path = "cache/two_pass_continuous_calibration.json"
    score = 1.0 # fail-safe
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
            if text_hash in cache:
                score = cache[text_hash].get("escalation_score", 1.0)
    
    return {
        "intent": intent_res.intent.value,
        "confidence": intent_res.confidence,
        "escalation_score": score,
        "escalation_reason": "Pass 2 Calibration"
    }

def evaluate_system(dataset_path: str = None):
    # Priority: explicit arg > EVAL_DATASET_PATH env var > CLI arg > default
    if dataset_path is None:
        dataset_path = os.environ.get("EVAL_DATASET_PATH")
    if dataset_path is None:
        import sys as _sys
        if len(_sys.argv) > 1:
            dataset_path = _sys.argv[1]
        else:
            dataset_path = "eval/golden_test.json"

    print(f"Loading Golden Set from {dataset_path}...")
    try:
        with open(dataset_path, "r") as f:
            golden_set = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {dataset_path}.")
        import sys as _sys2; _sys2.exit(1)

    # Ensure all predictions are cached first
    print("Collecting predictions (this will use cache if available)...")
    golden_predictions = []
    
    start_time = time.time()
    for row in golden_set:
        text = row["text_customer"]
        pred = run_sentinel_ai(text)
        golden_predictions.append({
            "text": text,
            "gold_intent": row["gold_intent"],
            "gold_escalate": row["gold_escalate"],
            "pred_intent": pred["intent"],
            "confidence": pred["confidence"],
            "escalation_score": pred["escalation_score"],
            "escalation_reason": pred["escalation_reason"]
        })
    end_time = time.time()
    avg_latency = (end_time - start_time) / len(golden_set)
    
    print("\n=== THRESHOLD SWEEP ===")
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    sweep_results = []
    
    for th in thresholds:
        os.environ["ESCALATION_THRESHOLD"] = str(th)
        
        y_true_intent, y_pred_intent = [], []
  Optimize the current Sentinel-AI Block 9 implementation for efficiency, reliability, and reviewer experience.

IMPORTANT:
Do not change the core evaluation methodology, benchmark definitions, scoring logic, or expected evaluation behavior. The goal is to make the existing implementation faster, cleaner, less wasteful, and more robust while preserving the same results.

First inspect:
- run.sh
- run_eval.py
- eval/evaluate.py
- all cache-loading/writing logic
- FAISS initialization/search logic
- classifier cache
- zero-shot cache
- calibration cache
- requirements.txt
- .gitignore
- eval/golden_test.json

==================================================
1. ELIMINATE UNNECESSARY API CALLS
==================================================

The biggest priority is preventing expensive Groq/Gemini calls during evaluation.

Trace the complete execution path starting from:

    bash run.sh

Determine exactly when an external API call happens.

The final reviewer evaluation should use existing committed caches whenever possible.

Implement cache-first behavior:

    request/input
         ↓
    cache lookup
      ↙       ↘
   HIT        MISS
    ↓           ↓
 return      API call
 result          ↓
              cache result

For the committed golden test set, all required results should already exist in cache.

If a cache miss occurs during evaluation:
- clearly report it
- do not silently make hundreds of expensive API calls
- ideally provide a clear error explaining which cache is missing

Do NOT hide API calls behind the evaluation script.

==================================================
2. OPTIMIZE CACHE ACCESS
==================================================

Inspect the current `_get_cache`, cache loading, and cache writing implementation.

Avoid repeatedly doing:

    open(json)
    json.load(...)
    close()

inside every test-case iteration.

Instead, where practical:

    load cache once
        ↓
    keep dictionary in memory
        ↓
    perform all lookups
        ↓
    write only if necessary

For example:

    cache = load_cache_once()

    for row in dataset:
        result = cache.get(cache_key)

Do not repeatedly parse the same large JSON file 250 times.

Preserve deterministic behavior.

==================================================
3. PREVENT UNNECESSARY CACHE WRITES
==================================================

Do not rewrite cache JSON files when nothing has changed.

Only write a cache when:
- a new result was generated
- an existing result was updated
- the cache actually changed

This reduces disk I/O and avoids unnecessary Git diffs.

==================================================
4. FAISS EFFICIENCY
==================================================

Inspect how the FAISS index is loaded.

The index should be loaded once rather than repeatedly initialized for every test case.

Prefer:

    load FAISS index once
          ↓
    perform all searches
          ↓
    evaluate results

If the current implementation performs individual embedding/model initialization repeatedly, optimize this.

Do not change retrieval semantics or similarity thresholds unless absolutely necessary.

The optimization must preserve existing benchmark results.

==================================================
5. BATCH OPERATIONS WHERE SAFE
==================================================

Look for loops that perform expensive operations one item at a time.

Where the underlying library supports batching, consider batching:

- embeddings
- FAISS searches
- classifier inference
- zero-shot inference

However:

DO NOT batch external API calls if doing so would change:
- cache keys
- deterministic behavior
- rate-limit handling
- existing outputs

Only introduce batching where it is safe and behavior-preserving.

==================================================
6. AVOID DUPLICATE WORK
==================================================

Inspect the pipeline for repeated processing of the same:

- customer text
- embedding
- classification
- zero-shot prediction
- retrieval result

If the same input appears multiple times, reuse the cached/computed result.

Use appropriate in-memory dictionaries where useful.

Example:

    text -> embedding

or:

    cache_key -> prediction

Do not introduce complicated caching architecture for trivial operations.

Keep the implementation understandable.

==================================================
7. MEMORY EFFICIENCY
==================================================

Review how datasets are loaded.

The golden test set is relatively small, so do NOT over-engineer streaming unnecessarily.

However:
- avoid loading the same dataset multiple times
- avoid unnecessary DataFrame copies
- avoid retaining large intermediate objects after they are no longer needed

For FAISS and model artifacts, avoid duplicate copies in memory.

==================================================
8. SUBPROCESS EFFICIENCY
==================================================

Review `run_eval.py`.

It should remain a thin orchestration layer.

Do not unnecessarily spawn multiple Python processes.

Continue using:

    sys.executable

to ensure the current environment is used.

Make sure environment variables are passed correctly.

Do not move the entire evaluation implementation into `run_eval.py`.

==================================================
9. run.sh OPTIMIZATION
==================================================

Review:

    run.sh

The current script installs dependencies every time.

Determine whether this is appropriate for the reviewer workflow.

Do NOT remove dependency installation if Block 9 requires a one-command clean-clone experience.

However, make it safe and efficient.

For example:
- use `python -m pip` rather than relying on a potentially unrelated `pip`
- fail immediately on errors
- provide clear progress messages
- avoid installing unnecessary packages
- don't perform expensive cache generation

The expected command remains:

    bash run.sh

==================================================
10. PREPOPULATION SCRIPT
==================================================

Inspect `prepopulate_caches.py`.

This script is a preparation/build-time operation, NOT part of the normal evaluation path.

Ensure:

    bash run.sh

does NOT execute it.

The architecture should be:

    DEVELOPMENT / ARTIFACT BUILD
              ↓
      prepopulate caches
              ↓
       commit artifacts
              ↓
    ─────────────────────
       REVIEWER / CI
              ↓
          run.sh
              ↓
       cached evaluation
              ↓
          results

If the prepopulation script is currently doing unnecessary duplicate API calls, optimize it too.

It should:
- check whether a cache entry already exists
- skip already-cached inputs
- retry transient API failures
- use exponential backoff
- avoid regenerating existing artifacts
- save progress safely so an interrupted run can resume

Do not delete valid existing cached results.

==================================================
11. RATE LIMITING
==================================================

For the one-time prepopulation process, make API usage efficient.

Avoid unnecessary requests.

Use:
- cache checking before requests
- reasonable batching where supported
- exponential backoff for rate limits
- resumability

Do not implement aggressive concurrency that could make rate limiting worse.

Reliability is more important than theoretical maximum throughput.

==================================================
12. DETERMINISM
==================================================

Optimization must NOT change reproducibility.

The following should remain deterministic:

    same code
    + same dataset
    + same committed caches
    = same evaluation results

Do not introduce uncontrolled randomness.

Preserve:
- random seeds
- cache keys
- model configuration
- temperature settings
- evaluation thresholds
- benchmark definitions

unless a change is strictly necessary for optimization.

==================================================
13. PRESERVE RESULTS
==================================================

Before optimization, record the current evaluation results.

Then run the optimized pipeline.

Compare:

    BEFORE vs AFTER

The following should remain equivalent unless there is an explicitly documented reason:

- predictions
- confusion matrices
- Macro-F1
- benchmark scores
- number of evaluated cases
- results.csv contents

Performance improvements are only valid if evaluation behavior remains correct.

==================================================
14. ADD PERFORMANCE MEASUREMENT
==================================================

Add lightweight timing information where useful.

For example:

    Dataset loading: 0.12s
    Cache loading: 0.08s
    FAISS initialization: 0.21s
    Evaluation: 2.34s
    Results writing: 0.03s
    Total: 2.78s

Do not add excessive logging.

The goal is to make performance bottlenecks visible.

==================================================
15. FAILURE HANDLING
==================================================

The evaluation should fail fast and clearly.

Examples:

Missing dataset:

    [ERROR] Target dataset not found: ...

Missing required cache:

    [ERROR] Required cache artifact missing: ...

Unexpected API call:

    [ERROR] Evaluation requires uncached external inference...

Do not silently fall back to expensive API calls during the deterministic reviewer evaluation.

==================================================
16. GIT ARTIFACT SIZE
==================================================

Inspect the size of:

    cache/faiss_index.bin
    cache/*.json

Do not unnecessarily duplicate large artifacts.

Do not commit:
- temporary cache files
- debug output
- duplicate datasets
- old evaluation results
- API credentials
- virtual environments

Keep only artifacts genuinely required for reproducible evaluation.

==================================================
17. CODE QUALITY
==================================================

Keep the implementation simple.

Prefer:

    clear functions
    type hints where useful
    meaningful variable names
    small reusable helpers
    comments explaining architectural decisions

Avoid:

    unnecessary abstractions
    premature optimization
    complex frameworks
    duplicate implementations
    magic constants

This is an engineering assessment, so readability matters as much as speed.

==================================================
18. ACCEPTANCE TEST
==================================================

After optimization, perform the following tests.

TEST 1 — Normal execution

    bash run.sh

Expected:
- dependencies install successfully
- evaluation starts
- no unnecessary external API calls
- caches are used
- evaluation completes
- eval/results.csv is created
- exit code = 0

TEST 2 — Repeat execution

    bash run.sh
    bash run.sh

Expected:
- second execution should not regenerate anything
- no unnecessary API calls
- results should remain identical

TEST 3 — Direct execution

    python run_eval.py --subset eval/golden_test.json

Expected:
- same evaluation behavior
- same metrics
- exit code = 0

TEST 4 — Missing dataset

    python run_eval.py --subset does_not_exist.json

Expected:
- clear error
- non-zero exit code
- no evaluation starts

TEST 5 — Cache verification

Temporarily verify that the evaluation can find and load all required committed caches.

Expected:
- no cache regeneration
- no external API dependency

TEST 6 — Result comparison

Compare optimized results against the pre-optimization baseline.

Expected:
- same number of test cases
- same predictions
- same headline metrics
- no unexplained regression

==================================================
FINAL OUTPUT
==================================================

After implementation, report:

1. What was inefficient before.
2. What was optimized.
3. Which API calls were eliminated from the reviewer path.
4. Which caches are loaded once instead of repeatedly.
5. Whether FAISS/model initialization was optimized.
6. Whether prepopulation became resumable/cache-aware.
7. Before vs after execution time.
8. Whether results changed.
9. Whether repeated runs are deterministic.
10. Whether `bash run.sh` works without API keys.
11. Any remaining bottlenecks.

IMPORTANT FINAL PRINCIPLE:

Optimize the system, not the benchmark.

Do NOT manipulate:
- test cases
- labels
- scoring thresholds
- benchmark definitions
- evaluation metrics

to make the numbers look better.

The objective is:

    SAME RESULTS
    + LESS COMPUTE
    + FEWER API CALLS
    + LESS I/O
    + FASTER EXECUTION
    + MORE RELIABLE REPRODUCTION      y_true_esc, y_pred_esc = [], []
        
        for p in golden_predictions:
            y_true_intent.append(p["gold_intent"])
            y_pred_intent.append(p["pred_intent"])
            
            y_true_esc.append(p["gold_escalate"])
            
            # Apply threshold manually for the sweep (bypassing gatekeeper so we don't re-run deterministic rules here unless needed. Wait, deterministic rules should still apply!)
            from src.escalation import rule_based_escalation
            rule_fails, _ = rule_based_escalation(p["text"], p["confidence"])
            if rule_fails:
                y_pred_esc.append(True)
            else:
                y_pred_esc.append(p["escalation_score"] >= th)
            
        precision = precision_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)
        f1 = f1_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true_esc, y_pred_esc, labels=[False, True]).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        prec_esc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        sweep_results.append({
            "Threshold": th,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "FNR": round(fnr, 3),
            "Precision": round(prec_esc, 3),
            "Macro-F1": round(f1, 3),
            "Latency": round(avg_latency, 3)
        })
        
    df_sweep = pd.DataFrame(sweep_results)
    print(df_sweep.to_string(index=False))
    os.makedirs("eval", exist_ok=True)
    df_sweep.to_csv("eval/results.csv", index=False)
    df_sweep.to_csv("eval/sweep_results.csv", index=False)
    print("\nHeadline metrics saved to: eval/results.csv")

if __name__ == "__main__":
    evaluate_system()