#!/usr/bin/env python3
"""
scripts/evaluate_rag_triad.py — RAG Triad Generative Quality Evaluation

Evaluates Faithfulness (groundedness against retrieved FAISS context) and
Answer Relevance (direct addressing of customer queries) for non-escalated cases
using a strict LLM-as-a-Judge protocol powered by Groq.

Outputs:
  - Console summary table and failure mode analysis
  - JSON benchmark receipt: cache/rag_triad_results.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from eval.rag_triad_schema import GenerativeEvalResult
from src.reply_drafter import draft_reply, apply_grounding_safeguards
from src.retriever import retrieve

GOLDEN_TEST_PATH = os.path.join(_ROOT, "eval", "golden_test.json")
JUDGE_CACHE_PATH = os.path.join(_ROOT, "cache", "rag_triad_judge_cache.json")
OUTPUT_PATH = os.path.join(_ROOT, "cache", "rag_triad_results.json")

JUDGE_SYSTEM_PROMPT = """You are an expert QA auditor. Evaluate the generated support draft based on the user query and the retrieved context.

FAITHFULNESS: The draft must not contain any facts, policies, or promises not explicitly present in the retrieved context. If it hallucinates, is_faithful=False.

RELEVANCE: The draft must address the user's core issue. If it dodges the question, is_relevant=False."""


def _get_judge_cache() -> dict:
    if os.path.exists(JUDGE_CACHE_PATH):
        try:
            with open(JUDGE_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_judge_cache(cache: dict):
    os.makedirs(os.path.dirname(JUDGE_CACHE_PATH), exist_ok=True)
    with open(JUDGE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def judge_case(
    client: Groq,
    user_query: str,
    retrieved_context: List[str],
    generated_draft: str,
    model: str = "openai/gpt-oss-120b",
    cache: Optional[dict] = None
) -> GenerativeEvalResult:
    """Invokes LLM-as-a-Judge with exponential backoff and persistent caching."""
    context_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(retrieved_context))
    cache_key = hashlib.sha256(
        f"{model}||{user_query.strip()}||{generated_draft.strip()}||{context_text.strip()}".encode("utf-8")
    ).hexdigest()

    if cache is not None and cache_key in cache:
        return GenerativeEvalResult(**cache[cache_key])

    user_prompt = f"""USER QUERY:
"{user_query}"

RETRIEVED CONTEXT (FAISS):
{context_text}

GENERATED SUPPORT DRAFT:
"{generated_draft}"

Evaluate FAITHFULNESS and RELEVANCE according to the system instructions.
Return a valid JSON object matching:
{{
  "is_faithful": true/false,
  "is_relevant": true/false,
  "reasoning": "brief justification"
}}"""

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            eval_result = GenerativeEvalResult.model_validate_json(content)

            if cache is not None:
                cache[cache_key] = eval_result.model_dump()
                _save_judge_cache(cache)

            return eval_result
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [ERROR] Judge call failed after {max_retries} attempts: {e}")
                # Conservative fallback: flag as unverified
                return GenerativeEvalResult(
                    is_faithful=False,
                    is_relevant=False,
                    reasoning=f"Judge API Error: {e!s}"
                )
            backoff = 2 * (2 ** attempt)
            time.sleep(backoff)


def run_rag_triad_evaluation(
    sample_size: int = 40,
    model: str = "openai/gpt-oss-120b",
    output_file: str = OUTPUT_PATH
) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL: GROQ_API_KEY environment variable is missing. Please check your .env file.")

    client = Groq(api_key=api_key)
    judge_cache = _get_judge_cache()

    if not os.path.exists(GOLDEN_TEST_PATH):
        raise FileNotFoundError(f"Golden test dataset missing: {GOLDEN_TEST_PATH}")

    with open(GOLDEN_TEST_PATH, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    # Filter out escalated cases — evaluate cases where drafts are actually delivered
    non_escalated_cases = [c for c in golden_cases if not c.get("gold_escalate", False)]
    print(f"Loaded {len(golden_cases)} total golden cases. Found {len(non_escalated_cases)} non-escalated cases.")

    selected_cases = non_escalated_cases[:sample_size]
    print(f"Selected sample size of {len(selected_cases)} cases for RAG Triad Evaluation.")
    print(f"Judge Model: {model} (Temperature: 0.0, Schema: GenerativeEvalResult)\n")

    results = []
    faithful_count = 0
    relevant_count = 0

    print("=" * 90)
    print(f"{'#':<3} | {'Tweet ID':<10} | {'Faithful':<9} | {'Relevant':<9} | {'Reasoning Summary':<52}")
    print("-" * 90)

    for i, case in enumerate(selected_cases, 1):
        query = case.get("text_customer", "").strip()
        tweet_id = case.get("tweet_id_customer", f"case_{i}")

        # 1. Retrieve FAISS context
        context_chunks = retrieve(query, k=3, exclude_exact_match=True)

        # 2. Get drafted reply
        draft_obj = draft_reply(query, with_rag=True, use_cache=True, exclude_exact_match=True)
        draft_text = apply_grounding_safeguards(draft_obj.draft_reply)

        # 3. Judge evaluation
        eval_result = judge_case(
            client=client,
            user_query=query,
            retrieved_context=context_chunks,
            generated_draft=draft_text,
            model=model,
            cache=judge_cache
        )

        if eval_result.is_faithful:
            faithful_count += 1
        if eval_result.is_relevant:
            relevant_count += 1

        f_str = "✓ True" if eval_result.is_faithful else "✗ FALSE"
        r_str = "✓ True" if eval_result.is_relevant else "✗ FALSE"
        reason_snippet = (eval_result.reasoning[:49] + "...") if len(eval_result.reasoning) > 52 else eval_result.reasoning

        print(f"{i:<3} | {tweet_id:<10} | {f_str:<9} | {r_str:<9} | {reason_snippet:<52}")

        results.append({
            "index": i,
            "tweet_id": tweet_id,
            "user_query": query,
            "retrieved_context": context_chunks,
            "generated_draft": draft_text,
            "is_faithful": eval_result.is_faithful,
            "is_relevant": eval_result.is_relevant,
            "reasoning": eval_result.reasoning,
        })

        # Subtle pacing to respect Groq rate limits
        time.sleep(0.15)

    total = len(selected_cases)
    faithfulness_pct = round((faithful_count / total) * 100, 2)
    relevance_pct = round((relevant_count / total) * 100, 2)

    # Isolate failure modes
    unfaithful_cases = [r for r in results if not r["is_faithful"]]
    irrelevant_cases = [r for r in results if not r["is_relevant"]]

    summary = {
        "benchmark": "RAG Triad Generative Quality Evaluation",
        "dataset": "eval/golden_test.json (Non-Escalated Stratum)",
        "sample_size": total,
        "judge_model": model,
        "metrics": {
            "faithfulness_count": faithful_count,
            "faithfulness_percentage": f"{faithfulness_pct}%",
            "faithfulness_ratio": round(faithful_count / total, 4),
            "relevance_count": relevant_count,
            "relevance_percentage": f"{relevance_pct}%",
            "relevance_ratio": round(relevant_count / total, 4),
            "hallucination_rate": f"{round(100 - faithfulness_pct, 2)}%",
            "irrelevance_rate": f"{round(100 - relevance_pct, 2)}%"
        },
        "failure_mode_analysis": {
            "unfaithful_failures": [
                {
                    "index": c["index"],
                    "tweet_id": c["tweet_id"],
                    "query": c["user_query"],
                    "draft": c["generated_draft"],
                    "reasoning": c["reasoning"]
                }
                for c in unfaithful_cases
            ],
            "irrelevant_failures": [
                {
                    "index": c["index"],
                    "tweet_id": c["tweet_id"],
                    "query": c["user_query"],
                    "draft": c["generated_draft"],
                    "reasoning": c["reasoning"]
                }
                for c in irrelevant_cases
            ]
        },
        "detailed_results": results
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("RAG TRIAD EVALUATION SUMMARY RECEIPTS:")
    print("=" * 90)
    print(f"Evaluated Cases:       {total}")
    print(f"Faithfulness Score:    {faithfulness_pct}% ({faithful_count}/{total})")
    print(f"Answer Relevance Score:{relevance_pct}% ({relevant_count}/{total})")
    print(f"Hallucination Rate:    {round(100 - faithfulness_pct, 2)}%")
    print(f"Irrelevance Rate:      {round(100 - relevance_pct, 2)}%")
    print(f"\nSaved full artifact to: {output_file}")

    if unfaithful_cases:
        print("\nDocumented Faithfulness Failures (Hallucinations):")
        for u in unfaithful_cases:
            print(f"  - Case #{u['index']} ({u['tweet_id']}): {u['reasoning']}")

    if irrelevant_cases:
        print("\nDocumented Relevance Failures:")
        for r in irrelevant_cases:
            print(f"  - Case #{r['index']} ({r['tweet_id']}): {r['reasoning']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Triad Generative Quality on non-escalated cases.")
    parser.add_argument("--sample-size", type=int, default=40, help="Number of non-escalated cases to evaluate (default: 40).")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="Groq model ID for LLM-as-a-Judge.")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Path to write JSON results artifact.")
    args = parser.parse_args()

    run_rag_triad_evaluation(
        sample_size=args.sample_size,
        model=args.model,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
