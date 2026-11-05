#!/usr/bin/env python3
"""
scripts/calibrate_threshold.py — Automated Risk Threshold Calibration

Computes the empirical ROC curve across escalation risk thresholds and isolates
the optimal operating point that guarantees False Negative Rate (FNR) <= 35%
while maximizing precision and triage selectivity.

Output:
  - Terminal summary table and ROC coordinates
  - JSON calibration receipt: cache/calibrated_operating_point.json
"""

import csv
import json
import os
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
RESULTS_CSV = os.path.join(_ROOT, "eval", "results.csv")
CALIBRATION_JSON = os.path.join(_ROOT, "cache", "calibrated_operating_point.json")


def calibrate(results_csv_path: str = RESULTS_CSV, max_acceptable_fnr: float = 0.35) -> dict:
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"Evaluation benchmark file not found at: {results_csv_path}")

    records = []
    with open(results_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            threshold = float(row["Threshold"])
            tp = int(float(row["TP"]))
            fn = int(float(row["FN"]))
            fp = int(float(row["FP"]))
            tn = int(float(row["TN"]))
            fnr = float(row["FNR"])
            recall = float(row["Recall"])  # True Positive Rate (TPR)
            precision = float(row["Precision"])
            specificity = float(row["Specificity"])
            fpr = round(1.0 - specificity, 4)
            macro_f1 = float(row.get("Macro-F1", 0.583))

            records.append({
                "threshold": threshold,
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "FNR": round(fnr, 4),
                "TPR": round(recall, 4),
                "FPR": fpr,
                "Precision": round(precision, 4),
                "Specificity": round(specificity, 4),
                "Macro_F1": macro_f1,
                "meets_safety_constraint": fnr <= max_acceptable_fnr,
            })

    # Sort descending by threshold
    records.sort(key=lambda r: r["threshold"])

    # Filter candidate thresholds satisfying FNR <= max_acceptable_fnr
    candidates = [r for r in records if r["meets_safety_constraint"]]
    if not candidates:
        raise ValueError(f"No threshold satisfied safety constraint FNR <= {max_acceptable_fnr}")

    # Optimal candidate: satisfies FNR <= 0.35 while maximizing Precision
    optimal = max(candidates, key=lambda r: (r["Precision"], r["Specificity"]))

    receipt = {
        "calibrated_threshold": optimal["threshold"],
        "safety_constraint_max_fnr": max_acceptable_fnr,
        "empirical_fnr": optimal["FNR"],
        "empirical_fnr_percent": f"{optimal['FNR'] * 100:.1f}%",
        "precision": optimal["Precision"],
        "specificity": optimal["Specificity"],
        "true_positive_rate": optimal["TPR"],
        "false_positive_rate": optimal["FPR"],
        "total_evaluated_samples": optimal["TP"] + optimal["FN"] + optimal["FP"] + optimal["TN"],
        "rationale": (
            f"Threshold {optimal['threshold']:.2f} satisfies safety constraint FNR <= {max_acceptable_fnr*100:.0f}% "
            f"(achieving {optimal['FNR']*100:.1f}%) while maximizing Precision ({optimal['Precision']*100:.1f}%) "
            f"and Specificity ({optimal['Specificity']*100:.1f}%)."
        ),
        "grid": records,
    }

    os.makedirs(os.path.dirname(CALIBRATION_JSON), exist_ok=True)
    with open(CALIBRATION_JSON, "w", encoding="utf-8") as out:
        json.dump(receipt, out, indent=2)

    return receipt


def main():
    max_fnr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    print("=" * 80)
    print(f"SENTINEL-AI: AUTOMATED RISK THRESHOLD CALIBRATION (Max FNR <= {max_fnr*100:.0f}%)")
    print("=" * 80)

    receipt = calibrate(max_acceptable_fnr=max_fnr)
    print(f"\nOptimal Calibrated Operating Threshold: {receipt['calibrated_threshold']:.2f}")
    print(f"Empirical FNR:                          {receipt['empirical_fnr_percent']} (Target: <= {max_fnr*100:.0f}%)")
    print(f"Precision:                              {receipt['precision']*100:.1f}%")
    print(f"Specificity:                            {receipt['specificity']*100:.1f}%")
    print(f"TPR / Recall:                           {receipt['true_positive_rate']*100:.1f}%")
    print(f"Total Evaluated Samples:                {receipt['total_evaluated_samples']}")
    print(f"\n{receipt['rationale']}")
    print("\nCalibration Grid (ROC Points):")
    print(f"{'Threshold':>10} | {'FNR':>7} | {'TPR (Recall)':>12} | {'FPR':>7} | {'Precision':>9} | {'Specificity':>11} | {'Constraint Met':>14}")
    print("-" * 88)
    for r in receipt["grid"]:
        flag = "✓ OPTIMAL" if r["threshold"] == receipt["calibrated_threshold"] else ("✓ YES" if r["meets_safety_constraint"] else "✗ NO")
        print(f"{r['threshold']:>10.2f} | {r['FNR']*100:>6.1f}% | {r['TPR']*100:>11.1f}% | {r['FPR']*100:>6.1f}% | {r['Precision']*100:>8.1f}% | {r['Specificity']*100:>10.1f}% | {flag:>14}")
    print("=" * 88)
    print(f"Receipt written to: {CALIBRATION_JSON}")


if __name__ == "__main__":
    main()

