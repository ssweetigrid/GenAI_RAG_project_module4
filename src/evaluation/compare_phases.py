"""
src/evaluation/compare_phases.py
──────────────────────────────────
Load all saved RAGAS evaluation JSON files and print a comparison table
showing how each phase improved (or regressed) against the baseline.

Usage:
    python src/evaluation/compare_phases.py

Or import and call from a notebook:
    from src.evaluation.compare_phases import compare_all_phases
    compare_all_phases()
"""

import json
from pathlib import Path

import pandas as pd

from config.settings import ROOT_DIR

RESULTS_DIR = ROOT_DIR / "data" / "processed"

# Canonical metric display order
METRIC_ORDER = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
]


def load_all_ragas_results(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    """
    Scan the processed data directory for all eval_ragas_*.json files
    and return them as a dict: {pipeline_name: {metric: score}}.
    """
    found = {}
    for f in sorted(results_dir.glob("eval_ragas_*.json")):
        name = f.stem.replace("eval_ragas_", "")
        with open(f) as fh:
            found[name] = json.load(fh)
    return found


def compare_all_phases(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """
    Build and print a comparison table of all saved evaluation runs.

    Returns a DataFrame with pipelines as rows and RAGAS metrics as columns.
    """
    all_results = load_all_ragas_results(results_dir)

    if not all_results:
        print("⚠️  No evaluation results found.")
        print(f"   Run `python scripts/run_phase2_eval.py` first.")
        print(f"   Results are saved to: {results_dir}/eval_ragas_*.json")
        return pd.DataFrame()

    # Build a tidy DataFrame
    rows = []
    for pipeline_name, scores in all_results.items():
        row = {"pipeline": pipeline_name}
        for metric in METRIC_ORDER:
            row[metric] = round(scores.get(metric, float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("pipeline")

    # Reorder columns to canonical order (only include those that exist)
    cols = [m for m in METRIC_ORDER if m in df.columns]
    df   = df[cols]

    # Print the table
    print("\n" + "=" * 70)
    print("  Phase Comparison – RAGAS Evaluation Scores")
    print("=" * 70)
    print(df.to_string())

    # Print delta vs first row (baseline)
    if len(df) > 1:
        baseline = df.iloc[0]
        print("\n  Δ vs baseline (first row):")
        print("-" * 70)
        delta = df.iloc[1:].subtract(baseline)
        for col in delta.columns:
            print(f"\n  {col}:")
            for idx, val in delta[col].items():
                arrow = "↑" if val > 0 else ("↓" if val < 0 else "→")
                print(f"    {idx:<35} {arrow} {val:+.4f}")

    # Highlight the best score per metric
    print("\n  🏆 Best score per metric:")
    print("-" * 70)
    for col in df.columns:
        best_val = df[col].max()
        best_idx = df[col].idxmax()
        print(f"  {col:<25} {best_val:.4f}  ({best_idx})")

    print("=" * 70 + "\n")
    return df


def save_comparison_csv(df: pd.DataFrame, out_path: Path = None) -> Path:
    """Save the comparison DataFrame to CSV."""
    if out_path is None:
        out_path = RESULTS_DIR / "eval_comparison_all_phases.csv"
    df.to_csv(out_path)
    print(f"💾 Comparison saved → {out_path}")
    return out_path


if __name__ == "__main__":
    df = compare_all_phases()
    if not df.empty:
        save_comparison_csv(df)
