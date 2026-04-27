#!/usr/bin/env python3
"""
compare_v1_v2.py

Prints a side-by-side comparison of V1 (original) vs V2 (canonical + IDF)
metrics, so you can see whether the embedding upgrade helped.

Reads:
    results/metrics/summary.tsv         (V1 — your original pipeline)
    results/metrics_v2/summary.tsv      (V2 — produced by run_v2_pipeline.sh)
    results/metrics/per_genus_opq.tsv
    results/metrics_v2/per_genus_opq.tsv

USAGE
    conda activate bioenv
    cd ~/CompressedRescue
    python scripts/compare_v1_v2.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.home() / "CompressedRescue"
V1_DIR       = PROJECT_ROOT / "results" / "metrics"
V2_DIR       = PROJECT_ROOT / "results" / "metrics_v2"


def load_or_die(path):
    if not path.exists():
        sys.exit(f"ERROR: not found: {path}")
    return pd.read_csv(path, sep="\t")


def fmt_delta(v1, v2, pct=True):
    """Format a v2-v1 delta with a sign and color-ish symbol."""
    if pd.isna(v1) or pd.isna(v2):
        return "n/a"
    d = v2 - v1
    sign = "+" if d >= 0 else ""
    arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
    if pct:
        return f"{sign}{d*100:.1f}pp  {arrow}"
    return f"{sign}{d:.3f}  {arrow}"


def main():
    print("=" * 70)
    print(f"{'V1 vs V2 COMPARISON':^70}")
    print("=" * 70)

    # ── Headline summary ─────────────────────────────────────────────────────
    v1_sum = load_or_die(V1_DIR / "summary.tsv")
    v2_sum = load_or_die(V2_DIR / "summary.tsv")

    print(f"\nSUMMARY (genus-level F1 by method)")
    print("-" * 70)
    print(f"  V1 columns: {list(v1_sum.columns)}")
    print(f"  V2 columns: {list(v2_sum.columns)}")
    print()
    print(f"V1 summary:")
    print(v1_sum.to_string(index=False))
    print()
    print(f"V2 summary:")
    print(v2_sum.to_string(index=False))

    # Try to print a comparison if columns line up
    method_col = next((c for c in ["method", "approach", "name"]
                       if c in v1_sum.columns and c in v2_sum.columns), None)
    f1_col = next((c for c in ["genus_f1", "f1", "F1"]
                   if c in v1_sum.columns and c in v2_sum.columns), None)
    if method_col and f1_col:
        print(f"\nF1 by method (V1 → V2):")
        print(f"  {'Method':<30} {'V1':>8} {'V2':>8}  {'Δ':>12}")
        print(f"  " + "-" * 60)
        merged = v1_sum.merge(v2_sum, on=method_col,
                              suffixes=("_v1", "_v2"))
        for _, r in merged.iterrows():
            v1_v = r[f"{f1_col}_v1"]
            v2_v = r[f"{f1_col}_v2"]
            print(f"  {str(r[method_col]):<30} "
                  f"{v1_v:>8.3f} {v2_v:>8.3f}  {fmt_delta(v1_v, v2_v):>12}")

    # ── Per-genus comparison (OPQ) ───────────────────────────────────────────
    v1_pg_path = V1_DIR / "per_genus_opq.tsv"
    v2_pg_path = V2_DIR / "per_genus_opq.tsv"

    if v1_pg_path.exists() and v2_pg_path.exists():
        v1_pg = pd.read_csv(v1_pg_path, sep="\t")
        v2_pg = pd.read_csv(v2_pg_path, sep="\t")

        print(f"\n\nPER-GENUS RECALL — OPQ (V1 vs V2)")
        print("-" * 70)

        genus_col = next((c for c in ["genus", "true_genus"]
                          if c in v1_pg.columns and c in v2_pg.columns), None)
        recall_col = next((c for c in ["recall", "Recall"]
                           if c in v1_pg.columns and c in v2_pg.columns), None)

        if genus_col and recall_col:
            merged = v1_pg.merge(v2_pg, on=genus_col,
                                  suffixes=("_v1", "_v2"))
            print(f"  {'Genus':<20} {'V1 recall':>10} "
                  f"{'V2 recall':>10}  {'Δ':>12}")
            print(f"  " + "-" * 60)
            # Sort by V1 recall ascending so the worst-performing genera
            # (where we hope to see the biggest improvement) are at the top.
            merged = merged.sort_values(f"{recall_col}_v1")
            for _, r in merged.iterrows():
                v1_v = r[f"{recall_col}_v1"]
                v2_v = r[f"{recall_col}_v2"]
                print(f"  {str(r[genus_col]):<20} "
                      f"{v1_v:>10.1%} {v2_v:>10.1%}  "
                      f"{fmt_delta(v1_v, v2_v):>12}")
        else:
            print(f"\n  V1 columns: {list(v1_pg.columns)}")
            print(f"  V2 columns: {list(v2_pg.columns)}")
            print(f"  Could not auto-detect genus/recall columns. "
                  f"Inspect manually.")

    print()
    print("=" * 70)
    print(f"Full V1 metrics: {V1_DIR}")
    print(f"Full V2 metrics: {V2_DIR}")


if __name__ == "__main__":
    main()
