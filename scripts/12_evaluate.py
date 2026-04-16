#!/usr/bin/env python3
"""
12_evaluate.py
==============
Evaluate the CompressedRescue pipeline: Kraken-only vs hybrid (Kraken +
FAISS rescue).  Computes genus-level precision, recall, F1, and a
rescue-improvement breakdown (improved / worsened / still-wrong /
unchanged-correct).

Inputs
------
  metadata/rescue_reads.tsv          — per-read truth + Kraken call
  results/rescue_predictions_pq.tsv  — hybrid prediction (PQ index)
  results/rescue_predictions_opq.tsv — hybrid prediction (OPQ index)

Outputs
-------
  results/metrics/summary.tsv        — headline metrics per index
  results/metrics/rescue_breakdown_pq.tsv
  results/metrics/rescue_breakdown_opq.tsv
  results/metrics/per_genus_pq.tsv
  results/metrics/per_genus_opq.tsv

Usage
-----
  conda activate bioenv
  cd ~/CompressedRescue
  python scripts/12_evaluate.py
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

# ── defaults ────────────────────────────────────────────────────────────
ROOT        = os.path.expanduser("~/CompressedRescue")
RESCUE_TSV  = os.path.join(ROOT, "metadata", "rescue_reads.tsv")
RESULTS_DIR = os.path.join(ROOT, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
LOG_DIR     = os.path.join(ROOT, "logs")


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "12_evaluate.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


# ── metric helpers ──────────────────────────────────────────────────────

def prf(predicted: pd.Series, truth: pd.Series, missing_pred="") -> dict:
    """
    Precision / recall / F1 at a single taxonomic level.

    • A prediction counts as a 'call' unless it equals `missing_pred`.
    • Precision = correct_calls / calls_made
    • Recall    = correct_calls / all_reads_with_truth
    • F1        = harmonic mean
    """
    n_total   = len(truth)
    made_mask = predicted != missing_pred
    n_made    = int(made_mask.sum())
    n_correct = int(((predicted == truth) & made_mask).sum())

    precision = n_correct / n_made   if n_made   > 0 else 0.0
    recall    = n_correct / n_total  if n_total  > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "n_total":   n_total,
        "n_made":    n_made,
        "n_correct": n_correct,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def classify_rescue_row(k_genus: str, h_genus: str, true_genus: str) -> str:
    """Bucket each rescue read by what rescue did to it."""
    k_ok = (k_genus == true_genus)
    h_ok = (h_genus == true_genus)
    k_missing = (k_genus in ("", "-", None) or pd.isna(k_genus))

    if k_missing and h_ok:          return "rescue_improved"      # Kraken missed, rescue got it
    if k_missing and not h_ok:      return "rescue_still_wrong"   # Kraken missed, rescue also wrong
    if k_ok and h_ok:               return "both_correct"         # rescue didn't change right call
    if k_ok and not h_ok:           return "rescue_worsened"      # rescue broke a right call
    if not k_ok and h_ok:           return "rescue_fixed"         # rescue fixed a wrong Kraken call
    if not k_ok and not h_ok:       return "both_wrong"           # both wrong (possibly different wrong)
    return "other"


# ── main evaluation per index ───────────────────────────────────────────

def evaluate_index(tag: str, rescue_df: pd.DataFrame,
                   pred_path: str, metrics_dir: str) -> dict:
    """Run all metrics for one index (pq/opq/flat)."""
    logging.info(f"Evaluating: {tag}")
    logging.info(f"  loading {pred_path}")
    pred_df = pd.read_csv(pred_path, sep="\t")
    logging.info(f"    {len(pred_df):,} predictions")

    # join on read_id
    df = rescue_df.merge(pred_df, on="read_id", how="inner",
                         suffixes=("_rescue", ""))
    logging.info(f"  joined on read_id: {len(df):,} rows")
    if len(df) != len(rescue_df):
        logging.warning(
            f"  {len(rescue_df) - len(df):,} rescue reads had no prediction"
        )

    # normalise Kraken genus column: '-' means no call
    df["kraken_genus_norm"] = df["kraken_genus"].replace("-", "")

    # ── genus-level metrics ──
    kraken_metrics = prf(df["kraken_genus_norm"], df["true_genus"],
                         missing_pred="")
    hybrid_metrics = prf(df["predicted_genus"],  df["true_genus"],
                         missing_pred="")

    logging.info(f"  genus — Kraken-only : "
                 f"P={kraken_metrics['precision']}  "
                 f"R={kraken_metrics['recall']}  "
                 f"F1={kraken_metrics['f1']}")
    logging.info(f"  genus — Hybrid ({tag}): "
                 f"P={hybrid_metrics['precision']}  "
                 f"R={hybrid_metrics['recall']}  "
                 f"F1={hybrid_metrics['f1']}")

    # ── species-level (hybrid only; Kraken often doesn't call species) ──
    species_metrics = prf(df["predicted_species"], df["true_species"],
                          missing_pred="")
    logging.info(f"  species — Hybrid ({tag}): "
                 f"P={species_metrics['precision']}  "
                 f"R={species_metrics['recall']}  "
                 f"F1={species_metrics['f1']}  "
                 f"(calls made: {species_metrics['n_made']:,})")

    # ── rescue breakdown ──
    df["rescue_bucket"] = [
        classify_rescue_row(k, h, t)
        for k, h, t in zip(df["kraken_genus_norm"],
                           df["predicted_genus"],
                           df["true_genus"])
    ]
    breakdown = (df["rescue_bucket"].value_counts()
                 .rename_axis("bucket").reset_index(name="count"))
    breakdown["percent"] = (breakdown["count"] * 100.0 /
                            len(df)).round(2)
    bp = os.path.join(metrics_dir, f"rescue_breakdown_{tag}.tsv")
    breakdown.to_csv(bp, sep="\t", index=False)
    logging.info(f"  rescue breakdown:")
    for _, row in breakdown.iterrows():
        logging.info(f"    {row['bucket']:25s}  "
                     f"{row['count']:>10,}  ({row['percent']:.2f}%)")
    logging.info(f"  ✓ saved → {bp}")

    # ── per-genus metrics ──
    rows = []
    for g in sorted(df["true_genus"].unique()):
        sub = df[df["true_genus"] == g]
        m_k = prf(sub["kraken_genus_norm"], sub["true_genus"], missing_pred="")
        m_h = prf(sub["predicted_genus"],   sub["true_genus"], missing_pred="")
        rows.append({
            "genus": g,
            "n_reads":        m_k["n_total"],
            "kraken_recall":  m_k["recall"],
            "hybrid_recall":  m_h["recall"],
            "kraken_precision": m_k["precision"],
            "hybrid_precision": m_h["precision"],
            "delta_recall":   round(m_h["recall"] - m_k["recall"], 4),
        })
    per_genus = pd.DataFrame(rows).sort_values("n_reads", ascending=False)
    pgp = os.path.join(metrics_dir, f"per_genus_{tag}.tsv")
    per_genus.to_csv(pgp, sep="\t", index=False)
    logging.info(f"  ✓ saved → {pgp}")

    # ── summary row ──
    return {
        "index":              tag,
        "n_rescue_reads":     len(df),
        "kraken_genus_P":     kraken_metrics["precision"],
        "kraken_genus_R":     kraken_metrics["recall"],
        "kraken_genus_F1":    kraken_metrics["f1"],
        "hybrid_genus_P":     hybrid_metrics["precision"],
        "hybrid_genus_R":     hybrid_metrics["recall"],
        "hybrid_genus_F1":    hybrid_metrics["f1"],
        "hybrid_species_P":   species_metrics["precision"],
        "hybrid_species_R":   species_metrics["recall"],
        "hybrid_species_F1":  species_metrics["f1"],
        "delta_genus_F1":     round(hybrid_metrics["f1"] -
                                    kraken_metrics["f1"], 4),
    }


# ── main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CompressedRescue pipeline")
    p.add_argument("--rescue_tsv",  default=RESCUE_TSV)
    p.add_argument("--results_dir", default=RESULTS_DIR)
    p.add_argument("--metrics_dir", default=METRICS_DIR)
    p.add_argument("--log_dir",     default=LOG_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_dir)

    logging.info("=" * 60)
    logging.info("12_evaluate.py   — CompressedRescue")
    logging.info("=" * 60)

    os.makedirs(args.metrics_dir, exist_ok=True)

    # ── load truth + Kraken ──
    logging.info(f"Loading rescue truth: {args.rescue_tsv}")
    rescue_df = pd.read_csv(args.rescue_tsv, sep="\t")
    logging.info(f"  {len(rescue_df):,} rescue reads")
    logging.info(f"  columns: {list(rescue_df.columns)}")

    # ── find prediction files ──
    pred_files = sorted([
        f for f in os.listdir(args.results_dir)
        if f.startswith("rescue_predictions_") and f.endswith(".tsv")
    ])
    if not pred_files:
        sys.exit(f"ERROR: no rescue_predictions_*.tsv in {args.results_dir}")

    logging.info(f"Found {len(pred_files)} prediction file(s): {pred_files}")

    # ── evaluate each ──
    summary_rows = []
    for pf in pred_files:
        tag = pf.replace("rescue_predictions_", "").replace(".tsv", "")
        path = os.path.join(args.results_dir, pf)
        logging.info("")
        summary_rows.append(evaluate_index(tag, rescue_df, path,
                                           args.metrics_dir))

    # ── save summary ──
    summary = pd.DataFrame(summary_rows)
    sp = os.path.join(args.metrics_dir, "summary.tsv")
    summary.to_csv(sp, sep="\t", index=False)
    logging.info("")
    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info("=" * 60)
    logging.info("\n" + summary.to_string(index=False))
    logging.info(f"\n✓ saved → {sp}")
    logging.info("")
    logging.info("Done.  Next → 13_make_figures.py")


if __name__ == "__main__":
    main()
