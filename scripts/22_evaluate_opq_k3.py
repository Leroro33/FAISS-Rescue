#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESCUE_TSV = ROOT / "metadata" / "rescue_reads.tsv"
PRED_TSV   = ROOT / "results" / "rescue_predictions_opq_k3.tsv"
OUT_DIR    = ROOT / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TSV = OUT_DIR / "summary_opq_k3.tsv"


def prf(correct: int, total_pred: int, total_true: int):
    precision = correct / total_pred if total_pred else 0.0
    recall = correct / total_true if total_true else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def main():
    print("=== 22_evaluate_opq_k3.py ===")
    print(f"Loading:\n  {RESCUE_TSV}\n  {PRED_TSV}")

    rescue = pd.read_csv(RESCUE_TSV, sep="\t")
    pred = pd.read_csv(PRED_TSV, sep="\t")

    df = rescue.merge(pred, on="read_id", how="inner")

    if "true_genus" not in df.columns:
        raise ValueError("Column 'true_genus' missing after merge")
    if "predicted_genus" not in df.columns:
        raise ValueError("Column 'predicted_genus' missing after merge")

    if "true_species" not in df.columns:
        raise ValueError("Column 'true_species' missing after merge")
    if "predicted_species" not in df.columns:
        raise ValueError("Column 'predicted_species' missing after merge")

    df["true_genus"] = df["true_genus"].astype(str)
    df["predicted_genus"] = df["predicted_genus"].astype(str)
    df["true_species"] = df["true_species"].astype(str)
    df["predicted_species"] = df["predicted_species"].astype(str)

    genus_correct = int((df["true_genus"] == df["predicted_genus"]).sum())
    species_correct = int((df["true_species"] == df["predicted_species"]).sum())

    n = len(df)

    genus_p, genus_r, genus_f1 = prf(genus_correct, n, n)
    species_p, species_r, species_f1 = prf(species_correct, n, n)

    out = pd.DataFrame([{
        "index": "opq_k3",
        "n_rescue_reads": n,
        "hybrid_genus_P": genus_p,
        "hybrid_genus_R": genus_r,
        "hybrid_genus_F1": genus_f1,
        "hybrid_species_P": species_p,
        "hybrid_species_R": species_r,
        "hybrid_species_F1": species_f1,
    }])

    out.to_csv(OUT_TSV, sep="\t", index=False)

    print(f"\nWritten: {OUT_TSV}")
    print("\nSummary:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
