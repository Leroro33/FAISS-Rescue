#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESCUE_TSV = ROOT / "metadata" / "rescue_reads.tsv"
PRED_TSV   = ROOT / "results" / "rescue_predictions_opq.tsv"
OUT_DIR    = ROOT / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONFUSION = OUT_DIR / "confusion_opq.tsv"
OUT_TOP       = OUT_DIR / "top_confusions_opq.tsv"
OUT_MATRIX    = OUT_DIR / "confusion_matrix_opq.tsv"


def main():
    print("=== 16_confusion_by_genus.py ===")
    print(f"Loading:\n  {RESCUE_TSV}\n  {PRED_TSV}")

    rescue = pd.read_csv(RESCUE_TSV, sep="\t")
    pred   = pd.read_csv(PRED_TSV, sep="\t")

    df = rescue.merge(pred, on="read_id", how="inner")

    if "predicted_genus" not in df.columns:
        for alt in ["genus", "genus_pred", "pred_genus"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "predicted_genus"})
                break

    if "true_genus" not in df.columns:
        raise ValueError("Column 'true_genus' not found after merge.")
    if "predicted_genus" not in df.columns:
        raise ValueError("Column 'predicted_genus' not found after merge.")

    df["true_genus"] = df["true_genus"].astype(str)
    df["predicted_genus"] = df["predicted_genus"].astype(str)

    # Long-format confusion table
    confusion = (
        df.groupby(["true_genus", "predicted_genus"], as_index=False)
          .size()
          .rename(columns={"size": "count"})
          .sort_values(["true_genus", "count"], ascending=[True, False])
          .reset_index(drop=True)
    )
    confusion.to_csv(OUT_CONFUSION, sep="\t", index=False)

    # Only wrong predictions
    wrong = confusion[confusion["true_genus"] != confusion["predicted_genus"]].copy()
    wrong = wrong.sort_values("count", ascending=False).reset_index(drop=True)
    wrong.to_csv(OUT_TOP, sep="\t", index=False)

    # Matrix form
    matrix = (
        confusion.pivot(index="true_genus", columns="predicted_genus", values="count")
        .fillna(0)
        .astype(int)
    )
    matrix.to_csv(OUT_MATRIX, sep="\t")

    print(f"\nWritten:")
    print(f"  {OUT_CONFUSION}")
    print(f"  {OUT_TOP}")
    print(f"  {OUT_MATRIX}")

    print("\nTop 20 wrong genus confusions:")
    print(wrong.head(20).to_string(index=False))

    print("\nTop predictions within each true genus:")
    top_per_true = (
        confusion.sort_values(["true_genus", "count"], ascending=[True, False])
                 .groupby("true_genus", as_index=False)
                 .head(5)
    )
    print(top_per_true.to_string(index=False))

    print("\nRow-normalized confusion (top 5 predicted genera per true genus):")
    row_sums = matrix.sum(axis=1)
    for true_genus in matrix.index:
        row = matrix.loc[true_genus]
        frac = (row / row_sums[true_genus]).sort_values(ascending=False).head(5)
        print(f"\n{true_genus}:")
        for pred_genus, value in frac.items():
            print(f"  {pred_genus}: {value:.4f}")


if __name__ == "__main__":
    main()
