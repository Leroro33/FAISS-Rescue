#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESCUE_TSV = ROOT / "metadata" / "rescue_reads.tsv"
PRED_TSV   = ROOT / "results" / "rescue_predictions_opq_k3.tsv"
OUT_DIR    = ROOT / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TSV = OUT_DIR / "per_genus_diagnostics_opq_k3.tsv"


def main():
    print("=== 23_per_genus_diagnostics_opq_k3.py ===")
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
    df["is_correct_genus"] = df["true_genus"] == df["predicted_genus"]

    rows = []

    for genus, sub in df.groupby("true_genus", sort=False):
        n_rescue_reads = len(sub)
        n_correct = int(sub["is_correct_genus"].sum())
        n_wrong = n_rescue_reads - n_correct
        recall = n_correct / n_rescue_reads if n_rescue_reads else 0.0

        wrong = sub.loc[~sub["is_correct_genus"], "predicted_genus"]
        if len(wrong) > 0:
            vc = wrong.value_counts()
            most_common_wrong_genus = vc.index[0]
            most_common_wrong_count = int(vc.iloc[0])
        else:
            most_common_wrong_genus = ""
            most_common_wrong_count = 0

        rows.append({
            "true_genus": genus,
            "n_rescue_reads": n_rescue_reads,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "recall": recall,
            "most_common_wrong_genus": most_common_wrong_genus,
            "most_common_wrong_count": most_common_wrong_count,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["recall", "n_rescue_reads"], ascending=[False, False]).reset_index(drop=True)
    out.to_csv(OUT_TSV, sep="\t", index=False)

    print(f"\nWritten: {OUT_TSV}")
    print("\nPer-genus diagnostics (OPQ k=3):")
    print(out.to_string(index=False))

    print("\nTop difficult genera (lowest recall):")
    print(out.sort_values("recall", ascending=True).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
