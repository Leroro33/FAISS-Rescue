#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]

RESCUE_TSV = ROOT / "metadata" / "rescue_reads.tsv"
HITS_TSV   = ROOT / "results" / "retrieval_hits" / "hits_opq.tsv"
OUT_DIR    = ROOT / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_DIR / "vote_sensitivity_opq.tsv"
OUT_PRED = OUT_DIR / "vote_sensitivity_predictions_opq.tsv"


def detect_hit_genus_column(df: pd.DataFrame) -> str:
    for c in ["genus_hit", "hit_genus", "genus", "neighbor_genus"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find hit genus column. Columns: {list(df.columns)}")


def detect_score_column(df: pd.DataFrame) -> str | None:
    for c in ["score", "similarity", "hit_score"]:
        if c in df.columns:
            return c
    return None


def predict_group(sub: pd.DataFrame, genus_col: str, score_col: str | None, k: int, mode: str) -> str:
    sub = sub.head(k)

    if mode == "unweighted":
        cnt = Counter(sub[genus_col].astype(str))
        return cnt.most_common(1)[0][0]

    weights = defaultdict(float)
    if score_col is None:
        for g in sub[genus_col].astype(str):
            weights[g] += 1.0
    else:
        for _, row in sub.iterrows():
            weights[str(row[genus_col])] += float(row[score_col])

    return sorted(weights.items(), key=lambda x: (-x[1], x[0]))[0][0]


def compute_metrics(df: pd.DataFrame) -> dict:
    correct = (df["true_genus"] == df["predicted_genus"]).sum()
    total = len(df)
    recall = correct / total if total else 0.0
    precision = recall
    f1 = recall
    return {
        "n_reads": total,
        "n_correct": int(correct),
        "genus_precision": precision,
        "genus_recall": recall,
        "genus_f1": f1,
    }


def main():
    print("=== 20_vote_sensitivity.py ===")
    print(f"Loading:\n  {RESCUE_TSV}\n  {HITS_TSV}")

    rescue = pd.read_csv(RESCUE_TSV, sep="\t")
    hits = pd.read_csv(HITS_TSV, sep="\t")

    genus_col = detect_hit_genus_column(hits)
    score_col = detect_score_column(hits)

    if "read_id" not in rescue.columns or "true_genus" not in rescue.columns:
        raise ValueError("rescue_reads.tsv must contain read_id and true_genus")

    if "read_id" not in hits.columns:
        raise ValueError("hits_opq.tsv must contain read_id")

    if "rank" in hits.columns:
        hits = hits.sort_values(["read_id", "rank"])
    else:
        hits = hits.sort_values(["read_id"])

    ks = [3, 5, 10]
    modes = ["unweighted", "weighted"]

    pred_rows = []
    summary_rows = []

    grouped = hits.groupby("read_id", sort=False)
    truth_map = rescue.set_index("read_id")["true_genus"].astype(str).to_dict()

    for k in ks:
        for mode in modes:
            print(f"Running mode={mode}, k={k}")
            rows = []

            for read_id, sub in grouped:
                pred_genus = predict_group(sub, genus_col, score_col, k, mode)
                true_genus = truth_map.get(read_id)
                if true_genus is None:
                    continue
                rows.append({
                    "read_id": read_id,
                    "true_genus": true_genus,
                    "predicted_genus": pred_genus,
                    "k": k,
                    "mode": mode,
                    "is_correct": pred_genus == true_genus,
                })

            pred_df = pd.DataFrame(rows)
            metrics = compute_metrics(pred_df)
            metrics["k"] = k
            metrics["mode"] = mode
            summary_rows.append(metrics)
            pred_rows.append(pred_df)

    summary = pd.DataFrame(summary_rows).sort_values(["mode", "k"]).reset_index(drop=True)
    preds = pd.concat(pred_rows, ignore_index=True)

    summary.to_csv(OUT_SUMMARY, sep="\t", index=False)
    preds.to_csv(OUT_PRED, sep="\t", index=False)

    print(f"\nWritten:")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_PRED}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
