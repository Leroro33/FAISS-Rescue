#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESCUE_TSV = ROOT / "metadata" / "rescue_reads.tsv"
HITS_TSV   = ROOT / "results" / "retrieval_hits" / "hits_opq.tsv"
OUT_DIR    = ROOT / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG = OUT_DIR / "hit_composition_by_true_genus_opq.tsv"
OUT_TOP  = OUT_DIR / "hit_composition_top_opq.tsv"


def detect_hit_genus_column(df: pd.DataFrame) -> str:
    for c in ["genus_hit", "hit_genus", "genus", "neighbor_genus"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find hit genus column. Columns: {list(df.columns)}")


def main():
    print("=== 17_hit_composition_by_genus.py ===")
    print(f"Loading:\n  {RESCUE_TSV}\n  {HITS_TSV}")

    rescue = pd.read_csv(RESCUE_TSV, sep="\t")
    hits   = pd.read_csv(HITS_TSV, sep="\t")

    if "read_id" not in rescue.columns:
        raise ValueError("Column 'read_id' missing in rescue_reads.tsv")
    if "true_genus" not in rescue.columns:
        raise ValueError("Column 'true_genus' missing in rescue_reads.tsv")
    if "read_id" not in hits.columns:
        raise ValueError("Column 'read_id' missing in hits_opq.tsv")

    hit_genus_col = detect_hit_genus_column(hits)

    # Keep only needed columns
    rescue_small = rescue[["read_id", "true_genus"]].copy()
    hits_small   = hits[["read_id", hit_genus_col]].copy().rename(columns={hit_genus_col: "hit_genus"})

    merged = rescue_small.merge(hits_small, on="read_id", how="inner")
    merged["true_genus"] = merged["true_genus"].astype(str)
    merged["hit_genus"] = merged["hit_genus"].astype(str)

    print(f"Merged rows (all hit rows): {len(merged):,}")

    counts = (
        merged.groupby(["true_genus", "hit_genus"], as_index=False)
        .size()
        .rename(columns={"size": "hit_count"})
    )

    total_hits = (
        merged.groupby("true_genus", as_index=False)
        .size()
        .rename(columns={"size": "total_hit_rows"})
    )

    out = counts.merge(total_hits, on="true_genus", how="left")
    out["fraction_of_hits"] = out["hit_count"] / out["total_hit_rows"]
    out = out.sort_values(["true_genus", "hit_count"], ascending=[True, False]).reset_index(drop=True)

    out.to_csv(OUT_LONG, sep="\t", index=False)

    top = (
        out.sort_values(["true_genus", "hit_count"], ascending=[True, False])
           .groupby("true_genus", as_index=False)
           .head(10)
           .reset_index(drop=True)
    )
    top.to_csv(OUT_TOP, sep="\t", index=False)

    print(f"\nWritten:")
    print(f"  {OUT_LONG}")
    print(f"  {OUT_TOP}")

    print("\nTop hit composition per true genus:")
    for genus in top["true_genus"].drop_duplicates():
        sub = top[top["true_genus"] == genus]
        print(f"\n{genus}:")
        for _, r in sub.iterrows():
            print(f"  {r['hit_genus']}: {r['hit_count']:,} ({r['fraction_of_hits']:.4f})")

    print("\nTop 20 most frequent true_genus -> hit_genus pairs:")
    print(
        out.sort_values("hit_count", ascending=False)
           .head(20)
           .to_string(index=False)
    )


if __name__ == "__main__":
    main()
