#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

HITS_TSV = ROOT / "results" / "retrieval_hits" / "hits_opq.tsv"
OUT_TSV  = ROOT / "results" / "rescue_predictions_opq_k3.tsv"


def detect_hit_genus_column(df: pd.DataFrame) -> str:
    for c in ["genus_hit", "hit_genus", "genus", "neighbor_genus"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find hit genus column. Columns: {list(df.columns)}")


def detect_hit_species_column(df: pd.DataFrame) -> str | None:
    for c in ["species_hit", "hit_species", "species", "neighbor_species"]:
        if c in df.columns:
            return c
    return None


def detect_score_column(df: pd.DataFrame) -> str | None:
    for c in ["score", "similarity", "hit_score"]:
        if c in df.columns:
            return c
    return None


def main():
    print("=== 21_revote_opq_k3.py ===")
    print(f"Loading: {HITS_TSV}")

    hits = pd.read_csv(HITS_TSV, sep="\t")

    if "read_id" not in hits.columns:
        raise ValueError("hits_opq.tsv must contain read_id")

    genus_col = detect_hit_genus_column(hits)
    species_col = detect_hit_species_column(hits)
    score_col = detect_score_column(hits)

    if "rank" in hits.columns:
        hits = hits.sort_values(["read_id", "rank"])
        hits = hits[hits["rank"] <= 3].copy()
    else:
        hits = hits.sort_values(["read_id"]).groupby("read_id", as_index=False).head(3).copy()

    rows = []

    for read_id, sub in hits.groupby("read_id", sort=False):
        genera = list(sub[genus_col].astype(str))
        genus_counts = Counter(genera)
        predicted_genus, genus_votes = genus_counts.most_common(1)[0]

        predicted_species = None
        if species_col is not None:
            species = list(sub[species_col].astype(str))
            species_counts = Counter(species)
            predicted_species, _ = species_counts.most_common(1)[0]

        top_score = None
        second_score = None
        margin = None
        if score_col is not None:
            scores = sorted([float(x) for x in sub[score_col]], reverse=True)
            if len(scores) >= 1:
                top_score = scores[0]
            if len(scores) >= 2:
                second_score = scores[1]
                margin = top_score - second_score
            elif len(scores) == 1:
                margin = top_score

        agreement_fraction_top3 = genus_votes / len(sub)

        rows.append({
            "read_id": read_id,
            "predicted_genus": predicted_genus,
            "predicted_species": predicted_species if predicted_species is not None else "",
            "genus_votes_top3": genus_votes,
            "agreement_fraction_top3": agreement_fraction_top3,
            "score": top_score if top_score is not None else "",
            "margin": margin if margin is not None else "",
            "mode": "unweighted",
            "k": 3,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TSV, sep="\t", index=False)

    print(f"Written: {OUT_TSV}")
    print(f"Rows: {len(out):,}")
    print("\nPreview:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
