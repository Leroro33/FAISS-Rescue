
#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import random

ROOT = Path(__file__).resolve().parents[1]

RESCUE_TSV = ROOT / "metadata" / "rescue_reads.tsv"
PRED_TSV   = ROOT / "results" / "rescue_predictions_opq.tsv"
HITS_TSV   = ROOT / "results" / "retrieval_hits" / "hits_opq.tsv"
OUT_DIR    = ROOT / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PER_GENUS = 25
N_RANDOM = 25
RANDOM_SEED = 42


def load_data():
    rescue = pd.read_csv(RESCUE_TSV, sep="\t")
    pred   = pd.read_csv(PRED_TSV, sep="\t")
    hits   = pd.read_csv(HITS_TSV, sep="\t")
    return rescue, pred, hits


def merge_truth_predictions(rescue, pred):
    df = rescue.merge(pred, on="read_id", how="inner", suffixes=("_truth", "_pred"))

    # normalize likely column names
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "true_genus":
            rename_map[c] = "true_genus"
        elif lc == "true_species":
            rename_map[c] = "true_species"
        elif lc == "predicted_genus":
            rename_map[c] = "predicted_genus"
        elif lc == "predicted_species":
            rename_map[c] = "predicted_species"
        elif lc == "score":
            rename_map[c] = "score"
        elif lc == "margin":
            rename_map[c] = "margin"
    df = df.rename(columns=rename_map)

    if "predicted_genus" not in df.columns:
        # try to infer common alternatives
        for alt in ["genus_pred", "genus", "pred_genus"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "predicted_genus"})
                break

    if "predicted_species" not in df.columns:
        for alt in ["species_pred", "species", "pred_species"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "predicted_species"})
                break

    if "score" not in df.columns:
        for alt in ["predicted_genus_score", "genus_score", "top_score"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "score"})
                break

    if "margin" not in df.columns:
        for alt in ["top2_margin", "genus_margin"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "margin"})
                break

    df["is_correct_genus"] = df["true_genus"].astype(str) == df["predicted_genus"].astype(str)
    return df


def summarize_top_hits(hits):
    # infer likely columns
    cols = set(hits.columns)

    read_col = "read_id"
    rank_col = "rank" if "rank" in cols else None
    genus_col = None
    score_col = None

    for c in ["genus_hit", "hit_genus", "genus", "neighbor_genus"]:
        if c in cols:
            genus_col = c
            break

    for c in ["score", "similarity", "distance", "hit_score"]:
        if c in cols:
            score_col = c
            break

    if genus_col is None:
        raise ValueError(f"Could not find genus column in {HITS_TSV}. Columns: {sorted(hits.columns)}")

    # keep top 10 if rank exists
    if rank_col is not None:
        hits2 = hits[hits[rank_col] <= 10].copy()
        if score_col is not None:
            hits2 = hits2.sort_values([read_col, rank_col])
        else:
            hits2 = hits2.sort_values([read_col, rank_col])
    else:
        hits2 = hits.copy()

    def agg_one(g):
        genera = list(g[genus_col].astype(str))
        if score_col is not None:
            scores = list(g[score_col])
            pairs = [f"{gen}:{round(float(sc),4)}" for gen, sc in zip(genera, scores)]
            return pd.Series({
                "top_hit_genera": "; ".join(genera[:10]),
                "top_hit_genera_with_scores": " | ".join(pairs[:10]),
            })
        else:
            return pd.Series({
                "top_hit_genera": "; ".join(genera[:10]),
                "top_hit_genera_with_scores": "; ".join(genera[:10]),
            })

    out = hits2.groupby(read_col, as_index=False).apply(agg_one).reset_index()
    if "level_0" in out.columns:
        out = out.drop(columns=["level_0"])
    return out


def sample_group(df, genus_name, n, seed):
    sub = df[df["true_genus"].astype(str).str.lower() == genus_name.lower()].copy()
    if len(sub) == 0:
        return sub
    n = min(n, len(sub))
    return sub.sample(n=n, random_state=seed).copy()


def main():
    print("=== 14_audit_extremes.py ===")
    print(f"Loading:\n  {RESCUE_TSV}\n  {PRED_TSV}\n  {HITS_TSV}")

    rescue, pred, hits = load_data()
    merged = merge_truth_predictions(rescue, pred)
    hit_summary = summarize_top_hits(hits)

    merged = merged.merge(hit_summary, on="read_id", how="left")

    print(f"Loaded rescue rows: {len(rescue):,}")
    print(f"Loaded prediction rows: {len(pred):,}")
    print(f"Merged rows: {len(merged):,}")

    myco = sample_group(merged, "Mycobacterium", N_PER_GENUS, RANDOM_SEED)
    salm = sample_group(merged, "Salmonella", N_PER_GENUS, RANDOM_SEED + 1)
    rand = merged.sample(n=min(N_RANDOM, len(merged)), random_state=RANDOM_SEED + 2).copy()

    keep_cols = [
        "read_id",
        "kraken_status",
        "kraken_genus",
        "kraken_species",
        "true_genus",
        "true_species",
        "predicted_genus",
        "predicted_species",
        "is_correct_genus",
        "score",
        "margin",
        "top_hit_genera",
        "top_hit_genera_with_scores",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]

    myco_out = myco[keep_cols].sort_values(["is_correct_genus", "read_id"], ascending=[True, True])
    salm_out = salm[keep_cols].sort_values(["is_correct_genus", "read_id"], ascending=[True, True])
    rand_out = rand[keep_cols].sort_values(["is_correct_genus", "read_id"], ascending=[True, True])

    myco_file = OUT_DIR / "audit_mycobacterium_opq.tsv"
    salm_file = OUT_DIR / "audit_salmonella_opq.tsv"
    rand_file = OUT_DIR / "audit_random_opq.tsv"

    myco_out.to_csv(myco_file, sep="\t", index=False)
    salm_out.to_csv(salm_file, sep="\t", index=False)
    rand_out.to_csv(rand_file, sep="\t", index=False)

    def quick_stats(name, df):
        if len(df) == 0:
            print(f"{name}: no rows found")
            return
        correct = int(df["is_correct_genus"].sum())
        print(f"{name}: {correct}/{len(df)} correct ({correct/len(df):.1%})")

    print("\nQuick audit stats:")
    quick_stats("Mycobacterium", myco_out)
    quick_stats("Salmonella", salm_out)
    quick_stats("Random", rand_out)

    print("\nWritten:")
    print(f"  {myco_file}")
    print(f"  {salm_file}")
    print(f"  {rand_file}")

    print("\nPreview: Mycobacterium")
    print(myco_out.head(10).to_string(index=False))

    print("\nPreview: Salmonella")
    print(salm_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()