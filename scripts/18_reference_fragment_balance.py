#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

GENOMES_TSV   = ROOT / "metadata" / "genomes.tsv"
SPLITS_TSV    = ROOT / "metadata" / "splits.tsv"
FRAGMENTS_TSV = ROOT / "metadata" / "fragments.tsv"
OUT_DIR       = ROOT / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TSV = OUT_DIR / "reference_fragment_balance.tsv"
OUT_GENOME_TSV = OUT_DIR / "reference_fragments_per_genome.tsv"


def unify_column(df, target, candidates):
    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            return df
    raise ValueError(f"Could not find a column for '{target}'. Available columns: {list(df.columns)}")


def main():
    print("=== 18_reference_fragment_balance.py ===")
    print(f"Loading:\n  {GENOMES_TSV}\n  {SPLITS_TSV}\n  {FRAGMENTS_TSV}")

    genomes = pd.read_csv(GENOMES_TSV, sep="\t")
    splits = pd.read_csv(SPLITS_TSV, sep="\t")
    frags = pd.read_csv(FRAGMENTS_TSV, sep="\t")

    if "genome_id" not in genomes.columns:
        raise ValueError("genomes.tsv must contain genome_id")
    if "genome_id" not in splits.columns or "split" not in splits.columns:
        raise ValueError("splits.tsv must contain genome_id and split")
    if "genome_id" not in frags.columns:
        raise ValueError("fragments.tsv must contain genome_id")

    genomes = unify_column(genomes, "genus", ["genus"])
    genomes = unify_column(genomes, "species", ["species"])

    train = splits[splits["split"] == "train_ref"][["genome_id", "split"]].copy()
    genomes_train = genomes.merge(train, on="genome_id", how="inner")

    # Only attach genus/species from genomes_train if not already needed from fragments
    frags_train = frags.merge(
        genomes_train[["genome_id", "genus", "species"]],
        on="genome_id",
        how="inner",
        suffixes=("_frag", "_genome")
    )

    # Normalize genus/species after merge
    frags_train = unify_column(
        frags_train,
        "genus_norm",
        ["genus", "genus_genome", "genus_frag", "genus_x", "genus_y"]
    )
    frags_train = unify_column(
        frags_train,
        "species_norm",
        ["species", "species_genome", "species_frag", "species_x", "species_y"]
    )

    # fragments per genome
    per_genome = (
        frags_train.groupby(["genome_id", "genus_norm", "species_norm"], as_index=False)
        .size()
        .rename(columns={
            "size": "n_fragments",
            "genus_norm": "genus",
            "species_norm": "species"
        })
        .sort_values(["genus", "species", "genome_id"])
        .reset_index(drop=True)
    )
    per_genome.to_csv(OUT_GENOME_TSV, sep="\t", index=False)

    genomes_per_genus = (
        genomes_train.groupby("genus", as_index=False)["genome_id"]
        .nunique()
        .rename(columns={"genome_id": "n_train_genomes"})
    )

    frags_per_genus = (
        per_genome.groupby("genus", as_index=False)["n_fragments"]
        .sum()
        .rename(columns={"n_fragments": "n_reference_fragments"})
    )

    stats = (
        per_genome.groupby("genus")["n_fragments"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={
            "mean": "mean_fragments_per_genome",
            "median": "median_fragments_per_genome",
            "min": "min_fragments_per_genome",
            "max": "max_fragments_per_genome",
        })
    )

    out = genomes_per_genus.merge(frags_per_genus, on="genus", how="outer")
    out = out.merge(stats, on="genus", how="outer")

    total_frags = out["n_reference_fragments"].sum()
    out["fraction_of_reference_fragments"] = out["n_reference_fragments"] / total_frags

    out = out.sort_values("n_reference_fragments", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_TSV, sep="\t", index=False)

    print("\nWritten:")
    print(f"  {OUT_TSV}")
    print(f"  {OUT_GENOME_TSV}")

    print("\nPer-genus reference balance:")
    print(out.to_string(index=False))

    print("\nMost overrepresented genera by fragment fraction:")
    print(
        out[["genus", "n_train_genomes", "n_reference_fragments", "fraction_of_reference_fragments"]]
        .sort_values("fraction_of_reference_fragments", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nGenomes with most fragments:")
    print(
        per_genome.sort_values("n_fragments", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
