#!/usr/bin/env python3
"""
01b_add_family_taxonomy.py

Adds a 'family' column to metadata/genomes.tsv mapping each genus to its
taxonomic family (NCBI). Used by Path C (hierarchical classifier).

The hierarchical classifier first routes a read to a family, then runs
a within-family genus prediction. Salmonella, E. coli, and Klebsiella
all live in Enterobacteriaceae — within that family they have equal
data weight (no class-imbalance bias toward majority neighbors).
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.home() / "CompressedRescue"
GENOMES_TSV  = PROJECT_ROOT / "metadata" / "genomes.tsv"

GENUS_TO_FAMILY = {
    "Acinetobacter":  "Moraxellaceae",
    "Escherichia":    "Enterobacteriaceae",
    "Klebsiella":     "Enterobacteriaceae",
    "Listeria":       "Listeriaceae",
    "Mycobacterium":  "Mycobacteriaceae",
    "Pseudomonas":    "Pseudomonadaceae",
    "Salmonella":     "Enterobacteriaceae",
    "Staphylococcus": "Staphylococcaceae",
    "Streptococcus":  "Streptococcaceae",
}


def main():
    df = pd.read_csv(GENOMES_TSV, sep="\t")
    print(f"Loaded {len(df)} genomes")

    if "family" in df.columns:
        print("ABORT: 'family' column already exists. Refusing to overwrite.")
        sys.exit(1)

    df["family"] = df["genus"].map(GENUS_TO_FAMILY)

    missing = df[df["family"].isna()]
    if len(missing) > 0:
        print(f"ABORT: {len(missing)} rows have a genus not in GENUS_TO_FAMILY:")
        print(missing[["genome_id", "genus"]].to_string(index=False))
        sys.exit(1)

    df.to_csv(GENOMES_TSV, sep="\t", index=False)
    print(f"Wrote {GENOMES_TSV}")

    print("\nFamily distribution (number of genomes per family):")
    print(df["family"].value_counts().to_string())

    print("\nGenera per family:")
    for family, group in df.groupby("family"):
        genera = sorted(group["genus"].unique())
        print(f"  {family}: {', '.join(genera)}")


if __name__ == "__main__":
    main()
