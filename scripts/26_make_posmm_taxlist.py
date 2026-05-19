#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]

GENOMES_TSV = ROOT / "metadata" / "genomes.tsv"
SPLITS_TSV = ROOT / "metadata" / "splits.tsv"
OUT_TXT = ROOT / "posmm_train_taxlist.txt"

def extract_gcf(text: str):
    m = re.search(r"(GCF_\d+\.\d+)", text)
    return m.group(1) if m else None

def main():
    genomes = pd.read_csv(GENOMES_TSV, sep="\t")
    splits = pd.read_csv(SPLITS_TSV, sep="\t")

    train_ids = set(splits.loc[splits["split"] == "train_ref", "genome_id"])
    train = genomes[genomes["genome_id"].isin(train_ids)].copy()

    gcfs = []
    for _, row in train.iterrows():
        text = " ".join([str(x) for x in row.tolist()])
        gcf = extract_gcf(text)
        if gcf:
            gcfs.append(gcf)

    gcfs = sorted(set(gcfs))

    with open(OUT_TXT, "w") as f:
        for gcf in gcfs:
            f.write(gcf + "\n")

    print(f"Written: {OUT_TXT}")
    print(f"GCF accessions: {len(gcfs)}")

if __name__ == "__main__":
    main()
