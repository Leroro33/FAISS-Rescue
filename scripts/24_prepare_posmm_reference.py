#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import gzip
import shutil

ROOT = Path(__file__).resolve().parents[1]

GENOMES_TSV = ROOT / "metadata" / "genomes.tsv"
SPLITS_TSV = ROOT / "metadata" / "splits.tsv"
OUT_DIR = ROOT / "posmm_ref_genomes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES_TO_TAXID = {
    ("Escherichia", "coli"): 562,
    ("Staphylococcus", "aureus"): 1280,
    ("Klebsiella", "pneumoniae"): 573,
    ("Salmonella", "enterica"): 28901,
    ("Listeria", "monocytogenes"): 1639,
    ("Streptococcus", "pneumoniae"): 1313,
    ("Streptococcus", "pyogenes"): 1314,
    ("Streptococcus", "agalactiae"): 1311,
    ("Mycobacterium", "tuberculosis"): 1773,
    ("Pseudomonas", "aeruginosa"): 287,
    ("Acinetobacter", "baumannii"): 470,
}

def pick_fasta_path(row):
    for col in ["fasta_path", "genome_path", "path", "file_path"]:
        if col in row and pd.notna(row[col]):
            return Path(str(row[col]))
    return None

def main():
    print("=== 24_prepare_posmm_reference.py ===")
    genomes = pd.read_csv(GENOMES_TSV, sep="\t")
    splits = pd.read_csv(SPLITS_TSV, sep="\t")

    train_ids = set(splits.loc[splits["split"] == "train_ref", "genome_id"])
    train = genomes[genomes["genome_id"].isin(train_ids)].copy()

    written = 0
    skipped = 0

    print(f"train_ref genomes: {len(train)}")

    for _, row in train.iterrows():
        genus = str(row["genus"])
        species = str(row["species"])
        genome_id = str(row["genome_id"])

        fasta_path = pick_fasta_path(row)
        key = (genus, species)
        taxid = SPECIES_TO_TAXID.get(key)

        if taxid is None:
            print(f"SKIP no taxid mapping for {genus} {species} ({genome_id})")
            skipped += 1
            continue

        if fasta_path is None or not fasta_path.exists():
            print(f"SKIP missing fasta path for {genome_id}: {fasta_path}")
            skipped += 1
            continue

        stem = fasta_path.name
        if stem.endswith(".gz"):
            stem = stem[:-3]

        if not stem.endswith(".fna"):
            stem = stem + ".fna"

        out_name = f"{taxid}.{genome_id}.{stem}"
        out_path = OUT_DIR / out_name

        if str(fasta_path).endswith(".gz"):
            with gzip.open(fasta_path, "rb") as fin, open(out_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            shutil.copyfile(fasta_path, out_path)

        written += 1

    print(f"Written genomes: {written}")
    print(f"Skipped genomes: {skipped}")
    print(f"Output dir: {OUT_DIR}")

if __name__ == "__main__":
    main()
