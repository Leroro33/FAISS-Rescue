#!/usr/bin/env python3
"""
04_build_kraken_db.py

Builds a Kraken2 database from train_ref genomes.

Steps:
    1. Download NCBI taxonomy
    2. Add train_ref genome FASTAs to library
    3. Build the database

Inputs:
    metadata/genomes.tsv
    metadata/splits.tsv
    data/raw_genomes/*.fna.gz

Outputs:
    data/kraken_db/  (Kraken2 database files)
"""

import os
import gzip
import shutil
import subprocess
import csv

# ── config ────────────────────────────────────────────────────────────────────
GENOMES_TSV  = "metadata/genomes.tsv"
SPLITS_TSV   = "metadata/splits.tsv"
KRAKEN_DB    = "data/kraken_db"
THREADS      = 8
# ─────────────────────────────────────────────────────────────────────────────


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run(cmd, desc):
    print(f"\n>>> {desc}")
    print(f"    {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    print("=== 04_build_kraken_db.py ===\n")

    os.makedirs(KRAKEN_DB, exist_ok=True)

    # ── get train_ref genomes ─────────────────────────────────────────────────
    genomes = {g["genome_id"]: g for g in load_tsv(GENOMES_TSV)}
    splits  = {s["genome_id"]: s["split"] for s in load_tsv(SPLITS_TSV)}

    train_genomes = [
        genomes[gid] for gid, split in splits.items()
        if split == "train_ref"
    ]
    print(f"Train reference genomes: {len(train_genomes)}")

    # ── step 1: download taxonomy ─────────────────────────────────────────────
    run(
        ["kraken2-build", "--download-taxonomy", "--db", KRAKEN_DB, "--use-ftp"],
        "Downloading NCBI taxonomy (via FTP)..."
    )

    # ── step 2: add genomes to library ────────────────────────────────────────
    print(f"\n>>> Adding {len(train_genomes)} genomes to Kraken2 library...")

    tmp_dir = os.path.join(KRAKEN_DB, "tmp_fasta")
    os.makedirs(tmp_dir, exist_ok=True)

    for i, g in enumerate(train_genomes):
        gid      = g["genome_id"]
        fasta_gz = g["fasta_path"]
        genus    = g["genus"]
        species  = g["species"]

        # decompress to tmp
        tmp_fna = os.path.join(tmp_dir, f"{gid}.fna")
        with gzip.open(fasta_gz, "rb") as f_in:
            with open(tmp_fna, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # add to library
        subprocess.run(
            ["kraken2-build", "--add-to-library", tmp_fna, "--db", KRAKEN_DB],
            check=True, capture_output=True
        )

        print(f"  [{i+1}/{len(train_genomes)}] Added {gid} — {genus} {species}")

    # cleanup tmp fastas
    shutil.rmtree(tmp_dir)
    print("  Temp files cleaned up")

    # ── step 3: build database ────────────────────────────────────────────────
    run(
        ["kraken2-build", "--build", "--db", KRAKEN_DB, "--threads", str(THREADS), "--max-db-size", "8000000000"],
        "Building Kraken2 database (this may take 30-60 min)..."
    )

    #��─ cleanup taxonomy folder ───────────────────────────────────────────────
    print("\nCleaning up taxonomy folder (not needed after build)...")
    shutil.rmtree(os.path.join(KRAKEN_DB, "taxonomy"), ignore_errors=True)
    print("Taxonomy folder removed.")

    # ── report size ───────────────────────────────────────────────────────────
    result = subprocess.run(
        ["du", "-sh", KRAKEN_DB],
        capture_output=True, text=True
    )
    print(f"\nDatabase size: {result.stdout.strip()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
