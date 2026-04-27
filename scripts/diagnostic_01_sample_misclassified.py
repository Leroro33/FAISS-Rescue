#!/usr/bin/env python3
"""
diagnostic_01_sample_misclassified.py

STEP 1 of the Salmonella misclassification diagnostic.

WHAT IT DOES
------------
1. Loads your rescue predictions (results/rescue_predictions_opq.tsv)
2. Joins with ground truth (metadata/rescue_reads.tsv)
3. Filters to reads where:
     - true genus  == Salmonella
     - predicted genus != Salmonella
4. Samples N of these misclassified reads
5. Pulls their sequences out of the original FASTQ files
6. Writes them to a FASTA file ready for mapping in step 2

WHY
---
We want to know WHERE in the Salmonella genome these reads came from.
If most come from rRNA / tRNA / ribosomal-protein genes, the problem is
biological (E. coli and Salmonella are nearly identical there) and no
embedding will fix it. If most come from elsewhere, the embedding
itself is failing and embedding optimisation is the right next step.

HOW TO RUN
----------
    conda activate bioenv
    cd ~/CompressedRescue
    python scripts/diagnostic_01_sample_misclassified.py

OUTPUTS
-------
    diagnostic/misclassified_salmonella.fasta
    diagnostic/misclassified_salmonella_metadata.tsv
"""

import sys
import gzip
from pathlib import Path

import pandas as pd
from Bio import SeqIO

# ─── Configuration ────────────────────────────────────────────────────────────
# Adjust if your project lives somewhere else.
PROJECT_ROOT     = Path.home() / "CompressedRescue"
PREDICTIONS_TSV  = PROJECT_ROOT / "results"  / "rescue_predictions_opq.tsv"
RESCUE_READS_TSV = PROJECT_ROOT / "metadata" / "rescue_reads.tsv"
READ_TRUTH_TSV   = PROJECT_ROOT / "metadata" / "read_truth.tsv"
FASTQ_DIR        = PROJECT_ROOT / "data"     / "simulated_reads"

OUTPUT_DIR    = PROJECT_ROOT / "diagnostic"
OUTPUT_FASTA  = OUTPUT_DIR / "misclassified_salmonella.fasta"
OUTPUT_META   = OUTPUT_DIR / "misclassified_salmonella_metadata.tsv"

TARGET_GENUS  = "Salmonella"
N_SAMPLE      = 1000
RANDOM_SEED   = 42
# ──────────────────────────────────────────────────────────────────────────────


def find_column(df, candidates, label):
    """Return the first matching column name, or exit with a clear error."""
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(
        f"ERROR: could not find {label} column. "
        f"Tried {candidates}. Available columns: {list(df.columns)}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load predictions ──────────────────────────────────────────────────
    print(f"[1/5] Loading predictions: {PREDICTIONS_TSV}")
    if not PREDICTIONS_TSV.exists():
        sys.exit(f"ERROR: predictions not found: {PREDICTIONS_TSV}")
    pred = pd.read_csv(PREDICTIONS_TSV, sep="\t")
    print(f"      {len(pred):,} prediction rows; columns: {list(pred.columns)}")

    pred_genus_col = find_column(
        pred, ["predicted_genus", "pred_genus"], "predicted genus"
    )

    # ── 2. Load truth ────────────────────────────────────────────────────────
    truth_path = RESCUE_READS_TSV if RESCUE_READS_TSV.exists() else READ_TRUTH_TSV
    print(f"\n[2/5] Loading truth: {truth_path}")
    truth = pd.read_csv(truth_path, sep="\t")
    print(f"      {len(truth):,} truth rows; columns: {list(truth.columns)}")

    truth_genus_col = find_column(
        truth, ["true_genus", "truth_genus", "genus"], "true genus"
    )

    # ── 3. Join + filter ─────────────────────────────────────────────────────
    print(f"\n[3/5] Joining predictions with truth on read_id...")
    keep_cols = ["read_id", truth_genus_col]
    if "true_species" in truth.columns:
        keep_cols.append("true_species")
    if "true_genome" in truth.columns:
        keep_cols.append("true_genome")
    merged = pred.merge(truth[keep_cols], on="read_id", how="inner")
    print(f"      Joined: {len(merged):,} rows")

    is_salm  = merged[truth_genus_col] == TARGET_GENUS
    is_wrong = merged[pred_genus_col] != TARGET_GENUS
    miscls   = merged[is_salm & is_wrong].copy()

    n_salm = int(is_salm.sum())
    print(f"\n      Total Salmonella reads in rescue:  {n_salm:,}")
    print(f"      Misclassified (not Salmonella):    {len(miscls):,}")
    if n_salm:
        print(f"      Misclassification rate:            "
              f"{len(miscls)/n_salm:.1%}")

    if not len(miscls):
        sys.exit("ERROR: no misclassified Salmonella reads found.")

    # Sample
    n_take = min(N_SAMPLE, len(miscls))
    sample = miscls.sample(n=n_take, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"\n      Sampled {n_take:,} reads")

    print(f"\n      Misclassified AS:")
    for genus, count in sample[pred_genus_col].value_counts().head(10).items():
        print(f"        {genus:<25} {count:>5}")

    sample.to_csv(OUTPUT_META, sep="\t", index=False)
    print(f"\n      Wrote metadata: {OUTPUT_META}")

    # ── 4. Extract sequences from FASTQ files ────────────────────────────────
    print(f"\n[4/5] Extracting sequences from FASTQ files in {FASTQ_DIR}")
    if not FASTQ_DIR.exists():
        sys.exit(f"ERROR: FASTQ dir not found: {FASTQ_DIR}")

    # Build set of needed read IDs (handle /1 /2 paired-end suffixes)
    needed = set()
    for rid in sample["read_id"].astype(str):
        needed.add(rid)
        needed.add(rid.split("/")[0])  # base ID without mate suffix

    print(f"      Looking for {len(sample):,} unique read IDs")

    fastq_files = sorted(
        list(FASTQ_DIR.rglob("*.fastq")) +
        list(FASTQ_DIR.rglob("*.fq")) +
        list(FASTQ_DIR.rglob("*.fastq.gz")) +
        list(FASTQ_DIR.rglob("*.fq.gz"))
    )
    if not fastq_files:
        sys.exit(f"ERROR: no FASTQ files found under {FASTQ_DIR}")
    print(f"      Found {len(fastq_files)} FASTQ file(s)")

    found = {}  # read_id -> sequence
    for i, fq in enumerate(fastq_files, 1):
        if len(found) >= len(sample):
            break
        opener = gzip.open if fq.suffix == ".gz" else open
        try:
            with opener(fq, "rt") as h:
                for rec in SeqIO.parse(h, "fastq"):
                    if rec.id in needed or rec.id.split("/")[0] in needed:
                        found[rec.id] = str(rec.seq)
        except Exception as e:
            print(f"      WARNING: could not parse {fq.name}: {e}")
            continue

        if i % 10 == 0 or i == len(fastq_files):
            print(f"      Scanned {i}/{len(fastq_files)} files; "
                  f"recovered {len(found):,}/{len(sample):,}")

    print(f"\n      Recovered {len(found):,} of {len(sample):,} sequences")
    if len(found) < 0.5 * len(sample):
        print("      WARNING: <50% recovered. Check that read IDs in your "
              "metadata match the IDs in the FASTQ files.")

    # ── 5. Write FASTA ───────────────────────────────────────────────────────
    print(f"\n[5/5] Writing FASTA: {OUTPUT_FASTA}")
    with open(OUTPUT_FASTA, "w") as out:
        for rid, seq in found.items():
            out.write(f">{rid}\n{seq}\n")
    print(f"      Wrote {len(found):,} sequences")

    print("\n=== DONE ===")
    print("Next step: bash diagnostic_02_run_mapping.sh")


if __name__ == "__main__":
    main()
