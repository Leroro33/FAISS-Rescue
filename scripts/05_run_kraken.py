#!/usr/bin/env python3
"""
05_run_kraken.py

Runs Kraken2 on all simulated test reads.
Produces classification output, reports, and
classified/unclassified FASTQ files.

Inputs:
    metadata/splits.tsv
    metadata/genomes.tsv
    data/simulated_reads/<genome_id>.fastq
    data/kraken_db/

Outputs:
    results/kraken_output/<genome_id>.kraken.out
    results/kraken_output/<genome_id>.report
    results/kraken_output/<genome_id>_classified.fq
    results/kraken_output/<genome_id>_unclassified.fq
"""

import os
import subprocess
import csv

# ── config ────────────────────────────────────────────────────────────────────
GENOMES_TSV   = "metadata/genomes.tsv"
SPLITS_TSV    = "metadata/splits.tsv"
READS_DIR     = "data/simulated_reads"
KRAKEN_DB     = "data/kraken_db"
KRAKEN_OUT    = "results/kraken_output"
THREADS       = 4
# ─────────────────────────────────────────────────────────────────────────────


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    print("=== 05_run_kraken.py ===\n")

    os.makedirs(KRAKEN_OUT, exist_ok=True)

    genomes = {g["genome_id"]: g for g in load_tsv(GENOMES_TSV)}
    splits  = {s["genome_id"]: s["split"] for s in load_tsv(SPLITS_TSV)}

    test_genomes = [
        genomes[gid] for gid, split in splits.items()
        if split == "test_heldout"
    ]
    print(f"Test genomes to classify: {len(test_genomes)}\n")

    total_reads       = 0
    total_classified  = 0
    total_unclassified = 0

    for i, g in enumerate(test_genomes):
        gid     = g["genome_id"]
        genus   = g["genus"]
        species = g["species"]
        fastq   = os.path.join(READS_DIR, f"{gid}.fastq")

        if not os.path.exists(fastq):
            print(f"  SKIP {gid} — fastq not found: {fastq}")
            continue

        print(f"[{i+1}/{len(test_genomes)}] {gid} — {genus} {species}")

        out_kraken       = os.path.join(KRAKEN_OUT, f"{gid}.kraken.out")
        out_report       = os.path.join(KRAKEN_OUT, f"{gid}.report")
        out_classified   = os.path.join(KRAKEN_OUT, f"{gid}_classified.fq")
        out_unclassified = os.path.join(KRAKEN_OUT, f"{gid}_unclassified.fq")

        cmd = [
            "kraken2",
            "--db",                KRAKEN_DB,
            "--output",            out_kraken,
            "--report",            out_report,
            "--classified-out",    out_classified,
            "--unclassified-out",  out_unclassified,
            "--threads",           str(THREADS),
            fastq
        ]
        subprocess.run(cmd, check=True)

        # count classified vs unclassified
        classified   = sum(1 for l in open(out_kraken) if l.startswith("C"))
        unclassified = sum(1 for l in open(out_kraken) if l.startswith("U"))
        total         = classified + unclassified
        pct           = classified / total * 100 if total > 0 else 0

        print(f"    Classified:   {classified:>8,} ({pct:.1f}%)")
        print(f"    Unclassified: {unclassified:>8,} ({100-pct:.1f}%)")

        total_reads        += total
        total_classified   += classified
        total_unclassified += unclassified

    # summary
    pct_total = total_classified / total_reads * 100 if total_reads > 0 else 0
    print(f"\n=== SUMMARY ===")
    print(f"Total reads:        {total_reads:>10,}")
    print(f"Classified:         {total_classified:>10,} ({pct_total:.1f}%)")
    print(f"Unclassified:       {total_unclassified:>10,} ({100-pct_total:.1f}%)")
    print("\nDone.")


if __name__ == "__main__":
    main()
