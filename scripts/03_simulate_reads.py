#!/usr/bin/env python3
"""
03_simulate_reads.py

Simulates 150bp single-end Illumina reads from test_heldout genomes
using ART. Generates FASTQ files and read_truth.tsv ground truth.

Inputs:
    metadata/splits.tsv
    metadata/genomes.tsv
    data/raw_genomes/*.fna.gz

Outputs:
    data/simulated_reads/<genome_id>.fastq
    metadata/read_truth.tsv
"""

import os
import re
import gzip
import shutil
import subprocess
import csv

# ── config ────────────────────────────────────────────────────────────────────
GENOMES_TSV    = "metadata/genomes.tsv"
SPLITS_TSV     = "metadata/splits.tsv"
READS_DIR      = "data/simulated_reads"
READ_TRUTH_TSV = "metadata/read_truth.tsv"
READ_LENGTH    = 150
FOLD_COVERAGE  = 10
RANDOM_SEED    = 42
ERROR_PROFILE  = "clean"   # label for truth table
# ─────────────────────────────────────────────────────────────────────────────


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def decompress_fasta(fna_gz_path, out_path):
    with gzip.open(fna_gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def run_art(fasta_path, output_prefix, read_length, coverage, seed):
    cmd = [
        "art_illumina",
        "--seqSys",  "HS25",
        "--in",      fasta_path,
        "--out",     output_prefix,
        "--len",     str(read_length),
        "--fcov",    str(coverage),
        "--rndSeed", str(seed),
        "--noALN",
        "--quiet",
    ]
    subprocess.run(cmd, check=True)


def parse_art_reads(fastq_path, genome_id, genus, species, read_length, error_profile):
    """
    Parse ART fastq output.
    ART read IDs look like: @genome_id-<start_pos>/1
    We extract start and compute end = start + read_length - 1
    """
    rows = []
    read_counter = 1
    with open(fastq_path) as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            f.readline()   # sequence
            f.readline()   # +
            f.readline()   # quality

            # parse start position from ART header
            # ART format: @ref_name-<start>/1
            start = None
            match = re.search(r"-(\d+)(?:/\d+)?$", header)
            if match:
                start = int(match.group(1))
            end = (start + read_length - 1) if start is not None else None

            read_id = f"{genome_id}_read_{read_counter}"
            rows.append({
                "read_id":          read_id,
                "source_genome_id": genome_id,
                "genus_true":       genus,
                "species_true":     species,
                "start":            start if start is not None else "NA",
                "end":              end   if end   is not None else "NA",
                "error_profile":    error_profile,
            })
            read_counter += 1
    return rows


def main():
    print("=== 03_simulate_reads.py ===\n")

    os.makedirs(READS_DIR, exist_ok=True)

    genomes = {g["genome_id"]: g for g in load_tsv(GENOMES_TSV)}
    splits  = {s["genome_id"]: s["split"] for s in load_tsv(SPLITS_TSV)}

    test_genomes = [
        genomes[gid] for gid, split in splits.items()
        if split == "test_heldout"
    ]
    print(f"Test genomes to simulate: {len(test_genomes)}\n")

    all_truth_rows = []
    tmp_fasta      = os.path.join(READS_DIR, "_tmp.fna")

    for i, g in enumerate(test_genomes):
        gid      = g["genome_id"]
        fasta_gz = g["fasta_path"]
        genus    = g["genus"]
        species  = g["species"]
        prefix   = os.path.join(READS_DIR, gid)
        final    = prefix + ".fastq"

        print(f"[{i+1}/{len(test_genomes)}] {gid} — {genus} {species}")

        decompress_fasta(fasta_gz, tmp_fasta)
        run_art(tmp_fasta, prefix, READ_LENGTH, FOLD_COVERAGE, RANDOM_SEED + i)

        art_out = prefix + ".fq"
        if os.path.exists(art_out):
            os.rename(art_out, final)

        rows = parse_art_reads(final, gid, genus, species, READ_LENGTH, ERROR_PROFILE)
        all_truth_rows.extend(rows)
        print(f"    Generated {len(rows)} reads → {final}")

    if os.path.exists(tmp_fasta):
        os.remove(tmp_fasta)

    fieldnames = ["read_id", "source_genome_id", "genus_true",
                  "species_true", "start", "end", "error_profile"]
    with open(READ_TRUTH_TSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_truth_rows)

    print(f"\nWritten: {READ_TRUTH_TSV} ({len(all_truth_rows)} total reads)")
    print("\nDone.")


if __name__ == "__main__":
    main()
