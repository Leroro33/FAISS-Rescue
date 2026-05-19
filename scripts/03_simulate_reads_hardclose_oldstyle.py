#!/usr/bin/env python3
import os
import re
import gzip
import shutil
import subprocess
import csv
import pandas as pd

GENOMES_TSV    = "metadata_hardclose/genomes.tsv"
SPLITS_TSV     = "metadata_hardclose/splits.tsv"
READS_DIR      = "data_hardclose_oldstyle/simulated_reads"
READ_TRUTH_TSV = "metadata_hardclose_oldstyle/read_truth.tsv"
READ_LENGTH    = 150
FOLD_COVERAGE  = 10
RANDOM_SEED    = 42
ERROR_PROFILE  = "clean"

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

def rewrite_fastq_and_collect_truth(in_fastq, out_fastq, genome_id, genus, species, read_length, error_profile):
    rows = []
    read_counter = 1

    with open(in_fastq) as fin, open(out_fastq, "w") as fout:
        while True:
            header = fin.readline().strip()
            if not header:
                break
            seq  = fin.readline().strip()
            plus = fin.readline().strip()
            qual = fin.readline().strip()

            start = None
            match = re.search(r"-(\d+)(?:/\d+)?$", header)
            if match:
                start = int(match.group(1))
            end = (start + read_length - 1) if start is not None else None

            read_id = f"{genome_id}_read_{read_counter}"

            fout.write(f"@{read_id}\n")
            fout.write(seq + "\n")
            fout.write(plus + "\n")
            fout.write(qual + "\n")

            rows.append({
                "read_id":          read_id,
                "source_genome_id": genome_id,
                "true_genus":       genus,
                "true_species":     species,
                "start":            start if start is not None else "NA",
                "end":              end   if end   is not None else "NA",
                "error_profile":    error_profile,
            })
            read_counter += 1

    return rows

def main():
    print("=== 03_simulate_reads_hardclose_oldstyle.py ===")
    os.makedirs(READS_DIR, exist_ok=True)
    os.makedirs("metadata_hardclose_oldstyle", exist_ok=True)

    genomes = pd.read_csv(GENOMES_TSV, sep="\t")
    splits  = pd.read_csv(SPLITS_TSV, sep="\t")
    test_ids = set(splits.loc[splits["split"] == "test_genome", "genome_id"])
    test_df = genomes[genomes["genome_id"].isin(test_ids)].copy()

    all_truth_rows = []
    tmp_fasta = os.path.join(READS_DIR, "_tmp.fna")

    for i, (_, g) in enumerate(test_df.iterrows()):
        gid      = g["genome_id"]
        fasta_gz = g["fasta_path"]
        genus    = g["genus"]
        species  = g["species"]
        prefix   = os.path.join(READS_DIR, gid)
        final    = prefix + ".fastq"

        print(f"[{i+1}/{len(test_df)}] {gid} — {genus} {species}")

        decompress_fasta(fasta_gz, tmp_fasta)
        run_art(tmp_fasta, prefix, READ_LENGTH, FOLD_COVERAGE, RANDOM_SEED + i)

        art_out = prefix + ".fq"
        rows = rewrite_fastq_and_collect_truth(
            art_out, final, gid, genus, species, READ_LENGTH, ERROR_PROFILE
        )
        if os.path.exists(art_out):
            os.remove(art_out)

        all_truth_rows.extend(rows)
        print(f"    Generated {len(rows)} reads → {final}")

    if os.path.exists(tmp_fasta):
        os.remove(tmp_fasta)

    fieldnames = ["read_id", "source_genome_id", "true_genus", "true_species", "start", "end", "error_profile"]
    with open(READ_TRUTH_TSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_truth_rows)

    print(f"\nWritten: {READ_TRUTH_TSV} ({len(all_truth_rows)} total reads)")

if __name__ == "__main__":
    main()
