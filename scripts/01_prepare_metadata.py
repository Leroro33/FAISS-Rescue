#!/usr/bin/env python3
"""
01_prepare_metadata.py

Scans data/raw_genomes/ for assembly report files, parses genome metadata,
assigns genome IDs, and writes:
  - metadata/genomes.tsv
  - metadata/splits.tsv
"""

import os
import re
import random
import csv
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
RAW_GENOMES_DIR = "data/raw_genomes"
GENOMES_TSV     = "metadata/genomes.tsv"
SPLITS_TSV      = "metadata/splits.tsv"
TRAIN_RATIO     = 0.75   # 75% train, 25% test
RANDOM_SEED     = 42     # makes the split reproducible
# ─────────────────────────────────────────────────────────────────────────────


def parse_assembly_report(report_path):
    """
    Read a *_assembly_report.txt and extract key metadata fields.
    Returns a dict with genus, species, accession, assembly_level.
    """
    data = {
        "organism": None,
        "accession": None,
        "assembly_level": None,
    }

    with open(report_path, "r") as f:
        for line in f:
            if not line.startswith("#"):
                break  # header section is over
            if "Organism name:" in line:
                data["organism"] = line.split("Organism name:")[-1].strip()
            elif "Assembly level:" in line:
                data["assembly_level"] = line.split("Assembly level:")[-1].strip()
            elif line.strip().startswith("# Assembly name:"):
                pass  # skip, not needed
            # accession is in the filename itself, parsed separately

    return data


def extract_accession(filename):
    """
    Pull the GCF accession from a filename like:
    GCF_000005845.2_ASM584v2_assembly_report.txt
    """
    match = re.match(r"(GCF_\d+\.\d+)", filename)
    return match.group(1) if match else None


def parse_organism(organism_str):
    """
    Split 'Escherichia coli K-12' into genus='Escherichia', species='coli'.
    """
    if not organism_str:
        return "Unknown", "Unknown"
    parts = organism_str.strip().split()
    genus   = parts[0] if len(parts) >= 1 else "Unknown"
    species = parts[1] if len(parts) >= 2 else "Unknown"
    return genus, species


def find_fasta_path(accession, raw_dir):
    """
    Find the .fna.gz file matching this accession in raw_dir.
    """
    for fname in os.listdir(raw_dir):
        if fname.startswith(accession) and fname.endswith(".fna.gz"):
            return os.path.join(raw_dir, fname)
    return None


def make_splits(genomes, train_ratio, seed):
    """
    Assign train_ref / test_heldout per species.
    Guarantees every species has at least one test genome.
    Returns a dict: genome_id -> split label
    """
    random.seed(seed)

    # group genome_ids by species
    by_species = defaultdict(list)
    for g in genomes:
        key = (g["genus"], g["species"])
        by_species[key].append(g["genome_id"])

    splits = {}
    for (genus, species), ids in by_species.items():
        random.shuffle(ids)
        n = len(ids)

        if n == 1:
            # only one genome — put it in train, flag it
            splits[ids[0]] = "train_ref"
            print(f"  WARNING: {genus} {species} has only 1 genome, assigned to train_ref")
        elif n == 2:
            # one train, one test
            splits[ids[0]] = "train_ref"
            splits[ids[1]] = "test_heldout"
        else:
            n_train = max(1, round(n * train_ratio))
            n_test  = n - n_train
            # guarantee at least 1 test
            if n_test < 1:
                n_train -= 1
                n_test   = 1
            for i, gid in enumerate(ids):
                splits[gid] = "train_ref" if i < n_train else "test_heldout"

    return splits


def main():
    print("=== 01_prepare_metadata.py ===\n")

    # ── collect all assembly reports ─────────────────────────────────────────
    report_files = [
        f for f in os.listdir(RAW_GENOMES_DIR)
        if f.endswith("_assembly_report.txt")
    ]
    print(f"Found {len(report_files)} assembly reports\n")

    genomes = []
    skipped = 0

    for i, report_fname in enumerate(sorted(report_files)):
        report_path = os.path.join(RAW_GENOMES_DIR, report_fname)
        accession   = extract_accession(report_fname)

        if not accession:
            print(f"  SKIP (no accession): {report_fname}")
            skipped += 1
            continue

        meta          = parse_assembly_report(report_path)
        genus, species = parse_organism(meta["organism"])
        fasta_path    = find_fasta_path(accession, RAW_GENOMES_DIR)

        if not fasta_path:
            print(f"  SKIP (no fasta found): {accession}")
            skipped += 1
            continue

        genomes.append({
            "genome_id":      f"G{i+1}",
            "genus":          genus,
            "species":        species,
            "accession":      accession,
            "assembly_level": meta["assembly_level"] or "Unknown",
            "fasta_path":     fasta_path,
        })

    print(f"Parsed:   {len(genomes)} genomes")
    print(f"Skipped:  {skipped} genomes\n")

    # ── write genomes.tsv ─────────────────────────────────────────────────────
    os.makedirs("metadata", exist_ok=True)
    fieldnames = ["genome_id", "genus", "species", "accession",
                  "assembly_level", "fasta_path"]

    with open(GENOMES_TSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(genomes)

    print(f"Written:  {GENOMES_TSV}  ({len(genomes)} rows)\n")

    # ── make splits ───────────────────────────────────────────────────────────
    print("Generating train/test splits...")
    splits = make_splits(genomes, TRAIN_RATIO, RANDOM_SEED)

    split_rows = []
    for g in genomes:
        split_rows.append({
            "genome_id": g["genome_id"],
            "genus":     g["genus"],
            "species":   g["species"],
            "split":     splits[g["genome_id"]],
        })

    with open(SPLITS_TSV, "w", newline="") as f:
        writer = csv.DictWriter(f,
                    fieldnames=["genome_id", "genus", "species", "split"],
                    delimiter="\t")
        writer.writeheader()
        writer.writerows(split_rows)

    # ── summary ───────────────────────────────────────────────────────────────
    n_train = sum(1 for r in split_rows if r["split"] == "train_ref")
    n_test  = sum(1 for r in split_rows if r["split"] == "test_heldout")

    print(f"Written:  {SPLITS_TSV}")
    print(f"  train_ref:    {n_train}")
    print(f"  test_heldout: {n_test}")
    print("\nDone.")


if __name__ == "__main__":
    main()
