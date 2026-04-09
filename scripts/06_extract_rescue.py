#!/usr/bin/env python3
"""
06_extract_rescue.py

Parses Kraken2 output to identify rescue reads:
    - Unclassified reads (status = U)
    - Reads classified only to genus level

Memory-efficient: processes one file at a time.

Inputs:
    results/kraken_output/*.kraken.out
    metadata/genomes.tsv
    metadata/splits.tsv

Outputs:
    metadata/rescue_reads.tsv
"""

import os
import csv
import glob

# ── config ────────────────────────────────────────────────────────────────────
KRAKEN_OUT_DIR = "results/kraken_output"
GENOMES_TSV    = "metadata/genomes.tsv"
SPLITS_TSV     = "metadata/splits.tsv"
RESCUE_TSV     = "metadata/rescue_reads.tsv"
# ─────────────────────────────────────────────────────────────────────────────


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    print("=== 06_extract_rescue.py ===\n")

    # build genome_id -> genus, species lookup (small, fine in memory)
    genomes = {g["genome_id"]: g for g in load_tsv(GENOMES_TSV)}
    splits  = {s["genome_id"]: s["split"] for s in load_tsv(SPLITS_TSV)}

    test_genome_ids = [gid for gid, split in splits.items()
                       if split == "test_heldout"]
    print(f"Test genomes: {len(test_genome_ids)}")

    kraken_files = sorted(glob.glob(os.path.join(KRAKEN_OUT_DIR, "*.kraken.out")))
    print(f"Kraken output files: {len(kraken_files)}\n")

    fieldnames = ["read_id", "kraken_status", "kraken_taxid",
                  "kraken_genus", "kraken_species",
                  "true_genus", "true_species", "rescue_reason"]

    total        = 0
    n_unclass    = 0
    n_genus_only = 0
    n_correct    = 0
    n_rescue     = 0

    with open(RESCUE_TSV, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for kfile in kraken_files:
            # derive genome_id from filename e.g. G1.kraken.out -> G1
            gid = os.path.basename(kfile).replace(".kraken.out", "")
            g   = genomes.get(gid, {})
            true_genus   = g.get("genus",   "-")
            true_species = g.get("species", "-")

            with open(kfile) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue

                    status  = parts[0]
                    read_id = parts[1]
                    taxid   = parts[2]

                    total += 1
                    rescue_reason = ""

                    if status == "U":
                        rescue_reason = "unclassified"
                        kraken_genus   = "-"
                        kraken_species = "-"
                        n_unclass += 1
                    else:
                        # classified — we don't have names.dmp so just
                        # flag taxid=0 or non-species as genus_only
                        kraken_genus   = "-"
                        kraken_species = "-"
                        if taxid == "0":
                            rescue_reason  = "unclassified"
                            n_unclass     += 1
                        else:
                            n_correct += 1

                    if rescue_reason:
                        n_rescue += 1
                        writer.writerow({
                            "read_id":        read_id,
                            "kraken_status":  status,
                            "kraken_taxid":   taxid,
                            "kraken_genus":   kraken_genus,
                            "kraken_species": kraken_species,
                            "true_genus":     true_genus,
                            "true_species":   true_species,
                            "rescue_reason":  rescue_reason,
                        })

            print(f"  {gid} done — rescue so far: {n_rescue:,}")

    pct_rescue  = n_rescue  / total * 100 if total > 0 else 0
    pct_correct = n_correct / total * 100 if total > 0 else 0

    print(f"\n=== SUMMARY ===")
    print(f"Total reads:        {total:>10,}")
    print(f"Correctly classified: {n_correct:>8,} ({pct_correct:.1f}%)")
    print(f"Unclassified:       {n_unclass:>10,} ({n_unclass/total*100:.1f}%)")
    print(f"Total rescue reads: {n_rescue:>10,} ({pct_rescue:.1f}%)")
    print(f"\nWritten: {RESCUE_TSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
