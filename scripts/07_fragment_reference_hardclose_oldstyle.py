#!/usr/bin/env python3
import os
import gzip
import csv
import pandas as pd

GENOMES_TSV    = "metadata_hardclose/genomes.tsv"
SPLITS_TSV     = "metadata_hardclose/splits.tsv"
FRAGMENTS_DIR  = "data_hardclose_oldstyle/reference_fragments"
FRAGMENTS_TSV  = "metadata_hardclose_oldstyle/fragments.tsv"
WINDOW         = 250
STRIDE         = 125

def read_fasta_gz(path):
    sequences = []
    current_header = None
    current_seq = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            sequences.append((current_header, "".join(current_seq)))
    return sequences

def fragment_sequence(seq, window, stride):
    fragments = []
    i = 0
    while i + window <= len(seq):
        frag = seq[i:i + window]
        fragments.append((i + 1, i + window, frag))
        i += stride
    return fragments

def main():
    print("=== 07_fragment_reference_hardclose_oldstyle.py ===")
    os.makedirs(FRAGMENTS_DIR, exist_ok=True)
    os.makedirs("metadata_hardclose_oldstyle", exist_ok=True)

    genomes = pd.read_csv(GENOMES_TSV, sep="\t")
    splits  = pd.read_csv(SPLITS_TSV, sep="\t")
    train_ids = set(splits.loc[splits["split"] == "train_ref", "genome_id"])
    train_df = genomes[genomes["genome_id"].isin(train_ids)].copy()

    fieldnames = ["fragment_id", "genome_id", "genus", "species", "start", "end", "length"]
    total_fragments = 0

    with open(FRAGMENTS_TSV, "w", newline="") as tsv_f:
        writer = csv.DictWriter(tsv_f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for i, (_, g) in enumerate(train_df.iterrows()):
            gid     = g["genome_id"]
            genus   = g["genus"]
            species = g["species"]
            fasta_gz = g["fasta_path"]

            print(f"[{i+1}/{len(train_df)}] {gid} — {genus} {species}")
            sequences = read_fasta_gz(fasta_gz)

            out_fasta = os.path.join(FRAGMENTS_DIR, f"{gid}.fasta")
            frag_count = 0

            with open(out_fasta, "w") as fasta_f:
                for contig_header, seq in sequences:
                    frags = fragment_sequence(seq, WINDOW, STRIDE)
                    for start, end, frag_seq in frags:
                        frag_count += 1
                        frag_id = f"{gid}_frag{frag_count:05d}"
                        fasta_f.write(f">{frag_id}|{gid}|{start}-{end}\n")
                        fasta_f.write(f"{frag_seq}\n")
                        writer.writerow({
                            "fragment_id": frag_id,
                            "genome_id": gid,
                            "genus": genus,
                            "species": species,
                            "start": start,
                            "end": end,
                            "length": len(frag_seq),
                        })

            total_fragments += frag_count
            print(f"    {frag_count:,} fragments → {out_fasta}")

    print(f"\nTotal fragments: {total_fragments:,}")
    print(f"Written: {FRAGMENTS_TSV}")

if __name__ == "__main__":
    main()
