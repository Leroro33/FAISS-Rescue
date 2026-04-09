#!/usr/bin/env python3
"""
07_fragment_reference.py

Slices all train_ref genomes into overlapping fragments.
Outputs fragment FASTA files and metadata/fragments.tsv.

Inputs:
    metadata/genomes.tsv
    metadata/splits.tsv
    data/raw_genomes/*.fna.gz

Outputs:
    data/reference_fragments/<genome_id>.fasta
    metadata/fragments.tsv
"""

import os
import gzip
import csv

# ── config ────────────────────────────────────────────────────────────────────
GENOMES_TSV    = "metadata/genomes.tsv"
SPLITS_TSV     = "metadata/splits.tsv"
FRAGMENTS_DIR  = "data/reference_fragments"
FRAGMENTS_TSV  = "metadata/fragments.tsv"
WINDOW         = 250    # fragment length in bp
STRIDE         = 125    # step size in bp
# ─────────────────────────────────────────────────────────────────────────────


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_fasta_gz(path):
    """
    Read a gzipped FASTA file.
    Returns list of (header, sequence) tuples.
    Concatenates all contigs into one sequence per genome.
    """
    sequences = []
    current_header = None
    current_seq    = []

    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq    = []
            else:
                current_seq.append(line)
        if current_header is not None:
            sequences.append((current_header, "".join(current_seq)))

    return sequences


def fragment_sequence(seq, window, stride):
    """
    Slide a window over a sequence.
    Returns list of (start, end, fragment_seq).
    Start/end are 1-based.
    """
    fragments = []
    i = 0
    while i + window <= len(seq):
        frag = seq[i:i + window]
        fragments.append((i + 1, i + window, frag))
        i += stride
    return fragments


def main():
    print("=== 07_fragment_reference.py ===\n")
    print(f"Window: {WINDOW} bp  Stride: {STRIDE} bp\n")

    os.makedirs(FRAGMENTS_DIR, exist_ok=True)

    genomes = {g["genome_id"]: g for g in load_tsv(GENOMES_TSV)}
    splits  = {s["genome_id"]: s["split"] for s in load_tsv(SPLITS_TSV)}

    train_genomes = [
        genomes[gid] for gid, split in splits.items()
        if split == "train_ref"
    ]
    print(f"Train reference genomes to fragment: {len(train_genomes)}\n")

    total_fragments = 0

    fieldnames = ["fragment_id", "genome_id", "genus",
                  "species", "start", "end", "length"]

    with open(FRAGMENTS_TSV, "w", newline="") as tsv_f:
        writer = csv.DictWriter(tsv_f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for i, g in enumerate(train_genomes):
            gid     = g["genome_id"]
            genus   = g["genus"]
            species = g["species"]
            fasta_gz = g["fasta_path"]

            print(f"[{i+1}/{len(train_genomes)}] {gid} — {genus} {species}")

            # read genome
            sequences = read_fasta_gz(fasta_gz)

            # output fasta for this genome
            out_fasta = os.path.join(FRAGMENTS_DIR, f"{gid}.fasta")
            frag_count = 0

            with open(out_fasta, "w") as fasta_f:
                for contig_header, seq in sequences:
                    frags = fragment_sequence(seq, WINDOW, STRIDE)

                    for start, end, frag_seq in frags:
                        frag_count += 1
                        frag_id = f"{gid}_frag{frag_count:05d}"

                        # write to FASTA
                        fasta_f.write(f">{frag_id}|{gid}|{start}-{end}\n")
                        fasta_f.write(f"{frag_seq}\n")

                        # write to TSV
                        writer.writerow({
                            "fragment_id": frag_id,
                            "genome_id":   gid,
                            "genus":       genus,
                            "species":     species,
                            "start":       start,
                            "end":         end,
                            "length":      len(frag_seq),
                        })

            total_fragments += frag_count
            print(f"    {frag_count:,} fragments → {out_fasta}")

    print(f"\nTotal fragments: {total_fragments:,}")
    print(f"Written: {FRAGMENTS_TSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
