#!/usr/bin/env python3
"""
08_compute_embeddings.py

Computes hashed k-mer embeddings for:
    1. All reference fragments (train_ref)
    2. All rescue reads (unclassified by Kraken2)

Uses MurmurHash3 for stable, reproducible hashing across runs.

Inputs:
    data/reference_fragments/<genome_id>.fasta
    data/simulated_reads/<genome_id>.fastq
    metadata/fragments.tsv
    metadata/rescue_reads.tsv

Outputs:
    embeddings/reference_emb.npy     - shape (n_fragments, d)
    embeddings/reference_ids.tsv     - fragment_id, genome_id, genus, species
    embeddings/rescue_emb.npy        - shape (n_rescue_reads, d)
    embeddings/rescue_ids.tsv        - read_id, true_genus, true_species
"""

import os
import csv
import glob
import numpy as np
import mmh3

# ── config ────────────────────────────────────────────────────────────────────
FRAGMENTS_DIR  = "data/reference_fragments"
READS_DIR      = "data/simulated_reads"
RESCUE_TSV     = "metadata/rescue_reads.tsv"
FRAGMENTS_TSV  = "metadata/fragments.tsv"
EMBEDDINGS_DIR = "embeddings"
K              = 5      # k-mer size
D              = 512    # embedding dimension
BATCH_SIZE     = 50000  # save to disk every N embeddings to save RAM
# ─────────────────────────────────────────────────────────────────────────────


def embed_sequence(seq, k=K, d=D):
    """
    Compute a hashed k-mer frequency vector for a DNA sequence.
    Uses MurmurHash3 for stable cross-run hashing.
    Returns a normalized numpy float32 vector of shape (d,).
    """
    vec = np.zeros(d, dtype=np.float32)
    seq = seq.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if 'N' in kmer:
            continue
        idx = mmh3.hash(kmer, signed=False) % d
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def read_fasta(path):
    """Yield (header, sequence) from a FASTA file."""
    header = None
    seq    = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header = line[1:]
                seq    = []
            else:
                seq.append(line)
    if header is not None:
        yield header, "".join(seq)


def read_fastq_ids(path):
    """Yield (read_id) from a FASTQ file."""
    with open(path) as f:
        while True:
            header = f.readline().strip()
            seq    = f.readline().strip()
            f.readline()   # +
            f.readline()   # quality
            if not header:
                break
            yield header[1:], seq   # strip @ from header


def main():
    print("=== 08_compute_embeddings.py ===\n")
    print(f"k={K}  d={D}\n")

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # ── 1. Reference fragment embeddings ──────────────────────────────────────
    print("--- Reference fragment embeddings ---")

    fasta_files = sorted(glob.glob(os.path.join(FRAGMENTS_DIR, "*.fasta")))
    print(f"Fragment FASTA files: {len(fasta_files)}")

    ref_embs = []
    ref_ids  = []

    for i, fasta_path in enumerate(fasta_files):
        gid = os.path.basename(fasta_path).replace(".fasta", "")
        for header, seq in read_fasta(fasta_path):
            parts      = header.split("|")
            frag_id    = parts[0]
            genome_id  = parts[1] if len(parts) > 1 else gid
            coords     = parts[2] if len(parts) > 2 else ""

            emb = embed_sequence(seq)
            ref_embs.append(emb)
            ref_ids.append({
                "fragment_id": frag_id,
                "genome_id":   genome_id,
                "coords":      coords,
            })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(fasta_files)} genomes "
                  f"({len(ref_embs):,} fragments so far)")

    print(f"\nTotal reference embeddings: {len(ref_embs):,}")

    # load genus/species from fragments.tsv
    frag_meta = {}
    with open(FRAGMENTS_TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            frag_meta[row["fragment_id"]] = {
                "genus":   row["genus"],
                "species": row["species"],
            }

    # save reference embeddings
        ref_emb_path = os.path.join(EMBEDDINGS_DIR, "reference_emb.npy")
    ref_arr = np.lib.format.open_memmap(
        ref_emb_path, mode="w+",
        dtype=np.float32, shape=(len(ref_embs), D)
    )
    for j, emb in enumerate(ref_embs):
        ref_arr[j] = emb
    ref_arr.flush()
    print(f"Saved: embeddings/reference_emb.npy  shape={ref_arr.shape}")

    with open(os.path.join(EMBEDDINGS_DIR, "reference_ids.tsv"), "w", newline="") as f:
        writer = csv.DictWriter(f,
            fieldnames=["fragment_id", "genome_id", "genus", "species"],
            delimiter="\t")
        writer.writeheader()
        for r in ref_ids:
            meta = frag_meta.get(r["fragment_id"], {})
            writer.writerow({
                "fragment_id": r["fragment_id"],
                "genome_id":   r["genome_id"],
                "genus":       meta.get("genus",   "-"),
                "species":     meta.get("species", "-"),
            })
    print(f"Saved: embeddings/reference_ids.tsv  ({len(ref_ids):,} rows)\n")

# ── 2. Rescue read embeddings ─────────────────────────────────────────────
    print("--- Rescue read embeddings ---")

    rescue_read_ids = set()
    rescue_meta     = {}
    with open(RESCUE_TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rescue_read_ids.add(row["read_id"])
            rescue_meta[row["read_id"]] = {
                "true_genus":   row["true_genus"],
                "true_species": row["true_species"],
            }
    n_rescue = len(rescue_read_ids)
    print(f"Rescue reads to embed: {n_rescue:,}")

    fastq_files = sorted(glob.glob(os.path.join(READS_DIR, "*.fastq")))

    # pre-allocate output file on disk using memmap
    rescue_emb_path = os.path.join(EMBEDDINGS_DIR, "rescue_emb.npy")
    rescue_arr = np.lib.format.open_memmap(
        rescue_emb_path, mode="w+",
        dtype=np.float32, shape=(n_rescue, D)
    )

    ids_path = os.path.join(EMBEDDINGS_DIR, "rescue_ids.tsv")
    ids_file = open(ids_path, "w", newline="")
    ids_writer = csv.DictWriter(ids_file,
        fieldnames=["read_id", "true_genus", "true_species"],
        delimiter="\t")
    ids_writer.writeheader()

    idx = 0
    for i, fastq_path in enumerate(fastq_files):
        gid   = os.path.basename(fastq_path).replace(".fastq", "")
        count = 0
        for read_id_raw, seq in read_fastq_ids(fastq_path):
            if read_id_raw in rescue_read_ids:
                rescue_arr[idx] = embed_sequence(seq)
                meta = rescue_meta[read_id_raw]
                ids_writer.writerow({
                    "read_id":      read_id_raw,
                    "true_genus":   meta["true_genus"],
                    "true_species": meta["true_species"],
                })
                idx   += 1
                count += 1
        print(f"  [{i+1}/{len(fastq_files)}] {gid}: {count:,} rescue reads embedded")

    ids_file.close()
    rescue_arr.flush()

    # cleanup tmp folder if it exists
    import shutil
    tmp_dir = os.path.join(EMBEDDINGS_DIR, "tmp_rescue")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print("Temp rescue chunks cleaned up.")

    print(f"\nTotal rescue embeddings: {idx:,}")
    print(f"Saved: embeddings/rescue_emb.npy  shape=({idx}, {D})")
    print(f"Saved: embeddings/rescue_ids.tsv  ({idx:,} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
