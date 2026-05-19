#!/usr/bin/env python3
import os
import csv
import glob
import numpy as np
import mmh3

FRAGMENTS_DIR  = "data_hardclose_oldstyle/reference_fragments"
READS_DIR      = "data_hardclose_oldstyle/simulated_reads"
READ_TRUTH_TSV = "metadata_hardclose_oldstyle/read_truth.tsv"
FRAGMENTS_TSV  = "metadata_hardclose_oldstyle/fragments.tsv"
EMBEDDINGS_DIR = "embeddings_hardclose_oldstyle"
K              = 5
D              = 512

def embed_sequence(seq, k=K, d=D):
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
    header = None
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if header is not None:
        yield header, "".join(seq)

def read_fastq_ids(path):
    with open(path) as f:
        while True:
            header = f.readline().strip()
            seq    = f.readline().strip()
            f.readline()
            f.readline()
            if not header:
                break
            yield header[1:], seq

def main():
    print("=== 08_compute_embeddings_hardclose_oldstyle.py ===")
    print(f"k={K} d={D}")
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # reference
    fasta_files = sorted(glob.glob(os.path.join(FRAGMENTS_DIR, "*.fasta")))
    ref_embs = []
    ref_ids = []

    for i, fasta_path in enumerate(fasta_files):
        gid = os.path.basename(fasta_path).replace(".fasta", "")
        for header, seq in read_fasta(fasta_path):
            parts = header.split("|")
            frag_id = parts[0]
            genome_id = parts[1] if len(parts) > 1 else gid
            emb = embed_sequence(seq)
            ref_embs.append(emb)
            ref_ids.append({"fragment_id": frag_id, "genome_id": genome_id})
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(fasta_files)} genomes ({len(ref_embs):,} fragments so far)")

    frag_meta = {}
    with open(FRAGMENTS_TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            frag_meta[row["fragment_id"]] = {"genus": row["genus"], "species": row["species"]}

    ref_emb_path = os.path.join(EMBEDDINGS_DIR, "reference_emb.npy")
    ref_arr = np.lib.format.open_memmap(ref_emb_path, mode="w+", dtype=np.float32, shape=(len(ref_embs), D))
    for j, emb in enumerate(ref_embs):
        ref_arr[j] = emb
    ref_arr.flush()

    with open(os.path.join(EMBEDDINGS_DIR, "reference_ids.tsv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fragment_id", "genome_id", "genus", "species"], delimiter="\t")
        writer.writeheader()
        for r in ref_ids:
            meta = frag_meta.get(r["fragment_id"], {})
            writer.writerow({
                "fragment_id": r["fragment_id"],
                "genome_id": r["genome_id"],
                "genus": meta.get("genus", "-"),
                "species": meta.get("species", "-"),
            })

    print(f"Saved: {ref_emb_path} shape={ref_arr.shape}")

    # reads
    truth_meta = {}
    read_ids = []
    with open(READ_TRUTH_TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            truth_meta[row["read_id"]] = {
                "true_genus": row["true_genus"],
                "true_species": row["true_species"],
            }
            read_ids.append(row["read_id"])

    n_reads = len(read_ids)
    rescue_emb_path = os.path.join(EMBEDDINGS_DIR, "rescue_emb.npy")
    rescue_arr = np.lib.format.open_memmap(rescue_emb_path, mode="w+", dtype=np.float32, shape=(n_reads, D))

    ids_path = os.path.join(EMBEDDINGS_DIR, "rescue_ids.tsv")
    ids_file = open(ids_path, "w", newline="")
    ids_writer = csv.DictWriter(ids_file, fieldnames=["read_id", "true_genus", "true_species"], delimiter="\t")
    ids_writer.writeheader()

    idx = 0
    fastq_files = sorted(glob.glob(os.path.join(READS_DIR, "*.fastq")))
    truth_set = set(read_ids)

    for i, fastq_path in enumerate(fastq_files):
        gid = os.path.basename(fastq_path).replace(".fastq", "")
        count = 0
        for rid, seq in read_fastq_ids(fastq_path):
            if rid in truth_set:
                rescue_arr[idx] = embed_sequence(seq)
                meta = truth_meta[rid]
                ids_writer.writerow({
                    "read_id": rid,
                    "true_genus": meta["true_genus"],
                    "true_species": meta["true_species"],
                })
                idx += 1
                count += 1
        print(f"  [{i+1}/{len(fastq_files)}] {gid}: {count:,} reads embedded")

    ids_file.close()
    rescue_arr.flush()
    print(f"Saved: {rescue_emb_path} shape=({idx}, {D})")

if __name__ == "__main__":
    main()
