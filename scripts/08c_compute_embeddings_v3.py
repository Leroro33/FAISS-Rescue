#!/usr/bin/env python3
"""
08c_compute_embeddings_v3.py

V2 EMBEDDING: canonical k-mers + IDF weighting.

WHAT'S DIFFERENT FROM V1
------------------------
V1 (08_compute_embeddings.py) hashes raw k-mers into a 512-dim vector,
counts them, L2-normalizes. That's it.

V2 changes two things:

  1) CANONICAL K-MERS
     For every k-mer, we also compute its reverse complement and use
     whichever is lexicographically smaller. This means a read and its
     reverse complement produce the same vector — eliminating a major
     source of noise. (In V1, half your reads effectively carried
     randomized signal because the strand was arbitrary.)

  2) IDF WEIGHTING
     "Inverse Document Frequency" — borrowed from text retrieval.
     We compute how often each hash bucket appears across all reference
     fragments. Buckets that appear in nearly every fragment (think:
     k-mers from conserved housekeeping genes, ribosomal proteins) are
     down-weighted, because they don't help discriminate genus.
     Buckets that appear in only a few genomes get up-weighted.

EXPECTED IMPACT
---------------
+5-10% F1 from canonical k-mers (helps everyone but especially Salmonella)
+3-7% F1 from IDF weighting (helps the confused Enterobacteriaceae)
Total: should bring genus F1 from 49% to roughly 60% — final number depends
on the data.

INPUT/OUTPUT NAMING
-------------------
V2 outputs go to embeddings/*_v3.npy etc. so they don't overwrite V1.
This lets you keep V1 results for comparison.

INPUTS
------
    metadata/fragments.tsv          (built by script 07)
    metadata/rescue_reads.tsv       (built by script 06)
    data/reference_fragments/       (built by script 07)
    data/simulated_reads/           (built by script 03)

OUTPUTS
-------
    embeddings/reference_emb_v3.npy        (3.7M × 512 float32)
    embeddings/rescue_emb_v3.npy           (2.7M × 512 float32)
    embeddings/reference_ids_v3.tsv        (same content as V1)
    embeddings/rescue_ids_v3.tsv           (same content as V1)
    embeddings/idf_weights_v3.npy          (512-dim float32)

USAGE
-----
    conda activate bioenv
    cd ~/CompressedRescue
    python scripts/08c_compute_embeddings_v3.py
"""

import sys
import gzip
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path.home() / "CompressedRescue"

FRAGMENTS_TSV     = PROJECT_ROOT / "metadata" / "fragments.tsv"
RESCUE_TSV        = PROJECT_ROOT / "metadata" / "rescue_reads.tsv"
FRAGMENT_DIR      = PROJECT_ROOT / "data"     / "reference_fragments"
FASTQ_DIR         = PROJECT_ROOT / "data"     / "simulated_reads"

EMB_DIR           = PROJECT_ROOT / "embeddings"
REF_EMB_OUT       = EMB_DIR / "reference_emb_v3.npy"
RESCUE_EMB_OUT    = EMB_DIR / "rescue_emb_v3.npy"
REF_IDS_OUT       = EMB_DIR / "reference_ids_v3.tsv"
RESCUE_IDS_OUT    = EMB_DIR / "rescue_ids_v3.tsv"
IDF_OUT           = EMB_DIR / "idf_weights_v3.npy"

K                 = 7      # k-mer length (same as V1 — keep apples-to-apples)
DIM               = 2048    # embedding dimension (same as V1)
HASH_SEED         = 0xC0FFEE
# ──────────────────────────────────────────────────────────────────────────────


# ─── K-mer machinery ──────────────────────────────────────────────────────────
# Translation table for reverse complement
COMPLEMENT = bytes.maketrans(b"ACGTacgtNn", b"TGCAtgcaNn")


def revcomp(seq_bytes: bytes) -> bytes:
    """Reverse complement of a DNA sequence (as bytes, very fast)."""
    return seq_bytes.translate(COMPLEMENT)[::-1]


def canonical_kmers(seq: str, k: int):
    """
    Yield CANONICAL k-mers from a sequence.
    A canonical k-mer is min(kmer, revcomp(kmer)) lexicographically.
    Skips any k-mer containing 'N'.
    """
    seq_b = seq.encode("ascii").upper()
    n = len(seq_b)
    for i in range(n - k + 1):
        km = seq_b[i:i + k]
        if b"N" in km:
            continue
        rc = revcomp(km)
        yield km if km <= rc else rc


def hash_kmer(kmer_bytes: bytes, dim: int = DIM, seed: int = HASH_SEED) -> int:
    """
    Hash a k-mer into a bucket [0, dim).
    Python's built-in hash() is fast and reasonable here. We seed it
    via XOR for reproducibility across runs (Python's hash is
    randomized per-process otherwise — we use bytes hash which is stable
    only if PYTHONHASHSEED is fixed, so we roll our own simple FNV-like).
    """
    # FNV-1a 64-bit — deterministic, fast, no dependencies
    h = 0xcbf29ce484222325 ^ seed
    for b in kmer_bytes:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h % dim


def embed_sequence(seq: str, dim: int = DIM, k: int = K) -> np.ndarray:
    """
    Build a k-mer count vector (no normalization, no IDF).
    Normalization and IDF weighting happen at a later stage so we can
    compute IDF first.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for km in canonical_kmers(seq, k):
        vec[hash_kmer(km, dim)] += 1.0
    return vec


# ─── Pass 1: build raw counts for all reference fragments ─────────────────────
def build_reference_counts():
    """
    Read all reference fragment FASTA files, build the raw count matrix.
    Returns (counts: np.ndarray (N, dim), ids: list of (frag_id, genome, genus, species)).
    """
    print(f"[1/4] Reading reference fragments from {FRAGMENT_DIR}")
    fragments_meta = pd.read_csv(FRAGMENTS_TSV, sep="\t")
    print(f"      {len(fragments_meta):,} fragments expected")
    print(f"      Columns: {list(fragments_meta.columns)}")

    # We need to know which FASTA file each fragment lives in. The exact
    # column names vary; we autodetect.
    id_col     = next((c for c in ["fragment_id", "frag_id", "id"]
                       if c in fragments_meta.columns), None)
    genome_col = next((c for c in ["genome_id", "genome"]
                       if c in fragments_meta.columns), None)
    genus_col  = next((c for c in ["genus"] if c in fragments_meta.columns), None)
    species_col = next((c for c in ["species"] if c in fragments_meta.columns), None)
    if not (id_col and genome_col and genus_col):
        sys.exit(f"ERROR: required columns missing in {FRAGMENTS_TSV}. "
                 f"Found: {list(fragments_meta.columns)}")

    n_frags = len(fragments_meta)
    counts  = np.zeros((n_frags, DIM), dtype=np.float32)
    ids_rows = []

    # Build a lookup from fragment_id -> row index, so we can fill in any order
    id_to_row = {}
    for i, row in enumerate(fragments_meta.itertuples(index=False)):
        fid = getattr(row, id_col)
        id_to_row[fid] = i
        ids_rows.append({
            "fragment_id": fid,
            "genome_id":   getattr(row, genome_col),
            "genus":       getattr(row, genus_col),
            "species":     getattr(row, species_col) if species_col else "",
        })

    # Now stream through every FASTA file and embed each record
    fasta_files = sorted(
        list(FRAGMENT_DIR.rglob("*.fna")) +
        list(FRAGMENT_DIR.rglob("*.fasta")) +
        list(FRAGMENT_DIR.rglob("*.fa")) +
        list(FRAGMENT_DIR.rglob("*.fna.gz")) +
        list(FRAGMENT_DIR.rglob("*.fa.gz"))
    )
    if not fasta_files:
        sys.exit(f"ERROR: no FASTA files found under {FRAGMENT_DIR}")
    print(f"      {len(fasta_files)} fragment FASTA file(s)")

    n_done = 0
    t0 = time.time()
    for fp in fasta_files:
        opener = gzip.open if fp.suffix == ".gz" else open
        with opener(fp, "rt") as h:
            for rec in SeqIO.parse(h, "fasta"):
                row_i = id_to_row.get(rec.id.split("|", 1)[0])
                if row_i is None:
                    continue
                counts[row_i] = embed_sequence(str(rec.seq))
                n_done += 1
                if n_done % 100000 == 0:
                    rate = n_done / (time.time() - t0)
                    eta = (n_frags - n_done) / max(rate, 1)
                    print(f"      Embedded {n_done:,}/{n_frags:,}  "
                          f"({rate:,.0f}/sec, eta {eta/60:.1f} min)")

    print(f"      Total embedded: {n_done:,}/{n_frags:,}")
    if n_done < n_frags:
        print(f"      WARNING: {n_frags - n_done:,} fragments not found in FASTA files")

    return counts, pd.DataFrame(ids_rows)


# ─── Pass 2: compute IDF weights ──────────────────────────────────────────────
def compute_idf(counts: np.ndarray) -> np.ndarray:
    """
    IDF weight for bucket b = log(N / (1 + df_b))
    where df_b = number of fragments in which bucket b had count > 0.
    The "+1" prevents division by zero and softens the weight for very
    rare buckets.
    """
    N = counts.shape[0]
    df = np.sum(counts > 0, axis=0).astype(np.float32)   # (DIM,)
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0           # smoothed IDF
    return idf.astype(np.float32)


# ─── Pass 3: apply IDF + L2-normalize ─────────────────────────────────────────
def apply_idf_and_normalize(counts: np.ndarray, idf: np.ndarray) -> np.ndarray:
    """In-place: multiply each row by IDF, then L2-normalize."""
    counts *= idf[np.newaxis, :]
    norms = np.linalg.norm(counts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    counts /= norms
    return counts


# ─── Pass 4: embed rescue reads ───────────────────────────────────────────────
def build_rescue_embeddings(idf: np.ndarray):
    """
    Embed every rescue read (reads where Kraken2 returned 'U') using the
    same canonical-k-mer scheme + the IDF weights computed from the
    reference set.
    """
    print(f"\n[3/4] Reading rescue read metadata: {RESCUE_TSV}")
    rescue = pd.read_csv(RESCUE_TSV, sep="\t")
    print(f"      {len(rescue):,} rescue reads")
    print(f"      Columns: {list(rescue.columns)}")

    n_reads = len(rescue)
    embs = np.zeros((n_reads, DIM), dtype=np.float32)
    id_to_row = {rid: i for i, rid in enumerate(rescue["read_id"].astype(str))}
    needed = set(id_to_row.keys())
    needed_bases = {rid.split("/")[0] for rid in needed}

    print(f"\n      Streaming FASTQ files in {FASTQ_DIR}...")
    fastq_files = sorted(
        list(FASTQ_DIR.rglob("*.fastq")) +
        list(FASTQ_DIR.rglob("*.fq")) +
        list(FASTQ_DIR.rglob("*.fastq.gz")) +
        list(FASTQ_DIR.rglob("*.fq.gz"))
    )
    if not fastq_files:
        sys.exit(f"ERROR: no FASTQ files in {FASTQ_DIR}")
    print(f"      {len(fastq_files)} FASTQ file(s)")

    n_done = 0
    t0 = time.time()
    for i, fp in enumerate(fastq_files, 1):
        opener = gzip.open if fp.suffix == ".gz" else open
        with opener(fp, "rt") as h:
            for rec in SeqIO.parse(h, "fastq"):
                rid = rec.id
                base = rid.split("/")[0]
                row_i = id_to_row.get(rid) or id_to_row.get(base)
                if row_i is None:
                    continue
                embs[row_i] = embed_sequence(str(rec.seq))
                n_done += 1
        if i % 5 == 0 or i == len(fastq_files):
            rate = n_done / (time.time() - t0 + 1e-9)
            print(f"      Scanned {i}/{len(fastq_files)} files; "
                  f"embedded {n_done:,}/{n_reads:,}  ({rate:,.0f}/s)")

    print(f"      Total embedded: {n_done:,}/{n_reads:,}")

    # Apply IDF + L2-normalize
    print(f"\n      Applying IDF weights and L2-normalizing...")
    embs = apply_idf_and_normalize(embs, idf)
    return embs, rescue["read_id"].astype(str).tolist()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"{'V2 EMBEDDING — canonical k-mers + IDF weighting':^70}")
    print("=" * 70)

    # Reference: counts -> IDF -> normalize
    ref_counts, ref_ids = build_reference_counts()

    print(f"\n[2/4] Computing IDF weights across {len(ref_counts):,} fragments")
    idf = compute_idf(ref_counts)
    print(f"      IDF range: min={idf.min():.3f}, max={idf.max():.3f}, "
          f"mean={idf.mean():.3f}")
    print(f"      ({(idf < idf.mean()).sum()} buckets below mean — these are "
          f"the 'common everywhere' buckets that get downweighted)")

    print(f"\n      Applying IDF + L2-normalize to reference embeddings...")
    ref_emb = apply_idf_and_normalize(ref_counts, idf)

    np.save(REF_EMB_OUT, ref_emb)
    np.save(IDF_OUT, idf)
    ref_ids.to_csv(REF_IDS_OUT, sep="\t", index=False)
    print(f"      Wrote {REF_EMB_OUT}  shape={ref_emb.shape}")
    print(f"      Wrote {IDF_OUT}")
    print(f"      Wrote {REF_IDS_OUT}")

    # Free memory before doing the rescue pass
    del ref_counts, ref_emb, ref_ids

    # Rescue
    rescue_emb, rescue_ids = build_rescue_embeddings(idf)
    np.save(RESCUE_EMB_OUT, rescue_emb)
    pd.DataFrame({"read_id": rescue_ids}).to_csv(RESCUE_IDS_OUT,
                                                  sep="\t", index=False)
    print(f"\n[4/4] Wrote {RESCUE_EMB_OUT}  shape={rescue_emb.shape}")
    print(f"      Wrote {RESCUE_IDS_OUT}")

    print("\n" + "=" * 70)
    print(f"{'V2 EMBEDDING DONE':^70}")
    print("=" * 70)
    print("Next: bash scripts/run_v3_pipeline.sh")


if __name__ == "__main__":
    main()
