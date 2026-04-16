#!/usr/bin/env python3
"""
10_search_index.py
==================
Query FAISS indices with rescue-read embeddings and map hits to taxonomy.

Inputs
------
  embeddings/rescue_emb.npy      — shape (Q, d), float32, L2-normalised
  embeddings/rescue_ids.tsv      — read_id column (row order matches rescue_emb)
  embeddings/reference_ids.tsv   — fragment_id, genome_id, genus, species, start, end
  indices/faiss_pq.index         — PQ index
  indices/faiss_opq.index        — OPQ index
  (optional) indices/faiss_flat.index

Outputs
-------
  results/retrieval_hits/hits_pq.tsv
  results/retrieval_hits/hits_opq.tsv
  (optional) results/retrieval_hits/hits_flat.tsv

  Each file has columns:
    read_id  rank  fragment_id  score  neighbor_genome  neighbor_genus  neighbor_species

Usage
-----
  conda activate bioenv
  cd ~/CompressedRescue
  python scripts/10_search_index.py                # defaults: k=10
  python scripts/10_search_index.py --k 20         # more neighbors
  python scripts/10_search_index.py --batch_size 25000  # smaller batches if RAM tight
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    sys.exit(
        "ERROR: faiss not installed.\n"
        "Run:  conda install -c pytorch -c conda-forge faiss-cpu"
    )

# ── defaults ────────────────────────────────────────────────────────────
ROOT       = os.path.expanduser("~/CompressedRescue")
RESCUE_EMB = os.path.join(ROOT, "embeddings", "rescue_emb.npy")
RESCUE_IDS = os.path.join(ROOT, "embeddings", "rescue_ids.tsv")
REF_IDS    = os.path.join(ROOT, "embeddings", "reference_ids.tsv")
INDEX_DIR  = os.path.join(ROOT, "indices")
OUT_DIR    = os.path.join(ROOT, "results", "retrieval_hits")
LOG_DIR    = os.path.join(ROOT, "logs")

K          = 10        # top-k neighbors per query
BATCH_SIZE = 50_000    # queries per batch


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "10_search_index.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


def load_rescue_embeddings(path: str) -> np.ndarray:
    """Load rescue embeddings as memmap."""
    logging.info(f"Loading rescue embeddings (memmap): {path}")
    emb = np.load(path, mmap_mode="r")
    logging.info(f"  shape = ({emb.shape[0]:,}, {emb.shape[1]})  dtype = {emb.dtype}")
    return emb


def load_rescue_ids(path: str) -> list:
    """Load rescue read IDs — must match row order of rescue_emb.npy."""
    logging.info(f"Loading rescue IDs: {path}")
    df = pd.read_csv(path, sep="\t")
    col = df.columns[0]  # first column is the read ID
    ids = df[col].tolist()
    logging.info(f"  {len(ids):,} rescue read IDs loaded")
    return ids


def load_reference_ids(path: str) -> pd.DataFrame:
    """Load reference fragment metadata for mapping neighbor indices."""
    logging.info(f"Loading reference IDs: {path}")
    df = pd.read_csv(path, sep="\t")
    logging.info(f"  {len(df):,} reference fragments,  columns: {list(df.columns)}")
    return df


def search_index(
    index_path: str,
    rescue_emb: np.ndarray,
    k: int,
    batch_size: int,
) -> tuple:
    """
    Search a FAISS index in batches.
    Returns (all_distances, all_indices) each of shape (Q, k).
    """
    index_name = os.path.basename(index_path)
    logging.info(f"Loading index: {index_name}")
    index = faiss.read_index(index_path)
    logging.info(f"  ntotal = {index.ntotal:,}")

    Q = rescue_emb.shape[0]
    d = rescue_emb.shape[1]
    all_D = np.empty((Q, k), dtype=np.float32)
    all_I = np.empty((Q, k), dtype=np.int64)

    logging.info(f"  Searching {Q:,} queries  (k={k}, batch_size={batch_size:,})")
    t0 = time.time()

    for start in range(0, Q, batch_size):
        end = min(start + batch_size, Q)
        # deep copy from memmap (avoids FAISS segfault)
        batch = np.empty((end - start, d), dtype=np.float32)
        batch[:] = rescue_emb[start:end]
        faiss.normalize_L2(batch)  # safety re-normalise

        D, I = index.search(batch, k)
        all_D[start:end] = D
        all_I[start:end] = I
        logging.info(f"    searched {end:>10,} / {Q:,}")

    elapsed = time.time() - t0
    qps = Q / elapsed if elapsed > 0 else 0
    logging.info(
        f"  ✓ Search done  time = {elapsed:.1f}s  "
        f"({qps:,.0f} queries/sec)"
    )
    return all_D, all_I


def build_hits_table(
    all_D: np.ndarray,
    all_I: np.ndarray,
    rescue_ids: list,
    ref_df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Expand search results into a tidy TSV table."""
    logging.info("Building hits table...")

    rows = []
    for q in range(len(rescue_ids)):
        read_id = rescue_ids[q]
        for rank in range(k):
            ref_idx = int(all_I[q, rank])
            score = float(all_D[q, rank])

            if ref_idx < 0:
                # FAISS returns -1 for missing results
                continue

            ref_row = ref_df.iloc[ref_idx]
            rows.append({
                "read_id":           read_id,
                "rank":              rank + 1,
                "fragment_id":       ref_row.get("fragment_id", f"frag_{ref_idx}"),
                "score":             round(score, 6),
                "neighbor_genome":   ref_row.get("genome_id", ""),
                "neighbor_genus":    ref_row.get("genus", ""),
                "neighbor_species":  ref_row.get("species", ""),
            })

    df = pd.DataFrame(rows)
    logging.info(f"  {len(df):,} hit rows generated")
    return df


def save_hits(df: pd.DataFrame, out_path: str) -> None:
    """Save hits table to TSV."""
    df.to_csv(out_path, sep="\t", index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    logging.info(f"  ✓ Saved → {out_path}  ({size_mb:.1f} MB)")


# ── main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Search FAISS indices with rescue-read embeddings"
    )
    p.add_argument("--rescue_emb", default=RESCUE_EMB,  help="path to rescue_emb.npy")
    p.add_argument("--rescue_ids", default=RESCUE_IDS,  help="path to rescue_ids.tsv")
    p.add_argument("--ref_ids",    default=REF_IDS,     help="path to reference_ids.tsv")
    p.add_argument("--index_dir",  default=INDEX_DIR,   help="directory with .index files")
    p.add_argument("--out_dir",    default=OUT_DIR,     help="output directory for hit tables")
    p.add_argument("--log_dir",    default=LOG_DIR,     help="log directory")
    p.add_argument("--k",         type=int, default=K,          help="top-k neighbors")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="queries per batch")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_dir)

    logging.info("=" * 60)
    logging.info("10_search_index.py   — CompressedRescue")
    logging.info("=" * 60)

    # ── load data ──────────────────────────────────────────────
    rescue_emb = load_rescue_embeddings(args.rescue_emb)
    rescue_ids = load_rescue_ids(args.rescue_ids)
    ref_df     = load_reference_ids(args.ref_ids)

    # sanity check: row counts must match
    if len(rescue_ids) != rescue_emb.shape[0]:
        sys.exit(
            f"ERROR: rescue_ids has {len(rescue_ids)} rows but "
            f"rescue_emb has {rescue_emb.shape[0]} rows — they must match."
        )

    os.makedirs(args.out_dir, exist_ok=True)

    # ── discover indices ───────────────────────────────────────
    index_files = sorted([
        f for f in os.listdir(args.index_dir)
        if f.endswith(".index")
    ])
    if not index_files:
        sys.exit(f"ERROR: no .index files found in {args.index_dir}")

    logging.info(f"Found {len(index_files)} index(es): {index_files}")

    # ── search each index ──────────────────────────────────────
    for idx_file in index_files:
        idx_path = os.path.join(args.index_dir, idx_file)
        tag = idx_file.replace("faiss_", "").replace(".index", "")

        all_D, all_I = search_index(idx_path, rescue_emb, args.k, args.batch_size)
        hits_df = build_hits_table(all_D, all_I, rescue_ids, ref_df, args.k)

        out_path = os.path.join(args.out_dir, f"hits_{tag}.tsv")
        save_hits(hits_df, out_path)

    # ── summary ────────────────────────────────────────────────
    logging.info("")
    logging.info("Output files:")
    for f in sorted(os.listdir(args.out_dir)):
        fpath = os.path.join(args.out_dir, f)
        mb = os.path.getsize(fpath) / 1e6
        logging.info(f"  {f:30s}  {mb:8.1f} MB")

    logging.info("")
    logging.info("Done.  Next → 11_vote_taxonomy.py")


if __name__ == "__main__":
    main()
