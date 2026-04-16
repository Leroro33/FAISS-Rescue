#!/usr/bin/env python3
"""
09_build_faiss_index.py
=======================
Build FAISS indices (Flat, PQ, OPQ) from precomputed reference embeddings.

Inputs
------
  embeddings/reference_emb.npy   — shape (N, d), float32, L2-normalised
  (loaded via memmap to avoid WSL2 RAM spikes)

Outputs
-------
  indices/faiss_flat.index   — IndexFlatIP  (exact inner-product search)
  indices/faiss_pq.index     — IndexPQ      (product-quantised)
  indices/faiss_opq.index    — OPQ rotation + PQ via index_factory

Usage
-----
  conda activate bioenv
  cd ~/CompressedRescue
  python scripts/09_build_faiss_index.py          # defaults
  python scripts/09_build_faiss_index.py --pq_m 16 --pq_nbits 8   # custom PQ

Notes
-----
  • Vectors must already be L2-normalised (08_compute_embeddings.py does this).
    We re-normalise here as a safety net.
  • Batch adding keeps peak RAM manageable on WSL2.
  • Install:  conda install -c pytorch -c conda-forge faiss-cpu
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit(
        "ERROR: faiss not installed.\n"
        "Run:  conda install -c pytorch -c conda-forge faiss-cpu"
    )

# ── defaults ────────────────────────────────────────────────────────────
ROOT        = os.path.expanduser("~/CompressedRescue")
EMB_PATH    = os.path.join(ROOT, "embeddings", "reference_emb.npy")
OUT_DIR     = os.path.join(ROOT, "indices")
LOG_DIR     = os.path.join(ROOT, "logs")

PQ_M       = 8       # number of PQ sub-quantisers
PQ_NBITS   = 8       # bits per sub-quantiser  (256 centroids)
OPQ_M      = 8       # same M for OPQ variant
BATCH_SIZE = 200_000  # vectors added per batch (keeps RAM in check)
TRAIN_FRAC = 0.25     # fraction of vectors used to train PQ/OPQ codebooks
TRAIN_CAP  = 500_000  # absolute cap on training set size


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "09_build_faiss_index.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings as a memory-mapped array, then copy batches on demand."""
    logging.info(f"Loading embeddings (memmap): {path}")
    emb = np.load(path, mmap_mode="r")          # read-only memmap
    n, d = emb.shape
    logging.info(f"  shape = ({n:,}, {d})  dtype = {emb.dtype}")
    return emb


def get_training_set(emb_mmap: np.ndarray, frac: float, cap: int) -> np.ndarray:
    """Return a contiguous float32 copy of a random subset for training."""
    n = emb_mmap.shape[0]
    n_train = min(int(n * frac), cap, n)
    logging.info(f"Sampling {n_train:,} vectors for PQ/OPQ training")
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(n, size=n_train, replace=False)
    idx.sort()                                    # sequential access → faster
    train = np.empty((n_train, emb_mmap.shape[1]), dtype=np.float32)
    train[:] = emb_mmap[idx]                      # deep copy from memmap
    faiss.normalize_L2(train)                     # safety re-normalise
    return train


def add_in_batches(index, emb_mmap: np.ndarray, batch_size: int) -> None:
    """Add vectors to *index* in chunks to limit peak RAM."""
    n = emb_mmap.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.empty((end - start, emb_mmap.shape[1]), dtype=np.float32)
        batch[:] = emb_mmap[start:end]                # deep copy from memmap
        faiss.normalize_L2(batch)                 # safety re-normalise
        index.add(batch)
        logging.info(f"  added {end:>10,} / {n:,}  vectors")


# ── index builders ──────────────────────────────────────────────────────

def build_flat(emb_mmap: np.ndarray, out_dir: str, batch_size: int) -> None:
    """IndexFlatIP — exact inner-product (= cosine on unit vectors)."""
    d = emb_mmap.shape[1]
    logging.info(f"Building IndexFlatIP  (d={d})")
    index = faiss.IndexFlatIP(d)
    t0 = time.time()
    add_in_batches(index, emb_mmap, batch_size)
    elapsed = time.time() - t0

    path = os.path.join(out_dir, "faiss_flat.index")
    faiss.write_index(index, path)
    size_mb = os.path.getsize(path) / 1e6
    logging.info(
        f"  ✓ Flat index saved  →  {path}\n"
        f"    ntotal = {index.ntotal:,}   size = {size_mb:.1f} MB   "
        f"time = {elapsed:.1f}s"
    )


def build_pq(
    emb_mmap: np.ndarray,
    train_vecs: np.ndarray,
    out_dir: str,
    m: int,
    nbits: int,
    batch_size: int,
) -> None:
    """IndexPQ — product-quantised approximate index."""
    d = emb_mmap.shape[1]
    logging.info(f"Building IndexPQ  (d={d}, M={m}, nbits={nbits})")

    factory_str = f"PQ{m}x{nbits}"
    index = faiss.index_factory(d, factory_str, faiss.METRIC_INNER_PRODUCT)

    t0 = time.time()
    index.train(train_vecs)
    logging.info(f"  PQ training done  ({time.time() - t0:.1f}s)")

    t1 = time.time()
    add_in_batches(index, emb_mmap, batch_size)
    elapsed = time.time() - t0

    path = os.path.join(out_dir, "faiss_pq.index")
    faiss.write_index(index, path)
    size_mb = os.path.getsize(path) / 1e6
    logging.info(
        f"  ✓ PQ index saved   →  {path}\n"
        f"    ntotal = {index.ntotal:,}   size = {size_mb:.1f} MB   "
        f"time = {elapsed:.1f}s"
    )


def build_opq(
    emb_mmap: np.ndarray,
    train_vecs: np.ndarray,
    out_dir: str,
    m: int,
    nbits: int,
    batch_size: int,
) -> None:
    """OPQ (optimised product quantisation) via index_factory."""
    d = emb_mmap.shape[1]
    factory_str = f"OPQ{m},PQ{m}x{nbits}"
    logging.info(f"Building OPQ index  (factory='{factory_str}')")

    index = faiss.index_factory(d, factory_str, faiss.METRIC_INNER_PRODUCT)

    t0 = time.time()
    index.train(train_vecs)
    logging.info(f"  OPQ training done  ({time.time() - t0:.1f}s)")

    t1 = time.time()
    add_in_batches(index, emb_mmap, batch_size)
    elapsed = time.time() - t0

    path = os.path.join(out_dir, "faiss_opq.index")
    faiss.write_index(index, path)
    size_mb = os.path.getsize(path) / 1e6
    logging.info(
        f"  ✓ OPQ index saved  →  {path}\n"
        f"    ntotal = {index.ntotal:,}   size = {size_mb:.1f} MB   "
        f"time = {elapsed:.1f}s"
    )


# ── quick sanity check ──────────────────────────────────────────────────

def sanity_check(emb_mmap: np.ndarray, out_dir: str, k: int = 5) -> None:
    """Search each index with a few query vectors and report top-k scores."""
    logging.info(f"Sanity check: querying 3 random vectors (k={k})")

    rng = np.random.default_rng(seed=99)
    q_idx = rng.choice(emb_mmap.shape[0], size=3, replace=False)
    queries = np.empty((3, emb_mmap.shape[1]), dtype=np.float32)
    queries[:] = emb_mmap[q_idx]                  # deep copy from memmap
    faiss.normalize_L2(queries)

    for name in ["faiss_flat.index", "faiss_pq.index", "faiss_opq.index"]:
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            continue
        idx = faiss.read_index(path)
        D, I = idx.search(queries, k)
        logging.info(f"  {name}:")
        for i in range(len(queries)):
            scores = ", ".join(f"{s:.4f}" for s in D[i])
            ids     = ", ".join(str(int(j)) for j in I[i])
            logging.info(f"    q[{q_idx[i]}] → IDs [{ids}]  scores [{scores}]")


# ── main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build FAISS indices (Flat / PQ / OPQ) for CompressedRescue"
    )
    p.add_argument("--emb",        default=EMB_PATH,   help="path to reference_emb.npy")
    p.add_argument("--out_dir",    default=OUT_DIR,     help="output directory for .index files")
    p.add_argument("--log_dir",    default=LOG_DIR,     help="log directory")
    p.add_argument("--pq_m",       type=int, default=PQ_M,      help="PQ sub-quantisers M")
    p.add_argument("--pq_nbits",   type=int, default=PQ_NBITS,  help="PQ bits per sub-vector")
    p.add_argument("--opq_m",      type=int, default=OPQ_M,     help="OPQ sub-quantisers M")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="batch size for adding vectors")
    p.add_argument("--train_frac", type=float, default=TRAIN_FRAC, help="fraction for PQ/OPQ training")
    p.add_argument("--train_cap",  type=int, default=TRAIN_CAP,  help="max training vectors")
    p.add_argument("--skip_flat",  action="store_true", help="skip building flat index")
    p.add_argument("--skip_pq",    action="store_true", help="skip building PQ index")
    p.add_argument("--skip_opq",   action="store_true", help="skip building OPQ index")
    p.add_argument("--no_sanity",  action="store_true", help="skip sanity-check queries")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_dir)

    logging.info("=" * 60)
    logging.info("09_build_faiss_index.py   — CompressedRescue")
    logging.info("=" * 60)

    # ── load ────────────────────────────────────────────────────
    emb = load_embeddings(args.emb)
    n, d = emb.shape

    # validate d is divisible by M
    for label, m in [("PQ_M", args.pq_m), ("OPQ_M", args.opq_m)]:
        if d % m != 0:
            sys.exit(f"ERROR: d={d} not divisible by {label}={m}. "
                     f"Choose M ∈ {{{', '.join(str(x) for x in [4,8,16,32,64] if d % x == 0)}}}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── training set (shared by PQ + OPQ) ──────────────────────
    train_vecs = None
    if not (args.skip_pq and args.skip_opq):
        train_vecs = get_training_set(emb, args.train_frac, args.train_cap)

    # ── build indices ──────────────────────────────────────────
    if not args.skip_flat:
        build_flat(emb, args.out_dir, args.batch_size)

    if not args.skip_pq:
        build_pq(emb, train_vecs, args.out_dir, args.pq_m, args.pq_nbits, args.batch_size)

    if not args.skip_opq:
        build_opq(emb, train_vecs, args.out_dir, args.opq_m, args.pq_nbits, args.batch_size)

    # ── sanity check ───────────────────────────────────────────
    if not args.no_sanity:
        sanity_check(emb, args.out_dir)

    # ── summary ────────────────────────────────────────────────
    logging.info("")
    logging.info("Index summary:")
    for name in ["faiss_flat.index", "faiss_pq.index", "faiss_opq.index"]:
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path):
            mb = os.path.getsize(path) / 1e6
            idx = faiss.read_index(path)
            logging.info(f"  {name:25s}  {idx.ntotal:>10,} vectors   {mb:8.1f} MB")

    logging.info("")
    logging.info("Done.  Next → 10_search_index.py")


if __name__ == "__main__":
    main()
