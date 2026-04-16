#!/usr/bin/env python3
"""
11_vote_taxonomy.py
===================
Aggregate k nearest neighbors per rescue read into a genus (and optional
species) prediction via weighted voting.

Method
------
For each read, sum neighbor scores per genus.  The genus with the highest
total weight is the prediction.  A top1/top2 margin is recorded as a
confidence signal.  Species is called only when one species dominates
clearly (configurable margin); otherwise left blank (genus-only call).

Inputs
------
  results/retrieval_hits/hits_pq.tsv
  results/retrieval_hits/hits_opq.tsv
  (optional) results/retrieval_hits/hits_flat.tsv

  Each row: read_id, rank, fragment_id, score, neighbor_genome,
            neighbor_genus, neighbor_species

Outputs
-------
  results/rescue_predictions_pq.tsv
  results/rescue_predictions_opq.tsv
  (optional) results/rescue_predictions_flat.tsv

  Columns:
    read_id, predicted_genus, genus_score, genus_margin, genus_agreement,
    predicted_species, species_score, species_margin, n_neighbors

Usage
-----
  conda activate bioenv
  cd ~/CompressedRescue
  python scripts/11_vote_taxonomy.py
  python scripts/11_vote_taxonomy.py --weight exp      # use exp(score)
  python scripts/11_vote_taxonomy.py --species_margin 0.7
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

# ── defaults ────────────────────────────────────────────────────────────
ROOT     = os.path.expanduser("~/CompressedRescue")
HITS_DIR = os.path.join(ROOT, "results", "retrieval_hits")
OUT_DIR  = os.path.join(ROOT, "results")
LOG_DIR  = os.path.join(ROOT, "logs")

WEIGHT_MODE      = "score"   # {"score", "exp"}
SPECIES_MARGIN   = 0.5       # require (top - second) / top ≥ this for species
CHUNK_SIZE       = 2_000_000  # rows per pandas read_csv chunk


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "11_vote_taxonomy.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


def compute_weights(scores: np.ndarray, mode: str) -> np.ndarray:
    """Convert raw FAISS scores to voting weights."""
    if mode == "score":
        # clip to 0 so negative IP scores can't subtract votes
        return np.clip(scores, 0.0, None)
    elif mode == "exp":
        # exp(score) — emphasises top neighbors more
        return np.exp(scores)
    else:
        sys.exit(f"ERROR: unknown weight mode '{mode}'")


def vote_for_read(group: pd.DataFrame, weight_mode: str,
                  species_margin: float) -> dict:
    """
    Given the k neighbor rows for one read, return a prediction dict.
    """
    weights = compute_weights(group["score"].to_numpy(), weight_mode)

    # --- genus vote ---
    genus_sums = defaultdict(float)
    for g, w in zip(group["neighbor_genus"].to_numpy(), weights):
        genus_sums[g] += w

    sorted_genus = sorted(genus_sums.items(), key=lambda kv: kv[1], reverse=True)
    top_genus, top_g_score = sorted_genus[0]
    second_g_score = sorted_genus[1][1] if len(sorted_genus) > 1 else 0.0
    g_margin = ((top_g_score - second_g_score) / top_g_score
                if top_g_score > 0 else 0.0)

    # agreement = fraction of neighbors whose genus == top_genus
    agreement = float((group["neighbor_genus"] == top_genus).sum()) / len(group)

    # --- species vote (restricted to neighbors of top_genus) ---
    in_top_genus = group["neighbor_genus"] == top_genus
    species_weights = weights[in_top_genus.to_numpy()]
    species_labels  = group.loc[in_top_genus, "neighbor_species"].to_numpy()

    predicted_species = ""
    top_s_score       = 0.0
    s_margin          = 0.0

    if len(species_labels) > 0:
        sp_sums = defaultdict(float)
        for s, w in zip(species_labels, species_weights):
            sp_sums[s] += w
        sorted_sp = sorted(sp_sums.items(), key=lambda kv: kv[1], reverse=True)
        top_sp, top_s_score = sorted_sp[0]
        second_s_score = sorted_sp[1][1] if len(sorted_sp) > 1 else 0.0
        s_margin = ((top_s_score - second_s_score) / top_s_score
                    if top_s_score > 0 else 0.0)
        # only call species if dominant
        if s_margin >= species_margin:
            predicted_species = top_sp

    return {
        "predicted_genus":   top_genus,
        "genus_score":       round(top_g_score, 6),
        "genus_margin":      round(g_margin, 4),
        "genus_agreement":   round(agreement, 4),
        "predicted_species": predicted_species,
        "species_score":     round(top_s_score, 6),
        "species_margin":    round(s_margin, 4),
        "n_neighbors":       len(group),
    }


def process_hits_file(hits_path: str, out_path: str,
                      weight_mode: str, species_margin: float,
                      chunk_size: int) -> None:
    """Stream a hits TSV in chunks, vote per read_id, write predictions."""
    logging.info(f"Processing: {hits_path}")
    size_mb = os.path.getsize(hits_path) / 1e6
    logging.info(f"  file size: {size_mb:.1f} MB")

    t0 = time.time()

    # Read in chunks — but a single read_id must not be split across
    # chunks.  The hits files ARE already grouped by read_id (search loop
    # emits them in query order) but we buffer any partial tail group
    # between chunks just to be safe.
    buffer = pd.DataFrame()
    predictions = []
    n_reads = 0

    reader = pd.read_csv(hits_path, sep="\t", chunksize=chunk_size)
    for chunk_i, chunk in enumerate(reader):
        if not buffer.empty:
            chunk = pd.concat([buffer, chunk], ignore_index=True)

        # Split off the last read_id — it might be incomplete if it
        # continues into the next chunk.  Process all complete read_ids
        # now and buffer the incomplete one.
        last_read = chunk["read_id"].iloc[-1]
        is_last   = chunk["read_id"] == last_read
        buffer    = chunk[is_last].copy()
        complete  = chunk[~is_last]

        for read_id, grp in complete.groupby("read_id", sort=False):
            pred = vote_for_read(grp, weight_mode, species_margin)
            pred["read_id"] = read_id
            predictions.append(pred)
            n_reads += 1

        logging.info(f"    chunk {chunk_i + 1}: processed {n_reads:,} reads "
                     f"(buffered {len(buffer)} rows)")

    # flush remaining buffer
    if not buffer.empty:
        for read_id, grp in buffer.groupby("read_id", sort=False):
            pred = vote_for_read(grp, weight_mode, species_margin)
            pred["read_id"] = read_id
            predictions.append(pred)
            n_reads += 1

    # ── save ──
    df = pd.DataFrame(predictions)
    # nice column order
    cols = ["read_id", "predicted_genus", "genus_score", "genus_margin",
            "genus_agreement", "predicted_species", "species_score",
            "species_margin", "n_neighbors"]
    df = df[cols]
    df.to_csv(out_path, sep="\t", index=False)

    elapsed = time.time() - t0
    out_mb = os.path.getsize(out_path) / 1e6
    logging.info(f"  ✓ {n_reads:,} predictions  →  {out_path}  "
                 f"({out_mb:.1f} MB)  time = {elapsed:.1f}s")

    # ── quick summary ──
    n_sp_called = (df["predicted_species"] != "").sum()
    pct_sp = 100.0 * n_sp_called / n_reads if n_reads > 0 else 0.0
    logging.info(f"  species-level calls: {n_sp_called:,} / {n_reads:,} "
                 f"({pct_sp:.1f}%)")
    logging.info(f"  top 5 predicted genera:")
    for g, c in df["predicted_genus"].value_counts().head(5).items():
        logging.info(f"    {g:25s}  {c:>10,}")


# ── main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Vote k-NN neighbors into taxonomy predictions"
    )
    p.add_argument("--hits_dir",       default=HITS_DIR,
                   help="folder with hits_*.tsv files")
    p.add_argument("--out_dir",        default=OUT_DIR,
                   help="output folder for rescue_predictions_*.tsv")
    p.add_argument("--log_dir",        default=LOG_DIR,
                   help="log directory")
    p.add_argument("--weight",         default=WEIGHT_MODE,
                   choices=["score", "exp"],
                   help="weight mode: 'score' (raw) or 'exp' (exponential)")
    p.add_argument("--species_margin", type=float, default=SPECIES_MARGIN,
                   help="min (top-second)/top ratio to call species")
    p.add_argument("--chunk_size",     type=int, default=CHUNK_SIZE,
                   help="rows per pandas read_csv chunk")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_dir)

    logging.info("=" * 60)
    logging.info("11_vote_taxonomy.py   — CompressedRescue")
    logging.info("=" * 60)
    logging.info(f"weight mode     = {args.weight}")
    logging.info(f"species margin  = {args.species_margin}")
    logging.info("")

    os.makedirs(args.out_dir, exist_ok=True)

    hits_files = sorted([
        f for f in os.listdir(args.hits_dir)
        if f.startswith("hits_") and f.endswith(".tsv")
    ])
    if not hits_files:
        sys.exit(f"ERROR: no hits_*.tsv files in {args.hits_dir}")

    logging.info(f"Found {len(hits_files)} hits file(s): {hits_files}")

    for hf in hits_files:
        hits_path = os.path.join(args.hits_dir, hf)
        tag = hf.replace("hits_", "").replace(".tsv", "")
        out_path = os.path.join(args.out_dir, f"rescue_predictions_{tag}.tsv")
        process_hits_file(hits_path, out_path,
                          args.weight, args.species_margin, args.chunk_size)
        logging.info("")

    logging.info("Done.  Next → 12_evaluate.py")


if __name__ == "__main__":
    main()
