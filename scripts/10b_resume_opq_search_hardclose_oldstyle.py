#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import faiss

ROOT = Path(__file__).resolve().parents[1]

EMB_DIR = ROOT / "embeddings_hardclose_oldstyle"
INDEX_DIR = ROOT / "indices_hardclose_oldstyle"
RESULTS_DIR = ROOT / "results_hardclose_oldstyle" / "retrieval_hits"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESCUE_EMB = EMB_DIR / "rescue_emb.npy"
RESCUE_IDS = EMB_DIR / "rescue_ids.tsv"
REF_IDS = EMB_DIR / "reference_ids.tsv"

INDEX_PATH = INDEX_DIR / "faiss_opq.index"
OUT_PATH = RESULTS_DIR / "hits_opq.tsv"

K = 10
BATCH_SIZE = 25000


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def main():
    print("=== 10b_resume_opq_search_hardclose_oldstyle.py ===")

    if not OUT_PATH.exists():
        raise FileNotFoundError(f"Existing partial file not found: {OUT_PATH}")

    print(f"Reading existing output: {OUT_PATH}")
    n_lines = count_lines(OUT_PATH)
    print(f"Existing lines: {n_lines:,}")

    if n_lines < 2:
        raise RuntimeError("hits_opq.tsv exists but has no data rows.")

    n_data = n_lines - 1  # exclude header
    if n_data % K != 0:
        raise RuntimeError(
            f"Data rows ({n_data}) are not divisible by K={K}. "
            f"File may be truncated/corrupted."
        )

    completed_reads = n_data // K
    print(f"Completed reads already present: {completed_reads:,}")

    rescue_emb = np.load(RESCUE_EMB, mmap_mode="r")
    rescue_ids = pd.read_csv(RESCUE_IDS, sep="\t")
    ref_ids = pd.read_csv(REF_IDS, sep="\t")

    total_queries, dim = rescue_emb.shape
    print(f"Total queries: {total_queries:,}, dim={dim}")
    print(f"Reference rows: {len(ref_ids):,}")

    if completed_reads >= total_queries:
        print("OPQ search is already complete. Nothing to do.")
        return

    print(f"Resuming from read index: {completed_reads:,}")
    print(f"Remaining reads: {total_queries - completed_reads:,}")

    print(f"Loading OPQ index: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))

    start_global = completed_reads

    for start in range(start_global, total_queries, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_queries)
        xb = np.asarray(rescue_emb[start:end], dtype="float32")

        scores, idxs = index.search(xb, K)
        batch_ids = rescue_ids.iloc[start:end].reset_index(drop=True)

        rows = []
        for i in range(len(batch_ids)):
            read_id = batch_ids.at[i, "read_id"]
            true_genus = batch_ids.at[i, "true_genus"]
            true_species = batch_ids.at[i, "true_species"]

            for rank in range(K):
                ref_idx = int(idxs[i, rank])
                score = float(scores[i, rank])

                if ref_idx < 0 or ref_idx >= len(ref_ids):
                    continue

                ref_row = ref_ids.iloc[ref_idx]
                rows.append({
                    "read_id": read_id,
                    "true_genus": true_genus,
                    "true_species": true_species,
                    "rank": rank + 1,
                    "score": score,
                    "fragment_id": ref_row["fragment_id"],
                    "genome_id": ref_row["genome_id"],
                    "genus_hit": ref_row["genus"],
                    "species_hit": ref_row["species"],
                })

        batch_df = pd.DataFrame(rows)
        batch_df.to_csv(
            OUT_PATH,
            sep="\t",
            index=False,
            mode="a",
            header=False
        )

        print(f"  opq resumed: processed {end:,} / {total_queries:,}")

    print(f"\nResume complete: {OUT_PATH}")


if __name__ == "__main__":
    main()
