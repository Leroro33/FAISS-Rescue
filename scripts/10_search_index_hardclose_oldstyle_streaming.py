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

INDEXES = {
    "pq": INDEX_DIR / "faiss_pq.index",
    "opq": INDEX_DIR / "faiss_opq.index",
}

K = 10
BATCH_SIZE = 25000

def main():
    print("=== 10_search_index_hardclose_oldstyle_streaming.py ===")

    rescue_emb = np.load(RESCUE_EMB, mmap_mode="r")
    rescue_ids = pd.read_csv(RESCUE_IDS, sep="\t")
    ref_ids = pd.read_csv(REF_IDS, sep="\t")

    n_queries, dim = rescue_emb.shape
    print(f"Queries: {n_queries:,}, dim={dim}")
    print(f"Reference rows: {len(ref_ids):,}")

    for name, index_path in INDEXES.items():
        print(f"\nLoading index: {name} -> {index_path}")
        index = faiss.read_index(str(index_path))

        out_path = RESULTS_DIR / f"hits_{name}.tsv"
        if out_path.exists():
            out_path.unlink()

        header_written = False

        for start in range(0, n_queries, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_queries)
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
                out_path,
                sep="\t",
                index=False,
                mode="a",
                header=not header_written
            )
            header_written = True

            print(f"  {name}: processed {end:,} / {n_queries:,}")

        print(f"Saved: {out_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
