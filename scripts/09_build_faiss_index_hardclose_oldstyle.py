#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parents[1]

EMB_DIR = ROOT / "embeddings_hardclose_oldstyle"
INDEX_DIR = ROOT / "indices_hardclose_oldstyle"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

REF_EMB = EMB_DIR / "reference_emb.npy"

OUT_PQ  = INDEX_DIR / "faiss_pq.index"
OUT_OPQ = INDEX_DIR / "faiss_opq.index"

def main():
    print("=== 09_build_faiss_index_hardclose_oldstyle.py ===")
    x = np.load(REF_EMB, mmap_mode="r")
    x = np.asarray(x, dtype="float32")

    n, d = x.shape
    print(f"Reference embeddings: n={n:,}, d={d}")

    m = 32
    nbits = 8

    print("Building PQ index...")
    index_pq = faiss.IndexPQ(d, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index_pq.train(x)
    index_pq.add(x)
    faiss.write_index(index_pq, str(OUT_PQ))
    print(f"Saved: {OUT_PQ}")

    print("Building OPQ index...")
    opq = faiss.OPQMatrix(d, m)
    pq = faiss.IndexPQ(d, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index_opq = faiss.IndexPreTransform(opq, pq)
    index_opq.train(x)
    index_opq.add(x)
    faiss.write_index(index_opq, str(OUT_OPQ))
    print(f"Saved: {OUT_OPQ}")

    print("Done.")

if __name__ == "__main__":
    main()
