#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESCUE_TSV = ROOT / "metadata" / "rescue_reads.tsv"
SIM_READS_DIR = ROOT / "data" / "simulated_reads"
OUT_FASTA = ROOT / "posmm_rescue_reads.fna"
OUT_MAP = ROOT / "results" / "metrics" / "posmm_rescue_read_order.tsv"
OUT_MAP.parent.mkdir(parents=True, exist_ok=True)

def read_fastq_sequences(path):
    seqs = {}
    with open(path, "r") as f:
        while True:
            h = f.readline().strip()
            if not h:
                break
            s = f.readline().strip()
            f.readline()
            f.readline()
            if h.startswith("@"):
                seqs[h[1:]] = s
    return seqs

def main():
    print("=== 25_make_rescue_fasta_for_posmm.py ===")
    rescue = pd.read_csv(RESCUE_TSV, sep="\t")

    needed = set(rescue["read_id"].astype(str))
    print(f"Need rescue reads: {len(needed):,}")

    found = {}
    for fq in sorted(SIM_READS_DIR.glob("*.fastq")):
        seqs = read_fastq_sequences(fq)
        overlap = needed.intersection(seqs.keys())
        if overlap:
            for rid in overlap:
                found[rid] = seqs[rid]
        print(f"{fq.name}: found cumulative {len(found):,}")

    missing = needed - set(found.keys())
    if missing:
        print(f"WARNING: missing {len(missing):,} rescue reads")
    else:
        print("All rescue reads found.")

    rescue = rescue.copy()
    rescue["read_id"] = rescue["read_id"].astype(str)
    rescue["sequence"] = rescue["read_id"].map(found)

    rescue_ok = rescue.dropna(subset=["sequence"]).reset_index(drop=True)

    with open(OUT_FASTA, "w") as out:
        for _, row in rescue_ok.iterrows():
            out.write(f">{row['read_id']}\n{row['sequence']}\n")

    order_df = rescue_ok[["read_id", "true_genus", "true_species"]].copy()
    order_df.insert(0, "line_number", range(1, len(order_df) + 1))
    order_df.to_csv(OUT_MAP, sep="\t", index=False)

    print(f"Written FASTA: {OUT_FASTA}")
    print(f"Written order map: {OUT_MAP}")
    print(f"Rows written: {len(rescue_ok):,}")

if __name__ == "__main__":
    main()
