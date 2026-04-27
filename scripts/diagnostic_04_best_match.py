#!/usr/bin/env python3
"""
diagnostic_04_best_match.py

WHAT IT DOES
------------
Maps the 1000 misclassified Salmonella reads (from diagnostic_01)
against ALL of your reference genomes (the 165 from data/raw_genomes/),
then reports — for each read — which genome and genus it best matches.

WHY
---
The earlier mapping to LT2 alone was too narrow. Salmonella has a huge
strain-specific accessory genome (plasmids, prophages, pathogenicity
islands) that simply isn't in any single reference. Mapping against
your full reference set tells us *what these reads actually look like*:

  • Best match to Salmonella  → embedding is failing → improve embedding
  • Best match to E. coli /
    Klebsiella / etc.         → cross-genus similarity (mobile elements)
                                → use abstention; embedding won't help
  • No match anywhere         → very strain-specific accessory genome
                                → use abstention based on retrieval score

DEPENDENCIES: minimap2  (you already have it)

DISK USAGE
----------
The combined reference FASTA will be ~600-800 MB. Make sure you have
~2 GB free in ~/CompressedRescue.

USAGE
-----
    conda activate bioenv
    cd ~/CompressedRescue
    python scripts/diagnostic_04_best_match.py

OUTPUTS
-------
    diagnostic/all_genome_mapping/all_references.fna   (combined reference)
    diagnostic/all_genome_mapping/reads_vs_all.sam
    diagnostic/results/best_match_per_read.tsv
    diagnostic/results/best_match_summary.tsv
"""

import sys
import gzip
import subprocess
from pathlib import Path
from collections import Counter

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path.home() / "CompressedRescue"
DIAG          = PROJECT_ROOT / "diagnostic"
INPUT_FASTA   = DIAG / "misclassified_salmonella.fasta"
GENOMES_TSV   = PROJECT_ROOT / "metadata" / "genomes.tsv"

WORK_DIR      = DIAG / "all_genome_mapping"
COMBINED_REF  = WORK_DIR / "all_references.fna"
CONTIG_MAP    = WORK_DIR / "contig_to_genome.tsv"
SAM_OUT       = WORK_DIR / "reads_vs_all.sam"
LOG_OUT       = WORK_DIR / "minimap2.log"

RESULT_TSV    = DIAG / "results" / "best_match_per_read.tsv"
SUMMARY_TSV   = DIAG / "results" / "best_match_summary.tsv"
# ──────────────────────────────────────────────────────────────────────────────


def find_col(df, candidates, what):
    """Return the first candidate column that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(f"ERROR: could not find {what} column. "
             f"Tried {candidates}. Available: {list(df.columns)}")


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (DIAG / "results").mkdir(parents=True, exist_ok=True)

    if not INPUT_FASTA.exists():
        sys.exit(f"ERROR: input FASTA not found: {INPUT_FASTA}\n"
                 f"Run diagnostic_01_sample_misclassified.py first.")

    # ── 1. Load genome metadata ──────────────────────────────────────────────
    print(f"[1/5] Loading genome metadata: {GENOMES_TSV}")
    genomes = pd.read_csv(GENOMES_TSV, sep="\t")
    print(f"      {len(genomes)} genomes; columns: {list(genomes.columns)}")

    id_col      = find_col(genomes, ["genome_id", "id"], "genome ID")
    genus_col   = find_col(genomes, ["genus"], "genus")
    species_col = next((c for c in ["species"] if c in genomes.columns), None)
    file_col    = find_col(genomes, ["fasta_path", "file_path", "path", "filename"], "file path")

    # genome_id -> (genus, species)
    id2tax = {
        row[id_col]: (row[genus_col],
                      row[species_col] if species_col else "")
        for _, row in genomes.iterrows()
    }

    # ── 2. Build combined reference ──────────────────────────────────────────
    if COMBINED_REF.exists() and CONTIG_MAP.exists():
        print(f"\n[2/5] Combined reference already exists: {COMBINED_REF}")
    else:
        print(f"\n[2/5] Building combined reference (1-2 min)...")
        n_contigs = 0
        n_missing = 0
        with open(COMBINED_REF, "w") as out, open(CONTIG_MAP, "w") as cmap:
            cmap.write("contig\tgenome_id\n")
            for _, row in genomes.iterrows():
                gid   = row[id_col]
                gpath = PROJECT_ROOT / row[file_col]
                if not gpath.exists():
                    n_missing += 1
                    continue
                opener = gzip.open if str(gpath).endswith(".gz") else open
                with opener(gpath, "rt") as h:
                    for line in h:
                        if line.startswith(">"):
                            contig = line[1:].split()[0]
                            new_name = f"{gid}__{contig}"
                            cmap.write(f"{new_name}\t{gid}\n")
                            out.write(f">{new_name}\n")
                            n_contigs += 1
                        else:
                            out.write(line)
        print(f"      Wrote {n_contigs:,} contigs from "
              f"{len(genomes) - n_missing}/{len(genomes)} genomes")
        if n_missing:
            print(f"      WARNING: {n_missing} genome files missing")

    # Load contig -> genome map
    contig2genome = {}
    with open(CONTIG_MAP) as f:
        next(f)  # header
        for line in f:
            contig, gid = line.rstrip("\n").split("\t")
            contig2genome[contig] = gid
    print(f"      {len(contig2genome):,} contigs in lookup")

    # ── 3. Map reads ─────────────────────────────────────────────────────────
    if SAM_OUT.exists():
        print(f"\n[3/5] SAM already exists: {SAM_OUT}")
    else:
        print(f"\n[3/5] Mapping reads against combined reference...")
        with open(SAM_OUT, "w") as sam_f, open(LOG_OUT, "w") as log_f:
            subprocess.run(
                ["minimap2",
                 "-ax", "sr",
                 "-t", "4",
                 "--secondary=no",
                 str(COMBINED_REF), str(INPUT_FASTA)],
                stdout=sam_f, stderr=log_f, check=True
            )
        print(f"      Done. SAM: {SAM_OUT}")

    # ── 4. Parse SAM, find best alignment per read ──────────────────────────
    print(f"\n[4/5] Parsing SAM, finding best match per read...")

    best = {}  # read_id -> dict
    with open(SAM_OUT) as f:
        for line in f:
            if line.startswith("@"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            qname = parts[0]
            flag  = int(parts[1])
            rname = parts[2]
            mapq  = int(parts[4])

            if flag & 4:  # unmapped
                if qname not in best:
                    best[qname] = {"genome_id": None, "AS": -1,
                                   "mapq": 0, "rname": None}
                continue
            if flag & 256 or flag & 2048:   # secondary / supplementary
                continue

            # Pull AS:i: tag (alignment score)
            as_score = 0
            for tag in parts[11:]:
                if tag.startswith("AS:i:"):
                    as_score = int(tag[5:])
                    break

            gid = contig2genome.get(rname)
            if (qname not in best) or (best[qname]["AS"] < as_score):
                best[qname] = {
                    "genome_id": gid, "AS": as_score,
                    "mapq": mapq, "rname": rname,
                }

    n_total   = len(best)
    n_unmapped = sum(1 for v in best.values() if v["genome_id"] is None)
    n_mapped   = n_total - n_unmapped
    print(f"      Total reads: {n_total:,}")
    print(f"      Mapped:      {n_mapped:,} ({n_mapped/max(n_total,1):.1%})")
    print(f"      Unmapped:    {n_unmapped:,} ({n_unmapped/max(n_total,1):.1%})")

    # ── 5. Tabulate ─────────────────────────────────────────────────────────
    print(f"\n[5/5] Tallying best-match genus per read...")

    rows           = []
    genus_counter  = Counter()
    genome_counter = Counter()

    for rid, v in best.items():
        gid = v["genome_id"]
        if gid is None:
            genus, species = "(unmapped)", ""
        else:
            genus, species = id2tax.get(gid, ("?", "?"))
        rows.append({
            "read_id":      rid,
            "best_genome":  gid or "",
            "best_contig":  v["rname"] or "",
            "best_genus":   genus,
            "best_species": species,
            "AS_score":     v["AS"],
            "mapq":         v["mapq"],
        })
        genus_counter[genus]  += 1
        if gid:
            genome_counter[gid] += 1

    pd.DataFrame(rows).to_csv(RESULT_TSV, sep="\t", index=False)
    pd.DataFrame([
        {"best_match_genus": g, "n_reads": c, "pct": c / max(n_total, 1)}
        for g, c in genus_counter.most_common()
    ]).to_csv(SUMMARY_TSV, sep="\t", index=False)

    # ─── Report ──────────────────────────────────────────────────────────────
    print()
    print("="*70)
    print(f"{'BEST-MATCH GENUS DISTRIBUTION':^70}")
    print("="*70)
    print(f"For {n_total} reads where TRUTH = Salmonella but prediction ≠ Salmonella,")
    print(f"the genus they map BEST to in your full reference set is:\n")
    print(f"{'Best-match genus':<25} {'Count':>8} {'Pct':>8}")
    print("-"*45)
    for genus, count in genus_counter.most_common():
        pct = count / max(n_total, 1)
        bar = "▓" * int(pct * 30)
        print(f"  {genus:<23} {count:>8,} {pct:>7.1%}  {bar}")

    # Top individual genomes
    print(f"\nTop 10 individual genomes the reads map to:")
    for gid, count in genome_counter.most_common(10):
        genus, species = id2tax.get(gid, ("?", "?"))
        print(f"  {gid:<8} {genus} {species:<15}  {count:>4} reads")

    # ─── Interpretation ──────────────────────────────────────────────────────
    salm_n  = genus_counter.get("Salmonella", 0)
    unmap_n = genus_counter.get("(unmapped)", 0)
    other_n = n_total - salm_n - unmap_n

    print()
    print("="*70)
    print(f"{'INTERPRETATION':^70}")
    print("="*70)

    if salm_n / max(n_total, 1) >= 0.5:
        print(f"  ►► {salm_n/n_total:.0%} of reads map BEST to Salmonella in your reference set.")
        print(f"     The reads ARE Salmonella, and your references contain similar")
        print(f"     sequence — but your embedding fails to put them close to Salmonella")
        print(f"     fragments in vector space.")
        print(f"")
        print(f"     RECOMMENDATION: Improve the embedding.")
        print(f"     1. Switch to canonical k-mers (collapses fwd/reverse)")
        print(f"     2. Add IDF weighting to downweight ubiquitous k-mers")
        print(f"     3. Try k=7 with 2048-dim hashing")

    elif other_n / max(n_total, 1) >= 0.5:
        # Identify which other genera are dominant
        top_other = [(g, c) for g, c in genus_counter.most_common()
                     if g not in ("Salmonella", "(unmapped)")][:3]
        print(f"  ►► {other_n/n_total:.0%} of reads map BEST to a NON-Salmonella genus.")
        print(f"     Top non-Salmonella matches:")
        for g, c in top_other:
            print(f"       {g}: {c} reads ({c/n_total:.0%})")
        print(f"")
        print(f"     These reads have stronger similarity to other Enterobacteriaceae")
        print(f"     than to your Salmonella references. Almost certainly horizontally")
        print(f"     transferred mobile elements (plasmids, prophages, integrative")
        print(f"     conjugative elements). These are biologically near-identical")
        print(f"     across genus boundaries — no embedding can resolve them.")
        print(f"")
        print(f"     RECOMMENDATION: Abstention, not embedding tuning.")
        print(f"     1. Add a genus_margin threshold — refuse to predict when low")
        print(f"     2. Optionally, identify and exclude mobile elements from the")
        print(f"        FAISS reference index (using geNomad or PlasmidFinder)")

    elif unmap_n / max(n_total, 1) >= 0.5:
        print(f"  ►► {unmap_n/n_total:.0%} of reads don't map anywhere in your reference set.")
        print(f"     These come from very strain-specific accessory genome that")
        print(f"     even your own Salmonella references don't contain.")
        print(f"")
        print(f"     RECOMMENDATION: Abstention based on retrieval score.")
        print(f"     If FAISS retrieval scores are low, refuse to predict.")
    else:
        print(f"  ►► Mixed picture. See the table above.")
        print(f"     Salmonella: {salm_n/n_total:.0%}, "
              f"Other: {other_n/n_total:.0%}, "
              f"Unmapped: {unmap_n/n_total:.0%}")

    print(f"\nFull per-read table: {RESULT_TSV}")
    print(f"Summary: {SUMMARY_TSV}")


if __name__ == "__main__":
    main()
