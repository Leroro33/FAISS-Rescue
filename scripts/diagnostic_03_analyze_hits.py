#!/usr/bin/env python3
"""
diagnostic_03_analyze_hits.py

STEP 3 of the Salmonella misclassification diagnostic.

WHAT IT DOES
------------
1. Reads the SAM file from step 2 (reads aligned to Salmonella LT2).
2. Reads the GFF annotation file (gene/rRNA/tRNA/CDS coordinates).
3. For every mapped read, finds which annotated feature it overlaps.
4. Bins each read into a category: rRNA / tRNA / ribosomal-protein /
   other CDS / intergenic / unmapped.
5. Prints a summary table and writes it to TSV.
6. Tells you what the answer means for your project.

NO DEPENDENCIES BEYOND PANDAS
-----------------------------
We parse SAM and GFF as plain TSV — no pysam, no samtools needed.

HOW TO RUN
----------
    conda activate bioenv
    cd ~/CompressedRescue
    python scripts/diagnostic_03_analyze_hits.py

OUTPUT
------
    diagnostic/results/region_breakdown.tsv
    diagnostic/results/per_read_categories.tsv
    Console: human-readable summary
"""

import sys
import re
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.home() / "CompressedRescue"
DIAG         = PROJECT_ROOT / "diagnostic"
SAM_PATH     = DIAG / "mappings"   / "reads_vs_salmonella.sam"
GFF_PATH     = DIAG / "references" / "salmonella_LT2.gff"
META_PATH    = DIAG / "misclassified_salmonella_metadata.tsv"

OUT_DIR      = DIAG / "results"
OUT_SUMMARY  = OUT_DIR / "region_breakdown.tsv"
OUT_PERREAD  = OUT_DIR / "per_read_categories.tsv"
# ──────────────────────────────────────────────────────────────────────────────


def parse_gff(gff_path):
    """
    Parse a GFF3 file from NCBI. Return a list of feature dicts:
      {chrom, start (0-based), end, type, gene, product}
    Only keeps features useful for our categorisation.
    """
    features = []
    keep_types = {"gene", "rRNA", "tRNA", "CDS", "ncRNA", "tmRNA"}

    with open(gff_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _src, ftype, start, end, _sc, _str, _ph, attrs = parts
            if ftype not in keep_types:
                continue

            # Attributes look like:  ID=gene-foo;gene=rrsA;product=16S ribosomal RNA
            attr_dict = {}
            for kv in attrs.split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    attr_dict[k.lower()] = v

            features.append({
                "chrom":   chrom,
                "start":   int(start) - 1,   # GFF is 1-based, we use 0-based
                "end":     int(end),
                "type":    ftype,
                "gene":    attr_dict.get("gene", "").lower(),
                "product": attr_dict.get("product", "").lower(),
            })
    return features


def categorize(feat):
    """
    Map an overlapping GFF feature to one of our buckets:
      rRNA_16S, rRNA_23S, rRNA_5S, rRNA_other,
      tRNA, ribosomal_protein, other_CDS, ncRNA, gene
    """
    ftype   = feat["type"]
    gene    = feat["gene"]      # e.g. "rrsa", "rpla", "gyrb"
    product = feat["product"]   # e.g. "16s ribosomal rna"

    # rRNA — recognised by feature type or gene/product text
    if ftype == "rRNA" or "ribosomal rna" in product:
        if "16s" in product or gene.startswith("rrs"):
            return "rRNA_16S"
        if "23s" in product or gene.startswith("rrl"):
            return "rRNA_23S"
        if "5s"  in product or gene.startswith("rrf"):
            return "rRNA_5S"
        return "rRNA_other"

    if ftype == "tRNA":
        return "tRNA"

    if ftype in ("CDS", "gene"):
        # Ribosomal protein gene names: rpsA, rplA, rpmA, ...
        if re.match(r"^(rps|rpl|rpm)[a-z]?\d*$", gene) or "ribosomal protein" in product:
            return "ribosomal_protein"
        return "other_CDS"

    if ftype == "ncRNA":
        return "ncRNA"
    if ftype == "tmRNA":
        return "tmRNA"

    return "other"


def build_lookup(features):
    """Group features by chromosome and sort by start position."""
    by_chrom = defaultdict(list)
    for f in features:
        by_chrom[f["chrom"]].append(f)
    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda x: x["start"])
    return by_chrom


def overlaps(chrom, r_start, r_end, lookup):
    """Return all features that overlap [r_start, r_end) on chrom."""
    if chrom not in lookup:
        return []
    hits = []
    for f in lookup[chrom]:
        if f["end"] < r_start:
            continue
        if f["start"] > r_end:
            break
        if f["start"] < r_end and f["end"] > r_start:
            hits.append(f)
    return hits


def parse_sam(sam_path):
    """
    Yield (read_id, chrom, start, end, mapq, is_unmapped, is_secondary)
    Approximates the alignment end as start + len(SEQ); fine for our
    purposes since features are kilobases long and reads are 150 bp.
    """
    with open(sam_path) as f:
        for line in f:
            if line.startswith("@"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            qname = parts[0]
            flag  = int(parts[1])
            rname = parts[2]
            pos   = int(parts[3])
            mapq  = int(parts[4])
            seq   = parts[9]

            unmapped     = bool(flag & 4)
            secondary    = bool(flag & 256)
            supplementary = bool(flag & 2048)

            if unmapped:
                yield qname, None, None, None, mapq, True, False
                continue
            if secondary or supplementary:
                # only count primary alignments
                continue

            start = pos - 1
            end   = start + max(1, len(seq))
            yield qname, rname, start, end, mapq, False, False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SAM_PATH.exists():
        sys.exit(f"ERROR: SAM not found: {SAM_PATH}\n"
                 f"Run diagnostic_02_run_mapping.sh first.")
    if not GFF_PATH.exists():
        sys.exit(f"ERROR: GFF not found: {GFF_PATH}")

    print(f"[1/3] Parsing GFF: {GFF_PATH}")
    features = parse_gff(GFF_PATH)
    print(f"      {len(features):,} features")
    lookup = build_lookup(features)

    # Optional: load metadata so we can report which predicted-genus the
    # rRNA reads got mistakenly assigned to.
    pred_lookup = {}
    if META_PATH.exists():
        meta = pd.read_csv(META_PATH, sep="\t")
        pred_col = next((c for c in ["predicted_genus", "pred_genus"]
                         if c in meta.columns), None)
        if pred_col:
            pred_lookup = dict(zip(meta["read_id"].astype(str), meta[pred_col]))

    print(f"\n[2/3] Categorising alignments from {SAM_PATH}")

    cat_counter      = Counter()
    pred_by_cat      = defaultdict(Counter)
    perread_rows     = []
    total            = 0

    # Priority order — if a read overlaps several features, pick the most
    # informative category.
    priority = [
        "rRNA_16S", "rRNA_23S", "rRNA_5S", "rRNA_other",
        "tRNA", "tmRNA", "ncRNA", "ribosomal_protein", "other_CDS",
    ]

    for qname, chrom, start, end, mapq, unmapped, _ in parse_sam(SAM_PATH):
        total += 1
        if unmapped:
            cat_counter["unmapped"] += 1
            perread_rows.append((qname, "unmapped", "", mapq))
            continue

        feats = overlaps(chrom, start, end, lookup)
        cats  = {categorize(f) for f in feats}
        chosen = "intergenic"
        for p in priority:
            if p in cats:
                chosen = p
                break

        cat_counter[chosen] += 1
        pred_by_cat[chosen][pred_lookup.get(qname, "?")] += 1
        perread_rows.append((qname, chosen, chrom, mapq))

    print(f"      Processed {total:,} alignment lines")

    # ─── Report ──────────────────────────────────────────────────────────────
    print(f"\n[3/3] Building report")

    # Group fine-grained categories into the headline buckets
    grouped = {
        "rRNA (any)": sum(cat_counter[k] for k in
                          ("rRNA_16S", "rRNA_23S", "rRNA_5S", "rRNA_other")),
        "  └─ 16S rRNA":        cat_counter["rRNA_16S"],
        "  └─ 23S rRNA":        cat_counter["rRNA_23S"],
        "  └─ 5S rRNA":         cat_counter["rRNA_5S"],
        "tRNA":                 cat_counter["tRNA"],
        "Ribosomal protein":    cat_counter["ribosomal_protein"],
        "Other CDS":            cat_counter["other_CDS"],
        "ncRNA / tmRNA":        cat_counter["ncRNA"] + cat_counter["tmRNA"],
        "Intergenic":           cat_counter["intergenic"],
        "Unmapped":             cat_counter["unmapped"],
    }

    print()
    print("=" * 70)
    print(f"{'DIAGNOSTIC RESULTS':^70}")
    print("=" * 70)
    print(f"Total reads in SAM:  {total:,}")
    print()
    print(f"{'Region category':<25} {'Count':>8} {'Pct':>8}")
    print("-" * 45)

    rows = []
    for cat, count in grouped.items():
        pct = count / max(total, 1)
        bar = "▓" * int(pct * 30)
        print(f"  {cat:<23} {count:>8,} {pct:>7.1%}  {bar}")
        if not cat.startswith("  "):  # don't double-count rRNA subtypes
            rows.append({"category": cat, "count": count, "pct": pct})

    pd.DataFrame(rows).to_csv(OUT_SUMMARY, sep="\t", index=False)
    pd.DataFrame(perread_rows,
                 columns=["read_id", "category", "chrom", "mapq"]
                 ).to_csv(OUT_PERREAD, sep="\t", index=False)
    print(f"\nSummary  → {OUT_SUMMARY}")
    print(f"Per-read → {OUT_PERREAD}")

    # ─── Interpretation ──────────────────────────────────────────────────────
    conserved = (
        cat_counter["rRNA_16S"] + cat_counter["rRNA_23S"] +
        cat_counter["rRNA_5S"]  + cat_counter["rRNA_other"] +
        cat_counter["tRNA"] + cat_counter["ribosomal_protein"]
    )
    cons_pct = conserved / max(total, 1)

    print()
    print("=" * 70)
    print(f"{'INTERPRETATION':^70}")
    print("=" * 70)
    print(f"Reads from CONSERVED regions (rRNA + tRNA + ribosomal proteins):")
    print(f"   {conserved:,} of {total:,}  →  {cons_pct:.1%}")
    print()

    if cons_pct >= 0.50:
        print("  ►► >50% from conserved regions — BIOLOGICAL CEILING.")
        print("     E. coli and Salmonella are nearly identical in these regions,")
        print("     so no amount of embedding tuning will recover them.")
        print()
        print("     RECOMMENDATIONS:")
        print("     1. Mask conserved regions out of the FAISS reference index")
        print("        (annotate genomes with prokka or barrnap, then exclude")
        print("        rRNA/tRNA/rRNA-protein fragments from script 07).")
        print("     2. Add an abstention threshold based on genus_margin —")
        print("        if margin is low, predict 'unknown Enterobacteriaceae'")
        print("        instead of guessing wrong.")
    elif cons_pct >= 0.25:
        print("  ►► 25–50% from conserved regions — MIXED.")
        print("     BOTH conserved-region masking AND embedding tweaks will help.")
        print()
        print("     RECOMMENDATIONS:")
        print("     1. Add canonical k-mers + IDF weighting to your embedding.")
        print("     2. Then mask conserved regions.")
        print("     3. Add abstention.")
    else:
        print("  ►► <25% from conserved regions — EMBEDDING IS THE ISSUE.")
        print("     The reads are coming from regions where Salmonella DOES")
        print("     differ from its neighbours, but your embedding isn't")
        print("     capturing the difference.")
        print()
        print("     RECOMMENDATIONS:")
        print("     1. Switch to canonical k-mers (collapses fwd / reverse).")
        print("     2. Add IDF weighting.")
        print("     3. Try k=7 with 2048 dimensions.")

    if pred_lookup:
        print()
        print("Predicted-genus breakdown for rRNA reads (where most confusion lives):")
        rrna_total = sum(cat_counter[k] for k in
                         ("rRNA_16S", "rRNA_23S", "rRNA_5S", "rRNA_other"))
        rrna_preds = Counter()
        for k in ("rRNA_16S", "rRNA_23S", "rRNA_5S", "rRNA_other"):
            rrna_preds.update(pred_by_cat[k])
        for genus, count in rrna_preds.most_common(8):
            pct = count / max(rrna_total, 1)
            print(f"   {genus:<25} {count:>5}  ({pct:.1%})")


if __name__ == "__main__":
    main()
