# FAISS-Rescue

A k-mer embedding pipeline for taxonomic rescue of metagenomic reads that fail Kraken2 classification. Uses FAISS vector similarity search with Product Quantization (PQ) and Optimized Product Quantization (OPQ) indices to recover genus- and species-level assignments for unclassified short reads.

---

## Background

Standard Kraken2 classification leaves a significant fraction of reads unclassified, especially in samples with low-abundance or divergent taxa. FAISS-Rescue targets these unclassified reads and attempts genus/species rescue by:

1. Fragmenting reference genomes into overlapping windows
2. Encoding each fragment as a hashed canonical 5-mer frequency vector
3. Building a compressed FAISS index over all reference fragments
4. Querying each unclassified read against the index
5. Assigning taxonomy by majority vote over the top-k nearest neighbors

---

## Repository structure

```
FAISS-Rescue/
├── data/
│   ├── raw_genomes/          # Reference genome FASTA files (not tracked)
│   ├── reference_fragments/  # Windowed fragments from reference (not tracked)
│   └── simulated_reads/      # ART Illumina simulated reads (not tracked)
├── embeddings/               # Hashed k-mer vectors (.npy, not tracked)
├── indices/                  # FAISS PQ/OPQ index files (.index, not tracked)
├── metadata/
│   ├── fragments.tsv         # Fragment-to-genome mapping (not tracked)
│   ├── read_truth.tsv        # Ground truth genus/species per read (not tracked)
│   └── rescue_reads.tsv      # Reads flagged for rescue (not tracked)
├── results/
│   ├── figures/              # Per-genus F1 plots, delta plots, summary tables
│   └── metrics/              # TSV diagnostic and summary tables
├── results_hardclose_oldstyle/
│   └── metrics/              # Hard-close benchmark results (PQ and OPQ)
├── scripts/                  # Full numbered pipeline (see below)
├── genera.txt                # List of genera included in the rescue set
├── V3_RESULTS.md             # Detailed V3 experiment results and analysis
└── .gitignore
```

---

## Pipeline overview

Scripts are numbered in execution order. Large intermediate files (embeddings, indices, raw reads) are excluded from the repository via `.gitignore`.

### Core rescue pipeline

| Script | Description |
|--------|-------------|
| `scripts/03_simulate_reads_hardclose_oldstyle.py` | Simulate Illumina short reads from reference genomes using ART |
| `scripts/07_fragment_reference_hardclose_oldstyle.py` | Fragment reference genomes into overlapping windows |
| `scripts/08_compute_embeddings_hardclose_oldstyle.py` | Compute hashed canonical 5-mer frequency embeddings |
| `scripts/09_build_faiss_index_hardclose_oldstyle.py` | Build PQ and OPQ FAISS indices from fragment embeddings |
| `scripts/10_search_index_hardclose_oldstyle_streaming.py` | Stream-query rescue reads against the FAISS index |
| `scripts/10b_resume_opq_search_hardclose_oldstyle.py` | Resume interrupted OPQ search runs |

### Validation and diagnostics

| Script | Description |
|--------|-------------|
| `scripts/14_audit_extremes.py` | Audit best- and worst-performing genera |
| `scripts/15_per_genus_diagnostics.py` | Per-genus precision, recall, F1 breakdown |
| `scripts/16_confusion_by_genus.py` | Confusion matrix between predicted and true genera |
| `scripts/17_hit_composition_by_genus.py` | Retrieval neighborhood composition per true genus |
| `scripts/18_reference_fragment_balance.py` | Reference fragment count per genome (imbalance check) |
| `scripts/20_vote_sensitivity.py` | Sensitivity analysis: weighted vs unweighted voting |
| `scripts/21_revote_opq_k3.py` | Re-run voting with k=3 nearest neighbors (OPQ) |
| `scripts/22_evaluate_opq_k3.py` | Evaluate OPQ k=3 predictions vs ground truth |
| `scripts/23_per_genus_diagnostics_opq_k3.py` | Per-genus diagnostics for OPQ k=3 |

### POSMM preparation (experimental)

| Script | Description |
|--------|-------------|
| `scripts/24_prepare_posmm_reference.py` | Format reference genomes for POSMM input |
| `scripts/25_make_rescue_fasta_for_posmm.py` | Export rescue reads as FASTA for POSMM |
| `scripts/26_make_posmm_taxlist.py` | Generate custom taxonomy list for POSMM |

> **Note:** POSMM lineage generation was not reproducible in the current WSL environment due to memory constraints. Preparation scripts are included for future use.

---

## Key results

### V3 experiment (OPQ, k-mer length 7, dim 768)

| Metric | V2 | V3 | Δ |
|--------|----|----|---|
| Genus F1 | 64.30% | **75.24%** | +10.94 pp |
| Species F1 | 64.20% | 75.12% | +10.92 pp |

V3 improved on V2's embedding by switching from k=5 to k=7 and expanding dimension from 512 to 768. Gains were concentrated in the two dominant Enterobacteriaceae (Klebsiella, Escherichia). Salmonella regressed (10.9% → 7.4%), confirming that the bottleneck is class imbalance, not embedding quality.

### Hard-close oldstyle benchmark (enteric genera, OPQ)

| Index | Genus/Species Accuracy |
|-------|------------------------|
| PQ | 0.8806 |
| OPQ | **0.8884** |

OPQ outperforms PQ across all enteric genera, with especially large gains for Enterobacter, Escherichia, Morganella, and Citrobacter. This benchmark uses the original hashed k-mer embedding (not the discarded transformer path), confirming the core method remains robust.

### Validation findings (PR #1)

- OPQ + k=3 voting outperforms OPQ + k=10 on the rescue subset overall
- The gain is genus-dependent; retrieval neighborhood purity is the main bottleneck
- Weighted voting gives negligible improvement over unweighted voting
- **Mycobacterium** is recovered with high fidelity (pure neighborhoods, high scores)
- **Salmonella** remains difficult due to strongly mixed retrieval neighborhoods
- **Streptococcus, Staphylococcus, Pseudomonas** frequently act as false-attractor classes

---

## Environment

The pipeline was developed and run under:

- Ubuntu (WSL2 on Windows)
- Python 3.x with `micromamba` environment management (`bioenv`)
- Key dependencies: `faiss-cpu` or `faiss-gpu`, `numpy`, `biopython`, `pandas`

> Large data files (genomes, reads, embeddings, indices) are excluded from the repository. See `.gitignore` for the full list.

---

## Notes and known issues

- Script 19 is intentionally absent (intermediate step was absorbed into adjacent scripts)
- The `results_hardclose_oldstyle/` directory mirrors the structure of `results/` for the hard-close benchmark experiment and is kept separate to avoid confusion with the main rescue results
- POSMM integration is incomplete pending a higher-memory compute environment

---

## Related files

- [`V3_RESULTS.md`](./V3_RESULTS.md) — detailed V3 experiment write-up with per-genus breakdown and interpretation
- [`genera.txt`](./genera.txt) — list of genera targeted for rescue
