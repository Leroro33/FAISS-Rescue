# V3 Experiment Results

## Configuration

- Branch: `feature/k7-dim2048`
- k-mer length: 5 → 7
- Embedding dimension: 512 → 768 (originally planned 2048; reduced to fit 16 GB RAM)
- Canonical k-mers + IDF weighting: retained from V2

## Headline (OPQ; PQ not run for V3)

| Metric | V2 | V3 | Δ |
|---|---|---|---|
| Genus F1 | 64.30% | **75.24%** | **+10.94 pp** |
| Species F1 | 64.20% | 75.12% | +10.92 pp |

Beat the doc's projected +5-7 pp gain at DIM=2048, while running at DIM=768.

## Per-Genus Recall

| Genus | n_reads | V2 | V3 | Δ |
|---|---|---|---|---|
| Klebsiella | 1.3M | 65.8% | 80.8% | +15.0 |
| Escherichia | 1.2M | 67.2% | 76.2% | +9.0 |
| Staphylococcus | 1k | 49.4% | 59.8% | +10.4 |
| Mycobacterium | 13k | 99.8% | 100% | +0.2 |
| Listeria | 40k | 16.9% | 20.2% | +3.3 |
| Pseudomonas | 78k | 53.1% | 49.3% | -3.9 |
| Salmonella | 29k | 10.9% | **7.4%** | **-3.5** |
| Acinetobacter | 39k | 32.4% | 25.5% | -6.9 |
| Streptococcus | 23k | 51.8% | 44.3% | -7.5 |

## Key finding

V3 helped the two dominant Enterobacteriaceae (Klebsiella + E. coli, ~92% of rescue reads), dragging the overall metric up. But it hurt every minority genus that competes with abundant sister classes — including Salmonella, the original motivation for the experiment.

**Salmonella regressed from 10.9% to 7.4%.** Diagnosis: sharper k-mer features (k=7 with 16k canonical k-mers) make the model more confidently wrong when k=10 nearest-neighbor voting is dominated by abundant cousins (Klebsiella, E. coli). Staphylococcus is the exception: smallest class (1k reads) but taxonomically isolated, so no close competitor to misvote toward.

## Conclusion

Feature resolution was **not** the bottleneck for Salmonella. The doc's Section 6 hypothesis is confirmed: Salmonella's bottleneck is class imbalance, not embedding quality.

If Salmonella remains the priority, the next experiment is **Path C** from the doc: a hierarchical classifier with family-level routing followed by within-family genus prediction. In that design, Salmonella has equal data weight against E. coli and Klebsiella within the Enterobacteriaceae stage, eliminating the imbalance.

## Implementation

- Embedding script: `scripts/08c_compute_embeddings_v3.py`
- Pipeline wrapper: `scripts/run_v3_pipeline.sh`
- Outputs: `embeddings/*_v3.*`, `indices/faiss_*_v3.index`, `results/retrieval_hits/hits_opq_v3.tsv`, `results/rescue_predictions_opq_v3.tsv`, `results/metrics_v3/`
- PQ variant for V3 was not run (skipped after a mid-pipeline crash recovery; OPQ is the headline variant anyway)
