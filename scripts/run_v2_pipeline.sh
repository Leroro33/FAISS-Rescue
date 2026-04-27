#!/usr/bin/env bash
# run_v2_pipeline.sh
#
# Runs scripts 09→10→11→12 on the V2 embeddings, then preserves all
# V2 outputs with "_v2" suffixes so you don't lose your V1 baseline.
#
# WHY THIS IS COMPLEX
# -------------------
# Your existing scripts 09-12 read from and write to FIXED paths
# (e.g. embeddings/reference_emb.npy). They don't have a --variant flag.
# So we need to physically swap V2 files in, run the pipeline, then
# move outputs to V2-suffixed names and restore V1 files.
#
# We use `set -e`: if anything fails, the script stops. If a failure
# happens mid-swap, see the recovery instructions below.
#
# RECOVERY (only if the script crashes mid-run)
# --------------------------------------------
# Files temporarily renamed have ".v1bak" extensions. To restore manually:
#   mv embeddings/reference_emb.npy.v1bak embeddings/reference_emb.npy
#   ...etc for any *.v1bak files you find
#
# USAGE
# -----
#   conda activate bioenv
#   cd ~/CompressedRescue
#   bash scripts/run_v2_pipeline.sh

set -euo pipefail

PROJECT="$HOME/CompressedRescue"
cd "$PROJECT"

EMB="embeddings"
IDX="indices"
RES="results"

echo "============================================================"
echo "       Running V2 pipeline (canonical + IDF embedding)"
echo "============================================================"

# ─── Sanity checks ───────────────────────────────────────────────────────────
for f in \
    "$EMB/reference_emb_v2.npy" \
    "$EMB/rescue_emb_v2.npy" \
    "$EMB/reference_ids_v2.tsv" \
    "$EMB/rescue_ids_v2.tsv" ; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run 08b_compute_embeddings_v2.py first."
        exit 1
    fi
done

# ─── Step 1: back up V1 input files (rename, don't copy — these are big) ─────
echo ""
echo "[1/6] Backing up V1 embedding files..."
for f in reference_emb.npy rescue_emb.npy reference_ids.tsv rescue_ids.tsv ; do
    if [ -f "$EMB/$f" ]; then
        mv "$EMB/$f" "$EMB/$f.v1bak"
        echo "      $EMB/$f  →  $EMB/$f.v1bak"
    fi
done

# ─── Step 2: back up V1 output files so the pipeline doesn't clobber them ────
echo ""
echo "[2/6] Backing up V1 output files..."
declare -a V1_OUTS=(
    "$IDX/faiss_pq.index"
    "$IDX/faiss_opq.index"
    "$RES/retrieval_hits/hits_pq.tsv"
    "$RES/retrieval_hits/hits_opq.tsv"
    "$RES/rescue_predictions_pq.tsv"
    "$RES/rescue_predictions_opq.tsv"
)
for f in "${V1_OUTS[@]}"; do
    if [ -f "$f" ]; then
        mv "$f" "$f.v1bak"
        echo "      $f  →  $f.v1bak"
    fi
done

# Move metrics directory aside if it exists
if [ -d "$RES/metrics" ]; then
    mv "$RES/metrics" "$RES/metrics_v1bak"
    echo "      $RES/metrics  →  $RES/metrics_v1bak"
fi

# ─── Step 3: promote V2 files to the canonical names ─────────────────────────
echo ""
echo "[3/6] Promoting V2 files to canonical names..."
mv "$EMB/reference_emb_v2.npy" "$EMB/reference_emb.npy"
mv "$EMB/rescue_emb_v2.npy"    "$EMB/rescue_emb.npy"
mv "$EMB/reference_ids_v2.tsv" "$EMB/reference_ids.tsv"
mv "$EMB/rescue_ids_v2.tsv"    "$EMB/rescue_ids.tsv"
echo "      done."

# ─── Step 4: run the pipeline ────────────────────────────────────────────────
echo ""
echo "[4/6] Running scripts 09 → 10 → 11 → 12 (this is the slow part)..."
echo ""

python scripts/09_build_faiss_index.py --skip_flat --batch_size 50000
python scripts/10_search_index.py
python scripts/11_vote_taxonomy.py
python scripts/12_evaluate.py

# ─── Step 5: rename all V2 outputs and the V2 inputs to *_v2 names ───────────
echo ""
echo "[5/6] Saving V2 outputs with _v2 suffix..."

# Rename V2 input embeddings back
mv "$EMB/reference_emb.npy"  "$EMB/reference_emb_v2.npy"
mv "$EMB/rescue_emb.npy"     "$EMB/rescue_emb_v2.npy"
mv "$EMB/reference_ids.tsv"  "$EMB/reference_ids_v2.tsv"
mv "$EMB/rescue_ids.tsv"     "$EMB/rescue_ids_v2.tsv"

# Rename V2 outputs
mv "$IDX/faiss_pq.index"                 "$IDX/faiss_pq_v2.index"
mv "$IDX/faiss_opq.index"                "$IDX/faiss_opq_v2.index"
mv "$RES/retrieval_hits/hits_pq.tsv"     "$RES/retrieval_hits/hits_pq_v2.tsv"
mv "$RES/retrieval_hits/hits_opq.tsv"    "$RES/retrieval_hits/hits_opq_v2.tsv"
mv "$RES/rescue_predictions_pq.tsv"      "$RES/rescue_predictions_pq_v2.tsv"
mv "$RES/rescue_predictions_opq.tsv"     "$RES/rescue_predictions_opq_v2.tsv"
mv "$RES/metrics"                        "$RES/metrics_v2"

# ─── Step 6: restore V1 files to their canonical names ───────────────────────
echo ""
echo "[6/6] Restoring V1 files to canonical names..."
for f in reference_emb.npy rescue_emb.npy reference_ids.tsv rescue_ids.tsv ; do
    if [ -f "$EMB/$f.v1bak" ]; then
        mv "$EMB/$f.v1bak" "$EMB/$f"
    fi
done

for f in "${V1_OUTS[@]}"; do
    if [ -f "$f.v1bak" ]; then
        mv "$f.v1bak" "$f"
    fi
done

if [ -d "$RES/metrics_v1bak" ]; then
    mv "$RES/metrics_v1bak" "$RES/metrics"
fi

echo ""
echo "============================================================"
echo "                   V2 PIPELINE DONE"
echo "============================================================"
echo "V1 files: canonical names (e.g. embeddings/reference_emb.npy)"
echo "V2 files: _v2 suffix (e.g. embeddings/reference_emb_v2.npy)"
echo ""
echo "Now run:  python scripts/compare_v1_v2.py"
