#!/usr/bin/env bash
# diagnostic_02_run_mapping.sh
#
# STEP 2 of the Salmonella misclassification diagnostic.
#
# WHAT IT DOES
# ------------
# 1. Downloads a well-annotated Salmonella enterica reference (LT2 strain,
#    GCF_000006945.2) — both the genome FASTA and its GFF annotation.
# 2. Runs minimap2 to map the misclassified reads (from step 1) against
#    that reference. We use `-ax sr` which is the preset for short reads.
# 3. Produces a SAM file ready for analysis in step 3.
#
# DEPENDENCIES
# ------------
# - minimap2  (install:  conda install -c bioconda minimap2)
# - wget      (already available on Ubuntu/WSL2)
#
# HOW TO RUN
# ----------
#   conda activate bioenv
#   cd ~/CompressedRescue
#   bash scripts/diagnostic_02_run_mapping.sh
#
# OUTPUTS
# -------
#   diagnostic/references/salmonella_LT2.fna
#   diagnostic/references/salmonella_LT2.gff
#   diagnostic/mappings/reads_vs_salmonella.sam

set -euo pipefail   # exit on any error, unset var, or failed pipe

PROJECT="${HOME}/CompressedRescue"
DIAG="${PROJECT}/diagnostic"
INPUT_FASTA="${DIAG}/misclassified_salmonella.fasta"

# Sanity checks ───────────────────────────────────────────────────────────────
if [ ! -f "${INPUT_FASTA}" ]; then
    echo "ERROR: input FASTA not found: ${INPUT_FASTA}"
    echo "Run diagnostic_01_sample_misclassified.py first."
    exit 1
fi

if ! command -v minimap2 &> /dev/null; then
    echo "ERROR: minimap2 not found."
    echo "Install with:  conda install -c bioconda minimap2"
    exit 1
fi

mkdir -p "${DIAG}/references" "${DIAG}/mappings"

# Download reference ──────────────────────────────────────────────────────────
# Salmonella enterica subsp. enterica serovar Typhimurium LT2  (GCF_000006945.2)
# — a standard, well-annotated reference strain.
NCBI_BASE="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/945/GCF_000006945.2_ASM694v2"

REF_FNA_GZ="${DIAG}/references/salmonella_LT2.fna.gz"
REF_GFF_GZ="${DIAG}/references/salmonella_LT2.gff.gz"
REF_FNA="${DIAG}/references/salmonella_LT2.fna"
REF_GFF="${DIAG}/references/salmonella_LT2.gff"

if [ ! -f "${REF_FNA}" ]; then
    echo "[1/3] Downloading Salmonella LT2 genome..."
    wget -q --show-progress -O "${REF_FNA_GZ}" \
        "${NCBI_BASE}/GCF_000006945.2_ASM694v2_genomic.fna.gz"
    gunzip -f "${REF_FNA_GZ}"
else
    echo "[1/3] Reference genome already present: ${REF_FNA}"
fi

if [ ! -f "${REF_GFF}" ]; then
    echo "      Downloading Salmonella LT2 annotation (GFF)..."
    wget -q --show-progress -O "${REF_GFF_GZ}" \
        "${NCBI_BASE}/GCF_000006945.2_ASM694v2_genomic.gff.gz"
    gunzip -f "${REF_GFF_GZ}"
else
    echo "      GFF already present: ${REF_GFF}"
fi

# Map reads ───────────────────────────────────────────────────────────────────
SAM="${DIAG}/mappings/reads_vs_salmonella.sam"
LOG="${DIAG}/mappings/minimap2.log"

echo ""
echo "[2/3] Mapping reads with minimap2 (this takes a minute or two)..."
minimap2 -ax sr -t 4 "${REF_FNA}" "${INPUT_FASTA}" \
    > "${SAM}" 2> "${LOG}"

# Quick stats ─────────────────────────────────────────────────────────────────
total_reads=$(grep -v "^@" "${SAM}" | wc -l)
mapped_reads=$(grep -v "^@" "${SAM}" | awk '$2 != 4' | wc -l)
echo ""
echo "[3/3] Mapping done."
echo "      SAM file:        ${SAM}"
echo "      Alignment lines: ${total_reads}"
echo "      Mapped reads:    ${mapped_reads}"
echo ""
echo "=== DONE ==="
echo "Next step: python scripts/diagnostic_03_analyze_hits.py"
