#!/bin/bash
# Extract DINOv2 embeddings for miniImageNet (Vinyals et al. 2016).
#
# miniImageNet: 100 classes from ImageNet, 600 images/class, 84x84 native.
# Note: learn2learn's train/validation/test splits use DIFFERENT classes
# (64/16/20 disjoint subsets), so there is no natural disjoint train/test
# within the same 100-class label space. We extract the full merged pool
# (~60K images, 100 global classes), and semicp_experiment.py / cs_ablation.py
# carve the unlabeled pool internally from this pool via
# `unlabeled_carve_per_class` (DATASETS dict already configured for this).
#
# Requires: learn2learn (auto-installed if missing).
#
# Usage (from repo root):
#   bash cluster/extract_miniimagenet.sh

set -euo pipefail

# ============================================================================
# Config
# ============================================================================
NUM_PER_CLASS="${NUM_PER_CLASS:-500}"            # 100 * 500 = 50K (matches local MEMORY layout)
DATA_DIR="${DATA_DIR:-data/miniimagenet}"
OUTPUT_NAME="${OUTPUT_NAME:-embeddings_miniimagenet.pt}"
MODEL=dinov2-base
INPUT_SIZE="${INPUT_SIZE:-518}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

# ============================================================================
# Banner + env
# ============================================================================
mkdir -p output cluster/logs
echo "=================================================="
echo "Job:      extract_miniimagenet"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
fi

# Ensure learn2learn is installed (the only nonstandard dep for miniImageNet)
python -c "import learn2learn" 2>/dev/null || {
    echo "Installing learn2learn..."
    PIP_CACHE_DIR="${PIP_CACHE_DIR:-/storage/ido/pip-cache}" pip install --no-cache-dir learn2learn
}

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ============================================================================
# Pipeline (single stage — merged 100-class pool; unlabeled carved by exp scripts)
# ============================================================================
mkdir -p "$DATA_DIR"
if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "[1/2] Downloading miniImageNet ($NUM_PER_CLASS/class merged across train/val/test)..."
    python src/download_datasets.py \
        --dataset miniimagenet \
        --output_dir "$DATA_DIR" \
        --num_per_class "$NUM_PER_CLASS"
else
    echo "[1/2] $DATA_DIR populated -- skipping download."
fi

echo "[2/2] Extracting embeddings -> output/$OUTPUT_NAME ..."
python src/extract_features.py \
    --data_dir "$DATA_DIR" \
    --output_name "$OUTPUT_NAME" \
    --model "$MODEL" \
    --input_size "$INPUT_SIZE" \
    --batch_size "$BATCH_SIZE"

echo "=================================================="
echo "Done:     $(date -Iseconds)"
ls -lh output/embeddings_miniimagenet*.pt 2>/dev/null || true
echo "=================================================="
