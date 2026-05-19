#!/bin/bash
# Extract DINOv2 embeddings for mini-ImageNet.
#
# Source: HuggingFace timm/mini-imagenet (avoids the abandoned learn2learn
# package, which fails to build on Python 3.11+ due to a Cython/longintrepr.h
# issue). 100 classes; HF train+val+test all use the same 100-class label
# space, so we can split them cleanly:
# - HF train (50K, 500/class)  -> labeled pool
# - HF val   (10K, 100/class)  -> unlabeled pool (naturally disjoint)
# - HF test  (5K,  50/class)   -> unused
#
# Usage (from repo root):
#   bash cluster/extract_miniimagenet.sh                         # both stages
#   STAGE=labeled bash cluster/extract_miniimagenet.sh           # labeled only
#   STAGE=unlabeled bash cluster/extract_miniimagenet.sh         # unlabeled only

set -euo pipefail

# ============================================================================
# Config
# ============================================================================
NUM_PER_CLASS_LABELED="${NUM_PER_CLASS_LABELED:-500}"     # full train: 100 * 500 = 50K
NUM_PER_CLASS_UNLABELED="${NUM_PER_CLASS_UNLABELED:-100}" # full val:   100 * 100 = 10K
DATA_DIR_LABELED="${DATA_DIR_LABELED:-data/miniimagenet}"
DATA_DIR_UNLABELED="${DATA_DIR_UNLABELED:-data/miniimagenet_unlabeled}"
OUTPUT_NAME_LABELED="${OUTPUT_NAME_LABELED:-embeddings_miniimagenet.pt}"
OUTPUT_NAME_UNLABELED="${OUTPUT_NAME_UNLABELED:-embeddings_miniimagenet_unlabeled.pt}"
MODEL=dinov2-base
INPUT_SIZE="${INPUT_SIZE:-518}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STAGE="${STAGE:-both}"   # labeled | unlabeled | both
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

# ============================================================================
# Banner + env
# ============================================================================
mkdir -p output cluster/logs
echo "=================================================="
echo "Job:      extract_miniimagenet  (STAGE=$STAGE)"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
fi

# Ensure the HuggingFace 'datasets' library is installed (no auth needed for
# timm/mini-imagenet). Installs ~50MB on first run; subsequent runs no-op.
python -c "import datasets" 2>/dev/null || {
    echo "Installing 'datasets' library..."
    PIP_CACHE_DIR="${PIP_CACHE_DIR:-/storage/ido/pip-cache}" pip install --no-cache-dir datasets
}

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ============================================================================
# Stage: labeled  (HF train split)
# ============================================================================
if [ "$STAGE" = "labeled" ] || [ "$STAGE" = "both" ]; then
    mkdir -p "$DATA_DIR_LABELED"
    if [ -z "$(ls -A "$DATA_DIR_LABELED" 2>/dev/null)" ]; then
        echo "[labeled 1/2] Downloading mini-ImageNet train split ($NUM_PER_CLASS_LABELED/class)..."
        python src/download_datasets.py \
            --dataset miniimagenet \
            --split train \
            --output_dir "$DATA_DIR_LABELED" \
            --num_per_class "$NUM_PER_CLASS_LABELED"
    else
        echo "[labeled 1/2] $DATA_DIR_LABELED populated -- skipping download."
    fi
    echo "[labeled 2/2] Extracting embeddings -> output/$OUTPUT_NAME_LABELED ..."
    python src/extract_features.py \
        --data_dir "$DATA_DIR_LABELED" \
        --output_name "$OUTPUT_NAME_LABELED" \
        --model "$MODEL" \
        --input_size "$INPUT_SIZE" \
        --batch_size "$BATCH_SIZE"
fi

# ============================================================================
# Stage: unlabeled  (HF validation split, disjoint from train within the same
#                    100-class label space)
# ============================================================================
if [ "$STAGE" = "unlabeled" ] || [ "$STAGE" = "both" ]; then
    mkdir -p "$DATA_DIR_UNLABELED"
    if [ -z "$(ls -A "$DATA_DIR_UNLABELED" 2>/dev/null)" ]; then
        echo "[unlabeled 1/2] Downloading mini-ImageNet val split ($NUM_PER_CLASS_UNLABELED/class)..."
        python src/download_datasets.py \
            --dataset miniimagenet \
            --split test \
            --output_dir "$DATA_DIR_UNLABELED" \
            --num_per_class "$NUM_PER_CLASS_UNLABELED"
    else
        echo "[unlabeled 1/2] $DATA_DIR_UNLABELED populated -- skipping download."
    fi
    echo "[unlabeled 2/2] Extracting embeddings -> output/$OUTPUT_NAME_UNLABELED ..."
    python src/extract_features.py \
        --data_dir "$DATA_DIR_UNLABELED" \
        --output_name "$OUTPUT_NAME_UNLABELED" \
        --model "$MODEL" \
        --input_size "$INPUT_SIZE" \
        --batch_size "$BATCH_SIZE"
fi

echo "=================================================="
echo "Done:     $(date -Iseconds)"
ls -lh output/embeddings_miniimagenet*.pt 2>/dev/null || true
echo "=================================================="
