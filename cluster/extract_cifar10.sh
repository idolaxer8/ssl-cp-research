#!/bin/bash
# Extract DINOv2 embeddings for CIFAR-10.
#
# CIFAR-10: 10 classes, 32x32 images.
# - Train split: 5000/class (= 50K images) -> labeled pool
# - Test split:  1000/class (= 10K images) -> unlabeled pool (naturally disjoint,
#                                              same trick as the CIFAR-100 setup)
#
# Usage (from repo root):
#   bash cluster/extract_cifar10.sh                          # both stages
#   STAGE=labeled bash cluster/extract_cifar10.sh            # labeled only
#   STAGE=unlabeled bash cluster/extract_cifar10.sh          # unlabeled only

set -euo pipefail

# ============================================================================
# Config — defaults aligned with cifar100 cluster runs
# ============================================================================
NUM_PER_CLASS_LABELED="${NUM_PER_CLASS_LABELED:-1000}"    # 10 * 1000 = 10K labeled
NUM_PER_CLASS_UNLABELED="${NUM_PER_CLASS_UNLABELED:-1000}" # 10 * 1000 = 10K unlabeled
DATA_DIR_LABELED="${DATA_DIR_LABELED:-data/cifar10}"
DATA_DIR_UNLABELED="${DATA_DIR_UNLABELED:-data/cifar10_unlabeled}"
OUTPUT_NAME_LABELED="${OUTPUT_NAME_LABELED:-embeddings_cifar10.pt}"
OUTPUT_NAME_UNLABELED="${OUTPUT_NAME_UNLABELED:-embeddings_cifar10_unlabeled.pt}"
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
echo "Job:      extract_cifar10  (STAGE=$STAGE)"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
fi

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ============================================================================
# Stage: labeled  (CIFAR-10 train split)
# ============================================================================
if [ "$STAGE" = "labeled" ] || [ "$STAGE" = "both" ]; then
    mkdir -p "$DATA_DIR_LABELED"
    if [ -z "$(ls -A "$DATA_DIR_LABELED" 2>/dev/null)" ]; then
        echo "[labeled 1/2] Downloading CIFAR-10 train split ($NUM_PER_CLASS_LABELED/class)..."
        python src/download_datasets.py \
            --dataset cifar10 \
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
# Stage: unlabeled  (CIFAR-10 test split — disjoint from train)
# ============================================================================
if [ "$STAGE" = "unlabeled" ] || [ "$STAGE" = "both" ]; then
    mkdir -p "$DATA_DIR_UNLABELED"
    if [ -z "$(ls -A "$DATA_DIR_UNLABELED" 2>/dev/null)" ]; then
        echo "[unlabeled 1/2] Downloading CIFAR-10 test split ($NUM_PER_CLASS_UNLABELED/class)..."
        python src/download_datasets.py \
            --dataset cifar10 \
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
ls -lh output/embeddings_cifar10*.pt 2>/dev/null || true
echo "=================================================="
