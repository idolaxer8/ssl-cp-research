#!/bin/bash
# Extract DINOv2 embeddings for the CIFAR-100 *test split* and save as the
# unlabeled pool used by MS-CS / PCA fitting.
#
# Why this works: the labeled pool (extract_cifar100_local.sh) was sampled from
# CIFAR-100 *train* split. The CIFAR-100 *test split* is 10000 disjoint images
# (100/class) — perfect drop-in for the unlabeled pool without any data leakage.
#
# Usage (from repo root):
#   bash cluster/extract_cifar100_unlabeled.sh
#
# Outputs:
#   data/cifar100_unlabeled/<class>/*.png   (10000 disjoint images)
#   output/embeddings_cifar100_unlabeled.pt (10000 x 768 tensor + labels)

set -euo pipefail

# ============================================================================
# Config
# ============================================================================
NUM_PER_CLASS=100                      # 100 classes x 100 = 10K images (full test split)
DATA_DIR=data/cifar100_unlabeled
OUTPUT_NAME=embeddings_cifar100_unlabeled.pt
MODEL=dinov2-base
INPUT_SIZE="${INPUT_SIZE:-336}"        # safe default for Run:AI fractional GPU
BATCH_SIZE="${BATCH_SIZE:-32}"
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

# ============================================================================
# Banner + env
# ============================================================================
mkdir -p "$DATA_DIR" output cluster/logs
echo "=================================================="
echo "Job:      extract_cifar100_unlabeled"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "PWD:      $(pwd)"
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
    echo "Activated venv: $SSL_CP_VENV"
fi

python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ============================================================================
# Pre-flight: real write test (Lustre PVC reports misleading df Avail)
# ============================================================================
PROBE="cluster/logs/.space_probe.$$"
if ! dd if=/dev/zero of="$PROBE" bs=1M count=10 status=none 2>/dev/null; then
    rm -f "$PROBE"
    echo "ERROR: cannot write 10 MB probe in $(pwd). Free space and retry."
    exit 1
fi
rm -f "$PROBE"

# ============================================================================
# Pipeline — CIFAR-100 *test split* as the unlabeled pool
# ============================================================================
if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "[1/2] Downloading CIFAR-100 test split ($NUM_PER_CLASS / class, disjoint from train)..."
    python src/download_datasets.py \
        --dataset cifar100 \
        --split test \
        --output_dir "$DATA_DIR" \
        --num_per_class "$NUM_PER_CLASS"
else
    echo "[1/2] $DATA_DIR already populated — skipping download."
fi

echo "[2/2] Extracting $MODEL embeddings (input=$INPUT_SIZE, batch=$BATCH_SIZE)..."
python src/extract_features.py \
    --data_dir "$DATA_DIR" \
    --output_name "$OUTPUT_NAME" \
    --model "$MODEL" \
    --input_size "$INPUT_SIZE" \
    --batch_size "$BATCH_SIZE"

echo "=================================================="
echo "Done:     $(date -Iseconds)"
ls -lh "output/$OUTPUT_NAME"
echo "=================================================="
