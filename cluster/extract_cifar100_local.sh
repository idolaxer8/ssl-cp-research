#!/bin/bash
# Extract DINOv2 embeddings for CIFAR-100 on a single-GPU machine
# (no SLURM / no scheduler — you SSH in and run this directly).
#
# Recommended usage — run inside tmux so the job survives ssh disconnects:
#
#   cd /storage/ido/ssl-cp/ssl-cp-research
#   tmux new -s extract                              # detach later with Ctrl-b d
#   bash cluster/extract_cifar100_local.sh
#
# Or fully background it:
#   nohup bash cluster/extract_cifar100_local.sh > cluster/logs/extract.log 2>&1 &
#
# Outputs:
#   data/cifar100/<class>/*.png        (raw images, ImageFolder layout)
#   output/embeddings_cifar100.pt      (10000 x 768 tensor + labels)

set -euo pipefail

# ============================================================================
# Config — edit if needed
# ============================================================================
NUM_PER_CLASS=100                  # 100 classes x 100 = 10K images
DATA_DIR=data/cifar100
OUTPUT_NAME=embeddings_cifar100.pt
MODEL=dinov2-base
# Defaults sized for a Run:AI fractional GPU slice (~5 GB VRAM).
# Override at invocation if you have more — e.g.:
#   INPUT_SIZE=518 BATCH_SIZE=64 bash cluster/extract_cifar100_local.sh
INPUT_SIZE="${INPUT_SIZE:-336}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Optional: point at a venv. Leave empty to use the default `python` on PATH.
VENV_PATH="${SSL_CP_VENV:-}"       # e.g. export SSL_CP_VENV=~/venvs/ssl-cp

# ============================================================================
# Banner
# ============================================================================
echo "=================================================="
echo "Job:      extract_cifar100 (local, no scheduler)"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "PWD:      $(pwd)"
echo "=================================================="

# ============================================================================
# Environment
# ============================================================================
if [ -n "$VENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    echo "Activated venv: $VENV_PATH"
fi

python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

if ! python -c "import timm" 2>/dev/null; then
    echo "ERROR: 'timm' not installed in this python. Install first:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# ============================================================================
# Pre-flight: actual writable-space check (df lies on Lustre/PVC mounts)
# ============================================================================
mkdir -p "$DATA_DIR" output cluster/logs
PROBE="cluster/logs/.space_probe.$$"
if ! dd if=/dev/zero of="$PROBE" bs=1M count=10 status=none 2>/dev/null; then
    rm -f "$PROBE"
    echo "ERROR: cannot write 10 MB probe in $(pwd)."
    echo "  Real 'no space left' — free up storage or move repo to a writable mount."
    exit 1
fi
rm -f "$PROBE"

# ============================================================================
# Pipeline
# ============================================================================
if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "[1/2] Downloading CIFAR-100 ($NUM_PER_CLASS / class)..."
    python src/download_datasets.py \
        --dataset cifar100 \
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
