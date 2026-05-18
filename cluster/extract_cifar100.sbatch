#!/bin/bash
# Extract DINOv2 embeddings for CIFAR-100 on a SLURM cluster.
#
# Usage (from repo root, on the cluster login node):
#   sbatch cluster/extract_cifar100.sbatch
#
# Logs land in cluster/logs/. Outputs:
#   data/cifar100/<class>/*.png        (raw images, ImageFolder layout)
#   output/embeddings_cifar100.pt      (10000 x 768 tensor + labels)
#
# Tune the >>> CONFIGURE <<< blocks below for your cluster before submitting.

# ============================================================================
# SLURM directives
# ============================================================================
#SBATCH --job-name=extract_cifar100
#SBATCH --output=cluster/logs/%x_%j.out
#SBATCH --error=cluster/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# >>> CONFIGURE <<< — uncomment + fill in if your cluster requires them:
# #SBATCH --partition=gpu
# #SBATCH --account=<your-account>
# #SBATCH --qos=<your-qos>

set -euo pipefail

echo "=================================================="
echo "Job:      ${SLURM_JOB_NAME:-extract_cifar100}"
echo "JobID:    ${SLURM_JOB_ID:-local}"
echo "Node:     $(hostname)"
echo "GPU:      ${CUDA_VISIBLE_DEVICES:-?}"
echo "Started:  $(date -Iseconds)"
echo "PWD:      $(pwd)"
echo "=================================================="

# ============================================================================
# Python environment
# ============================================================================
# >>> CONFIGURE <<< — pick ONE of the three options below and uncomment it.

# Option A: conda env (recommended; create once via cluster/setup_env.sh)
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate ssl-cp

# Option B: Lmod / Environment Modules (typical HPC)
# module purge
# module load python/3.10 cuda/11.8
# source ~/venvs/ssl-cp/bin/activate

# Option C: plain venv (you created it manually)
# source ~/venvs/ssl-cp/bin/activate

python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ============================================================================
# Pipeline
# ============================================================================
NUM_PER_CLASS=100             # 100 classes x 100 = 10K images
DATA_DIR=data/cifar100
OUTPUT_NAME=embeddings_cifar100.pt
MODEL=dinov2-base

# DINOv2-base at native 518 input + batch 64 — fine on >=16GB VRAM.
# If you hit OOM, drop to: --input_size 336 --batch_size 32
INPUT_SIZE=518
BATCH_SIZE=64

mkdir -p "$DATA_DIR" output cluster/logs

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
