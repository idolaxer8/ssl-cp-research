#!/bin/bash
# Very-small-cal sweep for the FCP variants ablation.
#
# Probes the regime cal = 50..300, including cal < K (= 100 for CIFAR-100).
# At cal < K, balanced_split falls back to uniform random sampling for cal,
# while test stays stratified. This reveals where each FCP variant breaks.
#
# Methods (6, from cs_ablation.py):
#   FCP, FCP+MS-CS, FCP+PCA, FCP+PCA+MS-CS, FCP+AE, FCP+AE+MS-CS
# NCM: geodesic_topk_mean (unchanged from main ablation).
#
# Usage (from repo root):
#   bash cluster/run_small_cal_sweep.sh
#
# Override via env vars:
#   N_TRIALS=10 bash cluster/run_small_cal_sweep.sh
#   CAL_SIZES="50 100 200" DATASET_TAG=tiny_imagenet bash cluster/run_small_cal_sweep.sh
#
# Outputs (under output/small_cal_sweep_<dataset>/):
#   ablation_reduction_mscs_summary.txt
#   ablation_reduction_mscs_results.json
#   ablation_reduction_mscs_*.png

set -euo pipefail

# ============================================================================
# Config — override via env vars
# ============================================================================
N_TRIALS="${N_TRIALS:-5}"
CAL_SIZES="${CAL_SIZES:-50 100 150 200 250 300}"
TEST_SIZE="${TEST_SIZE:-300}"
NCM="${NCM:-geodesic_topk_mean}"
ALPHA="${ALPHA:-0.1}"
PCA_DIM="${PCA_DIM:-128}"
AE_DIM="${AE_DIM:-32}"

# Dataset selection (defaults to CIFAR-100, but the sweep works for any dataset
# with extracted labeled + unlabeled .pt files):
DATASET_TAG="${DATASET_TAG:-cifar100}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-output/embeddings_${DATASET_TAG}.pt}"
UNLABELED_PATH="${UNLABELED_PATH:-output/embeddings_${DATASET_TAG}_unlabeled.pt}"
OUT_DIR="${OUT_DIR:-output/small_cal_sweep_${DATASET_TAG}}"
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

# ============================================================================
# Banner + env
# ============================================================================
mkdir -p cluster/logs "$OUT_DIR"
LOG_PATH="cluster/logs/small_cal_sweep_${DATASET_TAG}_$(date +%Y%m%d_%H%M%S).log"

echo "=================================================="
echo "Job:      small_cal_sweep  ($DATASET_TAG)"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "Log:      $LOG_PATH"
echo "=================================================="
echo "Dataset:    $DATASET_TAG"
echo "Embeddings: $EMBEDDINGS_PATH"
echo "Unlabeled:  $UNLABELED_PATH"
echo "Cal sizes:  $CAL_SIZES"
echo "Test size:  $TEST_SIZE"
echo "Trials:     $N_TRIALS"
echo "NCM:        $NCM"
echo "PCA dim:    $PCA_DIM   AE dim: $AE_DIM"
echo "Out dir:    $OUT_DIR"
echo "=================================================="
echo "Note: any cal < K_classes triggers uniform-random cal sampling (no stratification)."
echo "      Test stays stratified at TEST_SIZE/K per class for comparability."
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
fi

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ============================================================================
# Run
# ============================================================================
python src/cs_ablation.py \
    --cal_sizes $CAL_SIZES \
    --test_size "$TEST_SIZE" \
    --n_trials "$N_TRIALS" \
    --ncm "$NCM" \
    --alpha "$ALPHA" \
    --pca_dim "$PCA_DIM" \
    --ae_dim "$AE_DIM" \
    --embeddings_path "$EMBEDDINGS_PATH" \
    --unlabeled_path "$UNLABELED_PATH" \
    --out_dir "$OUT_DIR" \
    "$@" 2>&1 | tee "$LOG_PATH"

echo "=================================================="
echo "Done:     $(date -Iseconds)"
ls -lh "$OUT_DIR"/*
echo "=================================================="
