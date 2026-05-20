#!/bin/bash
# Sweep over unlabeled-pool size N to measure its effect on FCP+PCA+MS-CS.
#
# For a fixed cal_size, varies N in {0, 100, 250, 500, 1000, 2500, 5000, 10000}.
# At each N: random subsample of unlabeled pool, refit PCA, rebuild MS-CS clusters,
# evaluate 4 methods (FCP, FCP+PCA, FCP+MS-CS, FCP+PCA+MS-CS).
#
# Usage (from repo root):
#   bash cluster/run_unlabeled_size_sweep.sh
#
# Override via env vars:
#   CAL_SIZE=800 N_TRIALS=10 bash cluster/run_unlabeled_size_sweep.sh
#   DATASET_TAG=miniimagenet bash cluster/run_unlabeled_size_sweep.sh
#   UNLABELED_SIZES="0 250 1000 5000 10000" bash cluster/run_unlabeled_size_sweep.sh
#   MSCS_EXCHANGEABLE=1 bash cluster/run_unlabeled_size_sweep.sh   # ~500x slower, restores exchangeability
#
# Outputs (under output/unlabeled_size_sweep_<dataset>/):
#   unlabeled_size_sweep_summary.txt
#   unlabeled_size_sweep_results.json
#   unlabeled_size_sweep.png

set -euo pipefail

# ============================================================================
# Config — override via env vars
# ============================================================================
CAL_SIZE="${CAL_SIZE:-400}"
TEST_SIZE="${TEST_SIZE:-300}"
N_TRIALS="${N_TRIALS:-5}"
UNLABELED_SIZES="${UNLABELED_SIZES:-0 100 250 500 1000 2500 5000 10000}"
NCM="${NCM:-geodesic_topk_mean}"
ALPHA="${ALPHA:-0.1}"
PCA_DIM="${PCA_DIM:-128}"
MSCS_LAMBDA="${MSCS_LAMBDA:-0.05}"
MSCS_N_CLUSTERS="${MSCS_N_CLUSTERS:-20}"
MSCS_TAU_MULT="${MSCS_TAU_MULT:-0.5}"
MSCS_EXCHANGEABLE="${MSCS_EXCHANGEABLE:-0}"   # 1 = pass --mscs_exchangeable

# Dataset selection — defaults to CIFAR-100, but anything with extracted
# labeled + unlabeled .pt files works.
DATASET_TAG="${DATASET_TAG:-cifar100}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-output/embeddings_${DATASET_TAG}.pt}"
UNLABELED_PATH="${UNLABELED_PATH:-output/embeddings_${DATASET_TAG}_unlabeled.pt}"
OUT_DIR="${OUT_DIR:-output/unlabeled_size_sweep_${DATASET_TAG}}"
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

EXCH_FLAG=""
if [ "$MSCS_EXCHANGEABLE" = "1" ]; then
    EXCH_FLAG="--mscs_exchangeable"
fi

# ============================================================================
# Banner + env
# ============================================================================
mkdir -p cluster/logs "$OUT_DIR"
LOG_PATH="cluster/logs/unlabeled_size_sweep_${DATASET_TAG}_$(date +%Y%m%d_%H%M%S).log"

echo "=================================================="
echo "Job:      unlabeled_size_sweep  ($DATASET_TAG)"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "Log:      $LOG_PATH"
echo "=================================================="
echo "Dataset:    $DATASET_TAG"
echo "Embeddings: $EMBEDDINGS_PATH"
echo "Unlabeled:  $UNLABELED_PATH"
echo "Cal size:   $CAL_SIZE   test size: $TEST_SIZE"
echo "Trials:     $N_TRIALS"
echo "N values:   $UNLABELED_SIZES"
echo "NCM:        $NCM"
echo "PCA cap:    $PCA_DIM (auto-reduces if N-1 < pca_dim)"
echo "MS-CS:      lambda=$MSCS_LAMBDA clusters=$MSCS_N_CLUSTERS tau_mult=$MSCS_TAU_MULT exchangeable=$MSCS_EXCHANGEABLE"
echo "Out dir:    $OUT_DIR"
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
fi

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ============================================================================
# Run sweep
# ============================================================================
python src/unlabeled_size_sweep.py \
    --cal_size "$CAL_SIZE" \
    --test_size "$TEST_SIZE" \
    --n_trials "$N_TRIALS" \
    --unlabeled_sizes $UNLABELED_SIZES \
    --ncm "$NCM" \
    --alpha "$ALPHA" \
    --pca_dim "$PCA_DIM" \
    --mscs_lambda "$MSCS_LAMBDA" \
    --mscs_n_clusters "$MSCS_N_CLUSTERS" \
    --mscs_tau_mult "$MSCS_TAU_MULT" \
    --embeddings_path "$EMBEDDINGS_PATH" \
    --unlabeled_path "$UNLABELED_PATH" \
    --out_dir "$OUT_DIR" \
    --dataset_tag "$DATASET_TAG" \
    $EXCH_FLAG \
    "$@" 2>&1 | tee "$LOG_PATH"

echo "=================================================="
echo "Done:     $(date -Iseconds)"
ls -lh "$OUT_DIR"/* || true
echo "=================================================="
