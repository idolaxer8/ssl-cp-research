#!/bin/bash
# Run the FCP+MS-CS+PCA ablation on the cluster.
#
# Six methods compared: FCP, FCP+MS-CS, FCP+PCA, FCP+PCA+MS-CS, FCP+AE, FCP+AE+MS-CS.
# Metrics: coverage, set size, runtime per method.
#
# Usage (from repo root, after extracting embeddings_cifar100.pt):
#   bash cluster/run_ablation.sh
#
# To override defaults, set env vars OR pass extra args:
#   N_TRIALS=10 bash cluster/run_ablation.sh
#   bash cluster/run_ablation.sh --ncm geodesic_topk_asym --cal_sizes 400 800
#
# Outputs (under --out_dir):
#   ablation_reduction_mscs_summary.txt
#   ablation_reduction_mscs_results.json
#   ablation_reduction_mscs_cifar100.png

set -euo pipefail

# ============================================================================
# Config — override via env vars before invoking
# ============================================================================
N_TRIALS="${N_TRIALS:-5}"
NCM="${NCM:-geodesic_topk_asym}"               # best NCM per MEMORY.md
CAL_SIZES="${CAL_SIZES:-300 400 600 800 1000}"
ALPHA="${ALPHA:-0.1}"
PCA_DIM="${PCA_DIM:-128}"
AE_DIM="${AE_DIM:-32}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-output/embeddings_cifar100.pt}"
UNLABELED_PATH="${UNLABELED_PATH:-output/embeddings_cifar100_unlabeled.pt}"
OUT_DIR="${OUT_DIR:-output/ablation_cifar100}"
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

# ============================================================================
# Banner + env
# ============================================================================
mkdir -p cluster/logs "$OUT_DIR"
LOG_PATH="cluster/logs/ablation_$(date +%Y%m%d_%H%M%S).log"

echo "=================================================="
echo "Job:      cs_ablation (FCP+MS-CS+PCA)"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "PWD:      $(pwd)"
echo "Log:      $LOG_PATH"
echo "=================================================="
echo "NCM:        $NCM"
echo "Cal sizes:  $CAL_SIZES"
echo "Trials:     $N_TRIALS"
echo "PCA dim:    $PCA_DIM    AE dim: $AE_DIM"
echo "Embeddings: $EMBEDDINGS_PATH"
echo "Unlabeled:  $UNLABELED_PATH (auto-carve if missing)"
echo "Out dir:    $OUT_DIR"
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
    echo "Activated venv: $SSL_CP_VENV"
fi

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ============================================================================
# Run ablation — tee output so progress visible AND captured to log
# ============================================================================
python src/cs_ablation.py \
    --cal_sizes $CAL_SIZES \
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
