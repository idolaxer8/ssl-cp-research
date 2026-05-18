#!/bin/bash
# Head-to-head comparison: our winner (FCP+PCA+MS-CS) vs Split CP and SemiCP variants.
#
# 7 methods compared on CIFAR-100:
#   SCP-THR / SCP-APS / SCP-RAPS                (Split CP, 50/50 inner train/cal split)
#   SemiCP-THR / SemiCP-APS / SemiCP-RAPS       (Split CP + NNM augmentation from unlabeled)
#   FCP+PCA+MS-CS                               (our headline method)
#
# Total-label-budget framing: each method gets the same B labels.
# - SCP/SemiCP: split internally 50/50 between training and calibration.
# - FCP+PCA+MS-CS: uses all B for calibration (no separate training step needed).
#
# Usage (from repo root):
#   bash cluster/run_comparison.sh
#
# Override via env vars:
#   N_TRIALS=10 bash cluster/run_comparison.sh
#   CAL_SIZES="400 800" N_TRIALS=2 bash cluster/run_comparison.sh   # smoke test
#
# Outputs (under output/semicp_experiments/):
#   cifar100_semicp_results.json
#   cifar100_semicp_comparison.png

set -euo pipefail

# ============================================================================
# Config — override via env vars
# ============================================================================
N_TRIALS="${N_TRIALS:-5}"
CAL_SIZES="${CAL_SIZES:-300 400 600 800 1000}"
TEST_SIZE="${TEST_SIZE:-500}"
ALPHA="${ALPHA:-0.1}"
PCA_DIM="${PCA_DIM:-128}"
DATASETS="${DATASETS:-cifar100}"
OUTPUT_DIR="${OUTPUT_DIR:-output/semicp_experiments}"
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

# ============================================================================
# Banner + env
# ============================================================================
mkdir -p cluster/logs "$OUTPUT_DIR"
LOG_PATH="cluster/logs/comparison_$(date +%Y%m%d_%H%M%S).log"

echo "=================================================="
echo "Job:      semicp_comparison (FCP+PCA+MS-CS vs SCP/SemiCP)"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "PWD:      $(pwd)"
echo "Log:      $LOG_PATH"
echo "=================================================="
echo "Datasets:   $DATASETS"
echo "Cal sizes:  $CAL_SIZES"
echo "Test size:  $TEST_SIZE"
echo "Trials:     $N_TRIALS"
echo "PCA dim:    $PCA_DIM"
echo "Out dir:    $OUTPUT_DIR"
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
    echo "Activated venv: $SSL_CP_VENV"
fi

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ============================================================================
# Run comparison
# ============================================================================
python src/semicp_experiment.py \
    --datasets $DATASETS \
    --cal_sizes $CAL_SIZES \
    --test_size "$TEST_SIZE" \
    --n_trials "$N_TRIALS" \
    --alpha "$ALPHA" \
    --pca_dim "$PCA_DIM" \
    --output_dir "$OUTPUT_DIR" \
    "$@" 2>&1 | tee "$LOG_PATH"

echo "=================================================="
echo "Done:     $(date -Iseconds)"
ls -lh "$OUTPUT_DIR"/*
echo "=================================================="
