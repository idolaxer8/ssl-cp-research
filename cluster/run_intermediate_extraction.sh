#!/bin/bash
# Intermediate-layer DINOv2 embeddings for the MDCP layer-dimension direction
# (instructor 2026-07-07; Odonnat et al. CAO@ICLR-2026): block-output CLS taps
# at depths 3/6/9/12 (+ optional FFN-GeLU 'Act' taps) for
#   cifar100, aircraft, stanford_cars, miniimagenet   (labeled + unlabeled pair each)
#
# CRASH SAFETY -- three levels:
#   1. Each dataset+split is an independent STEP; finished steps are skipped
#      (final .pt exists). Re-run the same command after any crash.
#   2. Within a step, extraction checkpoints every CHUNK_SIZE images
#      (output/tmp_*/chunk_*.pt); a crash loses at most one chunk.
#   3. Downloads are skipped when the data dir is already populated.
#
# Usage (from repo ROOT on the cluster):
#   bash cluster/run_intermediate_extraction.sh                    # all 8 steps
#   STEPS="cifar100_labeled aircraft_unlabeled" bash cluster/run_intermediate_extraction.sh
#   DATASETS="cifar100 aircraft" bash cluster/run_intermediate_extraction.sh
#   ACT_LAYERS="6 9" bash cluster/run_intermediate_extraction.sh   # add FFN-GeLU taps
#   FORCE=1 ... redoes steps whose output already exists.
#
# Outputs: output/embeddings_<ds>_layers.pt and output/embeddings_<ds>_unlabeled_layers.pt
#   dict keys: labels, layer03, layer06, layer09, layer12 [, actKK...], final, meta
#   ('final' == the standard post-norm embedding, same convention as embeddings_<ds>.pt)

set -euo pipefail

MODEL="${MODEL:-dinov2-base}"
INPUT_SIZE="${INPUT_SIZE:-518}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LAYERS="${LAYERS:-3 6 9 12}"
ACT_LAYERS="${ACT_LAYERS:-}"             # e.g. "6 9" to also tap FFN GeLU (4x dims)
CHUNK_SIZE="${CHUNK_SIZE:-2048}"
NUM_WORKERS="${NUM_WORKERS:-0}"          # 0 = no /dev/shm use (K8s pods have ~64MB shm)
DATASETS="${DATASETS:-cifar100 aircraft stanford_cars miniimagenet}"
STEPS="${STEPS:-}"                       # explicit step list overrides DATASETS
FORCE="${FORCE:-0}"
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

mkdir -p output cluster/logs
echo "=================================================="
echo "Job:      run_intermediate_extraction"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "Model:    $MODEL @ $INPUT_SIZE | layers: $LAYERS | act: '${ACT_LAYERS}'"
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
    echo "Activated venv: $SSL_CP_VENV"
fi
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# Pre-flight write probe (Lustre PVC reports misleading df Avail)
PROBE="cluster/logs/.space_probe.$$"
if ! dd if=/dev/zero of="$PROBE" bs=1M count=10 status=none 2>/dev/null; then
    rm -f "$PROBE"; echo "ERROR: cannot write 10 MB probe in $(pwd)."; exit 1
fi
rm -f "$PROBE"

# ----------------------------------------------------------------------------
# step <name> <data_dir> <download_args...>  -- download if empty, then extract
# ----------------------------------------------------------------------------
run_step () {
    local name="$1" data_dir="$2" out_name="$3"; shift 3
    local final="output/$out_name"
    echo ""
    echo "----- STEP $name -----"
    if [ -f "$final" ] && [ "$FORCE" != "1" ]; then
        echo "SKIP: $final exists (FORCE=1 to redo)"
        return 0
    fi
    if [ -z "$(ls -A "$data_dir" 2>/dev/null)" ]; then
        echo "[download] $data_dir"
        python src/download_datasets.py --output_dir "$data_dir" "$@"
    else
        echo "[download] $data_dir already populated -- skipping"
    fi
    # shellcheck disable=SC2086
    python src/extract_intermediate_features.py \
        --data_dir "$data_dir" \
        --output_name "$out_name" \
        --model "$MODEL" \
        --input_size "$INPUT_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --layers $LAYERS \
        ${ACT_LAYERS:+--act_layers $ACT_LAYERS} \
        --chunk_size "$CHUNK_SIZE" \
        --num_workers "$NUM_WORKERS"
    ls -lh "$final"
}

want_step () {  # step selection: explicit STEPS wins, else DATASETS membership
    local step="$1" ds="$2"
    if [ -n "$STEPS" ]; then
        [[ " $STEPS " == *" $step "* ]]
    else
        [[ " $DATASETS " == *" $ds "* ]]
    fi
}

# ----------------------------------------------------------------------------
# The 8 steps (labeled + unlabeled per dataset). Download conventions follow
# src/download_datasets.py: 'train' split -> labeled pool, 'test' -> unlabeled.
# cifar100 mirrors cluster/extract_cifar100_*.sh (100/class from each split).
# ----------------------------------------------------------------------------
want_step cifar100_labeled cifar100 && \
    run_step cifar100_labeled data/cifar100 embeddings_cifar100_layers.pt \
        --dataset cifar100 --split train --num_per_class 100
want_step cifar100_unlabeled cifar100 && \
    run_step cifar100_unlabeled data/cifar100_unlabeled embeddings_cifar100_unlabeled_layers.pt \
        --dataset cifar100 --split test --num_per_class 100

want_step aircraft_labeled aircraft && \
    run_step aircraft_labeled data/aircraft embeddings_aircraft_layers.pt \
        --dataset aircraft --split train
want_step aircraft_unlabeled aircraft && \
    run_step aircraft_unlabeled data/aircraft_unlabeled embeddings_aircraft_unlabeled_layers.pt \
        --dataset aircraft --split test

want_step stanford_cars_labeled stanford_cars && \
    run_step stanford_cars_labeled data/stanford_cars embeddings_stanford_cars_layers.pt \
        --dataset stanford_cars --split train
want_step stanford_cars_unlabeled stanford_cars && \
    run_step stanford_cars_unlabeled data/stanford_cars_unlabeled embeddings_stanford_cars_unlabeled_layers.pt \
        --dataset stanford_cars --split test

want_step miniimagenet_labeled miniimagenet && \
    run_step miniimagenet_labeled data/miniimagenet embeddings_miniimagenet_layers.pt \
        --dataset miniimagenet --split train
want_step miniimagenet_unlabeled miniimagenet && \
    run_step miniimagenet_unlabeled data/miniimagenet_unlabeled embeddings_miniimagenet_unlabeled_layers.pt \
        --dataset miniimagenet --split test

echo ""
echo "=================================================="
echo "Done:     $(date -Iseconds)"
ls -lh output/embeddings_*_layers.pt 2>/dev/null || true
echo "=================================================="
