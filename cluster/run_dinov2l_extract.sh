#!/usr/bin/env bash
# DINOv2 ViT-L/14 extraction (+ optional headline rows) for the sec5-reorg
# large-encoder question (user call 2026-09-06): does the DINOv2 large
# encoder strengthen the Table 2 claim the way CLIP ViT-L strengthened the
# appendix one? Datasets = the 09-06 paper roster: cifar10, cifar100,
# eurosat. Same conventions as cluster/run_backbone_dwt.sh (suffix naming,
# idempotent skips, native resolution).
#
#   nohup bash cluster/run_dinov2l_extract.sh > cluster/logs/dinov2l.log 2>&1 &
#   # or inside tmux:  tmux new -s d2l ; bash cluster/run_dinov2l_extract.sh
#
# Prereqs on the pod (all under /storage/ido, which persists):
#   - venv at $SSL_CP_VENV (default /storage/ido/venvs/ssl-cp)
#   - the SAME raw ImageFolders the existing rows used:
#       data/<ds>            labeled
#       data/<ds>_unlabeled  10k held-out pool
#     For eurosat the labeled folder may be the carve data/eurosat_subset
#     (that is what the local dinov2-base finals used); this script falls
#     back to data/<ds>_subset{,_unlabeled} automatically and says so.
#   - dinov2-large runs at NATIVE 518 px (memo: never override res), so on
#     the ~5 GB fractional GPU slice the batch default is 8, not 32. Bump
#     BATCH_SIZE if you land a full GPU.
#
# Idempotent: existing .pt / results files are skipped, so re-running
# resumes. Set RUN_HEADLINE=0 to extract only.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PATH="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    echo "venv: $VENV_PATH"
fi
python -c "import torch,timm; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

EMB_DIR="${EMB_DIR:-output/from_cluster/backbone_dwt_emb}"
OUT_DIR="${OUT_DIR:-output/backbone_headline/dinov2-large}"
ON="${EMB_DIR#output/}"          # extract_features prepends output/
mkdir -p "$EMB_DIR" "$OUT_DIR" cluster/logs

PRESET="dinov2-large"            # vit_large_patch14_dinov2.lvd142m @ 518
SUF="_dinov2-large"
BATCH="${BATCH_SIZE:-8}"         # ViT-L @ 518 on a ~5 GB slice
DATASETS="${DATASETS:-cifar10 cifar100 eurosat}"
NTRIALS="${NTRIALS:-50}"
SHOTS="${SHOTS:-2 4 8 12}"       # sec5-reorg grid (tables cap at 12)
RUN_HEADLINE="${RUN_HEADLINE:-1}"

# resolve the labeled / pool ImageFolder for a dataset, allowing the
# eurosat-style carve layout data/<ds>_subset/{labeled,pool}.
# NEVER return a carve ROOT: ImageFolder would take the folder names
# labeled/pool as the two classes (this exact bug produced a broken
# 2-class eurosat file on 09-06; delete such a .pt before re-running).
resolve_dir() {                  # $1 = ds, $2 = "" | "_unlabeled"
    local ds="$1" part="$2" cands
    if [ "$part" = "_unlabeled" ]; then
        cands="data/${ds}_unlabeled data/${ds}_subset_unlabeled data/${ds}_subset/pool data/${ds}/pool"
    else
        cands="data/${ds} data/${ds}_subset/labeled data/${ds}/labeled"
    fi
    for cand in $cands; do
        if [ -d "$cand" ]; then
            if [ -d "$cand/labeled" ] || [ -d "$cand/pool" ]; then
                echo "[guard] $cand is a carve root, not an ImageFolder -- skipped" >&2
                continue
            fi
            echo "$cand"; return
        fi
    done
    echo ""
}

extract_one() {
    local ds="$1"
    local lab="$EMB_DIR/embeddings_${ds}${SUF}.pt"
    local unl="$EMB_DIR/embeddings_${ds}_unlabeled${SUF}.pt"
    if [ -f "$lab" ] && [ -f "$unl" ]; then
        echo "[skip] $PRESET / $ds already extracted"; return
    fi
    local lab_dir unl_dir
    lab_dir="$(resolve_dir "$ds" "")"
    unl_dir="$(resolve_dir "$ds" "_unlabeled")"
    if [ -z "$lab_dir" ]; then
        echo "[MISS] no labeled ImageFolder for $ds (tried data/${ds},"
        echo "       data/${ds}_subset/labeled, data/${ds}/labeled) -- put the"
        echo "       SAME folder the dinov2-base rows used on the pod, re-run."
        return
    fi
    echo "[extract] $PRESET / $ds  labeled=$lab_dir  (batch=$BATCH, 518 px)"
    [ -f "$lab" ] || python src/extract_features.py --data_dir "$lab_dir" \
        --model "$PRESET" --batch_size "$BATCH" \
        --output_name "${ON}/embeddings_${ds}${SUF}.pt"
    if [ -z "$unl_dir" ]; then
        echo "[MISS] no pool ImageFolder for $ds (tried data/${ds}_unlabeled,"
        echo "       data/${ds}_subset_unlabeled, data/${ds}_subset/pool,"
        echo "       data/${ds}/pool) -- the experiment needs the 10k pool;"
        echo "       extract it once the carve is on the pod."
        return
    fi
    echo "[extract] $PRESET / $ds  pool=$unl_dir"
    [ -f "$unl" ] || python src/extract_features.py --data_dir "$unl_dir" \
        --model "$PRESET" --batch_size "$BATCH" \
        --output_name "${ON}/embeddings_${ds}_unlabeled${SUF}.pt"
}

for ds in $DATASETS; do
    extract_one "$ds"
done

# === headline rows (same driver + protocol as Table 2) =====================
# Champion FRCP vs splitcp/cvplus/semicp, 50 trials, sec5-reorg shot grid.
# Results land in $OUT_DIR/results_<ds>.json (schema identical to
# output/headline), tabulated locally via make_headline_tables.py.
if [ "$RUN_HEADLINE" = "1" ]; then
    echo "=== headline: dinov2-large, $NTRIALS trials, shots $SHOTS ==="
    for ds in $DATASETS; do
        if [ ! -f "$EMB_DIR/embeddings_${ds}${SUF}.pt" ] || \
           [ ! -f "$EMB_DIR/embeddings_${ds}_unlabeled${SUF}.pt" ]; then
            echo "[skip-run] $ds: dinov2-large embeddings incomplete"
            continue
        fi
        if [ -f "$OUT_DIR/results_${ds}.json" ]; then
            echo "[skip-run] $ds: results_${ds}.json already exists"
            continue
        fi
        echo "=== headline: dinov2-large / $ds ==="
        # shellcheck disable=SC2086
        python src/headline_experiment.py --dataset "$ds" --emb_suffix "$SUF" \
            --data_dir "$EMB_DIR" --output_dir "$OUT_DIR" \
            --frozen_ncm prototype_softmax \
            --n_trials "$NTRIALS" --shots $SHOTS --device cuda \
            > "cluster/logs/headline_dinov2-large_${ds}.log" 2>&1 \
            || echo "  [warn] dinov2-large/$ds exited $?"
    done
fi

echo "DONE $(date -Iseconds)"
echo "Sync back to the laptop:"
echo "  $EMB_DIR/embeddings_*${SUF}.pt        -> output/from_cluster/backbone_dwt_emb/"
echo "  $OUT_DIR/results_*.json (+checkpoints) -> output/backbone_headline/dinov2-large/"
echo "Then tabulate/compare locally, e.g.:"
echo "  python src/make_headline_tables.py   # add --headline_dir output/backbone_headline/dinov2-large to inspect as main-table candidate"
