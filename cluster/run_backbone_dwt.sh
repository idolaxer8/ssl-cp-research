#!/usr/bin/env bash
# Backbone transfer study (C5): extract CLIP + ssl-resnet50 (CNN) embeddings
# and run the PAPER pipeline (headline_experiment.py: champion FRCP vs
# splitcp/cvplus/semicp) on each backbone, alongside the dinov2 anchor rows.
# Second backbone = ssl-resnet50 (user call 2026-08-30): a CNN, so the table
# shows the pipeline transfers across ARCHITECTURE, not just ViT objectives.
#
#   nohup bash cluster/run_backbone_dwt.sh > cluster/logs/backbone_dwt.log 2>&1 &
#   # or inside tmux:  tmux new -s bb ; bash cluster/run_backbone_dwt.sh
#
# Prereqs on the pod (all under /storage/ido, which persists):
#   - venv at $SSL_CP_VENV (default /storage/ido/venvs/ssl-cp), torch 2.7.1+cu118
#   - raw ImageFolders  data/<ds>  (labeled) and  data/<ds>_unlabeled  (pool)
#     for each dataset -- the SAME folders the dinov2 rows used, so backbones
#     are compared on identical images. cifar100 auto-downloads if missing;
#     aircraft's pool is a carve that must already be on the pod.
#   - the Run:AI GPU is a ~5 GB fractional slice (like the laptop), so batch 32
#     at NATIVE resolution (memo: upscaling hurt dinov3 -- never override res).
#
# Idempotent: any (backbone,dataset) whose .pt already exists is skipped, so
# re-running resumes and pre-extracted CLIP embeddings on the pod are reused.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PATH="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    echo "venv: $VENV_PATH"
fi
python -c "import torch,timm; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# Backbone embeddings already copied here (dinov2/clip-base/clip-large/dinov3/
# mae for cifar100+aircraft, labeled+unlabeled, 2026-08-30). Only ssl-resnet50
# is missing, so the idempotent skips below extract just that one.
EMB_DIR="${EMB_DIR:-output/from_cluster/backbone_dwt_emb}"
OUT_DIR="${OUT_DIR:-output/backbone_headline}"
ON="${EMB_DIR#output/}"          # output_name subpath (extract_features prepends output/)
mkdir -p "$EMB_DIR" "$OUT_DIR" cluster/logs

BATCH="${BATCH_SIZE:-32}"
# cifar100/aircraft already have clip+dinov3+mae in EMB_DIR; miniimagenet and
# cifar10 (added 2026-08-30) have only the dinov2 anchor, so clip-base,
# clip-large and ssl-resnet50 are extracted for them (heavy for mini: 50k
# labeled images x 3 backbones).
DATASETS="${DATASETS:-cifar100 aircraft miniimagenet cifar10}"
NTRIALS="${NTRIALS:-50}"
# where the pre-existing dinov2 finals (matched-518) live, to seed the anchor
# row for datasets whose dinov2 file is not already in EMB_DIR.
ANCHOR_DIR="${ANCHOR_DIR:-output/from_cluster/embeddings}"

# ensure the dinov2 anchor (no-suffix labeled+unlabeled) is in EMB_DIR
ensure_anchor() {
    local ds="$1"
    for part in "" "_unlabeled"; do
        local dst="$EMB_DIR/embeddings_${ds}${part}.pt"
        local src="$ANCHOR_DIR/embeddings_${ds}${part}.pt"
        if [ ! -f "$dst" ]; then
            if [ -f "$src" ]; then
                echo "[anchor] copy dinov2 $ds$part <- $ANCHOR_DIR"
                cp "$src" "$dst"
            else
                echo "[MISS] dinov2 anchor $src not found -- $ds dinov2 row will be skipped"
            fi
        fi
    done
}

# extract one (preset, file-suffix, dataset) at native res if not already done
extract_one() {
    local preset="$1" suf="$2" ds="$3"
    local lab="$EMB_DIR/embeddings_${ds}${suf}.pt"
    local unl="$EMB_DIR/embeddings_${ds}_unlabeled${suf}.pt"
    if [ -f "$lab" ] && [ -f "$unl" ]; then
        echo "[skip] $preset / $ds already extracted"; return
    fi
    if [ ! -d "data/${ds}" ]; then
        echo "[MISS] data/${ds} not found -- download it first, e.g."
        echo "       python src/download_datasets.py --dataset ${ds} --output_dir data/${ds} --num_per_class N"
        return
    fi
    echo "[extract] $preset / $ds  (batch=$BATCH, native res)"
    [ -f "$lab" ] || python src/extract_features.py --data_dir "data/${ds}" \
        --model "$preset" --batch_size "$BATCH" \
        --output_name "${ON}/embeddings_${ds}${suf}.pt"
    if [ ! -d "data/${ds}_unlabeled" ]; then
        echo "[MISS] data/${ds}_unlabeled (pool) not found -- extract it once the"
        echo "       unlabeled carve is on the pod; the experiment needs the pool."
        return
    fi
    [ -f "$unl" ] || python src/extract_features.py --data_dir "data/${ds}_unlabeled" \
        --model "$preset" --batch_size "$BATCH" \
        --output_name "${ON}/embeddings_${ds}_unlabeled${suf}.pt"
}

for ds in $DATASETS; do
    ensure_anchor "$ds"
    extract_one clip-base    _clip-base    "$ds"
    extract_one clip-large   _clip-large   "$ds"
    extract_one ssl-resnet50 _ssl-resnet50 "$ds"
done

# === backbone transfer, PAPER pipeline ====================================
# The SAME driver as Table 2: champion FRCP (pca128 -> lw_cluster whiten ->
# qe-post -> prototype-softmax FULL CP) vs splitcp / cvplus / semicp, via
# headline_experiment.py --emb_suffix. Results land per backbone in
# $OUT_DIR/<backbone>/results_<ds>.json (same schema as output/headline),
# so make_headline_tables.py / make_covgap_table.py can tabulate them.
# (backbone_dwt_experiment.py is a DIFFERENT instrument -- the qe-gate
# transfer diagnostic, raw/wt/qe_wt under cosine split CP -- not this table.)
echo "=== backbone transfer: paper pipeline, $NTRIALS trials, shots 2/4/8/14 ==="
# BACKBONES: which rows to run. dinov2 is EXCLUDED by default -- the anchor
# numbers already exist in output/headline (same driver, seeds and protocol;
# shots 2-14 is a superset), so re-running it measures nothing new. Add
# "dinov2" explicitly only if you want a fresh anchor in this output tree.
BACKBONES="${BACKBONES:-clip-base clip-large ssl-resnet50}"
frozen_ncm_for() {   # champion NCM per dataset, as in the paper
    case "$1" in
        aircraft|stanford_cars) echo unwhitened_topk_asym ;;
        *)                      echo prototype_softmax ;;
    esac
}
for ds in $DATASETS; do
    ncm="$(frozen_ncm_for "$ds")"
    for name in $BACKBONES; do
        if [ "$name" = "dinov2" ]; then suf=""; else suf="_$name"; fi
        if [ ! -f "$EMB_DIR/embeddings_${ds}${suf}.pt" ]; then
            echo "[skip-run] $name/$ds: embeddings_${ds}${suf}.pt not present"
            continue
        fi
        out="$OUT_DIR/$name"; mkdir -p "$out"
        if [ -f "$out/results_${ds}.json" ]; then
            echo "[skip-run] $name/$ds: results_${ds}.json already exists"
            continue
        fi
        echo "=== headline: $name / $ds (ncm=$ncm) ==="
        python src/headline_experiment.py --dataset "$ds" --emb_suffix "$suf" \
            --data_dir "$EMB_DIR" --output_dir "$out" --frozen_ncm "$ncm" \
            --n_trials "$NTRIALS" --shots 2 4 8 14 --device cuda \
            > "cluster/logs/headline_${name}_${ds}.log" 2>&1 \
            || echo "  [warn] $name/$ds exited $?"
    done
done
echo "DONE $(date -Iseconds)  ->  $OUT_DIR/<backbone>/results_<ds>.json"
echo "dinov2 anchor rows: reuse output/headline/results_<ds>.json (same protocol)."
