#!/bin/bash
# End-to-end FGVC-Aircraft run: (1) download -> (2) extract DINOv2 embeddings ->
# (3) ablate the current winning EXCHANGEABLE pipeline
# (DINOv2 -> pool-fit PCA-d + within-cluster whiten -> unwhitened geodesic top-k
#  NCM -> Full CP -> optional MS-CS penalty).
#
# WHY FGVC-Aircraft (the harder, no-overlap benchmark):
#   Fine-grained 100-class set from the CLIP/CoOp transfer suite (used by Conf-OT,
#   Silva-Rodriguez CVPR'25). Aircraft *variants* are NOT ImageNet labels, so --
#   unlike CUB-200 (birds) or mini-ImageNet -- there is no DINOv2 pretraining-LABEL
#   overlap; the backbone does NOT pre-solve the task at 2-shot, so prediction sets
#   are non-trivial and methods actually separate. K=100 keeps "cal=200 = 2 shots/
#   class" directly comparable to the mini-ImageNet result we validated.
#
# DATA LAYOUT (mirrors mini-ImageNet's labeled/unlabeled convention):
#   labeled pool   = torchvision 'trainval' split (~6667, ~67/class)  -> cal+test carved here
#   unlabeled pool = torchvision 'test'     split (~3333, ~33/class)  -> PCA/whiten/MS-CS fit here (disjoint)
#
# THE ABLATION (winning-pipeline decomposition; each rung -> one results_<tag>.json):
#   plain_fcp           raw 768-d, no transform, no penalty          (does the geometry need reducing?)
#   pca{d}              + pool-fit PCA-d only                         (PCA-dim sweep: fine-grained may need >128, cf. CUB=512)
#   pca{d}_whiten       + within-cluster whitening                   (the full exchangeable transform)
#   pca{d}_whiten_mscs  + MS-CS similarity penalty (lambda sweep)     (the full winning pipeline)
#   ...repeated for NCM in {unwhitened_topk_mean (headline), unwhitened_topk_asym (open: ~8% smaller?)}
#
# Coverage stays valid (>= 1-alpha) at every rung -- this ablates set-size EFFICIENCY,
# not validity. Split = balanced_both (lit few-shot protocol, conservative); set
# SPLIT=random for the exact-validity (~0.90, tight) reference arm.
#
# Usage (from repo ROOT):
#   bash cluster/run_aircraft_ablation.sh                       # all 3 stages
#   STAGE=data     bash cluster/run_aircraft_ablation.sh        # download + extract only
#   STAGE=ablation bash cluster/run_aircraft_ablation.sh        # ablation only (embeddings already on disk)
#   PCA_DIMS="128 256" N_TRIALS=10 CAL_SIZES="200 400 800" bash cluster/run_aircraft_ablation.sh
#   SPLIT=random   bash cluster/run_aircraft_ablation.sh        # exact-validity reference arm
#   # RESUME: just relaunch -- rungs whose results_<tag>.json already exist are skipped.
#   #         FORCE=1 bash cluster/run_aircraft_ablation.sh      # ignore existing JSONs, redo all
#
# SLURM: submit with a GPU node, e.g.
#   sbatch --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=06:00:00 \
#          --job-name=aircraft_abl --output=cluster/logs/%x_%j.out \
#          --wrap "bash cluster/run_aircraft_ablation.sh"

set -euo pipefail

# ============================================================================
# Config -- override any of these via environment variables
# ============================================================================
STAGE="${STAGE:-all}"                  # all | data | ablation
MODEL="${MODEL:-dinov2-base}"
INPUT_SIZE="${INPUT_SIZE:-518}"        # native DINOv2-base; drop to 336 if VRAM < 16GB
BATCH_SIZE="${BATCH_SIZE:-64}"

DATA_DIR_LABELED="${DATA_DIR_LABELED:-data/aircraft}"
DATA_DIR_UNLABELED="${DATA_DIR_UNLABELED:-data/aircraft_unlabeled}"
EMB_LABELED="${EMB_LABELED:-embeddings_aircraft.pt}"
EMB_UNLABELED="${EMB_UNLABELED:-embeddings_aircraft_unlabeled.pt}"

# Ablation knobs
CAL_SIZES="${CAL_SIZES:-200 400 600 800 1000}"   # K=100 -> 2..10 shots/class (labeled trainval has ~67/class)
N_TRIALS="${N_TRIALS:-20}"
ALPHA="${ALPHA:-0.1}"
PCA_DIMS="${PCA_DIMS:-128 256 512}"              # fine-grained sweep; 512 stresses cluster-whiten on the ~3.3k pool (shrinkage-regularised, same regime that worked for miniImageNet @100/cluster)
PCA_DIM_MSCS="${PCA_DIM_MSCS:-128}"              # PCA dim used for the MS-CS rung (override after seeing the sweep)
N_CLUSTERS_WHITEN="${N_CLUSTERS_WHITEN:-100}"    # ~n_classes; lower it to trade locality for more samples/cluster at high PCA dim
NCMS="${NCMS:-unwhitened_topk_mean unwhitened_topk_asym}"
LAMBDAS="${LAMBDAS:-0.0 0.05 0.1}"
SPLIT="${SPLIT:-balanced_both}"                  # balanced_both (default) | random (exact-validity arm)
DEVICE="${DEVICE:-cuda}"                         # MS-CS penalty backend; no-penalty rungs use CPU evaluate()
OUT_DIR="${OUT_DIR:-output/aircraft_ablation}"
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

# ============================================================================
# Banner + env
# ============================================================================
mkdir -p output cluster/logs "$OUT_DIR"
echo "=================================================="
echo "Job:      run_aircraft_ablation  (STAGE=$STAGE)"
echo "Host:     $(hostname)"
echo "Started:  $(date -Iseconds)"
echo "PWD:      $(pwd)"
echo "=================================================="
echo "Model:      $MODEL @ input=$INPUT_SIZE  batch=$BATCH_SIZE"
echo "Cal sizes:  $CAL_SIZES   trials=$N_TRIALS   alpha=$ALPHA   split=$SPLIT"
echo "PCA dims:   $PCA_DIMS   (MS-CS rung @ PCA-$PCA_DIM_MSCS)   whiten-clusters=$N_CLUSTERS_WHITEN"
echo "NCMs:       $NCMS"
echo "Lambdas:    $LAMBDAS   device=$DEVICE"
echo "Out dir:    $OUT_DIR"
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
    echo "Activated venv: $SSL_CP_VENV"
fi

python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ============================================================================
# Stage 1+2: download + extract embeddings (labeled = trainval, unlabeled = test)
# ============================================================================
extract_pool () {     # $1 = data_dir   $2 = split(train|test)   $3 = output_name
    local ddir="$1" split="$2" oname="$3"
    mkdir -p "$ddir"
    if [ -z "$(ls -A "$ddir" 2>/dev/null)" ]; then
        echo "[download] aircraft --split $split -> $ddir"
        python src/download_datasets.py --dataset aircraft --output_dir "$ddir" --split "$split"
    else
        echo "[download] $ddir already populated -- skipping."
    fi
    if [ ! -f "output/$oname" ]; then
        echo "[extract ] $ddir -> output/$oname  ($MODEL @ $INPUT_SIZE)"
        python src/extract_features.py \
            --data_dir "$ddir" --output_name "$oname" \
            --model "$MODEL" --input_size "$INPUT_SIZE" --batch_size "$BATCH_SIZE"
    else
        echo "[extract ] output/$oname exists -- skipping."
    fi
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "data" ]; then
    extract_pool "$DATA_DIR_LABELED"   train "$EMB_LABELED"
    extract_pool "$DATA_DIR_UNLABELED" test  "$EMB_UNLABELED"
    ls -lh output/embeddings_aircraft*.pt 2>/dev/null || true
fi

# ============================================================================
# Stage 3: ablation -- one exchangeable_fcp_experiment.py invocation per rung
# ============================================================================
EMB_PATH="output/$EMB_LABELED"
UNL_PATH="output/$EMB_UNLABELED"

run_rung () {         # $1 = tag   (rest = extra flags appended verbatim)
    local tag="$1"; shift
    local out="$OUT_DIR/results_${tag}.json"
    # Resume: a rung's results_<tag>.json is written only on full completion, so
    # its presence means that rung is done -> skip it. Re-running the script thus
    # picks up where it left off. FORCE=1 re-runs everything.
    if [ -z "${FORCE:-}" ] && [ -f "$out" ]; then
        echo ">>> rung: $tag -- already done ($out) -- SKIP (FORCE=1 to redo)"
        return 0
    fi
    echo ""
    echo ">>> rung: $tag"
    python src/exchangeable_fcp_experiment.py \
        --embeddings_path "$EMB_PATH" \
        --unlabeled_path  "$UNL_PATH" \
        --cal_sizes $CAL_SIZES --n_trials "$N_TRIALS" --alpha "$ALPHA" \
        --split "$SPLIT" --n_clusters_whiten "$N_CLUSTERS_WHITEN" \
        --device "$DEVICE" --output_dir "$OUT_DIR" --tag "$tag" \
        "$@"
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "ablation" ]; then
    if [ ! -f "$EMB_PATH" ] || [ ! -f "$UNL_PATH" ]; then
        echo "ERROR: missing embeddings ($EMB_PATH / $UNL_PATH). Run STAGE=data first." >&2
        exit 1
    fi

    # Rung 0: plain FCP on raw 768-d (no transform, no penalty). NCM-agnostic-ish;
    # report both NCMs for a clean per-NCM baseline.
    for ncm in $NCMS; do
        run_rung "plain_fcp_${ncm}" --ncm "$ncm" --pca_dim 0 --whiten none
    done

    # Rungs 1+2: PCA-dim sweep, with and without within-cluster whitening.
    for ncm in $NCMS; do
        for d in $PCA_DIMS; do
            run_rung "pca${d}_${ncm}"        --ncm "$ncm" --pca_dim "$d" --whiten none
            run_rung "pca${d}_whiten_${ncm}" --ncm "$ncm" --pca_dim "$d" --whiten cluster
        done
    done

    # Rung 3: full winning pipeline -- PCA + cluster-whiten + MS-CS (lambda sweep),
    # at the chosen MS-CS PCA dim, for each NCM.
    for ncm in $NCMS; do
        run_rung "pca${PCA_DIM_MSCS}_whiten_mscs_${ncm}" \
            --ncm "$ncm" --pca_dim "$PCA_DIM_MSCS" --whiten cluster \
            --mscs --lambdas $LAMBDAS
    done
fi

echo "=================================================="
echo "Done:     $(date -Iseconds)"
echo "Results JSON -> $OUT_DIR/results_*.json"
ls -lh "$OUT_DIR"/results_*.json 2>/dev/null || true
echo "Quick read:  python - <<'PY'"
echo "  import json,glob; [print(f.split('results_')[-1][:-5],"
echo "    [(r['cal'],r['method'],round(r['cov'],3),round(r['sz'],2)) for r in json.load(open(f))['rows']])"
echo "    for f in sorted(glob.glob('$OUT_DIR/results_*.json'))]"
echo "PY"
echo "=================================================="
