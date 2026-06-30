#!/bin/bash
# Subject 2 (weekly goals 06-30..07-04): Aircraft class-similarity + Split-CP +
# a 2nd fine-grained dataset. Three things, in the hard-backbone (DINOv2-can't-
# separate) regime where FGVC-Aircraft is our scope limit:
#
#   (cs)     Class-similarity penalty sweep on aircraft with the NEW softmax-native
#            M_ij = cos(mu_i, mu_j) (prototype-cosine M, goal 1) vs the existing
#            cluster-M and centroid-M -- does coarse class structure help where the
#            features can't? prototype_softmax NCM, exactly exchangeable, GPU.
#   (split)  Split-CP baseline on aircraft (SCP-THR/APS/RAPS + SemiCP) for a full
#            FCP-vs-Split head-to-head in the hard regime (we only had FCP-family).
#   (data2)  Get a 2nd DINOv2-struggles MULTI-CLASS set as the labeled/unlabeled
#            pair: Stanford Cars (K=196, default; same fine-grained CLIP-suite as
#            aircraft) via DATASET2; cub200/flowers102 also supported (carved).
#   (scope2) Replicate ALL THREE on the 2nd set (scope-limit + cs + split).
#            Single-label throughout -- NO multi-label CP extension.
#
# Coverage stays valid (>= 1-alpha) at every rung -- these ablate set-size
# EFFICIENCY, not validity. Split = balanced_both (lit few-shot protocol).
#
# Usage (from repo ROOT):
#   bash cluster/run_aircraft_cs_splitcp.sh                 # all stages
#   STAGE=cs     bash cluster/run_aircraft_cs_splitcp.sh    # aircraft class-similarity only
#   STAGE=split  bash cluster/run_aircraft_cs_splitcp.sh    # aircraft Split-CP only
#   STAGE=data2  bash cluster/run_aircraft_cs_splitcp.sh    # extract+carve 2nd dataset only
#   STAGE=scope2 bash cluster/run_aircraft_cs_splitcp.sh    # 2nd-dataset replication only
#   DATASET2=cub200 N_TRIALS=50 bash cluster/run_aircraft_cs_splitcp.sh
#   # RESUME: relaunch -- rungs whose results JSON already exists are skipped (FORCE=1 to redo).
#
# SLURM:
#   sbatch --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=08:00:00 \
#          --job-name=air_cs_split --output=cluster/logs/%x_%j.out \
#          --wrap "bash cluster/run_aircraft_cs_splitcp.sh"

set -euo pipefail

# ============================================================================
# Config -- override via environment
# ============================================================================
STAGE="${STAGE:-all}"                    # all | cs | split | data2 | scope2
MODEL="${MODEL:-dinov2-base}"
INPUT_SIZE="${INPUT_SIZE:-518}"
BATCH_SIZE="${BATCH_SIZE:-64}"

DATASET2="${DATASET2:-stanford_cars}"    # 2nd DINOv2-struggles multi-class set:
                                         #   stanford_cars (K=196, native train/test) |
                                         #   cub200 | flowers102 (merge-only -> carve)
DATA2_PER_CLASS="${DATA2_PER_CLASS:-0}"  # images/class to download (0/empty = all)
UNLABELED_PER_CLASS="${UNLABELED_PER_CLASS:-30}"  # carve-only datasets: holdout/class

CAL_SIZES="${CAL_SIZES:-200 400 800 1200}"        # K=100 -> 2..12 shots/class
N_TRIALS="${N_TRIALS:-50}"               # paper-quality; MS-CS GPU path makes this cheap
SPLIT_N_TRIALS="${SPLIT_N_TRIALS:-20}"   # Split-CP head-to-head (FCP transductive is heavier)
ALPHA="${ALPHA:-0.1}"
PCA_DIM="${PCA_DIM:-128}"                # aircraft: PCA payoff ~vanishes; 128 matches ablation
LAMBDAS="${LAMBDAS:-0.0 0.05 0.1 0.2 0.4}"
# Default to the ESTABLISHED (old) cluster MS-CS. The new score-aligned M
# (prototype-cosine / centered_cosine) is being finalised in a separate session;
# until then opt in explicitly, e.g. SIMILARITIES="prototype cluster".
SIMILARITIES="${SIMILARITIES:-cluster}"   # M variant(s): cluster (old) | prototype | centroid
DEVICE="${DEVICE:-cuda}"
TEST_SIZE_SPLIT="${TEST_SIZE_SPLIT:-1000}"
FORCE="${FORCE:-0}"

OUT_CS="${OUT_CS:-output/mscs_softmax}"          # mscs_softmax_experiment default
OUT_SPLIT="${OUT_SPLIT:-output/semicp_experiments}"
OUT_SCOPE="${OUT_SCOPE:-output/fca_family_cluster}"
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

mkdir -p output cluster/logs "$OUT_CS" "$OUT_SPLIT" "$OUT_SCOPE"
echo "=================================================="
echo "Job:      run_aircraft_cs_splitcp  (STAGE=$STAGE)"
echo "Host:     $(hostname)   Started: $(date -Iseconds)"
echo "PWD:      $(pwd)"
echo "2nd set:  $DATASET2   cal=$CAL_SIZES  trials=$N_TRIALS  M={$SIMILARITIES}"
echo "Lambdas:  $LAMBDAS   device=$DEVICE   pca=$PCA_DIM"
echo "=================================================="

if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"; echo "Activated venv: $SSL_CP_VENV"
fi
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

skip_if_done () {   # $1 = results path; honours FORCE=1
    [ "$FORCE" != "1" ] && [ -f "$1" ] && { echo "[skip] $1 exists (FORCE=1 to redo)"; return 0; }
    return 1
}

# ============================================================================
# (cs) class-similarity penalty sweep -- one results_<ds>_<sim>.json per M variant
# ============================================================================
run_cs () {     # $1 = dataset
    local ds="$1"
    for sim in $SIMILARITIES; do
        local tag; [ "$sim" = "cluster" ] && tag="" || tag="_$sim"
        local out="$OUT_CS/results_${ds}${tag}.json"
        if skip_if_done "$out"; then continue; fi
        echo "[cs] $ds  similarity=$sim"
        python src/mscs_softmax_experiment.py --dataset "$ds" --device "$DEVICE" \
            --ncms prototype_softmax --similarity "$sim" \
            --cal_sizes $CAL_SIZES --lambdas $LAMBDAS --n_trials "$N_TRIALS" \
            --pca_dim "$PCA_DIM" --alpha "$ALPHA" --output_dir "$OUT_CS"
    done
}

# ============================================================================
# (split) FCP-vs-SplitCP head-to-head
# ============================================================================
run_split () {  # $1 = dataset
    local ds="$1"
    local out="$OUT_SPLIT/${ds}_semicp_results.json"
    if skip_if_done "$out"; then return 0; fi
    echo "[split] $ds  FCP vs SCP/SemiCP"
    python src/semicp_experiment.py --datasets "$ds" \
        --cal_sizes $CAL_SIZES --test_size "$TEST_SIZE_SPLIT" \
        --n_trials "$SPLIT_N_TRIALS" --pca_dim "$PCA_DIM" \
        --output_dir "$OUT_SPLIT"
}

# ============================================================================
# (scope) prototype-vs-geodesic set-size scope limit (fca_family)
# ============================================================================
run_scope () {  # $1 = dataset
    local ds="$1"
    local out="$OUT_SCOPE/results_${ds}.json"
    if skip_if_done "$out"; then return 0; fi
    echo "[scope] $ds  prototype vs geodesic NCM set sizes"
    python src/fca_family_cluster_experiment.py --datasets "$ds" --device "$DEVICE" \
        --cal_sizes $CAL_SIZES --n_trials "$N_TRIALS" --pca_dim "$PCA_DIM" \
        --output_dir "$OUT_SCOPE" --plot
}

# ============================================================================
# (data2) get the 2nd dataset as the labeled/unlabeled embeddings pair.
#   stanford_cars: native train/test -> extract two pools (like aircraft).
#   flowers102 / cub200: merge-only -> extract all + carve a disjoint pool.
# ============================================================================
download_extract () {   # $1=dataset $2=split(train|test|"") $3=output_name $4=data_subdir
    local ds="$1" split="$2" oname="$3" ddir="data/$4"
    local dl_args=(--dataset "$ds" --output_dir "$ddir")
    [ -n "$split" ] && dl_args+=(--split "$split")
    [ -n "$DATA2_PER_CLASS" ] && [ "$DATA2_PER_CLASS" != "0" ] && dl_args+=(--num_per_class "$DATA2_PER_CLASS")
    if [ -z "$(ls -A "$ddir" 2>/dev/null)" ]; then
        echo "[data2] download $ds ${split:+--split $split} -> $ddir"
        python src/download_datasets.py "${dl_args[@]}"
    fi
    if [ ! -f "output/$oname" ]; then
        echo "[data2] extract $ddir -> output/$oname ($MODEL @ $INPUT_SIZE)"
        python src/extract_features.py --data_dir "$ddir" --output_name "$oname" \
            --model "$MODEL" --input_size "$INPUT_SIZE" --batch_size "$BATCH_SIZE"
    fi
}

run_data2 () {
    local ds="$DATASET2"
    if [ -f "output/embeddings_${ds}.pt" ] && [ -f "output/embeddings_${ds}_unlabeled.pt" ] \
       && [ "$FORCE" != "1" ]; then
        echo "[data2] output/embeddings_${ds}{,_unlabeled}.pt exist -- skipping."
        return 0
    fi
    case "$ds" in
        stanford_cars|miniimagenet|aircraft)   # native train/test -> two pools
            download_extract "$ds" train "embeddings_${ds}.pt"           "${ds}"
            download_extract "$ds" test  "embeddings_${ds}_unlabeled.pt" "${ds}_unlabeled"
            ;;
        *)                                       # merge-only -> extract all + carve
            download_extract "$ds" "" "embeddings_${ds}_all.pt" "${ds}"
            echo "[data2] carve disjoint unlabeled pool ($UNLABELED_PER_CLASS/class)"
            python src/carve_unlabeled.py \
                --in "output/embeddings_${ds}_all.pt" \
                --out_labeled "output/embeddings_${ds}.pt" \
                --out_unlabeled "output/embeddings_${ds}_unlabeled.pt" \
                --unlabeled_per_class "$UNLABELED_PER_CLASS"
            ;;
    esac
}

# ============================================================================
# Dispatch
# ============================================================================
case "$STAGE" in
    cs)     run_cs aircraft ;;
    split)  run_split aircraft ;;
    data2)  run_data2 ;;
    scope2) run_data2; run_scope "$DATASET2"; run_cs "$DATASET2"; run_split "$DATASET2" ;;
    all)
        run_cs aircraft
        run_split aircraft
        run_scope aircraft
        run_data2
        run_scope "$DATASET2"
        run_cs "$DATASET2"
        run_split "$DATASET2"
        ;;
    *) echo "unknown STAGE='$STAGE'"; exit 1 ;;
esac

echo "=================================================="
echo "Done: $STAGE   $(date -Iseconds)"
echo "Results: $OUT_CS (class-similarity), $OUT_SPLIT (Split-CP), $OUT_SCOPE (scope-limit)"
echo "=================================================="
