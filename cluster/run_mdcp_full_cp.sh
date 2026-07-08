#!/bin/bash
# Full-CP MDCP (pilot D) ladder on the cluster GPU (48GB -- no VRAM cliff:
# large candidate chunks + a dense pool-false cloud become affordable).
#
# Strategy per user 2026-07-08: relaxed setting FIRST (balanced cal=800),
# then gradually smaller / more extreme cal. Each rung is skipped if its
# results JSON already exists (FORCE=1 to redo) -- crash-safe stepwise.
#
# Usage (from repo ROOT):
#   bash cluster/run_mdcp_full_cp.sh                 # full ladder
#   RUNGS="bal800" bash cluster/run_mdcp_full_cp.sh  # one rung
#   N_TRIALS=20 N_TEST=200 bash cluster/run_mdcp_full_cp.sh
#
# SLURM-ish: nohup bash cluster/run_mdcp_full_cp.sh > cluster/logs/mdcp_fullcp.log 2>&1 &

set -euo pipefail

EMB_DIR="${EMB_DIR:-output/from_cluster/embeddings}"
DIMS="${DIMS:-proto:final__pca128_cw proto:final__pca32_cw}"
N_TRIALS="${N_TRIALS:-20}"
N_TEST="${N_TEST:-150}"
POOL_SUB="${POOL_SUB:-100000}"
MEM_BUDGET="${MEM_BUDGET:-8.0e9}"        # 48GB card: allow ~8GB distance tensors
RUNGS="${RUNGS:-bal800 bal400 bal200 rand200 rand400}"
FORCE="${FORCE:-0}"
SSL_CP_VENV="${SSL_CP_VENV:-/storage/ido/venvs/ssl-cp}"

mkdir -p cluster/logs
if [ -n "$SSL_CP_VENV" ] && [ -d "$SSL_CP_VENV" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$SSL_CP_VENV/bin/activate"
fi
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

run_rung () {
    local name="$1" split="$2" cal="$3" trials="$4"
    local outdir="output/mdcp_pool_pilot/full_cp_${name}"
    if [ -f "$outdir/mdcp_full_cp_results.json" ] && [ "$FORCE" != "1" ]; then
        echo "SKIP rung $name (results exist)"
        return 0
    fi
    echo "===== RUNG $name ($split cal=$cal, $trials trials x $N_TEST test) ====="
    # shellcheck disable=SC2086
    python -u src/mdcp_full_cp.py \
        --embeddings_path "$EMB_DIR/embeddings_cifar100_layers.pt" \
        --unlabeled_path  "$EMB_DIR/embeddings_cifar100_unlabeled_layers.pt" \
        --dims $DIMS \
        --cal_sizes "$cal" --splits "$split" \
        --n_trials "$trials" --n_test "$N_TEST" \
        --pool_subsample "$POOL_SUB" --mem_budget "$MEM_BUDGET" \
        --dtype float32 --output_dir "$outdir"
}

for rung in $RUNGS; do
    case "$rung" in
        bal800)  run_rung bal800  balanced_both 800 "$N_TRIALS" ;;
        bal400)  run_rung bal400  balanced_both 400 "$N_TRIALS" ;;
        bal200)  run_rung bal200  balanced_both 200 "$N_TRIALS" ;;
        rand200) run_rung rand200 random        200 "$N_TRIALS" ;;
        rand400) run_rung rand400 random        400 "$N_TRIALS" ;;
        *) echo "unknown rung $rung"; exit 1 ;;
    esac
done
echo "ALL RUNGS DONE $(date -Iseconds)"
