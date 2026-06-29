#!/usr/bin/env bash
# Geometry-conditional Full CP — Stage-2 cluster sweep (run on the 48GB RTX 6000 Ada pod).
#
# Mondrian Full CP over label-free geometric strata closes the geometric coverage gap
# (see docs/findings.md §2c, docs/novelty_pilots_findings.md). This sweep reproduces
# the local Stage-0/1 result at paper scale: 30 trials, larger cal (incl. 1600, which
# OOM'd on the 4 GB laptop), all robustness axes, both splits, with the confidence
# control + RLCP arms.
#
# Arms per run (the runner adds them automatically): global FCP, global-static
# (== global at O(1/n), a design-C sanity check), Mondrian-geometry (primary),
# confidence-conditioning control (balanced only), RLCP (continuous secondary).
#
# Strategy: ONE-AXIS-AT-A-TIME around the center config (cal=800, alpha=0.1, G=5,
# density) — this is the robustness story without the 100+-run full crossing.
# Resumable: skips a config whose results JSON already exists.
#
# Embeddings expected in $EMB_DIR as embeddings_<ds>.pt + embeddings_<ds>_unlabeled.pt
# (keys 'embeddings','labels'), matching the other cluster scripts (--data_dir output).
#
# Usage on the pod (the code must be on the branch first):
#   git fetch origin && git checkout worktree-novelty-pilots && git pull
#   EMB_DIR=output bash cluster/run_geometric_conditional_cluster.sh
#   # knobs: TRIALS=50  RLCP=""(skip slow RLCP arm)  OUT=output/geo_cond
set -uo pipefail                      # NOT -e: one failed config must not kill the sweep
cd "$(dirname "$0")/.."               # repo root

PY=${PY:-python}
EMB_DIR=${EMB_DIR:-output}            # cluster embeddings dir (override if elsewhere)
OUT=${OUT:-output/geometric_conditional_cluster}
TRIALS=${TRIALS:-30}
RLCP=${RLCP:---rlcp}                  # set RLCP="" to skip the (slow, NumPy-loop) RLCP arm
mkdir -p "$OUT"

# R DATASET CAL ALPHA G COVARIATE  -- resumable single run (both splits + controls)
R(){
  local DS=$1 CAL=$2 A=$3 G=$4 COV=$5
  local f="$OUT/$DS/stage0_${COV}_G${G}_cal${CAL}_a${A}.json"
  if [ -f "$f" ]; then echo "[skip] exists: $f"; return; fi
  echo "===== $DS cal=$CAL alpha=$A G=$G $COV (trials=$TRIALS) ====="
  $PY src/geometric_conditional_cp_experiment.py \
      --dataset "$DS" --cal "$CAL" --alpha "$A" --n_strata "$G" --covariate "$COV" \
      --embeddings_dir "$EMB_DIR" --out_dir "$OUT" --n_trials "$TRIALS" $RLCP --plot \
      || echo "[FAIL] $DS cal=$CAL alpha=$A G=$G $COV"
}

for DS in cifar100 miniimagenet; do
  # cal sweep (center alpha=0.1, G=5, density); cal=1600 is the new cluster-only point
  for CAL in 400 800 1600; do R "$DS" "$CAL" 0.1 5 density; done
  # alpha sweep (cal=800, G=5, density)
  R "$DS" 800 0.05 5 density
  R "$DS" 800 0.2  5 density
  # G sweep (cal=800, alpha=0.1, density)
  R "$DS" 800 0.1 3  density
  R "$DS" 800 0.1 10 density
  # covariate (cal=800, alpha=0.1, G=5, LID)
  R "$DS" 800 0.1 5 lid
done

# FGVC-Aircraft scope-boundary confirmation (K=100; expect ~no gap, Mondrian a no-op)
for A in 0.05 0.1 0.2; do R aircraft 800 "$A" 5 density; done
for CAL in 400 800;    do R aircraft "$CAL" 0.1 5 density; done

echo "GEOMETRIC_CONDITIONAL_CLUSTER_DONE -> $OUT"
echo "Pull results back, then: python src/plot_geometric_robustness.py --root $OUT"
