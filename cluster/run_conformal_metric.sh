#!/bin/bash
# G1 conformal-metric-learning runs on the Run:AI pod (high-trial confirm).
#
# One-time prereqs on the pod:
#   source /storage/ido/venvs/ssl-cp/bin/activate
#   cd /storage/ido/ssl-cp/ssl-cp-research
#   git fetch origin && git checkout worktree-conformal-metric && git pull
#
# Usage (from repo root, inside tmux -- no scheduler on this pod):
#   bash cluster/run_conformal_metric.sh
#   DATASETS="cifar100 aircraft" PHASES="rung1 benchmark" bash cluster/run_conformal_metric.sh
#   N_TRIALS=10 bash cluster/run_conformal_metric.sh          # quick pass
#
# cub200 needs the locally-carved heldout pair uploaded first (local-only data):
#   scp output/pca_pilots/heldout_data/embeddings_cub200*.pt \
#     root@<pod>:/storage/ido/ssl-cp/ssl-cp-research/output/pca_pilots/heldout_data/
# then: DATASETS=cub200 bash cluster/run_conformal_metric.sh
#
# Results land in $SSL_CP_MAIN/output/conformal_metric/{rung1,benchmark,validity}
# -- scp back into local output/from_cluster/conformal_metric/ when done.
set -euo pipefail

export SSL_CP_MAIN="${SSL_CP_MAIN:-$(pwd)}"
DATASETS="${DATASETS:-cifar100 aircraft miniimagenet}"
PHASES="${PHASES:-rung1 benchmark validity}"
N_TRIALS="${N_TRIALS:-50}"
OUT="$SSL_CP_MAIN/output/conformal_metric"
mkdir -p "$OUT"

# Embeddings layout shim: the code expects output/from_cluster/embeddings/;
# on the pod the extract scripts historically wrote flat output/embeddings_*.pt.
# Symlink whichever flat files exist into the expected directory.
EMB="$SSL_CP_MAIN/output/from_cluster/embeddings"
mkdir -p "$EMB"
for f in "$SSL_CP_MAIN"/output/embeddings_*.pt; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  [ -e "$EMB/$b" ] || ln -s "$f" "$EMB/$b"
done
echo "embeddings visible in $EMB:"
ls "$EMB" | sed 's/^/  /'

for phase in $PHASES; do
  extra=""
  if [ "$phase" = "benchmark" ]; then
    extra="--ablation --contaminated"   # stage/factor ablation + leak control
  fi
  echo "== phase $phase (datasets: $DATASETS, n_trials=$N_TRIALS) =="
  python -u src/conformal_metric_experiment.py --phase "$phase" \
    --datasets $DATASETS --n_trials "$N_TRIALS" --device cuda $extra \
    2>&1 | tee -a "$OUT/cluster_${phase}.log"
done
echo "ALL PHASES DONE -> $OUT"
