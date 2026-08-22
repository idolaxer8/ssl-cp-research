#!/usr/bin/env bash
# R1 headline v2: frozen champion vs SplitCP/CV+/SemiCP, 5 datasets,
# 50 trials, alphas {0.1, 0.05}. Resumable: re-run the same command after a
# cutoff and completed cell-trials are skipped (see headline_experiment.py).
#
#   nohup bash cluster/run_headline.sh > output/headline/run.log 2>&1 &
#
# cub200 embeddings are read from output/pca_pilots/heldout_data (the local
# 336px carved pair) -- sync them to the pod alongside from_cluster/embeddings
# or override --cub_dir.
set -e
cd "$(dirname "$0")/.."
mkdir -p output/headline
for ds in cifar100 miniimagenet cub200 aircraft stanford_cars; do
    echo "=== headline: $ds ==="
    python src/headline_experiment.py --dataset "$ds" --device cuda \
        --data_dir output/from_cluster/embeddings --n_trials 50
done
