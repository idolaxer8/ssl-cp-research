#!/bin/bash
# R1 headline run (ICLR 2027 Table 2): 4 main datasets, shots 2-14, 50 trials.
# RESUMABLE: each dataset checkpoints after every trial
# (output/r1_headline/r1_checkpoint_<ds>.json). If the pod dies, just
# re-run this script -- completed work is skipped, partial cells continue.
#
# Usage (on the cluster pod, from repo root):
#   bash cluster/run_r1_headline.sh                # all 4 datasets
#   bash cluster/run_r1_headline.sh aircraft       # one dataset
#   nohup bash cluster/run_r1_headline.sh > output/r1_headline/run.log 2>&1 &
set -u  # no -e: one dataset failing must not kill the sequence

DATA_DIR=${DATA_DIR:-output/from_cluster/embeddings}
OUT_DIR=${OUT_DIR:-output/r1_headline}
N_TRIALS=${N_TRIALS:-50}
SHOTS=${SHOTS:-"2 4 6 8 10 12 14"}
DATASETS=${1:-"cifar100 miniimagenet aircraft stanford_cars"}

mkdir -p "$OUT_DIR"
echo "R1 headline: datasets=[$DATASETS] shots=[$SHOTS] trials=$N_TRIALS"

for ds in $DATASETS; do
    echo ""
    echo "############ R1 $ds  $(date) ############"
    python src/r1_headline_experiment.py \
        --dataset "$ds" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUT_DIR" \
        --shots $SHOTS \
        --n_trials "$N_TRIALS" \
        --device cuda
    echo "############ R1 $ds exit=$? $(date) ############"
done

echo ""
echo "R1 sequence finished $(date). Checkpoints + results in $OUT_DIR/"
