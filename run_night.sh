#!/usr/bin/env bash
# Night run — runs all experiments on available embeddings (skips missing ones).
set -e

PYTHON="/c/Users/IDO/anaconda3/envs/ssl-cp/python.exe"
ROOT="/c/Users/IDO/Desktop/Ido_student/Msc/ssl-cp-research"
LOG="$ROOT/output/night_run.log"

cd "$ROOT"

echo "[$(date)] Starting night runs." | tee -a "$LOG"

# ── 1. NCM comparison on new datasets ───────────────────────────────────────
echo "[$(date)] Step 1a: NCM comparison — miniimagenet" | tee -a "$LOG"
$PYTHON src/compare_ncms.py \
    --embeddings_path output/embeddings_miniimagenet.pt \
    --cal_sizes 200,300,400,500,600,700,800 \
    --n_trials 3 2>&1 | tee -a "$LOG"

if [ -f "output/embeddings_tiny_imagenet.pt" ]; then
    echo "[$(date)] Step 1b: NCM comparison — tiny_imagenet" | tee -a "$LOG"
    $PYTHON src/compare_ncms.py \
        --embeddings_path output/embeddings_tiny_imagenet.pt \
        --cal_sizes 400,600,800,1000,1500 \
        --n_trials 3 2>&1 | tee -a "$LOG"
else
    echo "[$(date)] Step 1b: SKIP tiny_imagenet (embeddings not found)" | tee -a "$LOG"
fi

# ── 2. Multi-dataset base sweep (FCP vs CV+ vs SCP, no CS) ──────────────────
echo "[$(date)] Step 2: Multi-dataset base sweep" | tee -a "$LOG"
$PYTHON -u -m src.run_multi_dataset_experiments \
    --skip_cs 2>&1 | tee -a "$LOG"

# ── 3. CS experiment (100+ class datasets only) ──────────────────────────────
echo "[$(date)] Step 3: Class-similarity experiment" | tee -a "$LOG"
$PYTHON -u -m src.run_multi_dataset_experiments \
    --only_cs \
    --datasets cifar100 flowers102 cub200 miniimagenet 2>&1 | tee -a "$LOG"

# ── 4. Alpha sweep (miniimagenet only) ───────────────────────────────────────
echo "[$(date)] Step 4: Alpha sweep — miniimagenet" | tee -a "$LOG"
$PYTHON -u -m src.run_multi_dataset_experiments \
    --datasets miniimagenet \
    --alphas 0.05 0.10 0.20 \
    --skip_cs 2>&1 | tee -a "$LOG"

# ── 5. Few-shot sweep (all available datasets) ───────────────────────────────
echo "[$(date)] Step 5: Few-shot sweep" | tee -a "$LOG"
$PYTHON -u -m src.run_few_shot_experiment 2>&1 | tee -a "$LOG"

echo "[$(date)] All night runs complete." | tee -a "$LOG"
