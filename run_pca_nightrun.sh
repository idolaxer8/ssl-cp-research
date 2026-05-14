#!/bin/bash
# =============================================================================
# PCA Dimensionality Reduction — Full Night Run
# =============================================================================
#
# Runs PCA experiments across 3 datasets with stratified splits.
# Estimated runtime: ~2 hours
#
# Experiments:
#   1. CIFAR-100 — PCA on unlabeled pool (10K separate file)
#   2. CIFAR-100 — PCA on cal data (control: expected to fail at high dims)
#   3. CUB-200   — PCA on unlabeled pool (carved from main, 28/class)
#   4. miniImageNet — PCA on unlabeled pool (carved from main, 100/class)
#
# Usage:
#   cd ssl-cp-research
#   bash run_pca_nightrun.sh 2>&1 | tee output/pca_experiment/nightrun.log
#
# =============================================================================

set -e  # Exit on error

PYTHON="/c/Users/IDO/anaconda3/envs/ssl-cp/python.exe"
SCRIPT="src/pca_experiment.py"
BASE_DIR="output/pca_experiment"

echo "============================================================"
echo "PCA Night Run — $(date)"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# 1. CIFAR-100 — PCA on unlabeled pool (10K separate file)
# --------------------------------------------------------------------------
# Dataset: 2500 labeled (25/class x 100), 10K unlabeled (separate file)
# PCA fit on 10K unlabeled (n >> d=768, no overfitting)
# Cal sizes: 400 (4/cls), 600 (6/cls), 800 (8/cls)
# Test: 1000 (10/cls)
# 5 dims x 3 cal x 5 trials = 90 FCP runs, ~27 min

echo "============================================================"
echo "[1/4] CIFAR-100 — PCA source: unlabeled (10K)"
echo "============================================================"

$PYTHON -u $SCRIPT \
    --embeddings_path output/embeddings_cifar100.pt \
    --unlabeled_path output/embeddings_cifar100_unlabeled.pt \
    --pca_source unlabeled \
    --dims 32,64,128,256,512 \
    --cal_sizes 400,600,800 \
    --test_size 1000 \
    --n_trials 5 \
    --ncm geodesic_topk_mean \
    --output_dir $BASE_DIR/cifar100_unlabeled

echo ""
echo "[1/4] CIFAR-100 unlabeled — DONE ($(date))"
echo ""

# --------------------------------------------------------------------------
# 2. CIFAR-100 — PCA on calibration data (control experiment)
# --------------------------------------------------------------------------
# Same dataset, but PCA fit on cal data only (400-800 points in 768-D)
# Expected: under-coverage at high dims (PCA-256, PCA-512) due to overfitting
# This serves as the negative control to confirm unlabeled PCA is the key

echo "============================================================"
echo "[2/4] CIFAR-100 — PCA source: cal (control)"
echo "============================================================"

$PYTHON -u $SCRIPT \
    --embeddings_path output/embeddings_cifar100.pt \
    --pca_source cal \
    --dims 32,64,128,256,512 \
    --cal_sizes 400,600,800 \
    --test_size 1000 \
    --n_trials 5 \
    --ncm geodesic_topk_mean \
    --output_dir $BASE_DIR/cifar100_cal

echo ""
echo "[2/4] CIFAR-100 cal — DONE ($(date))"
echo ""

# --------------------------------------------------------------------------
# 3. CUB-200 — PCA on unlabeled pool (carved from main embeddings)
# --------------------------------------------------------------------------
# Dataset: ~10K total (41-50/class x 200 classes)
# Budget: 13/class labeled (test=5/cls + cal up to 8/cls), 28/class unlabeled
# Unlabeled pool: 28 x 200 = 5600 (> 768, safe for PCA)
# Cal sizes: 600 (3/cls), 800 (4/cls), 1000 (5/cls), 1500 (7.5/cls)
# Test: 1000 (5/cls)
# Note: cal=400 (2/cls) with 200 classes produces invalid results regardless
# of PCA — too few samples. Start from cal=600.

echo "============================================================"
echo "[3/4] CUB-200 — PCA source: unlabeled (carved, 28/class)"
echo "============================================================"

$PYTHON -u $SCRIPT \
    --embeddings_path output/embeddings_cub200.pt \
    --pca_source unlabeled \
    --unlabeled_per_class 28 \
    --dims 32,64,128,256,512 \
    --cal_sizes 600,800,1000,1500 \
    --test_size 1000 \
    --n_trials 5 \
    --ncm geodesic_topk_mean \
    --output_dir $BASE_DIR/cub200_unlabeled

echo ""
echo "[3/4] CUB-200 unlabeled — DONE ($(date))"
echo ""

# --------------------------------------------------------------------------
# 4. miniImageNet — PCA on unlabeled pool (carved from main embeddings)
# --------------------------------------------------------------------------
# Dataset: 50K total (500/class x 100 classes)
# Budget: 18/class labeled (test=10/cls + cal=8/cls), 100/class unlabeled
# Unlabeled pool: 100 x 100 = 10000 (matches CIFAR-100 unlabeled size)
# Cal sizes: 400 (4/cls), 600 (6/cls), 800 (8/cls)
# Test: 1000 (10/cls)

echo "============================================================"
echo "[4/4] miniImageNet — PCA source: unlabeled (carved, 100/class)"
echo "============================================================"

$PYTHON -u $SCRIPT \
    --embeddings_path output/embeddings_miniimagenet.pt \
    --pca_source unlabeled \
    --unlabeled_per_class 100 \
    --dims 32,64,128,256,512 \
    --cal_sizes 400,600,800 \
    --test_size 1000 \
    --n_trials 5 \
    --ncm geodesic_topk_mean \
    --output_dir $BASE_DIR/miniimagenet_unlabeled

echo ""
echo "[4/4] miniImageNet unlabeled — DONE ($(date))"
echo ""

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  $BASE_DIR/cifar100_unlabeled/"
echo "  $BASE_DIR/cifar100_cal/"
echo "  $BASE_DIR/cub200_unlabeled/"
echo "  $BASE_DIR/miniimagenet_unlabeled/"
echo ""
echo "Key files per directory:"
echo "  pca_comparison_geodesic_topk_mean_*.png   — line plots (sz/cov/time vs cal)"
echo "  pca_barplot_geodesic_topk_mean_*.png      — bar chart (sz vs dim)"
echo "  pca_results_geodesic_topk_mean_*.json     — raw data"
