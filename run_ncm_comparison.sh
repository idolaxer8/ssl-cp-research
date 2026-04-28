#!/bin/bash
# NCM comparison after empty-set fallback bug fix.
# Datasets: cifar10, cifar100, flowers102, miniimagenet
# Alpha: 0.1 (default). Secondary: 0.05, 0.2 if needed.

set -e
cd /c/Users/IDO/Desktop/Ido_student/Msc/ssl-cp-research
PYTHON=/c/Users/IDO/anaconda3/envs/ssl-cp/python.exe
OUT=output/ncm_comparison

echo "===== CIFAR-10 ====="
$PYTHON src/compare_ncms.py \
    --embeddings_path output/embeddings_cifar10.pt \
    --alphas 0.1 \
    --n_trials 5 \
    --cal_sizes 50,100,150,200,300,400 \
    --test_size 500 \
    --output_dir $OUT

echo ""
echo "===== CIFAR-100 ====="
$PYTHON src/compare_ncms.py \
    --embeddings_path output/embeddings_cifar100.pt \
    --alphas 0.1 \
    --n_trials 5 \
    --cal_sizes 100,200,300,400,600 \
    --test_size 300 \
    --output_dir $OUT

echo ""
echo "===== Flowers-102 ====="
$PYTHON src/compare_ncms.py \
    --embeddings_path output/embeddings_flowers102.pt \
    --alphas 0.1 \
    --n_trials 5 \
    --cal_sizes 200,400,600,800,1200 \
    --test_size 500 \
    --output_dir $OUT

echo ""
echo "===== miniImageNet ====="
$PYTHON src/compare_ncms.py \
    --embeddings_path output/embeddings_miniimagenet.pt \
    --alphas 0.1 \
    --n_trials 5 \
    --cal_sizes 200,400,600,800,1200 \
    --test_size 500 \
    --output_dir $OUT

echo ""
echo "All done. Results in $OUT/"
