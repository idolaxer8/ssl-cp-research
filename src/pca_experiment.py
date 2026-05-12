"""
PCA Dimensionality Reduction experiment for Full Conformal Prediction.

Compares FCP performance (coverage, set size, runtime) at various PCA dimensions
vs the full 768-D baseline. PCA is unsupervised so it preserves exchangeability.

Usage:
    python src/pca_experiment.py --embeddings_path output/embeddings_cifar100.pt
    python src/pca_experiment.py --embeddings_path output/embeddings_cifar100.pt --dims 32,64,128,256,512 --cal_sizes 400,600,800
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import argparse
import json
import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

from conformal_prediction import (
    FullConformalPredictor,
    create_ncm,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PCA dim reduction for FCP")
    parser.add_argument("--embeddings_path", type=str, default="output/embeddings_cifar100.pt")
    parser.add_argument("--dims", type=str, default="32,64,128,256,512",
                        help="Comma-separated PCA dimensions to test (768=full is always included)")
    parser.add_argument("--cal_sizes", type=str, default="400,600,800",
                        help="Comma-separated calibration sizes")
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ncm", type=str, default="geodesic_topk_mean",
                        choices=["geodesic_topk_mean", "geodesic_topk_asym", "whitened_geodesic",
                                 "mahal_nn_ratio"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output/pca_experiment")
    parser.add_argument("--pca_source", type=str, default="cal",
                        choices=["cal", "unlabeled", "pool"],
                        help="Where to fit PCA: 'cal' (calibration only), "
                             "'unlabeled' (separate unlabeled file or carved from main), "
                             "'pool' (cal+test)")
    parser.add_argument("--unlabeled_path", type=str, default=None,
                        help="Path to unlabeled embeddings .pt (for --pca_source unlabeled). "
                             "If not provided, unlabeled pool is carved from the main embeddings.")
    parser.add_argument("--unlabeled_per_class", type=int, default=None,
                        help="Per-class budget for unlabeled pool when carving from main embeddings. "
                             "Default: use all remaining data after cal+test allocation.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    data = torch.load(args.embeddings_path, map_location="cpu", weights_only=True)
    embeddings = data["embeddings"].numpy().astype(np.float32)
    labels = data["labels"].numpy()
    orig_dim = embeddings.shape[1]

    # Load or prepare unlabeled pool
    unlabeled_emb = None
    carve_unlabeled = False
    if args.pca_source == "unlabeled":
        if args.unlabeled_path is not None:
            u_data = torch.load(args.unlabeled_path, map_location="cpu", weights_only=True)
            unlabeled_emb = u_data["embeddings"].numpy().astype(np.float32)
            print(f"Loaded unlabeled embeddings from file: {unlabeled_emb.shape}")
        else:
            # Auto-detect file
            base = args.embeddings_path.replace(".pt", "_unlabeled.pt")
            if os.path.exists(base):
                u_data = torch.load(base, map_location="cpu", weights_only=True)
                unlabeled_emb = u_data["embeddings"].numpy().astype(np.float32)
                print(f"Loaded unlabeled embeddings from file: {unlabeled_emb.shape}")
            else:
                # Carve from main embeddings
                carve_unlabeled = True
                print("No unlabeled file found. Will carve unlabeled pool from main embeddings.")

    dims = [int(d.strip()) for d in args.dims.split(',')]
    dims = [d for d in dims if d < orig_dim]
    dims_with_full = dims + [orig_dim]  # always include full baseline
    cal_sizes = [int(s.strip()) for s in args.cal_sizes.split(',')]

    all_classes = np.unique(labels)
    n_classes = len(all_classes)

    print(f"Embeddings: {embeddings.shape}, Classes: {n_classes}")
    print(f"PCA dims: {dims} + full ({orig_dim})")
    print(f"Cal sizes: {cal_sizes}, Test size: {args.test_size}")
    print(f"NCM: {args.ncm}, Alpha: {args.alpha}, Trials: {args.n_trials}")
    print(f"PCA source: {args.pca_source}")

    # Pre-compute pool budget
    max_cal = max(cal_sizes)
    max_needed = args.test_size + max_cal
    min_class_count = min(int(np.sum(labels == c)) for c in all_classes)

    if carve_unlabeled:
        # Need extra samples for unlabeled pool
        unlabeled_per_class = args.unlabeled_per_class or max(10, min_class_count - math.ceil(max_needed / n_classes))
        labeled_per_class = math.ceil(max_needed / n_classes)
        num_per_class = labeled_per_class + unlabeled_per_class
        if num_per_class > min_class_count:
            # Reduce unlabeled budget to fit
            num_per_class = min_class_count
            unlabeled_per_class = num_per_class - labeled_per_class
            if unlabeled_per_class < 5:
                print(f"WARNING: only {unlabeled_per_class}/class for unlabeled pool. "
                      f"Consider using a separate unlabeled file.")
        print(f"Carving: {labeled_per_class}/class for cal+test, "
              f"{unlabeled_per_class}/class for unlabeled PCA pool")
    else:
        num_per_class = math.ceil(max_needed / n_classes)
        if num_per_class > min_class_count:
            num_per_class = min_class_count
            max_needed = num_per_class * n_classes

    # Results storage: results[dim][cal_size] = {coverages: [], set_sizes: [], times: []}
    results = {}
    for d in dims_with_full:
        results[d] = {}
        for cs in cal_sizes:
            results[d][cs] = {'coverages': [], 'set_sizes': [], 'times': []}

    for trial in range(args.n_trials):
        trial_seed = args.seed + trial * 1000
        np.random.seed(trial_seed)

        # Stratified subsample
        pool_indices = []
        for c in all_classes:
            c_idx = np.where(labels == c)[0]
            chosen = np.random.choice(c_idx, size=num_per_class, replace=False)
            pool_indices.append(chosen)
        pool_indices = np.concatenate(pool_indices)
        np.random.shuffle(pool_indices)

        X_pool_all = embeddings[pool_indices]
        y_pool_all = labels[pool_indices]

        if carve_unlabeled:
            # Carve unlabeled pool (stratified): take unlabeled_per_class from each class
            unlabeled_idx = []
            labeled_idx = []
            for c in all_classes:
                c_mask = np.where(y_pool_all == c)[0]
                unlabeled_idx.append(c_mask[:unlabeled_per_class])
                labeled_idx.append(c_mask[unlabeled_per_class:])
            unlabeled_idx = np.concatenate(unlabeled_idx)
            labeled_idx = np.concatenate(labeled_idx)
            np.random.shuffle(labeled_idx)

            unlabeled_emb = X_pool_all[unlabeled_idx]
            X_pool = X_pool_all[labeled_idx]
            y_pool = y_pool_all[labeled_idx]
            if trial == 0:
                print(f"Carved unlabeled pool: {unlabeled_emb.shape}, "
                      f"labeled pool: {X_pool.shape}")
        else:
            X_pool = X_pool_all
            y_pool = y_pool_all

        # Fixed test set
        X_test_full = X_pool[-args.test_size:]
        y_test = y_pool[-args.test_size:]
        X_remain_full = X_pool[:-args.test_size]
        y_remain = y_pool[:-args.test_size]

        print(f"\n{'='*60}")
        print(f"Trial {trial+1}/{args.n_trials} (seed={trial_seed})")
        print(f"{'='*60}")

        for cal_size in cal_sizes:
            if cal_size > len(X_remain_full):
                print(f"  cal={cal_size}: skipped (insufficient data)")
                continue

            X_cal_full = X_remain_full[:cal_size]
            y_cal = y_remain[:cal_size]

            for dim in dims_with_full:
                if dim == orig_dim:
                    # Baseline: no PCA
                    X_cal = X_cal_full
                    X_test = X_test_full
                    label = f"full-{orig_dim}"
                else:
                    # Fit PCA
                    if args.pca_source == "cal":
                        pca_fit_data = X_cal_full
                    elif args.pca_source == "unlabeled" and unlabeled_emb is not None:
                        pca_fit_data = unlabeled_emb
                    elif args.pca_source == "pool":
                        pca_fit_data = X_pool
                    else:
                        pca_fit_data = X_cal_full

                    # Skip if dim exceeds what PCA can handle
                    max_components = min(pca_fit_data.shape[0], pca_fit_data.shape[1])
                    if dim > max_components:
                        print(f"  cal={cal_size} {'PCA-'+str(dim):>10s}: "
                              f"SKIPPED (dim={dim} > max_components={max_components})")
                        continue

                    pca = PCA(n_components=dim)
                    pca.fit(pca_fit_data)
                    X_cal = pca.transform(X_cal_full)
                    X_test = pca.transform(X_test_full)
                    label = f"PCA-{dim}"

                # Run FCP
                ncm = create_ncm(args.ncm, k=args.k)
                cp = FullConformalPredictor(ncm, alpha=args.alpha)
                t0 = time.time()
                cp.calibrate(X_cal, y_cal, all_classes=all_classes)
                metrics = cp.evaluate(X_test, y_test, verbose=False)
                elapsed = time.time() - t0

                results[dim][cal_size]['coverages'].append(metrics['coverage'])
                results[dim][cal_size]['set_sizes'].append(metrics['avg_set_size'])
                results[dim][cal_size]['times'].append(elapsed)

                print(f"  cal={cal_size} {label:>10s}: "
                      f"cov={metrics['coverage']:.3f}  sz={metrics['avg_set_size']:.2f}  "
                      f"t={elapsed:.1f}s")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dim':<8}", end="")
    for cs in cal_sizes:
        print(f"  cal={cs:>4d} (cov/sz/t)", end="")
    print()
    print("-" * (8 + len(cal_sizes) * 30))

    for dim in dims_with_full:
        label = f"{dim}" if dim < orig_dim else f"{dim}*"
        print(f"{label:<8}", end="")
        for cs in cal_sizes:
            d = results[dim][cs]
            if not d['coverages']:
                print(f"  {'N/A':>28}", end="")
                continue
            cov = np.mean(d['coverages'])
            sz = np.mean(d['set_sizes'])
            t = np.mean(d['times'])
            cov_std = np.std(d['coverages'])
            sz_std = np.std(d['set_sizes'])
            print(f"  {cov:.3f}/{sz:.2f}+/-{sz_std:.2f}/{t:.1f}s", end="")
        print()

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(dims_with_full)))

    for ax_idx, (metric, ylabel) in enumerate([
        ('set_sizes', 'Avg Set Size'),
        ('coverages', 'Coverage'),
        ('times', 'Time (s)')
    ]):
        ax = axes[ax_idx]
        for i, dim in enumerate(dims_with_full):
            label = f"PCA-{dim}" if dim < orig_dim else f"Full-{orig_dim}"
            xs, ys, yerrs = [], [], []
            for cs in cal_sizes:
                d = results[dim][cs]
                if not d[metric]:
                    continue
                xs.append(cs)
                ys.append(np.mean(d[metric]))
                yerrs.append(np.std(d[metric]))
            if xs:
                ax.errorbar(xs, ys, yerr=yerrs, marker='o', label=label,
                            color=colors[i], capsize=3, linewidth=2)

        ax.set_xlabel('Calibration Size')
        ax.set_ylabel(ylabel)
        if ax_idx == 1:  # coverage
            ax.axhline(y=1 - args.alpha, color='red', linestyle='--', alpha=0.5, label=f'Target {1-args.alpha:.0%}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'PCA Dimensionality Reduction for FCP ({args.ncm})\n'
                 f'PCA source: {args.pca_source}', fontsize=12)
    plt.tight_layout()
    plot_path = Path(args.output_dir) / f"pca_comparison_{args.ncm}_{args.pca_source}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")
    plt.close()

    # --- Dim vs set size at each cal size (bar chart) ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bar_width = 0.8 / len(cal_sizes)
    x_pos = np.arange(len(dims_with_full))

    for j, cs in enumerate(cal_sizes):
        means = []
        stds = []
        for dim in dims_with_full:
            d = results[dim][cs]
            if d['set_sizes']:
                means.append(np.mean(d['set_sizes']))
                stds.append(np.std(d['set_sizes']))
            else:
                means.append(0)
                stds.append(0)
        ax2.bar(x_pos + j * bar_width, means, bar_width, yerr=stds,
                label=f'cal={cs}', capsize=2, alpha=0.8)

    ax2.set_xticks(x_pos + bar_width * (len(cal_sizes) - 1) / 2)
    ax2.set_xticklabels([f"PCA-{d}" if d < orig_dim else f"Full-{orig_dim}" for d in dims_with_full])
    ax2.set_ylabel('Avg Set Size')
    ax2.set_xlabel('Embedding Dimension')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title(f'Set Size vs PCA Dimension ({args.ncm}, PCA source: {args.pca_source})')
    plt.tight_layout()
    bar_path = Path(args.output_dir) / f"pca_barplot_{args.ncm}_{args.pca_source}.png"
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    print(f"Saved bar plot to {bar_path}")
    plt.close()

    # --- Save JSON ---
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        return obj

    json_path = Path(args.output_dir) / f"pca_results_{args.ncm}_{args.pca_source}.json"
    with open(json_path, 'w') as f:
        json.dump(to_serializable({
            'dims': dims_with_full,
            'cal_sizes': cal_sizes,
            'ncm': args.ncm,
            'alpha': args.alpha,
            'n_trials': args.n_trials,
            'pca_source': args.pca_source,
            'results': results,
        }), f, indent=2)
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
