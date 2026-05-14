"""
PCA Dimensionality Reduction experiment for Full Conformal Prediction.

Compares FCP performance (coverage, set size, runtime) at various PCA dimensions
vs the full 768-D baseline. PCA is unsupervised so it preserves exchangeability.

All splits are stratified (equal samples per class) to ensure fair comparison.

Usage:
    python src/pca_experiment.py --embeddings_path output/embeddings_cifar100.pt
    python src/pca_experiment.py --embeddings_path output/embeddings_cifar100.pt --dims 32,64,128,256,512 --cal_sizes 400,600,800
    python src/pca_experiment.py --embeddings_path output/embeddings_cub200.pt --pca_source unlabeled --unlabeled_per_class 15
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
from autoencoder_utils import EmbeddingAutoencoder


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
                             "Default: half of available after cal+test allocation.")
    parser.add_argument("--reduction", type=str, default="pca", choices=["pca", "ae"],
                        help="Reduction method: 'pca' (linear) or 'ae' (autoencoder)")
    return parser.parse_args()


def stratified_split(X, y, sizes, rng):
    """Split X, y into len(sizes)+1 stratified parts.

    sizes: list of ints — number of samples for each split.
    Returns: list of (X_part, y_part) for each size, plus the remainder.

    Each split gets ceil(size/K) per class (capped by available).
    """
    classes = np.unique(y)
    K = len(classes)

    # Build per-class index pools (shuffled)
    class_pools = {}
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        class_pools[c] = list(idx)

    parts = []
    for size in sizes:
        per_class = max(1, size // K)
        chosen = []
        # First pass: take per_class from each class
        for c in classes:
            take = min(per_class, len(class_pools[c]))
            chosen.extend(class_pools[c][:take])
            class_pools[c] = class_pools[c][take:]

        # If we need more (due to rounding), take extras round-robin
        deficit = size - len(chosen)
        if deficit > 0:
            for c in classes:
                if deficit <= 0:
                    break
                if len(class_pools[c]) > 0:
                    chosen.append(class_pools[c].pop(0))
                    deficit -= 1

        chosen = np.array(chosen[:size])
        rng.shuffle(chosen)
        parts.append((X[chosen], y[chosen]))

    # Remainder
    remaining = []
    for c in classes:
        remaining.extend(class_pools[c])
    remaining = np.array(remaining)
    if len(remaining) > 0:
        rng.shuffle(remaining)
        parts.append((X[remaining], y[remaining]))
    else:
        parts.append((np.empty((0, X.shape[1])), np.empty(0, dtype=y.dtype)))

    return parts


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
    min_class_count = min(int(np.sum(labels == c)) for c in all_classes)

    # Compute per-class budgets
    max_cal = max(cal_sizes)
    test_per_class = math.ceil(args.test_size / n_classes)
    max_cal_per_class = math.ceil(max_cal / n_classes)
    labeled_per_class = test_per_class + max_cal_per_class

    if carve_unlabeled:
        if args.unlabeled_per_class is not None:
            unlabeled_per_class = args.unlabeled_per_class
        else:
            # Default: half of remaining after labeled allocation
            unlabeled_per_class = max(10, (min_class_count - labeled_per_class) // 2)

        total_per_class = labeled_per_class + unlabeled_per_class
        if total_per_class > min_class_count:
            # Reduce unlabeled to fit
            unlabeled_per_class = min_class_count - labeled_per_class
            if unlabeled_per_class < 5:
                print(f"WARNING: only {unlabeled_per_class}/class for unlabeled pool. "
                      f"Consider using a separate unlabeled file or reducing cal/test sizes.")
            total_per_class = min_class_count

        print(f"Budget per class: {labeled_per_class} labeled (test={test_per_class} + cal={max_cal_per_class}), "
              f"{unlabeled_per_class} unlabeled, {total_per_class} total (available: {min_class_count})")
    else:
        total_per_class = min(labeled_per_class, min_class_count)
        unlabeled_per_class = 0
        print(f"Budget per class: {total_per_class} (test={test_per_class} + cal={max_cal_per_class}), "
              f"available: {min_class_count}")

    print(f"\nEmbeddings: {embeddings.shape}, Classes: {n_classes}")
    print(f"PCA dims: {dims} + full ({orig_dim})")
    print(f"Cal sizes: {cal_sizes}, Test size: {args.test_size}")
    print(f"NCM: {args.ncm}, Alpha: {args.alpha}, Trials: {args.n_trials}")
    print(f"Reduction: {args.reduction}, PCA source: {args.pca_source}")

    # Results storage
    results = {}
    for d in dims_with_full:
        results[d] = {}
        for cs in cal_sizes:
            results[d][cs] = {'coverages': [], 'set_sizes': [], 'times': []}

    for trial in range(args.n_trials):
        trial_seed = args.seed + trial * 1000
        rng = np.random.RandomState(trial_seed)

        # Stratified subsample: total_per_class from each class
        pool_indices = []
        for c in all_classes:
            c_idx = np.where(labels == c)[0]
            chosen = rng.choice(c_idx, size=total_per_class, replace=False)
            pool_indices.append(chosen)
        pool_indices = np.concatenate(pool_indices)
        rng.shuffle(pool_indices)

        X_pool_all = embeddings[pool_indices]
        y_pool_all = labels[pool_indices]

        # Carve unlabeled pool (stratified) if needed
        if carve_unlabeled:
            unlabeled_idx = []
            labeled_idx = []
            for c in all_classes:
                c_mask = np.where(y_pool_all == c)[0]
                # Shuffle within class to randomize which go to unlabeled vs labeled
                c_mask_shuffled = c_mask.copy()
                rng.shuffle(c_mask_shuffled)
                unlabeled_idx.append(c_mask_shuffled[:unlabeled_per_class])
                labeled_idx.append(c_mask_shuffled[unlabeled_per_class:])
            unlabeled_idx = np.concatenate(unlabeled_idx)
            labeled_idx = np.concatenate(labeled_idx)

            unlabeled_emb = X_pool_all[unlabeled_idx]
            X_pool = X_pool_all[labeled_idx]
            y_pool = y_pool_all[labeled_idx]
            if trial == 0:
                print(f"Carved unlabeled pool: {unlabeled_emb.shape}, "
                      f"labeled pool: {X_pool.shape}")
        else:
            X_pool = X_pool_all
            y_pool = y_pool_all

        # Stratified test/cal split
        splits = stratified_split(X_pool, y_pool, [args.test_size, max_cal], rng)
        (X_test_full, y_test) = splits[0]
        (X_remain_full, y_remain) = splits[1]
        # splits[2] is remainder (unused)

        actual_test = len(y_test)
        actual_remain = len(y_remain)

        print(f"\n{'='*60}")
        print(f"Trial {trial+1}/{args.n_trials} (seed={trial_seed}, "
              f"test={actual_test}, remain={actual_remain})")
        print(f"{'='*60}")

        for cal_size in cal_sizes:
            if cal_size > actual_remain:
                print(f"  cal={cal_size}: skipped (need {cal_size}, have {actual_remain})")
                continue

            # Stratified cal subset from remain
            if cal_size == actual_remain:
                X_cal_full = X_remain_full
                y_cal = y_remain
            else:
                cal_split = stratified_split(X_remain_full, y_remain, [cal_size], rng)
                (X_cal_full, y_cal) = cal_split[0]

            n_cal_classes = len(np.unique(y_cal))

            red_tag = "PCA" if args.reduction == "pca" else "AE"

            for dim in dims_with_full:
                if dim == orig_dim:
                    X_cal = X_cal_full
                    X_test = X_test_full
                    label = f"full-{orig_dim}"
                else:
                    if args.pca_source == "cal":
                        fit_data = X_cal_full
                    elif args.pca_source == "unlabeled" and unlabeled_emb is not None:
                        fit_data = unlabeled_emb
                    elif args.pca_source == "pool":
                        fit_data = np.vstack([X_cal_full, X_test_full])
                    else:
                        fit_data = X_cal_full

                    if args.reduction == "pca":
                        max_components = min(fit_data.shape[0], fit_data.shape[1])
                        if dim > max_components:
                            print(f"  cal={cal_size} {'PCA-'+str(dim):>10s}: "
                                  f"SKIPPED (dim={dim} > max_components={max_components})")
                            continue
                        reducer = PCA(n_components=dim)
                        reducer.fit(fit_data)
                    else:
                        reducer = EmbeddingAutoencoder(
                            bottleneck_dim=dim, seed=trial_seed)
                        reducer.fit(fit_data)

                    X_cal = reducer.transform(X_cal_full)
                    X_test = reducer.transform(X_test_full)
                    label = f"{red_tag}-{dim}"

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
                      f"t={elapsed:.1f}s  ({n_cal_classes}/{n_classes} cal classes)")

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
        red_label = "PCA" if args.reduction == "pca" else "AE"
        for i, dim in enumerate(dims_with_full):
            label = f"{red_label}-{dim}" if dim < orig_dim else f"Full-{orig_dim}"
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
        if ax_idx == 1:
            ax.axhline(y=1 - args.alpha, color='red', linestyle='--', alpha=0.5,
                        label=f'Target {1-args.alpha:.0%}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{args.reduction.upper()} Dimensionality Reduction for FCP ({args.ncm})\n'
                 f'fit source: {args.pca_source}', fontsize=12)
    plt.tight_layout()
    plot_path = Path(args.output_dir) / f"{args.reduction}_comparison_{args.ncm}_{args.pca_source}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")
    plt.close()

    # --- Dim vs set size bar chart ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bar_width = 0.8 / len(cal_sizes)
    x_pos = np.arange(len(dims_with_full))

    for j, cs in enumerate(cal_sizes):
        means, stds = [], []
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
    red_label2 = "PCA" if args.reduction == "pca" else "AE"
    ax2.set_xticklabels([f"{red_label2}-{d}" if d < orig_dim else f"Full-{orig_dim}" for d in dims_with_full])
    ax2.set_ylabel('Avg Set Size')
    ax2.set_xlabel('Embedding Dimension')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title(f'Set Size vs {args.reduction.upper()} Dimension ({args.ncm}, fit source: {args.pca_source})')
    plt.tight_layout()
    bar_path = Path(args.output_dir) / f"{args.reduction}_barplot_{args.ncm}_{args.pca_source}.png"
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

    json_path = Path(args.output_dir) / f"{args.reduction}_results_{args.ncm}_{args.pca_source}.json"
    with open(json_path, 'w') as f:
        json.dump(to_serializable({
            'dims': dims_with_full,
            'cal_sizes': cal_sizes,
            'ncm': args.ncm,
            'alpha': args.alpha,
            'n_trials': args.n_trials,
            'reduction': args.reduction,
            'pca_source': args.pca_source,
            'results': results,
        }), f, indent=2)
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
