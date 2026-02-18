"""
Sweep lambda_cs values for class similarity penalty and plot results.

Usage:
    python -m src.sweep_lambda_cs --embeddings_path output/embeddings_dinov2_base.pt
"""

import sys
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, "src")
from conformal_prediction import (
    FullConformalPredictor,
    FastNNRatio,
    build_class_similarity_matrix,
)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--ncm", type=str, default="nn_ratio",
                        choices=["nn_ratio", "geodesic_nn_ratio"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cal_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    # Lambda values to sweep
    lambda_values = np.round(np.arange(0.0, 1.05, 0.1), 2)

    # Load data
    print(f"Loading {args.embeddings_path}...")
    data = torch.load(args.embeddings_path)
    embeddings = data["embeddings"].numpy()
    labels = data["labels"].numpy()
    all_classes = np.unique(labels)
    n_classes = len(all_classes)
    print(f"  {embeddings.shape[0]} samples, {n_classes} classes, {embeddings.shape[1]}-dim")

    # Build similarity matrix from full dataset
    print("Building class similarity matrix...")
    M, sim_classes = build_class_similarity_matrix(embeddings, labels)
    print(f"  M shape: {M.shape}, avg={M.mean():.3f}, diag={np.diag(M).mean():.3f}")
    print(f"  Off-diag range: [{M[~np.eye(n_classes, dtype=bool)].min():.3f}, "
          f"{M[~np.eye(n_classes, dtype=bool)].max():.3f}]")

    # Stratified subsampling
    import math
    max_needed = args.cal_size + args.test_size
    num_per_class = math.ceil(max_needed / n_classes)
    min_class_count = min(np.sum(labels == c) for c in all_classes)
    if num_per_class > min_class_count:
        num_per_class = min_class_count
    print(f"  Stratified: {num_per_class}/class, pool={num_per_class * n_classes}")

    # Results storage
    results = {lam: {"coverages": [], "set_sizes": [], "singleton_rates": [],
                     "empty_rates": [], "times": []}
               for lam in lambda_values}

    from conformal_prediction import create_ncm

    for trial in range(args.n_trials):
        trial_seed = args.seed + trial * 1000
        np.random.seed(trial_seed)

        # Stratified subsample
        pool_idx = []
        for c in all_classes:
            c_idx = np.where(labels == c)[0]
            chosen = np.random.choice(c_idx, size=num_per_class, replace=False)
            pool_idx.append(chosen)
        pool_idx = np.concatenate(pool_idx)
        np.random.shuffle(pool_idx)

        X_pool = embeddings[pool_idx]
        y_pool = labels[pool_idx]

        # Carve test set (last test_size)
        actual_test = min(args.test_size, len(X_pool))
        X_test = X_pool[-actual_test:]
        y_test = y_pool[-actual_test:]
        X_remain = X_pool[:-actual_test]
        y_remain = y_pool[:-actual_test]

        actual_cal = min(args.cal_size, len(X_remain))
        X_cal = X_remain[:actual_cal]
        y_cal = y_remain[:actual_cal]

        print(f"\nTrial {trial + 1}/{args.n_trials} (seed={trial_seed}): "
              f"cal={len(X_cal)}, test={len(X_test)}")

        for lam in lambda_values:
            ncm = create_ncm(args.ncm)
            cp = FullConformalPredictor(
                ncm, alpha=args.alpha,
                similarity_matrix=M if lam > 0 else None,
                sim_classes=sim_classes if lam > 0 else None,
                lambda_cs=lam,
            )
            cp.calibrate(X_cal, y_cal, all_classes=all_classes)

            t0 = time.time()
            metrics = cp.evaluate(X_test, y_test, verbose=False)
            dt = time.time() - t0

            results[lam]["coverages"].append(metrics["coverage"])
            results[lam]["set_sizes"].append(metrics["avg_set_size"])
            results[lam]["singleton_rates"].append(metrics["singleton_rate"])
            results[lam]["empty_rates"].append(metrics["empty_set_rate"])
            results[lam]["times"].append(dt)

            print(f"  lambda={lam:.2f}: cov={metrics['coverage']:.3f}, "
                  f"size={metrics['avg_set_size']:.2f}, "
                  f"singleton={metrics['singleton_rate']:.3f}, "
                  f"empty={metrics['empty_set_rate']:.3f}, "
                  f"t={dt:.1f}s")

    # ---- PLOT ----
    Path(args.output_dir).mkdir(exist_ok=True)

    cov_means = [np.mean(results[l]["coverages"]) for l in lambda_values]
    cov_stds = [np.std(results[l]["coverages"]) for l in lambda_values]
    size_means = [np.mean(results[l]["set_sizes"]) for l in lambda_values]
    size_stds = [np.std(results[l]["set_sizes"]) for l in lambda_values]
    sing_means = [np.mean(results[l]["singleton_rates"]) for l in lambda_values]
    sing_stds = [np.std(results[l]["singleton_rates"]) for l in lambda_values]
    empty_means = [np.mean(results[l]["empty_rates"]) for l in lambda_values]
    empty_stds = [np.std(results[l]["empty_rates"]) for l in lambda_values]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Coverage
    ax = axes[0, 0]
    ax.errorbar(lambda_values, cov_means, yerr=cov_stds,
                fmt="o-", color="blue", linewidth=2, markersize=7, capsize=4)
    ax.axhline(y=1 - args.alpha, color="red", linestyle="--", linewidth=2,
               label=f"Target (1-α={1 - args.alpha:.2f})")
    ax.set_xlabel("λ (class similarity penalty)", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title("Coverage vs λ", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Average set size
    ax = axes[0, 1]
    ax.errorbar(lambda_values, size_means, yerr=size_stds,
                fmt="s-", color="green", linewidth=2, markersize=7, capsize=4)
    ax.set_xlabel("λ (class similarity penalty)", fontsize=12)
    ax.set_ylabel("Average Set Size", fontsize=12)
    ax.set_title("Average Set Size vs λ", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Singleton rate
    ax = axes[1, 0]
    ax.errorbar(lambda_values, sing_means, yerr=sing_stds,
                fmt="D-", color="orange", linewidth=2, markersize=7, capsize=4)
    ax.set_xlabel("λ (class similarity penalty)", fontsize=12)
    ax.set_ylabel("Singleton Rate", fontsize=12)
    ax.set_title("Singleton Rate vs λ", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Empty set rate
    ax = axes[1, 1]
    ax.errorbar(lambda_values, empty_means, yerr=empty_stds,
                fmt="^-", color="red", linewidth=2, markersize=7, capsize=4)
    ax.set_xlabel("λ (class similarity penalty)", fontsize=12)
    ax.set_ylabel("Empty Set Rate", fontsize=12)
    ax.set_title("Empty Set Rate vs λ", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Class Similarity Penalty Sweep — Full CP + {args.ncm}\n"
        f"DINOv2-base, {n_classes} classes, cal={args.cal_size}, test={args.test_size}, "
        f"α={args.alpha}, {args.n_trials} trials",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = Path(args.output_dir) / f"lambda_cs_sweep_{args.ncm}.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
    plt.close()

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY  ({args.n_trials} trials, NCM={args.ncm})")
    print(f"{'='*80}")
    print(f"{'lambda':>8} {'Coverage':>12} {'Avg Size':>12} {'Singleton':>12} {'Empty':>12}")
    print("-" * 60)
    for l in lambda_values:
        print(f"{l:>8.2f} "
              f"{np.mean(results[l]['coverages'])*100:>10.1f}% "
              f"{np.mean(results[l]['set_sizes']):>11.2f} "
              f"{np.mean(results[l]['singleton_rates'])*100:>10.1f}% "
              f"{np.mean(results[l]['empty_rates'])*100:>10.1f}%")


if __name__ == "__main__":
    main()
