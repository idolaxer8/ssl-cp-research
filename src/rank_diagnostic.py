"""
Rank Uniformity Diagnostic for Full CP.

Under perfect exchangeability with continuous scores, the rank of the test
point's score among the n_cal+1 augmented scores (under the TRUE label) must
be Uniform{1,...,n_cal+1}.

This script:
  1. Loads embeddings, draws a cal/test split.
  2. For each test point, computes:
       - test_score  = score of x_test under y_true
       - updated_cal = updated scores of all cal points when (x_test, y_true) is added
  3. Records:
       - rank      = position of test_score among updated_cal (0-indexed from lowest)
       - p_value   = (#{updated_cal >= test_score} + 1) / (n_cal + 1)
       - n_ties    = #{updated_cal == test_score}
       - score_val = actual test_score value
  4. Plots a histogram of ranks and a scatter of score values.

Usage:
  python src/rank_diagnostic.py --embeddings_path output/embeddings_cifar10.pt --ncm geodesic_nn_ratio
  python src/rank_diagnostic.py --embeddings_path output/embeddings_cifar10.pt --ncm nn_ratio
"""

import sys, os, argparse
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

from conformal_prediction import create_ncm


def stratified_split(X, y, n_cal, n_test, rng):
    classes = np.unique(y)
    K = len(classes)
    cal_per = max(1, n_cal // K)
    test_per = max(1, n_test // K)
    cal_idx, test_idx = [], []
    for c in classes:
        pool = np.where(y == c)[0]
        rng.shuffle(pool)
        need = cal_per + test_per
        if len(pool) < need:
            if len(pool) < 2:
                continue
            s = max(1, len(pool) // 2)
            cal_idx.extend(pool[:s].tolist())
            test_idx.extend(pool[s:].tolist())
        else:
            cal_idx.extend(pool[:cal_per].tolist())
            test_idx.extend(pool[cal_per:cal_per + test_per].tolist())
    return np.array(cal_idx), np.array(test_idx)


def run_diagnostic(X, y, ncm_name, cal_size, test_size, seed, tau, out_dir, dataset_tag):
    rng = np.random.default_rng(seed)
    cal_idx, test_idx = stratified_split(X, y, cal_size, test_size, rng)
    X_cal, y_cal = X[cal_idx], y[cal_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    n_cal = len(X_cal)

    print(f"\nNCM={ncm_name}  cal={n_cal}  test={len(X_test)}")

    ncm = create_ncm(ncm_name, tau=tau)
    ncm.fit(X_cal, y_cal)

    ranks = []          # rank of test score among n_cal+1 augmented scores (0-indexed, 0=smallest)
    p_values = []
    test_scores = []
    tie_counts = []
    zero_score_count = 0

    for i in range(len(X_test)):
        x = X_test[i]
        y_true = int(y_test[i])

        # Precompute Euclidean distances once
        dists = euclidean_distances([x], X_cal).flatten()

        try:
            ts = ncm.score_x(x, y_true, precomputed_dists=dists)
            uc = ncm.updated_calibration_scores_for(x, y_true, precomputed_dists=dists)
        except TypeError:
            ts = ncm.score_x(x, y_true)
            uc = ncm.updated_calibration_scores_for(x, y_true)

        n_ge = int(np.sum(uc >= ts))
        p_val = (n_ge + 1) / (n_cal + 1)
        n_ties = int(np.sum(uc == ts))

        # rank from smallest: how many updated_cal scores are STRICTLY less than ts
        rank = int(np.sum(uc < ts))   # 0 = test score is the smallest (most conforming)

        ranks.append(rank)
        p_values.append(p_val)
        test_scores.append(ts)
        tie_counts.append(n_ties)

        if ts == 0.0:
            zero_score_count += 1

    ranks = np.array(ranks)
    p_values = np.array(p_values)
    test_scores = np.array(test_scores)
    tie_counts = np.array(tie_counts)

    n_test_pts = len(ranks)
    empirical_cov = float(np.mean(p_values > 0.1))

    print(f"  Empirical coverage (alpha=0.1): {empirical_cov:.3f}")
    print(f"  Theoretical upper bound: {1 - 0.1 + 1/(n_cal+1):.4f}")
    print(f"  test_score=0.0 count: {zero_score_count}/{n_test_pts} ({100*zero_score_count/n_test_pts:.1f}%)")
    print(f"  Tie count (updated_cal == test_score): mean={tie_counts.mean():.1f}, max={tie_counts.max()}")
    print(f"  Rank stats: mean={ranks.mean():.1f} (expected ~{n_cal/2:.1f}), std={ranks.std():.1f}")
    print(f"  Rank distribution: min={ranks.min()}, p25={np.percentile(ranks,25):.0f}, "
          f"median={np.median(ranks):.0f}, p75={np.percentile(ranks,75):.0f}, max={ranks.max()}")

    # Check: how many test scores land in the top decile of cal scores
    frac_top10 = float(np.mean(ranks >= 0.9 * n_cal))
    print(f"  Fraction of test points with rank >= 90th percentile: {frac_top10:.3f} (uniform expect ~0.1)")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Rank Uniformity Diagnostic — {dataset_tag} | {ncm_name} | cal={n_cal}\n"
        f"Empirical coverage={empirical_cov:.3f}  "
        f"Upper bound≈{1-0.1+1/(n_cal+1):.3f}  "
        f"zero_score={100*zero_score_count/n_test_pts:.1f}%",
        fontsize=11
    )

    # Panel 1: Histogram of ranks
    ax = axes[0]
    ax.hist(ranks, bins=min(50, n_cal // 5 + 1), density=True, color='steelblue', alpha=0.7, label='Observed')
    uniform_density = 1.0 / (n_cal + 1)
    ax.axhline(uniform_density, color='red', linewidth=2, linestyle='--',
               label=f'Uniform density 1/{n_cal+1}')
    ax.set_xlabel("Rank of test_score (0=most conforming)")
    ax.set_ylabel("Density")
    ax.set_title("Rank Histogram (should be flat/uniform)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Histogram of test scores with annotation
    ax = axes[1]
    finite_ts = test_scores[np.isfinite(test_scores) & (test_scores < 1e8)]
    if len(finite_ts) > 0:
        ax.hist(finite_ts, bins=50, density=True, color='darkorange', alpha=0.7)
    ax.axvline(0.0, color='red', linewidth=2, linestyle='--', label='score=0.0')
    ax.set_xlabel("test_score (true label)")
    ax.set_ylabel("Density")
    ax.set_title(f"Test Score Distribution\n({100*zero_score_count/n_test_pts:.1f}% are exactly 0.0)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: p-value histogram (should be Uniform[0,1] under null)
    ax = axes[2]
    ax.hist(p_values, bins=20, density=True, color='green', alpha=0.7, label='Observed')
    ax.axhline(1.0, color='red', linewidth=2, linestyle='--', label='Uniform(0,1)')
    ax.set_xlabel("p-value (true label)")
    ax.set_ylabel("Density")
    ax.set_title("p-value Distribution (should be Uniform)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rank_diag_{dataset_tag}_{ncm_name}_cal{n_cal}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved -> {out_path}")

    # ------------------------------------------------------------------ #
    # Tie analysis: what fraction of cal scores == 0.0?
    # ------------------------------------------------------------------ #
    try:
        cal_scores = ncm.get_calibration_scores()
        n_cal_zero = int(np.sum(cal_scores == 0.0))
        print(f"  Calibration scores == 0.0: {n_cal_zero}/{n_cal} ({100*n_cal_zero/n_cal:.1f}%)")
        # Count how many cal scores are in each unique value
        unique_vals, counts = np.unique(cal_scores, return_counts=True)
        top_k = min(5, len(unique_vals))
        print(f"  Top-{top_k} most frequent cal score values:")
        order = np.argsort(-counts)[:top_k]
        for oi in order:
            print(f"    score={unique_vals[oi]:.6f}  count={counts[oi]}  ({100*counts[oi]/n_cal:.1f}%)")
    except Exception as e:
        print(f"  (Could not inspect cal scores: {e})")

    return {
        'ranks': ranks,
        'p_values': p_values,
        'test_scores': test_scores,
        'tie_counts': tie_counts,
        'empirical_cov': empirical_cov,
        'zero_score_frac': zero_score_count / n_test_pts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", default="output/embeddings_cifar10.pt")
    parser.add_argument("--ncms", default="nn_ratio,geodesic_nn_ratio,whitened_geodesic",
                        help="Comma-separated NCM names to diagnose")
    parser.add_argument("--cal_size", type=int, default=300)
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--output_dir", default="output/rank_diagnostic")
    args = parser.parse_args()

    print(f"Loading {args.embeddings_path} ...")
    data = torch.load(args.embeddings_path)
    X = data["embeddings"].numpy()
    y = data["labels"].numpy()
    K = len(np.unique(y))
    print(f"  {len(X)} samples | {K} classes | {X.shape[1]}D")

    dataset_tag = os.path.splitext(os.path.basename(args.embeddings_path))[0]

    for ncm_name in args.ncms.split(","):
        ncm_name = ncm_name.strip()
        run_diagnostic(X, y, ncm_name, args.cal_size, args.test_size,
                       args.seed, args.tau, args.output_dir, dataset_tag)


if __name__ == "__main__":
    main()
