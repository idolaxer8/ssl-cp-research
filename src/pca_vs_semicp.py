"""
Fair comparison: FCP variants that use the same unlabeled pool differently.

Methods:
- FCP: baseline (no unlabeled data)
- FCP+PCA: fit PCA on unlabeled, project everything, run FCP
- FCP+MS-CS: build class similarity from k-means on unlabeled, penalize FCP scores

Fairness constraints:
- Same labeled budget per trial (same stratified split)
- Same unlabeled pool
- Same test set (subsampled to TEST_SIZE, fixed across methods)
- All methods use full labeled budget as cal (no train/cal split needed)

Usage:
    python src/pca_vs_semicp.py
    python src/pca_vs_semicp.py --cal_sizes 400 600 800 --n_trials 3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_prediction import FullConformalPredictor, create_ncm
from mscs_unlabeled_experiment import build_cluster_similarity_matrix, run_fcp_with_mscs
from autoencoder_utils import EmbeddingAutoencoder

# Config
ALPHA = 0.1
SEED = 42
N_TRIALS = 5
CAL_SIZES = [300, 400, 600, 800, 1000]
TEST_SIZE = 500
PCA_DIM = 128
NCM = "geodesic_topk_mean"
MSCS_N_CLUSTERS = 20
MSCS_LAMBDA = 0.05
MSCS_TAU = -0.5  # negative = normalized by median_d^2

EMB_PATH = "output/embeddings_cifar100.pt"
TEST_PATH = "output/embeddings_cifar100_test.pt"
UNLABELED_PATH = "output/embeddings_cifar100_unlabeled.pt"
OUT_DIR = Path("output/pca_vs_semicp")


def stratified_cal(X, y, all_classes, cal_size, rng):
    """Stratified sample of cal_size from (X, y) with >=1 per class."""
    n_classes = len(all_classes)
    first = np.array([rng.choice(np.where(y == c)[0], 1, replace=False)[0]
                      for c in all_classes])
    rest = np.setdiff1d(np.arange(len(X)), first)
    n_extra = cal_size - n_classes
    if n_extra > 0:
        extra = rng.choice(rest, min(n_extra, len(rest)), replace=False)
        idx = np.concatenate([first, extra])
    else:
        idx = first[:cal_size]
    idx = rng.permutation(idx)
    return X[idx], y[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_sizes", type=int, nargs="+", default=CAL_SIZES)
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--pca_dim", type=int, default=PCA_DIM)
    parser.add_argument("--ae_dim", type=int, default=32)
    parser.add_argument("--test_size", type=int, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    data = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
    X_labeled = data["embeddings"].numpy()
    y_labeled = data["labels"].numpy()
    all_classes = np.unique(y_labeled)
    n_classes = len(all_classes)

    tdata = torch.load(TEST_PATH, map_location="cpu", weights_only=False)
    X_test_full = tdata["embeddings"].numpy()
    y_test_full = tdata["labels"].numpy()

    udata = torch.load(UNLABELED_PATH, map_location="cpu", weights_only=False)
    X_unlabeled = udata["embeddings"].numpy()

    # Stratified test subsample (fixed across all trials)
    rng_test = np.random.default_rng(args.seed)
    test_per_class = args.test_size // n_classes
    test_idx = []
    for c in all_classes:
        c_idx = np.where(y_test_full == c)[0]
        chosen = rng_test.choice(c_idx, min(test_per_class, len(c_idx)), replace=False)
        test_idx.append(chosen)
    test_idx = np.concatenate(test_idx)
    X_test = X_test_full[test_idx]
    y_test = y_test_full[test_idx]

    print(f"Labeled: {X_labeled.shape}, Unlabeled: {X_unlabeled.shape}, "
          f"Test: {X_test.shape}, Classes: {n_classes}")
    print(f"PCA dim: {args.pca_dim}, AE dim: {args.ae_dim}, NCM: {NCM}, alpha: {ALPHA}")
    print(f"Cal sizes: {args.cal_sizes}, Trials: {args.n_trials}")

    # --- Fit PCA on unlabeled pool (once, shared across trials) ---
    t_pca_fit = time.time()
    pca = PCA(n_components=args.pca_dim)
    pca.fit(X_unlabeled)
    t_pca_fit = time.time() - t_pca_fit
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA fit: {X_unlabeled.shape[1]}d -> {args.pca_dim}d, "
          f"explained variance: {explained:.3f}, fit time: {t_pca_fit:.2f}s")

    # Pre-project test (shared across trials)
    X_test_pca = pca.transform(X_test)

    # --- Fit AE on unlabeled pool (once, shared across trials) ---
    t_ae_fit = time.time()
    ae = EmbeddingAutoencoder(bottleneck_dim=args.ae_dim, seed=args.seed)
    ae.fit(X_unlabeled)
    t_ae_fit = time.time() - t_ae_fit
    ae_recon = ae.reconstruction_error(X_unlabeled)
    print(f"AE recon error: {ae_recon:.6f}, fit time: {t_ae_fit:.2f}s")

    X_test_ae = ae.transform(X_test)

    methods = ["FCP", "FCP+PCA", "FCP+AE", "FCP+MS-CS"]
    # {cal_size -> {method -> [trial_dicts]}}
    results = {cs: {m: [] for m in methods} for cs in args.cal_sizes}

    for cs in args.cal_sizes:
        t_cs = time.time()
        for trial in range(args.n_trials):
            rng = np.random.default_rng(args.seed + trial * 1000)

            # --- Same labeled split for all methods ---
            X_cal, y_cal = stratified_cal(X_labeled, y_labeled, all_classes, cs, rng)

            # --- FCP: full budget, full dim ---
            t0 = time.time()
            ncm = create_ncm(NCM, k=5)
            cp = FullConformalPredictor(ncm, alpha=ALPHA)
            cp.calibrate(X_cal, y_cal, all_classes=all_classes)
            m = cp.evaluate(X_test, y_test, verbose=False)
            results[cs]["FCP"].append({
                'coverage': m['coverage'], 'avg_set_size': m['avg_set_size'],
                'time': time.time() - t0})

            # --- FCP+PCA: full budget, PCA-projected ---
            X_cal_pca = pca.transform(X_cal)
            t0 = time.time()
            ncm_pca = create_ncm(NCM, k=5)
            cp_pca = FullConformalPredictor(ncm_pca, alpha=ALPHA)
            cp_pca.calibrate(X_cal_pca, y_cal, all_classes=all_classes)
            m = cp_pca.evaluate(X_test_pca, y_test, verbose=False)
            results[cs]["FCP+PCA"].append({
                'coverage': m['coverage'], 'avg_set_size': m['avg_set_size'],
                'time': time.time() - t0})

            # --- FCP+AE: full budget, AE-projected ---
            X_cal_ae = ae.transform(X_cal)
            t0 = time.time()
            ncm_ae = create_ncm(NCM, k=5)
            cp_ae = FullConformalPredictor(ncm_ae, alpha=ALPHA)
            cp_ae.calibrate(X_cal_ae, y_cal, all_classes=all_classes)
            m = cp_ae.evaluate(X_test_ae, y_test, verbose=False)
            results[cs]["FCP+AE"].append({
                'coverage': m['coverage'], 'avg_set_size': m['avg_set_size'],
                'time': time.time() - t0})

            # --- FCP+MS-CS: full budget, unlabeled for clustering ---
            t0 = time.time()
            (M_mscs, c2c, eff_tau, med_d2, cls_centroids,
             cls_counts, clust_centroids, clust_dists
             ) = build_cluster_similarity_matrix(
                X_unlabeled, X_cal, y_cal, all_classes,
                MSCS_N_CLUSTERS, tau=MSCS_TAU)
            cov_mscs, sz_mscs = run_fcp_with_mscs(
                X_cal, y_cal, X_test, y_test, all_classes,
                NCM, ALPHA, MSCS_LAMBDA, M_mscs)
            results[cs]["FCP+MS-CS"].append({
                'coverage': cov_mscs, 'avg_set_size': sz_mscs,
                'time': time.time() - t0})

        elapsed = time.time() - t_cs
        print(f"\ncal={cs} ({elapsed:.1f}s, {args.n_trials} trials)")
        for m_name in methods:
            vals = results[cs][m_name]
            covs = [v['coverage'] for v in vals]
            szs = [v['avg_set_size'] for v in vals]
            tms = [v['time'] for v in vals]
            flag = "OK" if np.mean(covs) >= 0.89 else "!!"
            print(f"  {m_name:15s}  cov={np.mean(covs):.3f}+/-{np.std(covs):.3f} {flag}  "
                  f"sz={np.mean(szs):.2f}+/-{np.std(szs):.2f}  "
                  f"t={np.mean(tms):.2f}s")

    # --- Save JSON ---
    def jsonify(o):
        if isinstance(o, dict):
            return {str(k): jsonify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonify(i) for i in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(out_dir / "results.json", "w") as f:
        json.dump(jsonify(results), f, indent=2)

    # --- Plot: 3-panel (set size, coverage, runtime) ---
    styles = {
        "FCP":          {"color": "#1f77b4", "ls": "-",  "marker": "o", "lw": 2.5},
        "FCP+PCA":      {"color": "#e377c2", "ls": "-",  "marker": "p", "lw": 2.5},
        "FCP+AE":       {"color": "#ff7f0e", "ls": "-",  "marker": "^", "lw": 2.5},
        "FCP+MS-CS":    {"color": "#ff9896", "ls": "-",  "marker": "d", "lw": 2.5},
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"CIFAR-100: FCP variants  (alpha={ALPHA}, PCA-{args.pca_dim}, AE-{args.ae_dim})",
                 fontsize=13, fontweight="bold")

    for m_name in methods:
        s = styles[m_name]
        xs, mu_sz, sd_sz, mu_cov, sd_cov, mu_t, sd_t = [], [], [], [], [], [], []

        for cs in args.cal_sizes:
            vals = results[cs][m_name]
            covs = [v['coverage'] for v in vals]
            szs = [v['avg_set_size'] for v in vals]
            tms = [v['time'] for v in vals]
            xs.append(cs)
            mu_cov.append(np.mean(covs)); sd_cov.append(np.std(covs))
            mu_sz.append(np.mean(szs)); sd_sz.append(np.std(szs))
            mu_t.append(np.mean(tms)); sd_t.append(np.std(tms))

        xs = np.array(xs)

        unlabeled_tag = ""
        if "PCA" in m_name:
            unlabeled_tag = f" [unlabeled: PCA-{args.pca_dim}]"
        elif "AE" in m_name:
            unlabeled_tag = f" [unlabeled: AE-{args.ae_dim}]"
        elif "MS-CS" in m_name:
            unlabeled_tag = " [unlabeled: k-means]"
        label = m_name + unlabeled_tag

        axes[0].plot(xs, mu_sz, color=s["color"], ls=s["ls"], marker=s["marker"],
                     lw=s["lw"], ms=6, label=label)
        axes[0].fill_between(xs, np.array(mu_sz) - np.array(sd_sz),
                             np.array(mu_sz) + np.array(sd_sz), alpha=0.12, color=s["color"])

        axes[1].plot(xs, np.array(mu_cov) * 100, color=s["color"], ls=s["ls"],
                     marker=s["marker"], lw=s["lw"], ms=6, label=label)
        axes[1].fill_between(xs, (np.array(mu_cov) - np.array(sd_cov)) * 100,
                             (np.array(mu_cov) + np.array(sd_cov)) * 100,
                             alpha=0.12, color=s["color"])

        axes[2].plot(xs, mu_t, color=s["color"], ls=s["ls"], marker=s["marker"],
                     lw=s["lw"], ms=6, label=label)
        axes[2].fill_between(xs, np.array(mu_t) - np.array(sd_t),
                             np.array(mu_t) + np.array(sd_t), alpha=0.12, color=s["color"])

    axes[1].axhline(90, color="black", ls="--", lw=1.2, label="Target 90%")

    axes[0].set_ylabel("Avg Prediction Set Size", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Coverage (%)", fontsize=11)
    axes[1].set_ylim(70, 102)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Runtime per trial (s)", fontsize=11)
    axes[2].set_xlabel("Labeled Budget", fontsize=11)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_dir / "fcp_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Text summary ---
    lines = [
        f"CIFAR-100: FCP variants  (alpha={ALPHA}, PCA-{args.pca_dim}, AE-{args.ae_dim}, {args.n_trials} trials)",
        f"Unlabeled pool: {X_unlabeled.shape[0]} points, PCA fit: {t_pca_fit:.2f}s, AE fit: {t_ae_fit:.2f}s",
        f"MS-CS: n_clusters={MSCS_N_CLUSTERS}, lambda={MSCS_LAMBDA}, tau={MSCS_TAU}",
        "=" * 80,
        f"{'Cal':>6}  {'Method':>15}  {'Coverage':>18}  {'Set Size':>18}  {'Time (s)':>14}",
        "-" * 80,
    ]
    for cs in args.cal_sizes:
        first = True
        for m_name in methods:
            vals = results[cs][m_name]
            covs = [v['coverage'] for v in vals]
            szs = [v['avg_set_size'] for v in vals]
            tms = [v['time'] for v in vals]
            prefix = f"{cs:>6}" if first else " " * 6
            lines.append(
                f"{prefix}  {m_name:>15}  "
                f"{np.mean(covs)*100:5.1f}%+/-{np.std(covs)*100:4.1f}%  "
                f"{np.mean(szs):7.2f}+/-{np.std(szs):5.2f}  "
                f"{np.mean(tms):6.2f}+/-{np.std(tms):4.2f}")
            first = False
        lines.append("-" * 80)

    txt = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(txt)

    print(f"\nSaved: {out_dir / 'results.json'}")
    print(f"Saved: {out_dir / 'fcp_comparison.png'}")
    print(f"Saved: {out_dir / 'summary.txt'}")
    print("\n" + txt)


if __name__ == "__main__":
    main()
