"""
SemiCP experiment: compare Split CP baselines (THR/APS) with SemiCP (APS/RAPS)
against Full CP and FCP+MS-CS.

This addresses the reviewer question: "Is FCP's advantage just because it
uses more data, or because it's a better method?" by giving Split CP access
to the same unlabeled pool via the NNM augmentation of Zhou et al. (2025).

Usage:
    python src/semicp_experiment.py
    python src/semicp_experiment.py --datasets cifar100 --cal_sizes 300 600 --n_trials 3
    python src/semicp_experiment.py --datasets cifar100 --unlabeled_sizes 0 1000 5000 10000
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_prediction import (
    FullConformalPredictor,
    SemiCP,
    SoftmaxSplitCP,
    create_ncm,
)
from mscs_unlabeled_experiment import (
    build_cluster_similarity_matrix,
    run_fcp_with_mscs,
)

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    "cifar100": {
        "labeled": "output/embeddings_cifar100.pt",
        "unlabeled": "output/embeddings_cifar100_unlabeled.pt",
        "test": "output/embeddings_cifar100_test.pt",
        "n_classes": 100,
        "pca_dim": 128,
    },
    "cub200": {
        "labeled": "output/embeddings_cub200.pt",
        "unlabeled": None,  # carved from labeled pool
        "test": None,
        "n_classes": 200,
        "pca_dim": 512,
        "unlabeled_carve_per_class": 28,  # ~5600 total from ~50/class
    },
    "miniimagenet": {
        # HuggingFace timm/mini-imagenet: train (labeled) + val (unlabeled) share
        # the same 100-class label space, so val is a naturally disjoint pool —
        # same train/test pattern as CIFAR-10/100.
        "labeled": "output/embeddings_miniimagenet.pt",
        "unlabeled": "output/embeddings_miniimagenet_unlabeled.pt",
        "test": None,
        "n_classes": 100,
        "pca_dim": 128,
    },
    "cifar10": {
        "labeled": "output/embeddings_cifar10.pt",
        "unlabeled": "output/embeddings_cifar10_unlabeled.pt",  # CIFAR-10 test split, naturally disjoint
        "test": None,
        "n_classes": 10,
        "pca_dim": 64,  # K=10 — lower-dim than CIFAR-100/miniImageNet (K=100)
    },
}

ALPHA = 0.1
SEED = 42
NCM_FCP = "geodesic_topk_mean"
SPLIT_TRAIN_RATIO = 0.5
# MS-CS defaults — matched to ablation (cs_ablation.py)
MSCS_LAMBDA = 0.05
MSCS_N_CLUSTERS = 20
MSCS_TAU_MULT = 0.5


# ---------------------------------------------------------------------------
# Stratified split utility
# ---------------------------------------------------------------------------

def balanced_sample(X, y, all_classes, n_total, rng):
    """Sample exactly n_total // K per class (balanced). Returns indices."""
    K = len(all_classes)
    per_class = n_total // K
    idx = []
    for c in all_classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, min(per_class, len(c_idx)), replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    return rng.permutation(idx)


def stratified_split(X, y, all_classes, cal_size, test_size, rng,
                     unlabeled_size=0):
    """
    Balanced split: pool -> cal + test (+ optional unlabeled carve-out).
    Each subset gets exactly size // K per class.

    Returns (X_cal, y_cal, X_test, y_test, X_unlabeled_out).
    X_unlabeled_out is empty array if unlabeled_size=0.
    """
    K = len(all_classes)
    cal_per_class = cal_size // K
    test_per_class = test_size // K

    cal_idx, test_idx = [], []
    for c in all_classes:
        c_idx = np.where(y == c)[0]
        c_idx = rng.permutation(c_idx)
        cal_idx.append(c_idx[:cal_per_class])
        test_idx.append(c_idx[cal_per_class:cal_per_class + test_per_class])
    cal_idx = rng.permutation(np.concatenate(cal_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))

    X_unlabeled_out = np.empty((0, X.shape[1]))
    return X[cal_idx], y[cal_idx], X[test_idx], y[test_idx], X_unlabeled_out


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

def random_train_cal_split(X, y, train_ratio, rng):
    """Random (non-stratified) split to preserve exchangeability for SCP."""
    n = len(X)
    perm = rng.permutation(n)
    n_tr = int(round(train_ratio * n))
    tr, ca = perm[:n_tr], perm[n_tr:]
    return X[tr], y[tr], X[ca], y[ca]


def run_trial(X_cal, y_cal, X_test, y_test, X_unlabeled, all_classes,
              alpha, ncm_fcp, split_train_ratio, rng,
              pca_model=None, X_test_pca=None, X_unlabeled_pca=None,
              mscs_lambda=MSCS_LAMBDA, mscs_n_clusters=MSCS_N_CLUSTERS,
              mscs_tau_mult=MSCS_TAU_MULT):
    """
    Run all methods for one trial and return a dict of metrics.

    Methods:
    - SCP-THR/APS/RAPS: Split CP without NNM
    - SemiCP-THR/APS/RAPS: Split CP + NNM from unlabeled
    - FCP: Full Conformal Prediction (no unlabeled)
    - FCP+PCA: FCP on PCA-projected embeddings (if pca_model provided)
    - FCP+PCA+MS-CS: FCP+PCA + MS-CS penalty (clusters built in PCA space)
    """
    results = {}
    n_cal = len(X_cal)
    n_classes = len(all_classes)

    # --- Random train/cal split for SCP/SemiCP (preserves exchangeability) ---
    X_train, y_train, X_cal_split, y_cal_split = random_train_cal_split(
        X_cal, y_cal, split_train_ratio, rng)
    n_train = len(X_train)
    n_cal_split_len = len(X_cal_split)

    scp_methods = ["SCP-THR", "SCP-APS", "SCP-RAPS",
                   "SemiCP-THR", "SemiCP-APS", "SemiCP-RAPS"]

    if n_train < 2 or n_cal_split_len < 2:
        for method in scp_methods:
            results[method] = {
                'coverage': np.nan, 'avg_set_size': np.nan,
                'time': np.nan, 'classifier_accuracy': np.nan
            }
    else:
        for score_fn in ["THR", "APS", "RAPS"]:
            # SCP (no NNM)
            t0 = time.time()
            scp = SemiCP(alpha=alpha, score_fn=score_fn)
            scp.fit(X_train, y_train)
            scp.calibrate(X_cal_split, y_cal_split, all_classes=all_classes)
            m = scp.evaluate(X_test, y_test, verbose=False)
            results[f"SCP-{score_fn}"] = {
                'coverage': m['coverage'], 'avg_set_size': m['avg_set_size'],
                'time': time.time() - t0,
                'classifier_accuracy': m['classifier_accuracy'],
            }

            # SemiCP (with NNM)
            t0 = time.time()
            semi = SemiCP(alpha=alpha, score_fn=score_fn)
            semi.fit(X_train, y_train)
            semi.calibrate(X_cal_split, y_cal_split,
                           X_unlabeled=X_unlabeled, all_classes=all_classes)
            m = semi.evaluate(X_test, y_test, verbose=False)
            results[f"SemiCP-{score_fn}"] = {
                'coverage': m['coverage'], 'avg_set_size': m['avg_set_size'],
                'time': time.time() - t0,
                'classifier_accuracy': m['classifier_accuracy'],
                'n_nnm': m['n_nnm_scores'],
            }

    # --- FCP (Full CP, uses all labeled budget) ---
    # approved: legacy whitened-NCM experiment (cal-fit whitening, non-exchangeable)
    t0 = time.time()
    ncm = create_ncm(ncm_fcp, k=5, allow_nonexchangeable=True)
    cp = FullConformalPredictor(ncm, alpha=alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)
    m = cp.evaluate(X_test, y_test, verbose=False)
    results["FCP"] = {
        'coverage': m['coverage'], 'avg_set_size': m['avg_set_size'],
        'time': time.time() - t0
    }

    # --- FCP+PCA (if PCA model provided) ---
    if pca_model is not None and X_test_pca is not None:
        t0 = time.time()
        X_cal_pca = pca_model.transform(X_cal)
        ncm_pca = create_ncm(ncm_fcp, k=5, allow_nonexchangeable=True)
        cp_pca = FullConformalPredictor(ncm_pca, alpha=alpha)
        cp_pca.calibrate(X_cal_pca, y_cal, all_classes=all_classes)
        m = cp_pca.evaluate(X_test_pca, y_test, verbose=False)
        results["FCP+PCA"] = {
            'coverage': m['coverage'], 'avg_set_size': m['avg_set_size'],
            'time': time.time() - t0
        }

    # --- FCP+PCA+MS-CS (the lead method — clusters built in PCA space) ---
    if (pca_model is not None and X_test_pca is not None
            and X_unlabeled_pca is not None and len(X_unlabeled_pca) > 0):
        t0 = time.time()
        X_cal_pca = pca_model.transform(X_cal)
        # Build cluster-similarity matrix in PCA space (4x faster than 768d)
        M, *_ = build_cluster_similarity_matrix(
            X_unlabeled_pca, X_cal_pca, y_cal, all_classes,
            mscs_n_clusters, tau=-mscs_tau_mult)
        cov, sz = run_fcp_with_mscs(
            X_cal_pca, y_cal, X_test_pca, y_test, all_classes,
            ncm_fcp, alpha, mscs_lambda, M, exchangeable=False,
            allow_nonexchangeable=True)
        results["FCP+PCA+MS-CS"] = {
            'coverage': cov, 'avg_set_size': sz,
            'time': time.time() - t0
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    "SCP-THR":      {"color": "#2ca02c", "ls": ":",  "marker": "^", "lw": 1.5},
    "SCP-APS":      {"color": "#9467bd", "ls": ":",  "marker": "v", "lw": 1.5},
    "SCP-RAPS":     {"color": "#8c564b", "ls": ":",  "marker": "<", "lw": 1.5},
    "SemiCP-THR":   {"color": "#17becf", "ls": "--", "marker": "P", "lw": 2.0},
    "SemiCP-APS":   {"color": "#d62728", "ls": "--", "marker": "s", "lw": 2.0},
    "SemiCP-RAPS":  {"color": "#ff7f0e", "ls": "--", "marker": "D", "lw": 2.0},
    "FCP":          {"color": "#1f77b4", "ls": "-",  "marker": "o", "lw": 2.5},
    "FCP+PCA":      {"color": "#e377c2", "ls": "-",  "marker": "p", "lw": 2.5},
    "FCP+PCA+MS-CS":{"color": "#bcbd22", "ls": "-",  "marker": "*", "lw": 3.0},
}


def plot_results(all_results, cal_sizes, methods, alpha, title, out_path):
    """Plot coverage and set size vs cal size for all methods."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for method in methods:
        style = METHOD_STYLES.get(method, {"color": "gray", "ls": "-", "marker": ".", "lw": 1})
        means_sz, stds_sz = [], []
        means_cov, stds_cov = [], []
        valid_cs = []

        for cs in cal_sizes:
            vals = all_results.get(cs, {}).get(method, [])
            if not vals:
                continue
            covs = [v['coverage'] for v in vals if not np.isnan(v['coverage'])]
            szs = [v['avg_set_size'] for v in vals if not np.isnan(v['avg_set_size'])]
            if covs:
                valid_cs.append(cs)
                means_cov.append(np.mean(covs))
                stds_cov.append(np.std(covs))
                means_sz.append(np.mean(szs))
                stds_sz.append(np.std(szs))

        if not valid_cs:
            continue

        xs = np.array(valid_cs)
        mu_sz = np.array(means_sz)
        sd_sz = np.array(stds_sz)
        mu_cov = np.array(means_cov)
        sd_cov = np.array(stds_cov)

        axes[0].plot(xs, mu_sz, color=style["color"], ls=style["ls"],
                     marker=style["marker"], lw=style["lw"], ms=6, label=method)
        axes[0].fill_between(xs, mu_sz - sd_sz, mu_sz + sd_sz,
                             alpha=0.12, color=style["color"])

        axes[1].plot(xs, mu_cov * 100, color=style["color"], ls=style["ls"],
                     marker=style["marker"], lw=style["lw"], ms=6, label=method)
        axes[1].fill_between(xs, (mu_cov - sd_cov) * 100, (mu_cov + sd_cov) * 100,
                             alpha=0.12, color=style["color"])

    target = (1 - alpha) * 100
    axes[1].axhline(target, color="black", ls="--", lw=1.2, label=f"Target {target:.0f}%")

    axes[0].set_ylabel("Avg Prediction Set Size", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Coverage (%)", fontsize=11)
    axes[1].set_xlabel("Labeled Budget (cal + train)", fontsize=11)
    axes[1].set_ylim(max(0, target - 20), 102)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_unlabeled_ablation(ablation_results, unlabeled_sizes, cal_size,
                            alpha, title, out_path):
    """Plot set size vs unlabeled pool size for SemiCP variants."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    methods = ["SCP-THR", "SemiCP-THR", "SemiCP-APS", "SemiCP-RAPS", "FCP"]
    for method in methods:
        style = METHOD_STYLES.get(method, {"color": "gray", "ls": "-", "marker": ".", "lw": 1})
        means_sz, stds_sz, means_cov, stds_cov = [], [], [], []
        valid_us = []

        for us in unlabeled_sizes:
            vals = ablation_results.get(us, {}).get(method, [])
            if not vals:
                continue
            covs = [v['coverage'] for v in vals if not np.isnan(v['coverage'])]
            szs = [v['avg_set_size'] for v in vals if not np.isnan(v['avg_set_size'])]
            if covs:
                valid_us.append(us)
                means_cov.append(np.mean(covs))
                stds_cov.append(np.std(covs))
                means_sz.append(np.mean(szs))
                stds_sz.append(np.std(szs))

        if not valid_us:
            continue
        xs = np.array(valid_us)
        axes[0].plot(xs, means_sz, color=style["color"], ls=style["ls"],
                     marker=style["marker"], lw=style["lw"], ms=6, label=method)
        axes[1].plot(xs, np.array(means_cov) * 100, color=style["color"],
                     ls=style["ls"], marker=style["marker"], lw=style["lw"],
                     ms=6, label=method)

    target = (1 - alpha) * 100
    axes[1].axhline(target, color="black", ls="--", lw=1.2, label=f"Target {target:.0f}%")

    axes[0].set_ylabel("Avg Set Size", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Coverage (%)", fontsize=11)
    axes[1].set_xlabel("Unlabeled Pool Size", fontsize=11)
    axes[1].set_ylim(max(0, target - 20), 102)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _jsonify(obj):
    """Recursively make obj JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description="SemiCP experiment")
    parser.add_argument("--datasets", nargs="+", default=["cifar100"],
                        choices=list(DATASETS.keys()))
    parser.add_argument("--cal_sizes", type=int, nargs="+",
                        default=[300, 400, 600, 800, 1000])
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--unlabeled_sizes", type=int, nargs="+",
                        default=None,
                        help="Unlabeled pool sizes for ablation (default: use all available)")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="output/semicp_experiments")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        labeled_path = cfg["labeled"]

        if not Path(labeled_path).exists():
            print(f"[SKIP] {ds_name}: embeddings not found at {labeled_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'='*70}")

        # Load labeled data
        data = torch.load(labeled_path, map_location="cpu", weights_only=False)
        X_labeled = data["embeddings"].numpy()
        y_labeled = data["labels"].numpy()
        all_classes = np.unique(y_labeled)
        n_classes = len(all_classes)
        print(f"Labeled: {X_labeled.shape}, {n_classes} classes")

        # Load external test set if available (stratified subsample)
        X_test_ext, y_test_ext = None, None
        if cfg.get("test") and Path(cfg["test"]).exists():
            tdata = torch.load(cfg["test"], map_location="cpu", weights_only=False)
            X_test_full = tdata["embeddings"].numpy()
            y_test_full = tdata["labels"].numpy()

            # Stratified subsample to test_size
            rng_test = np.random.default_rng(args.seed)
            test_per_class = args.test_size // n_classes
            test_idx = []
            for c in all_classes:
                c_idx = np.where(y_test_full == c)[0]
                chosen = rng_test.choice(c_idx, min(test_per_class, len(c_idx)),
                                         replace=False)
                test_idx.append(chosen)
            test_idx = np.concatenate(test_idx)
            X_test_ext = X_test_full[test_idx]
            y_test_ext = y_test_full[test_idx]
            print(f"External test: {X_test_ext.shape} (from {X_test_full.shape[0]})")

        # Load external unlabeled pool if available, or carve from labeled
        X_unlabeled_ext = None
        X_labeled_remaining = X_labeled  # data available for cal/test after carving
        y_labeled_remaining = y_labeled
        if cfg.get("unlabeled") and Path(cfg["unlabeled"]).exists():
            udata = torch.load(cfg["unlabeled"], map_location="cpu", weights_only=False)
            X_unlabeled_ext = udata["embeddings"].numpy()
            print(f"External unlabeled: {X_unlabeled_ext.shape}")
        elif cfg.get("unlabeled_carve_per_class"):
            # Carve unlabeled pool from labeled data (stratified)
            carve_per_class = cfg["unlabeled_carve_per_class"]
            rng_carve = np.random.default_rng(args.seed)
            carve_idx, remain_idx = [], []
            for c in all_classes:
                c_idx = np.where(y_labeled == c)[0]
                c_idx = rng_carve.permutation(c_idx)
                n_carve = min(carve_per_class, len(c_idx) - 5)  # keep >=5 for cal/test
                carve_idx.extend(c_idx[:n_carve])
                remain_idx.extend(c_idx[n_carve:])
            carve_idx = np.array(carve_idx)
            remain_idx = np.array(remain_idx)
            X_unlabeled_ext = X_labeled[carve_idx]
            X_labeled_remaining = X_labeled[remain_idx]
            y_labeled_remaining = y_labeled[remain_idx]
            print(f"Carved unlabeled: {X_unlabeled_ext.shape} "
                  f"(remaining labeled: {X_labeled_remaining.shape})")

        methods = ["SCP-THR", "SCP-APS", "SCP-RAPS",
                   "SemiCP-THR", "SemiCP-APS", "SemiCP-RAPS",
                   "FCP+PCA+MS-CS"]

        # ----------------------------------------------------------------
        # Fit PCA on unlabeled pool (once); pre-project unlabeled for MS-CS
        # ----------------------------------------------------------------
        pca_model, X_test_pca_ext, X_unlabeled_pca = None, None, None
        pca_dim = cfg.get("pca_dim", args.pca_dim)
        if X_unlabeled_ext is not None and len(X_unlabeled_ext) > 0:
            pca_model = PCA(n_components=pca_dim)
            pca_model.fit(X_unlabeled_ext)
            explained = pca_model.explained_variance_ratio_.sum()
            print(f"PCA: {X_unlabeled_ext.shape[1]}d -> {pca_dim}d "
                  f"(explained variance: {explained:.3f})")
            X_unlabeled_pca = pca_model.transform(X_unlabeled_ext)
            if X_test_ext is not None:
                X_test_pca_ext = pca_model.transform(X_test_ext)

        # ----------------------------------------------------------------
        # Main experiment: vary cal_size, fixed unlabeled pool
        # ----------------------------------------------------------------
        print(f"\n--- Main comparison: cal_sizes={args.cal_sizes} ---")
        all_results = {}  # {cal_size -> {method -> [trial_dicts]}}

        for cs in args.cal_sizes:
            all_results[cs] = {m: [] for m in methods}
            t_start = time.time()

            for trial in range(args.n_trials):
                rng = np.random.default_rng(args.seed + trial * 1000)
                rng_split = np.random.default_rng(args.seed + trial * 1000 + 1)

                if X_test_ext is not None:
                    # Use external test set, carve cal from labeled pool
                    X_test_t = X_test_ext
                    y_test_t = y_test_ext
                    X_test_pca_t = X_test_pca_ext

                    # Balanced cal from labeled pool (cs // K per class)
                    X_src, y_src = X_labeled_remaining, y_labeled_remaining
                    cal_idx = balanced_sample(X_src, y_src, all_classes, cs, rng)
                    X_cal_t = X_src[cal_idx]
                    y_cal_t = y_src[cal_idx]

                    # Unlabeled: use external/carved if available, else empty
                    X_u_t = X_unlabeled_ext if X_unlabeled_ext is not None else np.empty((0, X_src.shape[1]))
                else:
                    # Carve everything from remaining labeled pool
                    X_src, y_src = X_labeled_remaining, y_labeled_remaining
                    X_cal_t, y_cal_t, X_test_t, y_test_t, X_u_carved = stratified_split(
                        X_src, y_src, all_classes, cs, args.test_size, rng,
                        unlabeled_size=0)
                    X_test_pca_t = pca_model.transform(X_test_t) if pca_model else None
                    # Use pre-carved unlabeled for NNM (not trial-carved)
                    X_u_t = X_unlabeled_ext if X_unlabeled_ext is not None else np.empty((0, X_src.shape[1]))

                trial_results = run_trial(
                    X_cal_t, y_cal_t, X_test_t, y_test_t, X_u_t,
                    all_classes, args.alpha, NCM_FCP, SPLIT_TRAIN_RATIO,
                    rng_split, pca_model=pca_model, X_test_pca=X_test_pca_t,
                    X_unlabeled_pca=X_unlabeled_pca)

                for m in methods:
                    if m in trial_results:
                        all_results[cs][m].append(trial_results[m])

            elapsed = time.time() - t_start

            # Print summary
            print(f"\n  cal_size={cs} ({elapsed:.1f}s, {args.n_trials} trials)")
            for m in methods:
                vals = all_results[cs][m]
                covs = [v['coverage'] for v in vals if not np.isnan(v['coverage'])]
                szs = [v['avg_set_size'] for v in vals if not np.isnan(v['avg_set_size'])]
                if covs:
                    cov_mean = np.mean(covs)
                    sz_mean = np.mean(szs)
                    valid = "OK" if cov_mean >= 0.89 else "!!"
                    print(f"    {m:15s}  cov={cov_mean:.3f} {valid}  sz={sz_mean:.2f}")

        # Save results
        json_path = out_dir / f"{ds_name}_semicp_results.json"
        with open(json_path, "w") as f:
            json.dump(_jsonify(all_results), f, indent=2)
        print(f"\n  Saved JSON: {json_path}")

        # Plot
        plot_results(all_results, args.cal_sizes, methods, args.alpha,
                     f"{ds_name.upper()} -- SemiCP vs FCP (alpha={args.alpha})",
                     str(out_dir / f"{ds_name}_semicp_comparison.png"))

        # ----------------------------------------------------------------
        # Unlabeled ablation: fix cal_size, vary unlabeled pool size
        # ----------------------------------------------------------------
        if args.unlabeled_sizes and X_unlabeled_ext is not None:
            ablation_cal = args.cal_sizes[len(args.cal_sizes) // 2]  # middle cal size
            print(f"\n--- Unlabeled ablation: cal={ablation_cal}, "
                  f"unlabeled_sizes={args.unlabeled_sizes} ---")

            ablation_methods = ["SCP-THR", "SemiCP-THR", "SemiCP-APS",
                               "SemiCP-RAPS", "FCP"]
            ablation_results = {}  # {u_size -> {method -> [trial_dicts]}}

            for us in args.unlabeled_sizes:
                ablation_results[us] = {m: [] for m in ablation_methods}

                for trial in range(args.n_trials):
                    rng = np.random.default_rng(args.seed + trial * 1000)
                    rng_split = np.random.default_rng(args.seed + trial * 1000 + 1)

                    # Same cal/test split as main experiment
                    X_src, y_src = X_labeled_remaining, y_labeled_remaining
                    if X_test_ext is not None:
                        X_test_t = X_test_ext
                        y_test_t = y_test_ext
                        cal_idx = balanced_sample(X_src, y_src, all_classes,
                                                  ablation_cal, rng)
                        X_cal_t = X_src[cal_idx]
                        y_cal_t = y_src[cal_idx]
                    else:
                        X_cal_t, y_cal_t, X_test_t, y_test_t, _ = stratified_split(
                            X_src, y_src, all_classes, ablation_cal,
                            args.test_size, rng)

                    # Subsample unlabeled pool to size us
                    if us == 0:
                        X_u_t = np.empty((0, X_labeled.shape[1]))
                    elif X_unlabeled_ext is not None:
                        u_idx = rng.choice(len(X_unlabeled_ext),
                                           min(us, len(X_unlabeled_ext)), replace=False)
                        X_u_t = X_unlabeled_ext[u_idx]
                    else:
                        X_u_t = np.empty((0, X_labeled.shape[1]))

                    trial_results = run_trial(
                        X_cal_t, y_cal_t, X_test_t, y_test_t, X_u_t,
                        all_classes, args.alpha, NCM_FCP, SPLIT_TRAIN_RATIO,
                        rng_split)

                    for m in ablation_methods:
                        if m in trial_results:
                            ablation_results[us][m].append(trial_results[m])

                # Print
                print(f"  unlabeled={us}")
                for m in ablation_methods:
                    vals = ablation_results[us][m]
                    covs = [v['coverage'] for v in vals if not np.isnan(v['coverage'])]
                    szs = [v['avg_set_size'] for v in vals if not np.isnan(v['avg_set_size'])]
                    if covs:
                        print(f"    {m:15s}  cov={np.mean(covs):.3f}  sz={np.mean(szs):.2f}")

            # Save ablation
            ablation_path = out_dir / f"{ds_name}_unlabeled_ablation.json"
            with open(ablation_path, "w") as f:
                json.dump(_jsonify(ablation_results), f, indent=2)

            plot_unlabeled_ablation(
                ablation_results, args.unlabeled_sizes, ablation_cal,
                args.alpha,
                f"{ds_name.upper()} -- SemiCP Unlabeled Ablation (cal={ablation_cal})",
                str(out_dir / f"{ds_name}_unlabeled_ablation.png"))

        # ----------------------------------------------------------------
        # Print summary table
        # ----------------------------------------------------------------
        print(f"\n{'='*70}")
        print(f"SUMMARY: {ds_name.upper()}")
        print(f"{'='*70}")
        hdr = f"{'Cal':>6}  {'Method':>15}  {'Coverage':>10}  {'Set Size':>10}"
        print(hdr)
        print("-" * 50)
        for cs in args.cal_sizes:
            first = True
            for m in methods:
                vals = all_results[cs][m]
                covs = [v['coverage'] for v in vals if not np.isnan(v['coverage'])]
                szs = [v['avg_set_size'] for v in vals if not np.isnan(v['avg_set_size'])]
                if not covs:
                    continue
                prefix = f"{cs:>6}" if first else " " * 6
                cov_str = f"{np.mean(covs)*100:.1f}%+/-{np.std(covs)*100:.1f}%"
                sz_str = f"{np.mean(szs):.2f}+/-{np.std(szs):.2f}"
                print(f"{prefix}  {m:>15}  {cov_str:>10}  {sz_str:>10}")
                first = False
            print("-" * 50)

    print("\nDone.")


if __name__ == "__main__":
    main()
