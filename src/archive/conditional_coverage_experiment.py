"""
Class-Conditional Coverage Experiment.

Per-class coverage diagnostics for FCP variants vs SCP/SemiCP baselines, with
Clustered Conformal Prediction (Ding, Tibshirani & Ramdas, 2023; arXiv:2306.09335)
as the natural conditional-coverage baseline.

Reports per-class coverage AND per-class set size for every method, plus
Ding-et-al aggregate diagnostics (CovGap, worst-class coverage, fraction
under-covered, macro-coverage) and per-trial raw arrays in JSON.

Usage:
    # Full default (CIFAR-100 + miniImageNet, ~10-30 min):
    python src/conditional_coverage_experiment.py

    # Smoke test (~1-2 min on 4GB GPU):
    python src/conditional_coverage_experiment.py \
        --datasets cifar100 --cal_sizes 400 --test_size 500 --n_trials 2 \
        --output_dir output/conditional_coverage_smoke
"""
# ARCHIVE-CANDIDATE (review 2026-08-24): results recorded in findings §2b (CovGap + ClusterCP baseline, 2026-05-26); §10 item 8 (per-class coverage diagnostics) done; no substantive commit since. Re-flagged: the 2026-07-10 sweep's note never merged to main.

import argparse
import csv
import json
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
    ClusteredSplitCP,
    create_ncm,
)
from mscs_unlabeled_experiment import (
    build_cluster_similarity_matrix,
    run_fcp_with_mscs,
)
from semicp_experiment import (
    DATASETS,
    balanced_sample,
    stratified_split,
    random_train_cal_split,
    METHOD_STYLES as BASE_METHOD_STYLES,
    _jsonify,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA = 0.1
SEED = 42
NCM_FCP = "geodesic_topk_mean"
SPLIT_TRAIN_RATIO = 0.5
MSCS_LAMBDA = 0.05
MSCS_N_CLUSTERS = 20
MSCS_TAU_MULT = 0.5

METHODS = [
    "SCP-THR", "SCP-APS", "SCP-RAPS",
    "SemiCP-THR", "SemiCP-APS", "SemiCP-RAPS",
    "FCP", "FCP+PCA", "FCP+PCA+MS-CS",
    "ClusterCP",
]

METHOD_STYLES = dict(BASE_METHOD_STYLES)
METHOD_STYLES["ClusterCP"] = {
    "color": "#7f7f7f", "ls": "-.", "marker": "X", "lw": 2.0
}


# ---------------------------------------------------------------------------
# Per-class metric core
# ---------------------------------------------------------------------------

def per_class_metrics(prediction_sets, y_test, all_classes, alpha):
    """Per-class coverage/size + Ding-et-al aggregate diagnostics.

    Returns dict with arrays of length K=len(all_classes) and scalar diagnostics.
    Classes absent from y_test get NaN in the per-class arrays and are excluded
    from aggregate diagnostics (cov_gap_mean / worst_class_coverage / frac_undercovered).
    """
    K = len(all_classes)
    target = 1.0 - alpha
    cov_per_class = np.full(K, np.nan)
    size_per_class = np.full(K, np.nan)
    n_per_class = np.zeros(K, dtype=int)

    y_test = np.asarray(y_test)
    for i, c in enumerate(all_classes):
        ci = int(c)
        mask = (y_test == ci)
        n = int(mask.sum())
        n_per_class[i] = n
        if n == 0:
            continue
        idxs = np.where(mask)[0]
        cov_per_class[i] = float(np.mean(
            [ci in prediction_sets[j] for j in idxs]))
        size_per_class[i] = float(np.mean(
            [len(prediction_sets[j]) for j in idxs]))

    valid = ~np.isnan(cov_per_class)
    cov_gap_per_class = np.abs(cov_per_class[valid] - target) * 100.0

    marginal_cov = float(np.mean(
        [int(y_test[i]) in prediction_sets[i] for i in range(len(y_test))]))
    marginal_size = float(np.mean([len(s) for s in prediction_sets]))

    return {
        "cov_per_class": cov_per_class.tolist(),
        "size_per_class": size_per_class.tolist(),
        "n_per_class": n_per_class.tolist(),
        "marginal_coverage": marginal_cov,
        "marginal_set_size": marginal_size,
        "macro_coverage": float(np.nanmean(cov_per_class)),
        "cov_gap_mean": float(cov_gap_per_class.mean()) if cov_gap_per_class.size > 0 else float("nan"),
        "cov_gap_max": float(cov_gap_per_class.max()) if cov_gap_per_class.size > 0 else float("nan"),
        "worst_class_coverage": float(np.nanmin(cov_per_class)) if valid.any() else float("nan"),
        "frac_undercovered": float(np.mean(cov_per_class[valid] < target)) if valid.any() else float("nan"),
        "n_classes_in_test": int(valid.sum()),
    }


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

def run_trial(X_cal, y_cal, X_test, y_test, X_unlabeled, all_classes,
              alpha, ncm_fcp, split_train_ratio, rng,
              pca_model=None, X_test_pca=None, X_unlabeled_pca=None,
              cluster_n_clusters=5, cluster_n_thresh=None,
              mscs_lambda=MSCS_LAMBDA, mscs_n_clusters=MSCS_N_CLUSTERS,
              mscs_tau_mult=MSCS_TAU_MULT):
    """Run all 10 methods on one trial; return {method: metrics_dict}."""
    results = {}

    # --- Random train/cal split shared by SCP / SemiCP / ClusterCP ---
    X_train, y_train, X_cal_split, y_cal_split = random_train_cal_split(
        X_cal, y_cal, split_train_ratio, rng)

    split_ok = len(X_train) >= 2 and len(X_cal_split) >= 2

    scp_method_names = ["SCP-THR", "SCP-APS", "SCP-RAPS",
                        "SemiCP-THR", "SemiCP-APS", "SemiCP-RAPS",
                        "ClusterCP"]
    if not split_ok:
        for m in scp_method_names:
            results[m] = {"error": "insufficient cal for inner split"}
    else:
        for score_fn in ["THR", "APS", "RAPS"]:
            # Plain SCP (no NNM)
            t0 = time.time()
            scp = SemiCP(alpha=alpha, score_fn=score_fn)
            scp.fit(X_train, y_train)
            scp.calibrate(X_cal_split, y_cal_split, all_classes=all_classes)
            pred_sets = scp.predict(X_test)["prediction_sets"]
            m = per_class_metrics(pred_sets, y_test, all_classes, alpha)
            m["time"] = time.time() - t0
            results[f"SCP-{score_fn}"] = m

            # SemiCP (with NNM)
            t0 = time.time()
            semi = SemiCP(alpha=alpha, score_fn=score_fn)
            semi.fit(X_train, y_train)
            semi.calibrate(X_cal_split, y_cal_split,
                           X_unlabeled=X_unlabeled, all_classes=all_classes)
            pred_sets = semi.predict(X_test)["prediction_sets"]
            m = per_class_metrics(pred_sets, y_test, all_classes, alpha)
            m["time"] = time.time() - t0
            results[f"SemiCP-{score_fn}"] = m

        # ClusterCP (Ding et al. 2023)
        t0 = time.time()
        ccp = ClusteredSplitCP(alpha=alpha, n_clusters=cluster_n_clusters,
                               n_thresh=cluster_n_thresh)
        ccp.fit(X_train, y_train)
        ccp.calibrate(X_cal_split, y_cal_split, all_classes=all_classes)
        pred_sets = ccp.predict(X_test)["prediction_sets"]
        m = per_class_metrics(pred_sets, y_test, all_classes, alpha)
        m["time"] = time.time() - t0
        m["fraction_rare"] = ccp._fraction_rare
        m["effective_n_clusters"] = ccp._effective_n_clusters
        results["ClusterCP"] = m

    # --- FCP (Full CP) ---
    # approved: legacy whitened-NCM experiment (cal-fit whitening, non-exchangeable)
    t0 = time.time()
    ncm = create_ncm(ncm_fcp, k=5, allow_nonexchangeable=True)
    cp = FullConformalPredictor(ncm, alpha=alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)
    pred_sets = cp.predict(X_test, verbose=False)["prediction_sets"]
    m = per_class_metrics(pred_sets, y_test, all_classes, alpha)
    m["time"] = time.time() - t0
    results["FCP"] = m

    # --- FCP+PCA ---
    if pca_model is not None and X_test_pca is not None:
        t0 = time.time()
        X_cal_pca = pca_model.transform(X_cal)
        ncm_pca = create_ncm(ncm_fcp, k=5, allow_nonexchangeable=True)
        cp_pca = FullConformalPredictor(ncm_pca, alpha=alpha)
        cp_pca.calibrate(X_cal_pca, y_cal, all_classes=all_classes)
        pred_sets = cp_pca.predict(X_test_pca, verbose=False)["prediction_sets"]
        m = per_class_metrics(pred_sets, y_test, all_classes, alpha)
        m["time"] = time.time() - t0
        results["FCP+PCA"] = m

    # --- FCP+PCA+MS-CS (lead method) ---
    if (pca_model is not None and X_test_pca is not None
            and X_unlabeled_pca is not None and len(X_unlabeled_pca) > 0):
        t0 = time.time()
        X_cal_pca = pca_model.transform(X_cal)
        M, *_ = build_cluster_similarity_matrix(
            X_unlabeled_pca, X_cal_pca, y_cal, all_classes,
            mscs_n_clusters, tau=-mscs_tau_mult)
        _, _, pred_sets = run_fcp_with_mscs(
            X_cal_pca, y_cal, X_test_pca, y_test, all_classes,
            ncm_fcp, alpha, mscs_lambda, M, exchangeable=False,
            return_sets=True, allow_nonexchangeable=True)
        m = per_class_metrics(pred_sets, y_test, all_classes, alpha)
        m["time"] = time.time() - t0
        results["FCP+PCA+MS-CS"] = m

    return results


# ---------------------------------------------------------------------------
# Aggregation across trials
# ---------------------------------------------------------------------------

def aggregate_method(trials):
    """Aggregate a list of per_class_metrics dicts across trials.

    For per-class arrays: nanmean across trials.
    For scalars: mean + std across trials (NaN-aware).
    """
    if not trials or all("error" in t for t in trials):
        return {"n_trials_valid": 0}

    trials = [t for t in trials if "error" not in t]
    K = len(trials[0]["cov_per_class"])

    cov_arr = np.array([t["cov_per_class"] for t in trials])  # (n_trials, K)
    size_arr = np.array([t["size_per_class"] for t in trials])
    n_arr = np.array([t["n_per_class"] for t in trials])

    agg = {
        "n_trials_valid": len(trials),
        "cov_per_class_mean": np.nanmean(cov_arr, axis=0).tolist(),
        "cov_per_class_std": np.nanstd(cov_arr, axis=0).tolist(),
        "size_per_class_mean": np.nanmean(size_arr, axis=0).tolist(),
        "n_per_class_total": n_arr.sum(axis=0).tolist(),
    }

    for key in ("marginal_coverage", "marginal_set_size", "macro_coverage",
                "cov_gap_mean", "cov_gap_max", "worst_class_coverage",
                "frac_undercovered", "time"):
        vals = np.array([t.get(key, np.nan) for t in trials], dtype=float)
        agg[f"{key}_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
        agg[f"{key}_std"] = float(np.nanstd(vals)) if vals.size else float("nan")

    # Aggregate ClusterCP-specific diagnostics if present
    if "fraction_rare" in trials[0]:
        agg["fraction_rare_mean"] = float(np.mean(
            [t.get("fraction_rare", np.nan) for t in trials]))
        agg["effective_n_clusters_mean"] = float(np.mean(
            [t.get("effective_n_clusters", np.nan) for t in trials]))

    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_covgap_bar(agg_by_method, out_path, ds_name, cal_size, alpha):
    """Horizontal bar: CovGap (mean over classes of |cov - target|*100) per method."""
    methods = [m for m in METHODS if agg_by_method.get(m, {}).get("n_trials_valid", 0)]
    gaps = [agg_by_method[m]["cov_gap_mean_mean"] for m in methods]
    worst = [agg_by_method[m]["worst_class_coverage_mean"] for m in methods]

    order = np.argsort(gaps)
    methods = [methods[i] for i in order]
    gaps = [gaps[i] for i in order]
    worst = [worst[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [METHOD_STYLES.get(m, {}).get("color", "gray") for m in methods]
    bars = ax.barh(methods, gaps, color=colors, alpha=0.85)
    for bar, w in zip(bars, worst):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"worst={w:.2f}", va="center", fontsize=8)
    ax.set_xlabel("CovGap = mean_c |cov_c - (1-alpha)| (pp)")
    ax.set_title(f"{ds_name.upper()} cal={cal_size} — class-conditional CovGap "
                 f"(lower=better, target {1-alpha:.0%})")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_perclass_hist(agg_by_method, out_path, ds_name, cal_size, alpha):
    """Small-multiples: histogram of per-class coverage per method."""
    methods = [m for m in METHODS if agg_by_method.get(m, {}).get("n_trials_valid", 0)]
    n = len(methods)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    target = 1 - alpha
    bins = np.linspace(0, 1, 21)
    for i, m in enumerate(methods):
        ax = axes[i]
        cov = np.array(agg_by_method[m]["cov_per_class_mean"])
        cov = cov[~np.isnan(cov)]
        color = METHOD_STYLES.get(m, {}).get("color", "gray")
        ax.hist(cov, bins=bins, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(target, color="black", ls="--", lw=1.2, label=f"target {target:.0%}")
        gap = agg_by_method[m]["cov_gap_mean_mean"]
        worst = agg_by_method[m]["worst_class_coverage_mean"]
        ax.set_title(f"{m}\nCovGap={gap:.2f}pp worst={worst:.2f}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{ds_name.upper()} cal={cal_size} — per-class coverage distribution",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sorted_perclass(agg_by_method, out_path, ds_name, cal_size, alpha):
    """For each method, sorted per-class coverages — reveals tail behavior."""
    methods = [m for m in METHODS if agg_by_method.get(m, {}).get("n_trials_valid", 0)]
    fig, ax = plt.subplots(figsize=(9, 5))
    target = 1 - alpha
    for m in methods:
        cov = np.array(agg_by_method[m]["cov_per_class_mean"])
        cov = np.sort(cov[~np.isnan(cov)])
        style = METHOD_STYLES.get(m, {"color": "gray", "ls": "-", "lw": 1.5})
        ax.plot(np.arange(len(cov)), cov, color=style.get("color", "gray"),
                ls=style.get("ls", "-"), lw=style.get("lw", 1.5), label=m)
    ax.axhline(target, color="black", ls="--", lw=1.2, label=f"target {target:.0%}")
    ax.set_xlabel("Class rank (sorted by coverage)")
    ax.set_ylabel("Per-class coverage")
    ax.set_title(f"{ds_name.upper()} cal={cal_size} — per-class coverage (sorted)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_size_vs_class(agg_by_method, out_path, ds_name, cal_size):
    """Per-class set size scatter: which classes blow up sets, per method."""
    methods = [m for m in METHODS if agg_by_method.get(m, {}).get("n_trials_valid", 0)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in methods:
        sz = np.array(agg_by_method[m]["size_per_class_mean"])
        idx = np.arange(len(sz))
        valid = ~np.isnan(sz)
        style = METHOD_STYLES.get(m, {"color": "gray", "marker": "."})
        ax.scatter(idx[valid], sz[valid], s=12, alpha=0.55,
                   color=style.get("color", "gray"),
                   marker=style.get("marker", "."), label=m)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Mean prediction-set size for class")
    ax.set_yscale("log")
    ax.set_title(f"{ds_name.upper()} cal={cal_size} — per-class set size")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_covgap_vs_calsize(by_cal, out_path, ds_name, alpha):
    """CovGap vs cal_size, one line per method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cal_sizes = sorted(by_cal.keys())
    for m in METHODS:
        xs, ys, yerrs = [], [], []
        for cs in cal_sizes:
            agg = by_cal[cs].get(m, {})
            if agg.get("n_trials_valid", 0) == 0:
                continue
            xs.append(cs)
            ys.append(agg["cov_gap_mean_mean"])
            yerrs.append(agg["cov_gap_mean_std"])
        if not xs:
            continue
        style = METHOD_STYLES.get(m, {"color": "gray", "ls": "-", "marker": "o", "lw": 1.5})
        ax.errorbar(xs, ys, yerr=yerrs, color=style.get("color", "gray"),
                    ls=style.get("ls", "-"), marker=style.get("marker", "o"),
                    lw=style.get("lw", 1.5), capsize=3, label=m)
    ax.set_xlabel("Calibration size")
    ax.set_ylabel("CovGap (pp)")
    ax.set_title(f"{ds_name.upper()} — CovGap vs cal size (lower=better, target {1-alpha:.0%})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Class-conditional coverage experiment")
    parser.add_argument("--datasets", nargs="+", default=["cifar100", "miniimagenet"],
                        choices=list(DATASETS.keys()))
    parser.add_argument("--cal_sizes", type=int, nargs="+", default=[400, 800])
    parser.add_argument("--test_size", type=int, default=2000,
                        help="Test points (stratified). For K=100 default 2000 -> 20/class.")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ncm", type=str, default=NCM_FCP)
    parser.add_argument("--cluster_n_clusters", type=int, default=5,
                        help="Target n_clusters for ClusterCP (Ding et al. default ~5).")
    parser.add_argument("--cluster_n_thresh", type=int, default=None,
                        help="Minimum cal samples/class to be clustered. "
                             "Default: ceil((n_quantile_levels+1)/alpha).")
    parser.add_argument("--output_dir", type=str, default="output/conditional_coverage")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_rows = []

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        labeled_path = cfg["labeled"]
        if not Path(labeled_path).exists():
            print(f"[SKIP] {ds_name}: {labeled_path} not found")
            continue

        print(f"\n{'='*70}\nDataset: {ds_name.upper()}\n{'='*70}")
        data = torch.load(labeled_path, map_location="cpu", weights_only=False)
        X_labeled = data["embeddings"].numpy()
        y_labeled = data["labels"].numpy()
        all_classes = np.unique(y_labeled)
        n_classes = len(all_classes)
        print(f"Labeled: {X_labeled.shape}, {n_classes} classes")

        # External test set (stratified subsample)
        X_test_ext, y_test_ext = None, None
        if cfg.get("test") and Path(cfg["test"]).exists():
            tdata = torch.load(cfg["test"], map_location="cpu", weights_only=False)
            X_test_full = tdata["embeddings"].numpy()
            y_test_full = tdata["labels"].numpy()
            rng_t = np.random.default_rng(args.seed)
            test_per_class = max(1, args.test_size // n_classes)
            test_idx = []
            for c in all_classes:
                c_idx = np.where(y_test_full == c)[0]
                chosen = rng_t.choice(c_idx, min(test_per_class, len(c_idx)),
                                      replace=False)
                test_idx.append(chosen)
            test_idx = np.concatenate(test_idx)
            X_test_ext = X_test_full[test_idx]
            y_test_ext = y_test_full[test_idx]
            print(f"External test: {X_test_ext.shape} ({test_per_class}/class)")

        # External / carved unlabeled
        X_unlabeled_ext = None
        X_labeled_remaining = X_labeled
        y_labeled_remaining = y_labeled
        if cfg.get("unlabeled") and Path(cfg["unlabeled"]).exists():
            udata = torch.load(cfg["unlabeled"], map_location="cpu", weights_only=False)
            X_unlabeled_ext = udata["embeddings"].numpy()
            print(f"External unlabeled: {X_unlabeled_ext.shape}")
        else:
            # Configured external unlabeled file missing OR no unlabeled config.
            # Fall back to carving a stratified pool from labeled data so that
            # FCP+PCA / FCP+PCA+MS-CS aren't silently gated off.
            if cfg.get("unlabeled") and not Path(cfg["unlabeled"]).exists():
                print(f"[INFO] {ds_name}: configured unlabeled file "
                      f"{cfg['unlabeled']} not found, carving from labeled.")
            if cfg.get("unlabeled_carve_per_class"):
                carve_per_class = cfg["unlabeled_carve_per_class"]
            else:
                # Auto-pick: leave at least labeled budget room for cal+test.
                max_cal = max(args.cal_sizes)
                per_class_needed = max(1, max_cal // n_classes) + \
                                   max(1, args.test_size // n_classes) + 5
                avail = int(min(np.bincount(y_labeled.astype(int))))
                carve_per_class = max(5, min(50, avail - per_class_needed))
                print(f"[INFO] auto-carve: {carve_per_class}/class "
                      f"(avail {avail}, reserved {per_class_needed} for cal+test+slack)")

            rng_c = np.random.default_rng(args.seed)
            carve_idx, remain_idx = [], []
            for c in all_classes:
                c_idx = np.where(y_labeled == c)[0]
                c_idx = rng_c.permutation(c_idx)
                n_carve = min(carve_per_class, len(c_idx) - 5)
                carve_idx.extend(c_idx[:n_carve])
                remain_idx.extend(c_idx[n_carve:])
            carve_idx = np.array(carve_idx)
            remain_idx = np.array(remain_idx)
            X_unlabeled_ext = X_labeled[carve_idx]
            X_labeled_remaining = X_labeled[remain_idx]
            y_labeled_remaining = y_labeled[remain_idx]
            print(f"Carved unlabeled: {X_unlabeled_ext.shape}, "
                  f"remaining: {X_labeled_remaining.shape}")

        # Fit PCA on unlabeled pool, project test once
        pca_model = None
        X_test_pca_ext = None
        X_unlabeled_pca = None
        pca_dim = cfg.get("pca_dim", 128)
        if X_unlabeled_ext is not None and len(X_unlabeled_ext) > 0:
            pca_model = PCA(n_components=pca_dim)
            pca_model.fit(X_unlabeled_ext)
            print(f"PCA: {X_unlabeled_ext.shape[1]}d -> {pca_dim}d "
                  f"(EV {pca_model.explained_variance_ratio_.sum():.3f})")
            X_unlabeled_pca = pca_model.transform(X_unlabeled_ext)
            if X_test_ext is not None:
                X_test_pca_ext = pca_model.transform(X_test_ext)

        # Run experiment
        all_trial_results = {}    # {cal -> {method -> [trial dicts]}}
        all_agg_results = {}      # {cal -> {method -> agg dict}}

        for cs in args.cal_sizes:
            print(f"\n--- cal={cs} ---")
            all_trial_results[cs] = {m: [] for m in METHODS}
            t0_cs = time.time()

            for trial in range(args.n_trials):
                rng = np.random.default_rng(args.seed + trial * 1000)
                rng_split = np.random.default_rng(args.seed + trial * 1000 + 1)

                if X_test_ext is not None:
                    X_test_t = X_test_ext
                    y_test_t = y_test_ext
                    X_test_pca_t = X_test_pca_ext
                    cal_idx = balanced_sample(X_labeled_remaining, y_labeled_remaining,
                                              all_classes, cs, rng)
                    X_cal_t = X_labeled_remaining[cal_idx]
                    y_cal_t = y_labeled_remaining[cal_idx]
                    X_u_t = X_unlabeled_ext if X_unlabeled_ext is not None \
                            else np.empty((0, X_labeled_remaining.shape[1]))
                else:
                    X_cal_t, y_cal_t, X_test_t, y_test_t, _ = stratified_split(
                        X_labeled_remaining, y_labeled_remaining,
                        all_classes, cs, args.test_size, rng)
                    X_test_pca_t = pca_model.transform(X_test_t) if pca_model else None
                    X_u_t = X_unlabeled_ext if X_unlabeled_ext is not None \
                            else np.empty((0, X_labeled_remaining.shape[1]))

                trial_res = run_trial(
                    X_cal_t, y_cal_t, X_test_t, y_test_t, X_u_t, all_classes,
                    args.alpha, args.ncm, SPLIT_TRAIN_RATIO, rng_split,
                    pca_model=pca_model, X_test_pca=X_test_pca_t,
                    X_unlabeled_pca=X_unlabeled_pca,
                    cluster_n_clusters=args.cluster_n_clusters,
                    cluster_n_thresh=args.cluster_n_thresh,
                )
                for m in METHODS:
                    if m in trial_res:
                        all_trial_results[cs][m].append(trial_res[m])

                print(f"  trial {trial+1}/{args.n_trials} done "
                      f"({time.time() - t0_cs:.1f}s elapsed)")

            # Aggregate
            all_agg_results[cs] = {m: aggregate_method(all_trial_results[cs][m])
                                   for m in METHODS}

            # Print compact summary
            print(f"\n  cal={cs} summary ({time.time() - t0_cs:.1f}s)")
            print(f"  {'Method':<15} {'Marg.Cov':>9} {'MarSz':>7} "
                  f"{'CovGap':>7} {'Worst':>6} {'Frac<':>7}")
            for m in METHODS:
                agg = all_agg_results[cs][m]
                if agg.get("n_trials_valid", 0) == 0:
                    print(f"  {m:<15}  <no valid trials>")
                    continue
                print(f"  {m:<15} {agg['marginal_coverage_mean']:>9.3f} "
                      f"{agg['marginal_set_size_mean']:>7.2f} "
                      f"{agg['cov_gap_mean_mean']:>7.2f} "
                      f"{agg['worst_class_coverage_mean']:>6.2f} "
                      f"{agg['frac_undercovered_mean']:>7.2f}")
                if m == "ClusterCP" and "fraction_rare_mean" in agg:
                    print(f"  {' ':<15} ClusterCP: fraction_rare="
                          f"{agg['fraction_rare_mean']:.2f}, "
                          f"effective_clusters="
                          f"{agg['effective_n_clusters_mean']:.1f}")

                # Flat CSV row
                summary_csv_rows.append({
                    "dataset": ds_name, "cal_size": cs, "method": m,
                    "marginal_coverage": agg["marginal_coverage_mean"],
                    "marginal_set_size": agg["marginal_set_size_mean"],
                    "macro_coverage": agg["macro_coverage_mean"],
                    "cov_gap_mean": agg["cov_gap_mean_mean"],
                    "cov_gap_max": agg["cov_gap_max_mean"],
                    "worst_class_coverage": agg["worst_class_coverage_mean"],
                    "frac_undercovered": agg["frac_undercovered_mean"],
                    "time_sec": agg["time_mean"],
                    "n_trials": agg["n_trials_valid"],
                })

            # Plots for this (ds, cal)
            tag = f"{ds_name}_cs{cs}"
            plot_covgap_bar(all_agg_results[cs],
                            out_dir / f"{tag}_covgap_bar.png",
                            ds_name, cs, args.alpha)
            plot_perclass_hist(all_agg_results[cs],
                               out_dir / f"{tag}_perclass_hist.png",
                               ds_name, cs, args.alpha)
            plot_sorted_perclass(all_agg_results[cs],
                                 out_dir / f"{tag}_sorted_perclass.png",
                                 ds_name, cs, args.alpha)
            plot_size_vs_class(all_agg_results[cs],
                               out_dir / f"{tag}_size_vs_class.png",
                               ds_name, cs)
            print(f"  Saved 4 plots prefixed {tag}_*.png")

        # Across-cal summary plot
        plot_covgap_vs_calsize(all_agg_results,
                               out_dir / f"{ds_name}_covgap_vs_calsize.png",
                               ds_name, args.alpha)

        # Per-dataset JSON: per-trial raw + aggregated
        json_path = out_dir / f"{ds_name}_conditional_coverage.json"
        with open(json_path, "w") as f:
            json.dump(_jsonify({
                "trial_results": all_trial_results,
                "aggregated": all_agg_results,
                "config": {
                    "alpha": args.alpha,
                    "n_trials": args.n_trials,
                    "ncm": args.ncm,
                    "pca_dim": pca_dim,
                    "cluster_n_clusters": args.cluster_n_clusters,
                    "cluster_n_thresh": args.cluster_n_thresh,
                    "test_size": args.test_size,
                    "n_classes": int(n_classes),
                },
            }), f, indent=2)
        print(f"\n  Saved JSON: {json_path}")

    # Write flat summary CSV
    if summary_csv_rows:
        csv_path = out_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_csv_rows)
        print(f"\nSaved flat summary CSV: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
