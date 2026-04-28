"""
Multi-dataset experiment runner: Full CP vs CV+ vs Split CP with DINOv2 embeddings.

For each dataset (cifar10, cifar100, stl10, eurosat) runs Full CP vs CV+ vs Split CP
and generates a cross-dataset summary plot.

Usage:
    python src/run_multi_dataset_experiments.py
    python src/run_multi_dataset_experiments.py --datasets cifar10 stl10
    python src/run_multi_dataset_experiments.py --datasets eurosat --skip_base
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make sure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_prediction import (
    FullConformalPredictor,
    CrossValidationPlusPredictor,
    SoftmaxSplitCP,
    create_ncm,
    cal_test_split,
    train_cal_test_split,
)


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    "cifar10": {
        "embeddings": "output/embeddings_cifar10.pt",
        "n_classes": 10,
        # Dense before convergence (~150), sparse after, max 400
        "cal_sizes": [25, 50, 75, 100, 125, 150, 200, 300, 400],
        "test_size": 500,
        "color": "#1f77b4",
        "marker": "o",
    },
    "cifar100": {
        "embeddings": "output/embeddings_cifar100.pt",
        "n_classes": 100,
        # Crossover for 100-class (from existing results) around 400-700
        "cal_sizes": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "test_size": 500,
        "color": "#ff7f0e",
        "marker": "s",
    },
    "eurosat": {
        "embeddings": "output/embeddings_eurosat.pt",
        "n_classes": 10,
        # Dense before convergence (~150), sparse after, max 400
        "cal_sizes": [25, 50, 75, 100, 125, 150, 200, 300, 400],
        "test_size": 500,
        "color": "#d62728",
        "marker": "D",
    },
    "flowers102": {
        "embeddings": "output/embeddings_flowers102.pt",
        "n_classes": 102,
        # Needs ≥2/class = 204 minimum; crossover expected around 600–800
        "cal_sizes": [250, 400, 600, 800, 1000, 1500],
        "test_size": 500,
        "color": "#9467bd",
        "marker": "P",
    },
    "cub200": {
        "embeddings": "output/embeddings_cub200.pt",
        "n_classes": 200,
        # Needs ≥2/class = 400 minimum; crossover expected around 800–1200
        "cal_sizes": [400, 600, 800, 1000, 1500, 2000],
        "test_size": 500,
        "color": "#8c564b",
        "marker": "X",
    },
    "miniimagenet": {
        "embeddings": "output/embeddings_miniimagenet.pt",
        "n_classes": 100,
        # 100 classes × 500 images; similar to cifar100 but more images/class
        "cal_sizes": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "test_size": 500,
        "color": "#17becf",
        "marker": "v",
    },
    "tiny_imagenet": {
        "embeddings": "output/embeddings_tiny_imagenet.pt",
        "n_classes": 200,
        # 200 classes × 500 images; needs ≥2/class = 400 minimum
        "cal_sizes": [400, 600, 800, 1000, 1500, 2000],
        "test_size": 500,
        "color": "#bcbd22",
        "marker": "*",
    },
}

NCM_FULL_CP = "whitened_geodesic"   # best NCM from compare_ncms experiments
NCM_CV_PLUS = "geodesic_nn_ratio"   # faster & more principled than softmax for CV+
ALPHA = 0.1
N_TRIALS = 5
N_FOLDS = 5
SPLIT_TRAIN_RATIO = 0.5
SEED = 42


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cal_sizes: list,
    test_size: int,
    n_trials: int,
    ncm_type: str,
    ncm_cv_type: str,
    alpha: float,
    n_folds: int,
    split_train_ratio: float,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run Full CP vs CV+ vs Split CP for varying calibration sizes.

    Returns a results dict with per-cal-size metrics for each method.
    """
    all_classes = np.unique(labels)
    n_classes = len(all_classes)
    class_counts = {int(c): int(np.sum(labels == c)) for c in all_classes}
    min_class_count = min(class_counts.values())

    max_cal = max(cal_sizes)
    max_needed = test_size + max_cal
    num_per_class = math.ceil(max_needed / n_classes)
    if num_per_class > min_class_count:
        if verbose:
            print(f"  [warn] need {num_per_class}/class but min is {min_class_count}. Clamping.")
        num_per_class = min_class_count
        max_needed = num_per_class * n_classes

    if verbose:
        print(f"  Pool: {num_per_class}/class x {n_classes} classes = {num_per_class*n_classes}")
        print(f"  Test: {min(test_size, num_per_class*n_classes - max_cal)}  |  Max cal: {max_cal}")

    method_names = ["full_cp", "cv_plus", "split_cp"]
    results = {
        "cal_sizes": cal_sizes,
        "test_size": test_size,
        "alpha": alpha,
        "ncm": ncm_type,
        "ncm_cv": ncm_cv_type,
        "n_trials": n_trials,
        "n_folds": n_folds,
        "split_train_ratio": split_train_ratio,
        "num_per_class": num_per_class,
        "n_classes": n_classes,
    }
    for m in method_names:
        results[m] = {}
        for cs in cal_sizes:
            results[m][cs] = {
                "coverages": [], "set_sizes": [], "singleton_rates": [],
                "empty_rates": [], "times": [], "n_cal": [], "n_test": [],
            }

    for trial in range(n_trials):
        trial_seed = seed + trial * 1000
        rng = np.random.default_rng(trial_seed)

        # Stratified pool: num_per_class per class
        pool_indices = []
        for c in all_classes:
            c_idx = np.where(labels == c)[0]
            chosen = rng.choice(c_idx, size=num_per_class, replace=False)
            pool_indices.append(chosen)
        pool_indices = np.concatenate(pool_indices)
        rng.shuffle(pool_indices)

        X_pool = embeddings[pool_indices]
        y_pool = labels[pool_indices]

        # Fixed test set (last test_size examples)
        actual_test_size = min(test_size, len(X_pool))
        X_test = X_pool[-actual_test_size:]
        y_test = y_pool[-actual_test_size:]
        X_remaining = X_pool[:-actual_test_size]
        y_remaining = y_pool[:-actual_test_size]

        if verbose:
            print(f"  Trial {trial+1}/{n_trials} | test={len(X_test)}, cal pool={len(X_remaining)}")

        remaining_classes = np.unique(y_remaining)
        n_remaining_cls = len(remaining_classes)

        for cs in cal_sizes:
            if cs > len(X_remaining):
                if verbose:
                    print(f"    cal_size={cs}: not enough data ({len(X_remaining)} available), skipping")
                continue

            # Stratified cal sampling: guarantee ≥1 sample per class to maintain
            # FCP exchangeability (classes absent from cal → coverage violations).
            if cs >= n_remaining_cls:
                first_per_class = np.array([
                    rng.choice(np.where(y_remaining == c)[0], size=1, replace=False)[0]
                    for c in remaining_classes
                ])
                remaining_pool = np.setdiff1d(np.arange(len(X_remaining)), first_per_class)
                n_extra = cs - n_remaining_cls
                if n_extra > 0 and len(remaining_pool) >= n_extra:
                    extra = rng.choice(remaining_pool, size=n_extra, replace=False)
                    cal_idx = np.concatenate([first_per_class, extra])
                else:
                    cal_idx = first_per_class[:cs]
                cal_idx = rng.permutation(cal_idx)
            else:
                cal_idx = rng.choice(len(X_remaining), size=cs, replace=False)

            X_cal = X_remaining[cal_idx]
            y_cal = y_remaining[cal_idx]

            # ---- Full CP ----
            t0 = time.time()
            ncm_full = create_ncm(ncm_type, k=5, lambda_reg=1.0)
            cp_full = FullConformalPredictor(ncm_full, alpha=alpha)
            cp_full.calibrate(X_cal, y_cal, all_classes=all_classes)
            m_full = cp_full.evaluate(X_test, y_test, verbose=False)
            t_full = time.time() - t0

            results["full_cp"][cs]["coverages"].append(m_full["coverage"])
            results["full_cp"][cs]["set_sizes"].append(m_full["avg_set_size"])
            results["full_cp"][cs]["singleton_rates"].append(m_full["singleton_rate"])
            results["full_cp"][cs]["empty_rates"].append(m_full["empty_set_rate"])
            results["full_cp"][cs]["times"].append(t_full)
            results["full_cp"][cs]["n_cal"].append(cs)
            results["full_cp"][cs]["n_test"].append(len(X_test))

            # ---- CV+ ----
            t0 = time.time()
            ncm_cv_factory = lambda: create_ncm(ncm_cv_type, k=5, lambda_reg=1.0)
            cp_cv = CrossValidationPlusPredictor(
                ncm_cv_factory, alpha=alpha, n_folds=n_folds,
            )
            cp_cv.calibrate(X_cal, y_cal, all_classes=all_classes)
            m_cv = cp_cv.evaluate(X_test, y_test, verbose=False)
            t_cv = time.time() - t0

            results["cv_plus"][cs]["coverages"].append(m_cv["coverage"])
            results["cv_plus"][cs]["set_sizes"].append(m_cv["avg_set_size"])
            results["cv_plus"][cs]["singleton_rates"].append(m_cv["singleton_rate"])
            results["cv_plus"][cs]["empty_rates"].append(m_cv["empty_set_rate"])
            results["cv_plus"][cs]["times"].append(t_cv)
            results["cv_plus"][cs]["n_cal"].append(cs)
            results["cv_plus"][cs]["n_test"].append(len(X_test))

            # ---- Split CP ----
            # Split the cal budget into train (ratio) + cal (1-ratio)
            n_train = int(split_train_ratio * cs)
            n_cal_split = cs - n_train
            if n_train < n_classes or n_cal_split < 1:
                results["split_cp"][cs]["coverages"].append(np.nan)
                results["split_cp"][cs]["set_sizes"].append(np.nan)
                results["split_cp"][cs]["singleton_rates"].append(np.nan)
                results["split_cp"][cs]["empty_rates"].append(np.nan)
                results["split_cp"][cs]["times"].append(np.nan)
                results["split_cp"][cs]["n_cal"].append(n_cal_split)
                results["split_cp"][cs]["n_test"].append(len(X_test))
            else:
                # Re-split cal_idx deterministically: first n_train = train, rest = cal
                train_idx = cal_idx[:n_train]
                cal_split_idx = cal_idx[n_train:]
                X_tr, y_tr = X_remaining[train_idx], y_remaining[train_idx]
                X_cs, y_cs = X_remaining[cal_split_idx], y_remaining[cal_split_idx]

                t0 = time.time()
                cp_split = SoftmaxSplitCP(alpha=alpha)
                cp_split.fit(X_tr, y_tr)
                cp_split.calibrate(X_cs, y_cs, all_classes=all_classes)
                m_split = cp_split.evaluate(X_test, y_test, verbose=False)
                t_split = time.time() - t0

                results["split_cp"][cs]["coverages"].append(m_split["coverage"])
                results["split_cp"][cs]["set_sizes"].append(m_split["avg_set_size"])
                results["split_cp"][cs]["singleton_rates"].append(m_split["singleton_rate"])
                results["split_cp"][cs]["empty_rates"].append(m_split["empty_set_rate"])
                results["split_cp"][cs]["times"].append(t_split)
                results["split_cp"][cs]["n_cal"].append(n_cal_split)
                results["split_cp"][cs]["n_test"].append(len(X_test))

            if verbose:
                fcp = results["full_cp"][cs]
                ccp = results["cv_plus"][cs]
                scp = results["split_cp"][cs]
                print(
                    f"    cal={cs:4d} | "
                    f"FCP cov={np.mean(fcp['coverages']):.3f} sz={np.mean(fcp['set_sizes']):.2f} t={np.mean(fcp['times']):.2f}s | "
                    f"CV+ cov={np.mean(ccp['coverages']):.3f} sz={np.mean(ccp['set_sizes']):.2f} t={np.mean(ccp['times']):.2f}s | "
                    f"SCP cov={np.nanmean(scp['coverages']):.3f} sz={np.nanmean(scp['set_sizes']):.2f} t={np.nanmean(scp['times']):.2f}s"
                )

    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def agg(results: dict, method: str, metric: str):
    """Return (cal_sizes, means, stds) for a given method and metric."""
    cal_sizes = results["cal_sizes"]
    means, stds = [], []
    for cs in cal_sizes:
        vals = results[method][cs][metric]
        if len(vals) == 0:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            arr = np.array([v for v in vals if not (isinstance(v, float) and np.isnan(v))])
            means.append(np.nanmean(arr))
            stds.append(np.nanstd(arr))
    return np.array(cal_sizes), np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
# Per-dataset plots
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    "full_cp":  {"label": "Full CP (whitened_geodesic)",  "color": "#1f77b4", "ls": "-",  "lw": 2.5, "marker": "o"},
    "cv_plus":  {"label": "CV+ (geodesic_nn_ratio)",      "color": "#ff7f0e", "ls": "--", "lw": 2.0, "marker": "s"},
    "split_cp": {"label": "Split CP (softmax)",           "color": "#2ca02c", "ls": ":",  "lw": 2.0, "marker": "^"},
}


def plot_dataset_results(results: dict, title: str, out_path: str):
    """Main 2-row plot: top=set size, bottom=coverage. X-axis = cal size."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for method, style in METHOD_STYLES.items():
        xs, mus, sds = agg(results, method, "set_sizes")
        axes[0].plot(xs, mus, color=style["color"], ls=style["ls"], lw=style["lw"],
                     marker=style["marker"], ms=6, label=style["label"])
        axes[0].fill_between(xs, mus - sds, mus + sds, alpha=0.15, color=style["color"])

        xs, mus, sds = agg(results, method, "coverages")
        mus_pct = mus * 100
        sds_pct = sds * 100
        axes[1].plot(xs, mus_pct, color=style["color"], ls=style["ls"], lw=style["lw"],
                     marker=style["marker"], ms=6, label=style["label"])
        axes[1].fill_between(xs, mus_pct - sds_pct, mus_pct + sds_pct, alpha=0.15, color=style["color"])

    target_cov = (1 - results["alpha"]) * 100
    axes[1].axhline(target_cov, color="black", ls="--", lw=1.2, label=f"Target {target_cov:.0f}%")

    axes[0].set_ylabel("Avg Prediction Set Size", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Coverage (%)", fontsize=11)
    axes[1].set_xlabel("Calibration Set Size", fontsize=11)
    axes[1].set_ylim(max(0, target_cov - 20), 102)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_timing(results: dict, title: str, out_path: str):
    """Bar chart of mean run time per method per cal size."""
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(title + " — Runtime", fontsize=13)

    cal_sizes = results["cal_sizes"]
    x = np.arange(len(cal_sizes))
    w = 0.28
    for i, (method, style) in enumerate(METHOD_STYLES.items()):
        _, mus, _ = agg(results, method, "times")
        ax.bar(x + (i - 1) * w, mus, width=w, color=style["color"],
               alpha=0.85, label=style["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(cal_sizes)
    ax.set_xlabel("Calibration Set Size", fontsize=11)
    ax.set_ylabel("Mean Runtime per trial (s)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_text_report(results: dict, out_path: str):
    """Save human-readable table of mean ± std for all metrics."""
    lines = []
    cs_flag = ""
    lines.append(f"Full CP vs CV+ vs Split CP Comparison{cs_flag}")
    lines.append("=" * 72)
    lines.append(f"NCM (Full CP): {results['ncm']}")
    lines.append(f"NCM (CV+): {results['ncm_cv']}")
    lines.append(f"Alpha: {results['alpha']}  Target coverage: {(1-results['alpha'])*100:.0f}%")
    lines.append(f"Trials: {results['n_trials']}  CV+ folds: {results['n_folds']}")
    lines.append("")
    hdr = f"{'Cal':>6}  {'Method':<12}  {'Coverage':>18}  {'Avg Set Size':>18}  {'Time (s)':>14}"
    lines.append(hdr)
    lines.append("-" * 74)

    method_labels = {"full_cp": "Full CP", "cv_plus": "CV+", "split_cp": "Split CP"}
    for cs in results["cal_sizes"]:
        first = True
        for method, lbl in method_labels.items():
            vals_cov = results[method][cs]["coverages"]
            vals_sz  = results[method][cs]["set_sizes"]
            vals_t   = results[method][cs]["times"]
            if not vals_cov:
                continue
            mu_cov = np.nanmean(vals_cov) * 100
            sd_cov = np.nanstd(vals_cov) * 100
            mu_sz  = np.nanmean(vals_sz)
            sd_sz  = np.nanstd(vals_sz)
            mu_t   = np.nanmean(vals_t)
            sd_t   = np.nanstd(vals_t)
            prefix = f"{cs:>6}" if first else " " * 6
            lines.append(
                f"{prefix}  {lbl:<12}  {mu_cov:6.1f}%+/-{sd_cov:5.1f}%  "
                f"{mu_sz:8.2f}+/-{sd_sz:6.2f}  {mu_t:6.2f}+/-{sd_t:4.2f}"
            )
            first = False
        lines.append("-" * 74)

    text = "\n".join(lines) + "\n"
    Path(out_path).write_text(text)
    print(f"  Saved: {out_path}")
    print(text)


# ---------------------------------------------------------------------------
# Cross-dataset summary plot
# ---------------------------------------------------------------------------

def plot_summary(all_results: dict, out_dir: str):
    """
    One figure with 2 rows x N_datasets columns.
    Top row: set size vs cal size.  Bottom row: coverage vs cal size.
    Separate figures for base and class-similarity.
    """
    for variant in ("base", "cs"):
        keys = [k for k in all_results if k.endswith(f"_{variant}")]
        if not keys:
            continue

        n_ds = len(keys)
        fig, axes = plt.subplots(2, n_ds, figsize=(5 * n_ds, 8), sharey="row")
        if n_ds == 1:
            axes = np.array(axes).reshape(2, 1)

        suffix = "" if variant == "base" else " + Class Similarity"
        fig.suptitle(f"DINOv2: Full CP vs CV+ vs Split CP{suffix}  (α={ALPHA})",
                     fontsize=13, fontweight="bold")

        for col, key in enumerate(sorted(keys)):
            res = all_results[key]
            ds_name = key.replace(f"_{variant}", "")
            n_cls = res.get("n_classes", "?")

            for method, style in METHOD_STYLES.items():
                xs, mus, sds = agg(res, method, "set_sizes")
                axes[0, col].plot(xs, mus, color=style["color"], ls=style["ls"],
                                  lw=2, marker=style["marker"], ms=5,
                                  label=style["label"])
                axes[0, col].fill_between(xs, mus - sds, mus + sds,
                                          alpha=0.12, color=style["color"])

                xs, mus, sds = agg(res, method, "coverages")
                axes[1, col].plot(xs, mus * 100, color=style["color"], ls=style["ls"],
                                  lw=2, marker=style["marker"], ms=5)
                axes[1, col].fill_between(xs, (mus - sds) * 100, (mus + sds) * 100,
                                          alpha=0.12, color=style["color"])

            target = (1 - res["alpha"]) * 100
            axes[1, col].axhline(target, color="k", ls="--", lw=1.1)

            axes[0, col].set_title(f"{ds_name}\n({n_cls} classes)", fontsize=11)
            axes[0, col].grid(True, alpha=0.3)
            axes[1, col].grid(True, alpha=0.3)
            axes[1, col].set_ylim(max(0, target - 20), 102)
            axes[1, col].set_xlabel("Cal Size", fontsize=10)

        axes[0, 0].set_ylabel("Avg Set Size", fontsize=10)
        axes[1, 0].set_ylabel("Coverage (%)", fontsize=10)
        handles, labels_leg = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_leg, loc="lower center", ncol=3,
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(rect=[0, 0.04, 1, 1])

        fname = f"summary_{variant}.png"
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSummary saved: {fpath}")


# ---------------------------------------------------------------------------
# Multi-alpha summary plot
# ---------------------------------------------------------------------------

def plot_multi_alpha(ds_name: str, alpha_results: dict, ncm_full: str, out_path: str):
    """
    For a single dataset, show Full CP / CV+ / Split CP across multiple alpha values.

    alpha_results: { alpha_value -> results_dict }
    Layout: 2 rows x 3 cols
      Row 0: Avg Set Size  (Full CP | CV+ | Split CP)
      Row 1: Coverage      (Full CP | CV+ | Split CP)
    One line per alpha, color-coded. Target coverage shown as a dashed horizontal line.
    """
    alphas = sorted(alpha_results.keys())
    n_alpha = len(alphas)

    cmap = plt.cm.get_cmap("plasma", n_alpha)
    colors = [cmap(i) for i in range(n_alpha)]

    method_cols = {"full_cp": 0, "cv_plus": 1, "split_cp": 2}
    method_labels = {
        "full_cp": f"Full CP ({ncm_full})",
        "cv_plus": f"CV+ ({NCM_CV_PLUS})",
        "split_cp": "Split CP (softmax)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    fig.suptitle(
        f"{ds_name.upper()} — Multi-α sweep  (DINOv2, {ncm_full})\n"
        f"α ∈ {[f'{a:.2f}' for a in alphas]}",
        fontsize=13, fontweight="bold",
    )

    for col, method in enumerate(["full_cp", "cv_plus", "split_cp"]):
        axes[0, col].set_title(method_labels[method], fontsize=11)
        axes[0, col].set_ylabel("Avg Set Size" if col == 0 else "", fontsize=10)
        axes[1, col].set_ylabel("Coverage (%)" if col == 0 else "", fontsize=10)
        axes[1, col].set_xlabel("Cal Size", fontsize=10)
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].grid(True, alpha=0.3)

        for i, alpha in enumerate(alphas):
            res = alpha_results[alpha]
            color = colors[i]
            label = f"α={alpha:.2f}"

            xs, mus, sds = agg(res, method, "set_sizes")
            axes[0, col].plot(xs, mus, color=color, lw=2, marker="o", ms=4, label=label)
            axes[0, col].fill_between(xs, mus - sds, mus + sds, alpha=0.10, color=color)

            xs, mus, sds = agg(res, method, "coverages")
            target = (1 - alpha) * 100
            axes[1, col].plot(xs, mus * 100, color=color, lw=2, marker="o", ms=4, label=label)
            axes[1, col].fill_between(
                xs, (mus - sds) * 100, (mus + sds) * 100, alpha=0.10, color=color
            )
            axes[1, col].axhline(target, color=color, ls="--", lw=1.0)

        # Y-lim for coverage: from smallest target - 10 to 102
        min_target = (1 - max(alphas)) * 100
        axes[1, col].set_ylim(max(0, min_target - 10), 102)

    # Shared legend (alphas) from first axis
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=min(n_alpha, 6),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved multi-alpha plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                   choices=list(DATASETS.keys()),
                   help="Which datasets to run (default: all)")
    p.add_argument("--output_dir", default="output/multi_dataset_experiments",
                   help="Root output directory")
    p.add_argument("--skip_base", action="store_true",
                   help="Skip base experiment")
    p.add_argument("--ncm_full", type=str, default=NCM_FULL_CP,
                   choices=["nn_ratio", "geodesic_nn_ratio", "mahal_nn_ratio", "whitened_geodesic",
                            "geodesic_topk_mean", "geodesic_topk_asym"],
                   help="NCM for Full CP")
    p.add_argument("--alpha", type=float, default=ALPHA,
                   help="Single alpha value (ignored when --alphas is set)")
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="Multiple alpha values to sweep (overrides --alpha)")
    p.add_argument("--n_trials", type=int, default=N_TRIALS)
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def load_embeddings(path: str):
    print(f"  Loading: {path}")
    data = torch.load(path, map_location="cpu")
    embeddings = data["embeddings"].numpy()
    labels = data["labels"].numpy()
    print(f"  Shape: {embeddings.shape}, classes: {len(np.unique(labels))}, "
          f"min/max label: {labels.min()}/{labels.max()}")
    return embeddings, labels


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
    return obj


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve alpha list: --alphas takes priority over --alpha
    alphas_to_run = sorted(set(args.alphas)) if args.alphas else [args.alpha]
    multi_alpha = len(alphas_to_run) > 1

    # Update legend label to reflect the chosen Full CP NCM
    METHOD_STYLES["full_cp"]["label"] = f"Full CP ({args.ncm_full})"

    # all_results: keyed by f"{ds_name}_{variant}" for cross-dataset summary (last alpha)
    all_results = {}
    # per-dataset alpha results: { ds_name -> { alpha -> results } }
    ds_alpha_results = {ds: {} for ds in args.datasets}

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        emb_path = cfg["embeddings"]

        if not Path(emb_path).exists():
            print(f"\n[SKIP] Embeddings not found for {ds_name}: {emb_path}")
            continue

        print(f"\n{'#'*70}")
        print(f"# Dataset: {ds_name.upper()}")
        print(f"{'#'*70}")

        embeddings, labels = load_embeddings(emb_path)
        cal_sizes = cfg["cal_sizes"]
        test_size = cfg["test_size"]

        # ---- Base experiment (loop over alphas) ----
        if not args.skip_base:
            out_dir = out_root / ds_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for alpha in alphas_to_run:
                alpha_tag = f"a{alpha:.2f}"
                ncm_tag = args.ncm_full
                file_tag = f"{ncm_tag}_{alpha_tag}" if multi_alpha else ncm_tag

                print(f"\n--- [{ds_name}] Base experiment | alpha={alpha} ---")
                results = run_single_experiment(
                    embeddings=embeddings, labels=labels,
                    cal_sizes=cal_sizes, test_size=test_size,
                    n_trials=args.n_trials, ncm_type=args.ncm_full,
                    ncm_cv_type=NCM_CV_PLUS, alpha=alpha,
                    n_folds=args.n_folds, split_train_ratio=SPLIT_TRAIN_RATIO,
                    seed=args.seed,
                )
                ds_alpha_results[ds_name][alpha] = results
                all_results[f"{ds_name}_base"] = results  # last alpha wins for cross-ds plot

                with open(out_dir / f"cp_methods_{file_tag}.json", "w") as f:
                    json.dump(_jsonify(results), f, indent=2)

                plot_dataset_results(
                    results,
                    title=f"{ds_name.upper()} — Full CP vs CV+ vs Split CP  (DINOv2, α={alpha})",
                    out_path=str(out_dir / f"cp_methods_{file_tag}.png"),
                )
                plot_timing(
                    results,
                    title=f"{ds_name.upper()}  α={alpha}",
                    out_path=str(out_dir / f"cp_methods_timing_{file_tag}.png"),
                )
                save_text_report(results, str(out_dir / f"cp_methods_{file_tag}.txt"))

            # Multi-alpha summary plot for this dataset
            if multi_alpha and ds_alpha_results[ds_name]:
                plot_multi_alpha(
                    ds_name=ds_name,
                    alpha_results=ds_alpha_results[ds_name],
                    ncm_full=args.ncm_full,
                    out_path=str(out_dir / f"multi_alpha_{ncm_tag}.png"),
                )


            # Multi-alpha CS summary plot
            if multi_alpha and ds_alpha_cs_results:
                plot_multi_alpha(
                    ds_name=f"{ds_name} + CS",
                    alpha_results=ds_alpha_cs_results,
                    ncm_full=f"{ncm_tag} + CS",
                    out_path=str(out_dir_cs / f"multi_alpha_{ncm_tag}_cs.png"),
                )

    # ---- Cross-dataset summary ----
    if all_results:
        print(f"\n{'='*70}")
        print("Generating cross-dataset summary plots...")
        plot_summary(all_results, str(out_root))
    else:
        print("\n[warn] No results to summarize — check that embeddings exist.")

    print(f"\nAll done. Results in: {out_root}")


if __name__ == "__main__":
    main()
