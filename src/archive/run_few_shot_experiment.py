"""
Few-shot conformal prediction experiment.

For each k (labeled shots per class), uses all k*n_classes examples as
calibration and evaluates on a fixed held-out test set.
Compares FCP vs CV+ vs Split CP as k decreases toward the extreme low-data limit.

Usage:
    python -u -m src.run_few_shot_experiment
    python -u -m src.run_few_shot_experiment --datasets cifar100 --n_trials 3
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_prediction import (
    FullConformalPredictor,
    CrossValidationPlusPredictor,
    SoftmaxSplitCP,
    create_ncm,
)

# ---------------------------------------------------------------------------
# Dataset configs: k_shots and how many per-class test samples to reserve
# ---------------------------------------------------------------------------
DATASETS = {
    "cifar10": {
        "embeddings": "output/embeddings_cifar10.pt",
        "n_classes": 10,
        "k_shots": [2, 4, 6, 8, 10, 12, 15, 18, 20],
        "n_test_per_class": 50,   # 50*10=500 test points
    },
    "cifar100": {
        "embeddings": "output/embeddings_cifar100.pt",
        "n_classes": 100,
        "k_shots": [2, 4, 6, 8, 10, 12, 15, 18, 20],
        "n_test_per_class": 5,    # 5*100=500 test points
    },
    "eurosat": {
        "embeddings": "output/embeddings_eurosat.pt",
        "n_classes": 10,
        "k_shots": [2, 4, 6, 8, 10, 12, 15, 18, 20],
        "n_test_per_class": 50,   # 50*10=500 test points
    },
    "flowers102": {
        "embeddings": "output/embeddings_flowers102.pt",
        "n_classes": 102,
        "k_shots": [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30],
        "n_test_per_class": 5,    # 5*102=510 test points; 60/class downloaded → 30 max shot
        "ncm_cv": "geodesic_nn_ratio",  # softmax too slow at 102 classes
    },
    "cub200": {
        "embeddings": "output/embeddings_cub200.pt",
        "n_classes": 200,
        "k_shots": [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30],
        "n_test_per_class": 5,    # 5*200=1000 test points; 50/class downloaded → 30 max shot
        "ncm_cv": "geodesic_nn_ratio",  # softmax too slow at 200 classes
    },
    "miniimagenet": {
        "embeddings": "output/embeddings_miniimagenet.pt",
        "n_classes": 100,
        "k_shots": [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30],
        "n_test_per_class": 5,    # 5*100=500 test points; 500/class downloaded
        "ncm_cv": "geodesic_nn_ratio",  # softmax too slow at 100 classes
    },
    "tiny_imagenet": {
        "embeddings": "output/embeddings_tiny_imagenet.pt",
        "n_classes": 200,
        "k_shots": [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30],
        "n_test_per_class": 5,    # 5*200=1000 test points; 500/class downloaded
        "ncm_cv": "geodesic_nn_ratio",  # softmax too slow at 200 classes
    },
}

NCM_FULL_CP = "whitened_geodesic"
NCM_CV_PLUS = "softmax"
ALPHA = 0.1
N_TRIALS = 5
N_FOLDS = 5
SPLIT_TRAIN_RATIO = 0.5
SEED = 42

METHOD_STYLES = {
    "full_cp":  {"label": "Full CP (whitened_geodesic)", "color": "#1f77b4", "marker": "o", "ls": "-"},
    "cv_plus":  {"label": "CV+ (softmax)",      "color": "#2ca02c", "marker": "s", "ls": "--"},
    "split_cp": {"label": "Split CP (softmax)", "color": "#ff7f0e", "marker": "^", "ls": ":"},
}


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_few_shot(embeddings, labels, k_shots, n_test_per_class,
                 n_trials, ncm_type, ncm_cv_type, alpha,
                 n_folds, split_train_ratio, seed, verbose=True):
    """
    For each k in k_shots:
      - Reserve n_test_per_class per class as test (fixed within each trial).
      - Subsample k per class from the remaining pool for calibration.
      - Run FCP, CV+, SCP and record coverage / set size.

    Returns results dict keyed by k.
    """
    all_classes = np.unique(labels)
    n_classes = len(all_classes)
    min_class_count = min(np.sum(labels == c) for c in all_classes)

    # Clamp k_shots: need at least k + n_test_per_class per class
    max_cal_k = min_class_count - n_test_per_class
    k_shots = sorted(k for k in k_shots if k <= max_cal_k and k >= 1)
    if not k_shots:
        raise ValueError(f"No valid k_shots: need at least 1 + {n_test_per_class} per class "
                         f"but min_class_count={min_class_count}")
    max_k = max(k_shots)

    results = {k: {"coverages": {m: [] for m in METHOD_STYLES},
                   "set_sizes": {m: [] for m in METHOD_STYLES},
                   "times":     {m: [] for m in METHOD_STYLES}}
               for k in k_shots}

    for trial in range(n_trials):
        trial_seed = seed + trial * 1000
        rng = np.random.default_rng(trial_seed)

        # Build per-class index pools (shuffled)
        test_idx, cal_pool_idx = [], []
        for c in all_classes:
            c_idx = np.where(labels == c)[0]
            shuffled = rng.permutation(c_idx)
            test_idx.append(shuffled[:n_test_per_class])
            cal_pool_idx.append(shuffled[n_test_per_class:n_test_per_class + max_k])

        test_idx = np.concatenate(test_idx)
        cal_pool_per_class = cal_pool_idx   # list of arrays, one per class

        X_test = embeddings[test_idx]
        y_test = labels[test_idx]

        if verbose:
            print(f"  Trial {trial+1}/{n_trials}: test={len(X_test)}, "
                  f"cal pool up to {max_k}/class x {n_classes} = {max_k*n_classes}")

        for k in k_shots:
            # Subsample k per class from cal pool
            cal_idx = np.concatenate([
                rng.choice(cal_pool_per_class[i], size=k, replace=False)
                for i in range(n_classes)
            ])
            rng.shuffle(cal_idx)
            X_cal = embeddings[cal_idx]
            y_cal = labels[cal_idx]

            # ---- Full CP ----
            t0 = time.time()
            ncm = create_ncm(ncm_type, k=5, lambda_reg=1.0)
            cp = FullConformalPredictor(ncm, alpha=alpha)
            cp.calibrate(X_cal, y_cal, all_classes=all_classes)
            m = cp.evaluate(X_test, y_test, verbose=False)
            results[k]["coverages"]["full_cp"].append(m["coverage"])
            results[k]["set_sizes"]["full_cp"].append(m["avg_set_size"])
            results[k]["times"]["full_cp"].append(time.time() - t0)

            # ---- CV+ ----
            t0 = time.time()
            ncm_factory = lambda: create_ncm(ncm_cv_type, k=5, lambda_reg=1.0)
            cp_cv = CrossValidationPlusPredictor(ncm_factory, alpha=alpha, n_folds=n_folds)
            cp_cv.calibrate(X_cal, y_cal, all_classes=all_classes)
            m = cp_cv.evaluate(X_test, y_test, verbose=False)
            results[k]["coverages"]["cv_plus"].append(m["coverage"])
            results[k]["set_sizes"]["cv_plus"].append(m["avg_set_size"])
            results[k]["times"]["cv_plus"].append(time.time() - t0)

            # ---- Split CP ----
            n_train = int(split_train_ratio * k * n_classes)
            n_cal_split = k * n_classes - n_train
            if n_train < n_classes or n_cal_split < 1:
                for metric in ("coverages", "set_sizes", "times"):
                    results[k][metric]["split_cp"].append(np.nan)
            else:
                t0 = time.time()
                cp_split = SoftmaxSplitCP(alpha=alpha)
                cp_split.fit(X_cal[:n_train], y_cal[:n_train])
                cp_split.calibrate(X_cal[n_train:], y_cal[n_train:],
                                   all_classes=all_classes)
                m = cp_split.evaluate(X_test, y_test, verbose=False)
                results[k]["coverages"]["split_cp"].append(m["coverage"])
                results[k]["set_sizes"]["split_cp"].append(m["avg_set_size"])
                results[k]["times"]["split_cp"].append(time.time() - t0)

            if verbose:
                fcp_cov = np.mean(results[k]["coverages"]["full_cp"])
                fcp_sz  = np.mean(results[k]["set_sizes"]["full_cp"])
                cv_cov  = np.mean(results[k]["coverages"]["cv_plus"])
                cv_sz   = np.mean(results[k]["set_sizes"]["cv_plus"])
                scp_cov = np.nanmean(results[k]["coverages"]["split_cp"])
                scp_sz  = np.nanmean(results[k]["set_sizes"]["split_cp"])
                print(f"    k={k:3d}/cls ({k*n_classes:5d} total) | "
                      f"FCP={fcp_cov:.3f} sz={fcp_sz:.2f} | "
                      f"CV+={cv_cov:.3f} sz={cv_sz:.2f} | "
                      f"SCP={scp_cov:.3f} sz={scp_sz:.2f}")

    return results, k_shots


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_few_shot(ds_name, results, k_shots, n_classes, alpha, ncm_type, out_path):
    target = (1 - alpha) * 100
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{ds_name.upper()} — Few-Shot Conformal Prediction  (DINOv2, alpha={alpha})\n"
        f"{n_classes} classes, target coverage={target:.0f}%",
        fontsize=13, fontweight="bold",
    )

    for method, style in METHOD_STYLES.items():
        covs  = [np.nanmean(results[k]["coverages"][method]) * 100 for k in k_shots]
        c_std = [np.nanstd(results[k]["coverages"][method])  * 100 for k in k_shots]
        szs   = [np.nanmean(results[k]["set_sizes"][method])       for k in k_shots]
        s_std = [np.nanstd(results[k]["set_sizes"][method])        for k in k_shots]
        ts    = [np.nanmean(results[k]["times"][method])           for k in k_shots]

        axes[0].errorbar(k_shots, covs, yerr=c_std, label=style["label"],
                         color=style["color"], marker=style["marker"],
                         ls=style["ls"], lw=2, ms=7, capsize=4)
        axes[1].errorbar(k_shots, szs, yerr=s_std, label=style["label"],
                         color=style["color"], marker=style["marker"],
                         ls=style["ls"], lw=2, ms=7, capsize=4)
        axes[2].plot(k_shots, ts, label=style["label"],
                     color=style["color"], marker=style["marker"],
                     ls=style["ls"], lw=2, ms=7)

    axes[0].axhline(target, color="red", ls="--", lw=1.5, label=f"Target {target:.0f}%")
    for ax, ylabel, title in zip(axes,
                                  ["Coverage (%)", "Avg Set Size", "Time (s)"],
                                  ["Coverage vs k", "Avg Set Size vs k", "Runtime vs k"]):
        ax.set_xlabel("k (shots per class)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(max(0, target - 20), 102)

    # Secondary x-axis: total cal samples
    for ax in axes[:2]:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(k_shots)
        ax2.set_xticklabels([str(k * n_classes) for k in k_shots], fontsize=8, rotation=45)
        ax2.set_xlabel("Total cal samples", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_summary(all_results, out_dir, alpha, ncm_type, default_ncm_cv):
    datasets = list(all_results.keys())
    n_ds = len(datasets)
    fig, axes = plt.subplots(2, n_ds, figsize=(6 * n_ds, 10))
    if n_ds == 1:
        axes = axes.reshape(-1, 1)

    target = (1 - alpha) * 100

    for col, ds_name in enumerate(datasets):
        results, k_shots, n_classes, ncm_cv_type = all_results[ds_name]
        METHOD_STYLES["cv_plus"]["label"] = f"CV+ ({ncm_cv_type})"
        for method, style in METHOD_STYLES.items():
            covs  = [np.nanmean(results[k]["coverages"][method]) * 100 for k in k_shots]
            c_std = [np.nanstd(results[k]["coverages"][method])  * 100 for k in k_shots]
            szs   = [np.nanmean(results[k]["set_sizes"][method])       for k in k_shots]
            s_std = [np.nanstd(results[k]["set_sizes"][method])        for k in k_shots]

            axes[0, col].errorbar(k_shots, szs, yerr=s_std, label=style["label"],
                                   color=style["color"], marker=style["marker"],
                                   ls=style["ls"], lw=2, ms=6, capsize=3)
            axes[1, col].errorbar(k_shots, covs, yerr=c_std,
                                   color=style["color"], marker=style["marker"],
                                   ls=style["ls"], lw=2, ms=6, capsize=3)

        axes[1, col].axhline(target, color="red", ls="--", lw=1.2)
        axes[0, col].set_title(f"{ds_name.upper()} ({n_classes} classes)", fontsize=11)
        for row in range(2):
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlabel("k (shots per class)", fontsize=10)
        axes[1, col].set_ylim(max(0, target - 20), 102)

    axes[0, 0].set_ylabel("Avg Set Size", fontsize=10)
    axes[1, 0].set_ylabel("Coverage (%)", fontsize=10)
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Few-Shot CP — Full CP vs CV+ vs Split CP  (DINOv2, alpha={alpha})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(out_dir, f"summary_few_shot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSummary saved: {out_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(ds_name, results, k_shots, n_classes, alpha, ncm_cv_type="softmax"):
    target = (1 - alpha) * 100
    print(f"\n{'='*70}")
    print(f"FEW-SHOT RESULTS: {ds_name.upper()}  ({n_classes} classes, alpha={alpha}, target={target:.0f}%)")
    print(f"CV+ NCM: {ncm_cv_type}")
    print(f"{'='*70}")
    print(f"{'k':>5} {'Total':>6} | {'FCP cov':>8} {'FCP sz':>7} {'FCP t':>6} | "
          f"{'CV+ cov':>8} {'CV+ sz':>7} {'CV+ t':>6} | "
          f"{'SCP cov':>8} {'SCP sz':>7}")
    print("-" * 85)
    for k in k_shots:
        fcp_cov = np.nanmean(results[k]["coverages"]["full_cp"]) * 100
        fcp_sz  = np.nanmean(results[k]["set_sizes"]["full_cp"])
        fcp_t   = np.nanmean(results[k]["times"]["full_cp"])
        cv_cov  = np.nanmean(results[k]["coverages"]["cv_plus"]) * 100
        cv_sz   = np.nanmean(results[k]["set_sizes"]["cv_plus"])
        cv_t    = np.nanmean(results[k]["times"]["cv_plus"])
        scp_cov = np.nanmean(results[k]["coverages"]["split_cp"]) * 100
        scp_sz  = np.nanmean(results[k]["set_sizes"]["split_cp"])
        print(f"{k:>5} {k*n_classes:>6} | {fcp_cov:>7.1f}% {fcp_sz:>7.2f} {fcp_t:>5.1f}s | "
              f"{cv_cov:>7.1f}% {cv_sz:>7.2f} {cv_t:>5.1f}s | "
              f"{scp_cov:>7.1f}% {scp_sz:>7.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                   choices=list(DATASETS.keys()))
    p.add_argument("--output_dir", default="output/few_shot_experiments")
    p.add_argument("--ncm_full", default=NCM_FULL_CP,
                   choices=["nn_ratio", "geodesic_nn_ratio", "mahal_nn_ratio", "whitened_geodesic",
                            "relative_centroid", "ridge", "geodesic_topk_mean", "geodesic_topk_asym"])
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--n_trials", type=int, default=N_TRIALS)
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(i) for i in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def load_embeddings(path):
    print(f"  Loading: {path}")
    data = torch.load(path, map_location="cpu")
    emb = data["embeddings"].numpy()
    lab = data["labels"].numpy()
    print(f"  Shape: {emb.shape}, classes: {len(np.unique(lab))}")
    return emb, lab


def main():
    args = parse_args()
    METHOD_STYLES["full_cp"]["label"] = f"Full CP ({args.ncm_full})"

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        emb_path = cfg["embeddings"]

        if not Path(emb_path).exists():
            print(f"\n[SKIP] {ds_name}: {emb_path} not found")
            continue

        print(f"\n{'#'*60}")
        print(f"# Dataset: {ds_name.upper()}")
        print(f"{'#'*60}")

        embeddings, labels = load_embeddings(emb_path)
        n_classes = len(np.unique(labels))
        k_shots_cfg = cfg["k_shots"]
        n_test_per_class = cfg["n_test_per_class"]

        ncm_cv_type = cfg.get("ncm_cv", NCM_CV_PLUS)
        METHOD_STYLES["cv_plus"]["label"] = f"CV+ ({ncm_cv_type})"

        results, k_shots_used = run_few_shot(
            embeddings=embeddings, labels=labels,
            k_shots=k_shots_cfg, n_test_per_class=n_test_per_class,
            n_trials=args.n_trials, ncm_type=args.ncm_full,
            ncm_cv_type=ncm_cv_type, alpha=args.alpha,
            n_folds=args.n_folds, split_train_ratio=SPLIT_TRAIN_RATIO,
            seed=args.seed,
        )

        all_results[ds_name] = (results, k_shots_used, n_classes, ncm_cv_type)

        out_dir = out_root / ds_name
        out_dir.mkdir(exist_ok=True)

        with open(out_dir / f"few_shot_{args.ncm_full}.json", "w") as f:
            json.dump(_jsonify(results), f, indent=2)

        plot_few_shot(
            ds_name=ds_name, results=results, k_shots=k_shots_used,
            n_classes=n_classes, alpha=args.alpha, ncm_type=args.ncm_full,
            out_path=str(out_dir / f"few_shot_{args.ncm_full}.png"),
        )
        print_report(ds_name, results, k_shots_used, n_classes, args.alpha, ncm_cv_type)

    if len(all_results) > 1:
        plot_summary(all_results, str(out_root), args.alpha, args.ncm_full, NCM_CV_PLUS)

    print(f"\nAll done. Results in: {out_root}")


if __name__ == "__main__":
    main()
