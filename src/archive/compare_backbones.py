"""
Compare Full CP (geodesic_topk_mean) vs CV+ vs Split CP across SSL backbones.

Usage:
    python src/compare_backbones.py \
        --embeddings_paths output/embeddings_cifar100.pt output/embeddings_clip_base_cifar100.pt output/embeddings_beitv2_base_cifar100.pt \
        --backbone_names dinov2-base clip-base beitv2-base \
        --n_trials 5 --output_dir output/backbone_comparison
"""
# ARCHIVE-CANDIDATE (review 2026-06-24): HOLD for P2.11 (backbone sensitivity on cluster); archive if P2.11 is dropped.
# If unused by the review date, move to src/archive/ -- watch list: src/archive/README.md.


import sys, os, time, argparse, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from conformal_prediction import (
    FullConformalPredictor,
    CrossValidationPlusPredictor,
    SoftmaxSplitCP,
    create_ncm,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
NCM_FULL_CP = "geodesic_topk_mean"
NCM_CV_PLUS = "mahal_nn_ratio"
ALPHA = 0.1
N_TRIALS = 5
N_FOLDS = 5
SPLIT_TRAIN_RATIO = 0.5
SEED = 42
DEFAULT_CAL_SIZES = [200, 300, 400, 600, 800, 1000]
TEST_SIZE = 300


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------
def run_trial(X_cal, y_cal, X_test, y_test, all_classes, ncm_full, ncm_cv,
              alpha, n_folds, split_train_ratio):
    """Run FCP, CV+, SCP on one (cal, test) split. Returns dict of metrics."""
    n_classes = len(all_classes)
    out = {}

    # ---- Full CP ----
    # approved: legacy whitened-NCM experiment (cal-fit whitening, non-exchangeable)
    t0 = time.time()
    ncm = create_ncm(ncm_full, k=5, allow_nonexchangeable=True)
    cp = FullConformalPredictor(ncm, alpha=alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)
    m = cp.evaluate(X_test, y_test, verbose=False)
    out["full_cp"] = {
        "coverage": m["coverage"], "set_size": m["avg_set_size"],
        "singleton": m["singleton_rate"], "empty": m["empty_set_rate"],
        "time": time.time() - t0,
    }

    # ---- CV+ ----
    t0 = time.time()
    ncm_factory = lambda: create_ncm(ncm_cv, k=5, allow_nonexchangeable=True)
    cp_cv = CrossValidationPlusPredictor(ncm_factory, alpha=alpha, n_folds=n_folds)
    cp_cv.calibrate(X_cal, y_cal, all_classes=all_classes)
    m_cv = cp_cv.evaluate(X_test, y_test, verbose=False)
    out["cv_plus"] = {
        "coverage": m_cv["coverage"], "set_size": m_cv["avg_set_size"],
        "singleton": m_cv["singleton_rate"], "empty": m_cv["empty_set_rate"],
        "time": time.time() - t0,
    }

    # ---- Split CP ----
    n_train = int(split_train_ratio * len(X_cal))
    n_cal_split = len(X_cal) - n_train
    if n_train < n_classes or n_cal_split < 1:
        out["split_cp"] = {
            "coverage": np.nan, "set_size": np.nan,
            "singleton": np.nan, "empty": np.nan, "time": np.nan,
        }
    else:
        t0 = time.time()
        cp_s = SoftmaxSplitCP(alpha=alpha)
        cp_s.fit(X_cal[:n_train], y_cal[:n_train])
        cp_s.calibrate(X_cal[n_train:], y_cal[n_train:], all_classes=all_classes)
        m_s = cp_s.evaluate(X_test, y_test, verbose=False)
        out["split_cp"] = {
            "coverage": m_s["coverage"], "set_size": m_s["avg_set_size"],
            "singleton": m_s["singleton_rate"], "empty": m_s["empty_set_rate"],
            "time": time.time() - t0,
        }

    return out


def run_backbone_experiment(X, y, cal_sizes, test_size, n_trials, ncm_full,
                            ncm_cv, alpha, n_folds, split_train_ratio, seed):
    """Run all cal sizes x trials for one backbone. Returns results dict."""
    all_classes = np.unique(y)
    n_classes = len(all_classes)
    min_class_count = min(np.sum(y == c) for c in all_classes)

    max_cal = max(cal_sizes)
    num_per_class = math.ceil((test_size + max_cal) / n_classes)
    if num_per_class > min_class_count:
        num_per_class = min_class_count

    methods = ["full_cp", "cv_plus", "split_cp"]
    results = {m: {cs: {"coverages": [], "set_sizes": [], "times": []}
                   for cs in cal_sizes} for m in methods}

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial * 1000)

        # Stratified pool
        pool_idx = []
        for c in all_classes:
            c_idx = np.where(y == c)[0]
            chosen = rng.choice(c_idx, size=min(num_per_class, len(c_idx)), replace=False)
            pool_idx.append(chosen)
        pool_idx = np.concatenate(pool_idx)
        rng.shuffle(pool_idx)

        X_pool, y_pool = X[pool_idx], y[pool_idx]
        actual_test = min(test_size, len(X_pool))
        X_test, y_test = X_pool[-actual_test:], y_pool[-actual_test:]
        X_rem, y_rem = X_pool[:-actual_test], y_pool[:-actual_test]

        remaining_classes = np.unique(y_rem)

        for cs in cal_sizes:
            if cs > len(X_rem):
                continue

            # Stratified cal: >=1 per class
            if cs >= len(remaining_classes):
                first = np.array([
                    rng.choice(np.where(y_rem == c)[0], 1, replace=False)[0]
                    for c in remaining_classes
                ])
                rest_pool = np.setdiff1d(np.arange(len(X_rem)), first)
                n_extra = cs - len(remaining_classes)
                if n_extra > 0 and len(rest_pool) >= n_extra:
                    extra = rng.choice(rest_pool, n_extra, replace=False)
                    cal_idx = np.concatenate([first, extra])
                else:
                    cal_idx = first[:cs]
                cal_idx = rng.permutation(cal_idx)
            else:
                cal_idx = rng.choice(len(X_rem), cs, replace=False)

            X_cal, y_cal = X_rem[cal_idx], y_rem[cal_idx]

            trial_out = run_trial(
                X_cal, y_cal, X_test, y_test, all_classes,
                ncm_full, ncm_cv, alpha, n_folds, split_train_ratio,
            )

            for m in methods:
                results[m][cs]["coverages"].append(trial_out[m]["coverage"])
                results[m][cs]["set_sizes"].append(trial_out[m]["set_size"])
                results[m][cs]["times"].append(trial_out[m]["time"])

        print(f"    trial {trial+1}/{n_trials} done")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
BACKBONE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f"]
METHOD_STYLES = {
    "full_cp":  {"ls": "-",  "marker": "o"},
    "cv_plus":  {"ls": "--", "marker": "s"},
    "split_cp": {"ls": ":",  "marker": "^"},
}


def plot_results(all_backbone_results, backbone_names, cal_sizes, alpha,
                 ncm_full, out_dir):
    """Generate per-backbone 3-panel plots + cross-backbone comparison."""
    methods = ["full_cp", "cv_plus", "split_cp"]
    method_labels = {
        "full_cp": f"FCP ({ncm_full})",
        "cv_plus": f"CV+ ({NCM_CV_PLUS})",
        "split_cp": "Split CP (softmax)",
    }

    # --- Per-backbone plots ---
    for bi, bname in enumerate(backbone_names):
        res = all_backbone_results[bname]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{bname} — FCP vs CV+ vs SCP  (alpha={alpha})",
                     fontsize=13, fontweight='bold')

        for m in methods:
            style = METHOD_STYLES[m]
            covs = [np.nanmean(res[m][cs]["coverages"]) for cs in cal_sizes]
            szs  = [np.nanmean(res[m][cs]["set_sizes"]) for cs in cal_sizes]
            ts   = [np.nanmean(res[m][cs]["times"]) for cs in cal_sizes]

            axes[0].plot(cal_sizes, covs, marker=style["marker"],
                         linestyle=style["ls"], label=method_labels[m], linewidth=2)
            axes[1].plot(cal_sizes, szs,  marker=style["marker"],
                         linestyle=style["ls"], label=method_labels[m], linewidth=2)
            axes[2].plot(cal_sizes, ts,   marker=style["marker"],
                         linestyle=style["ls"], label=method_labels[m], linewidth=2)

        axes[0].axhline(1 - alpha, color='k', ls=':', lw=1.5, label=f"Target ({1-alpha:.0%})")
        axes[0].set_title("Coverage"); axes[0].set_xlabel("Cal size")
        axes[0].set_ylabel("Coverage"); axes[0].legend(fontsize=9)
        axes[0].set_ylim(0.75, 1.02); axes[0].grid(alpha=0.3)

        axes[1].set_title("Avg Set Size"); axes[1].set_xlabel("Cal size")
        axes[1].set_ylabel("Set size"); axes[1].legend(fontsize=9)
        axes[1].set_ylim(bottom=0); axes[1].grid(alpha=0.3)

        axes[2].set_title("Wall Time"); axes[2].set_xlabel("Cal size")
        axes[2].set_ylabel("Seconds"); axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        fname = f"{bname.replace('-','_')}_methods.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()
        print(f"  Saved: {fname}")

    # --- Cross-backbone comparison (FCP only) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Backbone Comparison — FCP ({ncm_full})  (alpha={alpha})",
                 fontsize=13, fontweight='bold')

    for bi, bname in enumerate(backbone_names):
        res = all_backbone_results[bname]
        c = BACKBONE_COLORS[bi % len(BACKBONE_COLORS)]
        covs = [np.nanmean(res["full_cp"][cs]["coverages"]) for cs in cal_sizes]
        szs  = [np.nanmean(res["full_cp"][cs]["set_sizes"]) for cs in cal_sizes]
        ts   = [np.nanmean(res["full_cp"][cs]["times"]) for cs in cal_sizes]

        axes[0].plot(cal_sizes, covs, marker='o', color=c, label=bname, lw=2)
        axes[1].plot(cal_sizes, szs,  marker='o', color=c, label=bname, lw=2)
        axes[2].plot(cal_sizes, ts,   marker='o', color=c, label=bname, lw=2)

    axes[0].axhline(1 - alpha, color='k', ls=':', lw=1.5, label=f"Target ({1-alpha:.0%})")
    axes[0].set_title("Coverage"); axes[0].set_xlabel("Cal size")
    axes[0].set_ylabel("Coverage"); axes[0].legend(fontsize=9)
    axes[0].set_ylim(0.75, 1.02); axes[0].grid(alpha=0.3)

    axes[1].set_title("Avg Set Size"); axes[1].set_xlabel("Cal size")
    axes[1].set_ylabel("Set size"); axes[1].legend(fontsize=9)
    axes[1].set_ylim(bottom=0); axes[1].grid(alpha=0.3)

    axes[2].set_title("Wall Time"); axes[2].set_xlabel("Cal size")
    axes[2].set_ylabel("Seconds"); axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fname = "backbone_fcp_comparison.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

    # --- Cross-backbone comparison (all methods, set size only) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f"Backbone x Method — Set Size  (alpha={alpha})",
                 fontsize=13, fontweight='bold')

    for bi, bname in enumerate(backbone_names):
        res = all_backbone_results[bname]
        c = BACKBONE_COLORS[bi % len(BACKBONE_COLORS)]
        for m in methods:
            style = METHOD_STYLES[m]
            szs = [np.nanmean(res[m][cs]["set_sizes"]) for cs in cal_sizes]
            label = f"{bname} {method_labels[m].split('(')[0].strip()}"
            ax.plot(cal_sizes, szs, marker=style["marker"], color=c,
                    linestyle=style["ls"], label=label, linewidth=1.5)

    ax.set_xlabel("Cal size"); ax.set_ylabel("Avg Set Size")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    plt.tight_layout()
    fname = "backbone_all_methods_setsize.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def print_table(all_backbone_results, backbone_names, cal_sizes, alpha):
    """Print a summary table to console."""
    methods = ["full_cp", "cv_plus", "split_cp"]
    method_short = {"full_cp": "FCP", "cv_plus": "CV+", "split_cp": "SCP"}

    print(f"\n{'='*90}")
    print(f"Summary Table  (alpha={alpha})")
    print(f"{'='*90}")
    print(f"{'Backbone':15s} {'Method':5s} ", end="")
    for cs in cal_sizes:
        print(f"  cal={cs:4d}       ", end="")
    print()
    print("-"*90)

    for bname in backbone_names:
        res = all_backbone_results[bname]
        for m in methods:
            tag = method_short[m]
            print(f"{bname:15s} {tag:5s} ", end="")
            for cs in cal_sizes:
                cov = np.nanmean(res[m][cs]["coverages"])
                sz  = np.nanmean(res[m][cs]["set_sizes"])
                valid = "OK" if cov >= (1 - alpha - 0.01) else "!!"
                print(f" {cov:.3f} sz={sz:5.2f}{valid}", end="")
            print()
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Compare backbones: FCP vs CV+ vs SCP")
    p.add_argument("--embeddings_paths", nargs="+", required=True)
    p.add_argument("--backbone_names", nargs="+", required=True)
    p.add_argument("--ncm_full", default=NCM_FULL_CP)
    p.add_argument("--ncm_cv", default=NCM_CV_PLUS)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--cal_sizes", default=",".join(str(x) for x in DEFAULT_CAL_SIZES),
                   help="Comma-separated cal sizes")
    p.add_argument("--test_size", type=int, default=TEST_SIZE)
    p.add_argument("--n_trials", type=int, default=N_TRIALS)
    p.add_argument("--n_folds", type=int, default=N_FOLDS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--output_dir", default="output/backbone_comparison")
    args = p.parse_args()

    if len(args.embeddings_paths) != len(args.backbone_names):
        raise ValueError("--embeddings_paths and --backbone_names must match in length")

    cal_sizes = [int(x) for x in args.cal_sizes.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"NCM (FCP): {args.ncm_full}  |  NCM (CV+): {args.ncm_cv}")
    print(f"alpha={args.alpha}  |  {args.n_trials} trials  |  test={args.test_size}")
    print(f"Cal sizes: {cal_sizes}")
    print(f"Backbones: {args.backbone_names}")
    print()

    # Load all embeddings
    data = {}
    for path, name in zip(args.embeddings_paths, args.backbone_names):
        print(f"Loading {path} ...")
        d = torch.load(path, map_location="cpu", weights_only=False)
        X = d["embeddings"].numpy()
        y = d["labels"].numpy()
        K = len(np.unique(y))
        print(f"  {X.shape[0]} samples, {K} classes, {X.shape[1]}D")
        data[name] = (X, y)

    all_results = {}
    for bname in args.backbone_names:
        X, y = data[bname]
        print(f"\n  [{bname}] Running FCP vs CV+ vs SCP ...")
        all_results[bname] = run_backbone_experiment(
            X, y, cal_sizes, args.test_size, args.n_trials,
            args.ncm_full, args.ncm_cv, args.alpha, args.n_folds,
            SPLIT_TRAIN_RATIO, seed=args.seed,
        )

    print_table(all_results, args.backbone_names, cal_sizes, args.alpha)
    plot_results(all_results, args.backbone_names, cal_sizes,
                 args.alpha, args.ncm_full, args.output_dir)

    # Save JSON
    json_out = {
        "ncm_full": args.ncm_full, "ncm_cv": args.ncm_cv,
        "alpha": args.alpha, "n_trials": args.n_trials,
        "cal_sizes": cal_sizes, "backbones": args.backbone_names,
    }
    for bname in args.backbone_names:
        json_out[bname] = {}
        for m in ["full_cp", "cv_plus", "split_cp"]:
            json_out[bname][m] = {}
            for cs in cal_sizes:
                r = all_results[bname][m][cs]
                json_out[bname][m][str(cs)] = {
                    "coverage": float(np.nanmean(r["coverages"])),
                    "set_size": float(np.nanmean(r["set_sizes"])),
                    "time": float(np.nanmean(r["times"])),
                }
    with open(os.path.join(args.output_dir, "backbone_comparison.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"\nAll done. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
