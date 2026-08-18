"""
NCM comparison after dimensionality reduction (PCA / AE / full baseline).

Compares all NCMs on the same stratified splits, with each reduction applied.
Uses separate unlabeled pool for PCA/AE fitting (no data leakage).

Usage:
    python src/ncm_comparison_reduced.py
    python src/ncm_comparison_reduced.py --cal_sizes 400,600,800 --n_trials 5
"""
# ARCHIVE-CANDIDATE (review 2026-06-24): results recorded 2026-05-14; HOLD for P0.3 (geodesic_topk_asym cluster confirmation).
# If unused by the review date, move to src/archive/ -- watch list: src/archive/README.md.


import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from conformal_prediction import FullConformalPredictor, create_ncm
from autoencoder_utils import EmbeddingAutoencoder

# Defaults
ALPHA = 0.1
SEED = 42
N_TRIALS = 3
CAL_SIZES = [300, 400, 600, 800]
TEST_SIZE = 500
PCA_DIM = 128
AE_DIM = 32

EMB_PATH = "output/embeddings_cifar100.pt"
TEST_PATH = "output/embeddings_cifar100_test.pt"
UNLABELED_PATH = "output/embeddings_cifar100_unlabeled.pt"

NCM_NAMES = [
    "mahal_nn_ratio",
    "whitened_geodesic",
    "geodesic_topk_mean",
    "geodesic_topk_asym",
    "unwhitened_topk_mean",
    "unwhitened_topk_asym",
]

STYLES = {
    "mahal_nn_ratio":      {"color": "#2ca02c", "ls": "--", "marker": "s"},
    "whitened_geodesic":   {"color": "#9467bd", "ls": "--", "marker": "D"},
    "geodesic_topk_mean":  {"color": "#d62728", "ls": "-",  "marker": "o"},
    "geodesic_topk_asym":  {"color": "#8c564b", "ls": "-.", "marker": "^"},
    "unwhitened_topk_mean":{"color": "#e377c2", "ls": ":",  "marker": "v"},
    "unwhitened_topk_asym":{"color": "#ff7f0e", "ls": ":",  "marker": "<"},
}


def stratified_cal(X, y, all_classes, cal_size, rng):
    """Balanced stratified sample: equal per class, then round-robin remainder."""
    n_classes = len(all_classes)
    per_class = cal_size // n_classes
    remainder = cal_size - per_class * n_classes

    idx = []
    leftover = []
    for c in all_classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, min(per_class, len(c_idx)), replace=False)
        idx.extend(chosen.tolist())
        rest = np.setdiff1d(c_idx, chosen)
        if len(rest) > 0:
            leftover.append(rng.choice(rest, 1, replace=False)[0])

    # Fill remainder round-robin from leftover
    if remainder > 0 and len(leftover) > 0:
        extra = np.array(leftover[:remainder])
        idx.extend(extra.tolist())

    idx = np.array(idx)
    return rng.permutation(idx)


def run_ncm_comparison(X_cal_full, y_cal, X_test, y_test, all_classes,
                       ncm_names, alpha):
    """Run all NCMs on one (cal, test) split. Returns {ncm: {cov, sz, time}}."""
    out = {}
    for ncm_name in ncm_names:
        # approved: comparison sweep may include whitened (cal-fit) NCMs
        ncm = create_ncm(ncm_name, k=5, allow_nonexchangeable=True)
        cp = FullConformalPredictor(ncm, alpha=alpha)
        t0 = time.time()
        cp.calibrate(X_cal_full, y_cal, all_classes=all_classes)
        m = cp.evaluate(X_test, y_test, verbose=False)
        elapsed = time.time() - t0
        out[ncm_name] = {
            "coverage": m["coverage"],
            "avg_set_size": m["avg_set_size"],
            "time": elapsed,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_sizes", type=str, default=",".join(str(c) for c in CAL_SIZES))
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--pca_dim", type=int, default=PCA_DIM)
    parser.add_argument("--ae_dim", type=int, default=AE_DIM)
    parser.add_argument("--test_size", type=int, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default="output/ncm_comparison")
    args = parser.parse_args()

    cal_sizes = [int(s) for s in args.cal_sizes.split(",")]
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

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

    # Stratified test subsample (fixed)
    rng_test = np.random.default_rng(args.seed)
    test_per_class = args.test_size // n_classes
    test_idx = []
    for c in all_classes:
        c_idx = np.where(y_test_full == c)[0]
        test_idx.append(rng_test.choice(c_idx, min(test_per_class, len(c_idx)), replace=False))
    test_idx = np.concatenate(test_idx)
    X_test_raw = X_test_full[test_idx]
    y_test = y_test_full[test_idx]

    print(f"Labeled: {X_labeled.shape}, Unlabeled: {X_unlabeled.shape}, "
          f"Test: {X_test_raw.shape}, Classes: {n_classes}")

    # --- Fit reductions on unlabeled pool ---
    pca = PCA(n_components=args.pca_dim)
    pca.fit(X_unlabeled)
    print(f"PCA-{args.pca_dim}: explained variance {pca.explained_variance_ratio_.sum():.3f}")

    ae = EmbeddingAutoencoder(bottleneck_dim=args.ae_dim, seed=args.seed)
    ae.fit(X_unlabeled)

    # Pre-project test sets
    reductions = {
        f"full-{X_labeled.shape[1]}": {"test": X_test_raw, "transform": lambda X: X},
        f"PCA-{args.pca_dim}": {"test": pca.transform(X_test_raw), "transform": pca.transform},
        f"AE-{args.ae_dim}": {"test": ae.transform(X_test_raw), "transform": ae.transform},
    }

    # --- Run comparison ---
    # results[reduction][cal_size][ncm] = [trial_dicts]
    results = {}
    for red_name in reductions:
        results[red_name] = {cs: {ncm: [] for ncm in NCM_NAMES} for cs in cal_sizes}

    for cs in cal_sizes:
        for trial in range(args.n_trials):
            rng = np.random.default_rng(args.seed + trial * 1000)
            cal_idx = stratified_cal(X_labeled, y_labeled, all_classes, cs, rng)
            X_cal_raw = X_labeled[cal_idx]
            y_cal = y_labeled[cal_idx]

            for red_name, red in reductions.items():
                X_cal = red["transform"](X_cal_raw)
                X_test = red["test"]

                trial_results = run_ncm_comparison(
                    X_cal, y_cal, X_test, y_test, all_classes, NCM_NAMES, ALPHA)

                for ncm_name, vals in trial_results.items():
                    results[red_name][cs][ncm_name].append(vals)

        # Print progress
        print(f"\n{'='*90}")
        print(f"cal={cs} ({args.n_trials} trials)")
        print(f"{'='*90}")
        print(f"{'Reduction':>12}  {'NCM':>25}  {'Coverage':>12}  {'Set Size':>14}  {'Time':>8}")
        print(f"{'-'*90}")
        for red_name in reductions:
            for ncm_name in NCM_NAMES:
                trials = results[red_name][cs][ncm_name]
                covs = [t["coverage"] for t in trials]
                szs = [t["avg_set_size"] for t in trials]
                tms = [t["time"] for t in trials]
                flag = "OK" if np.mean(covs) >= 0.89 else "!!"
                print(f"{red_name:>12}  {ncm_name:>25}  "
                      f"{np.mean(covs):.3f}+/-{np.std(covs):.3f} {flag}  "
                      f"{np.mean(szs):5.2f}+/-{np.std(szs):.2f}  "
                      f"{np.mean(tms):5.1f}s")
            print()

    # --- Save JSON ---
    def jsonify(o):
        if isinstance(o, dict):
            return {str(k): jsonify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonify(i) for i in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    json_path = os.path.join(out_dir, "ncm_comparison_reduced.json")
    with open(json_path, "w") as f:
        json.dump(jsonify({
            "cal_sizes": cal_sizes,
            "n_trials": args.n_trials,
            "alpha": ALPHA,
            "pca_dim": args.pca_dim,
            "ae_dim": args.ae_dim,
            "ncm_names": NCM_NAMES,
            "reductions": list(reductions.keys()),
            "results": results,
        }), f, indent=2)
    print(f"\nSaved: {json_path}")

    # --- Plot: one figure per reduction, 2 panels (set size + coverage) ---
    for red_name in reductions:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"NCM Comparison after {red_name}  (CIFAR-100, alpha={ALPHA}, {args.n_trials} trials)",
                     fontsize=12, fontweight="bold")

        for ncm_name in NCM_NAMES:
            s = STYLES[ncm_name]
            xs, mu_sz, sd_sz, mu_cov, sd_cov = [], [], [], [], []
            for cs in cal_sizes:
                trials = results[red_name][cs][ncm_name]
                covs = [t["coverage"] for t in trials]
                szs = [t["avg_set_size"] for t in trials]
                xs.append(cs)
                mu_cov.append(np.mean(covs)); sd_cov.append(np.std(covs))
                mu_sz.append(np.mean(szs)); sd_sz.append(np.std(szs))
            xs = np.array(xs)

            axes[0].plot(xs, mu_sz, color=s["color"], ls=s["ls"], marker=s["marker"],
                         lw=2, ms=6, label=ncm_name)
            axes[0].fill_between(xs, np.array(mu_sz) - np.array(sd_sz),
                                 np.array(mu_sz) + np.array(sd_sz),
                                 alpha=0.1, color=s["color"])

            axes[1].plot(xs, np.array(mu_cov) * 100, color=s["color"], ls=s["ls"],
                         marker=s["marker"], lw=2, ms=6, label=ncm_name)
            axes[1].fill_between(xs, (np.array(mu_cov) - np.array(sd_cov)) * 100,
                                 (np.array(mu_cov) + np.array(sd_cov)) * 100,
                                 alpha=0.1, color=s["color"])

        axes[0].set_ylabel("Avg Set Size"); axes[0].set_xlabel("Cal Size")
        axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

        axes[1].axhline(90, color="black", ls="--", lw=1.2, label="Target 90%")
        axes[1].set_ylabel("Coverage (%)"); axes[1].set_xlabel("Cal Size")
        axes[1].set_ylim(70, 102)
        axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = red_name.replace("-", "").replace("(", "").replace(")", "")
        plot_path = os.path.join(out_dir, f"ncm_comparison_{safe_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path}")

    # --- Summary table: best NCM per reduction per cal_size ---
    print(f"\n{'='*90}")
    print("BEST NCM per reduction (by set size, among valid coverage >= 89%)")
    print(f"{'='*90}")
    print(f"{'Cal':>5}  {'Reduction':>12}  {'Best NCM':>25}  {'Set Size':>10}  {'Coverage':>10}")
    print(f"{'-'*70}")
    for cs in cal_sizes:
        for red_name in reductions:
            best_ncm, best_sz, best_cov = None, 1e9, 0
            for ncm_name in NCM_NAMES:
                trials = results[red_name][cs][ncm_name]
                cov = np.mean([t["coverage"] for t in trials])
                sz = np.mean([t["avg_set_size"] for t in trials])
                if cov >= 0.89 and sz < best_sz:
                    best_ncm, best_sz, best_cov = ncm_name, sz, cov
            if best_ncm is None:
                best_ncm = "none valid"
            print(f"{cs:>5}  {red_name:>12}  {best_ncm:>25}  {best_sz:>10.2f}  {best_cov:>10.3f}")
        print()

    # --- Text summary ---
    lines = []
    lines.append(f"NCM Comparison after Dimensionality Reduction  (CIFAR-100, alpha={ALPHA}, {args.n_trials} trials)")
    lines.append(f"PCA-{args.pca_dim} on {X_unlabeled.shape[0]} unlabeled, AE-{args.ae_dim} on same pool")
    lines.append("=" * 100)
    for cs in cal_sizes:
        lines.append(f"\ncal={cs}")
        lines.append(f"{'Reduction':>12}  {'NCM':>25}  {'Coverage':>18}  {'Set Size':>18}")
        lines.append("-" * 80)
        for red_name in reductions:
            for ncm_name in NCM_NAMES:
                trials = results[red_name][cs][ncm_name]
                covs = [t["coverage"] for t in trials]
                szs = [t["avg_set_size"] for t in trials]
                flag = "OK" if np.mean(covs) >= 0.89 else "!!"
                lines.append(
                    f"{red_name:>12}  {ncm_name:>25}  "
                    f"{np.mean(covs)*100:5.1f}%+/-{np.std(covs)*100:4.1f}% {flag}  "
                    f"{np.mean(szs):7.2f}+/-{np.std(szs):5.2f}")
            lines.append("")

    txt = "\n".join(lines)
    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write(txt)
    print(f"Saved: {txt_path}")
    print("\n" + txt)


if __name__ == "__main__":
    main()
