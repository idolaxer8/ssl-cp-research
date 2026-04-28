"""
Compare Full CP NCMs: nn_ratio, geodesic_nn_ratio, mahal_nn_ratio, whitened_geodesic.
Sweeps calibration set sizes on a given embeddings file.

Usage:
    python src/compare_ncms.py --embeddings_path output/embeddings_cifar100.pt
"""

import sys, os, time, argparse, json
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

from conformal_prediction import FullConformalPredictor, create_ncm


# ---------------------------------------------------------------------------
def stratified_split(embeddings, labels, n_cal, n_test, rng):
    """Sample n_cal calibration + n_test test points, stratified by class."""
    classes, counts = np.unique(labels, return_counts=True)
    K = len(classes)

    cal_per_class  = max(1, n_cal  // K)
    test_per_class = max(1, n_test // K)

    cal_idx, test_idx = [], []
    for c in classes:
        pool = np.where(labels == c)[0]
        rng.shuffle(pool)
        need = cal_per_class + test_per_class
        if len(pool) < need:
            # Not enough — take what we can, skip if fewer than 2
            if len(pool) < 2:
                continue
            split = max(1, len(pool) // 2)
            cal_idx.extend(pool[:split].tolist())
            test_idx.extend(pool[split:].tolist())
        else:
            cal_idx.extend(pool[:cal_per_class].tolist())
            test_idx.extend(pool[cal_per_class:cal_per_class + test_per_class].tolist())

    return np.array(cal_idx), np.array(test_idx)


def run_one(X, y, ncm_name, cal_size, test_size, alpha, n_trials, seed, tau=0.1):
    """Run n_trials of FCP with given NCM and cal_size. Returns (coverage, set_size, time_s)."""
    rng = np.random.default_rng(seed)
    coverages, set_sizes, times = [], [], []

    for trial_i in range(n_trials):
        cal_idx, test_idx = stratified_split(X, y, cal_size, test_size, rng)
        if len(cal_idx) == 0 or len(test_idx) == 0:
            continue

        X_cal, y_cal   = X[cal_idx], y[cal_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        ncm = create_ncm(ncm_name, tau=tau)
        cp  = FullConformalPredictor(ncm, alpha=alpha)

        t0 = time.time()
        cp.calibrate(X_cal, y_cal)
        out = cp.predict(X_test, verbose=False)
        elapsed = time.time() - t0

        pred_sets = out['prediction_sets']
        coverage  = np.mean([y_test[i] in pred_sets[i] for i in range(len(y_test))])
        avg_size  = np.mean([len(s) for s in pred_sets])

        coverages.append(coverage)
        set_sizes.append(avg_size)
        times.append(elapsed)

    return np.mean(coverages), np.mean(set_sizes), np.mean(times)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", default="output/embeddings_cifar100.pt")
    parser.add_argument("--alpha",      type=float, default=None,
                        help="Single alpha (deprecated; use --alphas)")
    parser.add_argument("--alphas",    default="0.1",
                        help="Comma-separated alpha values (e.g. '0.1,0.5')")
    parser.add_argument("--n_trials",  type=int,   default=3)
    parser.add_argument("--test_size", type=int,   default=300,
                        help="Total test points per trial")
    parser.add_argument("--cal_sizes", default="100,200,300,400,600",
                        help="Comma-separated calibration sizes to sweep")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.1,
                        help="Temperature for tempered_density_ratio NCM")
    args = parser.parse_args()

    # Load
    print(f"Loading {args.embeddings_path}...")
    data = torch.load(args.embeddings_path)
    X    = data["embeddings"].numpy()
    y    = data["labels"].numpy()
    K    = len(np.unique(y))
    print(f"  {len(X)} samples  |  {K} classes  |  {X.shape[1]}D")

    # Resolve alphas: --alpha overrides --alphas for backwards compat
    if args.alpha is not None:
        alpha_list = [args.alpha]
    else:
        alpha_list = [float(a) for a in args.alphas.split(",")]

    cal_sizes = [int(s) for s in args.cal_sizes.split(",")]
    ncm_names = ["nn_ratio", "geodesic_nn_ratio", "mahal_nn_ratio",
                 "whitened_geodesic", "geodesic_topk_mean", "geodesic_topk_asym",
                 "tempered_density_ratio"]
    colors    = {"nn_ratio":                "#1f77b4",
                 "geodesic_nn_ratio":       "#ff7f0e",
                 "mahal_nn_ratio":          "#2ca02c",
                 "whitened_geodesic":       "#9467bd",
                 "geodesic_topk_mean":      "#d62728",
                 "geodesic_topk_asym":      "#8c564b",
                 "tempered_density_ratio":  "#e377c2"}
    labels_map = {"nn_ratio":               "nn_ratio (baseline)",
                  "geodesic_nn_ratio":      "geodesic_nn_ratio",
                  "mahal_nn_ratio":         "mahal_nn_ratio",
                  "whitened_geodesic":      "whitened_geodesic",
                  "geodesic_topk_mean":     "geodesic_topk_mean (sym)",
                  "geodesic_topk_asym":     "geodesic_topk_asym (new)",
                  "tempered_density_ratio": f"tempered_density_ratio (tau={args.tau})"}

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_tag = os.path.splitext(os.path.basename(args.embeddings_path))[0]

    for alpha in alpha_list:
        print(f"\n{'='*70}")
        print(f"  alpha = {alpha}")
        print(f"{'='*70}")

        results = {name: {"cov": [], "sz": [], "t": []} for name in ncm_names}

        for ncm_name in ncm_names:
            for cal_size in cal_sizes:
                print(f"  {ncm_name:25s}  cal={cal_size:4d} ...", end="", flush=True)
                cov, sz, t = run_one(X, y, ncm_name, cal_size, args.test_size,
                                      alpha, args.n_trials, args.seed, tau=args.tau)
                results[ncm_name]["cov"].append(cov)
                results[ncm_name]["sz"].append(sz)
                results[ncm_name]["t"].append(t)
                print(f"  cov={cov:.3f}  sz={sz:.2f}  t={t:.1f}s")

        # -------------------------------------------------------------------
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Full CP NCM Comparison — {os.path.basename(args.embeddings_path)}  "
            f"(alpha={alpha}, {args.n_trials} trials)",
            fontsize=13, fontweight='bold'
        )

        for ncm_name in ncm_names:
            c   = colors[ncm_name]
            lbl = labels_map[ncm_name]
            ls  = (0, (5, 2)) if ncm_name == 'tempered_density_ratio' \
                  else '--' if ncm_name in ('mahal_nn_ratio', 'whitened_geodesic') \
                  else '-.' if ncm_name == 'geodesic_topk_asym' \
                  else (0, (3, 1, 1, 1)) if ncm_name.startswith('lda_') else '-'
            axes[0].plot(cal_sizes, results[ncm_name]["cov"], marker='o',
                         color=c, label=lbl, linewidth=2, linestyle=ls)
            axes[1].plot(cal_sizes, results[ncm_name]["sz"],  marker='o',
                         color=c, label=lbl, linewidth=2, linestyle=ls)
            axes[2].plot(cal_sizes, results[ncm_name]["t"],   marker='o',
                         color=c, label=lbl, linewidth=2, linestyle=ls)

        # Coverage panel
        axes[0].axhline(1 - alpha, color='k', linestyle=':', linewidth=1.5,
                        label=f"Target ({1-alpha:.0%})")
        axes[0].set_title("Empirical Coverage"); axes[0].set_xlabel("Cal size")
        axes[0].set_ylabel("Coverage"); axes[0].legend(fontsize=9)
        axes[0].set_ylim(max(0.3, 1 - alpha - 0.3), 1.02); axes[0].grid(alpha=0.3)

        # Set size panel
        axes[1].set_title("Avg Prediction Set Size"); axes[1].set_xlabel("Cal size")
        axes[1].set_ylabel("Set size"); axes[1].legend(fontsize=9)
        axes[1].set_ylim(bottom=0); axes[1].grid(alpha=0.3)

        # Time panel
        axes[2].set_title("Wall Time (cal + predict)"); axes[2].set_xlabel("Cal size")
        axes[2].set_ylabel("Seconds"); axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        alpha_tag = f"alpha{int(alpha*100):02d}"
        out_path  = os.path.join(args.output_dir,
                                 f"ncm_comparison_{dataset_tag}_{alpha_tag}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\nSaved plot -> {out_path}")

        # -------------------------------------------------------------------
        # Summary table
        print("\n" + "="*80)
        print(f"{'NCM':25s}  {'cal':>5}  {'coverage':>9}  {'set_size':>9}  {'time(s)':>8}")
        print("-"*80)
        for cal_size, ci in zip(cal_sizes, range(len(cal_sizes))):
            for ncm_name in ncm_names:
                cov = results[ncm_name]["cov"][ci]
                sz  = results[ncm_name]["sz"][ci]
                t   = results[ncm_name]["t"][ci]
                valid = "OK" if cov >= (1 - alpha - 0.01) else "!!"
                print(f"{ncm_name:25s}  {cal_size:5d}  {cov:8.3f} {valid}  {sz:9.2f}  {t:8.1f}")
            print()

        # -------------------------------------------------------------------
        # Save JSON with raw numbers
        json_path = os.path.join(args.output_dir,
                                 f"ncm_comparison_{dataset_tag}_{alpha_tag}.json")
        json_data = {
            "dataset": dataset_tag,
            "alpha": alpha,
            "n_trials": args.n_trials,
            "test_size": args.test_size,
            "cal_sizes": cal_sizes,
            "results": {
                ncm: {
                    "cov": results[ncm]["cov"],
                    "sz":  results[ncm]["sz"],
                    "t":   results[ncm]["t"],
                } for ncm in ncm_names
            }
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved JSON  -> {json_path}")


if __name__ == "__main__":
    main()
