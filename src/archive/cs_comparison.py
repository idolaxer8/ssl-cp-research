"""
MA-CS vs MS-CS comparison on CIFAR-100.

Runs FCP baseline, FCP+MA-CS (optimal lambda), and FCP+MS-CS (optimal config)
across multiple cal sizes. Saves results and plot to output/class_similarity/.

Usage:
    python src/cs_comparison.py
    python src/cs_comparison.py --cal_sizes 300 400 600 800 --n_trials 5
"""

import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from conformal_prediction import FullConformalPredictor, create_ncm
from macs_experiment import run_fcp_with_macs, CIFAR100_FINE_TO_COARSE
from mscs_unlabeled_experiment import (
    build_cluster_similarity_matrix, run_fcp_with_mscs,
)

# ---------------------------------------------------------------------------
# Balanced split (equal per class)
# ---------------------------------------------------------------------------
def balanced_split(X, y, all_classes, cal_size, test_size, rng):
    K = len(all_classes)
    cal_pc = cal_size // K
    test_pc = test_size // K
    cal_idx, test_idx = [], []
    for c in all_classes:
        ci = np.where(y == c)[0]
        ci = rng.permutation(ci)
        cal_idx.append(ci[:cal_pc])
        test_idx.append(ci[cal_pc:cal_pc + test_pc])
    cal_idx = np.concatenate(cal_idx)
    test_idx = np.concatenate(test_idx)
    return X[cal_idx], y[cal_idx], X[test_idx], y[test_idx]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_sizes", type=int, nargs="+", default=[300, 400, 600, 800, 1000])
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ncm", type=str, default="geodesic_topk_mean")
    parser.add_argument("--alpha", type=float, default=0.1)
    # MA-CS config
    parser.add_argument("--macs_lambda", type=float, default=0.03)
    # MS-CS config (best from findings)
    parser.add_argument("--mscs_lambda", type=float, default=0.05)
    parser.add_argument("--mscs_n_clusters", type=int, default=20)
    parser.add_argument("--mscs_tau_mult", type=float, default=0.5,
                        help="tau = mult * median_d^2")
    args = parser.parse_args()

    out_dir = Path("output/class_similarity")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = torch.load("output/embeddings_cifar100.pt", map_location="cpu", weights_only=False)
    X = data["embeddings"].numpy()
    y = data["labels"].numpy()
    all_classes = np.unique(y)
    n_classes = len(all_classes)

    # Load unlabeled for MS-CS
    udata = torch.load("output/embeddings_cifar100_unlabeled.pt", map_location="cpu", weights_only=False)
    X_unlabeled = udata["embeddings"].numpy()

    print(f"CIFAR-100: {X.shape}, {n_classes} classes")
    print(f"Unlabeled: {X_unlabeled.shape}")
    print(f"NCM: {args.ncm}, alpha={args.alpha}, trials={args.n_trials}")
    print(f"MA-CS: lambda={args.macs_lambda} (superclass penalty)")
    print(f"MS-CS: lambda={args.mscs_lambda}, n_clusters={args.mscs_n_clusters}, "
          f"tau={args.mscs_tau_mult}*median_d^2")
    print()

    methods = ["FCP", "FCP+MA-CS", "FCP+MS-CS"]
    all_results = {}

    for cs in args.cal_sizes:
        all_results[cs] = {m: {"cov": [], "sz": []} for m in methods}
        t_start = time.time()

        for trial in range(args.n_trials):
            rng = np.random.default_rng(args.seed + trial * 1000)
            X_cal, y_cal, X_test, y_test = balanced_split(
                X, y, all_classes, cs, args.test_size, rng)

            # --- FCP baseline ---
            # approved: legacy whitened-NCM experiment (cal-fit whitening, non-exchangeable)
            ncm = create_ncm(args.ncm, k=5, allow_nonexchangeable=True)
            cp = FullConformalPredictor(ncm, alpha=args.alpha)
            cp.calibrate(X_cal, y_cal, all_classes=all_classes)
            m = cp.evaluate(X_test, y_test, verbose=False)
            all_results[cs]["FCP"]["cov"].append(m["coverage"])
            all_results[cs]["FCP"]["sz"].append(m["avg_set_size"])

            # --- FCP + MA-CS ---
            cov, sz = run_fcp_with_macs(
                X_cal, y_cal, X_test, y_test, all_classes,
                args.ncm, args.alpha, args.macs_lambda,
                allow_nonexchangeable=True)
            all_results[cs]["FCP+MA-CS"]["cov"].append(cov)
            all_results[cs]["FCP+MA-CS"]["sz"].append(sz)

            # --- FCP + MS-CS ---
            tau_arg = -args.mscs_tau_mult  # negative signals normalization
            (M, c2c, eff_tau, med_d2, cls_centroids,
             cls_counts, clust_centroids, clust_dists
             ) = build_cluster_similarity_matrix(
                X_unlabeled, X_cal, y_cal, all_classes,
                args.mscs_n_clusters, tau=tau_arg)

            cov, sz = run_fcp_with_mscs(
                X_cal, y_cal, X_test, y_test, all_classes,
                args.ncm, args.alpha, args.mscs_lambda, M,
                exchangeable=False, allow_nonexchangeable=True)
            all_results[cs]["FCP+MS-CS"]["cov"].append(cov)
            all_results[cs]["FCP+MS-CS"]["sz"].append(sz)

        elapsed = time.time() - t_start
        print(f"cal={cs} ({elapsed:.1f}s)")
        for m_name in methods:
            covs = all_results[cs][m_name]["cov"]
            szs = all_results[cs][m_name]["sz"]
            c_mean, s_mean = np.mean(covs), np.mean(szs)
            c_std, s_std = np.std(covs), np.std(szs)
            valid = "OK" if c_mean >= 0.89 else "!!"
            print(f"  {m_name:15s}  cov={c_mean:.3f}+/-{c_std:.3f} {valid}  "
                  f"sz={s_mean:.2f}+/-{s_std:.2f}")
        print()

    # --- Save summary ---
    summary_lines = []
    summary_lines.append(f"CIFAR-100: FCP vs MA-CS vs MS-CS ({args.ncm}, alpha={args.alpha}, {args.n_trials} trials)")
    summary_lines.append(f"MA-CS: lambda={args.macs_lambda}, MS-CS: lambda={args.mscs_lambda}, "
                         f"n_clusters={args.mscs_n_clusters}, tau={args.mscs_tau_mult}*median_d^2")
    summary_lines.append("=" * 80)
    hdr = f"{'Cal':>6}  {'Method':>15}  {'Coverage':>18}  {'Set Size':>18}"
    summary_lines.append(hdr)
    summary_lines.append("-" * 80)
    for cs in args.cal_sizes:
        first = True
        for m_name in methods:
            covs = all_results[cs][m_name]["cov"]
            szs = all_results[cs][m_name]["sz"]
            prefix = f"{cs:>6}" if first else " " * 6
            cov_str = f"{np.mean(covs)*100:.1f}%+/-{np.std(covs)*100:.1f}%"
            sz_str = f"{np.mean(szs):.2f}+/-{np.std(szs):.2f}"
            summary_lines.append(f"{prefix}  {m_name:>15}  {cov_str:>18}  {sz_str:>18}")
            first = False
        summary_lines.append("-" * 80)

    summary_txt = "\n".join(summary_lines)
    print("\n" + summary_txt)

    with open(out_dir / "macs_vs_mscs_summary.txt", "w") as f:
        f.write(summary_txt + "\n")

    # Save JSON
    json_results = {}
    for cs in args.cal_sizes:
        json_results[str(cs)] = {}
        for m_name in methods:
            json_results[str(cs)][m_name] = {
                "cov_mean": float(np.mean(all_results[cs][m_name]["cov"])),
                "cov_std": float(np.std(all_results[cs][m_name]["cov"])),
                "sz_mean": float(np.mean(all_results[cs][m_name]["sz"])),
                "sz_std": float(np.std(all_results[cs][m_name]["sz"])),
            }
    with open(out_dir / "macs_vs_mscs_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # --- Plot ---
    styles = {
        "FCP":        {"color": "#1f77b4", "marker": "o", "ls": "-",  "lw": 2.2},
        "FCP+MA-CS":  {"color": "#ff7f0e", "marker": "D", "ls": "--", "lw": 2.2},
        "FCP+MS-CS":  {"color": "#2ca02c", "marker": "s", "ls": "-.", "lw": 2.2},
    }

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(f"CIFAR-100 — FCP vs MA-CS vs MS-CS ({args.ncm}, {args.n_trials} trials)",
                 fontsize=13, fontweight="bold")

    for m_name in methods:
        st = styles[m_name]
        xs, mu_sz, sd_sz, mu_cov, sd_cov = [], [], [], [], []
        for cs in args.cal_sizes:
            covs = all_results[cs][m_name]["cov"]
            szs = all_results[cs][m_name]["sz"]
            xs.append(cs)
            mu_cov.append(np.mean(covs))
            sd_cov.append(np.std(covs))
            mu_sz.append(np.mean(szs))
            sd_sz.append(np.std(szs))

        xs = np.array(xs)
        mu_sz, sd_sz = np.array(mu_sz), np.array(sd_sz)
        mu_cov, sd_cov = np.array(mu_cov), np.array(sd_cov)

        axes[0].plot(xs, mu_sz, color=st["color"], ls=st["ls"], marker=st["marker"],
                     lw=st["lw"], ms=7, label=m_name)
        axes[0].fill_between(xs, mu_sz - sd_sz, mu_sz + sd_sz, alpha=0.12, color=st["color"])

        axes[1].plot(xs, mu_cov * 100, color=st["color"], ls=st["ls"], marker=st["marker"],
                     lw=st["lw"], ms=7, label=m_name)
        axes[1].fill_between(xs, (mu_cov - sd_cov) * 100, (mu_cov + sd_cov) * 100,
                             alpha=0.12, color=st["color"])

    axes[1].axhline(90, color="black", ls="--", lw=1.2, label="Target 90%")

    axes[0].set_ylabel("Avg Prediction Set Size", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Coverage (%)", fontsize=11)
    axes[1].set_xlabel("Calibration Set Size", fontsize=11)
    axes[1].set_ylim(max(80, min(mu_cov) * 100 - 5), 100)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(args.cal_sizes)

    plt.tight_layout()
    plot_path = out_dir / "macs_vs_mscs_cifar100.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {plot_path}")
    print(f"Saved summary: {out_dir / 'macs_vs_mscs_summary.txt'}")
    print(f"Saved JSON: {out_dir / 'macs_vs_mscs_results.json'}")


if __name__ == "__main__":
    main()
