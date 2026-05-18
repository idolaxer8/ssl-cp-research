"""
Ablation: dimensionality reduction x MS-CS on CIFAR-100.

6 methods (3 reductions x 2 CS options):
  FCP (768d), FCP+PCA-128, FCP+AE-32
  FCP+MS-CS, FCP+PCA-128+MS-CS, FCP+AE-32+MS-CS

MS-CS clusters are built in the REDUCED space (same as NCM operates in).
PCA/AE fit on unlabeled pool (exchangeability-safe).

Usage:
    python src/cs_ablation.py
    python src/cs_ablation.py --cal_sizes 400 600 800 --n_trials 5
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

from conformal_prediction import FullConformalPredictor, create_ncm
from mscs_unlabeled_experiment import build_cluster_similarity_matrix, run_fcp_with_mscs
from autoencoder_utils import EmbeddingAutoencoder


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
    return (X[np.concatenate(cal_idx)], y[np.concatenate(cal_idx)],
            X[np.concatenate(test_idx)], y[np.concatenate(test_idx)])


def run_fcp(X_cal, y_cal, X_test, y_test, all_classes, ncm_name, alpha):
    t0 = time.time()
    ncm = create_ncm(ncm_name, k=5)
    cp = FullConformalPredictor(ncm, alpha=alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)
    m = cp.evaluate(X_test, y_test, verbose=False)
    runtime = time.time() - t0
    return m["coverage"], m["avg_set_size"], runtime


def run_fcp_mscs(X_cal, y_cal, X_test, y_test, X_unlabeled,
                 all_classes, ncm_name, alpha, lam, n_clusters, tau_mult):
    t0 = time.time()
    tau_arg = -tau_mult
    (M, c2c, eff_tau, med_d2, cls_centroids,
     cls_counts, clust_centroids, clust_dists
     ) = build_cluster_similarity_matrix(
        X_unlabeled, X_cal, y_cal, all_classes, n_clusters, tau=tau_arg)
    cov, sz = run_fcp_with_mscs(
        X_cal, y_cal, X_test, y_test, all_classes,
        ncm_name, alpha, lam, M, exchangeable=False)
    runtime = time.time() - t0
    return cov, sz, runtime


def stratified_carve(X, y, all_classes, n_per_class, rng):
    """Carve n_per_class samples per class from (X, y). Returns (X_carve, y_carve, mask_kept)."""
    carve_idx, keep_mask = [], np.ones(len(y), dtype=bool)
    for c in all_classes:
        ci = np.where(y == c)[0]
        ci = rng.permutation(ci)
        take = ci[:n_per_class]
        carve_idx.append(take)
        keep_mask[take] = False
    carve_idx = np.concatenate(carve_idx)
    return X[carve_idx], y[carve_idx], keep_mask


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_sizes", type=int, nargs="+", default=[300, 400, 600, 800, 1000])
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ncm", type=str, default="geodesic_topk_mean")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--ae_dim", type=int, default=32)
    parser.add_argument("--mscs_lambda", type=float, default=0.05)
    parser.add_argument("--mscs_n_clusters", type=int, default=20)
    parser.add_argument("--mscs_tau_mult", type=float, default=0.5)
    parser.add_argument("--embeddings_path", type=str, default="output/embeddings_cifar100.pt")
    parser.add_argument("--unlabeled_path", type=str, default="output/embeddings_cifar100_unlabeled.pt",
                        help="If file missing, carve --unlabeled_per_class samples from labeled pool.")
    parser.add_argument("--unlabeled_per_class", type=int, default=40,
                        help="When carving unlabeled from labeled pool: samples per class.")
    parser.add_argument("--out_dir", type=str, default="output/class_similarity")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load labeled embeddings ---
    data = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)
    X = data["embeddings"].numpy()
    y = data["labels"].numpy()
    all_classes = np.unique(y)

    # --- Load (or carve) unlabeled embeddings ---
    if os.path.exists(args.unlabeled_path):
        udata = torch.load(args.unlabeled_path, map_location="cpu", weights_only=False)
        X_unlabeled = udata["embeddings"].numpy()
        print(f"Loaded unlabeled pool from {args.unlabeled_path}: {X_unlabeled.shape}")
    else:
        rng_carve = np.random.default_rng(args.seed)
        X_unlabeled, _, keep_mask = stratified_carve(
            X, y, all_classes, args.unlabeled_per_class, rng_carve)
        X, y = X[keep_mask], y[keep_mask]
        print(f"WARNING: {args.unlabeled_path} not found.")
        print(f"  Carved {X_unlabeled.shape[0]} samples ({args.unlabeled_per_class}/class) "
              f"from labeled pool as unlabeled.")
        print(f"  Remaining labeled pool: {X.shape[0]} samples.")

    print(f"CIFAR-100: {X.shape}, {len(all_classes)} classes")
    print(f"Unlabeled: {X_unlabeled.shape}")
    print(f"NCM: {args.ncm}, alpha={args.alpha}, trials={args.n_trials}")
    print(f"PCA-{args.pca_dim}, AE-{args.ae_dim}")
    print(f"MS-CS: lambda={args.mscs_lambda}, clusters={args.mscs_n_clusters}, "
          f"tau={args.mscs_tau_mult}*median_d^2")

    # --- Fit reductions on unlabeled pool (once) ---
    print("\nFitting PCA...")
    pca = PCA(n_components=args.pca_dim)
    pca.fit(X_unlabeled)
    print(f"  PCA-{args.pca_dim}: explained variance = {pca.explained_variance_ratio_.sum():.3f}")

    print("Fitting AE...")
    ae = EmbeddingAutoencoder(bottleneck_dim=args.ae_dim, seed=args.seed)
    ae.fit(X_unlabeled)

    # Pre-transform unlabeled pool for MS-CS cluster building
    X_u_pca = pca.transform(X_unlabeled)
    X_u_ae = ae.transform(X_unlabeled)

    methods = [
        "FCP",          "FCP+MS-CS",
        "FCP+PCA",      "FCP+PCA+MS-CS",
        "FCP+AE",       "FCP+AE+MS-CS",
    ]

    all_results = {}
    for cs in args.cal_sizes:
        all_results[cs] = {m: {"cov": [], "sz": [], "rt": []} for m in methods}
        t_start = time.time()

        for trial in range(args.n_trials):
            rng = np.random.default_rng(args.seed + trial * 1000)
            X_cal, y_cal, X_test, y_test = balanced_split(
                X, y, all_classes, cs, args.test_size, rng)

            # Reduce
            X_cal_pca = pca.transform(X_cal)
            X_test_pca = pca.transform(X_test)
            X_cal_ae = ae.transform(X_cal)
            X_test_ae = ae.transform(X_test)

            # --- FCP (768d) ---
            c, s, rt = run_fcp(X_cal, y_cal, X_test, y_test, all_classes, args.ncm, args.alpha)
            all_results[cs]["FCP"]["cov"].append(c)
            all_results[cs]["FCP"]["sz"].append(s)
            all_results[cs]["FCP"]["rt"].append(rt)

            # --- FCP+MS-CS (768d) ---
            c, s, rt = run_fcp_mscs(X_cal, y_cal, X_test, y_test, X_unlabeled,
                                    all_classes, args.ncm, args.alpha,
                                    args.mscs_lambda, args.mscs_n_clusters, args.mscs_tau_mult)
            all_results[cs]["FCP+MS-CS"]["cov"].append(c)
            all_results[cs]["FCP+MS-CS"]["sz"].append(s)
            all_results[cs]["FCP+MS-CS"]["rt"].append(rt)

            # --- FCP+PCA ---
            c, s, rt = run_fcp(X_cal_pca, y_cal, X_test_pca, y_test, all_classes, args.ncm, args.alpha)
            all_results[cs]["FCP+PCA"]["cov"].append(c)
            all_results[cs]["FCP+PCA"]["sz"].append(s)
            all_results[cs]["FCP+PCA"]["rt"].append(rt)

            # --- FCP+PCA+MS-CS (clusters in PCA space) ---
            c, s, rt = run_fcp_mscs(X_cal_pca, y_cal, X_test_pca, y_test, X_u_pca,
                                    all_classes, args.ncm, args.alpha,
                                    args.mscs_lambda, args.mscs_n_clusters, args.mscs_tau_mult)
            all_results[cs]["FCP+PCA+MS-CS"]["cov"].append(c)
            all_results[cs]["FCP+PCA+MS-CS"]["sz"].append(s)
            all_results[cs]["FCP+PCA+MS-CS"]["rt"].append(rt)

            # --- FCP+AE ---
            c, s, rt = run_fcp(X_cal_ae, y_cal, X_test_ae, y_test, all_classes, args.ncm, args.alpha)
            all_results[cs]["FCP+AE"]["cov"].append(c)
            all_results[cs]["FCP+AE"]["sz"].append(s)
            all_results[cs]["FCP+AE"]["rt"].append(rt)

            # --- FCP+AE+MS-CS (clusters in AE space) ---
            c, s, rt = run_fcp_mscs(X_cal_ae, y_cal, X_test_ae, y_test, X_u_ae,
                                    all_classes, args.ncm, args.alpha,
                                    args.mscs_lambda, args.mscs_n_clusters, args.mscs_tau_mult)
            all_results[cs]["FCP+AE+MS-CS"]["cov"].append(c)
            all_results[cs]["FCP+AE+MS-CS"]["sz"].append(s)
            all_results[cs]["FCP+AE+MS-CS"]["rt"].append(rt)

        elapsed = time.time() - t_start
        print(f"\ncal={cs} ({elapsed:.1f}s total wall)")
        for m_name in methods:
            covs = all_results[cs][m_name]["cov"]
            szs = all_results[cs][m_name]["sz"]
            rts = all_results[cs][m_name]["rt"]
            valid = "OK" if np.mean(covs) >= 0.89 else "!!"
            print(f"  {m_name:20s}  cov={np.mean(covs):.3f}+/-{np.std(covs):.3f} {valid}  "
                  f"sz={np.mean(szs):.2f}+/-{np.std(szs):.2f}  "
                  f"rt={np.mean(rts):.2f}s")

    # --- Summary ---
    print(f"\n{'='*108}")
    print(f"ABLATION: Reduction x MS-CS  (CIFAR-100, {args.ncm}, {args.n_trials} trials)")
    print(f"{'='*108}")
    hdr = f"{'Cal':>6}  {'Method':>20}  {'Coverage':>18}  {'Set Size':>18}  {'Runtime (s)':>16}"
    print(hdr)
    print("-" * 108)

    summary_lines = [
        f"CIFAR-100 Ablation: Reduction x MS-CS ({args.ncm}, alpha={args.alpha}, {args.n_trials} trials)",
        f"PCA-{args.pca_dim}, AE-{args.ae_dim}, MS-CS: lambda={args.mscs_lambda}, "
        f"clusters={args.mscs_n_clusters}, tau={args.mscs_tau_mult}*median_d^2",
        "=" * 108, hdr, "-" * 108,
    ]

    for cs in args.cal_sizes:
        first = True
        for m_name in methods:
            covs = all_results[cs][m_name]["cov"]
            szs = all_results[cs][m_name]["sz"]
            rts = all_results[cs][m_name]["rt"]
            prefix = f"{cs:>6}" if first else " " * 6
            cov_str = f"{np.mean(covs)*100:.1f}%+/-{np.std(covs)*100:.1f}%"
            sz_str = f"{np.mean(szs):.2f}+/-{np.std(szs):.2f}"
            rt_str = f"{np.mean(rts):.2f}+/-{np.std(rts):.2f}"
            line = f"{prefix}  {m_name:>20}  {cov_str:>18}  {sz_str:>18}  {rt_str:>16}"
            print(line)
            summary_lines.append(line)
            first = False
        print("-" * 108)
        summary_lines.append("-" * 108)

    with open(out_dir / "ablation_reduction_mscs_summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # Save JSON
    json_out = {}
    for cs in args.cal_sizes:
        json_out[str(cs)] = {}
        for m_name in methods:
            json_out[str(cs)][m_name] = {
                "cov_mean": float(np.mean(all_results[cs][m_name]["cov"])),
                "cov_std": float(np.std(all_results[cs][m_name]["cov"])),
                "sz_mean": float(np.mean(all_results[cs][m_name]["sz"])),
                "sz_std": float(np.std(all_results[cs][m_name]["sz"])),
                "rt_mean": float(np.mean(all_results[cs][m_name]["rt"])),
                "rt_std": float(np.std(all_results[cs][m_name]["rt"])),
            }
    with open(out_dir / "ablation_reduction_mscs_results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    # --- Plot ---
    styles = {
        "FCP":             {"color": "#1f77b4", "marker": "o", "ls": "-",  "lw": 2.0},
        "FCP+MS-CS":       {"color": "#1f77b4", "marker": "o", "ls": "--", "lw": 1.5},
        "FCP+PCA":         {"color": "#e377c2", "marker": "p", "ls": "-",  "lw": 2.0},
        "FCP+PCA+MS-CS":   {"color": "#e377c2", "marker": "p", "ls": "--", "lw": 2.5},
        "FCP+AE":          {"color": "#2ca02c", "marker": "s", "ls": "-",  "lw": 2.0},
        "FCP+AE+MS-CS":    {"color": "#2ca02c", "marker": "s", "ls": "--", "lw": 2.5},
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(f"CIFAR-100 — Ablation: Reduction × MS-CS ({args.ncm}, {args.n_trials} trials)",
                 fontsize=13, fontweight="bold")

    for m_name in methods:
        st = styles[m_name]
        xs, mu_sz, sd_sz, mu_cov, sd_cov, mu_rt, sd_rt = [], [], [], [], [], [], []
        for cs in args.cal_sizes:
            covs = all_results[cs][m_name]["cov"]
            szs = all_results[cs][m_name]["sz"]
            rts = all_results[cs][m_name]["rt"]
            xs.append(cs)
            mu_cov.append(np.mean(covs))
            sd_cov.append(np.std(covs))
            mu_sz.append(np.mean(szs))
            sd_sz.append(np.std(szs))
            mu_rt.append(np.mean(rts))
            sd_rt.append(np.std(rts))
        xs = np.array(xs)
        mu_sz, sd_sz = np.array(mu_sz), np.array(sd_sz)
        mu_cov, sd_cov = np.array(mu_cov), np.array(sd_cov)
        mu_rt, sd_rt = np.array(mu_rt), np.array(sd_rt)

        axes[0].plot(xs, mu_sz, color=st["color"], ls=st["ls"], marker=st["marker"],
                     lw=st["lw"], ms=7, label=m_name)
        axes[0].fill_between(xs, mu_sz - sd_sz, mu_sz + sd_sz, alpha=0.08, color=st["color"])
        axes[1].plot(xs, mu_cov * 100, color=st["color"], ls=st["ls"], marker=st["marker"],
                     lw=st["lw"], ms=7, label=m_name)
        axes[1].fill_between(xs, (mu_cov - sd_cov) * 100, (mu_cov + sd_cov) * 100,
                             alpha=0.08, color=st["color"])
        axes[2].plot(xs, mu_rt, color=st["color"], ls=st["ls"], marker=st["marker"],
                     lw=st["lw"], ms=7, label=m_name)
        axes[2].fill_between(xs, mu_rt - sd_rt, mu_rt + sd_rt, alpha=0.08, color=st["color"])

    axes[1].axhline(90, color="black", ls="--", lw=1.2, label="Target 90%")

    axes[0].set_ylabel("Avg Prediction Set Size", fontsize=11)
    axes[0].legend(fontsize=8.5, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Coverage (%)", fontsize=11)
    axes[1].set_ylim(85, 100)
    axes[1].legend(fontsize=8.5, ncol=2)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Runtime per trial (s)", fontsize=11)
    axes[2].set_xlabel("Calibration Set Size", fontsize=11)
    axes[2].legend(fontsize=8.5, ncol=2)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(args.cal_sizes)

    plt.tight_layout()
    plot_path = out_dir / "ablation_reduction_mscs_cifar100.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {out_dir / 'ablation_reduction_mscs_summary.txt'}")
    print(f"Saved: {out_dir / 'ablation_reduction_mscs_results.json'}")


if __name__ == "__main__":
    main()
