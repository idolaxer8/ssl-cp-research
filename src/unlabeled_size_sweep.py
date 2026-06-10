"""
Effect of unlabeled-pool size N on FCP+PCA+MS-CS performance.

For a fixed cal_size, we subsample the unlabeled pool to N points (random
without replacement) per trial, refit PCA and rebuild the MS-CS cluster matrix,
then evaluate four methods that use the unlabeled pool to varying degrees:

    FCP             — does not use the unlabeled pool (baseline, identical at all N)
    FCP+PCA         — uses unlabeled for PCA basis only
    FCP+MS-CS       — uses unlabeled for k-means clustering only (clusters in 768d)
    FCP+PCA+MS-CS   — uses unlabeled for both (clusters in PCA space — the winner)

Reports mean ± std over n_trials for coverage, set size, runtime at each N.

Usage:
    python src/unlabeled_size_sweep.py
    python src/unlabeled_size_sweep.py --cal_size 400 \\
        --unlabeled_sizes 0 100 250 500 1000 2500 5000 10000 --n_trials 5
"""
# ARCHIVE-CANDIDATE (review 2026-06-24): pool-size question settled (findings sec 3).
# If unused by the review date, move to src/archive/ -- watch list: src/archive/README.md.


import argparse
import json
import os
import re
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
from conformal_prediction import FullConformalPredictor, GeodesicTopKMeanNCM, create_ncm
from mscs_unlabeled_experiment import build_cluster_similarity_matrix, run_fcp_with_mscs

_USE_GPU = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Split (mirrors cs_ablation.balanced_split, simplified — assumes cal >= K)
# ---------------------------------------------------------------------------

def balanced_split(X, y, all_classes, cal_size, test_size, rng):
    K = len(all_classes)
    cal_pc = cal_size // K
    test_pc = test_size // K
    if cal_pc == 0:
        raise ValueError(
            f"cal_size={cal_size} < K={K}; this sweep requires stratified cal. "
            f"Use cal_size >= {K} (>= 2*K recommended)."
        )
    cal_idx, test_idx = [], []
    for c in all_classes:
        ci = np.where(y == c)[0]
        ci = rng.permutation(ci)
        cal_idx.append(ci[:cal_pc])
        test_idx.append(ci[cal_pc:cal_pc + test_pc])
    return (X[np.concatenate(cal_idx)], y[np.concatenate(cal_idx)],
            X[np.concatenate(test_idx)], y[np.concatenate(test_idx)])


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_fcp(X_cal, y_cal, X_test, y_test, all_classes, ncm_name, alpha,
            gpu_batch_size=256):
    """FCP with optional CUDA fast-path for GeodesicTopKMeanNCM-family NCMs."""
    t0 = time.time()
    # approved: legacy whitened-NCM experiment (cal-fit whitening, non-exchangeable)
    ncm = create_ncm(ncm_name, k=5, allow_nonexchangeable=True)
    cp = FullConformalPredictor(ncm, alpha=alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)
    if _USE_GPU and isinstance(ncm, GeodesicTopKMeanNCM):
        result = cp.predict(X_test, device="cuda",
                            gpu_batch_size=gpu_batch_size, verbose=False)
        prediction_sets = result["prediction_sets"]
        set_sizes = result["set_sizes"]
        coverage = float(np.mean([y_test[i] in ps for i, ps in enumerate(prediction_sets)]))
        avg_set_size = float(np.mean(set_sizes))
    else:
        m = cp.evaluate(X_test, y_test, verbose=False)
        coverage = m["coverage"]
        avg_set_size = m["avg_set_size"]
    return coverage, avg_set_size, time.time() - t0


def run_fcp_mscs(X_cal, y_cal, X_test, y_test, X_unlabeled,
                 all_classes, ncm_name, alpha, lam, n_clusters, tau_mult,
                 exchangeable=False):
    """FCP + MS-CS penalty.

    `exchangeable=True` rebuilds the cluster similarity matrix M and cal y_hat
    per (test, candidate) pair so the augmented bag is symmetric. Restores
    O(1/n_cal) exchangeability error at the cost of ~500x slowdown. Worth it
    when cal/n_classes < 5 (per-class centroid noise dominates); negligible
    gain above that ratio. See findings.md §4.4.
    """
    t0 = time.time()
    (M, c2c, eff_tau, _, cls_centroids, cls_counts,
     clust_centroids, clust_dists) = build_cluster_similarity_matrix(
        X_unlabeled, X_cal, y_cal, all_classes, n_clusters, tau=-tau_mult)
    if exchangeable:
        cov, sz = run_fcp_with_mscs(
            X_cal, y_cal, X_test, y_test, all_classes,
            ncm_name, alpha, lam, M, exchangeable=True,
            class_centroids=cls_centroids, class_counts=cls_counts,
            class_to_cluster=c2c, cluster_centroids=clust_centroids,
            cluster_dists=clust_dists, effective_tau=eff_tau,
            allow_nonexchangeable=True)
    else:
        cov, sz = run_fcp_with_mscs(
            X_cal, y_cal, X_test, y_test, all_classes,
            ncm_name, alpha, lam, M, exchangeable=False,
            allow_nonexchangeable=True)
    return cov, sz, time.time() - t0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_dataset_tag(path: str) -> str:
    """Pull a short dataset tag from an embeddings filename.

    e.g. 'output/embeddings_cifar100.pt'             -> 'cifar100'
         'output/embeddings_miniimagenet_extra.pt'   -> 'miniimagenet'
    Falls back to the bare stem if no 'embeddings_' prefix is present.
    """
    stem = Path(path).stem
    m = re.match(r"embeddings_([a-zA-Z0-9]+)", stem)
    return m.group(1) if m else stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_size", type=int, default=400)
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--unlabeled_sizes", type=int, nargs="+",
                        default=[0, 100, 250, 500, 1000, 2500, 5000, 10000])
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ncm", type=str, default="geodesic_topk_mean")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--mscs_lambda", type=float, default=0.05)
    parser.add_argument("--mscs_n_clusters", type=int, default=20)
    parser.add_argument("--mscs_tau_mult", type=float, default=0.5)
    parser.add_argument("--mscs_exchangeable", action="store_true",
                        help="Restore exchangeability in MS-CS by rebuilding M and "
                             "cal y_hat per candidate. ~500x slower but recovers "
                             "0.5-1.3%% coverage at cal/K<5 (findings.md §4.4).")
    parser.add_argument("--embeddings_path", type=str, default="output/embeddings_cifar100.pt")
    parser.add_argument("--unlabeled_path", type=str, default="output/embeddings_cifar100_unlabeled.pt")
    parser.add_argument("--out_dir", type=str, default="output/unlabeled_size_sweep")
    parser.add_argument("--dataset_tag", type=str, default=None,
                        help="Dataset name used in plot title. Inferred from "
                             "embeddings filename if not given.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load embeddings ---
    data = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)
    X = data["embeddings"].numpy()
    y = data["labels"].numpy()
    all_classes = np.unique(y)
    K = len(all_classes)
    dataset_tag = args.dataset_tag or infer_dataset_tag(args.embeddings_path)

    if not os.path.exists(args.unlabeled_path):
        raise FileNotFoundError(
            f"Unlabeled embeddings not found at {args.unlabeled_path}. "
            f"This sweep requires a pre-extracted disjoint unlabeled pool."
        )
    udata = torch.load(args.unlabeled_path, map_location="cpu", weights_only=False)
    X_unlabeled_full = udata["embeddings"].numpy()
    N_max = len(X_unlabeled_full)

    print(f"Dataset:    {dataset_tag}")
    print(f"Labeled:    {X.shape}  ({K} classes)")
    print(f"Unlabeled:  {X_unlabeled_full.shape}  (max N for sweep)")
    print(f"Cal size:   {args.cal_size}   test size: {args.test_size}   trials: {args.n_trials}")
    print(f"NCM:        {args.ncm}   alpha: {args.alpha}")
    print(f"PCA dim:    {args.pca_dim} (auto-reduces if N-1 < pca_dim)")
    print(f"MS-CS:      lambda={args.mscs_lambda}, n_clusters={args.mscs_n_clusters}, "
          f"tau={args.mscs_tau_mult}*median_d^2  (skipped if N < n_clusters)")
    print(f"MS-CS exchangeable: {args.mscs_exchangeable}")
    print(f"GPU FCP:    {_USE_GPU}")

    # Warn if cal/K is in the regime where non-exchangeable MS-CS under-covers
    cal_per_class = args.cal_size / K
    if cal_per_class < 5 and not args.mscs_exchangeable:
        print(f"WARNING: cal/K = {cal_per_class:.1f} < 5 — MS-CS non-exchangeable "
              f"approximation can under-cover by 0.5-1.3%. Pass --mscs_exchangeable "
              f"for a rigorous (but ~500x slower) run; see findings.md §4.4.")

    # Clamp unlabeled sizes to N_max
    requested = sorted(set(args.unlabeled_sizes))
    unlabeled_sizes = [min(N, N_max) for N in requested]
    if any(N > N_max for N in requested):
        print(f"WARNING: requested unlabeled sizes capped at N_max={N_max}.")

    methods = ["FCP", "FCP+PCA", "FCP+MS-CS", "FCP+PCA+MS-CS"]
    # results[N][method] = list of dicts {cov, sz, rt, pca_dim, n_used}
    results = {N: {m: [] for m in methods} for N in unlabeled_sizes}

    # ----------------------------------------------------------------
    # Sweep
    # ----------------------------------------------------------------
    for N in unlabeled_sizes:
        eff_pca_dim = min(args.pca_dim, max(N - 1, 1))
        t_block_start = time.time()
        print(f"\n--- N={N} (PCA-{eff_pca_dim}, MS-CS {'on' if N >= args.mscs_n_clusters else 'SKIPPED — N<n_clusters'}) ---")

        for trial in range(args.n_trials):
            rng = np.random.default_rng(args.seed + trial * 1000)
            rng_u = np.random.default_rng(args.seed + trial * 1000 + 7)
            X_cal, y_cal, X_test, y_test = balanced_split(
                X, y, all_classes, args.cal_size, args.test_size, rng)

            # FCP (no unlabeled used) — identical at all N, but rerun per trial
            # since the cal/test split changes per trial.
            cov, sz, rt = run_fcp(X_cal, y_cal, X_test, y_test, all_classes,
                                  args.ncm, args.alpha)
            results[N]["FCP"].append({"cov": cov, "sz": sz, "rt": rt,
                                      "pca_dim": None, "n_used": 0})

            if N == 0:
                # Other methods need unlabeled — skip them, record NaN sentinels
                for m in ["FCP+PCA", "FCP+MS-CS", "FCP+PCA+MS-CS"]:
                    results[N][m].append({"cov": np.nan, "sz": np.nan, "rt": np.nan,
                                          "pca_dim": None, "n_used": 0})
                continue

            # Random subsample of unlabeled pool
            u_idx = rng_u.permutation(N_max)[:N]
            X_u_sub = X_unlabeled_full[u_idx]

            # PCA refit on subsample
            pca = PCA(n_components=eff_pca_dim)
            pca.fit(X_u_sub)
            X_cal_pca = pca.transform(X_cal)
            X_test_pca = pca.transform(X_test)
            X_u_sub_pca = pca.transform(X_u_sub)

            # FCP+PCA (uses unlabeled for PCA basis only)
            cov, sz, rt = run_fcp(X_cal_pca, y_cal, X_test_pca, y_test, all_classes,
                                  args.ncm, args.alpha)
            results[N]["FCP+PCA"].append({"cov": cov, "sz": sz, "rt": rt,
                                          "pca_dim": eff_pca_dim, "n_used": N})

            if N >= args.mscs_n_clusters:
                # FCP+MS-CS (no reduction, clusters in 768d on X_u_sub)
                cov, sz, rt = run_fcp_mscs(
                    X_cal, y_cal, X_test, y_test, X_u_sub,
                    all_classes, args.ncm, args.alpha,
                    args.mscs_lambda, args.mscs_n_clusters, args.mscs_tau_mult,
                    exchangeable=args.mscs_exchangeable)
                results[N]["FCP+MS-CS"].append({"cov": cov, "sz": sz, "rt": rt,
                                                "pca_dim": None, "n_used": N})

                # FCP+PCA+MS-CS (clusters in PCA space, same N)
                cov, sz, rt = run_fcp_mscs(
                    X_cal_pca, y_cal, X_test_pca, y_test, X_u_sub_pca,
                    all_classes, args.ncm, args.alpha,
                    args.mscs_lambda, args.mscs_n_clusters, args.mscs_tau_mult,
                    exchangeable=args.mscs_exchangeable)
                results[N]["FCP+PCA+MS-CS"].append({"cov": cov, "sz": sz, "rt": rt,
                                                    "pca_dim": eff_pca_dim, "n_used": N})
            else:
                for m in ["FCP+MS-CS", "FCP+PCA+MS-CS"]:
                    results[N][m].append({"cov": np.nan, "sz": np.nan, "rt": np.nan,
                                          "pca_dim": None, "n_used": N})

        # Per-N summary
        elapsed = time.time() - t_block_start
        print(f"  block wall: {elapsed:.1f}s")
        for m in methods:
            covs = [r["cov"] for r in results[N][m] if not np.isnan(r["cov"])]
            szs = [r["sz"] for r in results[N][m] if not np.isnan(r["sz"])]
            rts = [r["rt"] for r in results[N][m] if not np.isnan(r["rt"])]
            if covs:
                tag = "OK" if np.mean(covs) >= 0.89 else "!!"
                print(f"  {m:18s}  cov={np.mean(covs):.3f}+/-{np.std(covs):.3f} {tag}  "
                      f"sz={np.mean(szs):.2f}+/-{np.std(szs):.2f}  "
                      f"rt={np.mean(rts):.2f}s")
            else:
                print(f"  {m:18s}  (skipped at N={N})")

    # ----------------------------------------------------------------
    # Summary table + JSON
    # ----------------------------------------------------------------
    hdr = f"{'N':>7}  {'Method':>18}  {'Coverage':>18}  {'Set Size':>18}  {'Runtime (s)':>14}"
    summary = [
        f"Unlabeled-pool size sweep — {dataset_tag} "
        f"({args.ncm}, alpha={args.alpha}, cal={args.cal_size}, {args.n_trials} trials)",
        f"PCA dim (cap): {args.pca_dim}; MS-CS: lambda={args.mscs_lambda}, "
        f"clusters={args.mscs_n_clusters}, tau={args.mscs_tau_mult}*median_d^2, "
        f"exchangeable={args.mscs_exchangeable}",
        "=" * 90, hdr, "-" * 90,
    ]
    print("\n" + "=" * 90)
    print(summary[0])
    print(summary[1])
    print("=" * 90)
    print(hdr)
    print("-" * 90)

    json_out = {
        "_meta": {
            "dataset_tag": dataset_tag,
            "n_classes": int(K),
            "cal_size": int(args.cal_size),
            "test_size": int(args.test_size),
            "n_trials": int(args.n_trials),
            "ncm": args.ncm,
            "alpha": float(args.alpha),
            "pca_dim_cap": int(args.pca_dim),
            "mscs_lambda": float(args.mscs_lambda),
            "mscs_n_clusters": int(args.mscs_n_clusters),
            "mscs_tau_mult": float(args.mscs_tau_mult),
            "mscs_exchangeable": bool(args.mscs_exchangeable),
        }
    }
    for N in unlabeled_sizes:
        first = True
        json_out[str(N)] = {}
        for m in methods:
            valid = [r for r in results[N][m] if not np.isnan(r["cov"])]
            if not valid:
                json_out[str(N)][m] = None
                line = f"{N:>7}" if first else " " * 7
                line += f"  {m:>18}  {'(skipped)':>18}  {'':>18}  {'':>14}"
                print(line); summary.append(line); first = False
                continue
            covs = np.array([r["cov"] for r in valid])
            szs = np.array([r["sz"] for r in valid])
            rts = np.array([r["rt"] for r in valid])
            json_out[str(N)][m] = {
                "cov_mean": float(covs.mean()), "cov_std": float(covs.std()),
                "sz_mean":  float(szs.mean()),  "sz_std":  float(szs.std()),
                "rt_mean":  float(rts.mean()),  "rt_std":  float(rts.std()),
                "n_used":   int(valid[0]["n_used"]),
                "pca_dim":  valid[0]["pca_dim"],
            }
            line = (f"{N:>7}" if first else " " * 7)
            line += (f"  {m:>18}  "
                     f"{covs.mean()*100:>8.1f}%+/-{covs.std()*100:.1f}%  "
                     f"{szs.mean():>8.2f}+/-{szs.std():.2f}  "
                     f"{rts.mean():>6.2f}+/-{rts.std():.2f}")
            print(line); summary.append(line); first = False
        print("-" * 90); summary.append("-" * 90)

    (out_dir / "unlabeled_size_sweep_summary.txt").write_text("\n".join(summary) + "\n")
    with open(out_dir / "unlabeled_size_sweep_results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    # ----------------------------------------------------------------
    # Plot — set size vs N (log x), coverage vs N
    # ----------------------------------------------------------------
    styles = {
        "FCP":             {"color": "#1f77b4", "marker": "o", "ls": "-"},
        "FCP+PCA":         {"color": "#e377c2", "marker": "p", "ls": "-"},
        "FCP+MS-CS":       {"color": "#2ca02c", "marker": "s", "ls": "-"},
        "FCP+PCA+MS-CS":   {"color": "#bcbd22", "marker": "*", "ls": "-"},
    }
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"Unlabeled-pool size sweep — {dataset_tag} "
        f"({args.ncm}, cal={args.cal_size}, {args.n_trials} trials)",
        fontsize=13, fontweight="bold")

    for m in methods:
        st = styles[m]
        xs, mu_sz, sd_sz, mu_cov, sd_cov = [], [], [], [], []
        for N in unlabeled_sizes:
            valid = [r for r in results[N][m] if not np.isnan(r["cov"])]
            if not valid:
                continue
            xs.append(max(N, 1))   # log-axis safe (N=0 plotted at 1)
            covs = np.array([r["cov"] for r in valid])
            szs = np.array([r["sz"] for r in valid])
            mu_sz.append(szs.mean()); sd_sz.append(szs.std())
            mu_cov.append(covs.mean()); sd_cov.append(covs.std())
        if not xs:
            continue
        xs = np.array(xs); mu_sz = np.array(mu_sz); sd_sz = np.array(sd_sz)
        mu_cov = np.array(mu_cov); sd_cov = np.array(sd_cov)
        axes[0].plot(xs, mu_sz, color=st["color"], marker=st["marker"], ls=st["ls"],
                     lw=2.0, ms=8, label=m)
        axes[0].fill_between(xs, mu_sz - sd_sz, mu_sz + sd_sz, alpha=0.12, color=st["color"])
        axes[1].plot(xs, mu_cov * 100, color=st["color"], marker=st["marker"], ls=st["ls"],
                     lw=2.0, ms=8, label=m)
        axes[1].fill_between(xs, (mu_cov - sd_cov) * 100, (mu_cov + sd_cov) * 100,
                             alpha=0.12, color=st["color"])

    axes[1].axhline((1 - args.alpha) * 100, color="black", ls="--", lw=1.2,
                    label=f"target {int((1-args.alpha)*100)}%")
    axes[0].set_ylabel("Avg Prediction Set Size", fontsize=11)
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)
    axes[1].set_ylabel("Coverage (%)", fontsize=11)
    axes[1].set_xlabel("Unlabeled pool size N  (symlog; N=0 shown at 1)", fontsize=11)
    axes[1].set_ylim(85, 100); axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)

    plt.tight_layout()
    plot_path = out_dir / "unlabeled_size_sweep.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {out_dir / 'unlabeled_size_sweep_summary.txt'}")
    print(f"Saved: {out_dir / 'unlabeled_size_sweep_results.json'}")


if __name__ == "__main__":
    main()
