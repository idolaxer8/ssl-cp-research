"""
MS-CS vs MA-CS head-to-head comparison (CIFAR-100).

Both MA-CS (Fargion et al. 2025, binary superclass penalty) and MS-CS
(continuous class-similarity penalty) are the SAME post-hoc FCP penalty

    s_lambda(x, y) = s(x, y) + lambda * (1 - M[y, yhat(x)])

differing only in the class-similarity matrix M:

  MA-CS          M[c,c'] = 1 if g(c)==g(c') else 0   (binary superclass
                 co-membership, from CIFAR-100 coarse labels). Then
                 1 - M[y,yhat] = I{g(y) != g(yhat)}, exactly the MA-CS
                 indicator penalty. Needs SUPERCLASS LABELS.
  MS-CS-centroid M[c,c'] = exp(-||mu_c - mu_c'||^2 / tau), centroids from the
                 CALIBRATION set only. Needs NOTHING extra (cal-only).
  MS-CS-cluster  k-means on the (transformed) EXTERNAL unlabeled pool -> cluster
                 co-assignment + inter-cluster distance kernel. Needs UNLABELED
                 DATA.

Running all three through the SAME engine (run_fcp_with_mscs) under IDENTICAL
splits, the SAME feature pipeline, the SAME NCM, the SAME yhat rule, and the SAME
exchangeable handling means the comparison isolates exactly one thing: which
similarity matrix M yields the tightest valid sets.

OPTIMAL PIPELINE (defaults here): the fully-exchangeable feature pipeline of
exchangeable_fcp_experiment.py / exchangeable_features.UnlabeledTransform:
PCA (fit on the unlabeled pool) + within-cluster whitening (fit on the unlabeled
pool) -> an UNWHITENED NCM (unwhitened_topk_mean) scores the transformed
features. Every data-dependent transform is fit on the independent unlabeled pool
(never on cal), so coverage >= 1-alpha holds at every cal size. The MS-CS cluster
matrix is built from the SAME transformed unlabeled pool. A uniform RANDOM
cal/test split keeps the guarantee exact and tight (~0.90); stratified over-covers.

Validity: all methods run in exchangeable mode (M and yhat recomputed per
augmented candidate set). For MA-CS the superclass M is data-independent (and
dimension-independent), so its exchangeable M-update is the identity (only yhat
is recomputed) and PCA leaves it untouched.

Outputs (output/mscs_vs_macs_pca/ by default):
  results.json      config + every (cal, method, lambda) row (cov, sz)
  summary.txt       human-readable best-lambda-per-method table + % reductions
  mscs_vs_macs.png  set size vs cal_size, best lambda per method
"""
import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import make_transform
from mscs_unlabeled_experiment import (
    run_fcp_with_mscs,
    build_cluster_similarity_matrix,
    build_centroid_similarity_matrix,
    update_centroid_M_for_candidate,
)
from macs_experiment import CIFAR100_FINE_TO_COARSE


def build_superclass_matrix(all_classes):
    """MA-CS similarity matrix: M[i,j] = 1 if same CIFAR-100 superclass else 0.

    Data- and dimension-independent (depends only on the fixed fine->coarse map),
    so in exchangeable mode its M-update is the identity and PCA never touches it.
    Feeding this M to the MS-CS engine reproduces the MA-CS indicator penalty:
        lambda * (1 - M[y, yhat]) = lambda * I{g(y) != g(yhat)}.
    """
    coarse = np.array([CIFAR100_FINE_TO_COARSE[int(c)] for c in all_classes])
    M = (coarse[:, None] == coarse[None, :]).astype(np.float64)
    np.fill_diagonal(M, 1.0)
    return M


# --------------------------- cal/test splits (index-based) -------------------
def random_split_idx(y, cal_size, test_size, all_classes, rng):
    """Uniform random partition -> exactly exchangeable (tight coverage)."""
    idx = rng.permutation(len(y))
    return idx[:cal_size], idx[cal_size:cal_size + test_size]


def stratified_split_idx(y, cal_size, test_size, all_classes, rng):
    """Stratified pool -> cal + test (mirrors mscs_unlabeled_experiment, but
    returns global indices so the transform is applied AFTER splitting).
    Label-dependent on cal -> empirically conservative (over-covers)."""
    K = len(all_classes)
    num_per_class = math.ceil((test_size + cal_size) / K)
    min_count = min(int((y == c).sum()) for c in all_classes)
    num_per_class = min(num_per_class, min_count)

    pool = []
    for c in all_classes:
        c_idx = np.where(y == c)[0]
        pool.append(rng.choice(c_idx, num_per_class, replace=False))
    pool = np.concatenate(pool)
    rng.shuffle(pool)

    test_idx = pool[-test_size:]
    rem = pool[:-test_size]
    rem_classes = np.unique(y[rem])
    first = np.array([rng.choice(rem[y[rem] == c], 1, replace=False)[0]
                      for c in rem_classes])
    rest = np.setdiff1d(rem, first)
    n_extra = cal_size - len(rem_classes)
    if n_extra > 0:
        extra = rng.choice(rest, min(n_extra, len(rest)), replace=False)
        cal_idx = np.concatenate([first, extra])
    else:
        cal_idx = first[:cal_size]
    cal_idx = rng.permutation(cal_idx)
    return cal_idx, test_idx


SPLITS = {"random": random_split_idx, "stratified": stratified_split_idx}


def eval_baseline_fcp(X_cal, y_cal, X_test, y_test, all_classes, ncm_name, alpha):
    """Plain FCP, no penalty (lambda = 0), on already-transformed features."""
    ncm = create_ncm(ncm_name, k=5)
    cp = FullConformalPredictor(ncm, alpha=alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)
    m = cp.evaluate(X_test, y_test, verbose=False)
    return m["coverage"], m["avg_set_size"]


def main():
    import argparse
    p = argparse.ArgumentParser(description="MS-CS vs MA-CS head-to-head (CIFAR-100, optimal PCA pipeline)")
    p.add_argument("--embeddings_path", type=str,
                   default="output/from_cluster/embeddings_cifar100.pt")
    p.add_argument("--unlabeled_path", type=str,
                   default="output/from_cluster/embeddings_cifar100_unlabeled.pt")
    p.add_argument("--output_dir", type=str, default="output/mscs_vs_macs_pca")
    # Optimal exchangeable pipeline defaults:
    p.add_argument("--ncm", type=str, default="unwhitened_topk_mean",
                   help="Use a whiten=False NCM for exchangeability with the "
                        "pool-fit whitening (unwhitened_topk_mean / _asym).")
    p.add_argument("--pca_dim", type=int, default=128, help="PCA dim (0/negative = off).")
    p.add_argument("--whiten", type=str, default="cluster",
                   choices=["cluster", "global", "none"],
                   help="Within-cluster (default) / global / no pool-fit whitening.")
    p.add_argument("--n_clusters_whiten", type=int, default=100,
                   help="k-means clusters for the pool-fit whitening (~n_classes).")
    p.add_argument("--split", type=str, default="random", choices=list(SPLITS),
                   help="random = exact exchangeable, tight; stratified = conservative.")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--cal_sizes", type=int, nargs="+", default=[300, 400, 600, 800])
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_clusters_mscs", type=int, default=20)
    p.add_argument("--tau", type=float, default=0.5,
                   help="MS-CS kernel temperature as a multiple of median_d^2.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tau_arg = -args.tau  # negative => effective_tau = |tau| * median_d^2
    whiten = None if args.whiten == "none" else args.whiten
    pca_dim = args.pca_dim if args.pca_dim and args.pca_dim > 0 else None

    # --- Load data ---
    data = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)
    X = data["embeddings"].numpy()
    y = data["labels"].numpy()
    all_classes = np.unique(y)
    Xu = torch.load(args.unlabeled_path, map_location="cpu",
                    weights_only=False)["embeddings"].numpy()

    # --- Fit the exchangeable feature transform on the unlabeled pool (ONCE) ---
    transform = make_transform(Xu, pca_dim=pca_dim, whiten=whiten,
                               n_clusters=args.n_clusters_whiten)
    Xu_mscs = getattr(transform, "Xu_transformed_", None)
    if Xu_mscs is None:
        Xu_mscs = transform.transform(Xu)

    print(f"Labeled (cal/test pool): {X.shape}, {len(all_classes)} classes")
    print(f"Unlabeled pool: {Xu.shape} -> {transform}")
    print(f"NCM={args.ncm}  alpha={args.alpha}  trials={args.n_trials}  "
          f"split={args.split}  exchangeable=True  yhat=ncm")
    print(f"Pipeline: PCA={pca_dim}, whiten={whiten} (n_clusters_whiten="
          f"{args.n_clusters_whiten}) fit on unlabeled pool")
    print(f"cal_sizes={args.cal_sizes}  lambdas={args.lambdas}  "
          f"MS-CS: n_clusters={args.n_clusters_mscs}, tau={args.tau}*median_d^2")
    exch = "exact (random split + pool-fit transform)" if args.split == "random" \
        else "conservative (stratified cal over-covers)"
    print(f"EXCHANGEABLE: {exch}")
    print()

    # MA-CS superclass M: fixed across all splits AND dimensions.
    M_super = build_superclass_matrix(all_classes)
    identity_update = lambda yc_idx, x_test, M_base: M_base
    n_super = len(np.unique([CIFAR100_FINE_TO_COARSE[int(c)] for c in all_classes]))
    off = (M_super.sum() - M_super.trace()) / (len(all_classes) ** 2 - len(all_classes))
    print(f"MA-CS superclass M: {M_super.shape}, {n_super} superclasses, "
          f"mean off-diag={off:.3f}\n")

    config = {
        "experiment": "mscs_vs_macs_pca",
        "dataset": "cifar100",
        "embeddings_path": args.embeddings_path,
        "unlabeled_path": args.unlabeled_path,
        "ncm": args.ncm,
        "pca_dim": pca_dim,
        "whiten": whiten,
        "n_clusters_whiten": args.n_clusters_whiten,
        "alpha": args.alpha,
        "exchangeable": True,
        "yhat_mode": "ncm",
        "split": args.split,
        "n_trials": args.n_trials,
        "cal_sizes": args.cal_sizes,
        "test_size": args.test_size,
        "lambdas": args.lambdas,
        "mscs_n_clusters": args.n_clusters_mscs,
        "mscs_tau_mult": args.tau,
        "methods": {
            "MA-CS": "binary superclass M (needs coarse labels)",
            "MS-CS-centroid": "cal-only class-centroid M (no side info)",
            "MS-CS-cluster": "k-means on transformed external unlabeled M",
        },
    }

    rows = []
    json_path = os.path.join(args.output_dir, "results.json")
    split_fn = SPLITS[args.split]

    for cal_size in args.cal_sizes:
        print("=" * 70)
        print(f"cal_size = {cal_size}")
        print("=" * 70)

        acc = {}

        def add(method, lam, cov, sz):
            acc.setdefault((method, lam), []).append((cov, sz))

        t_cal0 = time.time()
        for trial in range(args.n_trials):
            rng = np.random.default_rng(args.seed + trial * 1000)
            ci, ti = split_fn(y, cal_size, args.test_size, all_classes, rng)
            X_cal, y_cal = transform.transform(X[ci]), y[ci]
            X_test, y_test = transform.transform(X[ti]), y[ti]

            # --- Baseline FCP (shared by all methods, same split) ---
            cov, sz = eval_baseline_fcp(
                X_cal, y_cal, X_test, y_test, all_classes, args.ncm, args.alpha)
            add("FCP", 0.0, cov, sz)

            # --- Build the two data-dependent M matrices for this split ---
            (M_cent, eff_tau_c, _med_c, cc_c, cnt_c
             ) = build_centroid_similarity_matrix(X_cal, y_cal, all_classes, tau=tau_arg)
            cent_update = (lambda yc_idx, x_test, M_base:
                           update_centroid_M_for_candidate(
                               cc_c, cnt_c, eff_tau_c, yc_idx, x_test, M_base))

            (M_clu, c2c, eff_tau_k, _med_k, cc_k, cnt_k, clustC, clustD
             ) = build_cluster_similarity_matrix(
                Xu_mscs, X_cal, y_cal, all_classes, args.n_clusters_mscs, tau=tau_arg)

            # --- Penalty methods over the lambda grid ---
            for lam in args.lambdas:
                cov, sz = run_fcp_with_mscs(
                    X_cal, y_cal, X_test, y_test, all_classes, args.ncm,
                    args.alpha, lam, M_super, exchangeable=True,
                    update_M_fn=identity_update, yhat_mode="ncm")
                add("MA-CS", lam, cov, sz)

                cov, sz = run_fcp_with_mscs(
                    X_cal, y_cal, X_test, y_test, all_classes, args.ncm,
                    args.alpha, lam, M_cent, exchangeable=True,
                    class_centroids=cc_c, class_counts=cnt_c,
                    effective_tau=eff_tau_c, update_M_fn=cent_update,
                    yhat_mode="ncm")
                add("MS-CS-centroid", lam, cov, sz)

                cov, sz = run_fcp_with_mscs(
                    X_cal, y_cal, X_test, y_test, all_classes, args.ncm,
                    args.alpha, lam, M_clu, exchangeable=True,
                    class_centroids=cc_k, class_counts=cnt_k,
                    class_to_cluster=c2c, cluster_centroids=clustC,
                    cluster_dists=clustD, effective_tau=eff_tau_k,
                    yhat_mode="ncm")
                add("MS-CS-cluster", lam, cov, sz)

            print(f"  trial {trial+1}/{args.n_trials} done "
                  f"({time.time()-t_cal0:.0f}s elapsed)")

        # --- Aggregate + print this cal_size ---
        print(f"\n  {'method':16s} {'lambda':>7s} {'cov':>7s} {'sz':>7s}  valid")
        print(f"  {'-'*48}")
        for (method, lam), vals in sorted(acc.items(),
                                          key=lambda kv: (kv[0][0], kv[0][1])):
            covs = np.array([v[0] for v in vals])
            szs = np.array([v[1] for v in vals])
            cov_m, sz_m = float(covs.mean()), float(szs.mean())
            valid = "OK" if cov_m >= 0.89 else "!!"
            rows.append({
                "cal": cal_size, "method": method, "lam": round(lam, 4),
                "cov": cov_m, "sz": sz_m, "cov_std": float(covs.std()),
                "sz_std": float(szs.std()),
            })
            print(f"  {method:16s} {lam:7.2f} {cov_m:7.3f} {sz_m:7.2f}  {valid}")
        print()

        with open(json_path, "w") as f:
            json.dump({"config": config, "rows": rows}, f, indent=2)
        print(f"  [saved {json_path} with {len(rows)} rows]\n")

    write_summary(args.output_dir, config, rows)
    make_plot(args.output_dir, rows, args.cal_sizes)
    print("Done.")


def _best_per_method(rows, cal, methods):
    """For one cal_size, pick best lambda per method (min sz with cov>=0.89;
    if none valid, min sz overall). FCP has only lambda=0."""
    out = {}
    for method in methods:
        cand = [r for r in rows if r["cal"] == cal and r["method"] == method]
        if not cand:
            continue
        valid = [r for r in cand if r["cov"] >= 0.89]
        pick = min(valid or cand, key=lambda r: r["sz"])
        out[method] = pick
    return out


def write_summary(output_dir, config, rows):
    methods = ["FCP", "MA-CS", "MS-CS-centroid", "MS-CS-cluster"]
    cals = sorted(set(r["cal"] for r in rows))
    lines = []
    lines.append("MS-CS vs MA-CS  (CIFAR-100, DINOv2-518, OPTIMAL EXCHANGEABLE PIPELINE)")
    lines.append(f"NCM={config['ncm']}  PCA={config['pca_dim']}  "
                 f"whiten={config['whiten']}(n={config['n_clusters_whiten']}, pool-fit)  "
                 f"split={config['split']}")
    lines.append(f"alpha={config['alpha']}  n_trials={config['n_trials']}  "
                 f"test_size={config['test_size']}  "
                 f"MS-CS(n_clusters={config['mscs_n_clusters']}, "
                 f"tau={config['mscs_tau_mult']}*median_d^2)")
    lines.append("")
    lines.append("Best lambda per method (min set size with coverage >= 0.89):")
    lines.append("")
    header = f"  {'cal':>4s}  " + "  ".join(f"{m:>16s}" for m in methods)
    lines.append(header)
    lines.append(f"  {'':>4s}  " + "  ".join(f"{'cov / sz (lam)':>16s}" for _ in methods))
    lines.append("  " + "-" * (len(header)))
    for cal in cals:
        best = _best_per_method(rows, cal, methods)
        cells = []
        for m in methods:
            if m in best:
                r = best[m]
                cells.append(f"{r['cov']:.3f}/{r['sz']:.2f}({r['lam']:.2f})".rjust(16))
            else:
                cells.append(" " * 16)
        lines.append(f"  {cal:>4d}  " + "  ".join(cells))
    lines.append("")

    lines.append("Set-size reduction vs plain FCP (best valid lambda):")
    lines.append("")
    head2 = f"  {'cal':>4s}  " + "  ".join(f"{m:>16s}" for m in methods[1:])
    lines.append(head2)
    lines.append("  " + "-" * len(head2))
    for cal in cals:
        best = _best_per_method(rows, cal, methods)
        fcp_sz = best.get("FCP", {}).get("sz")
        cells = []
        for m in methods[1:]:
            if m in best and fcp_sz:
                red = 100.0 * (fcp_sz - best[m]["sz"]) / fcp_sz
                cells.append(f"{red:+5.1f}%".rjust(16))
            else:
                cells.append(" " * 16)
        lines.append(f"  {cal:>4d}  " + "  ".join(cells))
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Optimal pipeline: PCA + cluster-whiten fit on the unlabeled pool,")
    lines.append("    unwhitened NCM, random split -> exactly exchangeable, tight ~0.90.")
    lines.append("  - MA-CS requires CIFAR-100 superclass labels (PCA leaves its M unchanged).")
    lines.append("  - MS-CS-centroid uses NO side information (cal centroids only).")
    lines.append("  - MS-CS-cluster uses the transformed external unlabeled pool (no labels).")
    lines.append("  - All methods: identical splits, transform, NCM, yhat rule, exchangeable.")

    text = "\n".join(lines)
    path = os.path.join(output_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print("\n" + text + "\n")
    print(f"[saved {path}]")


def make_plot(output_dir, rows, cal_sizes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped: {e}]")
        return

    methods = ["FCP", "MA-CS", "MS-CS-centroid", "MS-CS-cluster"]
    colors = {"FCP": "k", "MA-CS": "tab:red",
              "MS-CS-centroid": "tab:green", "MS-CS-cluster": "tab:blue"}
    markers = {"FCP": "o", "MA-CS": "s",
               "MS-CS-centroid": "^", "MS-CS-cluster": "D"}
    cals = sorted(set(r["cal"] for r in rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in methods:
        szs, covs = [], []
        for cal in cals:
            best = _best_per_method(rows, cal, [m]).get(m)
            szs.append(best["sz"] if best else np.nan)
            covs.append(best["cov"] if best else np.nan)
        ax1.plot(cals, szs, marker=markers[m], color=colors[m], label=m)
        ax2.plot(cals, covs, marker=markers[m], color=colors[m], label=m)

    ax1.set_xlabel("calibration size"); ax1.set_ylabel("avg set size")
    ax1.set_title("Set size (best lambda per method)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.axhline(0.9, ls="--", color="gray", lw=1, label="target 0.90")
    ax2.set_xlabel("calibration size"); ax2.set_ylabel("marginal coverage")
    ax2.set_title("Coverage"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("MS-CS vs MA-CS - CIFAR-100, optimal exchangeable pipeline "
                 "(PCA-128 + cluster-whiten, unwhitened NCM)")
    fig.tight_layout()
    path = os.path.join(output_dir, "mscs_vs_macs.png")
    fig.savefig(path, dpi=130)
    print(f"[saved {path}]")


if __name__ == "__main__":
    main()
