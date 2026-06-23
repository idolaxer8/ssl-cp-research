"""
CIFAR-100 ablation: prototype_softmax (the FCA-inspired NCM, now with the fast
denominator-swap GPU path) vs the geodesic MEAN NCM, factored over two levers:

    PCA   : with PCA-128 vs without (full 768-D)   -- whitening (cluster) held on
    MS-CS : without penalty (plain FCP) vs with the exchangeable cluster penalty

=> 2 NCMs x 2 (PCA) x 2 (MS-CS) = 8 arms, swept over cal size, balanced cal+test.

Goal: see how much PCA and the MS-CS penalty each buy for the winning prototype,
against the geodesic-mean baseline, on the discriminating benchmark (CIFAR-100).

Pipeline (exchangeable): the feature transform (optional PCA + cluster-whiten) is
fit on the UNLABELED pool; the MS-CS cluster similarity matrix M is k-means on the
SAME transformed pool, updated per candidate (exchangeable=True) so validity is
exact. prototype_softmax uses a pilot-FIXED temperature (re-resolved per PCA arm,
since the feature scale changes); the geodesic NCM has no temperature.

DEVICE / RUNTIME:
  * No-MS-CS arms: FullConformalPredictor.predict on GPU for both NCMs (fast).
  * MS-CS arms: the geodesic NCM uses the GPU MS-CS path (fast); prototype_softmax
    has no GPU MS-CS kernel yet, so its MS-CS arm runs the CPU loop and is the
    runtime BOTTLENECK (minutes/trial at large cal). Start with modest
    --cal_sizes / --n_trials, scale up once you see the cost. (No-MS-CS prototype
    is fast on GPU via the denominator-swap path.)

Embeddings: output/embeddings_cifar100.pt (+ _unlabeled.pt). On the cluster the
default --data_dir is output/.

Examples
--------
# Cluster, full ablation:
python src/fca_ablation_cluster_experiment.py --device cuda --plot

# Local smoke:
python src/fca_ablation_cluster_experiment.py --data_dir output/from_cluster \
    --cal_sizes 200 400 --n_trials 3 --device cuda --plot
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import FullConformalPredictor, create_ncm, PrototypeSoftmaxNCM
from exchangeable_features import make_transform
from mscs_unlabeled_experiment import (build_cluster_similarity_matrix,
                                       run_fcp_with_mscs)

NCMS = ["prototype_softmax", "unwhitened_topk_mean"]   # prototype vs geodesic mean


def balanced_split(y, allc, m_cal, m_test, rng):
    ci, ti = [], []
    for c in allc:
        perm = rng.permutation(np.where(y == c)[0])
        ci.append(perm[:m_cal])
        ti.append(perm[m_cal:m_cal + m_test])
    return np.concatenate(ci), np.concatenate(ti)


def load_dataset(data_dir, ds):
    lab = os.path.join(data_dir, f"embeddings_{ds}.pt")
    unl = os.path.join(data_dir, f"embeddings_{ds}_unlabeled.pt")
    if not os.path.exists(lab):
        return None
    d = torch.load(lab, map_location="cpu", weights_only=False)
    X, y = d["embeddings"].numpy(), d["labels"].numpy()
    Xu = None
    if os.path.exists(unl):
        Xu = torch.load(unl, map_location="cpu", weights_only=False)["embeddings"].numpy()
    return X, y, Xu


def resolve_proto_T(transform, X, y, allc, args):
    """One pilot-fixed prototype temperature per PCA arm (the feature scale shifts
    with PCA, so T must be re-resolved). Constant across trials -> exchangeable."""
    K = len(allc)
    m = max(4, min(max(args.cal_sizes), 800) // K)
    rng = np.random.default_rng(args.seed)
    ci, _ = balanced_split(y, allc, m, 1, rng)
    return float(PrototypeSoftmaxNCM(temperature=None, logit=args.logit,
                                     allow_nonexchangeable=True
                                     ).fit(transform.transform(X[ci]), y[ci])._T)


def metrics_from_sets(sets, yt, allc, cls_to_j, pooled_cov, pooled_tot, alpha):
    covered = np.array([yt[i] in sets[i] for i in range(len(yt))])
    for c in allc:
        msk = yt == c
        if msk.any():
            j = cls_to_j[int(c)]
            pooled_cov[j] += float(covered[msk].sum())
            pooled_tot[j] += int(msk.sum())
    return float(covered.mean()), float(np.mean([len(s) for s in sets]))


def run(ds, args):
    loaded = load_dataset(args.data_dir, ds)
    if loaded is None:
        print(f"[skip] {ds}: no embeddings_{ds}.pt in {args.data_dir}/")
        return None
    X, y, Xu = loaded
    allc = np.unique(y)
    K = len(allc)
    cls_to_j = {int(c): j for j, c in enumerate(allc)}
    if Xu is None:
        print(f"[skip] {ds}: needs an unlabeled pool for the PCA/whiten transform "
              f"+ MS-CS clusters."); return None
    print(f"\n=== {ds}: X={X.shape}, K={K}, unlabeled={Xu.shape} ===")
    tau_arg = -args.tau   # negative => normalize by median squared inter-cluster dist

    rows = []
    for ncm in args.ncms:
        is_proto = (ncm == "prototype_softmax")
        for pca_dim in ([args.pca_dim, None] if args.pca else [args.pca_dim]):
            pca_on = pca_dim is not None
            transform = make_transform(Xu, pca_dim=pca_dim, whiten=args.whiten,
                                       n_clusters=args.n_clusters_whiten)
            proto_T = resolve_proto_T(transform, X, y, allc, args) if is_proto else None
            tag_t = f"  [{ncm} | PCA={'on' if pca_on else 'off'}]"
            print(f"{tag_t}  transform={transform}"
                  + (f"  T={proto_T:.4f}" if is_proto else ""))
            for mscs in [False, True]:
                for cal in args.cal_sizes:
                    m_cal = cal // K
                    if m_cal < 2:
                        continue
                    covs, szs, ts = [], [], []
                    pooled_cov, pooled_tot = np.zeros(K), np.zeros(K)
                    for t in range(args.n_trials):
                        rng = np.random.default_rng(args.seed + 1000 * t)
                        ci, ti = balanced_split(y, allc, m_cal, args.test_per_class, rng)
                        Xc, yc = transform.transform(X[ci]), y[ci]
                        Xt, yt = transform.transform(X[ti]), y[ti]
                        if not mscs:
                            obj = (create_ncm(ncm, temperature=proto_T, logit=args.logit)
                                   if is_proto else create_ncm(ncm, k=5))
                            cp = FullConformalPredictor(obj, alpha=args.alpha)
                            cp.calibrate(Xc, yc, all_classes=allc)
                            t0 = time.perf_counter()
                            sets = cp.predict(Xt, verbose=False, device=args.device
                                              )["prediction_sets"]
                            ts.append(time.perf_counter() - t0)
                        else:
                            (M, c2c, eff_tau, _med, cls_cen, cls_cnt, clu_cen, clu_d
                             ) = build_cluster_similarity_matrix(
                                transform.Xu_transformed_, Xc, yc, allc,
                                args.n_clusters_mscs, tau=tau_arg)
                            # prototype: no GPU MS-CS kernel -> CPU loop, 1-NN yhat
                            dev = "cpu" if is_proto else args.device
                            yhm = "1nn" if is_proto else "ncm"
                            t0 = time.perf_counter()
                            _c, _s, sets = run_fcp_with_mscs(
                                Xc, yc, Xt, yt, allc, ncm, args.alpha, args.lam, M,
                                exchangeable=True, yhat_mode=yhm,
                                class_centroids=cls_cen, class_counts=cls_cnt,
                                class_to_cluster=c2c, cluster_centroids=clu_cen,
                                cluster_dists=clu_d, effective_tau=eff_tau,
                                device=dev, return_sets=True,
                                temperature=proto_T, logit=args.logit)
                            ts.append(time.perf_counter() - t0)
                        cov, sz = metrics_from_sets(sets, yt, allc, cls_to_j,
                                                    pooled_cov, pooled_tot, args.alpha)
                        covs.append(cov); szs.append(sz)
                    valid = pooled_tot > 0
                    pcov = pooled_cov[valid] / pooled_tot[valid]
                    row = {"dataset": ds, "ncm": ncm,
                           "pca": "on" if pca_on else "off",
                           "mscs": "on" if mscs else "off", "cal": cal,
                           "cov": float(np.mean(covs)), "cov_sd": float(np.std(covs)),
                           "sz": float(np.mean(szs)), "sz_sd": float(np.std(szs)),
                           "covgap": float(100 * np.mean(np.abs(pcov - (1 - args.alpha)))),
                           "worst_cov": float(pcov.min()),
                           "runtime_s": float(np.mean(ts)), "n_trials": args.n_trials}
                    rows.append(row)
                    print(f"    PCA={row['pca']:3s} MS-CS={row['mscs']:3s} cal={cal:4d} "
                          f"{ncm:20s} cov={row['cov']:.4f} sz={row['sz']:6.2f} "
                          f"covgap={row['covgap']:5.2f}pp rt={row['runtime_s']:6.1f}s")
    return rows


def plot(ds, rows, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {("off", "off"): ("--", "o"), ("on", "off"): ("-", "o"),
              ("off", "on"): ("--", "s"), ("on", "on"): ("-", "s")}
    fig, axes = plt.subplots(len(args.ncms), 2, figsize=(12, 5 * len(args.ncms)),
                             squeeze=False)
    for r, ncm in enumerate(args.ncms):
        for col, (key, title) in enumerate([("sz", "avg set size (lower=better)"),
                                            ("cov", "marginal coverage")]):
            ax = axes[r][col]
            for (pca, mscs), (ls, mk) in styles.items():
                pts = sorted((d["cal"], d[key]) for d in rows
                             if d["ncm"] == ncm and d["pca"] == pca and d["mscs"] == mscs)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker=mk,
                            lw=2, label=f"PCA={pca}, MS-CS={mscs}")
            ax.set_title(f"{ncm}: {title}"); ax.set_xlabel("calibration size")
            ax.grid(alpha=0.3)
            if key == "cov":
                ax.axhline(1 - args.alpha, color="k", ls=":", lw=1)
            ax.legend(fontsize=8)
    fig.suptitle(f"{ds}: prototype vs geodesic-mean -- PCA x MS-CS ablation "
                 f"(balanced, {args.n_trials} trials, alpha={args.alpha})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.output_dir, f"ablation_{ds}.png")
    fig.savefig(out, dpi=150); print(f"  saved plot -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="CIFAR-100 PCA x MS-CS ablation: prototype_softmax vs geodesic mean")
    ap.add_argument("--datasets", nargs="+", default=["cifar100"])
    ap.add_argument("--data_dir", default="output")
    ap.add_argument("--ncms", nargs="+", default=NCMS)
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--test_per_class", type=int, default=10)
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pca", type=lambda s: s.lower() != "false", default=True,
                    help="if true (default) ablate PCA on AND off; if false only --pca_dim.")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--whiten", default="cluster", choices=["cluster", "global", "none"],
                    help="held constant across the PCA on/off arms.")
    ap.add_argument("--n_clusters_whiten", type=int, default=100)
    ap.add_argument("--lam", type=float, default=0.05, help="MS-CS penalty strength.")
    ap.add_argument("--n_clusters_mscs", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.5,
                    help="MS-CS kernel temp multiplier of median_d^2.")
    ap.add_argument("--logit", default="cosine", choices=["cosine", "dot"])
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--output_dir", default="output/fca_ablation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu"); args.device = "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"NCMs={args.ncms} | PCA ablate={args.pca} | cal={args.cal_sizes} | "
          f"trials={args.n_trials} | lam={args.lam} | device={args.device}")

    all_rows = []
    for ds in args.datasets:
        rows = run(ds, args)
        if not rows:
            continue
        all_rows += rows
        with open(os.path.join(args.output_dir, f"results_{ds}.json"), "w") as fh:
            json.dump({"dataset": ds, "config": vars(args), "rows": rows}, fh, indent=2)
        if args.plot:
            try:
                plot(ds, rows, args)
            except Exception as e:
                print(f"  [plot skipped: {e}]")
    if all_rows:
        with open(os.path.join(args.output_dir, "results_all.json"), "w") as fh:
            json.dump({"config": vars(args), "rows": all_rows}, fh, indent=2)
    print(f"\nDone. Saved -> {args.output_dir}/results_*.json")


if __name__ == "__main__":
    main()
