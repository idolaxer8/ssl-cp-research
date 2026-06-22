"""
Extensive cluster benchmark: the FCA-inspired `ridge_softmax` NCM vs the geodesic
NCMs (`unwhitened_topk_asym`, `unwhitened_topk_mean` = the exchangeable asym /
symmetric geodesics) under the BALANCED few-shot protocol.

Per (dataset, NCM, cal) it reports: marginal coverage, avg set size, CovGap
(class-conditional, Ding et al. 2023, via conditional_coverage_experiment.
per_class_metrics), worst-class coverage, and PREDICT RUNTIME.

Split: balanced ONLY -- balanced cal (cal//K per class) + a FIXED balanced test
(`--test_per_class` per class), disjoint, freshly sampled each trial. This is the
lit-comparable few-shot protocol; it is conservative (over-covers ~1-3pp), not the
exact guarantee -- the exact-validity (random) arm is established separately
(see exchangeable_fcp_experiment.py --split random). A FIXED test set (decoupled
from cal) is used so the only swept variable is cal.

Pipeline = exchangeable: PCA + cluster-whitening fit on the UNLABELED pool
(exchangeable_features.UnlabeledTransform); the NCM scores the transformed
features (no cal-fit whitening).

RUNTIME / DEVICE NOTE:
  * ALL NCMs here (ridge_softmax AND the geodesics) have a vectorized GPU path --
    pass --device cuda on the cluster. ridge_softmax's GPU path (Sherman-Morrison
    rank-1 + PRESS, batched over test points) matches the CPU loop bit-for-bit and
    is what makes the extensive sweep feasible. It uses float64 for that exact
    parity (RTX 6000 Ada has strong fp64). It requires every candidate class
    present in cal (always true for the balanced split); it silently falls back to
    CPU otherwise.
  * The `runtime_s` metric still records per-NCM predict time; for a strictly fair
    comparison keep all NCMs on the same --device.

KNOWN REGIME CAVEAT (verified on CIFAR-100, K=100):
  * ridge_softmax (a discriminative probe) needs enough samples per class. At
    m = cal//K <~ 4 (e.g. cal=200 at K=100 -> m=2) it UNDER-covers (~0.82) even on
    the conservative balanced split -- the m=2 LOO regime is degenerate (cf.
    [[fcp-missing-class-validity-fix]] Regime C). It becomes valid + tightest at
    cal>=400 (m>=4). The geodesic NN-ratio NCMs stay robust at tiny cal. cal=200
    is kept in the sweep on purpose, to MAP this breakdown -- expect the small-cal
    ridge rows to under-cover. `--lam_anchor` larger (-> nearest-class-mean
    prototype) is the lever to try for the small-cal regime.
  * Trade-off seen at cal>=400: ridge gives the SMALLEST sets but the geodesic
    NCMs give a slightly better (lower) CovGap. Hence the multi-metric comparison.

On the cluster, embeddings live at output/ (the default --data_dir). Expects
output/embeddings_<dataset>.pt and output/embeddings_<dataset>_unlabeled.pt with
keys {"embeddings", "labels"}.

Examples
--------
# cluster, geodesics on GPU, ridge on CPU, full sweep:
python src/ridge_softmax_cluster_experiment.py --device cuda --plot

# fair all-CPU runtime comparison, CIFAR-100 only, quick gauge:
python src/ridge_softmax_cluster_experiment.py --datasets cifar100 \
    --device cpu --n_trials 5 --cal_sizes 200 400 800
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import FullConformalPredictor, create_ncm, RidgeSoftmaxNCM
from exchangeable_features import make_transform

# NCMs that support the GPU fast path (GeodesicTopKMeanNCM family).
GEO_NCMS = {"unwhitened_topk_asym", "unwhitened_topk_mean",
            "geodesic_topk_asym", "geodesic_topk_mean",
            "geodesic_topk", "geodesic_1nn"}


def balanced_split(y, allc, m_cal, m_test, rng):
    """Balanced cal (m_cal/class) + balanced test (m_test/class), disjoint, fresh
    per-class permutation each trial."""
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
    X = d["embeddings"].numpy()
    y = d["labels"].numpy()
    Xu = None
    if os.path.exists(unl):
        Xu = torch.load(unl, map_location="cpu", weights_only=False)["embeddings"].numpy()
    return X, y, Xu


def resolve_ridge_T(transform, X, y, allc, args):
    """ONE pilot-fixed temperature per dataset, held constant across all cal/trials
    -> every trial's CP stays exactly exchangeable (a fixed T from a pilot draw is
    a hyperparameter on separate data). A per-fit auto-T would be cal-fit O(1/n)."""
    if str(args.temperature) != "auto":
        return float(args.temperature)
    K = len(allc)
    # Pilot T from a STABLE larger-m draw (not the smallest cal) -- a T estimated
    # at m=2/class is noisy and far from the optimum; one fixed T is used for all
    # cal sizes, so base it on a well-determined fit.
    m = max(4, min(max(args.cal_sizes), 800) // K)
    rng = np.random.default_rng(args.seed)
    ci, _ = balanced_split(y, allc, m, 1, rng)
    pncm = RidgeSoftmaxNCM(lam=args.lam_ridge, lam_anchor=args.lam_anchor,
                           temperature=None, loo=True,
                           allow_nonexchangeable=True).fit(transform.transform(X[ci]), y[ci])
    return float(pncm._T)


def run_dataset(ds, args):
    loaded = load_dataset(args.data_dir, ds)
    if loaded is None:
        print(f"[skip] {ds}: no embeddings_{ds}.pt in {args.data_dir}/")
        return None
    X, y, Xu = loaded
    allc = np.unique(y)
    K = len(allc)
    print(f"\n=== {ds}: X={X.shape}, K={K}, "
          f"unlabeled={None if Xu is None else Xu.shape} ===")

    whiten = None if args.whiten == "none" else args.whiten
    transform = (make_transform(Xu, pca_dim=args.pca_dim, whiten=whiten,
                                n_clusters=args.n_clusters_whiten)
                 if Xu is not None else make_transform(None))
    print(f"  transform: {transform}")

    ridge_T = resolve_ridge_T(transform, X, y, allc, args)
    print(f"  ridge_softmax fixed T = {ridge_T:.4f}")

    rows = []
    for cal in args.cal_sizes:
        m_cal = cal // K
        if m_cal < 2:
            print(f"  [skip cal={cal}] m_cal={m_cal} < 2 (need cal >= 2K = {2 * K})")
            continue
        for ncm_name in args.ncms:
            # ridge_softmax now has a vectorized GPU path too (it requires every
            # candidate class present in cal -> always true for the balanced split).
            dev = args.device
            cls_to_j = {int(c): j for j, c in enumerate(allc)}
            covs, szs, ts, gaps_pt = [], [], [], []
            pooled_cov = np.zeros(K)   # per-class covered count, accumulated over trials
            pooled_tot = np.zeros(K)   # per-class test count, accumulated over trials
            for t in range(args.n_trials):
                rng = np.random.default_rng(args.seed + 1000 * t)
                ci, ti = balanced_split(y, allc, m_cal, args.test_per_class, rng)
                Xc, yc = transform.transform(X[ci]), y[ci]
                Xt, yt = transform.transform(X[ti]), y[ti]
                if ncm_name == "ridge_softmax":
                    ncm = create_ncm(ncm_name, temperature=ridge_T,
                                     lam_ridge=args.lam_ridge,
                                     lam_anchor=args.lam_anchor, loo=True)
                else:
                    ncm = create_ncm(ncm_name, k=5)
                cp = FullConformalPredictor(ncm, alpha=args.alpha)
                cp.calibrate(Xc, yc, all_classes=allc)
                t0 = time.perf_counter()
                try:
                    res = cp.predict(Xt, verbose=False, device=dev)
                except (RuntimeError, ValueError):
                    res = cp.predict(Xt, verbose=False, device="cpu")  # safety fallback
                rt = time.perf_counter() - t0
                psets = res["prediction_sets"]
                covered = np.array([yt[i] in psets[i] for i in range(len(yt))])
                covs.append(float(covered.mean()))
                szs.append(float(np.mean([len(s) for s in psets])))
                ts.append(rt)
                # per-class accumulation (pooled CovGap) + per-trial CovGap (reference)
                gpt = []
                for c in allc:
                    msk = yt == c
                    if msk.any():
                        j = cls_to_j[int(c)]
                        cc = float(covered[msk].sum())
                        pooled_cov[j] += cc
                        pooled_tot[j] += int(msk.sum())
                        gpt.append(abs(cc / msk.sum() - (1 - args.alpha)))
                gaps_pt.append(100.0 * float(np.mean(gpt)))
            # POOLED CovGap (Ding et al.; stable at small test/class -- memory
            # [[default-split-balanced-cal-and-test]] recommends pooling at small m).
            valid = pooled_tot > 0
            pcov = pooled_cov[valid] / pooled_tot[valid]
            target = 1 - args.alpha
            row = {"dataset": ds, "ncm": ncm_name, "cal": cal, "device": dev,
                   "cov": float(np.mean(covs)), "cov_sd": float(np.std(covs)),
                   "sz": float(np.mean(szs)), "sz_sd": float(np.std(szs)),
                   "covgap": float(100 * np.mean(np.abs(pcov - target))),
                   "covgap_pertrial": float(np.mean(gaps_pt)),
                   "worst_cov": float(pcov.min()),
                   "frac_undercovered": float(np.mean(pcov < target)),
                   "runtime_s": float(np.mean(ts)), "n_trials": args.n_trials}
            rows.append(row)
            print(f"  cal={cal:4d} {ncm_name:22s} cov={row['cov']:.4f} "
                  f"sz={row['sz']:5.2f} covgap={row['covgap']:5.2f}pp "
                  f"worst={row['worst_cov']:.3f} rt={row['runtime_s']:6.1f}s [{dev}]")
    return rows


def plot_dataset(ds, rows, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncms = sorted({r["ncm"] for r in rows})
    panels = [("sz", "avg set size (lower = better)", False),
              ("cov", "marginal coverage", False),
              ("covgap", "CovGap (pp, lower = better)", False),
              ("runtime_s", "predict runtime (s, log)", True)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title, logy) in zip(axes.ravel(), panels):
        for ncm in ncms:
            pts = sorted((r["cal"], r[key]) for r in rows if r["ncm"] == ncm)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=2, label=ncm)
        ax.set_title(title)
        ax.set_xlabel("calibration size")
        ax.grid(alpha=0.3)
        if key == "cov":
            ax.axhline(1 - args.alpha, color="k", ls=":", lw=1, label="target 1-alpha")
        if logy:
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{ds}: ridge_softmax (FCA-inspired) vs geodesic NCMs\n"
                 f"balanced cal+test, FCP alpha={args.alpha}, {args.n_trials} trials, "
                 f"PCA-{args.pca_dim}+{args.whiten}-whiten")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(args.output_dir, f"compare_{ds}.png")
    fig.savefig(out, dpi=150)
    print(f"  saved plot -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Cluster benchmark: ridge_softmax (FCA) vs geodesic NCMs")
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "miniimagenet"])
    ap.add_argument("--data_dir", default="output",
                    help="dir with embeddings_<ds>.pt + _unlabeled.pt (cluster: output)")
    ap.add_argument("--cal_sizes", type=int, nargs="+",
                    default=[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800])
    ap.add_argument("--test_per_class", type=int, default=10,
                    help="balanced test points per class, fixed across cal sizes.")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--whiten", default="cluster", choices=["cluster", "global", "none"])
    ap.add_argument("--n_clusters_whiten", type=int, default=100)
    ap.add_argument("--ncms", nargs="+",
                    default=["ridge_softmax", "unwhitened_topk_asym", "unwhitened_topk_mean"])
    ap.add_argument("--temperature", default="auto",
                    help="ridge_softmax T: a float (exact) or 'auto' (pilot-fixed per dataset).")
    ap.add_argument("--lam_ridge", type=float, default=1.0)
    ap.add_argument("--lam_anchor", type=float, default=1.0)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                    help="geodesic NCMs use this (GPU fast path); ridge_softmax is "
                         "always cpu. For a FAIR runtime comparison use --device cpu.")
    ap.add_argument("--output_dir", default="output/ridge_softmax_cluster")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true", help="also save per-dataset plots.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but CUDA unavailable -> cpu")
        args.device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"NCMs={args.ncms} | cal={args.cal_sizes} | trials={args.n_trials} | "
          f"test/class={args.test_per_class} | device={args.device}")

    all_rows = []
    for ds in args.datasets:
        rows = run_dataset(ds, args)
        if not rows:
            continue
        all_rows += rows
        with open(os.path.join(args.output_dir, f"results_{ds}.json"), "w") as f:
            json.dump({"dataset": ds, "config": vars(args), "rows": rows}, f, indent=2)
        if args.plot:
            try:
                plot_dataset(ds, rows, args)
            except Exception as e:
                print(f"  [plot skipped: {e}]")

    if all_rows:
        with open(os.path.join(args.output_dir, "results_all.json"), "w") as f:
            json.dump({"config": vars(args), "rows": all_rows}, f, indent=2)
    print(f"\nDone. Saved -> {args.output_dir}/results_*.json")


if __name__ == "__main__":
    main()
