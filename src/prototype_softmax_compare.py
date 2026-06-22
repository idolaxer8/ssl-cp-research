"""
3-rung ablation: the FCA line translated into our pure-SSL Full-CP setting.

    rung 3  prototype_softmax   -- VANILLA text-free FCA: class-mean prototype ->
                                   softmax LAC. NO covariance term. The faithful
                                   translation + the clean control.
    rung 4  ridge_softmax       -- FCA + an added ridge covariance term (cal-fit
                                   whitening, refit per candidate). The "further
                                   improvement" under scrutiny.
    ours    unwhitened_topk_asym -- our exchangeable geodesic NN-ratio NCM.

The point of the trio: isolate whether ridge_softmax's edge is "the FCA idea"
(a refittable softmax head, already captured by prototype_softmax) or just the
extra discriminative whitening (the covariance term). prototype_softmax is the
missing rung that answers this.

Per (dataset, split, NCM, cal) reports: marginal coverage, avg set size, pooled
CovGap (Ding et al. 2023), worst-class coverage, predict runtime.

Splits (both reported):
  * balanced_both -- balanced cal (cal//K per class) + balanced test, disjoint,
                     fresh each trial. Lit-comparable few-shot protocol;
                     conservative (over-covers ~1-3pp). Under this split the
                     softmax NCMs BLOAT at small cal but do NOT under-cover.
  * random        -- uniform random cal/test; the exact-validity arm (tight ~0.90).
                     At small cal some classes are absent from cal -> structural
                     under-coverage shared by ALL methods (a property of the split).

Pipeline = exchangeable: PCA + cluster-whitening fit on the UNLABELED pool
(exchangeable_features.UnlabeledTransform); the NCM scores transformed features.

Temperature: the two softmax NCMs each get ONE pilot-fixed T per dataset, held
constant across trials -> every trial's CP stays exactly exchangeable (a fixed T
is a hyperparameter; a per-fit auto-T would be cal-fit O(1/n)). prototype_softmax
(cosine logits in [-1,1]) and ridge_softmax (unbounded ridge outputs) live on
different scales, so each is piloted separately.

Local scope: embeddings at output/embeddings_<ds>.pt (+ _unlabeled.pt).

Examples
--------
python src/prototype_softmax_compare.py --datasets cifar100 \
    --cal_sizes 200 400 800 --n_trials 10 --pca_dim 128 --whiten cluster --plot
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import (FullConformalPredictor, create_ncm,
                                   RidgeSoftmaxNCM, PrototypeSoftmaxNCM)
from exchangeable_features import make_transform

GEO_NCMS = {"unwhitened_topk_asym", "unwhitened_topk_mean",
            "geodesic_topk_asym", "geodesic_topk_mean",
            "geodesic_topk", "geodesic_1nn"}
SOFTMAX_NCMS = {"prototype_softmax", "ridge_softmax"}


def balanced_split(y, allc, m_cal, m_test, rng):
    """Balanced cal (m_cal/class) + balanced test (m_test/class), disjoint."""
    ci, ti = [], []
    for c in allc:
        perm = rng.permutation(np.where(y == c)[0])
        ci.append(perm[:m_cal])
        ti.append(perm[m_cal:m_cal + m_test])
    return np.concatenate(ci), np.concatenate(ti)


def random_split(y, cal, test_total, rng):
    """Uniform random cal/test (the exactly-exchangeable arm)."""
    idx = rng.permutation(len(y))
    return idx[:cal], idx[cal:cal + test_total]


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


def resolve_softmax_T(ncm_name, transform, X, y, allc, args):
    """ONE pilot-fixed T per (softmax NCM, dataset), constant across all trials ->
    exactly exchangeable. Piloted from a STABLE larger-m balanced draw (fixed
    seed). prototype_softmax and ridge_softmax are piloted with their own class
    (different logit scales)."""
    K = len(allc)
    m = max(4, min(max(args.cal_sizes), 800) // K)
    rng = np.random.default_rng(args.seed)
    ci, _ = balanced_split(y, allc, m, 1, rng)
    Zc = transform.transform(X[ci])
    if ncm_name == "prototype_softmax":
        pncm = PrototypeSoftmaxNCM(temperature=None, logit=args.logit,
                                   allow_nonexchangeable=True).fit(Zc, y[ci])
    else:  # ridge_softmax
        pncm = RidgeSoftmaxNCM(lam=args.lam_ridge, lam_anchor=args.lam_anchor,
                               temperature=None, loo=True,
                               allow_nonexchangeable=True).fit(Zc, y[ci])
    return float(pncm._T)


def make_ncm(ncm_name, T_map, args):
    if ncm_name == "prototype_softmax":
        return create_ncm(ncm_name, temperature=T_map[ncm_name], logit=args.logit)
    if ncm_name == "ridge_softmax":
        return create_ncm(ncm_name, temperature=T_map[ncm_name],
                          lam_ridge=args.lam_ridge, lam_anchor=args.lam_anchor, loo=True)
    return create_ncm(ncm_name, k=5)


def run_dataset(ds, args):
    loaded = load_dataset(args.data_dir, ds)
    if loaded is None:
        print(f"[skip] {ds}: no embeddings_{ds}.pt in {args.data_dir}/")
        return None
    X, y, Xu = loaded
    allc = np.unique(y)
    K = len(allc)
    cls_to_j = {int(c): j for j, c in enumerate(allc)}
    print(f"\n=== {ds}: X={X.shape}, K={K}, "
          f"unlabeled={None if Xu is None else Xu.shape} ===")

    whiten = None if args.whiten == "none" else args.whiten
    transform = (make_transform(Xu, pca_dim=args.pca_dim, whiten=whiten,
                                n_clusters=args.n_clusters_whiten)
                 if Xu is not None else make_transform(None))
    print(f"  transform: {transform}")

    T_map = {}
    for nm in args.ncms:
        if nm in SOFTMAX_NCMS:
            T_map[nm] = resolve_softmax_T(nm, transform, X, y, allc, args)
            print(f"  {nm} fixed T = {T_map[nm]:.4f}")

    test_total = args.test_per_class * K
    rows = []
    for split in args.splits:
        for cal in args.cal_sizes:
            m_cal = cal // K
            if split == "balanced_both" and m_cal < 2:
                print(f"  [skip {split} cal={cal}] m_cal={m_cal} < 2 (need cal >= 2K={2*K})")
                continue
            for ncm_name in args.ncms:
                dev = "cpu" if ncm_name not in GEO_NCMS else args.device
                covs, szs, ts, gaps_pt = [], [], [], []
                pooled_cov = np.zeros(K)
                pooled_tot = np.zeros(K)
                for t in range(args.n_trials):
                    rng = np.random.default_rng(args.seed + 1000 * t)
                    if split == "balanced_both":
                        ci, ti = balanced_split(y, allc, m_cal, args.test_per_class, rng)
                    else:
                        ci, ti = random_split(y, cal, test_total, rng)
                    Xc, yc = transform.transform(X[ci]), y[ci]
                    Xt, yt = transform.transform(X[ti]), y[ti]
                    ncm = make_ncm(ncm_name, T_map, args)
                    cp = FullConformalPredictor(ncm, alpha=args.alpha)
                    cp.calibrate(Xc, yc, all_classes=allc)
                    t0 = time.perf_counter()
                    res = cp.predict(Xt, verbose=False, device=dev)
                    rt = time.perf_counter() - t0
                    psets = res["prediction_sets"]
                    covered = np.array([yt[i] in psets[i] for i in range(len(yt))])
                    covs.append(float(covered.mean()))
                    szs.append(float(np.mean([len(s) for s in psets])))
                    ts.append(rt)
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
                valid = pooled_tot > 0
                pcov = pooled_cov[valid] / pooled_tot[valid]
                target = 1 - args.alpha
                row = {"dataset": ds, "split": split, "ncm": ncm_name, "cal": cal,
                       "device": dev,
                       "cov": float(np.mean(covs)), "cov_sd": float(np.std(covs)),
                       "sz": float(np.mean(szs)), "sz_sd": float(np.std(szs)),
                       "covgap": float(100 * np.mean(np.abs(pcov - target))),
                       "covgap_pertrial": float(np.mean(gaps_pt)),
                       "worst_cov": float(pcov.min()),
                       "frac_undercovered": float(np.mean(pcov < target)),
                       "runtime_s": float(np.mean(ts)), "n_trials": args.n_trials}
                rows.append(row)
                print(f"  [{split:13s}] cal={cal:4d} {ncm_name:20s} "
                      f"cov={row['cov']:.4f} sz={row['sz']:6.2f} "
                      f"covgap={row['covgap']:5.2f}pp worst={row['worst_cov']:.3f} "
                      f"rt={row['runtime_s']:6.1f}s [{dev}]")
    return rows


def plot_dataset(ds, rows, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [s for s in args.splits if any(r["split"] == s for r in rows)]
    ncms = sorted({r["ncm"] for r in rows})
    panels = [("sz", "avg set size (lower=better)", False),
              ("cov", "marginal coverage", False),
              ("covgap", "CovGap (pp, lower=better)", False),
              ("runtime_s", "predict runtime (s, log)", True)]
    for split in splits:
        srows = [r for r in rows if r["split"] == split]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for ax, (key, title, logy) in zip(axes.ravel(), panels):
            for ncm in ncms:
                pts = sorted((r["cal"], r[key]) for r in srows if r["ncm"] == ncm)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            marker="o", lw=2, label=ncm)
            ax.set_title(title); ax.set_xlabel("calibration size"); ax.grid(alpha=0.3)
            if key == "cov":
                ax.axhline(1 - args.alpha, color="k", ls=":", lw=1, label="target 1-alpha")
            if logy:
                ax.set_yscale("log")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{ds} [{split}]: 3-rung FCA ablation "
                     f"(prototype vs ridge vs geodesic)\n"
                     f"FCP alpha={args.alpha}, {args.n_trials} trials, "
                     f"PCA-{args.pca_dim}+{args.whiten}-whiten, logit={args.logit}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(args.output_dir, f"compare_{ds}_{split}.png")
        fig.savefig(out, dpi=150)
        print(f"  saved plot -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="3-rung FCA ablation: prototype_softmax vs ridge_softmax vs geodesic")
    ap.add_argument("--datasets", nargs="+", default=["cifar100"])
    ap.add_argument("--data_dir", default="output")
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--test_per_class", type=int, default=10)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--whiten", default="cluster", choices=["cluster", "global", "none"])
    ap.add_argument("--n_clusters_whiten", type=int, default=100)
    ap.add_argument("--ncms", nargs="+",
                    default=["prototype_softmax", "ridge_softmax", "unwhitened_topk_asym"])
    ap.add_argument("--splits", nargs="+", default=["balanced_both", "random"],
                    choices=["balanced_both", "random"])
    ap.add_argument("--logit", default="cosine", choices=["cosine", "dot"],
                    help="prototype_softmax logit form.")
    ap.add_argument("--lam_ridge", type=float, default=1.0)
    ap.add_argument("--lam_anchor", type=float, default=1.0)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                    help="geodesic NCMs use this (GPU fast path); softmax NCMs are cpu.")
    ap.add_argument("--output_dir", default="output/prototype_softmax_compare")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but CUDA unavailable -> cpu")
        args.device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"NCMs={args.ncms} | splits={args.splits} | cal={args.cal_sizes} | "
          f"trials={args.n_trials} | test/class={args.test_per_class} | device={args.device}")

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
