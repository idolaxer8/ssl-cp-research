"""
D1 score simplification -- is the softmax in the champion NCM redundant?
(Goal 1, week 08-17; docs/weekly_summary.md "Goals -- week 08-17 to 08-21".)

Theorem D1 (docs/dwt_denoise_theorem.md) is stated for a PLAIN raw-similarity
score s_c(x) = -cos(mu_c, x): one class at a time, no cross-class coupling. The
DEPLOYED champion NCM is `prototype_softmax`: the SAME cosine-to-prototype logits
f_c = cos(mu_c, x), but pushed through a softmax and scored by LAC 1 - p_c(x).
The softmax normalizer couples every class into each score -- exactly the part
D1 does NOT model. This script runs the clean ablation:

    prototype_softmax   f_c -> 1 - softmax_c(f / T)[y]      (champion; theory-blind
                                                             normalizer)
    prototype_cosine    f_c -> 1 - cos(mu_y, x)             (softmax removed; the
                                                             D1 score verbatim)

Same prototypes, same exact leave-one-out logits, same exchangeable pool
transform, SAME cal/test splits per trial (paired) -- the ONLY difference is the
softmax. Because softmax at fixed T is not a monotone per-class map (its
denominator couples all logits), the two give genuinely different sets: this is a
real ablation, not a reparam.

Verdict axis: if `prototype_cosine` matches (or beats) `prototype_softmax` on set
size / CovGap / correct-singleton rate across the grid, the softmax is redundant
and the theory covers the deployed score verbatim (drop it). If it loses, the
normalizer is load-bearing and we quantify the gap + where it lives.

Grid: cifar100 / miniimagenet / aircraft / stanford_cars / cub200, cal 200-800,
balanced_both (lit few-shot protocol) + random (exact-validity arm). Metrics per
(dataset, split, ncm, cal): marginal coverage, avg set size, pooled CovGap,
worst-class coverage, frac under-covered, singleton rate, correct-singleton rate,
predict runtime.

Pipeline = exchangeable (CLAUDE.md default): PCA-128 + cluster-whiten fit on the
UNLABELED pool. prototype_softmax gets ONE pilot-fixed T per dataset (held across
trials -> exact); prototype_cosine has NO temperature at all (strictly
exchangeable, no O(1/n) term).

Examples
--------
# Full grid, GPU, both arms, plots (cluster or 4GB-local for balanced):
python src/d1_softmax_ablation.py --plot

# Balanced only, three datasets, quick:
python src/d1_softmax_ablation.py --datasets cifar100 aircraft --splits balanced_both \
    --n_trials 10 --plot
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
                                   PrototypeSoftmaxNCM)
from exchangeable_features import make_transform

# both arms have a bit-exact vectorized GPU path (prototype_cosine shares the
# prototype_softmax GPU routine with a softmax-free scoring branch).
EMB = "output/from_cluster/embeddings"


# ---------------------------------------------------------------- data loaders
def load_dataset(ds, data_dir):
    """Return (X, y, Xu) float32/int arrays, or None if embeddings are missing.

    * cifar100 / miniimagenet / aircraft : embeddings_<ds>.pt + _unlabeled.pt.
    * stanford_cars : the intermediate-layer files (key 'final'); labeled +
      dedicated unlabeled pool.
    * cub200 : single all-in-one file (no separate pool) -> the FULL set, labels
      dropped, is the exchangeable transductive pool (fitting PCA+whiten on a
      fixed label-free superset of cal+test is exactly exchangeable; see the
      pool-source-transductive finding).
    """
    if ds == "stanford_cars":
        lab = os.path.join(data_dir, "embeddings_stanford_cars_layers.pt")
        unl = os.path.join(data_dir, "embeddings_stanford_cars_unlabeled_layers.pt")
        if not os.path.exists(lab):
            return None
        dl = torch.load(lab, map_location="cpu", weights_only=False)
        X = dl["final"].numpy(); y = dl["labels"].numpy()
        Xu = (torch.load(unl, map_location="cpu", weights_only=False)["final"].numpy()
              if os.path.exists(unl) else None)
        return X, y, Xu
    if ds == "cub200":
        p = "output/embeddings_cub200_all.pt"
        if not os.path.exists(p):
            return None
        dl = torch.load(p, map_location="cpu", weights_only=False)
        X = dl["embeddings"].numpy(); y = dl["labels"].numpy()
        return X, y, X.copy()          # full set (label-free) = transductive pool
    lab = os.path.join(data_dir, f"embeddings_{ds}.pt")
    unl = os.path.join(data_dir, f"embeddings_{ds}_unlabeled.pt")
    if not os.path.exists(lab):
        return None
    dl = torch.load(lab, map_location="cpu", weights_only=False)
    X = dl["embeddings"].numpy(); y = dl["labels"].numpy()
    Xu = (torch.load(unl, map_location="cpu", weights_only=False)["embeddings"].numpy()
          if os.path.exists(unl) else None)
    return X, y, Xu


# ------------------------------------------------------------------- splitting
def balanced_split(y, allc, m_cal, m_test, rng):
    ci, ti = [], []
    for c in allc:
        perm = rng.permutation(np.where(y == c)[0])
        ci.append(perm[:m_cal])
        ti.append(perm[m_cal:m_cal + m_test])
    return np.concatenate(ci), np.concatenate(ti)


def random_split(y, cal, test_total, rng):
    idx = rng.permutation(len(y))
    return idx[:cal], idx[cal:cal + test_total]


# -------------------------------------------------------------- softmax T pilot
def pilot_T(transform, X, y, allc, cal_max, logit, seed):
    """ONE pilot-fixed T for prototype_softmax per dataset (constant across trials
    -> exactly exchangeable). Piloted from a stable larger-m balanced draw."""
    K = len(allc)
    m = max(4, min(cal_max, 800) // K)
    rng = np.random.default_rng(seed)
    ci, _ = balanced_split(y, allc, m, 1, rng)
    Zc = transform.transform(X[ci])
    p = PrototypeSoftmaxNCM(temperature=None, logit=logit,
                            allow_nonexchangeable=True).fit(Zc, y[ci])
    return float(p._T)


# ----------------------------------------------------------------- metric block
def metrics_for_sets(psets, yt, allc, cls_to_j, pooled_cov, pooled_tot, alpha):
    covered = np.array([yt[i] in psets[i] for i in range(len(yt))])
    sizes = np.array([len(s) for s in psets])
    singleton = sizes == 1
    correct_singleton = singleton & covered
    for c in allc:
        msk = yt == c
        if msk.any():
            j = cls_to_j[int(c)]
            pooled_cov[j] += float(covered[msk].sum())
            pooled_tot[j] += int(msk.sum())
    return {"cov": float(covered.mean()), "sz": float(sizes.mean()),
            "singleton_rate": float(singleton.mean()),
            "correct_singleton_rate": float(correct_singleton.mean())}


def run_dataset(ds, args):
    loaded = load_dataset(ds, args.data_dir)
    if loaded is None:
        print(f"[skip] {ds}: embeddings not found")
        return None
    X, y, Xu = loaded
    allc = np.unique(y); K = len(allc)
    cls_to_j = {int(c): j for j, c in enumerate(allc)}
    print(f"\n=== {ds}: X={X.shape}, K={K}, "
          f"pool={None if Xu is None else Xu.shape} ===")

    whiten = None if args.whiten == "none" else args.whiten
    transform = (make_transform(Xu, pca_dim=args.pca_dim, whiten=whiten,
                                n_clusters=args.n_clusters_whiten)
                 if Xu is not None else make_transform(None))
    print(f"  transform: {transform}")

    T_soft = pilot_T(transform, X, y, allc, max(args.cal_sizes),
                     args.logit, args.seed)
    print(f"  prototype_softmax fixed T = {T_soft:.4f}")

    ncm_specs = [("prototype_softmax", dict(logit=args.logit, temperature=T_soft)),
                 ("prototype_cosine",  dict(logit=args.logit))]
    test_total = args.test_per_class * K
    rows = []
    for split in args.splits:
        for cal in args.cal_sizes:
            m_cal = cal // K
            if split == "balanced_both" and m_cal < 2:
                print(f"  [skip {split} cal={cal}] m_cal={m_cal} < 2 (need cal >= {2*K})")
                continue
            # paired accumulators: both NCMs see identical splits each trial
            acc = {nm: {"cov": [], "sz": [], "sr": [], "csr": [], "cg": [], "t": [],
                        "pc": np.zeros(K), "pt": np.zeros(K), "fb": 0}
                   for nm, _ in ncm_specs}
            for t in range(args.n_trials):
                rng = np.random.default_rng(args.seed + 1000 * t)
                if split == "balanced_both":
                    ci, ti = balanced_split(y, allc, m_cal, args.test_per_class, rng)
                else:
                    ci, ti = random_split(y, cal, test_total, rng)
                Xc, yc = transform.transform(X[ci]), y[ci]
                Xt, yt = transform.transform(X[ti]), y[ti]
                for nm, kw in ncm_specs:
                    ncm = create_ncm(nm, **kw)
                    cp = FullConformalPredictor(ncm, alpha=args.alpha)
                    cp.calibrate(Xc, yc, all_classes=allc)
                    t0 = time.perf_counter()
                    try:
                        res = cp.predict(Xt, verbose=False, device=args.device)
                    except (RuntimeError, ValueError):
                        res = cp.predict(Xt, verbose=False, device="cpu")
                        acc[nm]["fb"] += 1
                    rt = time.perf_counter() - t0
                    m = metrics_for_sets(res["prediction_sets"], yt, allc, cls_to_j,
                                         acc[nm]["pc"], acc[nm]["pt"], args.alpha)
                    acc[nm]["cov"].append(m["cov"]); acc[nm]["sz"].append(m["sz"])
                    acc[nm]["sr"].append(m["singleton_rate"])
                    acc[nm]["csr"].append(m["correct_singleton_rate"])
                    acc[nm]["t"].append(rt)
            target = 1 - args.alpha
            for nm, _ in ncm_specs:
                a = acc[nm]
                valid = a["pt"] > 0
                pcov = a["pc"][valid] / a["pt"][valid]
                used = ("cuda(+%dcpu)" % a["fb"]) if a["fb"] and args.device == "cuda" \
                    else args.device
                row = {"dataset": ds, "split": split, "ncm": nm, "cal": cal,
                       "device": used, "n_trials": args.n_trials,
                       "cov": float(np.mean(a["cov"])), "cov_sd": float(np.std(a["cov"])),
                       "sz": float(np.mean(a["sz"])), "sz_sd": float(np.std(a["sz"])),
                       "covgap": float(100 * np.mean(np.abs(pcov - target))),
                       "worst_cov": float(pcov.min()),
                       "frac_undercovered": float(np.mean(pcov < target)),
                       "singleton_rate": float(np.mean(a["sr"])),
                       "correct_singleton_rate": float(np.mean(a["csr"])),
                       "runtime_s": float(np.mean(a["t"]))}
                rows.append(row)
                print(f"  [{split:13s}] cal={cal:4d} {nm:18s} "
                      f"cov={row['cov']:.4f} sz={row['sz']:7.3f} "
                      f"covgap={row['covgap']:5.2f}pp csr={row['correct_singleton_rate']:.3f} "
                      f"[{used}]")
            # paired size delta (cosine relative to softmax), same splits
            sm = next(r for r in rows if r["ncm"] == "prototype_softmax"
                      and r["cal"] == cal and r["split"] == split)
            co = next(r for r in rows if r["ncm"] == "prototype_cosine"
                      and r["cal"] == cal and r["split"] == split)
            d = 100.0 * (co["sz"] - sm["sz"]) / max(sm["sz"], 1e-9)
            print(f"      -> cosine vs softmax set size: {d:+.1f}%  "
                  f"(covgap {co['covgap']-sm['covgap']:+.2f}pp, "
                  f"csr {co['correct_singleton_rate']-sm['correct_singleton_rate']:+.3f})")
    return rows


def plot_all(all_rows, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [d for d in args.datasets if any(r["dataset"] == d for r in all_rows)]
    for split in args.splits:
        srows = [r for r in all_rows if r["split"] == split]
        if not srows:
            continue
        panels = [("sz", "avg set size (lower=better)"),
                  ("cov", "marginal coverage"),
                  ("covgap", "CovGap (pp, lower=better)"),
                  ("correct_singleton_rate", "correct-singleton rate (higher=better)")]
        ncols = len(datasets)
        fig, axes = plt.subplots(len(panels), ncols,
                                 figsize=(3.6 * ncols, 3.1 * len(panels)),
                                 squeeze=False)
        colors = {"prototype_softmax": "tab:blue", "prototype_cosine": "tab:orange"}
        for cj, ds in enumerate(datasets):
            drows = [r for r in srows if r["dataset"] == ds]
            for pi, (key, title) in enumerate(panels):
                ax = axes[pi][cj]
                for nm in ["prototype_softmax", "prototype_cosine"]:
                    pts = sorted((r["cal"], r[key]) for r in drows if r["ncm"] == nm)
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                                marker="o", lw=2, color=colors[nm],
                                label=nm.replace("prototype_", ""))
                ax.grid(alpha=0.3)
                if key == "cov":
                    ax.axhline(1 - args.alpha, color="k", ls=":", lw=1)
                if pi == 0:
                    ax.set_title(ds, fontsize=11)
                if cj == 0:
                    ax.set_ylabel(title, fontsize=9)
                if pi == len(panels) - 1:
                    ax.set_xlabel("cal size")
                if pi == 0 and cj == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(f"D1 ablation: softmax vs plain 1-cos prototype NCM  [{split}]  "
                     f"(alpha={args.alpha}, {args.n_trials} trials, "
                     f"PCA-{args.pca_dim}+{args.whiten}-whiten, logit={args.logit})",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(args.output_dir, f"d1_ablation_{split}.png")
        fig.savefig(out, dpi=150)
        print(f"saved plot -> {out}")


def main():
    ap = argparse.ArgumentParser(description="D1 softmax-vs-plain-cosine prototype "
                                             "NCM ablation under Full CP.")
    ap.add_argument("--datasets", nargs="+",
                    default=["cifar100", "miniimagenet", "aircraft",
                             "stanford_cars", "cub200"])
    ap.add_argument("--data_dir", default=EMB)
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--test_per_class", type=int, default=10)
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--whiten", default="cluster", choices=["cluster", "global", "none"])
    ap.add_argument("--n_clusters_whiten", type=int, default=100)
    ap.add_argument("--splits", nargs="+", default=["balanced_both"],
                    choices=["balanced_both", "random"])
    ap.add_argument("--logit", default="cosine", choices=["cosine", "dot"])
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--output_dir", default="output/d1_softmax_ablation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu")
        args.device = "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"datasets={args.datasets} splits={args.splits} cal={args.cal_sizes} "
          f"trials={args.n_trials} device={args.device}")

    all_rows = []
    for ds in args.datasets:
        rows = run_dataset(ds, args)
        if not rows:
            continue
        all_rows += rows
        with open(os.path.join(args.output_dir, f"results_{ds}.json"), "w") as fjson:
            json.dump({"dataset": ds, "config": vars(args), "rows": rows},
                      fjson, indent=2)
    if all_rows:
        with open(os.path.join(args.output_dir, "results_all.json"), "w") as fjson:
            json.dump({"config": vars(args), "rows": all_rows}, fjson, indent=2)
        if args.plot:
            try:
                plot_all(all_rows, args)
            except Exception as e:
                print(f"[plot skipped: {e}]")
    print(f"\nDone -> {args.output_dir}/results_*.json")


if __name__ == "__main__":
    main()
