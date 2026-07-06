"""
Transform-control experiment: is the PCA step principled or ad hoc?

Pits the production transform (PCA-128 + within-cluster whitening, fit on the
unlabeled pool) against controls that each remove ONE ingredient, so the paper
can attribute the set-size win to the right mechanism:

    raw768         identity (no transform)                 -- baseline
    rp128          Johnson-Lindenstrauss Gaussian proj.    -- mere dimension
                   to d=128 (data-INDEPENDENT)                reduction
    rp128_cw       JL-128 + within-cluster whitening       -- adaptivity of the
                                                              PROJECTION isolated
    pca128         PCA-128, no whitening                   -- rotation+truncation
                                                              alone
    pca128_cw      PCA-128 + within-cluster whitening      -- the production
                                                              recipe
    lw_global768   full-768 ZCA whitening, Ledoit-Wolf     -- continuous shrinkage
                   shrunk GLOBAL covariance (no truncation)   alternative to
    lw_cluster768  full-768 ZCA whitening, LW shrunk          hard truncation
                   WITHIN-CLUSTER covariance

Predictions (from the metric-learning / spectral-regularization framing,
memory `pca-rationalization-audit`):
  - rp128 ~= raw768 (JL preserves distances; reduction per se is not the lever)
  - pca128_cw >> rp128_cw (the projection's data-adaptivity matters)
  - lw_cluster768 ~ pca128_cw would confirm "regularized precision metric" and
    defuse "d'=128 is magic"; lw << pca would justify hard truncation.

Every arm is a fixed map w.r.t. the cal/test bag (pool-fit or data-independent),
so ALL arms are exactly exchangeable -- this is an efficiency comparison only.

Usage (local, from repo root):
python src/transform_control_experiment.py --device cuda \
    --data_dir output/from_cluster --output_dir output/pca_pilots/transform_controls
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import (FullConformalPredictor, create_ncm,
                                  PrototypeSoftmaxNCM)
from exchangeable_features import UnlabeledTransform, IdentityTransform

SOFTMAX_NCMS = {"prototype_softmax"}

# arm -> UnlabeledTransform kwargs (None = IdentityTransform)
ARMS = {
    "raw768":        None,
    "rp128":         dict(pca_dim=128, whiten=None,         projection="random"),
    "rp128_cw":      dict(pca_dim=128, whiten="cluster",    projection="random"),
    "pca128":        dict(pca_dim=128, whiten=None,         projection="pca"),
    "pca128_cw":     dict(pca_dim=128, whiten="cluster",    projection="pca"),
    "pca512_cw":     dict(pca_dim=512, whiten="cluster",    projection="pca"),
    "lw_global768":  dict(pca_dim=None, whiten="lw_global", projection=None),
    "lw_cluster768": dict(pca_dim=None, whiten="lw_cluster", projection=None),
    # signal-in-the-tail probes (2026-07-06): keep the orthogonal complement of
    # the top-16 principal directions (16 ~ aircraft pool PR) / full-rank
    # diagonal cluster whitening
    "tail_r16":      dict(pca_dim=16, whiten=None,          projection="pca_tail"),
    "tailcw_r16":    dict(pca_dim=16, whiten="cluster",     projection="pca_tail"),
    "cw768":         dict(pca_dim=None, whiten="cluster",   projection="center"),
}


def pool_participation_ratio(Xu, n=4000, seed=0):
    """Effective dimensionality of the pool cloud: (sum ev)^2 / sum ev^2.
    Pool-only + label-free, so using it to SELECT the transform is itself a
    fixed (bag-independent) rule -> exchangeability preserved."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xu), min(n, len(Xu)), replace=False)
    Z = Xu[idx] - Xu[idx].mean(axis=0)
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    return float(ev.sum() ** 2 / (ev ** 2).sum())


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


def build_arm_transforms(arm, Xu, args):
    """List of fitted transforms for an arm (rp arms get --rp_repeats seeds,
    averaged over trials; deterministic arms get one)."""
    spec = ARMS[arm]
    if spec is None:
        return [IdentityTransform().fit()]
    if spec.get("projection") == "random":
        return [UnlabeledTransform(n_clusters=args.n_clusters_whiten,
                                   rp_seed=args.seed + 7919 * r, **spec).fit(Xu)
                for r in range(args.rp_repeats)]
    return [UnlabeledTransform(n_clusters=args.n_clusters_whiten, **spec).fit(Xu)]


def resolve_softmax_T(transform, X, y, allc, args):
    """Pilot-fixed prototype-softmax temperature, one per (arm, dataset):
    the cosine-logit distribution differs per feature space, so T must be
    re-piloted per arm. Fixed across trials -> exactly exchangeable."""
    if str(args.proto_temperature) != "auto":
        return float(args.proto_temperature)
    K = len(allc)
    m = max(4, min(max(args.cal_sizes), 800) // K)
    rng = np.random.default_rng(args.seed)
    ci, _ = balanced_split(y, allc, m, 1, rng)
    Zc = transform.transform(X[ci])
    pncm = PrototypeSoftmaxNCM(temperature=None, logit="cosine",
                               allow_nonexchangeable=True).fit(Zc, y[ci])
    return float(pncm._T)


def main():
    ap = argparse.ArgumentParser(description="PCA-vs-controls transform experiment (all arms exchangeable)")
    ap.add_argument("--data_dir", default="output/from_cluster")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS) + ["auto"],
                    help="'auto' = pool-only transform selection: pool "
                         "participation ratio >= --auto_pr_threshold -> "
                         "pca128_cw (signal in the top spectrum), else "
                         "lw_cluster768 (collapsed spectrum, signal in the "
                         "tail -> whiten, don't truncate).")
    ap.add_argument("--auto_pr_threshold", type=float, default=64.0,
                    help="PR switch point for the auto arm (d'/2 by default; "
                         "observed PR: aircraft 16, cifar10 117, cifar100 235, "
                         "mini 246).")
    ap.add_argument("--out_tag", default="",
                    help="suffix for results/plot filenames (sweep runs).")
    ap.add_argument("--ncms", nargs="+",
                    default=["unwhitened_topk_mean", "prototype_softmax"])
    ap.add_argument("--splits", nargs="+", default=["balanced_both", "random"],
                    choices=["balanced_both", "random"])
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--test_per_class", type=int, default=5,
                    help="balanced test shots/class; random split uses this*K total.")
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--rp_repeats", type=int, default=3,
                    help="independent JL matrices to average over (trial t uses "
                         "matrix t %% rp_repeats).")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_clusters_whiten", type=int, default=100)
    ap.add_argument("--random_min_cal_softmax", type=int, default=800,
                    help="skip softmax NCMs on the random split below this cal: "
                         "missing classes force the slow CPU fallback (~40-60s/trial "
                         "vs ~0.5s GPU) only to reproduce the known degenerate "
                         "corner (prototype at random cal<=400 -> sz~K). Set 0 to "
                         "run everything.")
    ap.add_argument("--proto_temperature", default="auto")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="output/pca_pilots/transform_controls")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu")
        args.device = "cpu"

    lab = os.path.join(args.data_dir, f"embeddings_{args.dataset}.pt")
    unl = os.path.join(args.data_dir, f"embeddings_{args.dataset}_unlabeled.pt")
    d = torch.load(lab, map_location="cpu", weights_only=False)
    X, y = d["embeddings"].numpy(), d["labels"].numpy()
    Xu = torch.load(unl, map_location="cpu", weights_only=False)["embeddings"].numpy()
    allc = np.unique(y)
    K = len(allc)
    cls_to_j = {int(c): j for j, c in enumerate(allc)}
    pool_pr = pool_participation_ratio(Xu, seed=args.seed)
    print(f"{args.dataset}: X={X.shape} K={K} pool={Xu.shape} "
          f"PR={pool_pr:.1f} | device={args.device}")

    rows = []
    for arm in args.arms:
        if arm == "auto":
            resolved = ("pca128_cw" if pool_pr >= args.auto_pr_threshold
                        else "lw_cluster768")
            print(f"\nauto arm: pool PR={pool_pr:.1f} vs threshold "
                  f"{args.auto_pr_threshold:g} -> {resolved}")
            spec_arm, arm = resolved, f"auto->{resolved}"
        else:
            spec_arm = arm
        t0 = time.time()
        tfs = build_arm_transforms(spec_arm, Xu, args)
        print(f"\n=== arm {arm}: {tfs[0]}"
              + (f" (x{len(tfs)} JL seeds)" if len(tfs) > 1 else "")
              + f"  [fit {time.time()-t0:.0f}s] ===")
        T_by_ncm = {}
        for nm in args.ncms:
            if nm in SOFTMAX_NCMS:
                T_by_ncm[nm] = resolve_softmax_T(tfs[0], X, y, allc, args)
                print(f"  {nm} fixed T = {T_by_ncm[nm]:.4f}")

        for split in args.splits:
            for cal in args.cal_sizes:
                m_cal = cal // K
                if split == "balanced_both" and m_cal < 2:
                    continue
                for nm in args.ncms:
                    if (split == "random" and nm in SOFTMAX_NCMS
                            and cal < args.random_min_cal_softmax):
                        continue
                    covs, szs = [], []
                    pooled_cov, pooled_tot = np.zeros(K), np.zeros(K)
                    n_fallback = 0
                    t0 = time.time()
                    for t in range(args.n_trials):
                        tf = tfs[t % len(tfs)]
                        rng = np.random.default_rng(args.seed + 1000 * t)
                        if split == "balanced_both":
                            ci, ti = balanced_split(y, allc, m_cal,
                                                    args.test_per_class, rng)
                        else:
                            ci, ti = random_split(y, cal, args.test_per_class * K, rng)
                        Xc, yc = tf.transform(X[ci]), y[ci]
                        Xt, yt = tf.transform(X[ti]), y[ti]
                        if nm in SOFTMAX_NCMS:
                            ncm = create_ncm(nm, temperature=T_by_ncm[nm],
                                             logit="cosine")
                        else:
                            ncm = create_ncm(nm, k=5)
                        cp = FullConformalPredictor(ncm, alpha=args.alpha)
                        cp.calibrate(Xc, yc, all_classes=allc)
                        try:
                            res = cp.predict(Xt, verbose=False, device=args.device)
                        except (RuntimeError, ValueError):
                            res = cp.predict(Xt, verbose=False, device="cpu")
                            n_fallback += 1
                        psets = res["prediction_sets"]
                        covered = np.array([yt[i] in psets[i]
                                            for i in range(len(yt))])
                        covs.append(float(covered.mean()))
                        szs.append(float(np.mean([len(s) for s in psets])))
                        for c in allc:
                            msk = yt == c
                            if msk.any():
                                j = cls_to_j[int(c)]
                                pooled_cov[j] += float(covered[msk].sum())
                                pooled_tot[j] += int(msk.sum())
                    valid = pooled_tot > 0
                    pcov = pooled_cov[valid] / pooled_tot[valid]
                    row = {"arm": arm, "split": split, "ncm": nm, "cal": cal,
                           "cov": float(np.mean(covs)), "cov_sd": float(np.std(covs)),
                           "sz": float(np.mean(szs)), "sz_sd": float(np.std(szs)),
                           "sz_se": float(np.std(szs) / np.sqrt(len(szs))),
                           "covgap": float(100 * np.mean(np.abs(pcov - (1 - args.alpha)))),
                           "n_trials": args.n_trials, "n_cpu_fallback": n_fallback}
                    if getattr(tfs[0], "lw_shrinkage_", None) is not None:
                        row["lw_shrinkage"] = tfs[0].lw_shrinkage_
                    rows.append(row)
                    print(f"  [{split:13s}] cal={cal:4d} {nm:22s} "
                          f"cov={row['cov']:.4f} sz={row['sz']:6.2f}"
                          f"+-{row['sz_se']:.2f} covgap={row['covgap']:5.2f}pp "
                          f"({time.time()-t0:.0f}s)")
        # free GPU between arms (768-dim arms are the heavy ones)
        if args.device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out = {"config": vars(args), "dataset": args.dataset,
           "pool_participation_ratio": pool_pr, "rows": rows}
    out_json = os.path.join(args.output_dir, f"results_{args.dataset}{tag}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_json}")

    if args.plot:
        plot(out, args)


def plot(out, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["rows"]
    present = {r["arm"] for r in rows}
    arm_order = ([a for a in ARMS if a in present]
                 + sorted(a for a in present if a not in ARMS))  # auto->... rows
    colors = plt.cm.tab10(np.linspace(0, 1, len(arm_order)))
    for split in {r["split"] for r in rows}:
        ncms = sorted({r["ncm"] for r in rows})
        fig, axes = plt.subplots(2, len(ncms), figsize=(6.5 * len(ncms), 8),
                                 squeeze=False)
        for j, nm in enumerate(ncms):
            for a, col in zip(arm_order, colors):
                rs = sorted([r for r in rows if r["arm"] == a and r["ncm"] == nm
                             and r["split"] == split], key=lambda r: r["cal"])
                if not rs:
                    continue
                cals = [r["cal"] for r in rs]
                axes[0][j].errorbar(cals, [r["sz"] for r in rs],
                                    yerr=[r["sz_se"] for r in rs],
                                    marker="o", label=a, color=col)
                axes[1][j].plot(cals, [r["cov"] for r in rs], marker="o",
                                label=a, color=col)
            axes[0][j].set_title(f"{nm} — set size ({split})")
            axes[0][j].set_xlabel("cal size"); axes[0][j].set_ylabel("avg set size")
            axes[0][j].legend(fontsize=8)
            axes[1][j].axhline(1 - args.alpha, ls="--", c="gray")
            axes[1][j].set_title(f"{nm} — coverage ({split})")
            axes[1][j].set_xlabel("cal size"); axes[1][j].set_ylabel("coverage")
            axes[1][j].set_ylim(0.85, 1.0)
        fig.suptitle(f"Transform controls — {out['dataset']} ({split}), "
                     f"{args.n_trials} trials, all arms exchangeable")
        fig.tight_layout()
        tag = f"_{args.out_tag}" if args.out_tag else ""
        p = os.path.join(args.output_dir,
                         f"transform_controls_{out['dataset']}_{split}{tag}.png")
        fig.savefig(p, dpi=140)
        print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
