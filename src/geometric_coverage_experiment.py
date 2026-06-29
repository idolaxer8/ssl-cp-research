"""
B-Stage-1 DIAGNOSTIC: is FCP coverage uniform across GEOMETRIC strata under a
balanced (class-count-equalised) protocol?

Reframing direction B (see memory novelty-directions-pac-and-geometric-coverage):
class-count imbalance is removed by the balanced protocol, so any residual
coverage non-uniformity is NOT a label-frequency artifact. We stratify TEST points
by a LABEL-FREE geometric covariate computed on the disjoint unlabeled POOL (local
kNN density, Levina-Bickel local intrinsic dimension, or k-means super-cluster),
which is a fixed function of U (independent of cal/test) -> exactly exchangeable
(repo theory.md Prop 2). Then we ask whether per-stratum coverage cov_geo(g) is
flat (== KILL the direction) or sloped (== real geometric difficulty -> Stage 2).

Metric: CovGap_geo = mean_g |cov_geo(g) - (1-alpha)| * 100 (pp), the geometry-axis
twin of the class-axis CovGap (Ding/Tibshirani/Ramdas 2023). Plus worst-stratum
coverage and a coverage-vs-covariate monotonicity (Spearman) check.

Bonus: per-class coverage vs class-prototype pool-LID scatter -- directly tests
whether the project's standing "~31% of classes under-covered even under balance"
tail is the high-LID / low-density geometric tail.

All math plain-text (repo convention). Run from the worktree; embeddings are read
from the main checkout's gitignored output/ via --embeddings_dir.
"""
import sys, os, json, argparse, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import make_transform
from mscs_unlabeled_experiment import (build_cluster_similarity_matrix,
                                        run_fcp_with_mscs)

MAIN_OUT = r"C:\Users\IDO\Desktop\Ido_student\Msc\ssl-cp-research\output\from_cluster"
# MS-CS best config (memory): lambda=0.05, 20 clusters, tau = 0.5 * median_d^2
MSCS_LAM, MSCS_NCLUST, MSCS_TAU = 0.05, 20, -0.5


@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


# ----------------------------- data ---------------------------------------
def load_emb(path):
    d = torch.load(path, map_location="cpu")
    X = np.asarray(d["embeddings"], dtype=np.float64)
    y = np.asarray(d["labels"]).astype(int).ravel()
    return X, y


def carve_test_and_pool(y, m_test, allc, rng):
    """Hold out a FIXED balanced test set (m_test/class); the rest of each class
    is the per-draw cal pool. Fixed test isolates calibration variance."""
    test_idx, pool_idx = [], []
    for c in allc:
        perm = rng.permutation(np.where(y == c)[0])
        test_idx.append(perm[:m_test])
        pool_idx.append(perm[m_test:])
    return np.concatenate(test_idx), np.concatenate(pool_idx)


def sample_cal(pool_idx, y, cal, balanced, allc, rng):
    if balanced:
        m = cal // len(allc)
        ci = []
        for c in allc:
            members = pool_idx[y[pool_idx] == c]
            ci.append(rng.choice(members, m, replace=False))
        return np.concatenate(ci)
    return rng.choice(pool_idx, cal, replace=False)


# ------------------------- geometry covariates ----------------------------
def knn_radii(nn, Xq, k, drop_self):
    """k neighbour radii for each query row. drop_self=True for pool self-query."""
    nq = k + 1 if drop_self else k
    dist, _ = nn.kneighbors(Xq, n_neighbors=nq)
    return dist[:, 1:k + 1] if drop_self else dist[:, :k]


def density_cov(radii):
    """Local sparsity proxy = distance to the k-th neighbour (large = sparse)."""
    return radii[:, -1]


def lid_mle(radii, eps=1e-12):
    """Levina-Bickel MLE local intrinsic dimension.
    LID = [ (1/(k-1)) sum_{i<k} log(r_k / r_i) ]^{-1}."""
    r = np.maximum(radii, eps)
    rk = r[:, -1:]
    inv = np.log(rk / r[:, :-1]).mean(axis=1)
    return 1.0 / np.maximum(inv, eps)


def strata_from_edges(values, edges):
    return np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)


# ------------------------------ metrics -----------------------------------
def per_stratum(covered, sizes, strata, n_strata, alpha):
    """covered: bool (N,), sizes: (N,), strata: int (N,). Aggregated over trials."""
    target = 1.0 - alpha
    rows, covs = [], []
    for g in range(n_strata):
        m = strata == g
        n = int(m.sum())
        if n == 0:
            rows.append(dict(stratum=g, n=0, cov=None, size=None)); continue
        cov = float(covered[m].mean()); sz = float(sizes[m].mean())
        rows.append(dict(stratum=g, n=n, cov=cov, size=sz)); covs.append(cov)
    covs = np.array(covs)
    covgap = float(np.mean(np.abs(covs - target)) * 100) if covs.size else None
    worst = float(covs.min()) if covs.size else None
    # monotonicity: Spearman corr of stratum index vs coverage
    if covs.size >= 3:
        from scipy.stats import spearmanr
        rho = float(spearmanr(np.arange(covs.size), covs).statistic)
    else:
        rho = None
    return dict(rows=rows, covgap_geo=covgap, worst=worst, spearman=rho,
                marg_cov=float(covered.mean()), marg_size=float(sizes.mean()))


# ------------------------------ run ---------------------------------------
def run(args):
    rng = np.random.default_rng(args.seed)
    allc = np.arange(args.n_classes)
    dev = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    Xb, yb = load_emb(os.path.join(args.embeddings_dir, f"embeddings_{args.dataset}.pt"))
    Xu, _ = load_emb(os.path.join(args.embeddings_dir, f"embeddings_{args.dataset}_unlabeled.pt"))
    print(f"[data] {args.dataset}: base {Xb.shape}, pool {Xu.shape}, device={dev}")

    # exchangeable transform fit on the pool only
    transform = make_transform(Xu, pca_dim=args.pca_dim, whiten="cluster",
                               n_clusters=args.n_clusters)
    Xu_t = transform.Xu_transformed_                      # pool in transformed space

    # geometry: kNN structure on the transformed pool (label-free, pool-fixed)
    nn = NearestNeighbors(n_neighbors=args.knn + 1).fit(Xu_t)
    pool_rad = knn_radii(nn, Xu_t, args.knn, drop_self=True)
    pool_cov = {"density": density_cov(pool_rad), "lid": lid_mle(pool_rad)}
    edges = {name: np.quantile(v, np.linspace(0, 1, args.n_bins + 1))
             for name, v in pool_cov.items()}
    # super-clusters: KMeans on the pool's k-means centroids -> coarse geo cells
    super_km = KMeans(n_clusters=args.n_super, random_state=42,
                      n_init=10).fit(transform.cluster_centroids_)
    cluster_to_super = super_km.labels_                   # (n_clusters,)

    # fixed pilot temperature for prototype_softmax (constant across trials => exact)
    T_fixed = None
    if "prototype" in args.arms:
        prng = np.random.default_rng(args.seed + 9999)
        ti, pi = carve_test_and_pool(yb, args.m_test, allc, prng)
        pci = sample_cal(pi, yb, min(max(args.cal_sizes), 800), True, allc, prng)
        with quiet():
            pncm = create_ncm("prototype_softmax", temperature=None,
                              logit="cosine", allow_nonexchangeable=True)
            pncm.fit(transform.transform(Xb[pci]), yb[pci])
            T_fixed = float(pncm._resolve_T())
        print(f"[pilot] prototype_softmax fixed T = {T_fixed:.4f}")

    def covariate_strata(Xq_t):
        rad = knn_radii(nn, Xq_t, args.knn, drop_self=False)
        s = {"density": strata_from_edges(density_cov(rad), edges["density"]),
             "lid": strata_from_edges(lid_mle(rad), edges["lid"]),
             "cluster": cluster_to_super[transform.cluster_of(Xq_t)]}
        return s

    n_strata = {"density": args.n_bins, "lid": args.n_bins, "cluster": args.n_super}
    arms = {"prototype": ("prototype_softmax", dict(temperature=T_fixed, logit="cosine")),
            "geodesic": ("unwhitened_topk_asym", dict(k=5))}

    results = {}
    # accumulators for the per-class-coverage-vs-LID scatter (cal = max, balanced)
    cls_cov = {a: {"hit": np.zeros(args.n_classes), "tot": np.zeros(args.n_classes)}
               for a in args.arms}

    for cal in args.cal_sizes:
        for split in args.splits:
            balanced = (split == "balanced")
            for arm in args.arms:
                if not balanced and arm != "geodesic":
                    print(f"[skip] {arm} on random split (needs all classes in cal)"); continue
                # accumulate test-point records across trials (fixed test each trial)
                cov_all, sz_all = [], []
                strat_all = {k: [] for k in n_strata}
                for t in range(args.n_trials):
                    trng = np.random.default_rng(args.seed + 1000 * cal + t)
                    test_idx, pool_idx = carve_test_and_pool(yb, args.m_test, allc, trng)
                    ci = sample_cal(pool_idx, yb, cal, balanced, allc, trng)
                    Xc_t = transform.transform(Xb[ci]); yc = yb[ci]
                    Xt_t = transform.transform(Xb[test_idx]); yt = yb[test_idx]
                    with quiet():
                        if arm.endswith("_mscs"):
                            base = arm[:-5]
                            ncm_name = ("prototype_softmax" if base == "prototype"
                                        else "unwhitened_topk_asym")
                            kw_m = (dict(temperature=T_fixed, logit="cosine")
                                    if base == "prototype" else {})
                            M, *_ = build_cluster_similarity_matrix(
                                transform.Xu_transformed_, Xc_t, yc, allc,
                                MSCS_NCLUST, tau=MSCS_TAU)
                            _, _, psets = run_fcp_with_mscs(
                                Xc_t, yc, Xt_t, yt, allc, ncm_name, args.alpha,
                                MSCS_LAM, M, exchangeable=False, return_sets=True,
                                allow_nonexchangeable=True, device=dev, **kw_m)
                        else:
                            ncm_type, kw = arms[arm]
                            ncm = create_ncm(ncm_type, **{k: v for k, v in kw.items()
                                                          if v is not None})
                            cp = FullConformalPredictor(ncm, alpha=args.alpha)
                            cp.calibrate(Xc_t, yc, all_classes=allc)
                            res = cp.predict(Xt_t, device=dev, verbose=False)
                            psets = res["prediction_sets"]
                    covered = np.array([yt[i] in set(ps) for i, ps in enumerate(psets)])
                    sizes = np.array([len(ps) for ps in psets], dtype=float)
                    st = covariate_strata(Xt_t)
                    cov_all.append(covered); sz_all.append(sizes)
                    for k in n_strata:
                        strat_all[k].append(st[k])
                    # per-class scatter accumulation at the largest balanced cal
                    if balanced and cal == max(args.cal_sizes):
                        for i in range(len(yt)):
                            c = yt[i]
                            cls_cov[arm]["tot"][c] += 1
                            cls_cov[arm]["hit"][c] += covered[i]
                cov_all = np.concatenate(cov_all); sz_all = np.concatenate(sz_all)
                key = f"{cal}_{split}_{arm}"
                results[key] = {"marg_cov": float(cov_all.mean()),
                                "marg_size": float(sz_all.mean()),
                                "by_covariate": {}}
                for k in n_strata:
                    sa = np.concatenate(strat_all[k])
                    m = per_stratum(cov_all, sz_all, sa, n_strata[k], args.alpha)
                    results[key]["by_covariate"][k] = m
                    print(f"[{key}] {k}: covgap_geo={m['covgap_geo']:.2f}pp "
                          f"worst={m['worst']:.3f} spearman={m['spearman']} "
                          f"marg={m['marg_cov']:.3f} sz={m['marg_size']:.2f}")

    # --- per-class coverage vs class-prototype pool-LID -----------------------
    scatter = {}
    for arm in args.arms:
        tot = cls_cov[arm]["tot"]
        if tot.sum() == 0:
            continue
        cov_c = np.where(tot > 0, cls_cov[arm]["hit"] / np.maximum(tot, 1), np.nan)
        # class prototype in transformed space (from all base data of the class)
        mu = np.array([transform.transform(Xb[yb == c]).mean(0) for c in allc])
        mu_rad = knn_radii(nn, mu, args.knn, drop_self=False)
        lid_c = lid_mle(mu_rad)
        from scipy.stats import spearmanr
        good = ~np.isnan(cov_c)
        rho = float(spearmanr(lid_c[good], cov_c[good]).statistic)
        scatter[arm] = {"cov_c": cov_c.tolist(), "lid_c": lid_c.tolist(),
                        "spearman_cov_vs_lid": rho,
                        "frac_undercov": float((cov_c[good] < 1 - args.alpha).mean())}
        print(f"[scatter:{arm}] spearman(cov_c, LID_c)={rho:.3f} "
              f"frac_undercov={scatter[arm]['frac_undercov']:.3f}")
    results["_scatter"] = scatter

    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "geometric_coverage.json"), "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"[done] wrote {out_dir}/geometric_coverage.json")
    if args.plot:
        make_plots(results, args, out_dir)


def make_plots(results, args, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    covs = [c for c in ["density", "lid", "cluster"]]
    for cal in args.cal_sizes:
        fig, axes = plt.subplots(1, len(covs), figsize=(4 * len(covs), 3.4), squeeze=False)
        for j, cv in enumerate(covs):
            ax = axes[0][j]
            for arm in args.arms:
                key = f"{cal}_balanced_{arm}"
                if key not in results:
                    continue
                m = results[key]["by_covariate"][cv]
                xs = [r["stratum"] for r in m["rows"] if r["cov"] is not None]
                ys = [r["cov"] for r in m["rows"] if r["cov"] is not None]
                ax.plot(xs, ys, "o-", label=f"{arm} (gap {m['covgap_geo']:.1f})")
            ax.axhline(1 - args.alpha, ls="--", c="k", lw=1)
            ax.set_title(f"{cv} | cal={cal}"); ax.set_xlabel(f"{cv} stratum (low->high)")
            ax.set_ylabel("coverage"); ax.legend(fontsize=7)
        fig.suptitle(f"{args.dataset}: per-geometric-stratum coverage (balanced)")
        fig.tight_layout()
        p = os.path.join(out_dir, f"perstratum_cal{cal}.png")
        fig.savefig(p, dpi=130); plt.close(fig); print(f"[plot] {p}")
    # scatter
    sc = results.get("_scatter", {})
    if sc:
        fig, axes = plt.subplots(1, len(sc), figsize=(4.2 * len(sc), 3.6), squeeze=False)
        for j, (arm, d) in enumerate(sc.items()):
            ax = axes[0][j]
            ax.scatter(d["lid_c"], d["cov_c"], s=14, alpha=0.6)
            ax.axhline(1 - args.alpha, ls="--", c="k", lw=1)
            ax.set_xlabel("class-prototype pool-LID"); ax.set_ylabel("class coverage")
            ax.set_title(f"{arm}: rho={d['spearman_cov_vs_lid']:.2f}")
        fig.suptitle(f"{args.dataset}: per-class coverage vs geometric difficulty (LID)")
        fig.tight_layout()
        p = os.path.join(out_dir, "perclass_cov_vs_lid.png")
        fig.savefig(p, dpi=130); plt.close(fig); print(f"[plot] {p}")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--embeddings_dir", default=MAIN_OUT)
    ap.add_argument("--out_dir", default="output/geometric_coverage")
    ap.add_argument("--n_classes", type=int, default=100)
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[400, 800])
    ap.add_argument("--m_test", type=int, default=20, help="held-out test per class")
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--splits", nargs="+", default=["balanced", "random"])
    ap.add_argument("--arms", nargs="+",
                    default=["prototype", "geodesic", "prototype_mscs", "geodesic_mscs"])
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--n_clusters", type=int, default=100)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--n_super", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--plot", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse())
