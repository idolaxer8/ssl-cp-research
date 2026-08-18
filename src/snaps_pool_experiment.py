"""
SNAPS-adaptation pilot: unlabeled-pool score correction for split-style CP.

Adapts SNAPS (Song et al., NeurIPS 2024, arXiv 2405.14303) to the label-free
SSL+CP stack. SNAPS corrects each point's nonconformity score PER CANDIDATE
LABEL by mixing in the scores of feature-similar points (column-wise smoothing
of the N x K score matrix). Their ImageNet variant draws the k neighbors from a
25k calibration set; at cal=200-800 with K=100 that pool is starved, so here
the neighbors come from the UNLABELED POOL instead (the adaptation the paper
never tests):

    s_hat(x, y) = (1 - eta) * s(x, y) + (eta / k) * sum_{u in kNN_pool(x)} s(u, y)

where s(u, y) is label-free (the score any point WOULD get at candidate y).
Same-label neighbors have LOW scores at the ego's true label and HIGH scores at
false labels, so the correction SHARPENS class discrimination -- provided the
neighbors actually share the ego's label (logged as `purity`).

Base score: prototype_softmax LAC, split-CP style:
  - cal points:      exact closed-form LOO scores (PrototypeSoftmaxNCM.alpha0)
  - test/pool points: plain full-prototype scores (one matvec), identical to
    the NCM's own score_x for out-of-bag points

VALIDITY: the pool scores are computed under CAL-FIT prototypes. The naive
correction (--correction naive) lets each cal point's own feature leak into its
correction through the ego -> prototype -> neighborhood loop: the class-y_i
prototype includes z_i, the kNN neighbors of x_i sit in x_i's region, so their
scores at y_i are biased LOW -> the quantile deflates -> UNDER-COVERAGE that
scales with eta and 1/n_c (measured 2026-07-28: cov 0.74/0.84/0.88 at
cal 200/400/800, eta=0.5 k=5; the oracle mode dodges it because random
same-class neighbors are not localized near the ego). The DEFAULT
--correction loo therefore mirrors the base-score construction: each cal
point's correction uses pool scores under the closed-form leave-one-out
prototype of its own class (mu_c^(-i) = (n_c mu_c - z_i)/(n_c - 1)), the same
LOO rule alpha0 itself uses; the test point's correction uses the full-cal
prototypes it is already excluded from. Still soft-gated (split-style, not the
full-CP symmetric construction); report the --split random arm empirically.

Modes:
  --neighbor_mode oracle  Stage 0 go/no-go ceiling: aggregate m random pool
                          points with the SAME TRUE LABEL as the ego (pool
                          labels used as oracle diagnostics only). Replicates
                          the paper's Fig 1a on our embeddings. Sweeps m at
                          --oracle_eta, plus an eta sweep at m=--oracle_m.
  --neighbor_mode knn     Stage 1 deployable: cosine kNN over raw pool
                          embeddings, no labels anywhere. Sweeps eta x k.

Examples
--------
# stage 0 (oracle ceiling)
python src/snaps_pool_experiment.py --neighbor_mode oracle --allow_nonexchangeable

# stage 1 (deployable) -- default balanced arm + the exact-validity random arm
python src/snaps_pool_experiment.py --neighbor_mode knn --allow_nonexchangeable
python src/snaps_pool_experiment.py --neighbor_mode knn --split random --allow_nonexchangeable
"""
import sys, os, math, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import (PrototypeSoftmaxNCM, GeodesicTopKMeanNCM,
                                  warn_nonexchangeable)
from exchangeable_features import make_transform
from exchangeable_fcp_experiment import SPLITS


# ---------------------------------------------------------------- score utils

def prototype_score_matrix(ncm, X, allc):
    """Split-style LAC scores for arbitrary out-of-bag points: 1 - softmax
    (z @ P / T) over the cal-fit full-mean prototypes. One column per class in
    ``allc``; classes absent from cal get the NCM's empty-class score 1.0.
    Matches ncm.score_x for out-of-bag points exactly."""
    Z = ncm._prep(X)
    F = Z @ ncm.P
    if not ncm._P_ok.all():
        F[:, ~ncm._P_ok] = -np.inf
    Pm = ncm._softmax_rows(F / ncm._T)
    S = np.ones((len(Z), len(allc)))
    for j, c in enumerate(allc):
        col = ncm._cls_to_col.get(int(c))
        if col is not None:
            S[:, j] = 1.0 - Pm[:, col]
    return S


def geodesic_score_matrix(ncm, X, allc):
    """Vectorized split-style geodesic scores for arbitrary out-of-bag points:
    s(x, y) = arccos(topk-mean sim to class-y cal) / (arccos(topk-mean sim to
    NOT-class-y cal) + 1e-8), bit-matching GeodesicTopKMeanNCM.score_x (mean
    variant). Returns (S, d_other, sims) — d_other/sims reused by the LOO cal
    correction. Pooled other-class top-k is exact via a global top-2k merge of
    per-class top-k values (any class contributes at most k candidates)."""
    Z = ncm._whiten_normalize(np.asarray(X, dtype=np.float64))
    sims = Z @ ncm.X_cal_wn.T                      # (n, n_cal)
    k, K, n = ncm.k, len(allc), len(Z)
    V = np.full((n, K, k), -np.inf)
    for j, c in enumerate(allc):
        cols = np.where(ncm.y_cal == c)[0]
        if len(cols) == 0:
            continue
        sc = sims[:, cols]
        kk = min(k, sc.shape[1])
        top = np.partition(sc, sc.shape[1] - kk, axis=1)[:, -kk:] if kk < sc.shape[1] else sc
        V[:, j, :kk] = top
    finite = np.isfinite(V)
    keff_same = finite.sum(axis=2)
    sum_same = np.where(finite, V, 0.0).sum(axis=2)
    d_same = np.arccos(np.clip(sum_same / np.maximum(keff_same, 1), -1.0, 1.0))
    # other side: global top-2k of per-class top-k values, filter out class y
    G = V.reshape(n, K * k)
    Lcls = np.repeat(np.arange(K), k)
    order = np.argsort(-G, axis=1)[:, :2 * k]
    topvals = np.take_along_axis(G, order, axis=1)  # (n, 2k) descending
    topcls = Lcls[order]
    d_other = np.empty((n, K))
    for j in range(K):
        m = (topcls != j) & np.isfinite(topvals)
        take = m & (m.cumsum(axis=1) <= k)
        s = np.where(take, topvals, 0.0).sum(axis=1)
        ko = take.sum(axis=1)
        d_other[:, j] = np.arccos(np.clip(s / np.maximum(ko, 1), -1.0, 1.0))
    return d_same / (d_other + 1e-8), d_other, sims


def geodesic_cal_correction_loo(ncm, allc, S_pool, d_other_pool, sims_pool_cal,
                                nbr, y_cal_col):
    """Correction for each CAL point i at its true label with the ego REMOVED
    from the reference set: only the same-class numerator of the neighbors'
    scores changes (the ego is never on the other side of its own label), so
    re-top-k the neighbors' sims to class y_i minus column i and reuse the
    precomputed pooled denominator."""
    n_cal = sims_pool_cal.shape[1]
    cols_by_class = {j: np.where(ncm.y_cal == allc[j])[0] for j in range(len(allc))}
    c = np.empty(len(y_cal_col))
    k = ncm.k
    for i in range(len(y_cal_col)):
        j = y_cal_col[i]
        cols = cols_by_class[j]
        keep = cols[cols != i]
        sc = sims_pool_cal[nbr[i]][:, keep] if len(keep) else None
        if sc is None or sc.shape[1] == 0:
            d_same = np.full(len(nbr[i]), np.arccos(0.0))
        else:
            kk = min(k, sc.shape[1])
            top = (np.partition(sc, sc.shape[1] - kk, axis=1)[:, -kk:]
                   if kk < sc.shape[1] else sc)
            d_same = np.arccos(np.clip(top.mean(axis=1), -1.0, 1.0))
        c[i] = float(np.mean(d_same / (d_other_pool[nbr[i], j] + 1e-8)))
    return c


def conformal_quantile(scores, alpha):
    n = len(scores)
    level = math.ceil((1 - alpha) * (n + 1)) / n
    if level > 1.0:
        return np.inf
    return float(np.quantile(scores, level, method="higher"))


def evaluate_sets(S_test, y_test, allc, qhat, alpha):
    """Coverage / mean size / singleton-hit / CovGap(pp) / SSCV from the
    (n_test, K) corrected score matrix and the calibrated quantile."""
    inset = S_test <= qhat
    sizes = inset.sum(axis=1)
    col_of = {int(c): j for j, c in enumerate(allc)}
    ycols = np.array([col_of[int(v)] for v in y_test])
    covered = inset[np.arange(len(y_test)), ycols]
    gaps = [abs(covered[ycols == j].mean() - (1 - alpha))
            for j in range(len(allc)) if (ycols == j).any()]
    sscv = 0.0
    for lo, hi in [(0, 1), (2, 3), (4, 10), (11, 100), (101, 10 ** 9)]:
        m = (sizes >= lo) & (sizes <= hi)
        if m.any():
            sscv = max(sscv, abs(covered[m].mean() - (1 - alpha)))
    return {"cov": float(covered.mean()), "sz": float(sizes.mean()),
            "sh": float(((sizes == 1) & covered).mean()),
            "covgap": float(np.mean(gaps) * 100), "sscv": float(sscv)}


# ---------------------------------------------------------------- neighbors

def _l2(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def pool_knn(X_pts, Xu_n, kmax):
    """Top-kmax pool neighbors per point by cosine similarity (raw embeddings),
    sorted by similarity so idx[:, :k] is the top-k for any k <= kmax."""
    A = _l2(X_pts) @ Xu_n.T
    idx = np.argpartition(-A, kmax - 1, axis=1)[:, :kmax]
    row = np.arange(len(A))[:, None]
    order = np.argsort(-A[row, idx], axis=1)
    return idx[row, order]


def oracle_neighbors(y_pts, pool_by_class, m, rng):
    """m random pool points with the SAME TRUE LABEL as each point (oracle
    diagnostic). Returns (n_pts, m) pool indices (fewer repeated if starved)."""
    out = np.empty((len(y_pts), m), dtype=np.int64)
    for i, c in enumerate(y_pts):
        cand = pool_by_class[int(c)]
        take = rng.choice(cand, size=m, replace=len(cand) < m)
        out[i] = take
    return out


def cal_correction_loo(ncm, F_pool, Zu_prep, nbr):
    """Correction term for each CAL point i at its true label, with the class
    of i re-prototyped LEAVE-ONE-OUT (mu^(-i)): kills the ego -> prototype ->
    neighborhood leak that deflates the quantile in the naive construction.
    Only column y_i of the neighbors' logit rows changes -> cheap."""
    n = len(ncm.Z)
    c = np.empty(n)
    for i in range(n):
        jc = ncm.y_col[i]
        p = ncm._proto(ncm.class_sum[:, jc] - ncm.Z[i], ncm.n_c[jc] - 1.0)
        rows = F_pool[nbr[i]].copy()
        rows[:, jc] = -np.inf if p is None else Zu_prep[nbr[i]] @ p
        Pm = ncm._softmax_rows(rows / ncm._T)
        c[i] = float(np.mean(1.0 - Pm[:, jc]))
    return c


# ---------------------------------------------------------------- experiment

def main():
    ap = argparse.ArgumentParser(description="SNAPS-style pool-kNN score correction (split-style prototype CP)")
    ap.add_argument("--embeddings_path", type=str,
                    default="output/from_cluster/embeddings/embeddings_cifar100.pt")
    ap.add_argument("--unlabeled_path", type=str,
                    default="output/from_cluster/embeddings/embeddings_cifar100_unlabeled.pt")
    ap.add_argument("--neighbor_mode", type=str, default="knn", choices=["knn", "oracle"])
    ap.add_argument("--base", type=str, default="prototype", choices=["prototype", "geodesic"],
                    help="Base NCM: prototype_softmax LAC (stage 1) or the champion "
                         "unwhitened geodesic top-k mean (stage 2), split-style.")
    ap.add_argument("--transform", type=str, default="none", choices=["none", "pool"],
                    help="'pool' fits UnlabeledTransform (PCA + cluster-whiten) on the "
                         "unlabeled pool; scores AND kNN run in the transformed space.")
    ap.add_argument("--pca_dim", type=int, default=128,
                    help="PCA dim for --transform pool (0/negative = full-rank whiten only).")
    ap.add_argument("--pre", type=str, default="none", choices=["none", "qe"],
                    help="'qe' = alpha-QE pool-neighbor feature smoothing before the "
                         "pool transform (qe x SNAPS stacking axis, pool-repr menu "
                         "round 2). Scores AND the SNAPS neighbor graph then live in "
                         "the qe-smoothed transformed space.")
    ap.add_argument("--n_clusters_whiten", type=int, default=100)
    ap.add_argument("--dataset", type=str, default=None,
                    help="Label recorded in config/tag (inferred from embeddings path if omitted).")
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50],
                    help="kNN neighbor counts (knn mode).")
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 40],
                    help="Oracle same-label pool points aggregated (oracle mode; 0 = baseline).")
    ap.add_argument("--oracle_eta", type=float, default=0.5, help="eta for the oracle m sweep.")
    ap.add_argument("--oracle_m", type=int, default=10, help="m for the oracle eta sweep.")
    ap.add_argument("--temperature", type=str, default="auto",
                    help="Softmax T: a float (exact) or 'auto' = pilot-fit once and held "
                         "fixed across all trials (ridge_softmax pattern).")
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--test_size", type=int, default=1000,
                    help="Test size for --split random (balanced_both ignores it).")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--split", type=str, default="balanced_both", choices=list(SPLITS))
    ap.add_argument("--correction", type=str, default="loo", choices=["loo", "naive"],
                    help="Cal-point correction: 'loo' (default) re-prototypes the ego's "
                         "class leave-one-out (kills the ego->prototype->neighborhood "
                         "leak); 'naive' keeps the leaky full-prototype pool scores "
                         "(under-covers, kept for the record).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow_nonexchangeable", action="store_true",
                    help="Approve the cal-fit-NCM pool-score correction (O(1/n) gate).")
    ap.add_argument("--output_dir", type=str, default="output/snaps_pool")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    def _load(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        emb = d["embeddings"] if "embeddings" in d else d["final"]  # layers files
        return emb.numpy(), d["labels"].numpy()

    X, y = _load(args.embeddings_path)
    Xu, yu = _load(args.unlabeled_path)
    allc = np.unique(y)
    K = len(allc)
    ds = args.dataset or os.path.basename(args.embeddings_path
                                          ).replace("embeddings_", "").replace(".pt", "")
    print(f"[{ds}] Labeled: {X.shape}, {K} classes | Pool: {Xu.shape} "
          f"(pool labels used ONLY for oracle mode / purity diagnostics)")

    # optional pool-fit feature transform (label-free, trial-independent);
    # scores AND the neighbor search both live in the transformed space
    if args.transform == "pool":
        pca_dim = args.pca_dim if args.pca_dim and args.pca_dim > 0 else None
        kw = {"pre": "qe"} if args.pre == "qe" else {}
        transform = make_transform(Xu, pca_dim=pca_dim, whiten="cluster",
                                   n_clusters=args.n_clusters_whiten, **kw)
        X = transform.transform(X)
        # reuse the fit-time pool cache: identical to transform(Xu) for linear
        # stages, and for pre='qe' it is the leave-self-out smoothing (the
        # apply-path would double-count each pool point's own vector)
        Xu_t = getattr(transform, "Xu_transformed_", None)
        Xu = Xu_t if Xu_t is not None else transform.transform(Xu)
        print(f"pool transform: {transform} -> features {X.shape[1]}-d")

    # correction validity gate (fires once; the base eta=0 arm is unaffected)
    warn_nonexchangeable(
        "SNAPS pool-score correction (pool scores under cal-fit prototypes)",
        "Corrected cal/test scores share a cal-fit NCM -> O(1/n) asymmetry "
        "(same character as cal-fit whitening). Report the --split random arm "
        "empirically; the exact full-CP wrapping is a later rung.",
        order="O(1/n)", allow=args.allow_nonexchangeable)

    # ONE fixed temperature across all trials (pilot draw, disjoint seed)
    T = None
    if args.base == "prototype":
        if args.temperature == "auto":
            rng0 = np.random.default_rng(args.seed + 987654)
            pci, _ = SPLITS["balanced_both"](X, y, max(args.cal_sizes), 0, allc, rng0)
            pilot = PrototypeSoftmaxNCM(temperature=None, allow_nonexchangeable=True
                                        ).fit(X[pci], y[pci])
            T = float(pilot._T)
            print(f"prototype_softmax: fixed T={T:.4f} (pilot; constant across trials)")
        else:
            T = float(args.temperature)

    # arm list: (label, eta, k_or_m)
    if args.neighbor_mode == "knn":
        arms = [("baseline", 0.0, 0)]
        arms += [(f"knn eta={e:g} k={k}", e, k)
                 for e in args.etas if e > 0 for k in args.ks]
        kmax = max(args.ks)
    else:
        arms = [("baseline", 0.0, 0)]
        arms += [(f"oracle m={m} eta={args.oracle_eta:g}", args.oracle_eta, m)
                 for m in args.ms if m > 0]
        arms += [(f"oracle m={args.oracle_m} eta={e:g}", e, args.oracle_m)
                 for e in args.etas if e > 0 and e != args.oracle_eta]
        pool_by_class = {int(c): np.where(yu == c)[0] for c in allc}
        for c in allc:
            if len(pool_by_class[int(c)]) == 0:
                raise ValueError(f"oracle mode: pool has no points of class {c}")

    Xu_n = _l2(Xu)
    split_fn = SPLITS[args.split]
    pre_tag = "" if args.pre == "none" else f"_{args.pre}"
    tag = args.tag or (f"{ds}_{args.base}_{args.transform}{pre_tag}_"
                       f"{args.neighbor_mode}_{args.split}_{args.correction}")
    print(f"base={args.base} | transform={args.transform} | mode={args.neighbor_mode} "
          f"| split={args.split} | correction={args.correction} | alpha={args.alpha} "
          f"| trials={args.n_trials} | arms={len(arms)}")

    results = {"tag": tag,
               "config": {"dataset": ds, "base": args.base, "transform": args.transform,
                          "pre": args.pre,
                          "pca_dim": args.pca_dim if args.transform == "pool" else None,
                          "neighbor_mode": args.neighbor_mode, "split": args.split,
                          "correction": args.correction,
                          "alpha": args.alpha, "n_trials": args.n_trials,
                          "temperature": T, "etas": args.etas,
                          "ks": args.ks if args.neighbor_mode == "knn" else args.ms,
                          "embeddings": args.embeddings_path,
                          "unlabeled": args.unlabeled_path},
               "rows": []}

    for cal in args.cal_sizes:
        print(f"\ncal={cal}")
        per_arm = {label: [] for label, _, _ in arms}
        t0 = time.time()
        for trial in range(args.n_trials):
            rng = np.random.default_rng(args.seed + trial * 1000)
            ci, ti = split_fn(X, y, cal, args.test_size, allc, rng)
            col_of = {int(c): j for j, c in enumerate(allc)}
            ycols_cal = np.array([col_of[int(v)] for v in y[ci]])

            if args.base == "prototype":
                ncm = PrototypeSoftmaxNCM(temperature=T).fit(X[ci], y[ci])
                Zu_prep = ncm._prep(Xu)                          # (n_pool, d)
                F_pool = Zu_prep @ ncm.P                         # (n_pool, K_cal)
                if not ncm._P_ok.all():
                    F_pool[:, ~ncm._P_ok] = -np.inf
                S_pool = prototype_score_matrix(ncm, Xu, allc)   # (n_pool, K)
                S_test = prototype_score_matrix(ncm, X[ti], allc)
                s_cal = ncm.alpha0         # cal base @ true label (exact LOO)
            else:
                ncm = GeodesicTopKMeanNCM(whiten=False)
                ncm.fit(X[ci], y[ci])
                S_pool, d_other_pool, sims_pool_cal = geodesic_score_matrix(ncm, Xu, allc)
                S_test, _, _ = geodesic_score_matrix(ncm, X[ti], allc)
                s_cal = ncm.get_calibration_scores()
                if trial == 0:
                    chk = np.random.default_rng(0)
                    for _ in range(20):     # parity vs the NCM's own score_x
                        i0 = int(chk.integers(len(ti))); j0 = int(chk.integers(K))
                        ref = ncm.score_x(X[ti][i0], int(allc[j0]))
                        assert abs(ref - S_test[i0, j0]) < 1e-8, \
                            f"geodesic matrix mismatch: {ref} vs {S_test[i0, j0]}"

            if args.neighbor_mode == "knn":
                nbr_all = pool_knn(np.vstack([X[ci], X[ti]]), Xu_n, kmax)
                nbr_cal_all, nbr_test_all = nbr_all[:len(ci)], nbr_all[len(ci):]
            purity, c_cal_cache, c_test_cache = {}, {}, {}

            def corrections_for(km):
                """(c_cal @ true label, c_test all-K) for km neighbors, cached."""
                if km in c_cal_cache:
                    return c_cal_cache[km], c_test_cache[km]
                if args.neighbor_mode == "knn":
                    nbr_cal, nbr_test = nbr_cal_all[:, :km], nbr_test_all[:, :km]
                    nl = yu[np.vstack([nbr_cal, nbr_test])]
                    tl = np.concatenate([y[ci], y[ti]])[:, None]
                    purity[km] = float((nl == tl).mean())
                else:
                    nbr_cal = oracle_neighbors(y[ci], pool_by_class, km, rng)
                    nbr_test = oracle_neighbors(y[ti], pool_by_class, km, rng)
                if args.correction != "loo":
                    c_cal = S_pool[nbr_cal].mean(axis=1)[np.arange(len(ci)), ycols_cal]
                elif args.base == "prototype":
                    c_cal = cal_correction_loo(ncm, F_pool, Zu_prep, nbr_cal)
                else:
                    c_cal = geodesic_cal_correction_loo(
                        ncm, allc, S_pool, d_other_pool, sims_pool_cal,
                        nbr_cal, ycols_cal)
                c_test = S_pool[nbr_test].mean(axis=1)           # (n_test, K)
                c_cal_cache[km], c_test_cache[km] = c_cal, c_test
                return c_cal, c_test

            for label, eta, km in arms:
                if eta == 0.0:
                    s_cal_hat, S_test_hat = s_cal, S_test
                else:
                    c_cal, c_test = corrections_for(km)
                    s_cal_hat = (1.0 - eta) * s_cal + eta * c_cal
                    S_test_hat = (1.0 - eta) * S_test + eta * c_test
                qhat = conformal_quantile(s_cal_hat, args.alpha)
                m = evaluate_sets(S_test_hat, y[ti], allc, qhat, args.alpha)
                m["purity"] = purity.get(km, float("nan")) if eta > 0 else float("nan")
                per_arm[label].append(m)

        n_tr = args.n_trials
        exact_target = 1 - math.floor(args.alpha * (cal + 1)) / (cal + 1)
        for label, eta, km in arms:
            tr = per_arm[label]
            agg = {k: float(np.mean([t[k] for t in tr]))
                   for k in ("cov", "sz", "sh", "covgap", "sscv", "purity")}
            agg["cov_se"] = float(np.std([t["cov"] for t in tr], ddof=1) / math.sqrt(n_tr))
            agg["sz_se"] = float(np.std([t["sz"] for t in tr], ddof=1) / math.sqrt(n_tr))
            row = {"cal": cal, "mode": args.neighbor_mode, "split": args.split,
                   "method": label, "eta": eta,
                   ("k" if args.neighbor_mode == "knn" else "m"): km,
                   "n_trials": n_tr, "exact_target": exact_target, **agg}
            results["rows"].append(row)
            pur = "" if math.isnan(agg["purity"]) else f"  pur={agg['purity']:.2f}"
            print(f"  {label:24s} cov={agg['cov']:.4f}±{agg['cov_se']:.4f}  "
                  f"sz={agg['sz']:6.2f}±{agg['sz_se']:.2f}  SH={agg['sh']*100:5.1f}%  "
                  f"CovGap={agg['covgap']:.2f}pp  SSCV={agg['sscv']:.3f}{pur}")
        print(f"  ({time.time()-t0:.1f}s)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"results_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {out_json}")


if __name__ == "__main__":
    main()
