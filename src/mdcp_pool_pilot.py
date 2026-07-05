"""
MDCP pool pilot (Pilot B): Multi-Dimensional Conformal Prediction adapted to
the SSL + small-cal regime as a SCORE-SPACE k-NN DENSITY-RATIO scalar NCM.

Background: Tawachi & Laufer-Goldshtein (ICLR 2025), "Multi-Dimensional
Conformal Prediction" (MDCP), stack n 1-D conformal scores into s(x,y) in R^n,
Voronoi-partition the score space on a held-out half of cal (Dcells), rank
cells by purity D_i = (F_i + T_i) / T_i, and calibrate the number of selected
cells on the other half (Dre-cal). Their own Proposition-2 proof reduces the
whole construction to standard 1-D CP on the scalar score "purity of the cell
containing s(x,y)".

This pilot implements that reduction directly, with two adaptations for our
regime (cal 200-800, K=100):

  1. GEOMETRY FROM THE POOL. The true/false score-vector reference cloud that
     defines the purity function is built from the (pseudo-labeled) unlabeled
     pool, not from a sacrificed half of cal. ALL of cal is kept for the
     conformal quantile. (Their 50/50 split at cal=400/K=100 leaves 2 true
     points per class to define the geometry -- pure count noise.)
  2. k-NN SMOOTHING. Purity at a query s = false/true ratio among its k_D
     nearest cloud points. Their Voronoi cells are the 1-NN special case with
     T_i = 1. Note: with a fixed-size combined-cloud k-NN query,
     D = (F+T)/T = k_D / T_count, i.e. purity is monotone in the true-fraction
     among neighbors.

Dimensions (both cal-anchored, computed on pool-fit PCA features):
  s1 = geodesic asym ratio: arccos(max same-class cos) / arccos(mean top-k
       other-class cos)   [= unwhitened_topk_asym convention]
  s2 = prototype cosine LAC: 1 - softmax_T(cos(z, mu_q))  [PrototypeSoftmaxNCM
       convention, pilot-fixed T]
A per-dimension rank (ECDF) transform, fit on the pool score cloud, makes the
two scales commensurate before the k-NN query.

Arms:
  raw1_geo / raw1_proto        plain SCP quantile on each raw dimension
  dratio1_geo / dratio1_proto  D-ratio machinery in 1-D (isolates machinery
                               from dimensionality)
  dratio2_oracle               2-D D-ratio, pool pseudo-labels = TRUE pool
                               labels (diagnostic upper bound only)
  dratio2_fullcal              2-D D-ratio, pool pseudo-labels = cal-anchored
                               prototype argmax (deployable arm)

Validity note: every arm shares the repo's SCP-geodesic scoring convention
(cal scored leave-self-out against cal, test scored against full cal; see
MEMORY scp-geodesic-isolates-ncm-vs-fcp -- empirically tracks FCP tightly).
The purity function additionally depends on cal through the dimension anchors
and (fullcal arm) the pseudo-labeler, so this pilot's guarantee is EMPIRICAL:
high-trial coverage is the check. The exact-guarantee version is the FCP
integration (pilot D), gated on this pilot showing an efficiency signal.

Run from repo root (embeddings live in the main checkout's output/):
python src/mdcp_pool_pilot.py \
    --embeddings_path output/from_cluster/embeddings_cifar100.pt \
    --unlabeled_path  output/from_cluster/embeddings_cifar100_unlabeled.pt \
    --cal_sizes 200 400 800 --splits balanced_both random --n_trials 20
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from scipy.spatial import cKDTree

from exchangeable_features import make_transform
from conformal_prediction import PrototypeSoftmaxNCM

EPS = 1e-8
D_CAP = 1e9  # purity when no true neighbor found (T_count = 0)


# ---------------------------------------------------------------------------
# Dimension scores (vectorized; conventions mirror the repo NCMs)
# ---------------------------------------------------------------------------

def _l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-10)


def geodesic_asym_scores(Zq, Zcal, y_cal, classes, k, self_is_cal=False):
    """(nq, K) geodesic asym scores: arccos(1-NN same cos)/arccos(mean top-k
    other cos). Missing class -> d_same = arccos(-1) = pi (symmetric missing-
    class convention). self_is_cal=True excludes the diagonal (cal-vs-cal LOO).
    """
    Zqn, Zcn = _l2n(Zq), _l2n(Zcal)
    S = Zqn @ Zcn.T                                    # (nq, m)
    if self_is_cal:
        np.fill_diagonal(S, -np.inf)
    nq, m = S.shape
    K = len(classes)

    # --- per-class max via reduceat over class-sorted columns ---
    order = np.argsort(y_cal, kind="stable")
    S_ord = S[:, order]
    y_ord = y_cal[order]
    starts = np.searchsorted(y_ord, classes)           # class block starts
    present = np.searchsorted(y_ord, classes, side="right") > starts
    max_same = np.full((nq, K), -1.0)
    if present.any():
        red = np.maximum.reduceat(S_ord, starts[present], axis=1)
        max_same[:, present] = red
    max_same[~np.isfinite(max_same)] = -1.0            # all-blocked block (LOO singleton)
    d_same = np.arccos(np.clip(max_same, -1.0, 1.0))   # (nq, K)

    # --- other-class mean top-k via global top-(k + max_class_count) trick ---
    counts = np.array([(y_cal == c).sum() for c in classes])
    kk = min(m, k + int(counts.max()))
    part = np.argpartition(S, -kk, axis=1)[:, -kk:]    # (nq, kk) global top-kk cols
    top_vals = np.take_along_axis(S, part, axis=1)
    top_lbls = y_cal[part]
    d_other = np.empty((nq, K))
    for j, c in enumerate(classes):
        vals = np.where(top_lbls != c, top_vals, -np.inf)
        ke = min(k, m - counts[j])
        if ke <= 0:
            d_other[:, j] = np.pi / 2
            continue
        tk = np.partition(vals, -ke, axis=1)[:, -ke:]
        tk_sum = np.where(np.isfinite(tk), tk, 0.0).sum(axis=1)
        tk_cnt = np.isfinite(tk).sum(axis=1)
        d_other[:, j] = np.arccos(np.clip(tk_sum / np.maximum(tk_cnt, 1), -1.0, 1.0))
    return d_same / (d_other + EPS)


def prototype_lac_scores(Zq, Zcal, y_cal, classes, T, loo=False):
    """(nq, K) prototype cosine LAC scores 1 - p(q|z), PrototypeSoftmaxNCM
    conventions: L2-normalized features + prototypes, fixed temperature T,
    missing/empty class -> logit -inf -> p=0 -> score 1. loo=True applies the
    closed-form leave-one-out class-mean to each cal point's own class."""
    Zqn, Zcn = _l2n(Zq), _l2n(Zcal)
    K = len(classes)
    d = Zcn.shape[1]
    cls_sum = np.zeros((K, d))
    n_c = np.zeros(K)
    for j, c in enumerate(classes):
        idx = (y_cal == c)
        n_c[j] = idx.sum()
        if n_c[j]:
            cls_sum[j] = Zcn[idx].sum(axis=0)
    ok = n_c > 0
    Mu = np.zeros((K, d))
    Mu[ok] = _l2n(cls_sum[ok] / n_c[ok, None])
    logits = Zqn @ Mu.T / T
    logits[:, ~ok] = -np.inf

    if loo:
        col = np.searchsorted(classes, y_cal)
        n_own = n_c[col]
        loo_vec = cls_sum[col] - Zqn                    # Zq IS Zcal here
        single = n_own <= 1
        loo_vec[~single] /= (n_own[~single, None] - 1)
        loo_dir = _l2n(loo_vec)
        own = (Zqn * loo_dir).sum(axis=1) / T
        own[single] = -np.inf
        logits[np.arange(len(Zqn)), col] = own

    mx = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - mx)
    p = e / e.sum(axis=1, keepdims=True)
    return 1.0 - p


# ---------------------------------------------------------------------------
# Rank transform + purity function
# ---------------------------------------------------------------------------

def ecdf_fit(pool_scores_flat):
    return np.sort(pool_scores_flat.ravel())


def ecdf_apply(ref_sorted, scores):
    return np.searchsorted(ref_sorted, scores, side="right") / (len(ref_sorted) + 1.0)


class PurityD:
    """MDCP purity function over the pool score cloud.

    mode="count": faithful MDCP purity D(s) = (F+T)/T among the k_d nearest
        cloud points (their Voronoi cells = the 1-NN special case). FAILS in
        the tail: when a query has zero true neighbors D saturates at D_CAP,
        and once >alpha of cal true scores are capped the quantile IS the cap
        -> sets collapse to all K classes (observed with pseudo-label clouds).
    mode="ratio" (default): smooth kNN density-ratio
        D(s) = dist to k_d-th nearest TRUE point / dist to k_d-th nearest
        FALSE point. Monotone in local false-vs-true density, never saturates
        -- the geodesic d_same/d_other ratio transplanted into score space.
    """

    def __init__(self, cloud, is_true, k_d, mode="ratio"):
        self.mode = mode
        self.k_d = k_d
        if mode == "count":
            self.tree = cKDTree(cloud)
            self.is_true = is_true.astype(np.float64)
        else:
            self.tree_t = cKDTree(cloud[is_true])
            self.tree_f = cKDTree(cloud[~is_true])

    def __call__(self, queries):
        if self.mode == "count":
            _, idx = self.tree.query(queries, k=self.k_d, workers=-1)
            if self.k_d == 1:
                idx = idx[:, None]
            t = self.is_true[idx].sum(axis=1)
            return np.where(t > 0, self.k_d / np.maximum(t, 1e-12), D_CAP)
        d_t, _ = self.tree_t.query(queries, k=self.k_d, workers=-1)
        d_f, _ = self.tree_f.query(queries, k=self.k_d, workers=-1)
        if self.k_d > 1:
            d_t, d_f = d_t[:, -1], d_f[:, -1]
        return d_t / (d_f + EPS)


def build_cloud(S_pool_dims, yhat_col):
    """Stack pool score vectors for all classes; flag the pseudo-true ones.
    S_pool_dims: (n_pool, K, n_dims) rank-space scores. Returns (cloud, is_true)."""
    n_pool, K, n_dims = S_pool_dims.shape
    cloud = S_pool_dims.reshape(n_pool * K, n_dims)
    is_true = np.zeros(n_pool * K, dtype=bool)
    is_true[np.arange(n_pool) * K + yhat_col] = True
    return cloud, is_true


# ---------------------------------------------------------------------------
# Conformal wrapper
# ---------------------------------------------------------------------------

def scp_eval(cal_true_scores, test_scores, y_test_col, alpha):
    """Standard split-CP ceil quantile + evaluation. test_scores: (n_test, K)."""
    r = len(cal_true_scores)
    qlevel = np.ceil((r + 1) * (1 - alpha)) / r
    qhat = np.quantile(cal_true_scores, min(qlevel, 1.0), method="higher")
    sets = test_scores <= qhat
    cov = sets[np.arange(len(sets)), y_test_col].mean()
    return float(cov), float(sets.sum(axis=1).mean()), float((sets.sum(axis=1) == 0).mean())


# ---------------------------------------------------------------------------
# Splits (conventions copied from exchangeable_fcp_experiment.py)
# ---------------------------------------------------------------------------

def balanced_both_split(y, classes, cal, rng):
    m = cal // len(classes)
    ci, ti = [], []
    for c in classes:
        perm = rng.permutation(np.where(y == c)[0])
        ci.append(perm[:m]); ti.append(perm[m:2 * m])
    return np.concatenate(ci), np.concatenate(ti)


def random_split(y, classes, cal, rng, test=1000):
    idx = rng.permutation(len(y))
    return idx[:cal], idx[cal:cal + test]


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def run_trial(Z, y, Zu, yu_true, classes, cal_size, split, alpha, k_d, T_proto, rng,
              purity_mode="ratio"):
    if split == "balanced_both":
        ci, ti = balanced_both_split(y, classes, cal_size, rng)
    else:
        ci, ti = random_split(y, classes, cal_size, rng)
    Zc, yc = Z[ci], y[ci]
    Zt, yt = Z[ti], y[ti]
    K = len(classes)
    y_test_col = np.searchsorted(classes, yt)
    k_geo = max(1, min(5, len(yc) // K))

    # --- dimension scores (cal-anchored) for pool / cal / test ---
    dims = {}
    dims["geo"] = dict(
        pool=geodesic_asym_scores(Zu, Zc, yc, classes, k_geo),
        cal=geodesic_asym_scores(Zc, Zc, yc, classes, k_geo, self_is_cal=True),
        test=geodesic_asym_scores(Zt, Zc, yc, classes, k_geo),
    )
    dims["proto"] = dict(
        pool=prototype_lac_scores(Zu, Zc, yc, classes, T_proto),
        cal=prototype_lac_scores(Zc, Zc, yc, classes, T_proto, loo=True),
        test=prototype_lac_scores(Zt, Zc, yc, classes, T_proto),
    )

    # --- rank space (ECDF fit on the pool cloud, per dimension) ---
    rank = {}
    for name, d in dims.items():
        ref = ecdf_fit(d["pool"])
        rank[name] = {part: ecdf_apply(ref, d[part]) for part in ("pool", "cal", "test")}

    cal_rows = np.arange(len(yc))
    cal_col = np.searchsorted(classes, yc)
    res = {}

    # --- raw 1-D arms ---
    for name in ("geo", "proto"):
        cal_true = dims[name]["cal"][cal_rows, cal_col]
        res[f"raw1_{name}"] = scp_eval(cal_true, dims[name]["test"], y_test_col, alpha)

    # --- pseudo-labels for the pool cloud ---
    yhat = {
        "oracle": np.searchsorted(classes, yu_true),
        "fullcal": np.argmin(dims["proto"]["pool"], axis=1),  # argmax posterior = argmin LAC
    }

    # --- D-ratio arms ---
    def dratio(dim_names, yhat_col):
        S_pool = np.stack([rank[n]["pool"] for n in dim_names], axis=-1)   # (n_pool, K, D)
        cloud, is_true = build_cloud(S_pool, yhat_col)
        D = PurityD(cloud, is_true, k_d, mode=purity_mode)
        q_cal = np.stack([rank[n]["cal"][cal_rows, cal_col] for n in dim_names], axis=-1)
        cal_true = D(q_cal)
        n_t, Kc = rank[dim_names[0]]["test"].shape
        q_test = np.stack([rank[n]["test"].ravel() for n in dim_names], axis=-1)
        test_D = D(q_test).reshape(n_t, Kc)
        return scp_eval(cal_true, test_D, y_test_col, alpha)

    res["dratio1_geo"] = dratio(("geo",), yhat["fullcal"])
    res["dratio1_proto"] = dratio(("proto",), yhat["fullcal"])
    res["dratio2_oracle"] = dratio(("geo", "proto"), yhat["oracle"])
    res["dratio2_fullcal"] = dratio(("geo", "proto"), yhat["fullcal"])
    res["_yhat_acc_fullcal"] = float((yhat["fullcal"] == yhat["oracle"]).mean())
    return res


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MDCP pool pilot (score-space k-NN density-ratio NCM)")
    ap.add_argument("--embeddings_path", required=True)
    ap.add_argument("--unlabeled_path", required=True)
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--splits", nargs="+", default=["balanced_both", "random"])
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--k_d", type=int, default=10)
    ap.add_argument("--purity", choices=["ratio", "count"], default="ratio")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--whiten", default="cluster")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="output/mdcp_pool_pilot")
    args = ap.parse_args()

    d = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)
    X, y = d["embeddings"].numpy().astype(np.float64), d["labels"].numpy()
    du = torch.load(args.unlabeled_path, map_location="cpu", weights_only=False)
    Xu, yu = du["embeddings"].numpy().astype(np.float64), du["labels"].numpy()
    classes = np.unique(y)
    print(f"labeled {X.shape}, pool {Xu.shape}, K={len(classes)}")

    tr = make_transform(unlabeled=Xu, pca_dim=args.pca_dim, whiten=args.whiten)
    Z, Zu = tr.transform(X), tr.transform(Xu)

    # pilot-fixed prototype temperature (repo convention: one stable balanced
    # draw at m=8/class, fixed seed -> constant across all trials)
    rng0 = np.random.default_rng(args.seed)
    ci0, _ = balanced_both_split(y, classes, 8 * len(classes), rng0)
    pncm = PrototypeSoftmaxNCM(temperature=None, logit="cosine",
                               allow_nonexchangeable=True)
    pncm.fit(Z[ci0], y[ci0])
    T_proto = float(pncm._T)
    print(f"pilot-fixed prototype T = {T_proto:.4f}")

    results = {}
    for split in args.splits:
        for cal in args.cal_sizes:
            arms = {}
            t0 = time.time()
            for t in range(args.n_trials):
                rng = np.random.default_rng(args.seed + 1000 * t + cal)
                r = run_trial(Z, y, Zu, yu, classes, cal, split, args.alpha,
                              args.k_d, T_proto, rng, purity_mode=args.purity)
                for k, v in r.items():
                    arms.setdefault(k, []).append(v)
            key = f"{split}_cal{cal}"
            results[key] = {}
            print(f"\n== {key}  ({args.n_trials} trials, {time.time()-t0:.0f}s) ==")
            print(f"{'arm':<18}{'cov':>8}{'+-':>7}{'size':>9}{'+-':>7}{'empty%':>8}")
            for arm, vals in arms.items():
                if arm.startswith("_"):
                    m = float(np.mean(vals))
                    results[key][arm] = m
                    print(f"{arm:<18}{m:>8.3f}")
                    continue
                a = np.array(vals)  # (trials, 3): cov, size, empty
                cov, sz, emp = a[:, 0], a[:, 1], a[:, 2]
                results[key][arm] = dict(
                    cov=float(cov.mean()), cov_se=float(cov.std(ddof=1) / np.sqrt(len(cov))),
                    size=float(sz.mean()), size_se=float(sz.std(ddof=1) / np.sqrt(len(sz))),
                    empty=float(emp.mean()))
                print(f"{arm:<18}{cov.mean():>8.3f}{cov.std(ddof=1)/np.sqrt(len(cov)):>7.3f}"
                      f"{sz.mean():>9.2f}{sz.std(ddof=1)/np.sqrt(len(sz)):>7.2f}{100*emp.mean():>8.1f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "mdcp_pool_pilot_results.json")
    with open(out, "w") as f:
        json.dump(dict(config=vars(args), results=results), f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
