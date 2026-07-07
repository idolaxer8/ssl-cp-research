"""
Pilot D: FULL-CP MDCP -- exact transductive score-space purity (approved plan
next-while-it-extracting-plan-steady-squirrel.md).

For test point x and candidate label y the bag is B_y = cal + {(x,y)}. Every
bag member gets ONE symmetric scalar:

    D(i) = dist to k_d-th nearest OTHER bag-TRUE score vector
           ------------------------------------------------- + eps-guard
           dist to k_d-th nearest FALSE-cloud vector

where the score vector of member i stacks n dimensions s_d(i) (prototype LAC
with leave-self-out class means, PrototypeSoftmaxNCM conventions: cosine
logits / fixed T / empty-class -> LAC 1). FALSE cloud arms (both exact):
  pool   -- every pool score vector s_d(u, q | B_y), unlabeled, NO
            pseudo-labels (contamination exactly 1/K);
  bag    -- the bag's own (m+1)(K-1) wrong-label vectors;
  count  -- instructor-faithful Voronoi purity over the bag cloud:
            centers = unique bag-true vectors, D = (F+T)/T of own cell.
p(y) = (1 + #{i in cal : D_i >= D_cand}) / (m+1); set = {y : p > alpha}.

EXACTNESS. All caches are cal-only; per candidate the class-y prototype gains
x, which moves (a) class-y members' own-class logits, (b) EVERYONE's softmax
normalizer through column y (handled by the incremental-Z trick:
Z_new = Z - exp(f_y_old) + exp(f_y_new)), (c) pool column y. The test point's
own logit row faces cal-only class means for every column, so it is
candidate-independent. reference_bag_D() recomputes everything from the bag
with NO cal/test asymmetry; tests/test_mdcp_full_cp.py asserts the engine
path is identical to it (the symmetry oracle).

Run:
python src/mdcp_full_cp.py \
    --embeddings_path .../embeddings_cifar100_layers.pt \
    --unlabeled_path  .../embeddings_cifar100_unlabeled_layers.pt \
    --dims proto:final__pca128_cw proto:final__pca32_cw \
    --cal_sizes 200 --splits balanced_both random --n_trials 20 --n_test 100
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import PrototypeSoftmaxNCM
from mdcp_pool_pilot import (balanced_both_split, random_split,
                             load_embedding_sources, build_view_feats,
                             prototype_lac_scores)

EPS = 1e-12


def _l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-10)


# ---------------------------------------------------------------------------
# Dimension: prototype LAC with bag-symmetric LOO (the BagDim contract).
# Batched-per-test-point API (deviation from the plan's per-candidate Delta,
# same semantics): fit_trial caches cal-only state; test_products(x) returns
# everything needed to realize ALL K candidate bags at once.
# ---------------------------------------------------------------------------

class PrototypeLACDim:
    """One score dimension: prototype cosine LAC on a feature view.

    Conventions (PrototypeSoftmaxNCM): features and class means L2-normalized,
    fixed temperature T, member's own-class column uses the leave-self-out
    mean, empty class -> logit -inf -> prob 0 -> LAC 1.
    """

    def __init__(self, view, T):
        self.view = view
        self.T = float(T)

    def fit_trial(self, Zc, yc, Zpool, classes):
        self.K = len(classes)
        self.classes = classes
        self.ycol = np.searchsorted(classes, yc)
        self.Zn = _l2n(Zc)                       # (m, d)
        self.Pn = _l2n(Zpool)                    # (n_pool, d)
        m, d = self.Zn.shape
        self.S = np.zeros((self.K, d))
        self.n_c = np.zeros(self.K)
        for j in range(self.K):
            idx = self.ycol == j
            self.n_c[j] = idx.sum()
            if self.n_c[j]:
                self.S[j] = self.Zn[idx].sum(axis=0)
        ok = self.n_c > 0
        self.Mu = np.zeros((self.K, d))
        self.Mu[ok] = _l2n(self.S[ok] / self.n_c[ok, None])
        self.ok = ok

        # cal logits, own-class column = LOO mean (empty after LOO -> -inf)
        L = self.Zn @ self.Mu.T
        L[:, ~ok] = -np.inf
        rows = np.arange(m)
        loo_vec = self.S[self.ycol] - self.Zn
        n_own = self.n_c[self.ycol]
        single = n_own <= 1
        denom = np.maximum(n_own - 1, 1)[:, None]
        loo_dir = _l2n(loo_vec / denom)
        own = (self.Zn * loo_dir).sum(axis=1)
        own[single] = -np.inf
        L[rows, self.ycol] = own
        self.L = L / self.T                      # (m, K) scaled logits
        self.E = np.exp(self.L)                  # exp(-inf) = 0
        self.Zrow = self.E.sum(axis=1)

        # pool logits vs cal-only means
        PL = self.Pn @ self.Mu.T
        PL[:, ~ok] = -np.inf
        self.PL = PL / self.T
        self.PE = np.exp(self.PL)
        self.PZ = self.PE.sum(axis=1)
        return self

    def test_products(self, x):
        """Everything candidate-dependent, for all K candidates at once."""
        xn = _l2n(x.reshape(1, -1))[0]
        # test row: every column faces cal-only means (bag \ {test} = cal)
        tl = xn @ self.Mu.T
        tl[~self.ok] = -np.inf
        tl = tl / self.T
        tE = np.exp(tl)
        tZ = tE.sum()
        t_probs = tE / max(tZ, EPS)              # (K,) candidate-independent

        # new class-c mean when the candidate joins class c: normalize(S_c + xn)
        MuNew = _l2n(self.S + xn[None, :])       # (K, d) valid even if n_c = 0
        C = (self.Zn @ MuNew.T) / self.T         # (m, K): member i vs new mu_c
        # class-y members' own-class LOO with the candidate inside:
        # mu = normalize(S_y - z_i + xn)  (defined even for singletons: = xn)
        rows = np.arange(len(self.Zn))
        spec_vec = _l2n(self.S[self.ycol] - self.Zn + xn[None, :])
        C[rows, self.ycol] = (self.Zn * spec_vec).sum(axis=1) / self.T
        Cexp = np.exp(C)

        PC = (self.Pn @ MuNew.T) / self.T        # (n_pool, K)
        PCexp = np.exp(PC)
        return dict(t_probs=t_probs, Cexp=Cexp, PCexp=PCexp)


# ---------------------------------------------------------------------------
# Reference implementation: score a bag with NO cal/test asymmetry at all.
# Used by the symmetry-oracle test and as the ground truth for the engine.
# ---------------------------------------------------------------------------

def bag_prob_rows(Z_bag, y_bag, T, K):
    """(m+1, K) prob rows, one symmetric rule per member (own-class LOO)."""
    return 1.0 - prototype_lac_scores(Z_bag, Z_bag, y_bag, np.arange(K), T, loo=True)


def reference_bag_D(Z_bag_views, y_bag, Zpool_views, Ts, k_d, arm,
                    pool_pairs=None):
    """D for every bag member, straight from the definitions.
    Z_bag_views/Zpool_views: list per dim of (n, d_view) arrays; Ts: per-dim T.
    arm: 'pool' | 'bag' | 'count'."""
    K = int(max(y_bag.max() + 1, 2))
    n = len(y_bag)
    rows_list, pool_list = [], []
    for Zb, Zp, T in zip(Z_bag_views, Zpool_views, Ts):
        P = bag_prob_rows(Zb, y_bag, T, K)               # (n, K)
        rows_list.append(1.0 - P)                        # LAC scores
        # pool vs FULL-BAG class means (symmetric: bag as a set)
        Zbn, Zpn = _l2n(Zb), _l2n(Zp)
        S = np.zeros((K, Zbn.shape[1])); cnt = np.zeros(K)
        for j in range(K):
            idx = y_bag == j
            cnt[j] = idx.sum()
            if cnt[j]:
                S[j] = Zbn[idx].sum(axis=0)
        ok = cnt > 0
        Mu = np.zeros_like(S); Mu[ok] = _l2n(S[ok] / cnt[ok, None])
        PL = (Zpn @ Mu.T); PL[:, ~ok] = -np.inf
        PP = np.exp(PL / T); PP = PP / np.maximum(PP.sum(1, keepdims=True), EPS)
        pool_list.append(1.0 - PP)                       # (n_pool, K) LAC
    nd = len(rows_list)
    idx = np.arange(n)
    true_cloud = np.stack([r[idx, y_bag] for r in rows_list], axis=-1)  # (n, nd)

    # numerator: k_d-th nearest OTHER true vector
    dt = np.linalg.norm(true_cloud[:, None, :] - true_cloud[None, :, :], axis=-1)
    np.fill_diagonal(dt, np.inf)
    kk = min(k_d, n - 1)
    num = np.sort(dt, axis=1)[:, kk - 1]

    if arm == "count":
        # Voronoi purity over the bag cloud (instructor-faithful)
        centers, inv, tcnt = np.unique(true_cloud, axis=0, return_inverse=True,
                                       return_counts=True)
        false_pts = _bag_false_cloud(rows_list, y_bag)
        dc = np.linalg.norm(false_pts[:, None, :] - centers[None, :, :], axis=-1)
        assign = dc.argmin(axis=1)
        F = np.bincount(assign, minlength=len(centers))
        Dcell = (F + tcnt) / tcnt
        return Dcell[inv]

    if arm == "bag":
        false_pts = _bag_false_cloud(rows_list, y_bag)
    else:  # pool
        pairs = pool_pairs if pool_pairs is not None else _all_pairs(pool_list[0].shape)
        false_pts = np.stack([p[pairs[0], pairs[1]] for p in pool_list], axis=-1)
    df = np.linalg.norm(true_cloud[:, None, :] - false_pts[None, :, :], axis=-1)
    kf = min(k_d, false_pts.shape[0])
    den = np.sort(df, axis=1)[:, kf - 1]
    return num / (den + 1e-9)


def _bag_false_cloud(rows_list, y_bag):
    n, K = rows_list[0].shape
    mask = np.ones((n, K), dtype=bool)
    mask[np.arange(n), y_bag] = False
    return np.stack([r[mask] for r in rows_list], axis=-1)   # (n*(K-1), nd)


def _all_pairs(shape):
    n_pool, K = shape
    u = np.repeat(np.arange(n_pool), K)
    q = np.tile(np.arange(K), n_pool)
    return u, q


# ---------------------------------------------------------------------------
# Engine: cached cal-only state + per-candidate incremental realization.
# torch for the distance kernels; numpy for the probability algebra.
# ---------------------------------------------------------------------------

class FullCPMDCP:
    def __init__(self, dims, k_d=10, alpha=0.1, arms=("pool", "bag", "count"),
                 pool_subsample=100_000, device="cpu", seed=0):
        self.dims = dims
        self.k_d = k_d
        self.alpha = alpha
        self.arms = arms
        self.pool_subsample = pool_subsample
        self.device = torch.device(device)
        self.seed = seed

    def fit_trial(self, feats_cal, yc, feats_pool, classes):
        self.K = len(classes)
        self.m = len(yc)
        self.ycol = np.searchsorted(classes, yc)
        for d, Zc, Zp in zip(self.dims, feats_cal, feats_pool):
            d.fit_trial(Zc, yc, Zp, classes)
        n_pool = self.dims[0].Pn.shape[0]
        rng = np.random.default_rng(self.seed)
        total = n_pool * self.K
        take = min(self.pool_subsample, total)
        flat = rng.choice(total, size=take, replace=False)
        self.pool_pairs = (flat // self.K, flat % self.K)     # fixed per trial
        # bag-false mask over cal rows (test column dropped per candidate)
        self.cal_false_mask = np.ones((self.m, self.K), dtype=bool)
        self.cal_false_mask[np.arange(self.m), self.ycol] = False
        return self

    def point_state(self, x_feats):
        """Candidate-independent state for one test point (torch tensors)."""
        prods = [d.test_products(x) for d, x in zip(self.dims, x_feats)]
        dev = self.device
        st = dict(
            E=[torch.as_tensor(d.E, device=dev) for d in self.dims],
            Zr=[torch.as_tensor(d.Zrow, device=dev) for d in self.dims],
            Cx=[torch.as_tensor(p["Cexp"], device=dev) for p in prods],
            tpr=[torch.as_tensor(p["t_probs"], device=dev) for p in prods],
            PE=[torch.as_tensor(d.PE, device=dev) for d in self.dims],
            PZ=[torch.as_tensor(d.PZ, device=dev) for d in self.dims],
            PCx=[torch.as_tensor(p["PCexp"], device=dev) for p in prods],
            ycol=torch.as_tensor(self.ycol, device=dev),
            rows=torch.arange(self.m, device=dev),
            u_idx=torch.as_tensor(self.pool_pairs[0], device=dev),
            q_idx=torch.as_tensor(self.pool_pairs[1], device=dev),
            calF_mask=torch.as_tensor(self.cal_false_mask, device=dev),
        )
        return st

    def candidate_D(self, st, y):
        """D for every bag member under candidate label y, per arm.
        Member order: cal[0..m-1], then the test point. This is the single
        code path the symmetry-oracle test checks against reference_bag_D."""
        m, kd = self.m, self.k_d
        out = {}
        true_c, calP, poolF = [], [], []
        for di in range(len(self.dims)):
            Znew = st["Zr"][di] - st["E"][di][:, y] + st["Cx"][di][:, y]
            own = torch.where(st["ycol"] == y, st["Cx"][di][:, y],
                              st["E"][di][st["rows"], st["ycol"]]) / Znew
            true_c.append(torch.cat([1.0 - own,
                                     (1.0 - st["tpr"][di][y]).reshape(1)]))
            if "bag" in self.arms or "count" in self.arms:
                Pfull = st["E"][di] / Znew[:, None]
                Pfull[:, y] = st["Cx"][di][:, y] / Znew
                calP.append(Pfull)
            if "pool" in self.arms:
                PZnew = (st["PZ"][di] - st["PE"][di][:, y] + st["PCx"][di][:, y])
                pv = torch.where(st["q_idx"] == y,
                                 st["PCx"][di][st["u_idx"], y],
                                 st["PE"][di][st["u_idx"], st["q_idx"]]) \
                    / PZnew[st["u_idx"]]
                poolF.append(1.0 - pv)
        T_cloud = torch.stack(true_c, dim=-1)                   # (m+1, nd)

        dt = torch.cdist(T_cloud, T_cloud)
        dt.fill_diagonal_(float("inf"))
        num = dt.kthvalue(min(kd, m), dim=1).values

        if "bag" in self.arms or "count" in self.arms:
            calF = torch.stack(
                [torch.cat([P[st["calF_mask"]], self._test_false(st["tpr"][di], y)])
                 for di, P in enumerate(calP)], dim=-1)
            calF = 1.0 - calF                                    # probs -> LAC
        if "bag" in self.arms:
            den = torch.cdist(T_cloud, calF).kthvalue(
                min(kd, calF.shape[0]), dim=1).values
            out["bag"] = num / (den + 1e-9)
        if "count" in self.arms:
            out["count"] = self._count_D(T_cloud, calF)
        if "pool" in self.arms:
            Fp = torch.stack(poolF, dim=-1)
            den = torch.cdist(T_cloud, Fp).kthvalue(
                min(kd, Fp.shape[0]), dim=1).values
            out["pool"] = num / (den + 1e-9)
        return out

    def predict_point(self, x_feats):
        """All K candidate p-values for one test point, per arm."""
        st = self.point_state(x_feats)
        p_out = {arm: np.zeros(self.K) for arm in self.arms}
        for y in range(self.K):
            D = self.candidate_D(st, y)
            for arm in self.arms:
                p_out[arm][y] = self._pvalue(D[arm])
        return p_out

    def _test_false(self, t_probs, y):
        mask = torch.ones(self.K, dtype=torch.bool, device=t_probs.device)
        mask[y] = False
        return t_probs[mask]

    def _count_D(self, T_cloud, F_pts):
        centers, inv, tcnt = torch.unique(T_cloud, dim=0, return_inverse=True,
                                          return_counts=True)
        assign = torch.cdist(F_pts, centers).argmin(dim=1)
        F = torch.bincount(assign, minlength=len(centers)).to(tcnt.dtype)
        Dcell = (F + tcnt) / tcnt
        return Dcell[inv]

    def _pvalue(self, D):
        d_cal, d_cand = D[:-1], D[-1]
        return float((1 + (d_cal >= d_cand).sum().item()) / (self.m + 1))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def pilot_T(Zview, y, classes, seed=0):
    rng = np.random.default_rng(seed)
    ci, _ = balanced_both_split(y, classes, 8 * len(classes), rng)
    p = PrototypeSoftmaxNCM(temperature=None, logit="cosine",
                            allow_nonexchangeable=True)
    p.fit(Zview[ci], y[ci])
    return float(p._T)


def main():
    ap = argparse.ArgumentParser(description="Full-CP MDCP (pilot D)")
    ap.add_argument("--embeddings_path", required=True)
    ap.add_argument("--unlabeled_path", required=True)
    ap.add_argument("--dims", nargs="+",
                    default=["proto:final__pca128_cw", "proto:final__pca32_cw"])
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200])
    ap.add_argument("--splits", nargs="+", default=["balanced_both", "random"])
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=100,
                    help="test points per trial (FCP cost scales with n_test*K)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--k_d", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=["pool", "bag", "count"])
    ap.add_argument("--pool_subsample", type=int, default=100_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default="output/mdcp_pool_pilot/full_cp")
    args = ap.parse_args()

    X_src, y = load_embedding_sources(args.embeddings_path)
    Xu_src, _ = load_embedding_sources(args.unlabeled_path)
    classes = np.unique(y)
    specs = [s.split(":") for s in args.dims]
    assert all(ncm == "proto" for ncm, _ in specs), \
        "pilot D implements PrototypeLACDim only (plan scope)"
    views = [v for _, v in specs]
    feats = build_view_feats(X_src, Xu_src, views)
    T_by_view = {v: pilot_T(feats[v][0], y, classes, args.seed) for v in set(views)}
    print("dims:", args.dims, "| T:", {v: round(t, 4) for v, t in T_by_view.items()})

    results = {}
    for split in args.splits:
        for cal in args.cal_sizes:
            per_arm = {a: [] for a in args.arms}
            t0 = time.time()
            for t in range(args.n_trials):
                rng = np.random.default_rng(args.seed + 1000 * t + cal)
                if split == "balanced_both":
                    ci, ti = balanced_both_split(y, classes, cal, rng)
                else:
                    ci, ti = random_split(y, classes, cal, rng)
                ti = ti[rng.permutation(len(ti))[:args.n_test]]
                dims = [PrototypeLACDim(v, T_by_view[v]) for v in views]
                eng = FullCPMDCP(dims, k_d=args.k_d, alpha=args.alpha,
                                 arms=tuple(args.arms),
                                 pool_subsample=args.pool_subsample,
                                 device=args.device,
                                 seed=args.seed + 7000 * t + cal)
                eng.fit_trial([feats[v][0][ci] for v in views], y[ci],
                              [feats[v][1] for v in views], classes)
                y_test_col = np.searchsorted(classes, y[ti])
                cov = {a: 0 for a in args.arms}
                size = {a: 0 for a in args.arms}
                for k, tidx in enumerate(ti):
                    p_by_arm = eng.predict_point([feats[v][0][tidx] for v in views])
                    for a in args.arms:
                        inset = p_by_arm[a] > args.alpha
                        cov[a] += bool(inset[y_test_col[k]])
                        size[a] += int(inset.sum())
                n = len(ti)
                for a in args.arms:
                    per_arm[a].append((cov[a] / n, size[a] / n))
                if t == 0:
                    print(f"  [{split} cal={cal}] trial 1/{args.n_trials}: "
                          f"{(time.time()-t0):.0f}s, "
                          + "  ".join(f"{a}: cov {cov[a]/n:.3f} sz {size[a]/n:.2f}"
                                      for a in args.arms))
            key = f"{split}_cal{cal}"
            results[key] = {}
            print(f"\n== FULL-CP {key}  ({args.n_trials} trials x {args.n_test} test, "
                  f"{time.time()-t0:.0f}s) ==")
            print(f"{'arm':<12}{'cov':>8}{'+-':>7}{'size':>9}{'+-':>7}")
            for a in args.arms:
                arr = np.array(per_arm[a])
                c, s = arr[:, 0], arr[:, 1]
                results[key][f"fcp_{a}"] = dict(
                    cov=float(c.mean()), cov_se=float(c.std(ddof=1) / np.sqrt(len(c))),
                    size=float(s.mean()), size_se=float(s.std(ddof=1) / np.sqrt(len(s))))
                print(f"{'fcp_'+a:<12}{c.mean():>8.3f}{c.std(ddof=1)/np.sqrt(len(c)):>7.3f}"
                      f"{s.mean():>9.2f}{s.std(ddof=1)/np.sqrt(len(s)):>7.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "mdcp_full_cp_results.json")
    with open(out, "w") as f:
        json.dump(dict(config=vars(args), results=results), f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
