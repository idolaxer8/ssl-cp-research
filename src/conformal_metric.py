"""Conformal metric learning (G1): learn the NCM's metric on the unlabeled
pool by optimizing a pool-only surrogate of conformal set size.

The transform
-------------
    T(x) = ClusterWhiten( diag(s) @ V^T @ (x - mu) )

  mu, V, lambda_j : pool eigenbasis (all 768 dims; Stage 0)
  s in R^768      : LEARNED per-dimension scales -- a soft spectral filter
                    subsuming hard truncation (s -> 0 on the tail) and
                    spectral reweighting (Stage 1)
  ClusterWhiten   : the existing pool-fit within-cluster whitening, re-fit on
                    the FILTERED pool (diag 'cluster' or full-matrix
                    'lw_cluster'; Stage 2, exchangeable_features.py unchanged)

Rung 1: s constrained to a 3-parameter spectral-filter family
            s_j = sigmoid((j0 - j)/w) * lambda_j^gamma
        (gate = WHICH directions participate, continuous d'; power = HOW
        strongly each is rescaled vs its pool variance, continuous whitening
        strength; filter-function view per Lo Gerfo/Rosasco 2008), optimized
        by grid + Nelder-Mead on the rehearsal-set-size pool objective.
Rung 2: free diagonal s (768 params), torch Adam on a ConfTr-style
        differentiable soft set size (Stutz et al., ICLR 2022 -- smoothed
        quantile + sigmoid set membership; our delta: we fit the METRIC on
        the unlabeled pseudo-labeled pool, not a classifier on labels, and
        full-CP exactness survives by Prop 2), TV smoothness + anchor to the
        rung-1 solution, early stopping on a pool-internal validation half.

Exchangeability: `fit_conformal_metric` sees ONLY the pool + a-priori
constants (K = label-space size, alpha, frozen CFG). Every internal selection
(grid, NM, early stopping, TV weight, rung showdown) is decided on pool half
B2. The shipped map pool -> (s, whiten mode) is one deterministic function,
so Proposition 2 (theory.md sec 2) gives EXACT coverage -- zero cost.
"""
import hashlib
import json
import os
import sys
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exchangeable_features import UnlabeledTransform  # noqa: E402
from pool_objective import (l2n, pseudo_task, class_means, centroid_dists,  # noqa: E402
                            score_matrix, margin_stats, rehearsal_setsize,
                            objective_on_half)

# Pre-registered constants (frozen BEFORE any cal/test data is looked at; the
# report records a hash of this dict so refits are checkable).
CFG = dict(
    seed=0,
    n_clusters_whiten=100,
    reg=1e-4,
    n_rep=20,
    grid_j0=[16, 32, 64, 128, 256, 512, 768],
    grid_w=[0.5, 8, 32, 128],
    grid_gamma=[-0.5, -0.25, 0.0, 0.25, 0.5],
    whiten_modes=["cluster", "lw_cluster"],
    nm_maxiter=150,
    n_real_finalists=8,
    lw_subsample=2000,
    # rung 2
    r2_lr=1e-2, r2_steps=1500, r2_eval_every=25, r2_patience=300,
    r2_tau_q=0.05, r2_tau_set_hi=0.2, r2_tau_set_lo=0.02, r2_anneal_steps=1000,
    r2_lam_tv=[0.003, 0.03, 0.3], r2_anchor=0.01,
)


def cfg_hash(cfg):
    return hashlib.sha1(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def pool_sha1(Xu):
    return hashlib.sha1(np.ascontiguousarray(
        Xu, dtype=np.float64).tobytes()).hexdigest()[:12]


def pool_eigenbasis(X):
    """Full-rank pool eigenbasis: mean mu, eigenvector columns V (sorted by
    descending eigenvalue), eigenvalues lam."""
    mu = X.mean(axis=0)
    Xc = X - mu
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = sv ** 2 / max(1, len(X) - 1)
    return mu, Vt.T, lam


def gate_scales(j0, w, gamma, lam):
    """The rung-1 spectral-filter family: s_j = gate(j; j0, w) * lambda_j^gamma.
    gate = sigmoid((j0 - j)/w) -- soft truncation at index j0, softness w.
    lambda^gamma -- spectral reweighting (gamma=0 PCA scale, -0.5 whitening).
    Normalized so the 90th percentile of s is 1 (pins the global scale against
    the whitening reg floor; global scale is inert for cosine NCMs)."""
    d = len(lam)
    j = np.arange(1, d + 1, dtype=np.float64)
    gate = 1.0 / (1.0 + np.exp(np.clip(-(j0 - j) / w, -60, 60)))
    lam_floor = np.maximum(lam, max(lam.max(), 1e-300) * 1e-12)
    s = gate * lam_floor ** gamma
    q = np.quantile(s, 0.90)
    if q < 1e-12 * s.max():
        q = s.max()
    return s / q


def composite_diag_metric(s, w_var, reg):
    """Closed form for filter + DIAG cluster whitening under FROZEN cluster
    assignments: within-cluster variance of filtered dim j is s_j^2 * w_j, so
    the two stages collapse to one diagonal
        a_j = s_j / sqrt(s_j^2 * w_j + reg_eff),
    with the deployed adaptive floor reg_eff = max(reg, 0.01 * median(s^2 w)).
    Note the saturation: a_j -> 1/sqrt(w_j) as s_j grows (fully whitened),
    a_j -> 0 as s_j -> 0 (dropped)."""
    v = s ** 2 * w_var
    reg_eff = max(reg, 0.01 * float(np.median(v)))
    return s / np.sqrt(v + reg_eff)


class PoolContext:
    """Precomputed pool-only state shared by all candidate evaluations.

    Pool split: half A (fit statistics: eigenbasis, pseudo-centroids, frozen
    whitening clusters), half B1 (fitting objective), half B2 (validation --
    every selection decision). Nothing here touches cal/test.
    """

    def __init__(self, Xu, K, alpha, cfg, verbose=True):
        seed = cfg["seed"]
        self.K, self.alpha, self.cfg = K, alpha, cfg
        self.cal_budget = cfg.get("cal_budget", 800)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Xu))
        half = len(Xu) // 2
        self.iA, iB = perm[:half], perm[half:]
        nb1 = len(iB) // 2
        self.iB1, self.iB2 = iB[:nb1], iB[nb1:]
        self.Xa = np.ascontiguousarray(Xu[self.iA], dtype=np.float64)
        Xb1 = np.ascontiguousarray(Xu[self.iB1], dtype=np.float64)
        Xb2 = np.ascontiguousarray(Xu[self.iB2], dtype=np.float64)

        # Stage-0 basis on half A (bake() re-derives it on the full pool).
        self.mu, self.V, self.lam = pool_eigenbasis(self.Xa)
        self.Ea = (self.Xa - self.mu) @ self.V
        self.Eb1 = (Xb1 - self.mu) @ self.V
        self.Eb2 = (Xb2 - self.mu) @ self.V
        self.Xb1, self.Xb2 = Xb1, Xb2

        # Pseudo-supervision (selector-pilot protocol): K-way k-means on raw
        # L2 half A; halves B1/B2 get predicted labels.
        yb_all_src = np.vstack([Xb1, Xb2])
        self.ya, yb_all, _ = pseudo_task(self.Xa, yb_all_src, K, seed)
        self.yb1, self.yb2 = yb_all[:len(Xb1)], yb_all[len(Xb1):]
        self.Ma = class_means(self.Ea, self.ya, K)

        # Frozen whitening clusters for the fast surrogate: k-means in the
        # (unscaled) eigenbasis of A == k-means on centered raw A (rotation-
        # invariant). Real finalist refits redo k-means in the filtered space.
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=cfg["n_clusters_whiten"], random_state=42,
                    n_init=10).fit(self.Ea)
        resid = self.Ea.copy()
        for c in range(cfg["n_clusters_whiten"]):
            m = km.labels_ == c
            if m.any():
                resid[m] -= self.Ea[m].mean(axis=0)
        self.resid = resid
        self.w_var = (resid ** 2).mean(axis=0)
        sub = np.random.default_rng(seed).choice(
            len(resid), min(cfg["lw_subsample"], len(resid)), replace=False)
        self.resid_sub = resid[sub]
        if verbose:
            print(f"  [ctx] pool {len(Xu)} -> A {len(self.Xa)} / B1 {len(Xb1)}"
                  f" / B2 {len(Xb2)}; K={K} alpha={alpha} "
                  f"cal_budget={self.cal_budget}", flush=True)

    # -- candidate evaluation (surrogate paths) -----------------------------
    def _project(self, s, whiten, E):
        """Map eigen-coordinates E to the candidate space for (s, whiten)."""
        if whiten == "cluster":
            a = composite_diag_metric(s, self.w_var, self.cfg["reg"])
            return E * a, None
        elif whiten == "lw_cluster":
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(assume_centered=True).fit(self.resid_sub * s)
            evals, evecs = np.linalg.eigh(lw.covariance_)
            evals = np.maximum(evals, 1e-12)
            W = evecs @ np.diag(evals ** -0.5) @ evecs.T
            return (E * s) @ W, W
        raise ValueError(whiten)

    def eval_candidate(self, s, whiten, half="B1", n_rep=None, seed_off=0):
        """Surrogate objective for one candidate: pseudo-centroids from A,
        rehearsal set size + margin stats on the requested B half."""
        E = self.Eb1 if half == "B1" else self.Eb2
        yb = self.yb1 if half == "B1" else self.yb2
        if whiten == "cluster":
            a = composite_diag_metric(s, self.w_var, self.cfg["reg"])
            Zb, Mu = E * a, self.Ma * a
        else:
            Zb, W = self._project(s, whiten, E)
            Mu = (self.Ma * s) @ W
        D = centroid_dists(Zb, Mu)
        out = margin_stats(D)
        out["rehearsal_sz"], out["rehearsal_se"] = rehearsal_setsize(
            D, yb, self.K, self.cal_budget, self.alpha,
            n_rep=n_rep or self.cfg["n_rep"], seed=self.cfg["seed"] + seed_off)
        return out

    def eval_real(self, s, whiten, half="B2"):
        """Exact re-evaluation: fit the REAL UnlabeledTransform (k-means refit
        in the filtered space) on raw half A, objective on the raw B half."""
        t = UnlabeledTransform(
            projection="spectral",
            spectral_filter={"mu": self.mu, "V": self.V, "s": s},
            pca_dim=None, whiten=whiten,
            n_clusters=self.cfg["n_clusters_whiten"], random_state=42,
        ).fit(self.Xa)
        Za = t.transform(self.Xa)
        Xb = self.Xb2 if half == "B2" else self.Xb1
        yb = self.yb2 if half == "B2" else self.yb1
        Zb = t.transform(Xb)
        return objective_on_half(Za, Zb, self.ya, yb, self.K,
                                 self.cal_budget, self.alpha,
                                 n_rep=self.cfg["n_rep"],
                                 seed=self.cfg["seed"] + 1)


# ---------------------------------------------------------------------------
# Rung 1: parametric family, grid + Nelder-Mead, real-refit finalists
# ---------------------------------------------------------------------------
def rung1_fit(ctx, verbose=True):
    cfg = ctx.cfg
    records = []
    for whiten in cfg["whiten_modes"]:
        for j0, w, gamma in product(cfg["grid_j0"], cfg["grid_w"],
                                    cfg["grid_gamma"]):
            s = gate_scales(j0, w, gamma, ctx.lam)
            obj = ctx.eval_candidate(s, whiten, half="B1")
            records.append(dict(j0=float(j0), w=float(w), gamma=float(gamma),
                                whiten=whiten, **obj))
        if verbose:
            best = min((r for r in records if r["whiten"] == whiten),
                       key=lambda r: r["rehearsal_sz"])
            print(f"  [grid:{whiten}] best rehearsal={best['rehearsal_sz']:.3f}"
                  f" @ j0={best['j0']:.0f} w={best['w']} gamma={best['gamma']}",
                  flush=True)

    # Nelder-Mead polish from the overall grid argmin (whiten mode fixed).
    best = min(records, key=lambda r: r["rehearsal_sz"])
    from scipy.optimize import minimize

    def nm_obj(theta):
        j0, w, gamma = np.exp(theta[0]), np.exp(theta[1]), theta[2]
        j0 = float(np.clip(j0, 1, len(ctx.lam)))
        w = float(np.clip(w, 0.25, 512))
        gamma = float(np.clip(gamma, -1.0, 1.0))
        s = gate_scales(j0, w, gamma, ctx.lam)
        return ctx.eval_candidate(s, best["whiten"], half="B1")["rehearsal_sz"]

    x0 = np.array([np.log(best["j0"]), np.log(best["w"]), best["gamma"]])
    nm = minimize(nm_obj, x0, method="Nelder-Mead",
                  options=dict(maxiter=cfg["nm_maxiter"], xatol=0.05,
                               fatol=1e-3))
    nm_params = dict(j0=float(np.clip(np.exp(nm.x[0]), 1, len(ctx.lam))),
                     w=float(np.clip(np.exp(nm.x[1]), 0.25, 512)),
                     gamma=float(np.clip(nm.x[2], -1.0, 1.0)),
                     whiten=best["whiten"], rehearsal_sz=float(nm.fun))
    if verbose:
        print(f"  [NM] {nm_params}", flush=True)

    # Finalists: top-N surrogate grid points (mixed modes) + the NM solution,
    # re-evaluated with the REAL transform on validation half B2.
    finalists = sorted(records, key=lambda r: r["rehearsal_sz"])
    finalists = finalists[:cfg["n_real_finalists"]] + [nm_params]
    for f in finalists:
        s = gate_scales(f["j0"], f["w"], f["gamma"], ctx.lam)
        real = ctx.eval_real(s, f["whiten"], half="B2")
        f["real_b2_sz"] = real["rehearsal_sz"]
        f["real_b2_margin_q90"] = real["margin_q90"]
        if verbose:
            print(f"  [real-B2] j0={f['j0']:6.1f} w={f['w']:6.2f} "
                  f"gamma={f['gamma']:+.2f} {f['whiten']:<10} "
                  f"surr={f['rehearsal_sz']:.3f} real={f['real_b2_sz']:.3f}",
                  flush=True)
    winner = min(finalists, key=lambda r: r["real_b2_sz"])
    s_win = gate_scales(winner["j0"], winner["w"], winner["gamma"], ctx.lam)
    report = dict(rung=1, winner=winner, finalists=finalists,
                  n_grid=len(records), grid=records)
    return s_win, winner["whiten"], report


# ---------------------------------------------------------------------------
# Rung 2: free diagonal s, ConfTr-style soft set size (diag-whiten branch
# only -- full-matrix ZCA undoes diag(s) up to shrinkage, s unidentifiable)
# ---------------------------------------------------------------------------
def _smooth_quantile_factory(torch):
    class SmoothQuantile(torch.autograd.Function):
        """q solving mean_i sigmoid((q - s_i)/tau) = level.
        Forward: 40 bisection steps (no grad). Backward: implicit function
        theorem, dq/ds_i = p_i / sum_j p_j, p_i = sigmoid'((q - s_i)/tau)."""

        @staticmethod
        def forward(ctx_, scores, level, tau):
            lo = scores.min() - 3 * tau
            hi = scores.max() + 3 * tau
            for _ in range(40):
                mid = (lo + hi) / 2
                frac = torch.sigmoid((mid - scores) / tau).mean()
                if frac < level:
                    lo = mid
                else:
                    hi = mid
            q = (lo + hi) / 2
            ctx_.save_for_backward(scores, q.detach().clone())
            ctx_.tau = tau
            return q

        @staticmethod
        def backward(ctx_, grad_out):
            scores, q = ctx_.saved_tensors
            sig = torch.sigmoid((q - scores) / ctx_.tau)
            p = sig * (1 - sig)
            wgt = p / (p.sum() + 1e-12)
            return grad_out * wgt, None, None

    return SmoothQuantile


def rung2_fit(ctx, s_init, device="cuda", verbose=True):
    """Free-diagonal refinement of s from the rung-1 init. Returns
    (s_best, report). Selection of lam_tv and best step is on pool half B2
    (hard rehearsal, surrogate metric); the caller runs the final rung-1 vs
    rung-2 showdown with eval_real."""
    import torch
    cfg = ctx.cfg
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    SQ = _smooth_quantile_factory(torch)

    Eb1 = torch.tensor(ctx.Eb1, dtype=torch.float32, device=dev)
    Ma = torch.tensor(ctx.Ma, dtype=torch.float32, device=dev)
    w_var = torch.tensor(ctx.w_var, dtype=torch.float32, device=dev)
    yb1 = torch.tensor(ctx.yb1, dtype=torch.long, device=dev)
    K, alpha = ctx.K, ctx.alpha
    n = len(ctx.yb1)
    m_cal = max(1, ctx.cal_budget // K)

    # Pre-generated balanced pseudo-cal splits of B1, cycled per step.
    rng = np.random.default_rng(cfg["seed"] + 7)
    splits = []
    for _ in range(cfg["n_rep"]):
        cal_idx = []
        for c in range(K):
            pc = np.where(ctx.yb1 == c)[0]
            if len(pc):
                cal_idx.append(rng.choice(pc, min(m_cal, len(pc)),
                                          replace=False))
        cal_idx = np.concatenate(cal_idx)
        mask = np.ones(n, bool)
        mask[cal_idx] = False
        splits.append((torch.tensor(cal_idx, dtype=torch.long, device=dev),
                       torch.tensor(np.where(mask)[0], dtype=torch.long,
                                    device=dev)))

    theta0 = torch.tensor(np.log(np.maximum(s_init, 1e-8)),
                          dtype=torch.float32, device=dev)
    ar = torch.arange(K, device=dev)

    def soft_size(theta, step):
        s = torch.exp(theta)
        v = s ** 2 * w_var
        reg_eff = torch.clamp(0.01 * v.median(), min=cfg["reg"]).detach()
        a = s / torch.sqrt(v + reg_eff)
        Zb = torch.nn.functional.normalize(Eb1 * a, dim=1)
        Mu = torch.nn.functional.normalize(Ma * a, dim=1)
        D = 1.0 - Zb @ Mu.T
        two = torch.topk(-D, 2, dim=1)
        min1, min2 = -two.values[:, 0], -two.values[:, 1]
        amin = two.indices[:, 0]
        other = torch.where(ar[None, :] == amin[:, None],
                            min2[:, None], min1[:, None])
        S = D / (other + 1e-12)
        ci, ti = splits[step % len(splits)]
        cal_scores = S[ci, yb1[ci]]
        level = float(np.ceil((len(ci) + 1) * (1 - alpha)) / len(ci))
        q = SQ.apply(cal_scores, min(level, 1.0 - 1e-6), cfg["r2_tau_q"])
        frac = min(1.0, step / cfg["r2_anneal_steps"])
        tau_set = (cfg["r2_tau_set_hi"]
                   + (cfg["r2_tau_set_lo"] - cfg["r2_tau_set_hi"])
                   * 0.5 * (1 - np.cos(np.pi * frac)))
        size = torch.sigmoid((q - S[ti]) / tau_set).sum(dim=1).mean()
        soft_cov = torch.sigmoid((q - S[ti, yb1[ti]]) / tau_set).mean()
        return size, soft_cov

    def hard_b2(theta):
        s = np.exp(theta.detach().cpu().numpy().astype(np.float64))
        return ctx.eval_candidate(s, "cluster", half="B2")["rehearsal_sz"]

    results = []
    for lam_tv in cfg["r2_lam_tv"]:
        theta = theta0.clone().requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=cfg["r2_lr"])
        best = dict(b2=hard_b2(theta0), theta=theta0.clone(), step=-1)
        cov_drift, since_best = 0, 0
        for step in range(cfg["r2_steps"]):
            opt.zero_grad()
            size, soft_cov = soft_size(theta, step)
            tv = (theta[1:] - theta[:-1]).abs().mean()
            anchor = (theta - theta0).abs().mean()
            loss = size + lam_tv * tv + cfg["r2_anchor"] * anchor
            loss.backward()
            opt.step()
            cov_drift = (cov_drift + 1
                         if abs(float(soft_cov) - (1 - alpha)) > 0.02 else 0)
            if (step + 1) % cfg["r2_eval_every"] == 0:
                b2 = hard_b2(theta)
                if b2 < best["b2"] - 1e-4:
                    best = dict(b2=b2, theta=theta.detach().clone(), step=step)
                    since_best = 0
                else:
                    since_best += cfg["r2_eval_every"]
                if since_best >= cfg["r2_patience"]:
                    break
        eff = float(torch.exp(best["theta"]).sum() ** 2
                    / (torch.exp(best["theta"]) ** 2).sum())
        results.append(dict(lam_tv=lam_tv, b2=best["b2"], step=best["step"],
                            eff_dim=eff, cov_drift_steps=cov_drift,
                            theta=best["theta"]))
        if verbose:
            print(f"  [r2 lam_tv={lam_tv}] best B2={best['b2']:.3f} "
                  f"@step {best['step']} eff_dim={eff:.1f}", flush=True)
    win = min(results, key=lambda r: r["b2"])
    s_best = np.exp(win["theta"].cpu().numpy().astype(np.float64))
    report = dict(rung=2, lam_tv=win["lam_tv"], b2=win["b2"],
                  eff_dim=win["eff_dim"],
                  configs=[{k: v for k, v in r.items() if k != "theta"}
                           for r in results])
    return s_best, report


# ---------------------------------------------------------------------------
# Bake + public API
# ---------------------------------------------------------------------------
def bake(Xu, s, whiten, cfg, ctx=None):
    """Final deployable transform: eigenbasis re-derived on the FULL pool
    (parity with how menu arms are fit), s reused by eigen-index, whitening
    re-fit on the filtered full pool. Returns (transform, stability dict)."""
    mu_f, V_f, lam_f = pool_eigenbasis(np.asarray(Xu, dtype=np.float64))
    t = UnlabeledTransform(
        projection="spectral",
        spectral_filter={"mu": mu_f, "V": V_f, "s": s},
        pca_dim=None, whiten=whiten,
        n_clusters=cfg["n_clusters_whiten"], random_state=42,
    ).fit(np.asarray(Xu, dtype=np.float64))
    stab = {}
    if ctx is not None:
        # Stability: does the A-basis objective survive the full-pool basis?
        Za = t.transform(ctx.Xa)
        Zb2 = t.transform(ctx.Xb2)
        full = objective_on_half(Za, Zb2, ctx.ya, ctx.yb2, ctx.K,
                                 ctx.cal_budget, ctx.alpha,
                                 n_rep=ctx.cfg["n_rep"],
                                 seed=ctx.cfg["seed"] + 2)
        stab = dict(full_basis_b2_sz=full["rehearsal_sz"])
    return t, stab


def fit_conformal_metric(Xu, K, alpha=0.1, cfg=None, rung=1, device="cuda",
                         verbose=True):
    """Public entry point. Sees ONLY the unlabeled pool Xu plus a-priori
    constants (K = label-space size, alpha, frozen cfg). Returns
    (fitted UnlabeledTransform, report dict).

    rung=1: parametric spectral-filter family (grid + NM + real finalists).
    rung=2: rung 1, then free-diagonal refinement; the shipped s is decided
            by the real-transform B2 showdown (pool-internal)."""
    Xu = np.asarray(Xu, dtype=np.float64)
    assert Xu.ndim == 2, "fit_conformal_metric takes the raw pool matrix only"
    cfg = dict(CFG, **(cfg or {}))
    ctx = PoolContext(Xu, K, alpha, cfg, verbose=verbose)
    s1, whiten1, rep1 = rung1_fit(ctx, verbose=verbose)
    report = dict(cfg_hash=cfg_hash(cfg), pool_sha1=pool_sha1(Xu),
                  alpha=alpha, K=K, rung=rung, rung1=rep1,
                  s_rung1=s1.tolist(), whiten_rung1=whiten1)
    s_final, whiten_final = s1, whiten1
    if rung >= 2:
        s2, rep2 = rung2_fit(ctx, s1, device=device, verbose=verbose)
        # Showdown on B2 with the REAL transform (pool-internal decision).
        real1 = rep1["winner"]["real_b2_sz"]
        real2 = ctx.eval_real(s2, "cluster", half="B2")["rehearsal_sz"]
        rep2["real_b2_r1"], rep2["real_b2_r2"] = real1, real2
        report["rung2"] = rep2
        report["s_rung2"] = s2.tolist()
        if real2 < real1:
            s_final, whiten_final = s2, "cluster"
        if verbose:
            print(f"  [showdown] r1={real1:.3f} vs r2={real2:.3f} -> "
                  f"{'r2' if real2 < real1 else 'r1'}", flush=True)
    t, stab = bake(Xu, s_final, whiten_final, cfg, ctx=ctx)
    report.update(s_final=s_final.tolist(), whiten_final=whiten_final,
                  bake_stability=stab,
                  s_eff_dim=float(s_final.sum() ** 2 / (s_final ** 2).sum()))
    if verbose:
        print(f"  [baked] {t} | stability={stab}", flush=True)
    return t, report
