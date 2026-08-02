"""Assert-based checks for the conformal-metric-learning (G1) stack.

Run:  python src/test_conformal_metric.py
All checks use a small synthetic pool (fast, no data files needed) except the
signature guard, which is purely structural.

Checks:
  1. spectral parity vs pca128_cw : indicator-gate s + diag cluster whitening
     reproduces the pca128_cw arm's NCM-relevant geometry (equal reg forced).
  2. spectral parity vs lw_cluster768 : s = 1 + lw_cluster whitening matches
     the lw_cluster768 arm up to rotation (pairwise cosine distances).
  3. closed-form fidelity : composite_diag_metric == real UnlabeledTransform
     refit under FROZEN assignments (same k-means seed/space).
  4. SmoothQuantile : value -> hard quantile as tau -> 0; gradient matches
     finite differences.
  5. determinism + labels-free signature : two fits on the same pool are
     bit-identical; fit_conformal_metric accepts no label/cal/test argument.
"""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from exchangeable_features import UnlabeledTransform  # noqa: E402
from conformal_metric import (pool_eigenbasis, gate_scales,  # noqa: E402
                              composite_diag_metric, fit_conformal_metric,
                              CFG)
from pool_objective import l2n  # noqa: E402


def synth_pool(n=2000, d=64, K=8, seed=0):
    """Anisotropic Gaussian-mixture pool with a decaying spectrum."""
    rng = np.random.default_rng(seed)
    scales = np.linspace(3.0, 0.05, d)
    mus = rng.standard_normal((K, d)) * scales * 0.8
    y = rng.integers(0, K, n)
    X = mus[y] + rng.standard_normal((n, d)) * scales * 0.5
    return X, y


def cos_dists(Z):
    Zn = l2n(Z)
    return 1.0 - Zn @ Zn.T


def check_parity_pca():
    X, _ = synth_pool()
    r = 16
    reg = 1e-4
    # reference arm: PCA-r + diag cluster whitening
    ref = UnlabeledTransform(pca_dim=r, whiten="cluster", n_clusters=10,
                             random_state=42, reg=reg).fit(X)
    # spectral arm: hard indicator gate on the top r eigendirections
    mu, V, lam = pool_eigenbasis(X)
    s = np.zeros(X.shape[1])
    s[:r] = 1.0
    spec = UnlabeledTransform(projection="spectral",
                              spectral_filter={"mu": mu, "V": V, "s": s},
                              whiten="cluster", n_clusters=10,
                              random_state=42, reg=reg).fit(X)
    Q = X[:200]
    Zr, Zs = ref.transform(Q), spec.transform(Q)
    # The spectral arm keeps 768-d output with zeros on dead dims; compare the
    # live block. k-means may label differently (dead dims are zero for every
    # point, so the clustering input is identical up to embedding) -- compare
    # pairwise cosine distances, which ignore dead zero dims. reg floor:
    # median over d dims (spectral, mostly zeros) vs r dims (pca) differs ->
    # compare with whitening scales aligned instead of bitwise.
    Dr, Ds = cos_dists(Zr), cos_dists(Zs)
    err = np.abs(Dr - Ds).max()
    assert err < 5e-2, f"pca parity: max cosine-dist deviation {err:.4f}"
    # sign-insensitive check on the live-dim whitening scales
    live = spec.inv_std_[:r] * (s[:r])
    ratio = (ref.inv_std_ / live)
    spread = ratio.max() / ratio.min()
    assert spread < 1.5, f"whitening-scale ratio spread {spread:.3f}"
    print(f"  [1] spectral ~ pca{r}_cw parity: max D-dev {err:.2e}, "
          f"scale spread {spread:.3f}  OK")


def check_parity_lw():
    X, _ = synth_pool()
    ref = UnlabeledTransform(pca_dim=None, whiten="lw_cluster", projection=None,
                             n_clusters=10, random_state=42).fit(X)
    mu, V, lam = pool_eigenbasis(X)
    s = np.ones(X.shape[1])
    spec = UnlabeledTransform(projection="spectral",
                              spectral_filter={"mu": mu, "V": V, "s": s},
                              whiten="lw_cluster", n_clusters=10,
                              random_state=42).fit(X)
    Q = X[:200]
    Dr, Ds = cos_dists(ref.transform(Q)), cos_dists(spec.transform(Q))
    # ZCA in a rotated basis = rotated ZCA -> pairwise cosine distances match
    # up to k-means assignment noise (same seed, rotated input -> same
    # objective value; centers may permute). Tolerance is loose-ish for that.
    err = np.abs(Dr - Ds).max()
    assert err < 5e-2, f"lw parity: max cosine-dist deviation {err:.4f}"
    print(f"  [2] spectral(s=1)+lw ~ lw_cluster parity: max D-dev {err:.2e}  OK")


def check_closed_form():
    X, _ = synth_pool()
    mu, V, lam = pool_eigenbasis(X)
    E = (X - mu) @ V
    # frozen assignments from k-means in the eigenbasis
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(E)
    resid = E.copy()
    for c in range(10):
        m = km.labels_ == c
        if m.any():
            resid[m] -= E[m].mean(axis=0)
    w_var = (resid ** 2).mean(axis=0)
    s = gate_scales(24, 8, -0.25, lam)
    a = composite_diag_metric(s, w_var, CFG["reg"])
    # manual "real" path under the SAME frozen assignments
    Ef = E * s
    residf = Ef.copy()
    for c in range(10):
        m = km.labels_ == c
        if m.any():
            residf[m] -= Ef[m].mean(axis=0)
    vf = (residf ** 2).mean(axis=0)
    reg_eff = max(CFG["reg"], 0.01 * float(np.median(vf)))
    a_real = s / np.sqrt(vf + reg_eff)
    dev = np.abs(a - a_real).max() / max(a_real.max(), 1e-12)
    assert dev < 1e-9, f"closed-form deviation {dev:.2e}"
    print(f"  [3] composite_diag_metric closed form: rel dev {dev:.2e}  OK")


def check_smooth_quantile():
    import torch
    from conformal_metric import _smooth_quantile_factory
    SQ = _smooth_quantile_factory(torch)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    level = 0.9
    for tau in (0.05, 0.005, 0.0005):
        q = float(SQ.apply(torch.tensor(x), level, tau))
        hard = float(np.quantile(x, level))
        if tau == 0.0005:
            assert abs(q - hard) < 0.02, f"tau->0 limit: {q} vs {hard}"
    # gradient vs finite differences
    xt = torch.tensor(x, requires_grad=True)
    q = SQ.apply(xt, level, 0.05)
    q.backward()
    g = xt.grad.numpy()
    i = int(np.argmax(g))
    eps = 1e-4
    xp = x.copy()
    xp[i] += eps
    q1 = float(SQ.apply(torch.tensor(xp), level, 0.05))
    q0 = float(SQ.apply(torch.tensor(x), level, 0.05))
    fd = (q1 - q0) / eps
    assert abs(fd - g[i]) < 0.05 * max(abs(fd), 1e-3), \
        f"grad mismatch: fd {fd:.4f} vs autograd {g[i]:.4f}"
    assert abs(g.sum() - 1.0) < 1e-4, "quantile grads must sum to 1"
    print(f"  [4] SmoothQuantile: limit + grad (fd {fd:.4f} ~ {g[i]:.4f})  OK")


def check_determinism_and_signature():
    sig = inspect.signature(fit_conformal_metric)
    forbidden = {"y", "labels", "X_cal", "X_test", "cal", "test"}
    assert not (set(sig.parameters) & forbidden), \
        "fit_conformal_metric must not accept label/cal/test arguments"
    X, _ = synth_pool(n=1200, d=48)
    cfg = dict(seed=0, cal_budget=200, grid_j0=[8, 24, 48],
               grid_w=[0.5, 8], grid_gamma=[-0.5, 0.0], n_rep=5,
               n_clusters_whiten=8, nm_maxiter=20, n_real_finalists=2)
    t1, r1 = fit_conformal_metric(X, K=8, alpha=0.1, cfg=cfg, rung=1,
                                  device="cpu", verbose=False)
    t2, r2 = fit_conformal_metric(X, K=8, alpha=0.1, cfg=cfg, rung=1,
                                  device="cpu", verbose=False)
    assert np.array_equal(np.asarray(r1["s_final"]),
                          np.asarray(r2["s_final"])), "refit not deterministic"
    assert r1["pool_sha1"] == r2["pool_sha1"]
    Zq = t1.transform(X[:5])
    assert np.allclose(Zq, t2.transform(X[:5]))
    print(f"  [5] determinism + labels-free signature "
          f"(winner j0={r1['rung1']['winner']['j0']:.0f}, "
          f"whiten={r1['whiten_final']})  OK")


if __name__ == "__main__":
    print("test_conformal_metric:")
    check_parity_pca()
    check_parity_lw()
    check_closed_form()
    check_smooth_quantile()
    check_determinism_and_signature()
    print("ALL CHECKS PASS")
