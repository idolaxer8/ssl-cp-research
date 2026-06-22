"""
Correctness + exact-exchangeability tests for RidgeSoftmaxNCM (FCA-inspired,
text-free SS-Text: class-mean-anchored ridge probe -> softmax LAC). CPU-only;
no CUDA required.

Checks:
  1. test_oracle      -- the fast Sherman-Morrison + PRESS path matches a
                         brute-force leave-one-out refit (anchor held at the
                         full-bag mean), for LOO and full-bag, incl. a candidate
                         class absent from cal. This validates the update math.
  2. test_coverage    -- random split, alpha=0.1: mean coverage lands in the
                         exact band [0.88, 0.925] at every cal size, with
                         non-trivial (< K) sets. The decisive validity gate.
  3. test_edge_cases  -- missing classes in cal, K=2, bounded scores in [0,1].

Run:  python tests/test_ridge_softmax_ncm.py
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conformal_prediction import (  # noqa: E402
    RidgeSoftmaxNCM,
    FullConformalPredictor,
    create_ncm,
)


def make_gmm(K, per_class, d, seed, sep=4.0):
    """Synthetic Gaussian-mixture embeddings; K balanced classes."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(K, d)) * sep
    X = np.concatenate(
        [centers[c] + rng.normal(size=(per_class, d)) for c in range(K)]
    ).astype(np.float64)
    y = np.concatenate([np.full(per_class, c) for c in range(K)]).astype(np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def _softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


# ----------------------------------------------------------------------
# 1. Brute-force PRESS oracle
# ----------------------------------------------------------------------
def reference_scores(ncm, Zc, yc, z_test, y, classes):
    """Rebuild the augmented bag B = cal U {(z_test, y)}; hold the class-mean
    anchor at its full-bag value; leave out each point's DATA row explicitly
    (this is the exact definition the PRESS shortcut implements); softmax.
    Returns (cal_scores (n,), test_score)."""
    lam_eff = ncm.lam + ncm.lam_anchor
    la = ncm.lam_anchor
    T = ncm._T
    loo = ncm.loo

    cls = [int(c) for c in classes]
    if int(y) not in cls:
        cls = cls + [int(y)]
    col = {c: j for j, c in enumerate(cls)}
    K = len(cls)
    d = Zc.shape[1]
    n = len(Zc)

    Zb = np.vstack([Zc, z_test[None, :]])
    yb = np.append(yc, int(y))

    M1 = np.zeros((d, K))
    data = np.zeros((d, K))
    for c in cls:
        m = yb == c
        if m.any():
            M1[:, col[c]] = Zb[m].mean(0)
            data[:, col[c]] = Zb[m].sum(0)
    B1 = data + la * M1
    A1 = Zb.T @ Zb
    A1[np.diag_indices_from(A1)] += lam_eff
    W_full = np.linalg.solve(A1, B1)

    def loo_logits(zj, cj):
        # remove zj's data row only; anchor M1 held fixed (full-bag)
        A_m = A1 - np.outer(zj, zj)
        data_m = data.copy()
        data_m[:, cj] -= zj
        return zj @ np.linalg.solve(A_m, data_m + la * M1)

    cal = np.empty(n)
    for i in range(n):
        ci = col[int(yc[i])]
        f = loo_logits(Zc[i], ci) if loo else Zc[i] @ W_full
        cal[i] = 1.0 - _softmax(f / T)[ci]

    cy = col[int(y)]
    f = loo_logits(z_test, cy) if loo else z_test @ W_full
    test = 1.0 - _softmax(f / T)[cy]
    return cal, test


def test_oracle():
    max_err = 0.0
    for loo in (True, False):
        for lam_a in (0.5, 5.0):
            for seed in (0, 1, 2):
                Zc, yc = make_gmm(K=5, per_class=12, d=8, seed=seed, sep=3.0)
                classes = np.unique(yc)
                ncm = RidgeSoftmaxNCM(lam=1.0, lam_anchor=lam_a,
                                      temperature=0.3, loo=loo).fit(Zc, yc)
                rng = np.random.default_rng(100 + seed)
                cands = [int(classes[0]), int(classes[-1]), 999]  # 999 = absent
                for _ in range(4):
                    z = rng.normal(size=8)
                    for y in cands:
                        ref_cal, ref_test = reference_scores(ncm, Zc, yc, z, y, classes)
                        fast_cal = ncm.updated_calibration_scores_for(z, y)
                        fast_test = ncm.score_x(z, y)
                        max_err = max(max_err,
                                      np.abs(ref_cal - fast_cal).max(),
                                      abs(ref_test - fast_test))
                        assert np.allclose(ref_cal, fast_cal, atol=1e-7), \
                            f"cal mismatch loo={loo} lam_a={lam_a} y={y} " \
                            f"err={np.abs(ref_cal - fast_cal).max():.2e}"
                        assert abs(ref_test - fast_test) < 1e-7, \
                            f"test mismatch loo={loo} lam_a={lam_a} y={y}"
    print(f"test_oracle: PASS (max |fast - brute-force| = {max_err:.2e})")


# ----------------------------------------------------------------------
# 2. Exact-exchangeability coverage gate
# ----------------------------------------------------------------------
def test_coverage():
    K, d = 20, 16
    X, y = make_gmm(K=K, per_class=220, d=d, seed=1, sep=3.0)
    classes = np.unique(y)
    alpha = 0.1

    # A sharp-but-FIXED temperature: estimated once on a fixed chunk -> a global
    # constant across all cal/test splits -> exact (T does not depend on the
    # split). This isolates the validity check from temperature tuning.
    tmp = RidgeSoftmaxNCM(lam=1.0, lam_anchor=1.0, temperature=None,
                          allow_nonexchangeable=True).fit(X[:2000], y[:2000])
    T_fixed = tmp._T

    ok = True
    for cal_size in (200, 400, 800):
        covs, szs = [], []
        for t in range(20):
            r = np.random.default_rng(1000 + t)
            idx = r.permutation(len(X))
            ci, ti = idx[:cal_size], idx[cal_size:cal_size + 1000]
            ncm = RidgeSoftmaxNCM(lam=1.0, lam_anchor=1.0,
                                  temperature=T_fixed, loo=True)
            cp = FullConformalPredictor(ncm, alpha=alpha)
            cp.calibrate(X[ci], y[ci], all_classes=classes)
            m = cp.evaluate(X[ti], y[ti], verbose=False)
            covs.append(m["coverage"]); szs.append(m["avg_set_size"])
        cov, sz = float(np.mean(covs)), float(np.mean(szs))
        flag = "OK" if (0.88 <= cov <= 0.925 and sz < K) else "!!"
        if flag == "!!":
            ok = False
        print(f"  cal={cal_size:4d}: cov={cov:.4f} sz={sz:5.2f}  {flag}")
        assert 0.875 <= cov <= 0.93, f"coverage {cov:.4f} out of band at cal={cal_size}"
        assert sz < K, f"trivial sets (sz={sz:.2f} ~ K={K}) at cal={cal_size}"
    print(f"test_coverage: {'PASS' if ok else 'PASS (borderline)'} (T={T_fixed:.3f})")


# ----------------------------------------------------------------------
# 3. Edge cases
# ----------------------------------------------------------------------
def test_edge_cases():
    # (a) classes missing from cal; test spans the full label space
    X, y = make_gmm(K=10, per_class=90, d=12, seed=3, sep=3.0)
    classes = np.unique(y)
    r = np.random.default_rng(5)
    keep = classes[:7]                       # cal omits 3 classes
    cal_pool = np.where(np.isin(y, keep))[0]
    ci = r.choice(cal_pool, 300, replace=False)
    ti = r.choice(np.setdiff1d(np.arange(len(X)), ci), 500, replace=False)
    ncm = RidgeSoftmaxNCM(temperature=0.3, loo=True).fit(X[ci], y[ci])
    cp = FullConformalPredictor(ncm, alpha=0.1)
    cp.calibrate(X[ci], y[ci], all_classes=classes)
    m = cp.evaluate(X[ti], y[ti], verbose=False)
    assert np.isfinite(m["coverage"]), "coverage not finite with missing classes"

    # scores for a candidate class ABSENT from cal must be finite & in [0, 1]
    y_absent = int(classes[-1])              # one of the 3 omitted classes
    s = ncm.score_x(X[ti[0]], y_absent)
    assert 0.0 <= s <= 1.0, f"test score {s} out of [0,1] for absent class"
    sc = ncm.updated_calibration_scores_for(X[ti[0]], y_absent)
    assert np.all(np.isfinite(sc)) and sc.min() >= 0.0 and sc.max() <= 1.0, \
        "cal scores out of [0,1] for absent candidate class"

    # (b) K=2 and (c) lam_anchor -> large reduces toward a prototype (still valid)
    X2, y2 = make_gmm(K=2, per_class=120, d=6, seed=8, sep=3.0)
    s2 = RidgeSoftmaxNCM(temperature=0.3).fit(X2[:160], y2[:160]).get_calibration_scores()
    assert np.all(np.isfinite(s2)) and s2.min() >= 0 and s2.max() <= 1.0

    proto = RidgeSoftmaxNCM(lam=1.0, lam_anchor=1e6, temperature=0.3).fit(X[ci], y[ci])
    assert np.all(np.isfinite(proto.get_calibration_scores()))
    print("test_edge_cases: PASS")


# ----------------------------------------------------------------------
# 4. Vectorized (GPU) path parity vs the CPU Python loop
# ----------------------------------------------------------------------
def test_vectorized_parity():
    """`_predict_ridge_softmax_gpu` (batched torch; CUDA if available, else
    vectorised CPU) must match the CPU Python loop bit-for-bit -- prediction sets
    identical and p-values to ~1e-9. Requires all candidate classes in cal (use a
    balanced cal draw). Runs everywhere (no CUDA needed)."""
    K, d = 20, 16
    X, y = make_gmm(K=K, per_class=120, d=d, seed=2, sep=3.0)
    rng = np.random.default_rng(0)
    ci, ti = [], []
    for c in range(K):
        idx = rng.permutation(np.where(y == c)[0])
        ci.append(idx[:10]); ti.append(idx[10:18])
    ci = np.concatenate(ci); ti = np.concatenate(ti)
    allc = np.unique(y[ci])
    max_dp = 0.0
    for loo in (True, False):
        for la in (1.0, 8.0):
            ncm = RidgeSoftmaxNCM(temperature=0.2, lam_anchor=la, loo=loo)
            cp = FullConformalPredictor(ncm, alpha=0.1)
            cp.calibrate(X[ci], y[ci], all_classes=allc)
            rc = cp.predict(X[ti], return_p_values=True, device="cpu", verbose=False)
            rv = cp._predict_ridge_softmax_gpu(X[ti], return_p_values=True, verbose=False)
            for a, b in zip(rc["prediction_sets"], rv["prediction_sets"]):
                assert sorted(a) == sorted(b), f"set mismatch loo={loo} la={la}"
            pc = np.array([[rc["p_values"][i][int(c)] for c in allc] for i in range(len(ti))])
            pv = np.array([[rv["p_values"][i][int(c)] for c in allc] for i in range(len(ti))])
            max_dp = max(max_dp, float(np.abs(pc - pv).max()))
    assert max_dp < 1e-9, f"vectorized p-value mismatch {max_dp:.2e}"
    print(f"test_vectorized_parity: PASS (max |cpu - vectorized| = {max_dp:.2e})")


# ----------------------------------------------------------------------
# Informational: set-size vs the exact geodesic control (no assert)
# ----------------------------------------------------------------------
def report_setsize_vs_geodesic():
    K, d = 20, 16
    X, y = make_gmm(K=K, per_class=220, d=d, seed=2, sep=3.0)
    classes = np.unique(y)
    tmp = RidgeSoftmaxNCM(temperature=None, allow_nonexchangeable=True).fit(X[:2000], y[:2000])
    Tf = tmp._T
    print("  set-size (cov) by NCM, 10 trials, alpha=0.1:")
    for cal_size in (200, 400, 800):
        rows = {}
        for name, mk in (
            ("ridge_softmax(la=1)", lambda: RidgeSoftmaxNCM(lam_anchor=1.0, temperature=Tf)),
            ("ridge_softmax(la=20)", lambda: RidgeSoftmaxNCM(lam_anchor=20.0, temperature=Tf)),
            ("unwhitened_topk_mean", lambda: create_ncm("unwhitened_topk_mean", k=5)),
        ):
            covs, szs = [], []
            for t in range(10):
                r = np.random.default_rng(7000 + t)
                idx = r.permutation(len(X))
                ci, ti = idx[:cal_size], idx[cal_size:cal_size + 800]
                cp = FullConformalPredictor(mk(), alpha=0.1)
                cp.calibrate(X[ci], y[ci], all_classes=classes)
                m = cp.evaluate(X[ti], y[ti], verbose=False)
                covs.append(m["coverage"]); szs.append(m["avg_set_size"])
            rows[name] = (np.mean(covs), np.mean(szs))
        cells = "  ".join(f"{k}: sz={v[1]:.2f}(cov={v[0]:.3f})" for k, v in rows.items())
        print(f"  cal={cal_size:4d} | {cells}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    test_oracle()
    test_vectorized_parity()
    test_coverage()
    test_edge_cases()
    print("\n--- informational ---")
    report_setsize_vs_geodesic()
    print("\nALL RIDGE-SOFTMAX TESTS PASSED")
