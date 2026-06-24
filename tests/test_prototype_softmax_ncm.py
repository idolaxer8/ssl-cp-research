"""
Correctness + exact-exchangeability tests for PrototypeSoftmaxNCM (the VANILLA
text-free FCA NCM: class-mean prototype -> softmax LAC, no covariance term).
CPU-only; no CUDA required.

Checks:
  1. test_oracle              -- the fast closed-form leave-one-out / per-candidate
                                update matches a brute-force re-score of the whole
                                augmented bag, for logit in {cosine, dot}, incl. a
                                candidate class absent from cal. Validates the math.
  2. test_coverage           -- random split, alpha=0.1: mean coverage lands in the
                                exact band [0.88, 0.925] at every cal size with
                                non-trivial (< K) sets. The decisive validity gate.
  3. test_small_cal_bloat    -- the KEY property: under BALANCED cal at m=2/class the
                                method BLOATS (large sets) but does NOT under-cover
                                (cov >= 0.88). Contrast: ridge_softmax under-covered
                                (~0.82) at m=2 -- the exact prototype LOO does not.
  4. test_edge_cases         -- missing classes in cal, absent candidate class,
                                K=2, singleton cal class; bounded scores in [0,1].
  5. test_temperature_invariance -- validity holds for ANY fixed T; auto-T warns.

Run:  python tests/test_prototype_softmax_ncm.py
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conformal_prediction import (  # noqa: E402
    PrototypeSoftmaxNCM,
    FullConformalPredictor,
    ExchangeabilityWarning,
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


def balanced_split(y, allc, m_cal, m_test, rng):
    """Disjoint balanced cal/test: m_cal then m_test samples per class."""
    ci, ti = [], []
    for c in allc:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        ci.extend(idx[:m_cal])
        ti.extend(idx[m_cal:m_cal + m_test])
    return np.array(ci), np.array(ti)


def _softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


# ----------------------------------------------------------------------
# 1. Brute-force leave-one-out oracle
# ----------------------------------------------------------------------
def _prep(X, cosine, eps=1e-9):
    Z = np.asarray(X, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[None, :]
    if cosine:
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), eps)
    return Z


def _proto(s, count, cosine, eps=1e-9):
    if count <= 0:
        return None
    if cosine:
        nrm = float(np.linalg.norm(s))
        return None if nrm < eps else s / nrm
    return s / count


def reference_scores(Zc_raw, yc, z_test_raw, y, cosine, T):
    """Rebuild the augmented bag B = cal U {(z_test, y)}; for EVERY point leave it
    out of its own class, recompute every prototype, softmax. The exact definition
    the fast path implements. Returns (cal_scores (n,), test_score)."""
    Zc = _prep(Zc_raw, cosine)
    zt = _prep(z_test_raw, cosine).ravel()
    Zb = np.vstack([Zc, zt[None, :]])
    yb = np.append(np.asarray(yc), int(y))
    classes = np.unique(yb)
    col = {int(c): j for j, c in enumerate(classes)}
    K = len(classes)
    n = len(Zc)

    def score_point(idx):
        zj = Zb[idx]
        yj = int(yb[idx])
        logits = np.full(K, -np.inf)
        for c in classes:
            mask = (yb == int(c))
            mask[idx] = False                       # leave point idx out
            p = _proto(Zb[mask].sum(0), int(mask.sum()), cosine)
            logits[col[int(c)]] = -np.inf if p is None else float(zj @ p)
        return 1.0 - _softmax(logits / T)[col[yj]]

    cal = np.array([score_point(i) for i in range(n)])
    test = score_point(n)
    return cal, test


def test_oracle():
    max_err = 0.0
    for logit in ("cosine", "dot"):
        for seed in (0, 1, 2):
            Zc, yc = make_gmm(K=5, per_class=12, d=8, seed=seed, sep=3.0)
            classes = np.unique(yc)
            ncm = PrototypeSoftmaxNCM(temperature=0.3, logit=logit).fit(Zc, yc)
            rng = np.random.default_rng(100 + seed)
            cands = [int(classes[0]), int(classes[-1]), 999]   # 999 = absent
            for _ in range(4):
                z = rng.normal(size=8)
                for y in cands:
                    ref_cal, ref_test = reference_scores(Zc, yc, z, y, ncm.cosine, ncm._T)
                    fast_cal = ncm.updated_calibration_scores_for(z, y)
                    fast_test = ncm.score_x(z, y)
                    max_err = max(max_err,
                                  np.abs(ref_cal - fast_cal).max(),
                                  abs(ref_test - fast_test))
                    assert np.allclose(ref_cal, fast_cal, atol=1e-9), \
                        f"cal mismatch logit={logit} y={y} " \
                        f"err={np.abs(ref_cal - fast_cal).max():.2e}"
                    assert abs(ref_test - fast_test) < 1e-9, \
                        f"test mismatch logit={logit} y={y} " \
                        f"err={abs(ref_test - fast_test):.2e}"
    print(f"test_oracle: PASS (max |fast - brute-force| = {max_err:.2e})")


# ----------------------------------------------------------------------
# 2. Exact-exchangeability coverage gate (random split)
# ----------------------------------------------------------------------
def test_coverage():
    K, d = 20, 16
    X, y = make_gmm(K=K, per_class=220, d=d, seed=1, sep=3.0)
    classes = np.unique(y)
    alpha = 0.1

    # A sharp-but-FIXED temperature: estimated once on a fixed chunk -> a global
    # constant across all cal/test splits -> exact (T independent of the split).
    tmp = PrototypeSoftmaxNCM(temperature=None,
                              allow_nonexchangeable=True).fit(X[:2000], y[:2000])
    T_fixed = tmp._T

    ok = True
    for cal_size in (200, 400, 800):
        covs, szs = [], []
        for t in range(20):
            r = np.random.default_rng(1000 + t)
            idx = r.permutation(len(X))
            ci, ti = idx[:cal_size], idx[cal_size:cal_size + 1000]
            ncm = PrototypeSoftmaxNCM(temperature=T_fixed)
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
# 3. Small-cal: BLOAT, never under-coverage (the key property)
# ----------------------------------------------------------------------
def test_small_cal_bloat():
    K, d = 20, 16
    X, y = make_gmm(K=K, per_class=220, d=d, seed=4, sep=3.0)
    classes = np.unique(y)
    tmp = PrototypeSoftmaxNCM(temperature=None,
                              allow_nonexchangeable=True).fit(X[:2000], y[:2000])
    T_fixed = tmp._T

    sizes = {}
    for m_cal in (2, 4, 8):
        covs, szs = [], []
        for t in range(20):
            r = np.random.default_rng(2000 + t)
            ci, ti = balanced_split(y, classes, m_cal=m_cal, m_test=8, rng=r)
            ncm = PrototypeSoftmaxNCM(temperature=T_fixed)
            cp = FullConformalPredictor(ncm, alpha=0.1)
            cp.calibrate(X[ci], y[ci], all_classes=classes)
            mtr = cp.evaluate(X[ti], y[ti], verbose=False)
            covs.append(mtr["coverage"]); szs.append(mtr["avg_set_size"])
        cov, sz = float(np.mean(covs)), float(np.mean(szs))
        sizes[m_cal] = sz
        print(f"  balanced m_cal={m_cal} (cal={m_cal*K}): cov={cov:.4f} sz={sz:5.2f}")
        # COVERAGE HELD: balanced cal over-covers (conservative); exact LOO means
        # NO under-coverage even at m=2 -- unlike ridge_softmax (~0.82 at m=2).
        assert cov >= 0.88, \
            f"under-coverage at balanced m_cal={m_cal}: cov={cov:.4f} (should bloat, not under-cover)"
    # BLOAT: smaller cal -> larger sets (noisy prototypes -> near-uniform softmax)
    assert sizes[2] > sizes[8], \
        f"expected bloat at small cal: sz(m=2)={sizes[2]:.2f} !> sz(m=8)={sizes[8]:.2f}"
    print("test_small_cal_bloat: PASS (coverage held + sets bloat at small cal)")


# ----------------------------------------------------------------------
# 4. Edge cases
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
    ncm = PrototypeSoftmaxNCM(temperature=0.3).fit(X[ci], y[ci])
    cp = FullConformalPredictor(ncm, alpha=0.1)
    cp.calibrate(X[ci], y[ci], all_classes=classes)
    m = cp.evaluate(X[ti], y[ti], verbose=False)
    assert np.isfinite(m["coverage"]), "coverage not finite with missing classes"

    # candidate class ABSENT from cal -> bounded scores in [0, 1]
    y_absent = int(classes[-1])              # one of the 3 omitted classes
    s = ncm.score_x(X[ti[0]], y_absent)
    assert s == 1.0, f"absent-class test score should be 1.0, got {s}"
    sc = ncm.updated_calibration_scores_for(X[ti[0]], y_absent)
    assert np.all(np.isfinite(sc)) and sc.min() >= 0.0 and sc.max() <= 1.0, \
        "cal scores out of [0,1] for absent candidate class"

    # (b) K=2 and (c) a singleton cal class (empty LOO -> score 1.0, no NaN)
    X2, y2 = make_gmm(K=2, per_class=120, d=6, seed=8, sep=3.0)
    s2 = PrototypeSoftmaxNCM(temperature=0.3).fit(X2[:160], y2[:160]).get_calibration_scores()
    assert np.all(np.isfinite(s2)) and s2.min() >= 0 and s2.max() <= 1.0

    # singleton: build a cal set where one class has exactly 1 point
    Xs = np.vstack([X2[y2 == 0][:5], X2[y2 == 1][:1]])
    ys = np.array([0, 0, 0, 0, 0, 1])
    ss = PrototypeSoftmaxNCM(temperature=0.3).fit(Xs, ys).get_calibration_scores()
    assert np.all(np.isfinite(ss)), "singleton class produced non-finite cal score"
    assert abs(ss[-1] - 1.0) < 1e-12, f"singleton point should score 1.0, got {ss[-1]}"
    print("test_edge_cases: PASS")


# ----------------------------------------------------------------------
# 5. Temperature: validity is T-independent; auto-T warns
# ----------------------------------------------------------------------
def test_temperature_invariance():
    K, d = 20, 16
    X, y = make_gmm(K=K, per_class=200, d=d, seed=6, sep=3.0)
    classes = np.unique(y)
    for T in (0.05, 0.1, 0.3):
        covs = []
        for t in range(12):
            r = np.random.default_rng(3000 + t)
            idx = r.permutation(len(X))
            ci, ti = idx[:400], idx[400:1400]
            ncm = PrototypeSoftmaxNCM(temperature=T)
            cp = FullConformalPredictor(ncm, alpha=0.1)
            cp.calibrate(X[ci], y[ci], all_classes=classes)
            covs.append(cp.evaluate(X[ti], y[ti], verbose=False)["coverage"])
        cov = float(np.mean(covs))
        print(f"  T={T:.2f}: cov={cov:.4f}")
        assert 0.875 <= cov <= 0.93, f"coverage {cov:.4f} out of band at T={T}"

    # auto-T (temperature=None) must emit ExchangeabilityWarning unless approved
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PrototypeSoftmaxNCM(temperature=None).fit(X[:400], y[:400])
        assert any(issubclass(wi.category, ExchangeabilityWarning) for wi in w), \
            "auto-T did not warn"
    # approved -> silent
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PrototypeSoftmaxNCM(temperature=None,
                            allow_nonexchangeable=True).fit(X[:400], y[:400])
        assert not any(issubclass(wi.category, ExchangeabilityWarning) for wi in w), \
            "approved auto-T still warned"
    print("test_temperature_invariance: PASS")


# ----------------------------------------------------------------------
# 6. GPU/torch-path parity vs the (proven-exact) CPU loop
# ----------------------------------------------------------------------
def test_gpu_parity():
    """The vectorized predict(device='cuda') path must match the numpy CPU loop
    set-for-set and p-value-for-p-value. Routes to torch on CUDA if available,
    else vectorized CPU torch -- either way a real parity check vs the CPU loop
    proven exact by test_oracle. Requires all candidate classes present in cal."""
    import torch
    for logit in ("cosine", "dot"):
        X, y = make_gmm(K=10, per_class=80, d=12, seed=11, sep=3.0)
        classes = np.unique(y)
        r = np.random.default_rng(11)
        ci, ti = balanced_split(y, classes, m_cal=20, m_test=12, rng=r)
        mk = lambda: PrototypeSoftmaxNCM(temperature=0.2, logit=logit)
        cp_cpu = FullConformalPredictor(mk(), alpha=0.1)
        cp_cpu.calibrate(X[ci], y[ci], all_classes=classes)
        r_cpu = cp_cpu.predict(X[ti], return_p_values=True, verbose=False, device="cpu")
        cp_gpu = FullConformalPredictor(mk(), alpha=0.1)
        cp_gpu.calibrate(X[ci], y[ci], all_classes=classes)
        r_gpu = cp_gpu.predict(X[ti], return_p_values=True, verbose=False, device="cuda")
        sets_cpu = [sorted(s) for s in r_cpu["prediction_sets"]]
        sets_gpu = [sorted(s) for s in r_gpu["prediction_sets"]]
        assert sets_cpu == sets_gpu, f"set mismatch (logit={logit})"
        max_dp = 0.0
        for pc, pg in zip(r_cpu["p_values"], r_gpu["p_values"]):
            for k in pc:
                max_dp = max(max_dp, abs(pc[k] - pg[k]))
        assert max_dp < 1e-9, f"p-value mismatch (logit={logit}) max={max_dp:.2e}"
    dev = "cuda" if torch.cuda.is_available() else "cpu(torch)"
    print(f"test_gpu_parity: PASS (sets identical, max |dp| < 1e-9, ran on {dev})")


# ----------------------------------------------------------------------
# 7. GPU MS-CS path parity (prototype): run_prototype_mscs_torch vs CPU loop
# ----------------------------------------------------------------------
def test_mscs_gpu_parity():
    """The prototype GPU MS-CS kernel (exchangeable cluster-M, argmax yhat) must
    match the CPU run_fcp_with_mscs(prototype, yhat_mode='ncm') set-for-set."""
    import torch
    from mscs_unlabeled_experiment import (build_cluster_similarity_matrix,
                                           run_fcp_with_mscs)
    from mscs_gpu import run_prototype_mscs_torch
    X, y = make_gmm(K=20, per_class=60, d=16, seed=7, sep=3.0)
    classes = np.unique(y)
    r = np.random.default_rng(7)
    ci, ti = balanced_split(y, classes, m_cal=20, m_test=12, rng=r)
    pool = np.setdiff1d(np.arange(len(X)), np.concatenate([ci, ti]))
    Xc, yc, Xt, yt, Xu = X[ci], y[ci], X[ti], y[ti], X[pool]
    Tf = PrototypeSoftmaxNCM(temperature=None, allow_nonexchangeable=True,
                             logit="cosine").fit(Xc, yc)._T
    M, c2c, eff_tau, _m, ccen, ccnt, clcen, cld = build_cluster_similarity_matrix(
        Xu, Xc, yc, classes, 10, tau=-0.5)
    common = dict(exchangeable=True, yhat_mode="ncm", update_M_fn=None,
                  class_centroids=ccen, class_counts=ccnt, class_to_cluster=c2c,
                  cluster_centroids=clcen, cluster_dists=cld, effective_tau=eff_tau,
                  return_sets=True, temperature=Tf, logit="cosine")
    _, _, sets_cpu = run_fcp_with_mscs(Xc, yc, Xt, yt, classes, "prototype_softmax",
                                       0.1, 0.05, M, device="cpu", **common)
    ncm = create_ncm("prototype_softmax", temperature=Tf, logit="cosine")
    cp = FullConformalPredictor(ncm, alpha=0.1); cp.calibrate(Xc, yc, all_classes=classes)
    common.pop("temperature"); common.pop("logit")
    _, _, sets_gpu = run_prototype_mscs_torch(cp, Xc, yc, Xt, yt, classes, 0.1, 0.05, M,
                                              device="cuda", **common)
    assert all(sorted(a) == sorted(b) for a, b in zip(sets_cpu, sets_gpu)), \
        "prototype GPU MS-CS sets differ from CPU"
    dev = "cuda" if torch.cuda.is_available() else "cpu(torch)"
    print(f"test_mscs_gpu_parity: PASS (sets identical, ran on {dev})")


# ----------------------------------------------------------------------
# Informational: set-size of the 3 rungs (no assert)
# ----------------------------------------------------------------------
def report_setsize_3rungs():
    K, d = 20, 16
    X, y = make_gmm(K=K, per_class=220, d=d, seed=2, sep=3.0)
    classes = np.unique(y)
    Tf = PrototypeSoftmaxNCM(temperature=None,
                             allow_nonexchangeable=True).fit(X[:2000], y[:2000])._T
    print("  set-size (cov) by NCM, 10 trials, alpha=0.1:")
    for cal_size in (200, 400, 800):
        rows = {}
        for name, mk in (
            ("prototype_softmax", lambda: create_ncm("prototype_softmax", temperature=Tf)),
            ("ridge_softmax(la=1)", lambda: create_ncm("ridge_softmax", temperature=Tf)),
            ("unwhitened_topk_asym", lambda: create_ncm("unwhitened_topk_asym", k=5)),
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
    test_coverage()
    test_small_cal_bloat()
    test_edge_cases()
    test_temperature_invariance()
    test_gpu_parity()
    test_mscs_gpu_parity()
    print("\n--- informational ---")
    report_setsize_3rungs()
    print("\nALL PROTOTYPE-SOFTMAX TESTS PASSED")
