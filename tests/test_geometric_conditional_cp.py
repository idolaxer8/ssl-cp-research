"""
Validity tests for geometry-conditional Full CP (src/geometric_conditional_cp.py).

Gates the Stage-2 guarantees with synthetic data (no embeddings needed):
  1. global_sets reproduces FullConformalPredictor static-mode sets (design C sound).
  2. Mondrian per-stratum CP is exactly valid on a RANDOM (exchangeable) split:
     marginal coverage >= 1-alpha and per-stratum coverage >= 1-alpha (up to MC + the
     1/(n_g+1) Mondrian slack). Strata are a FIXED label-free function of x.
  3. Empty-set invariant: when a stratum threshold excludes all candidates the set is
     EMPTY (size 0), never the full-K set by fallback.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
try:
    import pytest
except ImportError:                                  # allow running as a plain script
    class _P:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda f: f
    pytest = _P()

from conformal_prediction import FullConformalPredictor, create_ncm
from geometric_conditional_cp import global_sets, mondrian_sets, rlcp_sets


def make_gmm(K, per, d, rng, sep=3.0):
    mus = rng.normal(size=(K, d)) * sep
    X, y = [], []
    for c in range(K):
        X.append(mus[c] + rng.normal(size=(per, d)))
        y += [c] * per
    return np.concatenate(X), np.array(y)


def coord_strata(X, edges):
    """Fixed, label-free stratifier: bin by first coordinate (pool-fixed edges)."""
    return np.clip(np.digitize(X[:, 0], edges[1:-1]), 0, len(edges) - 2)


def test_global_sets_matches_fcp_static():
    rng = np.random.default_rng(0)
    K, d = 6, 16
    Xc, yc = make_gmm(K, 30, d, rng)
    Xt, yt = make_gmm(K, 20, d, rng)
    allc = np.arange(K)
    cp = FullConformalPredictor(create_ncm("unwhitened_topk_asym", k=5), alpha=0.1)
    cp.calibrate(Xc, yc, all_classes=allc)
    res = cp.predict(Xt, update_calibration_scores=False, return_test_scores=True, verbose=False)
    gs = global_sets(res["test_scores"], res["cal_scores"], res["classes"], 0.1)
    for a, b in zip(res["prediction_sets"], gs):
        assert set(a) == set(b), "global_sets must reproduce FCP static-mode sets"


@pytest.mark.parametrize("ncm_name", ["unwhitened_topk_asym", "prototype_softmax"])
def test_mondrian_valid_random_split(ncm_name):
    """Random (exchangeable) split => Mondrian marginal & per-stratum coverage valid."""
    alpha, K, d, G, n_strata = 0.1, 5, 16, 4, 4
    edges = np.quantile(np.random.default_rng(1).normal(size=5000), np.linspace(0, 1, G + 1))
    cov_all, strat_all = [], []
    for t in range(40):
        rng = np.random.default_rng(100 + t)
        X, y = make_gmm(K, 80, d, rng)          # 400 points, K=5
        idx = rng.permutation(len(X))
        ci, ti = idx[:200], idx[200:]           # uniform random split (exact)
        Xc, yc, Xt, yt = X[ci], y[ci], X[ti], y[ti]
        kw = dict(temperature=0.3, logit="cosine") if "prototype" in ncm_name else dict(k=5)
        cp = FullConformalPredictor(create_ncm(ncm_name, **kw), alpha=alpha)
        cp.calibrate(Xc, yc, all_classes=np.arange(K))
        res = cp.predict(Xt, return_test_scores=True, verbose=False)
        s_cal, s_test = coord_strata(Xc, edges), coord_strata(Xt, edges)
        sets = mondrian_sets(res["test_scores"], res["cal_scores"], s_test, s_cal,
                             res["classes"], alpha)
        cov_all.append(np.array([yt[i] in set(s) for i, s in enumerate(sets)]))
        strat_all.append(s_test)
    cov = np.concatenate(cov_all); strat = np.concatenate(strat_all)
    assert cov.mean() >= 1 - alpha - 0.02, f"marginal under-covers: {cov.mean():.3f}"
    assert cov.mean() <= 0.98, f"marginal grossly over-covers: {cov.mean():.3f}"
    for g in range(n_strata):
        m = strat == g
        if m.sum() > 200:
            assert cov[m].mean() >= 1 - alpha - 0.03, \
                f"stratum {g} under-covers: {cov[m].mean():.3f}"


def test_empty_set_invariant_no_allclasses_fallback():
    """A test point whose every candidate score exceeds the stratum quantile must get
    an EMPTY set, never the full-K set."""
    classes = np.arange(5)
    cal_scores = np.linspace(0.0, 1.0, 100)          # one global stratum
    strata_cal = np.zeros(100, dtype=int)
    # test point with all candidate scores ABOVE everything in cal -> p ~ 1/(n+1) <= alpha
    test_scores = np.full((1, 5), 5.0)
    strata_test = np.zeros(1, dtype=int)
    sets = mondrian_sets(test_scores, cal_scores, strata_test, strata_cal, classes,
                         alpha=0.1, n_min=10)
    assert sets[0] == [], "must return EMPTY set, not all-K fallback"
    # and a very conforming point (score below all cal) -> non-empty
    sets2 = mondrian_sets(np.full((1, 5), -1.0), cal_scores, strata_test, strata_cal,
                          classes, alpha=0.1, n_min=10)
    assert len(sets2[0]) == 5


def test_rlcp_marginal_valid_random_split():
    """RLCP randomized localized CP is exactly marginally valid on a random split."""
    alpha, K, d = 0.1, 5, 16
    cov = []
    for t in range(40):
        rng = np.random.default_rng(200 + t)
        X, y = make_gmm(K, 80, d, rng)
        idx = rng.permutation(len(X)); ci, ti = idx[:200], idx[200:]
        cp = FullConformalPredictor(create_ncm("unwhitened_topk_asym", k=5), alpha=alpha)
        cp.calibrate(X[ci], y[ci], all_classes=np.arange(K))
        res = cp.predict(X[ti], return_test_scores=True, verbose=False)
        v_cal, v_test = X[ci][:, 0], X[ti][:, 0]      # label-free covariate
        h = 0.5 * (np.percentile(X[:, 0], 75) - np.percentile(X[:, 0], 25))
        sets = rlcp_sets(res["test_scores"], res["cal_scores"], v_test, v_cal,
                         res["classes"], alpha, h, rng)
        cov.append(np.mean([y[ti][i] in set(s) for i, s in enumerate(sets)]))
    assert 1 - alpha - 0.02 <= np.mean(cov) <= 0.98, f"RLCP marginal {np.mean(cov):.3f}"
