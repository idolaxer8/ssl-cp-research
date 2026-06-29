"""
Stage 2 — geometry-conditional Full CP (close the geometric coverage gap).

Mondrian Full CP over LABEL-FREE geometric strata (design C): one global NCM
(strong neighbour pool, all classes) gives the test-score matrix s(x,y) and the
static LOO calibration scores; the conformal THRESHOLD is then computed PER
STRATUM. A test point is ranked only against calibration points in its own
pool-fixed geometric stratum, so

    p_g(x,y) = (1 + #{ j in cal : g(x_j)=g(x), s_j >= s(x,y) }) / (n_g + 1)
    C(x)     = { y : p_g(x,y) > alpha }

Per-stratum guarantee P(Y in C | g) >= 1-alpha (exact up to O(1/n_g), the same
concession as the SCP-geodesic==FCP result). Empty sets are ALWAYS allowed and
NEVER replaced by the all-classes set (that old fallback over-covered above the
1-alpha+1/(n+1) bound and broke validity). Fewer empties in sparse strata come
only from the correctly-calibrated higher local threshold.

Strata are a fixed function of the unlabeled pool (theory.md Prop 2) => exactly
exchangeable. Reuses the Stage-1 geometric helpers in
geometric_coverage_experiment.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.neighbors import NearestNeighbors

from geometric_coverage_experiment import knn_radii, density_cov, lid_mle, strata_from_edges


class GeometryStratifier:
    """Label-free geometric stratifier fit on the unlabeled pool (exchangeable).

    mode: 'density' (k-th NN radius; sparse=large) or 'lid' (Levina-Bickel local
    intrinsic dimension). Bin edges are pool quantiles -> fixed given the pool.
    """

    def __init__(self, transform, mode="density", n_strata=5, knn=10):
        assert mode in ("density", "lid")
        self.transform = transform
        self.mode = mode
        self.n_strata = n_strata
        self.knn = knn
        Xu_t = transform.Xu_transformed_
        self.nn = NearestNeighbors(n_neighbors=knn + 1).fit(Xu_t)
        pool_cov = self._cov(knn_radii(self.nn, Xu_t, knn, drop_self=True))
        self.edges = np.quantile(pool_cov, np.linspace(0, 1, n_strata + 1))
        q75, q25 = np.percentile(pool_cov, [75, 25])
        self.pool_iqr = float(q75 - q25)            # bandwidth scale for RLCP

    def _cov(self, radii):
        return density_cov(radii) if self.mode == "density" else lid_mle(radii)

    def covariate_of(self, X_transformed):
        """Continuous label-free covariate v(x) (for RLCP localization)."""
        rad = knn_radii(self.nn, np.asarray(X_transformed), self.knn, drop_self=False)
        return self._cov(rad)

    def stratum_of(self, X_transformed):
        """Assign each (already pool-transformed) point to its geometric stratum."""
        return strata_from_edges(self.covariate_of(X_transformed), self.edges)


class ConfidenceStratifier:
    """CONTROL stratifier: bins by the model's own confidence (max posterior prob)
    instead of geometry. Tests whether geometry is NECESSARY -- if conditioning on
    confidence also flattens the geometric CovGap, geometry adds nothing.

    Prototype-softmax only (needs a posterior). Built per-trial from the fitted NCM;
    edges = pool-confidence quantiles. Cal-fit => only O(1/n)-exchangeable (a control).
    """

    def __init__(self, ncm, transform, n_strata=5):
        self.ncm = ncm
        self.n_strata = n_strata
        pool_conf = self._conf(transform.Xu_transformed_)
        self.edges = np.quantile(pool_conf, np.linspace(0, 1, n_strata + 1))

    def _conf(self, X_transformed):
        Z = np.asarray(X_transformed, dtype=np.float64)
        if getattr(self.ncm, "cosine", True):
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), self.ncm.eps)
        logits = (Z @ self.ncm.P) / float(self.ncm._T)            # (n, Kc)
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(axis=1, keepdims=True)
        return p.max(axis=1)                                      # max posterior = confidence

    def stratum_of(self, X_transformed):
        return strata_from_edges(self._conf(X_transformed), self.edges)


def _pvalues_against(sorted_ref, test_row):
    """Conformal p-values of a test point's K candidate scores against a SORTED
    (ascending) reference score vector: p = (1 + #{ref >= s}) / (n_ref + 1)."""
    n_ref = len(sorted_ref)
    # #{ref >= s} = n_ref - #{ref < s} = n_ref - searchsorted(side='left')
    n_geq = n_ref - np.searchsorted(sorted_ref, test_row, side="left")
    return (n_geq + 1.0) / (n_ref + 1.0)


def global_sets(test_scores, cal_scores, classes, alpha):
    """Plain (global-threshold) static FCP sets from a shared score matrix. This is
    the G=1 special case of Mondrian and the baseline the Mondrian arm must beat on
    CovGap_geo (it reproduces standard FCP set-for-set at O(1/n))."""
    classes = np.asarray(classes)
    ref = np.sort(np.asarray(cal_scores))
    sets = []
    for i in range(len(test_scores)):
        p = _pvalues_against(ref, test_scores[i])
        sets.append(classes[p > alpha].astype(int).tolist())
    return sets


def mondrian_sets(test_scores, cal_scores, strata_test, strata_cal, classes, alpha,
                  n_min=None, return_diag=False):
    """Mondrian per-stratum-threshold FCP sets (design C).

    test_scores: (n_test, K) s(x,y); column j -> classes[j].
    cal_scores:  (n_cal,) static LOO scores.
    strata_test/strata_cal: int stratum id per test/cal point (pool-fixed).
    n_min: strata with fewer than n_min cal points fall back to the GLOBAL ref
           (mirror ClusterCP null cluster). Default max(ceil(1/alpha), 10).
    Empty sets allowed; NO all-classes fallback ever.
    """
    classes = np.asarray(classes)
    if n_min is None:
        n_min = max(int(np.ceil(1.0 / alpha)), 10)
    cal_scores = np.asarray(cal_scores)
    global_ref = np.sort(cal_scores)
    ref_by_stratum, n_by_stratum, fellback = {}, {}, set()
    for g in np.unique(strata_cal):
        ref_g = np.sort(cal_scores[strata_cal == g])
        n_by_stratum[g] = len(ref_g)
        if len(ref_g) < n_min:
            ref_by_stratum[g] = global_ref
            fellback.add(int(g))
        else:
            ref_by_stratum[g] = ref_g

    sets = []
    for i in range(len(test_scores)):
        g = int(strata_test[i])
        ref = ref_by_stratum.get(g, global_ref)
        p = _pvalues_against(ref, test_scores[i])
        sets.append(classes[p > alpha].astype(int).tolist())
    if return_diag:
        diag = {"n_by_stratum": {int(k): int(v) for k, v in n_by_stratum.items()},
                "fellback_strata": sorted(fellback), "n_min": int(n_min)}
        return sets, diag
    return sets


def rlcp_sets(test_scores, cal_scores, v_test, v_cal, classes, alpha, h, rng):
    """RLCP — Randomized Localized CP (Hore & Barber 2024, arXiv:2310.07850).

    Continuous geometry conditioning: weight each cal point by a Gaussian kernel on
    the LABEL-FREE covariate v (pool-fixed center v(x) => exchangeable). A per-test
    randomization U restores EXACT marginal coverage:

        w_i = exp(-(v(x)-v(x_i))^2 / (2 h^2)),  w_x = 1
        p(x,y) = ( sum_i w_i 1{s_i > s(x,y)} + U (w_x + sum_i w_i 1{s_i == s(x,y)}) )
                 / (w_x + sum_i w_i)
        include y iff p(x,y) > alpha       (one U per test point; shared across y)

    h -> inf recovers global FCP; h -> 0 approaches per-point conditioning. Empty
    sets allowed; no all-classes fallback.
    """
    classes = np.asarray(classes)
    cal = np.asarray(cal_scores)[:, None]                 # (n_cal, 1)
    inv2h2 = 1.0 / (2.0 * h * h)
    sets = []
    for i in range(len(test_scores)):
        w = np.exp(-((v_test[i] - v_cal) ** 2) * inv2h2)  # (n_cal,)
        W = 1.0 + w.sum()
        U = float(rng.random())
        s = test_scores[i][None, :]                       # (1, K)
        wc = w[:, None]
        A = (wc * (cal > s)).sum(0)                       # (K,)
        Tie = (wc * (cal == s)).sum(0)
        p = (A + U * (1.0 + Tie)) / W
        sets.append(classes[p > alpha].astype(int).tolist())
    return sets
