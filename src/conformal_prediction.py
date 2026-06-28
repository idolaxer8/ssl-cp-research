"""
Full Conformal Prediction with Optimizations
Based on: Cherubin et al. 2021 - "Exact and Approximate Conformal Inference for Multi-Output Regression"
https://proceedings.mlr.press/v139/cherubin21a/cherubin21a.pdf

This implementation provides:
1. Full Conformal Prediction (FCP) - exact p-values for each test example
2. Efficient nonconformity measures for SSL embeddings
3. Optimizations: caching, vectorization, early stopping
"""

import math
import time
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Exchangeability guard
# ----------------------------------------------------------------------------
# FCP's 1 - alpha coverage guarantee requires every nonconformity score to be a
# permutation-symmetric function of the calibration+test bag. The DEFAULT path
# in this codebase is fully exchangeable: data-dependent transforms (whitening,
# PCA, k-means, RBF bandwidth) are fit on an INDEPENDENT unlabeled pool -- see
# exchangeable_features.UnlabeledTransform -- which is a fixed map w.r.t. the
# bag. Any variant that instead fits such a transform on the CALIBRATION set
# breaks exchangeability (even at O(1/n)) and voids the exact guarantee. Every
# such variant must route through warn_nonexchangeable() so the loss of validity
# is surfaced, and must be explicitly approved via allow_nonexchangeable=True.

class ExchangeabilityWarning(UserWarning):
    """Emitted when a code path breaks FCP's exchangeability assumption, so the
    1 - alpha coverage guarantee is no longer exactly valid."""
    pass


def warn_nonexchangeable(source: str, detail: str = "",
                         order: str = "O(1/n)", allow: bool = False) -> None:
    """Surface a validity warning when a non-exchangeable (cal-fit /
    data-dependent) transform is used in the FCP path.

    The fully-exchangeable path (transforms fit on an independent unlabeled
    pool, see exchangeable_features.UnlabeledTransform) is the DEFAULT and never
    calls this. Any variant that breaks exchangeability -- even at O(1/n) -- must
    route through here so the loss of the coverage guarantee is explicit.

    Soft gate (warn-only): the offending variant still runs and is numerically
    unchanged; this only emits a warning. Pass allow=True (the caller's explicit
    approval via allow_nonexchangeable=True) to accept the broken guarantee and
    silence the warning.

    Args:
        source: short name of the offending transform (e.g. "cal-fit whitening").
        detail: how to recover exactness (e.g. point to the unlabeled-pool path).
        order:  magnitude of the exchangeability break ("O(1/n)", "O(1/n^2)", ...).
        allow:  approval flag. If True the caller has explicitly accepted the
                broken guarantee, so the warning is suppressed.
    """
    if allow:
        return
    warnings.warn(
        f"VALIDITY WARNING: {source} breaks FCP exchangeability ({order}); the "
        f"1 - alpha coverage guarantee is NOT exactly valid. "
        f"{('' if not detail else detail + ' ')}"
        f"The exchangeable default fits this transform on an independent "
        f"unlabeled pool (exchangeable_features.UnlabeledTransform). "
        f"Pass allow_nonexchangeable=True to approve and silence this warning.",
        ExchangeabilityWarning, stacklevel=3,
    )


class NonconformityMeasure:
    """Base class for nonconformity measures."""

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Fit on calibration data."""
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity score for examples.
        Higher score = more nonconforming = less confident
        """
        raise NotImplementedError

    def score_x_cv(self, x: np.ndarray, y: int) -> float:
        """
        Score a single point for CV+/Jackknife+ (no hypothetical addition).

        For most NCMs this is identical to score_x. Override in NCMs where
        score_x includes a hypothetical test-point addition (e.g. Ridge with
        Sherman-Morrison).
        """
        return self.score_x(x, y)


class MahalNNRatio(NonconformityMeasure):
    """
    Point-to-Point Mahalanobis Nearest-Neighbor Ratio NCM.

    Uses Mahalanobis distance with a pooled within-class diagonal covariance (LDA-style).

    Score:
        alpha_i = min_{j: y_j=y_i, j!=i} mahal(xi, xj)
                  / min_{k: y_k!=y_i}       mahal(xi, xk)

    where  mahal(a, b) = sqrt( sum_d  (a_d - b_d)^2 / (pooled_var_d + reg) )

    This is equivalent to running nn_ratio in a "whitened" feature space where each
    dimension d is rescaled by 1/sqrt(pooled_var_d).  Dimensions with high within-class
    variance are compressed (less informative); tight dimensions are amplified.

    Why pooled (LDA-style) variance?
    - With K=100 classes and n_cal=300 points, per-class has only ~3 samples in 768D
      => completely unreliable.
    - Pooled var uses all n_cal points (one shared estimator): ~100x more stable.

    Exchangeability (O(1/n) approximation):
    - pooled_var is fixed at fit() time from calibration data only. For exact
      exchangeability, pooled_var should be recomputed on the augmented bag
      {z1,...,zn,(x*,y*)} for each candidate — but this would break O(1) updates.
    - The asymmetry is O(1/n): adding one point to n calibration points changes
      pooled_var by O(1/n), so the metric shift is negligible for n >= 200.
    - Empirically valid at all tested calibration sizes (coverage matches theory).
    - See Fan & Sesia (2025, arXiv 2512.15383) "transductive standardization"
      for formal analysis of data-dependent preprocessing in conformal prediction.
    """

    def __init__(self, reg: float = 1e-4, allow_nonexchangeable: bool = False):
        self.reg = reg
        # MahalNNRatio ALWAYS fits pooled within-class whitening on the
        # calibration set, so it is intrinsically non-exchangeable (O(1/n)).
        # allow_nonexchangeable=True approves this and silences the warning.
        self.allow_nonexchangeable = allow_nonexchangeable
        self.X_cal = None
        self.y_cal = None
        self.inv_std = None      # (d,) element-wise 1/sqrt(pooled_var + reg)
        self.lookup_same = None  # (n_cal,) min Mahal dist to same-class neighbor
        self.lookup_other = None # (n_cal,) min Mahal dist to other-class neighbor
        self.alpha0 = None
        # Cache: per-test-point Mahalanobis distances (same x, different y_candidates)
        self._mcache_key = None
        self._mcache_dists = None

    def _compute_mahal_dists_from_x(self, x: np.ndarray) -> np.ndarray:
        """Mahalanobis distances from x to all cal points, with per-x caching.

        Uses x.ctypes.data (stable pointer into X_test's buffer) as cache key,
        so the O(n*d) computation runs once per test point across all y-candidates.
        """
        cache_key = x.ctypes.data
        if cache_key != self._mcache_key:
            diff = (x - self.X_cal) * self.inv_std
            self._mcache_dists = np.sqrt(np.maximum((diff * diff).sum(axis=1), 0))
            self._mcache_key = cache_key
        return self._mcache_dists

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        warn_nonexchangeable(
            "MahalNNRatio cal-fit pooled within-class whitening",
            "Use create_ncm('unwhitened_topk_*') with an unlabeled-pool transform "
            "for an exact guarantee.",
            order="O(1/n)", allow=self.allow_nonexchangeable)
        self.X_cal = X_cal
        self.y_cal = y_cal
        n_cal = len(X_cal)
        classes = np.unique(y_cal)

        # 1. Pooled within-class diagonal variance
        residuals = X_cal.copy()
        for c in classes:
            idx = np.where(y_cal == c)[0]
            residuals[idx] -= X_cal[idx].mean(axis=0)
        pooled_var = (residuals ** 2).mean(axis=0)
        self.inv_std = 1.0 / np.sqrt(pooled_var + self.reg)

        # 2. Whitened calibration matrix for fast pairwise distance
        X_w = X_cal * self.inv_std   # (n_cal, d)

        # 3. Full pairwise Mahalanobis distance matrix (= Euclidean in whitened space)
        sq = np.sum(X_w ** 2, axis=1, keepdims=True)   # (n_cal, 1)
        D_sq = sq + sq.T - 2.0 * (X_w @ X_w.T)         # (n_cal, n_cal)
        D = np.sqrt(np.maximum(D_sq, 0))
        np.fill_diagonal(D, np.inf)

        # 4. Build lookup tables for NN-ratio lookup
        self.lookup_same  = np.full(n_cal, np.inf)
        self.lookup_other = np.full(n_cal, np.inf)
        for c in classes:
            idx_same  = np.where(y_cal == c)[0]
            idx_other = np.where(y_cal != c)[0]
            if len(idx_same) > 1:
                self.lookup_same[idx_same] = D[np.ix_(idx_same, idx_same)].min(axis=1)
            if len(idx_other) > 0:
                self.lookup_other[idx_same] = D[np.ix_(idx_same, idx_other)].min(axis=1)

        # 5. Baseline calibration scores
        self.alpha0 = self.lookup_same / (self.lookup_other + 1e-8)
        self.alpha0[~np.isfinite(self.alpha0)] = 1e9

    def get_calibration_scores(self) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()

    def score_x(self, x: np.ndarray, y: int) -> float:
        dists = self._compute_mahal_dists_from_x(x)
        mask_same  = (self.y_cal == y)
        mask_other = (self.y_cal != y)
        if not np.any(mask_same):
            return 1e9
        d_same  = np.min(dists[mask_same])
        d_other = np.min(dists[mask_other]) if np.any(mask_other) else 1e-8
        return float(d_same / (d_other + 1e-8))

    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        dists_x = self._compute_mahal_dists_from_x(x)
        updated = self.alpha0.copy()

        # CASE A: same-class points — x is a new same-class neighbor candidate
        idx_same = np.where(self.y_cal == y)[0]
        if len(idx_same) > 0:
            new_same = np.minimum(self.lookup_same[idx_same], dists_x[idx_same])
            updated[idx_same] = new_same / (self.lookup_other[idx_same] + 1e-8)

        # CASE B: other-class points — x is a new other-class neighbor candidate
        idx_other = np.where(self.y_cal != y)[0]
        if len(idx_other) > 0:
            new_other = np.minimum(self.lookup_other[idx_other], dists_x[idx_other])
            updated[idx_other] = self.lookup_same[idx_other] / (new_other + 1e-8)

        return updated


class WhitenedGeodesicNNRatio(NonconformityMeasure):
    """
    Whitened Geodesic NN Ratio NCM: pooled whitening + L2 renorm + geodesic arccos.

    Combines two orthogonal improvements over plain nn_ratio:
    1. Dimension whitening (pooled within-class var) — amplifies discriminative dims
    2. Geodesic (arccos) metric on the re-normalized unit sphere — natural for SSL

    Pipeline per embedding x:
        x' = x * inv_std              (dimension-wise whitening, inv_std = 1/sqrt(pv+reg))
        z  = x' / ||x'||              (project back onto unit sphere S^{d-1})
        score = arccos(max_same_cos(z)) / (arccos(max_other_cos(z)) + eps)

    Exchangeability (O(1/n) approximation):
    - Whitening (inv_std) is fixed at fit() time from calibration data only.
      For exact exchangeability, pooled_var should be recomputed on the augmented
      bag {z1,...,zn,(x*,y*)} for each candidate label. The asymmetry is O(1/n)
      and negligible for n >= 200.
    - After whitening + renorm, Full CP update is identical to geodesic NN ratio:
      only lookup_same_sim / lookup_other_sim change by a max() comparison.
    - See Fan & Sesia (2025, arXiv 2512.15383) on transductive standardization.
    """

    def __init__(self, reg: float = 1e-4, allow_nonexchangeable: bool = False):
        """
        Args:
            reg: Whitening regularization floor (adaptive version used in fit()).
            allow_nonexchangeable: WhitenedGeodesicNNRatio ALWAYS fits pooled
                whitening on the calibration set, so it is intrinsically
                non-exchangeable (O(1/n)). Set True to approve and silence the
                validity warning.
        """
        self.reg = reg
        self.allow_nonexchangeable = allow_nonexchangeable
        self.inv_std = None
        self.X_cal_wn = None        # whitened + L2-normalized cal embeddings (n_cal, d)
        self.y_cal = None
        self.lookup_same_sim  = None  # (n_cal,) max cosine sim to same-class neighbor
        self.lookup_other_sim = None  # (n_cal,) max cosine sim to other-class neighbor
        self.alpha0 = None
        # Cache: per-test-point whitened similarities
        self._wcache_key = None
        self._wcache_sims = None

    @staticmethod
    def _geodesic_ratio(same_sim: np.ndarray, other_sim: np.ndarray) -> np.ndarray:
        """Compute arccos(same) / arccos(other) ratio."""
        eps = 1e-8
        d_same  = np.arccos(np.clip(same_sim,  -1.0, 1.0))
        d_other = np.arccos(np.clip(other_sim, -1.0, 1.0))
        return d_same / (d_other + eps)

    def _whiten_normalize(self, X: np.ndarray) -> np.ndarray:
        """Whiten then L2-normalize each row."""
        X_w = X * self.inv_std
        norms = np.linalg.norm(X_w, axis=1, keepdims=True)
        return X_w / np.maximum(norms, 1e-10)

    def _get_sims(self, x: np.ndarray) -> np.ndarray:
        """Cosine sims from whitened-normalized x to all cal points, with caching.

        Uses x.ctypes.data as cache key (stable view pointer into X_test's buffer).
        """
        cache_key = x.ctypes.data
        if cache_key != self._wcache_key:
            z = self._whiten_normalize(x.reshape(1, -1))[0]
            self._wcache_sims = self.X_cal_wn @ z
            self._wcache_key = cache_key
        return self._wcache_sims

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        warn_nonexchangeable(
            "WhitenedGeodesicNNRatio cal-fit pooled whitening",
            "Use create_ncm('unwhitened_topk_*') with an unlabeled-pool transform "
            "for an exact guarantee.",
            order="O(1/n)", allow=self.allow_nonexchangeable)
        self.y_cal = y_cal
        n_cal = len(X_cal)
        classes = np.unique(y_cal)

        # 1. Pooled within-class diagonal variance
        residuals = X_cal.copy()
        for c in classes:
            idx = np.where(y_cal == c)[0]
            residuals[idx] -= X_cal[idx].mean(axis=0)
        pooled_var = (residuals ** 2).mean(axis=0)
        # Fix 4: adaptive reg caps noise-amplification on well-separated datasets.
        # Fixed reg=1e-4 lets 1/sqrt(var+reg) blow up for near-zero-var dims, amplifying
        # float noise into apparent angular distances. Bounding reg >= 1% of median(pooled_var)
        # limits the worst-case amplification factor to ~10×.
        adaptive_reg = max(self.reg, 0.01 * float(np.median(pooled_var)))
        self.inv_std = 1.0 / np.sqrt(pooled_var + adaptive_reg)

        # 2. Whiten + L2-normalize calibration embeddings
        self.X_cal_wn = self._whiten_normalize(X_cal)  # (n_cal, d)

        # 3. Cosine similarity matrix in whitened space
        S = self.X_cal_wn @ self.X_cal_wn.T
        np.fill_diagonal(S, -np.inf)

        # 4. Build lookup tables (same structure as geodesic NN ratio)
        self.lookup_same_sim  = np.full(n_cal, -1.0)
        self.lookup_other_sim = np.full(n_cal, -1.0)
        for c in classes:
            idx_same  = np.where(y_cal == c)[0]
            idx_other = np.where(y_cal != c)[0]
            if len(idx_same) > 1:
                self.lookup_same_sim[idx_same] = S[np.ix_(idx_same, idx_same)].max(axis=1)
            if len(idx_other) > 0:
                self.lookup_other_sim[idx_same] = S[np.ix_(idx_same, idx_other)].max(axis=1)

        self.alpha0 = self._geodesic_ratio(self.lookup_same_sim, self.lookup_other_sim)
        self.alpha0[~np.isfinite(self.alpha0)] = 1e9

    def get_calibration_scores(self) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()

    def score_x(self, x: np.ndarray, y: int) -> float:
        sims = self._get_sims(x)
        mask_same  = (self.y_cal == y)
        if not np.any(mask_same):
            return 1e9
        max_same  = float(np.max(sims[mask_same]))
        max_other = float(np.max(sims[~mask_same])) if np.any(~mask_same) else -1.0
        return float(self._geodesic_ratio(np.array([max_same]), np.array([max_other]))[0])

    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        sims_x = self._get_sims(x)

        updated_same  = self.lookup_same_sim.copy()
        updated_other = self.lookup_other_sim.copy()

        idx_same = np.where(self.y_cal == y)[0]
        if len(idx_same) > 0:
            updated_same[idx_same] = np.maximum(self.lookup_same_sim[idx_same], sims_x[idx_same])

        idx_other = np.where(self.y_cal != y)[0]
        if len(idx_other) > 0:
            updated_other[idx_other] = np.maximum(self.lookup_other_sim[idx_other], sims_x[idx_other])

        scores = self._geodesic_ratio(updated_same, updated_other)
        scores[~np.isfinite(scores)] = 1e9
        return scores


class GeodesicTopKMeanNCM(NonconformityMeasure):
    """
    Geodesic Top-k Mean NCM: whitened geodesic ratio using k-NN averaged cosine similarity.

    Addresses the Trojan Horse degeneracy of 1-NN ratio NCMs on well-separated datasets.

    Three modes controlled by `topk_same` / `topk_other`:

    Symmetric  (topk_same=True,  topk_other=True):   "geodesic_topk_mean"
        score = arccos(mean_k(sim_same)) / (arccos(mean_k(sim_other)) + eps)
        Top-k on both sides. Best at small cal; slightly worse than whitened_geodesic
        at large cal because denominator grows with averaging.

    Asymmetric (topk_same=False, topk_other=True):   "geodesic_topk_asym"  ← recommended
        score = arccos(max(sim_same)) / (arccos(mean_k(sim_other)) + eps)
        1-NN numerator keeps correct-class scores tight; k-NN mean denominator dilutes
        the Trojan Horse collapse. Best of both worlds: smaller sets at all cal sizes.

    Sym-wrong  (topk_same=True,  topk_other=False):  not registered — hurts efficiency.

    k selection (adaptive): k = max(1, min(K_MAX, n_cal // n_classes)).
    At k=1 all modes reduce to WhitenedGeodesicNNRatio.

    Exchangeability:
    - DEFAULT whiten=False is fully exchangeable: the NCM applies only the
      per-point L2 normalisation + geodesic top-k, which is a fixed map. Feed it
      features from an exchangeable_features.UnlabeledTransform to recover the
      whitening benefit while keeping the exact guarantee.
    - whiten=True (must be approved via allow_nonexchangeable=True) fixes inv_std
      at fit() time from calibration data only, which breaks exchangeability at
      O(1/n): for exactness pooled_var would have to be recomputed on the
      augmented bag {z1,...,zn,(x*,y*)} per candidate. Negligible for n >= 200 but
      NOT exact. See Fan & Sesia (2025, arXiv 2512.15383).

    FCP O(N) compatibility: CASE A update is O(1) per calibration point.
    """

    K_MAX = 5  # cap; diminishing Trojan-Horse protection beyond k=5

    def __init__(self, reg: float = 1e-4, k: Optional[int] = None,
                 topk_same: bool = True, topk_other: bool = True,
                 numerator_only: bool = False, whiten: bool = False,
                 allow_nonexchangeable: bool = False):
        """
        Args:
            reg:            Whitening regularization floor.
            k:              Neighbors for mean pooling. None → adaptive.
            topk_same:      If True, average same-class sims (numerator). Default True.
            topk_other:     If True, average other-class sims (denominator). Default True.
            numerator_only: If True, score = d_same only (no ratio). Avoids denominator
                            collapse on well-separated datasets. Ignores topk_other.
            whiten:         If True, fit pooled within-class whitening on the
                            CALIBRATION set. This breaks exchangeability (O(1/n))
                            and voids the exact coverage guarantee, so it now
                            DEFAULTS to False (fully-exchangeable path). To get the
                            whitening benefit exactly, feed features through an
                            exchangeable_features.UnlabeledTransform (whitening fit
                            on an independent unlabeled pool) and keep whiten=False.
            allow_nonexchangeable: Approve cal-fit whitening (whiten=True) and
                            silence the validity warning.
        """
        self.reg           = reg
        self.k_override    = k
        self.topk_same     = topk_same
        self.topk_other    = topk_other
        self.numerator_only = numerator_only
        self.whiten     = whiten
        self.allow_nonexchangeable = allow_nonexchangeable
        self.k          = None      # resolved at fit()
        self.inv_std    = None
        self.X_cal_wn   = None
        self.y_cal      = None
        # Same-class lookup — top-k or 1-NN
        self.sum_same_sims  = None   # topk_same only
        self.kth_same_sim   = None   # topk_same only
        self._k_same_eff    = None   # topk_same only
        self.lookup_same_sim = None  # not topk_same (max sim)
        # Other-class lookup — top-k or 1-NN
        self.sum_other_sims  = None  # topk_other only
        self.kth_other_sim   = None  # topk_other only
        self._k_other_eff    = None  # topk_other only
        self.lookup_other_sim = None  # not topk_other (max sim)
        self.alpha0 = None
        # Test-point similarity cache
        self._wcache_key  = None
        self._wcache_sims = None

    # ------------------------------------------------------------------
    # Preprocessing helpers (identical to WhitenedGeodesicNNRatio)
    # ------------------------------------------------------------------

    def _whiten_normalize(self, X: np.ndarray) -> np.ndarray:
        X_w = X * self.inv_std
        norms = np.linalg.norm(X_w, axis=1, keepdims=True)
        return X_w / np.maximum(norms, 1e-10)

    def _get_sims(self, x: np.ndarray) -> np.ndarray:
        cache_key = x.ctypes.data
        if cache_key != self._wcache_key:
            z = self._whiten_normalize(x.reshape(1, -1))[0]
            self._wcache_sims = self.X_cal_wn @ z
            self._wcache_key  = cache_key
        return self._wcache_sims

    # ------------------------------------------------------------------
    # Top-k helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _topk_stats_block(S_block: np.ndarray, k: int):
        """
        For each row in S_block compute (sum_topk, kth_sim, k_eff).
        -inf entries are excluded (used for the diagonal of same-class blocks).
        Returns three (n_rows,) arrays.
        """
        n_rows = S_block.shape[0]
        sums   = np.zeros(n_rows)
        kths   = np.full(n_rows, -1.0)
        k_effs = np.zeros(n_rows)
        for i in range(n_rows):
            row   = S_block[i]
            valid = row[np.isfinite(row)]
            ke    = min(k, len(valid))
            if ke == 0:
                continue
            if ke < len(valid):
                top = valid[np.argpartition(valid, -ke)[-ke:]]
            else:
                top = valid
            sums[i]   = top.sum()
            kths[i]   = top.min()
            k_effs[i] = ke
        return sums, kths, k_effs

    @staticmethod
    def _d_topk(sim_sum: np.ndarray, k_eff: np.ndarray) -> np.ndarray:
        """arccos of mean-k sim."""
        return np.arccos(np.clip(sim_sum / np.maximum(k_eff, 1), -1.0, 1.0))

    @staticmethod
    def _d_1nn(max_sim: np.ndarray) -> np.ndarray:
        """arccos of max sim (1-NN geodesic distance)."""
        return np.arccos(np.clip(max_sim, -1.0, 1.0))

    def _score_ratio(self,
                     same_sum, same_k, same_max,
                     other_sum, other_k, other_max) -> np.ndarray:
        """Compute score using the mode flags (topk_same / topk_other / numerator_only)."""
        d_same = self._d_topk(same_sum, same_k) if self.topk_same else self._d_1nn(same_max)
        if self.numerator_only:
            return d_same
        eps = 1e-8
        d_other = self._d_topk(other_sum, other_k) if self.topk_other else self._d_1nn(other_max)
        return d_same / (d_other + eps)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        self.y_cal = y_cal
        n_cal   = len(X_cal)
        classes = np.unique(y_cal)

        # Adaptive k
        n_per_class = n_cal // len(classes)
        self.k = self.k_override if self.k_override is not None \
                 else max(1, min(self.K_MAX, n_per_class))

        # Pooled whitening (Fix 4: adaptive reg) — skip if whiten=False (default,
        # fully-exchangeable). whiten=True fits on cal and breaks exchangeability.
        if self.whiten:
            warn_nonexchangeable(
                "GeodesicTopKMeanNCM cal-fit pooled whitening (whiten=True)",
                "Set whiten=False and source whitening from an "
                "exchangeable_features.UnlabeledTransform for an exact guarantee.",
                order="O(1/n)", allow=self.allow_nonexchangeable)
            residuals = X_cal.copy()
            for c in classes:
                idx = np.where(y_cal == c)[0]
                residuals[idx] -= X_cal[idx].mean(axis=0)
            pooled_var    = (residuals ** 2).mean(axis=0)
            adaptive_reg  = max(self.reg, 0.01 * float(np.median(pooled_var)))
            self.inv_std  = 1.0 / np.sqrt(pooled_var + adaptive_reg)
        else:
            self.inv_std = np.ones(X_cal.shape[1], dtype=np.float64)

        # Whiten + L2-normalize
        self.X_cal_wn = self._whiten_normalize(X_cal)

        # Cosine similarity matrix
        S = self.X_cal_wn @ self.X_cal_wn.T
        np.fill_diagonal(S, -np.inf)

        # Build lookup tables
        k = self.k
        if self.topk_same:
            self.sum_same_sims  = np.zeros(n_cal)
            self.kth_same_sim   = np.full(n_cal, -1.0)
            self._k_same_eff    = np.zeros(n_cal)
        else:
            self.lookup_same_sim = np.full(n_cal, -1.0)

        if not self.numerator_only:
            if self.topk_other:
                self.sum_other_sims = np.zeros(n_cal)
                self.kth_other_sim  = np.full(n_cal, -1.0)
                self._k_other_eff   = np.zeros(n_cal)
            else:
                self.lookup_other_sim = np.full(n_cal, -1.0)

        for c in classes:
            idx_same  = np.where(y_cal == c)[0]
            idx_other = np.where(y_cal != c)[0]

            # Same-class block (diagonal is -inf → excluded by _topk_stats_block)
            S_same = S[np.ix_(idx_same, idx_same)]
            if self.topk_same:
                s, th, ke = self._topk_stats_block(S_same, k)
                self.sum_same_sims[idx_same] = s
                self.kth_same_sim[idx_same]  = th
                self._k_same_eff[idx_same]   = ke
            else:
                # max per row, excluding -inf self
                self.lookup_same_sim[idx_same] = np.where(
                    np.isfinite(S_same), S_same, -np.inf
                ).max(axis=1)

            if not self.numerator_only and len(idx_other) > 0:
                S_other = S[np.ix_(idx_same, idx_other)]
                if self.topk_other:
                    s, th, ke = self._topk_stats_block(S_other, k)
                    self.sum_other_sims[idx_same] = s
                    self.kth_other_sim[idx_same]  = th
                    self._k_other_eff[idx_same]   = ke
                else:
                    self.lookup_other_sim[idx_same] = S_other.max(axis=1)

        _z = np.zeros(n_cal)
        self.alpha0 = self._score_ratio(
            self.sum_same_sims,  self._k_same_eff,   self.lookup_same_sim,
            getattr(self, 'sum_other_sims',  _z),
            getattr(self, '_k_other_eff',    _z + 1),
            getattr(self, 'lookup_other_sim', _z - 1),
        )
        self.alpha0[~np.isfinite(self.alpha0)] = 1e9

    def get_calibration_scores(self) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()

    def score_x(self, x: np.ndarray, y: int) -> float:
        sims = self._get_sims(x)
        k    = self.k
        mask_same  = (self.y_cal == y)
        mask_other = ~mask_same
        has_same = bool(np.any(mask_same))

        def _topk_scalar(sim_arr):
            ke = min(k, len(sim_arr))
            if ke == 0:
                return 0.0, 0.0
            top = sim_arr[np.argpartition(sim_arr, -ke)[-ke:]] if ke < len(sim_arr) else sim_arr
            return float(top.sum()), float(ke)

        # Same-class. When y is ABSENT from cal (no same-class neighbour) use the
        # SAME zero-neighbour convention fit() applies to a sole-member cal class
        # (topk: sum=0,k_eff=0 -> arccos(0)=pi/2; 1-NN: max_sim=-1 -> arccos(-1)=pi).
        # This keeps a test point and a cal point in the identical situation
        # EXCHANGEABLE. The old `return 1e9` made a test point whose class is
        # missing from cal non-exchangeable with cal points, which under-covered
        # coverage whenever classes were absent from cal (small-cal regime).
        if self.topk_same:
            sum_s, ke_s = _topk_scalar(sims[mask_same]) if has_same else (0.0, 0.0)
            max_s = None
        else:
            sum_s = ke_s = None
            max_s = float(np.max(sims[mask_same])) if has_same else -1.0

        # Other-class
        if not self.numerator_only and np.any(mask_other):
            if self.topk_other:
                sum_o, ke_o = _topk_scalar(sims[mask_other])
                max_o = None
            else:
                sum_o = ke_o = None
                max_o = float(np.max(sims[mask_other]))
        else:
            sum_o = ke_o = None; max_o = -1.0

        return float(self._score_ratio(
            np.array([sum_s  if sum_s  is not None else 0.0]),
            np.array([ke_s   if ke_s   is not None else 1.0]),
            np.array([max_s  if max_s  is not None else -1.0]),
            np.array([sum_o  if sum_o  is not None else 0.0]),
            np.array([ke_o   if ke_o   is not None else 1.0]),
            np.array([max_o  if max_o  is not None else -1.0]),
        )[0])

    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        sims_x = self._get_sims(x)
        k      = self.k
        n = len(self.y_cal)

        # -- Same-class update --
        idx_same = np.where(self.y_cal == y)[0]
        if self.topk_same:
            upd_sum_same = self.sum_same_sims.copy()
            upd_k_same   = self._k_same_eff.copy()
            if len(idx_same) > 0:
                sim  = sims_x[idx_same]
                full = self._k_same_eff[idx_same] >= k
                enter = full & (sim > self.kth_same_sim[idx_same])
                upd_sum_same[idx_same[enter]] += sim[enter] - self.kth_same_sim[idx_same[enter]]
                grow = ~full
                upd_sum_same[idx_same[grow]] += sim[grow]
                upd_k_same[idx_same[grow]]   += 1
            upd_max_same = None
        else:
            upd_sum_same = upd_k_same = None
            upd_max_same = self.lookup_same_sim.copy()
            if len(idx_same) > 0:
                upd_max_same[idx_same] = np.maximum(
                    self.lookup_same_sim[idx_same], sims_x[idx_same]
                )

        # -- Other-class update (skipped when numerator_only) --
        if self.numerator_only:
            upd_sum_other = upd_k_other = upd_max_other = None
        else:
            idx_other = np.where(self.y_cal != y)[0]
            if self.topk_other:
                upd_sum_other = self.sum_other_sims.copy()
                upd_k_other   = self._k_other_eff.copy()
                if len(idx_other) > 0:
                    sim  = sims_x[idx_other]
                    full = self._k_other_eff[idx_other] >= k
                    enter = full & (sim > self.kth_other_sim[idx_other])
                    upd_sum_other[idx_other[enter]] += sim[enter] - self.kth_other_sim[idx_other[enter]]
                    grow = ~full
                    upd_sum_other[idx_other[grow]] += sim[grow]
                    upd_k_other[idx_other[grow]]   += 1
                upd_max_other = None
            else:
                upd_sum_other = upd_k_other = None
                upd_max_other = self.lookup_other_sim.copy()
                if len(idx_other) > 0:
                    upd_max_other[idx_other] = np.maximum(
                        self.lookup_other_sim[idx_other], sims_x[idx_other]
                    )

        def _arr(v, fallback):
            return v if v is not None else np.full(n, fallback)

        scores = self._score_ratio(
            _arr(upd_sum_same,  0.0), _arr(upd_k_same,  1.0), _arr(upd_max_same,  -1.0),
            _arr(upd_sum_other, 0.0), _arr(upd_k_other, 1.0), _arr(upd_max_other, -1.0),
        )
        scores[~np.isfinite(scores)] = 1e9
        return scores

    # ------------------------------------------------------------------
    # NCM-consistent y_hat for MS-CS (variant A): argmax_c top-k mean sim.
    # ŷ(x) = the class whose top-k whitened-cosine neighbours are most similar
    # to x = argmin_c d_same(x,c) = the NCM numerator's own class prediction.
    # Reuses the whitened space; no raw-Euclidean distances. All quantities are
    # ones the NCM already computes during scoring, so this adds no redundant
    # work in the MS-CS loop (it replaces the separate 1-NN distance matrices).
    # ------------------------------------------------------------------

    def _topk_mean_per_class(self, sims: np.ndarray) -> np.ndarray:
        """Top-k mean similarity of one point to each class, from a (n_cal,)
        similarity vector. Returns means aligned to self._yh_classes."""
        classes = self._yh_classes
        means = np.full(len(classes), -np.inf)
        for j, c in enumerate(classes):
            sc = sims[self.y_cal == c]
            ke = min(self.k, len(sc))
            if ke == 0:
                continue
            top = sc[np.argpartition(sc, -ke)[-ke:]] if ke < len(sc) else sc
            means[j] = float(top.sum()) / ke
        return means

    def _ensure_cal_yhat(self):
        """Lazily compute, for every calibration point (LOO), its top-k mean
        similarity to each class and the resulting NCM prediction
        cal_y_hat = argmax_c. Reuses the whitened cal matrix (recomputed once on
        demand, not stored by fit) so plain-FCP runs are unaffected."""
        if getattr(self, "cal_y_hat", None) is not None:
            return
        if self.X_cal_wn is None:
            raise ValueError("Must call fit() before _ensure_cal_yhat()")
        S = self.X_cal_wn @ self.X_cal_wn.T
        np.fill_diagonal(S, -np.inf)                       # LOO: exclude self
        classes = np.unique(self.y_cal)
        n_cal, nC = len(self.y_cal), len(classes)
        self._yh_classes = classes
        self._yh_col = {int(c): j for j, c in enumerate(classes)}
        self.cal_topk_sum  = np.zeros((n_cal, nC))
        self.cal_topk_kth  = np.full((n_cal, nC), -1.0)
        self.cal_topk_keff = np.zeros((n_cal, nC))
        for j, c in enumerate(classes):
            idx_c = np.where(self.y_cal == c)[0]
            s, th, ke = self._topk_stats_block(S[:, idx_c], self.k)
            self.cal_topk_sum[:, j]  = s
            self.cal_topk_kth[:, j]  = th
            self.cal_topk_keff[:, j] = ke
        self.cal_topk_mean = self.cal_topk_sum / np.maximum(self.cal_topk_keff, 1)
        self.cal_topk_best = self.cal_topk_mean.max(axis=1)
        self.cal_y_hat = classes[np.argmax(self.cal_topk_mean, axis=1)]

    def predict_class(self, x: np.ndarray) -> int:
        """Variant-A ŷ for a test point: argmax_c top-k mean similarity to
        class c (the NCM numerator). Reuses _get_sims; no raw distances."""
        if self.X_cal_wn is None:
            raise ValueError("Must call fit() before predict_class()")
        if getattr(self, "_yh_classes", None) is None:
            self._yh_classes = np.unique(self.y_cal)
        means = self._topk_mean_per_class(self._get_sims(x))
        return int(self._yh_classes[int(np.argmax(means))])

    def predict_class_augmented_cal(self, x: np.ndarray, y: int) -> np.ndarray:
        """Variant-A ŷ for every cal point in the augmented set {cal} u {(x,y)}
        (exchangeable mode). Adding (x, label y) can only raise each cal point's
        top-k similarity to class y; every other class is unchanged. So the new
        argmax is y where the augmented class-y mean beats the old best, else the
        base prediction. O(n_cal) per call; reuses cached per-class top-k stats
        and the same 'enter top-k' rule as updated_calibration_scores_for."""
        self._ensure_cal_yhat()
        sims_x = self._get_sims(x)
        y = int(y)
        if y in self._yh_col:
            col_y = self._yh_col[y]
            sum_y  = self.cal_topk_sum[:, col_y].copy()
            kth_y  = self.cal_topk_kth[:, col_y]
            keff_y = self.cal_topk_keff[:, col_y].copy()
            full  = keff_y >= self.k
            enter = full & (sims_x > kth_y)
            grow  = ~full
            sum_y[enter] += sims_x[enter] - kth_y[enter]
            sum_y[grow]  += sims_x[grow]
            keff_y[grow] += 1
            mean_y_aug = sum_y / np.maximum(keff_y, 1)
        else:
            # y absent from cal: x becomes the sole class-y exemplar (k_eff=1)
            mean_y_aug = sims_x
        return np.where(mean_y_aug >= self.cal_topk_best, y, self.cal_y_hat)


class RBFDensityNCM(NonconformityMeasure):
    """
    RBF (Gaussian) kernel density NCM.

    Score for point i:
        alpha_i = -log Σ_{j ≠ i, y_j = y_i}  exp(-||x_i - x_j||² / (2 σ²))

    Lower alpha = denser within-class neighborhood = more conforming.

    Why this NCM:
    - Nonlinear in x: can exploit curved class manifolds that a linear NCM
      (whitened cosine) discards. Pairs with AE-style nonlinear features.
    - O(n_cal/K) per-hypothesis update via logaddexp on a single
      precomputed pairwise kernel matrix.

    Exchangeability (O(1/n) approximation, same regime as MahalNNRatio /
    WhitenedGeodesicNNRatio):
    - The bandwidth σ is fit on calibration data only at fit() time (median
      pairwise distance heuristic). For exact exchangeability σ should be
      recomputed on the augmented bag {z_1, ..., z_n, (x*, y*)}; the asymmetry
      is O(1/n²) and negligible for n >= 200. See Fan & Sesia (2025,
      arXiv 2512.15383).
    - Alternative: pass a precomputed σ from an independent unlabeled pool
      (set sigma= at init) for an exactly-symmetric bandwidth.

    FCP O(N) compatibility: per-hypothesis only n_cal/K calibration scores
    change (those with y_i == y_hyp). All others reuse the baseline value.
    """

    def __init__(self, sigma: Optional[float] = None,
                 bandwidth_rule: str = "median",
                 sigma_scale: float = 1.0,
                 ratio: bool = False,
                 same_agg: str = "sum",
                 whiten: bool = False,
                 reg: float = 1e-4,
                 eps_log: float = 1e-30,
                 allow_nonexchangeable: bool = False):
        """
        Args:
            sigma: Kernel bandwidth. If None, set at fit() via bandwidth_rule.
            bandwidth_rule: 'median' (median pairwise dist), 'knn' (median dist to
                            k=10 NN), 'within_class_median' (median of same-class
                            pairwise dist).
            sigma_scale: Multiplier on the auto-chosen sigma. 1.0 = use as-is.
            ratio: If True, score = -log(agg_same K) + log(Σ_other K) (log-odds).
                   If False, score = -log(agg_same K) (within-class density only).
            same_agg: 'sum' = log-sum-exp over all same-class neighbors (symmetric).
                      'max' = 1-NN same class (asymmetric, mirrors geodesic_topk_asym).
                      Same-class side uses 'max' helps efficiency by tightening the
                      numerator on well-clustered features. Other side stays soft.
            whiten: If True, apply pooled-within-class whitening before kernel dists.
            reg: Whitening regularization floor.
            eps_log: Floor inside the log when a class has no neighbors at all.
            allow_nonexchangeable: Approve and silence the validity warning that
                fires for cal-fit bandwidth (sigma=None) and/or cal-fit whitening
                (whiten=True). For an exact guarantee, pass a precomputed sigma
                from an independent unlabeled pool AND keep whiten=False.
        """
        self.sigma = sigma
        self._sigma_provided = sigma is not None  # external (exchangeable) bandwidth?
        self.bandwidth_rule = bandwidth_rule
        self.sigma_scale = sigma_scale
        self.ratio = ratio
        self.same_agg = same_agg
        self.whiten = whiten
        self.allow_nonexchangeable = allow_nonexchangeable
        self.reg = reg
        self.eps_log = eps_log
        self.inv_std_ = None
        # Fit state
        self.X_cal = None
        self.y_cal = None
        self.gamma_ = None              # 1 / (2 sigma²)
        self.K_cal_ = None              # (n_cal, n_cal) kernel matrix
        self.baseline_lse_ = None       # (n_cal,) log Σ_{j≠i, y_j=y_i} K_ij
        self.alpha0 = None
        # Test-point cache: keyed by x.ctypes.data
        self._test_cache_key = None
        self._test_cache_k = None        # (n_cal,) K(x_test, x_j)

    @staticmethod
    def _pairwise_dist2(X: np.ndarray) -> np.ndarray:
        sq = (X ** 2).sum(axis=1)
        D2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
        return np.maximum(D2, 0.0)

    @classmethod
    def _auto_sigma(cls, X: np.ndarray, y: np.ndarray, rule: str,
                    rng_seed: int = 42) -> float:
        """Compute an auto bandwidth. All rules are symmetric in calibration
        points (exchangeable up to the standard O(1/n) bandwidth approximation)."""
        rng = np.random.default_rng(rng_seed)
        D2 = cls._pairwise_dist2(X)
        D = np.sqrt(D2)
        np.fill_diagonal(D, np.inf)

        if rule == "median":
            iu = np.triu_indices_from(D, k=1)
            d = D[iu]
            return float(np.median(d[np.isfinite(d)]))
        if rule == "knn":
            k = min(10, len(X) - 1)
            nn_d = np.partition(D, k - 1, axis=1)[:, :k]
            # Median of mean k-NN distance per point
            return float(np.median(nn_d.mean(axis=1)))
        if rule == "within_class_median":
            dists = []
            for c in np.unique(y):
                idx = np.where(y == c)[0]
                if len(idx) < 2:
                    continue
                sub = D[np.ix_(idx, idx)]
                iu = np.triu_indices_from(sub, k=1)
                dists.append(sub[iu][np.isfinite(sub[iu])])
            if not dists:
                return float(np.median(D[np.isfinite(D)]))
            return float(np.median(np.concatenate(dists)))
        raise ValueError(f"Unknown bandwidth_rule: {rule}")

    @staticmethod
    def _row_logsumexp(sub: np.ndarray) -> np.ndarray:
        """log-sum-exp over rows, treating -inf entries correctly."""
        row_max = sub.max(axis=1)
        finite = np.isfinite(row_max)
        out = np.full(sub.shape[0], -np.inf)
        if finite.any():
            shifted = sub[finite] - row_max[finite, None]
            out[finite] = np.log(np.exp(shifted).sum(axis=1)) + row_max[finite]
        return out

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        # Surface any cal-fit (non-exchangeable) component: auto bandwidth from
        # cal (sigma not supplied externally) and/or cal-fit whitening.
        _reasons = []
        if not self._sigma_provided:
            _reasons.append("cal-fit bandwidth (sigma auto from cal)")
        if self.whiten:
            _reasons.append("cal-fit pooled whitening")
        if _reasons:
            warn_nonexchangeable(
                "RBFDensityNCM " + " + ".join(_reasons),
                "Pass a precomputed sigma from an independent unlabeled pool "
                "(and keep whiten=False) for an exact guarantee.",
                order="O(1/n)", allow=self.allow_nonexchangeable)
        X_cal = X_cal.astype(np.float64)
        self.y_cal = np.asarray(y_cal)
        n_cal, _ = X_cal.shape

        # --- Optional pooled whitening (LDA-style) ---
        if self.whiten:
            residuals = X_cal.copy()
            for c in np.unique(self.y_cal):
                idx = np.where(self.y_cal == c)[0]
                residuals[idx] -= X_cal[idx].mean(axis=0)
            pooled_var = (residuals ** 2).mean(axis=0)
            adaptive_reg = max(self.reg, 0.01 * float(np.median(pooled_var)))
            self.inv_std_ = 1.0 / np.sqrt(pooled_var + adaptive_reg)
            self.X_cal = X_cal * self.inv_std_
        else:
            self.inv_std_ = None
            self.X_cal = X_cal

        # --- Bandwidth (computed on whitened space if whitened) ---
        if self.sigma is None:
            auto = self._auto_sigma(self.X_cal, self.y_cal, self.bandwidth_rule)
            self.sigma = max(auto * self.sigma_scale, 1e-6)
        self.gamma_ = 1.0 / (2.0 * self.sigma ** 2)

        # --- log-kernel matrix (cal × cal) ---
        D2 = self._pairwise_dist2(self.X_cal)
        self.logK_cal_ = -self.gamma_ * D2
        np.fill_diagonal(self.logK_cal_, -np.inf)

        # --- Baseline same-class aggregate ---
        # 'sum' => log Σ exp(logK)  (smooth/soft);
        # 'max' => max logK (= -gamma * d²_1nn; asymmetric, 1-NN style).
        self.baseline_lse_same_ = np.full(n_cal, -np.inf)
        for c in np.unique(self.y_cal):
            idx = np.where(self.y_cal == c)[0]
            sub = self.logK_cal_[np.ix_(idx, idx)]
            if self.same_agg == "sum":
                self.baseline_lse_same_[idx] = self._row_logsumexp(sub)
            elif self.same_agg == "max":
                # max along axis=1 ignoring -inf diagonal naturally
                self.baseline_lse_same_[idx] = sub.max(axis=1)
            else:
                raise ValueError(f"Unknown same_agg: {self.same_agg}")

        if self.ratio:
            # --- Baseline cross-class log-sum-exp ---
            self.baseline_lse_other_ = np.full(n_cal, -np.inf)
            for c in np.unique(self.y_cal):
                idx_in = np.where(self.y_cal == c)[0]
                idx_out = np.where(self.y_cal != c)[0]
                if len(idx_out) == 0:
                    continue
                sub = self.logK_cal_[np.ix_(idx_in, idx_out)]
                self.baseline_lse_other_[idx_in] = self._row_logsumexp(sub)
            # nonconformity = -log Σ_same + log Σ_other = log(other/same)
            self.alpha0 = -self.baseline_lse_same_ + self.baseline_lse_other_
        else:
            self.alpha0 = -self.baseline_lse_same_
        self.alpha0[~np.isfinite(self.alpha0)] = 1e9

    def get_calibration_scores(self) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()

    def _get_logk_test(self, x: np.ndarray) -> np.ndarray:
        """Return log K(x_test, x_j) for j in cal. Cached on x.ctypes.data."""
        key = x.ctypes.data
        if key != self._test_cache_key:
            x_proj = x * self.inv_std_ if self.whiten else x
            d2 = ((self.X_cal - x_proj[None, :]) ** 2).sum(axis=1)
            self._test_cache_k = -self.gamma_ * d2
            self._test_cache_key = key
        return self._test_cache_k

    @staticmethod
    def _lse_vec(v: np.ndarray) -> float:
        m = v.max()
        if not np.isfinite(m):
            return -np.inf
        return float(np.log(np.exp(v - m).sum()) + m)

    def score_x(self, x: np.ndarray, y: int) -> float:
        """Nonconformity of (x, y)."""
        logk = self._get_logk_test(x)
        mask_same = (self.y_cal == y)
        if not np.any(mask_same):
            return 1e9
        if self.same_agg == "sum":
            agg_same = self._lse_vec(logk[mask_same])
        elif self.same_agg == "max":
            agg_same = float(logk[mask_same].max())
        else:
            raise ValueError(f"Unknown same_agg: {self.same_agg}")
        if not self.ratio:
            return float(-agg_same)
        mask_other = ~mask_same
        lse_other = self._lse_vec(logk[mask_other]) if np.any(mask_other) else -np.inf
        return float(-agg_same + (lse_other if np.isfinite(lse_other) else np.log(self.eps_log)))

    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        """Augmented cal scores after adding (x, y).

        Density form: only y_i = y points see a numerator update.
        Ratio form: y_i = y see numerator update, y_i != y see denominator update.
        """
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        logk = self._get_logk_test(x)
        idx_same = np.where(self.y_cal == y)[0]

        new_lse_same = self.baseline_lse_same_.copy()
        if len(idx_same) > 0:
            if self.same_agg == "sum":
                new_lse_same[idx_same] = np.logaddexp(
                    self.baseline_lse_same_[idx_same], logk[idx_same])
            elif self.same_agg == "max":
                # 1-NN update: new max is max(old, K(x_i, x_test))
                new_lse_same[idx_same] = np.maximum(
                    self.baseline_lse_same_[idx_same], logk[idx_same])
            else:
                raise ValueError(f"Unknown same_agg: {self.same_agg}")

        if not self.ratio:
            scores = -new_lse_same
        else:
            idx_other = np.where(self.y_cal != y)[0]
            new_lse_other = self.baseline_lse_other_.copy()
            if len(idx_other) > 0:
                new_lse_other[idx_other] = np.logaddexp(
                    self.baseline_lse_other_[idx_other], logk[idx_other])
            scores = -new_lse_same + new_lse_other
        scores[~np.isfinite(scores)] = 1e9
        return scores


class SoftmaxNonconformity(NonconformityMeasure):
    """
    Logistic Regression (Softmax) nonconformity measure.

    Score: s(x, y) = 1 - p(y | x)   where p comes from a softmax classifier.

    During fit(), a logistic regression model is trained on the calibration
    data.  This makes the NCM **data-adaptive** — unlike geometric NCMs that
    only measure distances, it learns a decision boundary.

    Designed for use with **CV+** (CrossValidationPlusPredictor):
    * Each fold trains its own classifier on the training portion.
    * Held-out points are scored with ``score_x_cv``.
    * At test time each fold's classifier scores the test point.

    NOT suitable for Full CP (no efficient hypothetical-addition update).
    """

    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter
        self.classifier = None
        self.scaler = None
        self.X_cal = None
        self.y_cal = None
        self.alpha0 = None
        self._class_to_idx = None

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.X_cal = X_cal
        self.y_cal = y_cal

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_cal)

        self.classifier = LogisticRegression(
            max_iter=self.max_iter, solver='lbfgs', random_state=42
        )
        self.classifier.fit(X_scaled, y_cal)
        self._class_to_idx = {c: i for i, c in enumerate(self.classifier.classes_)}

        # Calibration scores: 1 - p(y_true | x)
        probs = self.classifier.predict_proba(X_scaled)
        self.alpha0 = np.array([
            1.0 - probs[i, self._class_to_idx[y_cal[i]]]
            if y_cal[i] in self._class_to_idx else 1.0
            for i in range(len(y_cal))
        ])

    def get_calibration_scores(self) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()

    def score_x(self, x: np.ndarray, y: int) -> float:
        return self.score_x_cv(x, y)

    def score_x_cv(self, x: np.ndarray, y: int) -> float:
        """Score for CV+: 1 - p(y | x) using the fitted classifier."""
        if self.classifier is None:
            raise ValueError("Must call fit() first")
        X_scaled = self.scaler.transform(x.reshape(1, -1))
        probs = self.classifier.predict_proba(X_scaled)[0]
        if y in self._class_to_idx:
            return 1.0 - float(probs[self._class_to_idx[y]])
        return 1.0  # unseen class → max nonconformity

    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        raise NotImplementedError(
            "SoftmaxNonconformity does not support Full CP updates. "
            "Use with CV+ or Split CP only."
        )


class RidgeSoftmaxNCM(NonconformityMeasure):
    """FCA-inspired softmax NCM (text-free SS-Text) for Full CP.

    A class-mean-ANCHORED ridge (least-squares) linear probe whose softmax gives
    the LAC/THR score  s(x, y) = 1 - p(y | x).  This is the pure-SSL translation
    of "Full Conformal Adaptation of Medical VLMs" (Silva-Rodriguez et al., IPMI
    2025, arXiv:2506.06076): we keep their closed-form, per-candidate-refittable
    probe but, lacking a zero-shot TEXT encoder (we use DINOv2), replace the text
    anchor by the class MEAN and recover discriminativeness via the ridge
    covariance term ``A^{-1} = (Z^T Z + lam I)^{-1}`` that their first-order
    SS-Text linearisation discards.

    Per class c the probe weight solves the anchored ridge
        w_c = A^{-1} ( Z^T y_c  +  lam_a mu_c ),   A = Z^T Z + (lam + lam_a) I,
    with mu_c the class mean (anchor), y_c the {0,1} one-hot column.  Then
        f_c(x) = z^T w_c ,   p(y|x) = softmax_c( f_c(x) / T ) ,   s = 1 - p(y|x).
        lam_a -> inf  =>  w_c -> mu_c (pure nearest-class-mean prototype; the
                          scarce-data-safe limit, mirroring FCA's text anchor),
        lam_a -> 0    =>  discriminative ridge/LDA fit.

    Full CP.  For each test point x and candidate y, (z, y) is folded into the
    bag.  ``A`` gets ONE rank-1 Sherman-Morrison update (independent of the
    candidate LABEL, so it is shared across all K candidates of a test point);
    only the class-y column of ``Z^T Y + lam_a M`` moves.  Calibration points are
    re-scored with leave-one-out predictions via the ridge PRESS identity
        f_c^{LOO}(z_i) = ( f_c(z_i) - h_ii y_ic ) / (1 - h_ii),
        h_ii = z_i^T A^{-1} z_i      (leverage),
    which removes each point's self-fit.  The class-mean anchor is held at its
    full-bag value (an O(1/n_c) self-inclusion kept for the clean closed form);
    every point -- cal AND the hypothesised test point -- is treated by the SAME
    formula, so the n+1 scores stay exchangeable.

    EXCHANGEABILITY.  The probe is refit per candidate on the augmented bag (a
    symmetric function of it), NOT trained once on cal -- so unlike
    ``SoftmaxNonconformity`` (cal-trained logistic, Full-CP-invalid) this is
    exactly exchangeable, PROVIDED the upstream feature map, ``lam``, ``lam_a``
    and the temperature ``T`` are fixed / pool-sourced.  A cal-fit temperature
    (``temperature=None`` -> auto) breaks it at O(1/n) and routes through
    ``warn_nonexchangeable`` (same policy as RBFDensityNCM's cal-fit bandwidth).
    Validity is independent of ``T`` for any FIXED value; ``T`` is an
    efficiency knob (sharper softmax -> tighter sets).

    Input features are assumed ALREADY projected by the exchangeable pool
    transform (exchangeable_features.UnlabeledTransform); this NCM adds no
    further whitening.

    Args:
        lam:        ridge stabiliser (added to the diagonal alongside lam_anchor).
        lam_anchor: class-mean anchor strength (large -> prototype, 0 -> LDA fit).
        temperature: softmax temperature T. A fixed float is exact; None means
                     auto from cal (O(1/n), warns).
        loo:        if True (default) use ridge PRESS leave-one-out predictions;
                    if False use full-bag (self-inclusive) predictions (also exact,
                    simpler, slightly more optimistic).
        eps:        numerical floor (leverage clamp / norm floor).
        allow_nonexchangeable: approve + silence the cal-fit-temperature warning.
    """

    def __init__(self, lam: float = 1.0, lam_anchor: float = 1.0,
                 temperature: Optional[float] = None, loo: bool = True,
                 eps: float = 1e-9, allow_nonexchangeable: bool = False):
        self.lam = float(lam)
        self.lam_anchor = float(lam_anchor)
        self.temperature = temperature
        self.loo = bool(loo)
        self.eps = float(eps)
        self.allow_nonexchangeable = allow_nonexchangeable
        # fitted state
        self.classes_ = None
        self._cls_to_col = None
        self.Z = None
        self.y_col = None        # (n,) cal labels mapped to 0..K-1
        self.Y = None            # (n, K) one-hot
        self.M = None            # (d, K) class-mean anchors
        self.n_c = None          # (K,) class counts
        self.A_inv = None        # (d, d) (Z^T Z + (lam+lam_a) I)^{-1}
        self.B = None            # (d, K) Z^T Y + lam_a M
        self.W = None            # (d, K) A_inv @ B
        self.alpha0 = None
        self._T = None
        # per-test cache, keyed on x.ctypes.data
        self._cache_key = None
        self._cache = None

    # -- softmax helpers (float64, stable) ----------------------------------
    @staticmethod
    def _softmax_rows(logits: np.ndarray) -> np.ndarray:
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _softmax_vec(logits: np.ndarray) -> np.ndarray:
        m = logits.max()
        e = np.exp(logits - m)
        return e / e.sum()

    def _resolve_T(self, Z: np.ndarray) -> float:
        if self.temperature is not None:
            return float(self.temperature)
        # auto temperature from cal -> data-dependent, O(1/n) non-exchangeable
        warn_nonexchangeable(
            "RidgeSoftmaxNCM cal-fit temperature (temperature=None)",
            "Pass a fixed temperature for an exact guarantee (sweep it for "
            "efficiency); validity holds for ANY fixed T.",
            order="O(1/n)", allow=self.allow_nonexchangeable)
        F = Z @ self.W
        n, K = F.shape
        if K < 2:
            return 1.0
        Ftrue = F[np.arange(n), self.y_col]
        Fm = F.copy()
        Fm[np.arange(n), self.y_col] = -np.inf
        Fother = Fm.max(axis=1)
        gap = float(np.median(Ftrue - Fother))
        # map the median true-vs-best-other gap to a softmax logit ~4 (p~0.98)
        return max(abs(gap), self.eps) / 4.0

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        Z = np.asarray(X_cal, dtype=np.float64)
        self.y_cal = np.asarray(y_cal)
        self.classes_ = np.unique(self.y_cal)
        self._cls_to_col = {int(c): j for j, c in enumerate(self.classes_)}
        n, d = Z.shape
        K = len(self.classes_)
        self.Z = Z
        self.y_col = np.array([self._cls_to_col[int(c)] for c in self.y_cal])
        Y = np.zeros((n, K))
        Y[np.arange(n), self.y_col] = 1.0
        self.Y = Y
        # class-mean anchors + counts
        M = np.zeros((d, K))
        n_c = np.zeros(K)
        for j in range(K):
            m = self.y_col == j
            n_c[j] = int(m.sum())
            if n_c[j] > 0:
                M[:, j] = Z[m].mean(axis=0)
        self.M = M
        self.n_c = n_c
        # anchored ridge
        lam_eff = self.lam + self.lam_anchor
        A = Z.T @ Z
        A[np.diag_indices_from(A)] += lam_eff
        self.A_inv = np.linalg.inv(A)
        self.B = Z.T @ Y + self.lam_anchor * M
        self.W = self.A_inv @ self.B
        self._T = self._resolve_T(Z)
        # baseline (cal-only bag) calibration scores, LOO if requested
        F = Z @ self.W
        if self.loo:
            ZA = Z @ self.A_inv
            h = np.einsum('ij,ij->i', ZA, Z)
            h = np.clip(h, None, 1.0 - self.eps)
            F = (F - h[:, None] * Y) / (1.0 - h)[:, None]
        P = self._softmax_rows(F / self._T)
        self.alpha0 = 1.0 - P[np.arange(n), self.y_col]
        self._cache_key = None
        self._cache = None
        return self

    def get_calibration_scores(self) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()

    # -- per-test shared work (candidate-label independent) -----------------
    def _ensure_cache(self, x: np.ndarray) -> Dict:
        key = x.ctypes.data
        if self._cache_key == key and self._cache is not None:
            return self._cache
        z = np.asarray(x, dtype=np.float64).ravel()
        u = self.A_inv @ z
        denom = 1.0 + float(z @ u)                 # Sherman-Morrison denom (>0)
        A1_inv = self.A_inv - np.outer(u, u) / denom
        h_test = float(z @ (A1_inv @ z))           # test-point leverage in aug. A1
        W1_base = A1_inv @ self.B                  # candidate-independent columns
        f_test_base = z @ W1_base                  # (K,)
        cache = {"z": z, "A1_inv": A1_inv, "h_test": min(h_test, 1.0 - self.eps),
                 "W1_base": W1_base, "f_test_base": f_test_base, "F1_base": None}
        if self.loo:
            ZA1 = self.Z @ A1_inv
            h = np.einsum('ij,ij->i', ZA1, self.Z)
            cache["h"] = np.clip(h, None, 1.0 - self.eps)
        self._cache_key = key
        self._cache = cache
        return cache

    def _w1_y(self, cache: Dict, y: int):
        """Augmented class-y weight column after folding (z, y) into the bag.
        Returns (col_or_None, w1_y); col is None when y is absent from cal."""
        z = cache["z"]
        yc = int(y)
        if yc in self._cls_to_col:
            col = self._cls_to_col[yc]
            n_y = self.n_c[col]
            mu_y = self.M[:, col]
            mu_y_new = (n_y * mu_y + z) / (n_y + 1.0)
            b1y = self.B[:, col] + z + self.lam_anchor * (mu_y_new - mu_y)
            return col, cache["A1_inv"] @ b1y
        # new class: z is its sole member, mu_y = z -> B1[:,y] = z + lam_a z
        return None, cache["A1_inv"] @ ((1.0 + self.lam_anchor) * z)

    def score_x(self, x: np.ndarray, y: int) -> float:
        cache = self._ensure_cache(x)
        col, w1y = self._w1_y(cache, y)
        f = cache["f_test_base"]
        f_y = float(cache["z"] @ w1y)
        if col is not None:
            logits = f.copy(); logits[col] = f_y; yi = col
        else:
            logits = np.append(f, f_y); yi = len(logits) - 1
        if self.loo:
            h = cache["h_test"]
            logits = logits / (1.0 - h)
            logits[yi] = (f_y - h) / (1.0 - h)     # PRESS at the hypothesised class
        p = self._softmax_vec(logits / self._T)
        return float(1.0 - p[yi])

    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        cache = self._ensure_cache(x)
        col, w1y = self._w1_y(cache, y)
        Z = self.Z
        n = len(Z)
        if cache["F1_base"] is None:
            cache["F1_base"] = Z @ cache["W1_base"]    # (n, K), once per test pt
        Fy = Z @ w1y                                   # updated class-y column
        if col is not None:
            F = cache["F1_base"].copy(); F[:, col] = Fy
        else:
            F = np.concatenate([cache["F1_base"], Fy[:, None]], axis=1)
        ycol = self.y_col                              # cal true-class columns
        if self.loo:
            h = cache["h"]
            # PRESS: subtract the leverage h_i only at each point's OWN true-class
            # column (target 1), divide every column by (1 - h_i).
            true_raw = F[np.arange(n), ycol].copy()    # raw true-class prediction
            F = F / (1.0 - h)[:, None]
            F[np.arange(n), ycol] = (true_raw - h) / (1.0 - h)
        P = self._softmax_rows(F / self._T)
        return 1.0 - P[np.arange(n), ycol]


class PrototypeSoftmaxNCM(NonconformityMeasure):
    """Vanilla text-free FCA NCM: class-mean prototype -> softmax LAC, for Full CP.

    The FAITHFUL text-free translation of "Full Conformal Adaptation of Medical
    VLMs" (Silva-Rodriguez et al., IPMI 2025, arXiv:2506.06076).  FCA's SS-Text
    probe is a class-mean prototype BLENDED WITH A ZERO-SHOT TEXT ANCHOR,
        w_c = a * mu_c (visual class mean)  +  b * t_c (text prototype),
    scored by LAC  s(x, y) = 1 - p(y | x).  We have no text encoder (DINOv2 is
    pure SSL), so we DROP the text anchor and keep the bare prototype: w_c = mu_c.
    Unlike RidgeSoftmaxNCM (rung 4) this adds NO ridge covariance term -- it is
    the minimal discriminative head, and the clean ablation control that isolates
    what the covariance term actually buys.

    ALGORITHM FLOW (plain text).  z = a feature AFTER the exchangeable pool
    transform (exchangeable_features.UnlabeledTransform; this NCM adds no further
    whitening); in ``logit="cosine"`` mode z and the prototypes are L2-normalised
    (cosine logits on the DINOv2 hypersphere -- the FCA-faithful default);
    ``logit="dot"`` uses raw inner products with mean prototypes.  T is a FIXED
    softmax temperature (FCA's tau analog), sourced off-cal; validity is
    T-independent, T is an efficiency knob.

    ONE symmetric score rule, applied to EVERY point i in a bag B (this is what
    makes it exactly exchangeable)::

        mu_c^{(-i)}  = class-c prototype over B \\ {i}   (leave i out of its OWN
                       class only; other classes are unchanged by removing i)
        f_c          = <z_i, mu_c^{(-i)}>
        p(y_i|z_i)   = softmax_c( f_c / T )
        s_i          = 1 - p(y_i | z_i)
        empty-class convention: if class y_i is EMPTY after leaving i out
                       (singleton cal class, or a candidate class absent from cal
                       applied to the test point) set f_{y_i} = -inf -> p = 0 ->
                       s_i = 1.  The SAME rule fires for the test point and for
                       singleton cal points -> symmetric -> exchangeable.

    Full CP.  Folding (x, y) into the bag changes ONLY the class-y prototype, so
    only column y of the cached cal logit matrix moves.  The test point's own
    class (y) prototype is always the pure cal class-y mean (leave the test point
    itself out), so ONE matvec z @ Mu gives the test logits for all K candidates.
    No matrix inverse, no Sherman-Morrison, no PRESS leverage: the prototype is
    LINEAR in the data, so leave-one-out is the closed-form class-mean update
        mu_c^{(-i)} = (n_c mu_c - z_i) / (n_c - 1).

    EXCHANGEABILITY.  Exactly exchangeable PROVIDED the upstream feature map and
    the temperature T are fixed / pool-sourced.  A cal-fit temperature
    (``temperature=None`` -> auto) breaks it at O(1/n) and routes through
    ``warn_nonexchangeable`` (same policy as RidgeSoftmaxNCM / RBFDensityNCM).
    Because LOO is exact and the empty-class rule is symmetric, small (balanced)
    cal yields BLOATED sets, never under-coverage.

    Args:
        temperature: softmax temperature T. A fixed float is exact; None auto-fits
                     from cal (O(1/n), warns).
        logit:       "cosine" (L2-normalise features + prototypes; FCA-faithful
                     default) or "dot" (raw inner product, mean prototypes).
        eps:         numerical floor (norm clamp).
        allow_nonexchangeable: approve + silence the cal-fit-temperature warning.
    """

    def __init__(self, temperature: Optional[float] = None, logit: str = "cosine",
                 eps: float = 1e-9, allow_nonexchangeable: bool = False):
        if logit not in ("cosine", "dot"):
            raise ValueError(f"logit must be 'cosine' or 'dot', got {logit!r}")
        self.temperature = temperature
        self.logit = logit
        self.cosine = (logit == "cosine")
        self.eps = float(eps)
        self.allow_nonexchangeable = allow_nonexchangeable
        # fitted state
        self.classes_ = None
        self._cls_to_col = None
        self.Z = None             # (n, d) prepped (normalised iff cosine) features
        self.y_cal = None
        self.y_col = None         # (n,) cal labels mapped to 0..K-1
        self.class_sum = None     # (d, K) per-class sums of prepped features
        self.n_c = None           # (K,) class counts
        self.P = None             # (d, K) full-mean prototype directions
        self._P_ok = None         # (K,) bool: prototype well-defined
        self.F_base = None        # (n, K) cal-only LOO logits (own-class = LOO)
        self.alpha0 = None        # (n,) cal-only LOO scores
        self._T = None
        # per-test cache, keyed on x.ctypes.data
        self._cache_key = None
        self._cache = None

    # -- softmax helpers (float64, stable; tolerate -inf logits) -------------
    @staticmethod
    def _softmax_rows(logits: np.ndarray) -> np.ndarray:
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _softmax_vec(logits: np.ndarray) -> np.ndarray:
        m = logits.max()
        e = np.exp(logits - m)
        return e / e.sum()

    # -- feature / prototype prep -------------------------------------------
    def _prep(self, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(X, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[None, :]
        if self.cosine:
            nrm = np.linalg.norm(Z, axis=1, keepdims=True)
            Z = Z / np.maximum(nrm, self.eps)
        return Z

    def _proto(self, s: np.ndarray, count: float):
        """Prototype from a class sum vector ``s`` over ``count`` members.
        Returns the (d,) prototype, or None if the class is empty/degenerate."""
        if count <= 0:
            return None
        if self.cosine:
            nrm = float(np.linalg.norm(s))
            if nrm < self.eps:
                return None
            return s / nrm
        return s / count

    def _resolve_T(self) -> float:
        if self.temperature is not None:
            return float(self.temperature)
        warn_nonexchangeable(
            "PrototypeSoftmaxNCM cal-fit temperature (temperature=None)",
            "Pass a fixed temperature for an exact guarantee (sweep it for "
            "efficiency); validity holds for ANY fixed T.",
            order="O(1/n)", allow=self.allow_nonexchangeable)
        F = self.F_base
        n, K = F.shape
        if K < 2:
            return 1.0
        Ftrue = F[np.arange(n), self.y_col]
        Fm = F.copy()
        Fm[np.arange(n), self.y_col] = -np.inf
        Fother = Fm.max(axis=1)
        diff = Ftrue - Fother
        diff = diff[np.isfinite(diff)]
        gap = float(np.median(diff)) if diff.size else 1.0
        # map the median true-vs-best-other gap to a softmax logit ~4 (p~0.98)
        return max(abs(gap), self.eps) / 4.0

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        Z = self._prep(X_cal)
        self.Z = Z
        self.y_cal = np.asarray(y_cal)
        self.classes_ = np.unique(self.y_cal)
        self._cls_to_col = {int(c): j for j, c in enumerate(self.classes_)}
        n, d = Z.shape
        K = len(self.classes_)
        self.y_col = np.array([self._cls_to_col[int(c)] for c in self.y_cal])
        # per-class sums + counts
        self.class_sum = np.zeros((d, K))
        self.n_c = np.zeros(K)
        for j in range(K):
            m = self.y_col == j
            self.n_c[j] = int(m.sum())
            self.class_sum[:, j] = Z[m].sum(axis=0)
        # full-mean prototype directions
        self.P = np.zeros((d, K))
        self._P_ok = np.zeros(K, dtype=bool)
        for j in range(K):
            p = self._proto(self.class_sum[:, j], self.n_c[j])
            if p is not None:
                self.P[:, j] = p
                self._P_ok[j] = True
        # cross logits with full class means; undefined cols -> -inf
        F = Z @ self.P
        if not self._P_ok.all():
            F[:, ~self._P_ok] = -np.inf
        # own-class logit replaced by the exact leave-one-out prototype
        f_own = np.empty(n)
        for i in range(n):
            j = self.y_col[i]
            p = self._proto(self.class_sum[:, j] - Z[i], self.n_c[j] - 1.0)
            f_own[i] = -np.inf if p is None else float(Z[i] @ p)
        F[np.arange(n), self.y_col] = f_own
        self.F_base = F
        self._T = self._resolve_T()
        P0 = self._softmax_rows(self.F_base / self._T)
        self.alpha0 = 1.0 - P0[np.arange(n), self.y_col]
        self._cache_key = None
        self._cache = None
        self.cal_y_hat = None        # reset MS-CS y_hat cache on (re)fit
        return self

    def get_calibration_scores(self) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()

    # -- per-test shared work (candidate-label independent) -----------------
    def _ensure_cache(self, x: np.ndarray) -> Dict:
        key = x.ctypes.data
        if self._cache_key == key and self._cache is not None:
            return self._cache
        z = self._prep(x).ravel()
        f_test = self.P.T @ z                       # (K,) = z @ P
        if not self._P_ok.all():
            f_test = f_test.copy()
            f_test[~self._P_ok] = -np.inf
        cache = {"z": z, "f_test": f_test}
        self._cache_key = key
        self._cache = cache
        return cache

    def score_x(self, x: np.ndarray, y: int) -> float:
        cache = self._ensure_cache(x)
        yc = int(y)
        if yc not in self._cls_to_col:
            # candidate class absent from cal: the test point is its sole member,
            # so leaving it out empties class y -> p(y)=0 -> score 1.0. This is
            # the SAME empty-class rule a singleton cal point gets (exchangeable).
            return 1.0
        col = self._cls_to_col[yc]
        p = self._softmax_vec(cache["f_test"] / self._T)
        return float(1.0 - p[col])

    def _augmented_col_logit(self, x: np.ndarray, y: int) -> np.ndarray:
        """Augmented class-y logit for EVERY cal point j in the bag {cal u (x, y)}:
          non-members          -> <z_j, proto(class_sum[y] + z, n_y + 1)>
          members (y_j == y)   -> leave j out: <z_j, proto(class_sum[y] + z - z_j, n_y)>
          y absent from cal    -> x is the sole class-y exemplar: <z_j, proto(z, 1)>
        This is exactly column y of the augmented logit matrix the cal SCORES use,
        so the MS-CS y_hat (argmax over this column together with F_base) is
        consistent with the score (and leave-one-out the same way)."""
        cache = self._ensure_cache(x)
        z = cache["z"]
        n = len(self.Z)
        yc = int(y)
        if yc not in self._cls_to_col:
            p_new = self._proto(z, 1.0)
            return (self.Z @ p_new) if p_new is not None else np.full(n, -np.inf)
        col = self._cls_to_col[yc]
        cs = self.class_sum[:, col]
        ny = self.n_c[col]
        # non-members: full augmented class-y mean (one matvec)
        p_full = self._proto(cs + z, ny + 1.0)
        colvec = (self.Z @ p_full) if p_full is not None else np.full(n, -np.inf)
        colvec = np.asarray(colvec, dtype=np.float64)
        # members (y_i == col): leave i out of the augmented class y, count ny
        members = np.where(self.y_col == col)[0]
        if members.size:
            S = (cs + z)[None, :] - self.Z[members]      # (m, d) leave-out sums
            if self.cosine:
                nrm = np.linalg.norm(S, axis=1)
                protos = S / np.maximum(nrm, self.eps)[:, None]
                vals = np.einsum('md,md->m', self.Z[members], protos)
                vals[nrm < self.eps] = -np.inf
            else:
                protos = S / ny
                vals = np.einsum('md,md->m', self.Z[members], protos)
            colvec[members] = vals
        return colvec

    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        n = len(self.Z)
        yc = int(y)
        colvec = np.asarray(self._augmented_col_logit(x, yc), dtype=np.float64)
        F = self.F_base.copy()
        if yc in self._cls_to_col:
            F[:, self._cls_to_col[yc]] = colvec
        else:
            # absent candidate class: append a new column (sole member = test point)
            F = np.concatenate([F, colvec[:, None]], axis=1)
        P = self._softmax_rows(F / self._T)
        return 1.0 - P[np.arange(n), self.y_col]

    # -- suspected-class y_hat for the MS-CS penalty ------------------------------
    # Enables yhat_mode="ncm" with this NCM (run_fcp_with_mscs) and the prototype
    # GPU MS-CS path (mscs_gpu.run_prototype_mscs_torch). y_hat(z) = argmax of the
    # NCM's OWN softmax logits (F_base; own-class leave-one-out), so y_hat is
    # consistent with the cal scores and leave-one-out the same way they are. Every
    # bag member then excludes only itself -> the penalty is a symmetric
    # (exchangeable) function of the bag. (The earlier full-prototype argmax,
    # F = Z @ P, was NON-LOO: a cal point "saw itself" in its own class while the
    # test point did not, which broke the prototype MS-CS penalty's exchangeability.)
    def _ensure_cal_yhat(self):
        if getattr(self, "cal_y_hat", None) is not None:
            return
        F = self.F_base                                  # (n, K) own-class-LOO logits
        n = len(F)
        rows = np.arange(n)
        top1 = F.argmax(axis=1)
        self._yh_top1_col = top1
        self._yh_top1_val = F[rows, top1]
        F2 = F.copy(); F2[rows, top1] = -np.inf          # exact 2nd-best per row
        top2 = F2.argmax(axis=1)
        self._yh_top2_col = top2
        self._yh_top2_val = F2[rows, top2]
        self.cal_y_hat = self.classes_[top1]             # (n,) non-augmented y_hat labels
        self._cal_best = self._yh_top1_val               # legacy alias (LOO top-1 logit)

    def predict_class(self, x: np.ndarray) -> int:
        z = self._prep(x).ravel()
        f = self.P.T @ z
        if not self._P_ok.all():
            f = f.copy(); f[~self._P_ok] = -np.inf
        return int(self.classes_[int(np.argmax(f))])

    def predict_class_augmented_cal(self, x: np.ndarray, y: int) -> np.ndarray:
        """Each cal point's y_hat in the bag {cal u (x, y)} = argmax of its AUGMENTED
        softmax logits = F_base with column y replaced by the augmented class-y
        logit (`_augmented_col_logit`; same column the cal scores use). Only column
        y changes, so the exact new argmax is y when its augmented logit beats the
        best of the OTHER columns, else that best -- read off the precomputed top-2
        of F_base. Leave-one-out + score-consistent -> exchangeable."""
        self._ensure_cal_yhat()
        yc = int(y)
        colvec = np.asarray(self._augmented_col_logit(x, yc), dtype=np.float64)
        if yc not in self._cls_to_col:
            # absent candidate class -> a brand-new column vs F_base's existing best
            return np.where(colvec >= self._yh_top1_val, yc,
                            self.classes_[self._yh_top1_col])
        col = self._cls_to_col[yc]
        # best of the OTHER columns: top-1 unless the changed column WAS the top-1
        is_top1 = (self._yh_top1_col == col)
        thresh = np.where(is_top1, self._yh_top2_val, self._yh_top1_val)
        fb_col = np.where(is_top1, self._yh_top2_col, self._yh_top1_col)
        out_col = np.where(colvec >= thresh, col, fb_col)
        return self.classes_[out_col]


class FullConformalPredictor:

    def __init__(
        self,
        nonconformity_measure: NonconformityMeasure,
        alpha: float = 0.1,
    ):
        """
        Args:
            nonconformity_measure: Nonconformity scoring function
            alpha: Significance level (e.g., 0.1 for 90% confidence)
        """
        self.ncm = nonconformity_measure
        self.alpha = alpha
        self.X_cal = None
        self.y_cal = None
        self.classes = None
        self.cal_scores = None  # Cached calibration scores

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                  all_classes: np.ndarray = None):
        """
        Calibrate the conformal predictor on calibration set.

        Args:
            X_cal: Calibration features (n_cal, d)
            y_cal: Calibration labels (n_cal,)
            all_classes: Full label space to iterate over during prediction.
                         If None, uses only classes present in y_cal.
        """
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(all_classes) if all_classes is not None else np.unique(y_cal)

        # Fit nonconformity measure
        t0 = time.time()
        self.ncm.fit(X_cal, y_cal)
        fit_time = time.time() - t0

        # Compute and cache calibration scores (new API; fallback to legacy .score)
        t0 = time.time()
        try:
            self.cal_scores = self.ncm.get_calibration_scores()
        except (AttributeError, NotImplementedError):
            self.cal_scores = self.ncm.score(X_cal, y_cal)
        score_time = time.time() - t0

        cal_classes = np.unique(y_cal)
        s = self.cal_scores
        print(f"Calibrated with {len(X_cal)} examples, {len(cal_classes)} cal classes, "
              f"{len(self.classes)} candidate classes")
        print(f"Calibration scores (base) - min: {s.min():.4f}, max: {s.max():.4f}, "
              f"mean: {s.mean():.4f}, std: {s.std():.4f}")
        print(f"Timing: fit={fit_time:.2f}s, score_cal={score_time:.2f}s")

    def predict(
        self,
        X_test: np.ndarray,
        return_p_values: bool = False,
        update_calibration_scores: bool = True,
        verbose: bool = True,
        device: str = "cpu",
        gpu_batch_size: int = 256,
    ) -> Dict:
        """
        Compute prediction sets for test examples using Full CP.

        Args:
            X_test: Test features (n_test, d)
            return_p_values: If True, return p-values for all classes
            verbose: Show progress bar
            device: "cpu" (default) runs the per-test Python loop. "cuda" dispatches
                    to a vectorised torch path that supports GeodesicTopKMeanNCM
                    (covers geodesic_topk_mean / _asym and their unwhitened/numerator-
                    only ablations). Raises on other NCMs.
            gpu_batch_size: Number of test points per GPU batch (only when device="cuda").
            update_calibration_scores: If True (default), recompute the calibration
                    scores for each augmented bag {cal ∪ (x_test, y)} — standard
                    transductive Full CP. If False, reuse the static leave-one-out
                    calibration scores from calibrate() (i.e. do NOT update the cal
                    scores at predict): this is the ONLY substantive difference from
                    FCP and was formerly the separate "SCP-geodesic" class. The NCM,
                    fit, and base score_x are identical either way. CPU only.
                    See docs/theory.md §5.

        Returns:
            Dictionary with:
            - 'prediction_sets': List of prediction sets (lists of labels)
            - 'set_sizes': Size of each prediction set
            - 'p_values': (optional) p-values for each (test_idx, class) pair
            - 'prediction_time': Total prediction time in seconds
        """
        if self.cal_scores is None:
            raise ValueError("Must call calibrate() before predict()")

        if device == "cuda":
            if not update_calibration_scores:
                raise NotImplementedError(
                    "update_calibration_scores=False is CPU-only; use device='cpu' "
                    "for the static-calibration (no cal-update) mode."
                )
            if isinstance(self.ncm, PrototypeSoftmaxNCM):
                return self._predict_prototype_softmax_gpu(
                    X_test, return_p_values=return_p_values,
                    verbose=verbose, batch_size=gpu_batch_size,
                )
            if isinstance(self.ncm, RidgeSoftmaxNCM):
                return self._predict_ridge_softmax_gpu(
                    X_test, return_p_values=return_p_values,
                    verbose=verbose, batch_size=gpu_batch_size,
                )
            return self._predict_geodesic_gpu(
                X_test, return_p_values=return_p_values,
                verbose=verbose, batch_size=gpu_batch_size,
            )

        start_time = time.time()
        n_test = len(X_test)
        n_cal = len(self.X_cal)

        prediction_sets = []
        set_sizes = []
        all_p_values = [] if return_p_values else None
        empty_count = 0

        iterator = tqdm(range(n_test), desc="Full CP") if verbose else range(n_test)

        for i in iterator:
            x_test = X_test[i]
            pred_set = []
            p_vals = {} if return_p_values else None

            # For each candidate label, compute p-value
            for y_candidate in self.classes:
                yc = int(y_candidate)

                test_score = self.ncm.score_x(x_test, yc)
                # Standard Full CP recomputes the calibration scores for the
                # augmented bag {cal ∪ (x_test, yc)} (the transductive step).
                # update_calibration_scores=False reuses the static LOO cal scores
                # instead — the sole difference from FCP (was "SCP-geodesic").
                if update_calibration_scores:
                    cal_scores_yc = self.ncm.updated_calibration_scores_for(x_test, yc)
                else:
                    cal_scores_yc = self.cal_scores

                n_greater = np.sum(cal_scores_yc >= test_score)
                p_value = (n_greater + 1) / (n_cal + 1)

                if return_p_values:
                    p_vals[yc] = p_value

                if p_value > self.alpha:
                    pred_set.append(yc)

            # Empty prediction sets are valid under CP: they represent the ~alpha fraction
            # of test points that the predictor correctly does NOT cover.  The CP guarantee
            # is probabilistic (coverage >= 1-alpha over randomness), NOT that every
            # individual set must be non-empty.  The old fallback (add all classes) was
            # artificially inflating empirical coverage above the theoretical upper bound
            # 1-alpha + 1/(n_cal+1) on well-separated datasets like CIFAR-10 with DINOv2.
            if len(pred_set) == 0:
                empty_count += 1

            prediction_sets.append(pred_set)
            set_sizes.append(len(pred_set))

            if return_p_values:
                all_p_values.append(p_vals)

        if empty_count > 0 and verbose:
            print(f"\nWarning: {empty_count}/{n_test} prediction sets are empty!")
            print(f"  Consider increasing alpha (current: {self.alpha}) or checking your data.")

        prediction_time = time.time() - start_time

        results = {
            'prediction_sets': prediction_sets,
            'set_sizes': np.array(set_sizes),
            'prediction_time': prediction_time,
        }

        if return_p_values:
            results['p_values'] = all_p_values

        if verbose:
            print(f"\nPrediction time: {prediction_time:.2f}s "
                  f"({prediction_time / n_test * 1000:.1f}ms per sample)")

        return results

    def _predict_geodesic_gpu(
        self,
        X_test: np.ndarray,
        return_p_values: bool = False,
        verbose: bool = True,
        batch_size: int = 256,
    ) -> Dict:
        """Vectorised GPU path for FCP with GeodesicTopKMeanNCM.

        Replaces the n_test × K Python loop in predict() with batched torch ops.
        Math is identical to the CPU path; see GeodesicTopKMeanNCM.score_x /
        updated_calibration_scores_for. Test points are processed in chunks of
        `batch_size` to bound the (B, K, n_cal) tensor memory footprint.
        """
        if not isinstance(self.ncm, GeodesicTopKMeanNCM):
            raise ValueError(
                "predict(device='cuda') only supports GeodesicTopKMeanNCM; "
                f"got {type(self.ncm).__name__}. Use device='cpu' for other NCMs."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")

        start_time = time.time()
        ncm = self.ncm
        dev = torch.device("cuda")
        eps = 1e-8
        neg_inf = float("-inf")

        n_test = len(X_test)
        classes_np = np.asarray(self.classes)
        K = len(classes_np)
        k = int(ncm.k)

        # --- Move NCM state to GPU once ---
        inv_std = torch.from_numpy(ncm.inv_std).to(dev, dtype=torch.float32)
        X_cal_wn = torch.from_numpy(ncm.X_cal_wn).to(dev, dtype=torch.float32)  # (n_cal, d)
        n_cal = X_cal_wn.shape[0]
        y_cal_t = torch.from_numpy(np.asarray(ncm.y_cal)).to(dev, dtype=torch.long)
        classes_t = torch.from_numpy(classes_np).to(dev, dtype=torch.long)
        same_mask = (y_cal_t.unsqueeze(0) == classes_t.unsqueeze(1))  # (K, n_cal) bool
        n_same_per_class = same_mask.sum(dim=1)  # (K,)

        if ncm.topk_same:
            sum_same = torch.from_numpy(ncm.sum_same_sims).to(dev, dtype=torch.float32)
            kth_same = torch.from_numpy(ncm.kth_same_sim).to(dev, dtype=torch.float32)
            k_same   = torch.from_numpy(ncm._k_same_eff).to(dev, dtype=torch.float32)
        else:
            max_same = torch.from_numpy(ncm.lookup_same_sim).to(dev, dtype=torch.float32)

        need_other = not ncm.numerator_only
        if need_other:
            if ncm.topk_other:
                sum_other = torch.from_numpy(ncm.sum_other_sims).to(dev, dtype=torch.float32)
                kth_other = torch.from_numpy(ncm.kth_other_sim).to(dev, dtype=torch.float32)
                k_other   = torch.from_numpy(ncm._k_other_eff).to(dev, dtype=torch.float32)
            else:
                max_other = torch.from_numpy(ncm.lookup_other_sim).to(dev, dtype=torch.float32)
            n_other_per_class = (n_cal - n_same_per_class)  # (K,)

        # Whiten + L2-normalise full X_test
        X_test_t = torch.from_numpy(np.asarray(X_test)).to(dev, dtype=torch.float32)
        X_test_w = X_test_t * inv_std
        X_test_wn = X_test_w / X_test_w.norm(dim=1, keepdim=True).clamp_min(1e-10)

        p_values_chunks = []

        def _topk_mean_along(values: torch.Tensor, k_max: int,
                              count_per_row: torch.Tensor) -> torch.Tensor:
            """Top-k mean along the last dim.

            values: (..., n) with masked-out entries set to -inf.
            count_per_row: (...) effective k per row (clamped at k_max).
            Returns (...) mean of the top-k finite entries.
            """
            top_vals, _ = values.topk(k=k_max, dim=-1)
            top_finite = torch.where(torch.isfinite(top_vals), top_vals,
                                      torch.zeros_like(top_vals))
            sums = top_finite.sum(dim=-1)
            return sums / count_per_row.clamp_min(1).float()

        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            X_b = X_test_wn[batch_start:batch_end]                       # (B, d)
            B = X_b.shape[0]

            # Similarities test->cal
            S = X_b @ X_cal_wn.T                                         # (B, n_cal)
            S_b = S.unsqueeze(1)                                          # (B, 1, n_cal)
            mask_c = same_mask.unsqueeze(0)                               # (1, K, n_cal)

            # ---- TEST scores (B, K) ----
            if ncm.topk_same:
                S_same_t = torch.where(mask_c, S_b, torch.full_like(S_b, neg_inf))  # (B, K, n_cal)
                ke_same = torch.clamp(n_same_per_class, max=k)            # (K,)
                mean_s = _topk_mean_along(S_same_t, k, ke_same.unsqueeze(0).expand(B, K))
                d_same_test = torch.arccos(mean_s.clamp(-1.0, 1.0))
            else:
                S_same_t = torch.where(mask_c, S_b, torch.full_like(S_b, neg_inf))
                max_s, _ = S_same_t.max(dim=-1)
                d_same_test = torch.arccos(max_s.clamp(-1.0, 1.0))

            if ncm.numerator_only:
                test_scores = d_same_test
            else:
                other_mask_c = ~mask_c
                if ncm.topk_other:
                    S_other_t = torch.where(other_mask_c, S_b, torch.full_like(S_b, neg_inf))
                    ke_other = torch.clamp(n_other_per_class, max=k)
                    mean_o = _topk_mean_along(S_other_t, k, ke_other.unsqueeze(0).expand(B, K))
                    d_other_test = torch.arccos(mean_o.clamp(-1.0, 1.0))
                else:
                    S_other_t = torch.where(other_mask_c, S_b, torch.full_like(S_b, neg_inf))
                    max_o, _ = S_other_t.max(dim=-1)
                    d_other_test = torch.arccos(max_o.clamp(-1.0, 1.0))
                test_scores = d_same_test / (d_other_test + eps)

            # Missing-class symmetry fix (mirrors the fixed GeodesicTopKMeanNCM.score_x):
            # a candidate class ABSENT from cal has an all -inf same-class row, so
            # _topk_mean_along already yields mean 0 -> arccos(0)=pi/2 -- the same
            # zero-neighbour convention fit() uses for a sole-member class. We must NOT
            # force 1e9 here: the old override made a test point whose class is missing
            # from cal non-exchangeable with cal points and under-covered at small cal.
            # Keep only the genuine non-finite guard (div-by-zero etc.).
            test_scores = torch.where(torch.isfinite(test_scores), test_scores,
                                       torch.full_like(test_scores, 1e9))

            # ---- UPDATED cal scores (B, K, n_cal) ----
            if ncm.topk_same:
                sum_b = sum_same.view(1, 1, -1)
                kth_b = kth_same.view(1, 1, -1)
                k_b   = k_same.view(1, 1, -1)
                full = (k_b >= k)                                         # (1, 1, n_cal)
                enter = full & (S_b > kth_b)                              # (B, 1, n_cal)
                grow = ~full                                              # (1, 1, n_cal)
                applies_enter = mask_c & enter
                applies_grow  = mask_c & grow
                delta_enter = (S_b - kth_b)
                new_sum_same = sum_b + torch.where(applies_enter, delta_enter, torch.zeros_like(S_b)) \
                                       + torch.where(applies_grow,  S_b,         torch.zeros_like(S_b))
                new_k_same = k_b + applies_grow.float()
                mean_same_upd = new_sum_same / new_k_same.clamp_min(1.0)
                d_same_upd = torch.arccos(mean_same_upd.clamp(-1.0, 1.0))
            else:
                max_b = max_same.view(1, 1, -1)
                new_max_same = torch.where(mask_c, torch.maximum(max_b, S_b), max_b.expand_as(S_b.expand(B, K, -1)))
                d_same_upd = torch.arccos(new_max_same.clamp(-1.0, 1.0))

            if ncm.numerator_only:
                updated_scores = d_same_upd
            else:
                other_mask_c = ~mask_c
                if ncm.topk_other:
                    sum_b_o = sum_other.view(1, 1, -1)
                    kth_b_o = kth_other.view(1, 1, -1)
                    k_b_o   = k_other.view(1, 1, -1)
                    full_o = (k_b_o >= k)
                    enter_o = full_o & (S_b > kth_b_o)
                    grow_o = ~full_o
                    applies_enter_o = other_mask_c & enter_o
                    applies_grow_o  = other_mask_c & grow_o
                    delta_enter_o = (S_b - kth_b_o)
                    new_sum_other = sum_b_o + torch.where(applies_enter_o, delta_enter_o, torch.zeros_like(S_b)) \
                                              + torch.where(applies_grow_o,  S_b,            torch.zeros_like(S_b))
                    new_k_other = k_b_o + applies_grow_o.float()
                    mean_other_upd = new_sum_other / new_k_other.clamp_min(1.0)
                    d_other_upd = torch.arccos(mean_other_upd.clamp(-1.0, 1.0))
                else:
                    max_b_o = max_other.view(1, 1, -1)
                    new_max_other = torch.where(other_mask_c, torch.maximum(max_b_o, S_b),
                                                 max_b_o.expand_as(S_b.expand(B, K, -1)))
                    d_other_upd = torch.arccos(new_max_other.clamp(-1.0, 1.0))
                updated_scores = d_same_upd / (d_other_upd + eps)

            updated_scores = torch.where(torch.isfinite(updated_scores), updated_scores,
                                           torch.full_like(updated_scores, 1e9))

            # ---- p-values: (1 + |{j: upd[b,c,j] >= test[b,c]}|) / (n_cal + 1) ----
            n_greater = (updated_scores >= test_scores.unsqueeze(-1)).sum(dim=-1)
            p_values = (n_greater.float() + 1.0) / (n_cal + 1.0)
            p_values_chunks.append(p_values.cpu().numpy())

            # Free batch tensors before next iter (helps under memory pressure)
            del S, S_b, mask_c, updated_scores, d_same_upd, test_scores
            if need_other:
                del d_other_upd

        p_values_arr = np.concatenate(p_values_chunks, axis=0)  # (n_test, K)

        prediction_sets = []
        set_sizes = np.zeros(n_test, dtype=int)
        empty_count = 0
        for i in range(n_test):
            include = p_values_arr[i] > self.alpha
            pred_set = classes_np[include].astype(int).tolist()
            if len(pred_set) == 0:
                empty_count += 1
            prediction_sets.append(pred_set)
            set_sizes[i] = len(pred_set)

        if empty_count > 0 and verbose:
            print(f"\nWarning: {empty_count}/{n_test} prediction sets are empty!")
            print(f"  Consider increasing alpha (current: {self.alpha}) or checking your data.")

        prediction_time = time.time() - start_time

        results = {
            'prediction_sets': prediction_sets,
            'set_sizes': set_sizes,
            'prediction_time': prediction_time,
        }

        if return_p_values:
            results['p_values'] = [
                {int(classes_np[c]): float(p_values_arr[i, c]) for c in range(K)}
                for i in range(n_test)
            ]

        if verbose:
            print(f"\nGPU FCP prediction time: {prediction_time:.2f}s "
                  f"({prediction_time / n_test * 1000:.2f}ms per sample)")

        return results

    def _predict_ridge_softmax_gpu(
        self,
        X_test: np.ndarray,
        return_p_values: bool = False,
        verbose: bool = True,
        batch_size: int = 128,
    ) -> Dict:
        """Vectorised (torch) path for FCP with RidgeSoftmaxNCM.

        Replaces the n_test x K Python loop with batched torch ops. Every per-test
        quantity is a rank-1 (Sherman-Morrison) correction to a cal-only precompute,
        followed by the ridge PRESS leave-one-out -- math identical to
        RidgeSoftmaxNCM.score_x / updated_calibration_scores_for. Runs on CUDA if
        available, else vectorised CPU (still far faster than the Python loop).

        REQUIRES every candidate class to be present in calibration (true for the
        balanced protocol). Raises otherwise so predict() can fall back to the CPU
        loop (the missing-class branch is only needed for class-dropping splits).
        """
        ncm = self.ncm
        if not isinstance(ncm, RidgeSoftmaxNCM):
            raise ValueError("ridge GPU path requires RidgeSoftmaxNCM; "
                             f"got {type(ncm).__name__}.")
        classes_np = np.asarray(self.classes)
        K = len(classes_np)
        cls_to_col = ncm._cls_to_col
        if not all(int(c) in cls_to_col for c in classes_np):
            raise ValueError(
                "ridge_softmax GPU path requires every candidate class present in "
                "cal; use device='cpu' for splits that drop classes.")

        start_time = time.time()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f = torch.float64
        T = float(ncm._T); la = float(ncm.lam_anchor); eps = float(ncm.eps)
        loo = bool(ncm.loo)

        Z = torch.as_tensor(ncm.Z, dtype=f, device=dev)            # (n, d)
        A_inv = torch.as_tensor(ncm.A_inv, dtype=f, device=dev)    # (d, d)
        W = torch.as_tensor(ncm.W, dtype=f, device=dev)            # (d, Kc)
        M = torch.as_tensor(ncm.M, dtype=f, device=dev)            # (d, Kc)
        n, Kc = Z.shape[0], W.shape[1]
        n_c = torch.as_tensor(ncm.n_c, dtype=f, device=dev)        # (Kc,)
        ycol = torch.as_tensor(ncm.y_col, dtype=torch.long, device=dev)   # (n,)
        col_of = torch.as_tensor([cls_to_col[int(c)] for c in classes_np],
                                 dtype=torch.long, device=dev)     # (K,)

        F_base = Z @ W                                             # (n, Kc)
        G = Z @ (A_inv @ M)                                        # (n, Kc)
        h0 = (Z @ A_inv * Z).sum(dim=1)                            # (n,)

        Xt = torch.as_tensor(np.asarray(X_test), dtype=f, device=dev)
        n_test = Xt.shape[0]
        p_chunks = []

        for s0 in range(0, n_test, batch_size):
            Xb = Xt[s0:min(s0 + batch_size, n_test)]               # (B, d)
            B = Xb.shape[0]
            U = Xb @ A_inv                                         # (B, d)
            rho = (Xb * U).sum(dim=1)                              # (B,)
            denom = 1.0 + rho
            inv_dn = 1.0 / denom                                   # (B,)
            Q = U @ Z.T                                            # (B, n)
            SW = Xb @ W                                            # (B, Kc)
            UM = U @ M                                             # (B, Kc)
            if loo:
                h_test = (rho * inv_dn).clamp(max=1.0 - eps)       # (B,)
                h_cal = (h0.unsqueeze(0) - Q * Q * inv_dn.unsqueeze(1)).clamp(max=1.0 - eps)
            else:
                h_test = torch.zeros(B, dtype=f, device=dev)
                h_cal = torch.zeros((B, n), dtype=f, device=dev)

            Qd = Q * inv_dn.unsqueeze(1)                           # (B, n)
            # F1[b,i,c] = F_base[i,c] - Q[b,i]*SW[b,c]/denom[b]
            F1 = F_base.unsqueeze(0) - Q.unsqueeze(2) * (SW * inv_dn.unsqueeze(1)).unsqueeze(1)
            # Fy[b,i,c] = F1 + Qd + la*(Qd - G[i,c] + Qd*UM[b,c])/(n_c[c]+1)
            anchor = (Qd.unsqueeze(2) - G.unsqueeze(0)
                      + Qd.unsqueeze(2) * UM.unsqueeze(1)) / (n_c.view(1, 1, Kc) + 1.0)
            Fy = F1 + Qd.unsqueeze(2) + la * anchor                # (B, n, Kc)

            f_test_base = SW * inv_dn.unsqueeze(1)                 # (B, Kc)
            f_y = (SW + (denom.unsqueeze(1) - 1.0)
                   + la * ((denom.unsqueeze(1) - 1.0) - UM) / (n_c.view(1, Kc) + 1.0)) \
                * inv_dn.unsqueeze(1)                              # (B, Kc)

            inv1mh_cal = 1.0 / (1.0 - h_cal)                       # (B, n)
            sub_cal = h_cal * inv1mh_cal                           # (B, n)
            inv1mh_test = 1.0 / (1.0 - h_test)                     # (B,)
            sub_test = h_test * inv1mh_test                        # (B,)
            ycol_b = ycol.view(1, n, 1).expand(B, n, 1)

            p_b = torch.empty((B, K), dtype=f, device=dev)
            for ci in range(K):
                c = int(col_of[ci])
                L = F1.clone()
                L[:, :, c] = Fy[:, :, c]
                Lp = L * inv1mh_cal.unsqueeze(2)
                Lp.scatter_add_(2, ycol_b, (-sub_cal).unsqueeze(2))   # PRESS at true class
                P = torch.softmax(Lp / T, dim=2)
                s_cal = 1.0 - P.gather(2, ycol_b).squeeze(2)          # (B, n)
                Lt = f_test_base.clone()
                Lt[:, c] = f_y[:, c]
                Ltp = Lt * inv1mh_test.unsqueeze(1)
                Ltp[:, c] = Ltp[:, c] - sub_test                      # PRESS at candidate (test's class)
                Pt = torch.softmax(Ltp / T, dim=1)
                s_test = 1.0 - Pt[:, c]                               # (B,)
                ng = (s_cal >= s_test.unsqueeze(1)).sum(dim=1)
                p_b[:, ci] = (ng.to(f) + 1.0) / (n + 1.0)
            p_chunks.append(p_b.cpu().numpy())
            del F1, Fy, Q, SW, UM

        p_values_arr = np.concatenate(p_chunks, axis=0)            # (n_test, K)
        prediction_sets, set_sizes, empty_count = [], np.zeros(n_test, dtype=int), 0
        for i in range(n_test):
            include = p_values_arr[i] > self.alpha
            pred_set = classes_np[include].astype(int).tolist()
            if not pred_set:
                empty_count += 1
            prediction_sets.append(pred_set)
            set_sizes[i] = len(pred_set)
        if empty_count > 0 and verbose:
            print(f"\nWarning: {empty_count}/{n_test} prediction sets are empty!")
        prediction_time = time.time() - start_time
        results = {'prediction_sets': prediction_sets, 'set_sizes': set_sizes,
                   'prediction_time': prediction_time}
        if return_p_values:
            results['p_values'] = [
                {int(classes_np[c]): float(p_values_arr[i, c]) for c in range(K)}
                for i in range(n_test)]
        if verbose:
            print(f"\nGPU ridge_softmax FCP time: {prediction_time:.2f}s "
                  f"({prediction_time / n_test * 1000:.2f}ms/sample) on {dev}")
        return results

    def _predict_prototype_softmax_gpu(
        self,
        X_test: np.ndarray,
        return_p_values: bool = False,
        verbose: bool = True,
        batch_size: int = 256,
    ) -> Dict:
        """Vectorised (torch) path for FCP with PrototypeSoftmaxNCM.

        Replaces the n_test x K Python loop with batched torch ops. Folding (x, y)
        into the bag changes ONLY the class-y prototype, so each cal point's score
        is a single-column softmax update of a cal-only precompute -- math identical
        to PrototypeSoftmaxNCM.score_x / updated_calibration_scores_for. float64 ->
        bit-exact with the CPU path. Runs on CUDA if available, else vectorised CPU.

        REQUIRES every candidate class present in cal (true for the balanced
        protocol). Raises otherwise so predict() can fall back to the CPU loop.
        """
        ncm = self.ncm
        if not isinstance(ncm, PrototypeSoftmaxNCM):
            raise ValueError("prototype GPU path requires PrototypeSoftmaxNCM; "
                             f"got {type(ncm).__name__}.")
        classes_np = np.asarray(self.classes)
        K = len(classes_np)
        cls_to_col = ncm._cls_to_col
        if not all(int(c) in cls_to_col for c in classes_np):
            raise ValueError(
                "prototype_softmax GPU path requires every candidate class present "
                "in cal; use device='cpu' for splits that drop classes.")

        start_time = time.time()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f = torch.float64
        T = float(ncm._T); eps = float(ncm.eps); cosine = bool(ncm.cosine)

        Z = torch.as_tensor(ncm.Z, dtype=f, device=dev)               # (n, d) prepped
        P = torch.as_tensor(ncm.P, dtype=f, device=dev)               # (d, Kc)
        class_sum = torch.as_tensor(ncm.class_sum, dtype=f, device=dev)  # (d, Kc)
        n_c = torch.as_tensor(ncm.n_c, dtype=f, device=dev)           # (Kc,)
        ycol = torch.as_tensor(ncm.y_col, dtype=torch.long, device=dev)  # (n,)
        F_base = torch.as_tensor(ncm.F_base, dtype=f, device=dev)     # (n, Kc) cal LOO
        n, Kc = Z.shape[0], P.shape[1]
        col_of = torch.as_tensor([cls_to_col[int(c)] for c in classes_np],
                                 dtype=torch.long, device=dev)        # (K,)

        # cal-only precomputes (independent of the test point)
        ZS = Z @ class_sum                                            # (n, Kc) <z_i, sum_c>
        znorm2 = (Z * Z).sum(dim=1)                                   # (n,)
        css_norm2 = (class_sum * class_sum).sum(dim=0)                # (Kc,) ||sum_c||^2
        eps2 = eps * eps
        NEG = torch.tensor(float("-inf"), dtype=f, device=dev)
        # cosine fast path: cache the softmax denominator. Cosine logits are in
        # [-1,1] so exp(./T) is bounded (no overflow); folding (x,y) swaps ONE
        # class-y exp term per cal point -> D' = D_base - e_old + e_new, O(n) per
        # candidate (O(n*K) total) instead of recomputing a K-way softmax per
        # candidate (O(n*K^2)). dot logits are unbounded -> keep the stable path.
        if cosine:
            E_base = torch.exp(F_base / T)                           # (n, Kc) bounded
            D_base = E_base.sum(dim=1)                               # (n,) denominator
            E_true = E_base.gather(1, ycol.view(n, 1)).squeeze(1)    # (n,) own-class exp

        Xt = torch.as_tensor(np.asarray(X_test), dtype=f, device=dev)
        n_test = Xt.shape[0]
        p_chunks = []

        for s0 in range(0, n_test, batch_size):
            Xb = Xt[s0:min(s0 + batch_size, n_test)]                 # (B, d) raw
            if cosine:
                Xb = Xb / torch.clamp(Xb.norm(dim=1, keepdim=True), min=eps)
            B = Xb.shape[0]
            Zzx = Xb @ Z.T                                           # (B, n) <z_x, z_i>
            csTzx = Xb @ class_sum                                   # (B, Kc) <z_x, sum_c>
            zxnorm2 = (Xb * Xb).sum(dim=1)                           # (B,)
            f_test = Xb @ P                                          # (B, Kc)
            Ptest = torch.softmax(f_test / T, dim=1)                 # (B, Kc)
            norm_aug2 = (css_norm2.view(1, Kc) + 2.0 * csTzx
                         + zxnorm2.view(B, 1))                       # (B, Kc) ||sum_c+z_x||^2
            ycol_idx = ycol.view(1, n, 1).expand(B, n, 1)           # gather index

            p_b = torch.empty((B, K), dtype=f, device=dev)
            for ci in range(K):
                c = int(col_of[ci])
                base = ZS[:, c].view(1, n) + Zzx                     # (B, n) <z_i, sum_c+z_x>
                na2 = norm_aug2[:, c].view(B, 1)                     # (B, 1)
                if cosine:
                    nonmem = base / torch.clamp(torch.sqrt(na2), min=eps)
                    nonmem = torch.where(na2 < eps2, NEG, nonmem)
                    mem_den2 = na2 - 2.0 * base + znorm2.view(1, n)  # (B, n)
                    mem = (base - znorm2.view(1, n)) / torch.sqrt(torch.clamp(mem_den2, min=eps2))
                    mem = torch.where(mem_den2 < eps2, NEG, mem)
                else:
                    nonmem = base / (n_c[c] + 1.0)
                    mem = (base - znorm2.view(1, n)) / n_c[c]
                is_member = (ycol == c).view(1, n)                   # (1, n)
                col_c = torch.where(is_member, mem, nonmem)          # (B, n)
                if cosine:
                    # denominator swap: only class c's exp term moves per cal point
                    e_new = torch.exp(col_c / T)                     # (B, n) bounded
                    D = D_base.view(1, n) - E_base[:, c].view(1, n) + e_new
                    num = torch.where(is_member, e_new, E_true.view(1, n))
                    s_cal = 1.0 - num / D                            # (B, n)
                else:
                    # dot logits unbounded -> stable full softmax over (B, n, Kc)
                    L = F_base.unsqueeze(0).expand(B, n, Kc).clone()
                    L[:, :, c] = col_c
                    s_cal = 1.0 - torch.softmax(L / T, dim=2).gather(
                        2, ycol_idx).squeeze(2)                      # (B, n)
                s_test = 1.0 - Ptest[:, c]                           # (B,)
                ng = (s_cal >= s_test.view(B, 1)).sum(dim=1)         # (B,)
                p_b[:, ci] = (ng.to(f) + 1.0) / (n + 1.0)
            p_chunks.append(p_b.cpu().numpy())
            del Zzx, csTzx, f_test, Ptest, norm_aug2

        p_values_arr = np.concatenate(p_chunks, axis=0)              # (n_test, K)
        prediction_sets, set_sizes, empty_count = [], np.zeros(n_test, dtype=int), 0
        for i in range(n_test):
            include = p_values_arr[i] > self.alpha
            pred_set = classes_np[include].astype(int).tolist()
            if not pred_set:
                empty_count += 1
            prediction_sets.append(pred_set)
            set_sizes[i] = len(pred_set)
        if empty_count > 0 and verbose:
            print(f"\nWarning: {empty_count}/{n_test} prediction sets are empty!")
        prediction_time = time.time() - start_time
        results = {'prediction_sets': prediction_sets, 'set_sizes': set_sizes,
                   'prediction_time': prediction_time}
        if return_p_values:
            results['p_values'] = [
                {int(classes_np[c]): float(p_values_arr[i, c]) for c in range(K)}
                for i in range(n_test)]
        if verbose:
            print(f"\nGPU prototype_softmax FCP time: {prediction_time:.2f}s "
                  f"({prediction_time / n_test * 1000:.2f}ms/sample) on {dev}")
        return results

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate conformal prediction on test set.

        Returns:
            Dictionary with coverage, efficiency metrics
        """
        results = self.predict(X_test, return_p_values=False, verbose=verbose)
        prediction_sets = results['prediction_sets']
        set_sizes = results['set_sizes']

        # Coverage: fraction of times true label is in prediction set
        coverage = np.mean([
            y_test[i] in pred_set
            for i, pred_set in enumerate(prediction_sets)
        ])
        avg_set_size = np.mean(set_sizes)

        # Singleton accuracy: accuracy when |prediction_set| = 1
        singleton_mask = set_sizes == 1
        if singleton_mask.sum() > 0:
            singleton_correct = np.mean([
                y_test[i] == prediction_sets[i][0]
                for i in np.where(singleton_mask)[0]
            ])
            singleton_rate = singleton_mask.mean()
        else:
            singleton_correct = 0.0
            singleton_rate = 0.0

        empty_rate = np.mean(set_sizes == 0)

        metrics = {
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'median_set_size': np.median(set_sizes),
            'singleton_rate': singleton_rate,
            'singleton_accuracy': singleton_correct,
            'empty_set_rate': empty_rate,
            'alpha': self.alpha,
            'target_coverage': 1 - self.alpha,
        }

        if verbose:
            print("\n" + "=" * 50)
            print("CONFORMAL PREDICTION EVALUATION")
            print("=" * 50)
            print(f"Significance level (alpha):    {metrics['alpha']:.3f}")
            print(f"Target coverage:               {metrics['target_coverage']:.3f}")
            print(f"Actual coverage:               {metrics['coverage']:.3f}")
            print(f"Average set size:              {metrics['avg_set_size']:.3f}")
            print(f"Median set size:               {metrics['median_set_size']:.1f}")
            print(f"Singleton rate:                {metrics['singleton_rate']:.3f}")
            print(f"Singleton accuracy:            {metrics['singleton_accuracy']:.3f}")
            print(f"Empty set rate:                {metrics['empty_set_rate']:.3f}")
            print("=" * 50)

        return metrics


# ---------------------------------------------------------------------------
# Split-CP baselines moved to split_cp_baselines.py (2026-06-10); re-exported
# here so existing `from conformal_prediction import ...` call sites work.
# ---------------------------------------------------------------------------
from split_cp_baselines import (  # noqa: F401
    compute_cp_scores,
    compute_cp_sets,
    SoftmaxSplitCP,
    SemiCP,
    ClusteredSplitCP,
)


class CrossValidationPlusPredictor:
    """
    Cross-Validation+ (CV+) Conformal Prediction, inspired by Jackknife+.

    Instead of retraining on every test point (Full CP) or wasting data on a
    separate training set (Split CP), CV+ uses K-fold cross-validation:

    1. Split calibration data into K folds
    2. For each fold k, fit an NCM on all data except fold k
    3. Score each calibration point with the NCM that excluded its fold
    4. At test time, score the test point with each fold's NCM and aggregate

    Advantages over Full CP:
    - Much faster at test time (K forward passes vs n_cal updates per label)
    - Same data-efficient calibration (no separate training set)

    Advantages over Split CP:
    - Uses all calibration data for both fitting and scoring
    - No data wasted on a separate training set for the classifier

    Coverage guarantee: P(Y in C(X)) >= 1 - 2*alpha (Barber et al. 2021).
    In practice, coverage is typically close to 1 - alpha.

    Reference: Barber, Candes, Ramdas & Tibshirani (2021)
               "Predictive Inference with the Jackknife+"
    """

    def __init__(self, ncm_factory, alpha: float = 0.1, n_folds: int = 5):
        """
        Args:
            ncm_factory: Callable returning a new NCM instance
                         (e.g. lambda: GeodesicTopKMeanNCM())
            alpha: Significance level (e.g. 0.1 for 90% target coverage)
            n_folds: Number of cross-validation folds
        """
        self.ncm_factory = ncm_factory
        self.alpha = alpha
        self.n_folds = n_folds

        self.X_cal = None
        self.y_cal = None
        self.classes = None

        self.fold_ncms = []              # K fitted NCMs
        self.fold_assignments = None     # (n_cal,) fold index per calibration point
        self.cal_scores = None           # (n_cal,) leave-fold-out scores

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                  all_classes: np.ndarray = None):
        """
        Calibrate using K-fold cross-validation.

        For each fold k:
        1. Fit NCM on all data except fold k
        2. Score points in fold k using that NCM

        Args:
            X_cal: Calibration features (n_cal, d)
            y_cal: Calibration labels (n_cal,)
            all_classes: Full label space to iterate over during prediction.
                         If None, uses only classes present in y_cal.
        """
        from sklearn.model_selection import StratifiedKFold

        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(all_classes) if all_classes is not None else np.unique(y_cal)

        # Cap n_folds at the smallest class size to avoid StratifiedKFold error
        min_class_size = min(np.sum(y_cal == c) for c in self.classes)
        actual_folds = min(self.n_folds, min_class_size)
        if actual_folds < self.n_folds:
            print(f"  CV+ note: reduced folds from {self.n_folds} to {actual_folds} "
                  f"(smallest class has {min_class_size} members)")
        if actual_folds < 2:
            actual_folds = 2
            print(f"  CV+ warning: forcing n_folds=2 (minimum for CV)")

        n_cal = len(X_cal)
        self.fold_assignments = np.zeros(n_cal, dtype=int)
        self.cal_scores = np.zeros(n_cal, dtype=float)
        self.fold_ncms = []

        self._actual_folds = actual_folds

        # Use non-stratified KFold when StratifiedKFold cannot be applied
        # (i.e. some class has fewer members than n_splits)
        if min_class_size < actual_folds:
            from sklearn.model_selection import KFold
            print(f"  CV+ note: using non-stratified KFold (min class size={min_class_size} < n_folds={actual_folds})")
            skf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        else:
            skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

        start_time = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_cal, y_cal)):
            ncm = self.ncm_factory()
            ncm.fit(X_cal[train_idx], y_cal[train_idx])
            self.fold_ncms.append(ncm)

            self.fold_assignments[val_idx] = fold_idx

            # Score held-out points with the NCM that excluded them
            for i in val_idx:
                self.cal_scores[i] = ncm.score_x_cv(X_cal[i], int(y_cal[i]))

        fit_time = time.time() - start_time

        cal_classes = np.unique(y_cal)
        print(f"CV+ Calibrated: {n_cal} examples, {len(cal_classes)} cal classes, "
              f"{len(self.classes)} candidate classes, {actual_folds} folds")
        print(f"Calibration scores - min: {self.cal_scores.min():.4f}, "
              f"max: {self.cal_scores.max():.4f}, mean: {self.cal_scores.mean():.4f}")
        print(f"Timing: calibrate={fit_time:.2f}s")

    def predict(
        self,
        X_test: np.ndarray,
        return_p_values: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Compute prediction sets using the CV+ formula.

        For each test point x and candidate label y:
        1. Compute K test scores (one per fold NCM)
        2. For each cal point i (in fold k(i)), compare R_i vs R_test^{k(i)}
        3. p_value = (1 + sum 1{R_i >= R_test^{k(i)}}) / (n_cal + 1)

        Returns:
            Dict with 'prediction_sets', 'set_sizes', 'prediction_time',
            and optionally 'p_values'.
        """
        if self.cal_scores is None:
            raise ValueError("Must call calibrate() before predict()")

        start_time = time.time()
        n_test = len(X_test)
        n_cal = len(self.X_cal)
        n_folds = self._actual_folds

        prediction_sets = []
        set_sizes = []
        all_p_values = [] if return_p_values else None
        empty_count = 0

        iterator = tqdm(range(n_test), desc="CV+") if verbose else range(n_test)

        for i in iterator:
            x_test = X_test[i]
            pred_set = []
            # We always populate p_vals so the verbose empty-set warning can
            # report the best (max) p-value; only emit it when requested.
            p_vals = {}

            for y_candidate in self.classes:
                yc = int(y_candidate)
                # K test scores, one per fold NCM
                test_scores_per_fold = np.array([
                    self.fold_ncms[k].score_x_cv(x_test, yc) for k in range(n_folds)
                ])

                # Vectorised CV+ comparison:
                # For each cal point j in fold k(j), compare R_j >= R_test^{k(j)}.
                test_score_per_cal = test_scores_per_fold[self.fold_assignments]
                n_greater = np.sum(self.cal_scores >= test_score_per_cal)

                p_value = (n_greater + 1) / (n_cal + 1)
                p_vals[yc] = p_value

                if p_value > self.alpha:
                    pred_set.append(yc)

            if len(pred_set) == 0:
                empty_count += 1
                if empty_count <= 3 and verbose:
                    best_class = max(p_vals, key=p_vals.get)
                    print(f"\nWarning: Empty prediction set for test example {i}")
                    print(f"  Best p-value: {p_vals[best_class]:.4f} (class {best_class}), "
                          f"alpha: {self.alpha}")

            prediction_sets.append(pred_set)
            set_sizes.append(len(pred_set))

            if return_p_values:
                all_p_values.append(p_vals)

        if empty_count > 0 and verbose:
            print(f"\nWarning: {empty_count}/{n_test} prediction sets are empty!")

        prediction_time = time.time() - start_time

        results = {
            'prediction_sets': prediction_sets,
            'set_sizes': np.array(set_sizes),
            'prediction_time': prediction_time,
        }

        if return_p_values:
            results['p_values'] = all_p_values

        if verbose:
            print(f"\nPrediction time: {prediction_time:.2f}s "
                  f"({prediction_time / n_test * 1000:.1f}ms per sample)")

        return results

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """Evaluate CV+ prediction sets on test data."""
        results = self.predict(X_test, return_p_values=False, verbose=verbose)
        prediction_sets = results['prediction_sets']
        set_sizes = results['set_sizes']

        coverage = np.mean([
            y_test[i] in pred_set
            for i, pred_set in enumerate(prediction_sets)
        ])
        avg_set_size = np.mean(set_sizes)

        singleton_mask = set_sizes == 1
        if singleton_mask.sum() > 0:
            singleton_correct = np.mean([
                y_test[i] == prediction_sets[i][0]
                for i in np.where(singleton_mask)[0]
            ])
            singleton_rate = singleton_mask.mean()
        else:
            singleton_correct = 0.0
            singleton_rate = 0.0

        empty_rate = np.mean(set_sizes == 0)

        metrics = {
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'median_set_size': np.median(set_sizes),
            'singleton_rate': singleton_rate,
            'singleton_accuracy': singleton_correct,
            'empty_set_rate': empty_rate,
            'alpha': self.alpha,
            'target_coverage': 1 - self.alpha
        }

        if verbose:
            print("\n" + "=" * 50)
            print("CV+ CONFORMAL PREDICTION EVALUATION")
            print("=" * 50)
            print(f"Significance level (alpha):    {metrics['alpha']:.3f}")
            print(f"Target coverage:               {metrics['target_coverage']:.3f}")
            print(f"Theoretical guarantee:         >={1 - 2 * self.alpha:.3f}")
            print(f"Actual coverage:               {metrics['coverage']:.3f}")
            print(f"Average set size:              {metrics['avg_set_size']:.3f}")
            print(f"Median set size:               {metrics['median_set_size']:.1f}")
            print(f"Singleton rate:                {metrics['singleton_rate']:.3f}")
            print(f"Singleton accuracy:            {metrics['singleton_accuracy']:.3f}")
            print(f"Empty set rate:                {metrics['empty_set_rate']:.3f}")
            print("=" * 50)

        return metrics


def create_ncm(ncm_type: str, k: int = 5,
               reg: float = 1e-4,
               allow_nonexchangeable: bool = False,
               *, temperature: Optional[float] = None,
               lam_ridge: float = 1.0, lam_anchor: float = 1.0,
               loo: bool = True, logit: str = "cosine") -> NonconformityMeasure:
    """
    Factory function to create NCM instances.

    Args:
        ncm_type: One of 'softmax', 'mahal_nn_ratio', 'whitened_geodesic',
                  'geodesic_topk_mean', 'geodesic_topk_asym',
                  'unwhitened_topk_mean', 'unwhitened_topk_asym',
                  'geodesic_topk', 'geodesic_1nn', 'rbf_density', 'ridge_softmax',
                  'prototype_softmax'
        k: Number of neighbors for top-k NCMs
        reg: Variance regularisation for Mahalanobis NCMs
        temperature, lam_ridge, lam_anchor, loo: 'ridge_softmax' only (keyword-
            only; all other NCM types ignore them). See RidgeSoftmaxNCM.
            temperature=None auto-fits T on cal (O(1/n), warns); pass a fixed
            float for the exact guarantee.
        temperature, logit: 'prototype_softmax' only (keyword-only). The vanilla
            text-free FCA NCM (class-mean prototype -> softmax LAC, no covariance
            term). logit='cosine' (default) or 'dot'. temperature=None warns
            (O(1/n)); pass a fixed float for the exact guarantee. See
            PrototypeSoftmaxNCM.
        allow_nonexchangeable: Approve (and silence the validity warning for) the
            NCM types that fit a data-dependent transform on the CALIBRATION set
            and therefore break FCP exchangeability at O(1/n): every whitened
            geodesic variant ('geodesic_topk_mean', 'geodesic_topk_asym',
            'geodesic_topk', 'geodesic_1nn', 'whitened_geodesic'),
            'mahal_nn_ratio', and 'rbf_density'. The 'unwhitened_topk_*' types and
            'softmax' are exchangeable and ignore this flag.

    Exchangeable default: the 'unwhitened_topk_*' types fit NO cal-dependent
    transform. To recover the whitening efficiency while staying exact, feed their
    features through an exchangeable_features.UnlabeledTransform (whitening/PCA fit
    on an independent unlabeled pool). The whitened named types below preserve
    their historical (cal-fit) numeric behavior but now require this approval flag
    to run without a validity warning.
    """
    nx = allow_nonexchangeable  # shorthand
    if ncm_type == "softmax":
        return SoftmaxNonconformity()
    elif ncm_type == "mahal_nn_ratio":
        return MahalNNRatio(reg=reg, allow_nonexchangeable=nx)
    elif ncm_type == "whitened_geodesic":
        return WhitenedGeodesicNNRatio(reg=reg, allow_nonexchangeable=nx)
    elif ncm_type == "geodesic_topk_mean":
        # Symmetric: topk on both same and other (cal-fit whitening -> approve)
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=True, topk_other=True,
                                   whiten=True, allow_nonexchangeable=nx)
    elif ncm_type == "geodesic_topk_asym":
        # Asymmetric: 1-NN for same (tight numerator), topk for other (Trojan Horse fix)
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=False, topk_other=True,
                                   whiten=True, allow_nonexchangeable=nx)
    elif ncm_type == "unwhitened_topk_mean":
        # Exchangeable: symmetric topk WITHOUT whitening
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=True, topk_other=True, whiten=False)
    elif ncm_type == "unwhitened_topk_asym":
        # Exchangeable: asymmetric topk WITHOUT whitening
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=False, topk_other=True, whiten=False)
    elif ncm_type == "geodesic_topk":
        # Numerator-only: mean of top-k same-class geodesic distances, no ratio
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=True, numerator_only=True,
                                   whiten=True, allow_nonexchangeable=nx)
    elif ncm_type == "geodesic_1nn":
        # Numerator-only: 1-NN same-class geodesic distance, no ratio
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=False, numerator_only=True,
                                   whiten=True, allow_nonexchangeable=nx)
    elif ncm_type == "rbf_density":
        # Gaussian-kernel density NCM. Bandwidth auto from cal median heuristic.
        return RBFDensityNCM(allow_nonexchangeable=nx)
    elif ncm_type == "ridge_softmax":
        # FCA-inspired (text-free SS-Text): class-mean-anchored ridge probe ->
        # softmax LAC. Exact for fixed temperature; temperature=None warns.
        return RidgeSoftmaxNCM(lam=lam_ridge, lam_anchor=lam_anchor,
                               temperature=temperature, loo=loo,
                               allow_nonexchangeable=nx)
    elif ncm_type == "prototype_softmax":
        # Vanilla text-free FCA: class-mean prototype -> softmax LAC. NO ridge
        # covariance term (cf. ridge_softmax) -- the clean ablation control.
        # Exact for fixed temperature; temperature=None warns.
        return PrototypeSoftmaxNCM(temperature=temperature, logit=logit,
                                   allow_nonexchangeable=nx)
    else:
        raise ValueError(f"Unknown NCM type: {ncm_type}")


def train_cal_test_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.4,
    cal_ratio: float = 0.3,
    random_state: int = 42
) -> Tuple:
    """
    Split data into train/calibration/test sets for Split CP.

    This is needed for SoftmaxSplitCP which requires a training set
    (unlike Full CP which only needs calibration + test).

    Args:
        embeddings: Feature vectors (n, d)
        labels: Labels (n,)
        train_ratio: Fraction for training
        cal_ratio: Fraction for calibration (rest goes to test)
        random_state: Random seed

    Returns:
        (X_train, y_train, X_cal, y_cal, X_test, y_test)
    """
    np.random.seed(random_state)
    n = len(embeddings)
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_cal = int(n * cal_ratio)

    train_idx = indices[:n_train]
    cal_idx = indices[n_train:n_train + n_cal]
    test_idx = indices[n_train + n_cal:]

    X_train, y_train = embeddings[train_idx], labels[train_idx]
    X_cal, y_cal = embeddings[cal_idx], labels[cal_idx]
    X_test, y_test = embeddings[test_idx], labels[test_idx]

    print(f"Split: Train={len(X_train)}, Cal={len(X_cal)}, Test={len(X_test)}")
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def cal_test_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cal_ratio: float = 0.5,
    random_state: int = 42
) -> Tuple:
    """
    Split data into calibration/test sets for Full CP.

    Note: Full CP doesn't require a separate training set.
    We work directly with calibration data and test data.

    Args:
        embeddings: Feature vectors (n, d)
        labels: Labels (n,)
        cal_ratio: Fraction for calibration (rest goes to test)
        random_state: Random seed

    Returns:
        (X_cal, y_cal, X_test, y_test)
    """
    np.random.seed(random_state)
    n = len(embeddings)
    indices = np.random.permutation(n)
    n_cal = int(n * cal_ratio)

    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]

    X_cal, y_cal = embeddings[cal_idx], labels[cal_idx]
    X_test, y_test = embeddings[test_idx], labels[test_idx]

    print(f"Split: Cal={len(X_cal)}, Test={len(X_test)}")
    return X_cal, y_cal, X_test, y_test


def stratified_cal_test_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cal_size: int,
    test_size: int,
    balanced: bool = True,
    random_state: int = 42
) -> Tuple:
    """
    Stratified calibration/test split guaranteeing class balance.

    Args:
        embeddings: Feature vectors (n, d)
        labels: Labels (n,)
        cal_size: Number of calibration samples
        test_size: Number of test samples
        balanced: If True, calibration has exactly cal_size//n_classes per class
                  (remainder distributed round-robin). If False, guarantees >=1/class
                  with rest random.
        random_state: Random seed

    Returns:
        (X_cal, y_cal, X_test, y_test)
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(labels)
    n_classes = len(classes)

    # Build per-class index pools
    class_indices = {c: np.where(labels == c)[0] for c in classes}
    for c in classes:
        class_indices[c] = rng.permutation(class_indices[c])

    # --- Test set: stratified (test_size // n_classes per class) ---
    test_per_class = test_size // n_classes
    test_idx = []
    for c in classes:
        available = class_indices[c]
        take = min(test_per_class, len(available))
        test_idx.append(available[:take])
        class_indices[c] = available[take:]
    test_idx = np.concatenate(test_idx)

    # --- Calibration set ---
    if balanced:
        cal_per_class = cal_size // n_classes
        remainder = cal_size - cal_per_class * n_classes
        cal_idx = []
        for i, c in enumerate(classes):
            available = class_indices[c]
            take = cal_per_class + (1 if i < remainder else 0)
            take = min(take, len(available))
            cal_idx.append(available[:take])
        cal_idx = np.concatenate(cal_idx)
    else:
        # Guarantee >=1 per class, rest random
        cal_idx = []
        for c in classes:
            available = class_indices[c]
            cal_idx.append(available[:1])
            class_indices[c] = available[1:]
        cal_idx_first = np.concatenate(cal_idx)

        # Remaining budget from all leftover indices
        all_remaining = np.concatenate([class_indices[c] for c in classes])
        n_extra = cal_size - len(cal_idx_first)
        if n_extra > 0:
            extra = rng.choice(all_remaining, size=min(n_extra, len(all_remaining)), replace=False)
            cal_idx = np.concatenate([cal_idx_first, extra])
        else:
            cal_idx = cal_idx_first[:cal_size]

    cal_idx = rng.permutation(cal_idx)

    X_cal = embeddings[cal_idx]
    y_cal = labels[cal_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]

    # Verify
    cal_classes = np.unique(y_cal)
    assert len(cal_classes) == n_classes, (
        f"Calibration missing classes: have {len(cal_classes)}, expected {n_classes}")

    if balanced:
        counts = np.array([np.sum(y_cal == c) for c in classes])
        expected = cal_size // n_classes
        assert np.all(counts >= expected), (
            f"Balance violated: min count {counts.min()}, expected {expected}")

    print(f"Stratified split: Cal={len(X_cal)} ({'balanced' if balanced else 'min-1/class'}), "
          f"Test={len(X_test)}, Classes={n_classes}")

    return X_cal, y_cal, X_test, y_test


def stratified_pool_split(X, y, cal_size, test_size, all_classes, rng):
    """Stratified POOL -> cal + test split with a random cal/test boundary.

    Balances only the carved pool (ceil((cal_size+test_size)/K) per class),
    then takes test as the last test_size of the shuffled pool; cal gets >=1
    sample per class surviving in the remainder, rest random. Unlike
    stratified_cal_test_split(balanced=True), cal itself is NOT forced
    class-balanced: the label-blind cal/test boundary keeps the bag exactly
    exchangeable (tight ~1-alpha coverage), but at small cal some classes can
    be missing from cal entirely (relies on the NCM's symmetric missing-class
    handling). Historically duplicated in macs_experiment.py /
    mscs_unlabeled_experiment.py as `stratified_split`; hoisted verbatim.
    """
    n_classes = len(all_classes)
    num_per_class = math.ceil((test_size + cal_size) / n_classes)
    min_count = min(np.sum(y == c) for c in all_classes)
    num_per_class = min(num_per_class, min_count)

    pool_idx = []
    for c in all_classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, num_per_class, replace=False)
        pool_idx.append(chosen)
    pool_idx = np.concatenate(pool_idx)
    rng.shuffle(pool_idx)

    X_pool, y_pool = X[pool_idx], y[pool_idx]
    X_test, y_test = X_pool[-test_size:], y_pool[-test_size:]
    X_rem, y_rem = X_pool[:-test_size], y_pool[:-test_size]

    # Stratified cal
    rem_classes = np.unique(y_rem)
    first = np.array([rng.choice(np.where(y_rem == c)[0], 1, replace=False)[0]
                      for c in rem_classes])
    rest = np.setdiff1d(np.arange(len(X_rem)), first)
    n_extra = cal_size - len(rem_classes)
    if n_extra > 0:
        extra = rng.choice(rest, min(n_extra, len(rest)), replace=False)
        cal_idx = np.concatenate([first, extra])
    else:
        cal_idx = first[:cal_size]
    cal_idx = rng.permutation(cal_idx)

    return X_rem[cal_idx], y_rem[cal_idx], X_test, y_test
