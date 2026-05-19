"""
Full Conformal Prediction with Optimizations
Based on: Cherubin et al. 2021 - "Exact and Approximate Conformal Inference for Multi-Output Regression"
https://proceedings.mlr.press/v139/cherubin21a/cherubin21a.pdf

This implementation provides:
1. Full Conformal Prediction (FCP) - exact p-values for each test example
2. Efficient nonconformity measures for SSL embeddings
3. Optimizations: caching, vectorization, early stopping
"""

import numpy as np
import torch
import time
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm



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

    def __init__(self, reg: float = 1e-4):
        self.reg = reg
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

    def __init__(self, reg: float = 1e-4):
        """
        Args:
            reg: Whitening regularization floor (adaptive version used in fit()).
        """
        self.reg = reg
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

    Exchangeability (O(1/n) approximation):
    - Whitening (inv_std) is fixed at fit() time from calibration data only.
      For exact exchangeability, pooled_var should be recomputed on the augmented
      bag {z1,...,zn,(x*,y*)} for each candidate. The asymmetry is O(1/n) and
      negligible for n >= 200. See Fan & Sesia (2025, arXiv 2512.15383).

    FCP O(N) compatibility: CASE A update is O(1) per calibration point.
    """

    K_MAX = 5  # cap; diminishing Trojan-Horse protection beyond k=5

    def __init__(self, reg: float = 1e-4, k: Optional[int] = None,
                 topk_same: bool = True, topk_other: bool = True,
                 numerator_only: bool = False, whiten: bool = True):
        """
        Args:
            reg:            Whitening regularization floor.
            k:              Neighbors for mean pooling. None → adaptive.
            topk_same:      If True, average same-class sims (numerator). Default True.
            topk_other:     If True, average other-class sims (denominator). Default True.
            numerator_only: If True, score = d_same only (no ratio). Avoids denominator
                            collapse on well-separated datasets. Ignores topk_other.
            whiten:         If False, skip pooled whitening (ablation). Default True.
        """
        self.reg           = reg
        self.k_override    = k
        self.topk_same     = topk_same
        self.topk_other    = topk_other
        self.numerator_only = numerator_only
        self.whiten     = whiten
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

        # Pooled whitening (Fix 4: adaptive reg) — skip if whiten=False (ablation)
        if self.whiten:
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
        if not np.any(mask_same):
            return 1e9

        def _topk_scalar(sim_arr):
            ke = min(k, len(sim_arr))
            if ke == 0:
                return 0.0, 0.0
            top = sim_arr[np.argpartition(sim_arr, -ke)[-ke:]] if ke < len(sim_arr) else sim_arr
            return float(top.sum()), float(ke)

        # Same-class
        if self.topk_same:
            sum_s, ke_s = _topk_scalar(sims[mask_same])
            max_s = None
        else:
            sum_s = ke_s = None
            max_s = float(np.max(sims[mask_same]))

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
                 eps_log: float = 1e-30):
        """
        Args:
            sigma: Kernel bandwidth. If None, set at fit() via bandwidth_rule.
            bandwidth_rule: 'median' (median pairwise dist), 'knn' (median dist to
                            k=10 NN), 'within_class_median' (median of same-class
                            pairwise dist; exchangeable so long as labels are
                            calibration-only).
            sigma_scale: Multiplier on the auto-chosen sigma. 1.0 = use as-is.
            ratio: If True, score = -log(Σ_same K) + log(Σ_other K) (log-odds).
                   If False, score = -log(Σ_same K) (pure within-class density).
            eps_log: Floor inside the log when a class has no neighbors at all.
        """
        self.sigma = sigma
        self.bandwidth_rule = bandwidth_rule
        self.sigma_scale = sigma_scale
        self.ratio = ratio
        self.eps_log = eps_log
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
        self.X_cal = X_cal.astype(np.float64)
        self.y_cal = np.asarray(y_cal)
        n_cal, _ = X_cal.shape

        # --- Bandwidth ---
        if self.sigma is None:
            auto = self._auto_sigma(self.X_cal, self.y_cal, self.bandwidth_rule)
            self.sigma = max(auto * self.sigma_scale, 1e-6)
        self.gamma_ = 1.0 / (2.0 * self.sigma ** 2)

        # --- log-kernel matrix (cal × cal) ---
        D2 = self._pairwise_dist2(self.X_cal)
        self.logK_cal_ = -self.gamma_ * D2
        np.fill_diagonal(self.logK_cal_, -np.inf)

        # --- Baseline within-class log-sum-exp ---
        self.baseline_lse_same_ = np.full(n_cal, -np.inf)
        for c in np.unique(self.y_cal):
            idx = np.where(self.y_cal == c)[0]
            sub = self.logK_cal_[np.ix_(idx, idx)]
            self.baseline_lse_same_[idx] = self._row_logsumexp(sub)

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
            d2 = ((self.X_cal - x[None, :]) ** 2).sum(axis=1)
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
        lse_same = self._lse_vec(logk[mask_same])
        if not self.ratio:
            return float(-lse_same)
        mask_other = ~mask_same
        lse_other = self._lse_vec(logk[mask_other]) if np.any(mask_other) else -np.inf
        return float(-lse_same + (lse_other if np.isfinite(lse_other) else np.log(self.eps_log)))

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
            new_lse_same[idx_same] = np.logaddexp(
                self.baseline_lse_same_[idx_same], logk[idx_same])

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
        start_time = time.time()
        self.ncm.fit(X_cal, y_cal)
        fit_time = time.time() - start_time

        # Compute and cache calibration scores
        start_time = time.time()
        # Try new API first (get_calibration_scores), fallback to legacy (score)
        try:
            self.cal_scores = self.ncm.get_calibration_scores()
        except (AttributeError, NotImplementedError):
            # Fallback for legacy NCMs
            self.cal_scores = self.ncm.score(X_cal, y_cal)
        score_time = time.time() - start_time

        cal_classes = np.unique(y_cal)
        print(f"Calibrated with {len(X_cal)} examples, {len(cal_classes)} cal classes, {len(self.classes)} candidate classes")
        print(f"Calibration scores (base) - min: {self.cal_scores.min():.4f}, max: {self.cal_scores.max():.4f}, mean: {self.cal_scores.mean():.4f}, std: {self.cal_scores.std():.4f}")
        print(f"Timing: fit={fit_time:.2f}s, score_cal={score_time:.2f}s")

    
    def predict(
        self,
        X_test: np.ndarray,
        return_p_values: bool = False,
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
        
        # Debug: track empty sets
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
                updated_scores = self.ncm.updated_calibration_scores_for(x_test, yc)

                n_greater = np.sum(updated_scores >= test_score)
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

            # Only return p_values if requested
            if return_p_values:
                all_p_values.append(p_vals)
        
        # Warn about empty sets
        if empty_count > 0 and verbose:
            print(f"\n⚠️  Warning: {empty_count}/{n_test} prediction sets are empty!")
            print(f"   Consider increasing alpha (current: {self.alpha}) or checking your data.")
        
        prediction_time = time.time() - start_time
        
        results = {
            'prediction_sets': prediction_sets,
            'set_sizes': np.array(set_sizes),
            'prediction_time': prediction_time
        }
        
        if return_p_values:
            results['p_values'] = all_p_values
        
        if verbose:
            print(f"\nPrediction time: {prediction_time:.2f}s ({prediction_time/n_test*1000:.1f}ms per sample)")

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

            # No same-class cal points => max nonconformity (matches CPU return 1e9 branch)
            no_same = (n_same_per_class == 0).unsqueeze(0).expand(B, K)
            test_scores = torch.where(no_same, torch.full_like(test_scores, 1e9), test_scores)
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
            print(f"\n⚠️  Warning: {empty_count}/{n_test} prediction sets are empty!")
            print(f"   Consider increasing alpha (current: {self.alpha}) or checking your data.")

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
                  f"({prediction_time/n_test*1000:.2f}ms per sample)")

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
        
        # Efficiency: average prediction set size
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
        
        # Empty set rate (should be 0 ideally)
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
            print("\n" + "="*50)
            print("CONFORMAL PREDICTION EVALUATION")
            print("="*50)
            print(f"Significance level (α):        {metrics['alpha']:.3f}")
            print(f"Target coverage:               {metrics['target_coverage']:.3f}")
            print(f"Actual coverage:               {metrics['coverage']:.3f}")
            print(f"Average set size:              {metrics['avg_set_size']:.3f}")
            print(f"Median set size:               {metrics['median_set_size']:.1f}")
            print(f"Singleton rate:                {metrics['singleton_rate']:.3f}")
            print(f"Singleton accuracy:            {metrics['singleton_accuracy']:.3f}")
            print(f"Empty set rate:                {metrics['empty_set_rate']:.3f}")
            print("="*50)
        
        return metrics


def compute_cp_scores(probs: np.ndarray, y_indices: np.ndarray,
                       score_fn: str = "APS",
                       k_reg: int = 5, lambda_raps: float = 0.01) -> np.ndarray:
    """
    Compute conformal prediction scores for a batch of examples.

    Args:
        probs: (n, K) softmax probabilities for each example over K classes
        y_indices: (n,) column index of the true class in probs for each example
        score_fn: "THR", "APS", or "RAPS"
        k_reg: RAPS regularization cutoff (only used when score_fn="RAPS")
        lambda_raps: RAPS penalty weight (only used when score_fn="RAPS")

    Returns:
        scores: (n,) nonconformity scores (higher = less conforming)
    """
    n, K = probs.shape
    scores = np.zeros(n)

    if score_fn == "THR":
        for i in range(n):
            scores[i] = 1.0 - probs[i, y_indices[i]]

    elif score_fn in ("APS", "RAPS"):
        for i in range(n):
            # Sort class probabilities descending
            sorted_idx = np.argsort(-probs[i])
            cumsum = 0.0
            rank = 0
            for j, idx in enumerate(sorted_idx):
                cumsum += probs[i, idx]
                rank = j + 1  # 1-based rank
                if idx == y_indices[i]:
                    break
            # APS score = cumulative probability up to and including y's position
            scores[i] = cumsum
            if score_fn == "RAPS":
                scores[i] += lambda_raps * max(0, rank - k_reg)
    else:
        raise ValueError(f"Unknown score_fn: {score_fn}. Use 'THR', 'APS', or 'RAPS'.")

    return scores


def compute_cp_sets(probs: np.ndarray, q_hat: float,
                    score_fn: str = "APS",
                    k_reg: int = 5, lambda_raps: float = 0.01) -> list:
    """
    Compute prediction sets for test examples given a calibrated quantile.

    Args:
        probs: (n, K) softmax probabilities
        q_hat: calibrated quantile threshold
        score_fn: "THR", "APS", or "RAPS"
        k_reg: RAPS regularization cutoff
        lambda_raps: RAPS penalty weight

    Returns:
        List of lists, each containing class indices included in the prediction set.
    """
    n, K = probs.shape
    prediction_sets = []

    if score_fn == "THR":
        threshold = 1.0 - q_hat
        for i in range(n):
            pred_set = [c for c in range(K) if probs[i, c] >= threshold]
            prediction_sets.append(pred_set)

    elif score_fn in ("APS", "RAPS"):
        for i in range(n):
            sorted_idx = np.argsort(-probs[i])
            cumsum = 0.0
            pred_set = []
            for j, idx in enumerate(sorted_idx):
                cumsum += probs[i, idx]
                rank = j + 1
                score = cumsum
                if score_fn == "RAPS":
                    score += lambda_raps * max(0, rank - k_reg)
                pred_set.append(int(idx))
                if score > q_hat:
                    break
            prediction_sets.append(pred_set)
    else:
        raise ValueError(f"Unknown score_fn: {score_fn}")

    return prediction_sets


class SoftmaxSplitCP:
    """
    Naive Softmax Split Conformal Prediction baseline.
    
    This is the standard "control" baseline for conformal prediction experiments.
    It uses a trained classifier's softmax probabilities as confidence scores.
    
    Protocol:
    1. Split data into D_train, D_calib, D_test
    2. Train a classifier (logistic regression) on D_train
    3. For each sample in D_calib: score s_i = 1 - p(y_true | x_i)
    4. Find quantile q at level (1 - alpha) * (n_cal + 1) / n_cal
    5. For test point x: prediction set = {y : 1 - p(y|x) <= q}
    
    This baseline represents what you get with a standard classifier + CP,
    without the sophisticated nonconformity measures used in Full CP.
    
    Reference: Shafer & Vovk (2008), Romano et al. (2020)
    """
    
    def __init__(self, alpha: float = 0.1, max_iter: int = 1000):
        """
        Args:
            alpha: Significance level (e.g., 0.1 for 90% coverage target)
            max_iter: Max iterations for logistic regression
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.classifier = None
        self.scaler = None
        self.q_hat = None  # Calibration quantile
        self.classes = None
        self.cal_scores = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the softmax classifier on training data.
        
        Args:
            X_train: Training features (n_train, d)
            y_train: Training labels (n_train,)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        self.classes = np.unique(y_train)
        
        # Standardize features for better convergence
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train logistic regression with softmax
        self.classifier = LogisticRegression(
            max_iter=self.max_iter,
            solver='lbfgs',
            random_state=42
        )
        self.classifier.fit(X_train_scaled, y_train)
        
        print(f"Trained softmax classifier on {len(X_train)} examples, {len(self.classes)} classes")
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                  all_classes: np.ndarray = None):
        """
        Calibrate using calibration set to find quantile threshold.

        Score: s_i = 1 - p(y_true | x_i) (higher = less confident = more nonconforming)

        Args:
            X_cal: Calibration features (n_cal, d)
            y_cal: Calibration labels (n_cal,)
            all_classes: Full label space for prediction sets.
                         If None, uses classifier's known classes.
        """
        if self.classifier is None:
            raise ValueError("Must call fit() before calibrate()")

        self._all_classes = all_classes  # store for predict()

        X_cal_scaled = self.scaler.transform(X_cal)

        # Get softmax probabilities
        probs = self.classifier.predict_proba(X_cal_scaled)

        # Compute nonconformity scores: s_i = 1 - p(y_true | x_i)
        # Need to map y_cal labels to classifier's class indices
        class_to_idx = {c: i for i, c in enumerate(self.classifier.classes_)}

        self.cal_scores = np.zeros(len(y_cal))
        for i, (prob, y_true) in enumerate(zip(probs, y_cal)):
            if y_true in class_to_idx:
                self.cal_scores[i] = 1.0 - prob[class_to_idx[y_true]]
            else:
                # Unknown class - assign max nonconformity
                self.cal_scores[i] = 1.0

        # Compute quantile at level (1 - alpha) * (n + 1) / n
        # This is the finite-sample correction for split CP
        n_cal = len(y_cal)
        quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        quantile_level = min(quantile_level, 1.0)  # Cap at 1.0

        self.q_hat = np.quantile(self.cal_scores, quantile_level)

        print(f"Calibrated on {n_cal} examples")
        print(f"Calibration scores - min: {self.cal_scores.min():.4f}, max: {self.cal_scores.max():.4f}, "
              f"mean: {self.cal_scores.mean():.4f}, std: {self.cal_scores.std():.4f}")
        print(f"Quantile threshold q_hat: {self.q_hat:.4f} (at level {quantile_level:.4f})")

    def predict(self, X_test: np.ndarray, return_p_values: bool = False) -> Dict:
        """
        Compute prediction sets for test examples.

        Prediction set: {y : 1 - p(y|x) <= q_hat}
        Equivalently: {y : p(y|x) >= 1 - q_hat}

        Iterates over the full label space (all_classes) if provided during
        calibrate(). Unseen classes get probability 0 and are only included
        if q_hat >= 1.0.

        Args:
            X_test: Test features (n_test, d)
            return_p_values: If True, return softmax probabilities as pseudo p-values

        Returns:
            Dict with 'prediction_sets' and optionally 'p_values'
        """
        if self.q_hat is None:
            raise ValueError("Must call calibrate() before predict()")

        X_test_scaled = self.scaler.transform(X_test)
        probs = self.classifier.predict_proba(X_test_scaled)

        n_test = len(X_test)
        prediction_sets = []

        # Threshold for inclusion: p(y|x) >= 1 - q_hat
        prob_threshold = 1.0 - self.q_hat

        # Build class -> prob-column-index map
        clf_class_to_idx = {c: i for i, c in enumerate(self.classifier.classes_)}

        # Determine full candidate label space
        if self._all_classes is not None:
            candidate_classes = np.unique(self._all_classes)
        else:
            candidate_classes = self.classifier.classes_

        for i in range(n_test):
            pred_set = []
            for c in candidate_classes:
                if c in clf_class_to_idx:
                    p = probs[i, clf_class_to_idx[c]]
                else:
                    p = 0.0  # unseen class gets zero probability
                if p >= prob_threshold:
                    pred_set.append(int(c))
            prediction_sets.append(pred_set)

        result = {'prediction_sets': prediction_sets}

        if return_p_values:
            # Return softmax probs as pseudo p-values (not true p-values, just for comparison)
            result['p_values'] = probs

        return result
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True) -> Dict:
        """
        Evaluate prediction sets on test data.
        
        Args:
            X_test: Test features
            y_test: True test labels
            verbose: Print results
            
        Returns:
            Dict with coverage, set sizes, and other metrics
        """
        predictions = self.predict(X_test)
        pred_sets = predictions['prediction_sets']
        
        n_test = len(y_test)
        
        # Coverage: fraction of test examples where true label is in prediction set
        covered = sum(1 for i, ps in enumerate(pred_sets) if y_test[i] in ps)
        coverage = covered / n_test
        
        # Set sizes
        set_sizes = [len(ps) for ps in pred_sets]
        avg_set_size = np.mean(set_sizes)
        median_set_size = np.median(set_sizes)
        
        # Additional metrics
        singleton_count = sum(1 for ps in pred_sets if len(ps) == 1)
        singleton_rate = singleton_count / n_test
        
        # Singleton accuracy: among singletons, how many are correct
        singleton_correct = sum(1 for i, ps in enumerate(pred_sets) 
                               if len(ps) == 1 and y_test[i] in ps)
        singleton_accuracy = singleton_correct / singleton_count if singleton_count > 0 else 0.0
        
        # Empty set rate
        empty_count = sum(1 for ps in pred_sets if len(ps) == 0)
        empty_set_rate = empty_count / n_test
        
        # Classifier accuracy (top-1)
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.classifier.predict(X_test_scaled)
        classifier_accuracy = np.mean(y_pred == y_test)
        
        metrics = {
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'median_set_size': median_set_size,
            'singleton_rate': singleton_rate,
            'singleton_accuracy': singleton_accuracy,
            'empty_set_rate': empty_set_rate,
            'classifier_accuracy': classifier_accuracy,
            'set_sizes': set_sizes,
        }
        
        if verbose:
            print("\n" + "="*50)
            print("SOFTMAX SPLIT CP EVALUATION")
            print("="*50)
            print(f"Test examples:                 {n_test}")
            print(f"Target coverage (1-α):         {1-self.alpha:.3f}")
            print(f"Achieved coverage:             {coverage:.3f}")
            print(f"Average set size:              {avg_set_size:.2f}")
            print(f"Median set size:               {median_set_size:.1f}")
            print(f"Singleton rate:                {singleton_rate:.3f}")
            print(f"Singleton accuracy:            {singleton_accuracy:.3f}")
            print(f"Empty set rate:                {empty_set_rate:.3f}")
            print(f"Classifier accuracy (top-1):   {classifier_accuracy:.3f}")
            print("="*50)
        
        return metrics


class SemiCP:
    """
    Semi-supervised Conformal Prediction (SemiCP).

    Augments Split CP calibration scores with bias-corrected pseudo-scores
    from unlabeled data using Nearest-Neighbor Matching (NNM).

    When X_unlabeled=None during calibrate(), this reduces to standard Split CP
    with the chosen score function (THR/APS/RAPS).

    Algorithm (Zhou et al., arXiv 2505.21147):
    1. Train classifier on D_train, score D_cal -> labeled scores {s_i}
    2. For each unlabeled point x_u:
       a. Pseudo-label: y_hat = argmax f(x_u)
       b. Pseudo-score: S(x_u, y_hat)
       c. NNM: find j* = argmin_j |S(x_u, y_hat) - S(x_j, y_hat_j)| among cal
       d. Bias-correct: s_u = S(x_u, y_hat) + S(x_j*, y_j*) - S(x_j*, y_hat_j*)
    3. Merge all_scores = labeled_scores + nnm_scores
    4. Quantile: q_hat at level ceil((n+N+1)(1-alpha)) / (n+N)

    Reference: Zhou et al. (2025), "Semi-supervised Conformal Prediction"
    """

    def __init__(self, alpha: float = 0.1, score_fn: str = "APS",
                 max_iter: int = 1000, k_reg: int = 5, lambda_raps: float = 0.01):
        self.alpha = alpha
        self.score_fn = score_fn
        self.max_iter = max_iter
        self.k_reg = k_reg
        self.lambda_raps = lambda_raps
        self.classifier = None
        self.scaler = None
        self.q_hat = None
        self._all_classes = None
        self._n_labeled = 0
        self._n_unlabeled = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train logistic regression classifier on training data."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.classifier = LogisticRegression(
            max_iter=self.max_iter, solver='lbfgs', random_state=42
        )
        self.classifier.fit(X_scaled, y_train)
        self._clf_classes = self.classifier.classes_
        self._class_to_idx = {c: i for i, c in enumerate(self._clf_classes)}

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                  X_unlabeled: np.ndarray = None, all_classes: np.ndarray = None):
        """
        Calibrate with labeled data and optionally augment with unlabeled data.

        Args:
            X_cal: Calibration features (n_cal, d)
            y_cal: Calibration labels (n_cal,)
            X_unlabeled: Optional unlabeled features (n_u, d) for NNM augmentation
            all_classes: Full label space for prediction sets
        """
        if self.classifier is None:
            raise ValueError("Must call fit() before calibrate()")

        self._all_classes = all_classes

        X_cal_scaled = self.scaler.transform(X_cal)
        probs_cal = self.classifier.predict_proba(X_cal_scaled)

        # Map true labels to classifier column indices
        # Classes absent from training get score=1.0 (max nonconformity)
        known_mask = np.array([y in self._class_to_idx for y in y_cal])
        y_cal_idx = np.array([self._class_to_idx.get(y, 0) for y in y_cal])

        # Labeled calibration scores
        cal_scores = compute_cp_scores(
            probs_cal, y_cal_idx, self.score_fn, self.k_reg, self.lambda_raps)
        cal_scores[~known_mask] = 1.0  # max score for unseen classes
        self._n_labeled = len(cal_scores)

        # --- NNM augmentation with unlabeled data ---
        nnm_scores = np.array([])
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            X_u_scaled = self.scaler.transform(X_unlabeled)
            probs_u = self.classifier.predict_proba(X_u_scaled)

            # Pseudo-labels for unlabeled
            y_hat_u = np.argmax(probs_u, axis=1)  # column indices

            # Pseudo-scores for unlabeled: S(x_u, y_hat_u)
            pseudo_scores_u = compute_cp_scores(
                probs_u, y_hat_u, self.score_fn, self.k_reg, self.lambda_raps)

            # Predicted labels for cal: y_hat_j = argmax f(x_j)
            y_hat_cal = np.argmax(probs_cal, axis=1)  # column indices

            # Scores under predicted label for cal: S(x_j, y_hat_j)
            cal_scores_pred = compute_cp_scores(
                probs_cal, y_hat_cal, self.score_fn, self.k_reg, self.lambda_raps)

            # NNM: for each unlabeled point, find closest cal point by pseudo-score
            # Per Zhou et al.: j* = argmin_j |S(x_u, y_hat_u) - S(x_j, y_hat_j)|
            # Matching across ALL cal points (no class restriction)
            # Vectorized: (n_u, 1) vs (1, n_cal) -> (n_u, n_cal)
            diffs = np.abs(pseudo_scores_u[:, None] - cal_scores_pred[None, :])
            j_stars = np.argmin(diffs, axis=1)

            # Bias-correct: s_u = S(x_u, y_hat_u) + S(x_j*, y_j*) - S(x_j*, y_hat_j*)
            nnm_scores = pseudo_scores_u + cal_scores[j_stars] - cal_scores_pred[j_stars]
            self._n_unlabeled = len(nnm_scores)

        # Merge scores
        all_scores = np.concatenate([cal_scores, nnm_scores]) if len(nnm_scores) > 0 else cal_scores
        n_total = len(all_scores)

        # Quantile with finite-sample correction
        quantile_level = np.ceil((n_total + 1) * (1 - self.alpha)) / n_total
        quantile_level = min(quantile_level, 1.0)
        self.q_hat = np.quantile(all_scores, quantile_level)

        print(f"SemiCP calibrated: {self._n_labeled} labeled + {self._n_unlabeled} NNM "
              f"= {n_total} total scores, score_fn={self.score_fn}")
        print(f"  q_hat={self.q_hat:.4f} (at quantile level {quantile_level:.4f})")

    def predict(self, X_test: np.ndarray) -> Dict:
        """Compute prediction sets for test examples."""
        if self.q_hat is None:
            raise ValueError("Must call calibrate() before predict()")

        X_scaled = self.scaler.transform(X_test)
        probs = self.classifier.predict_proba(X_scaled)
        n_test, K = probs.shape

        # Determine candidate label space
        if self._all_classes is not None:
            candidate_classes = np.unique(self._all_classes)
        else:
            candidate_classes = self._clf_classes

        # Build mapping from candidate class to probs column
        clf_to_col = {c: i for i, c in enumerate(self._clf_classes)}

        # Compute prediction sets using the score function
        prediction_sets = []

        if self.score_fn == "THR":
            threshold = 1.0 - self.q_hat
            for i in range(n_test):
                pred_set = []
                for c in candidate_classes:
                    p = probs[i, clf_to_col[c]] if c in clf_to_col else 0.0
                    if p >= threshold:
                        pred_set.append(int(c))
                prediction_sets.append(pred_set)

        elif self.score_fn in ("APS", "RAPS"):
            for i in range(n_test):
                # For each candidate label, compute its score and include if <= q_hat
                # More efficient: walk sorted probs and include until score > q_hat
                sorted_col = np.argsort(-probs[i])
                cumsum = 0.0
                pred_set = []
                for j, col_idx in enumerate(sorted_col):
                    # Map column back to class label
                    cls_label = self._clf_classes[col_idx]
                    if self._all_classes is not None and cls_label not in candidate_classes:
                        continue
                    cumsum += probs[i, col_idx]
                    rank = j + 1
                    score = cumsum
                    if self.score_fn == "RAPS":
                        score += self.lambda_raps * max(0, rank - self.k_reg)
                    pred_set.append(int(cls_label))
                    if score > self.q_hat:
                        break
                # Also include candidate classes not in classifier
                if self._all_classes is not None:
                    for c in candidate_classes:
                        if c not in clf_to_col and self.q_hat >= 1.0:
                            pred_set.append(int(c))
                prediction_sets.append(pred_set)

        return {'prediction_sets': prediction_sets}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 verbose: bool = True) -> Dict:
        """Evaluate prediction sets on test data."""
        predictions = self.predict(X_test)
        pred_sets = predictions['prediction_sets']
        n_test = len(y_test)

        covered = sum(1 for i, ps in enumerate(pred_sets) if y_test[i] in ps)
        coverage = covered / n_test

        set_sizes = [len(ps) for ps in pred_sets]
        avg_set_size = np.mean(set_sizes)
        median_set_size = np.median(set_sizes)

        singleton_count = sum(1 for ps in pred_sets if len(ps) == 1)
        singleton_rate = singleton_count / n_test
        singleton_correct = sum(1 for i, ps in enumerate(pred_sets)
                                if len(ps) == 1 and y_test[i] in ps)
        singleton_accuracy = singleton_correct / singleton_count if singleton_count > 0 else 0.0

        empty_count = sum(1 for ps in pred_sets if len(ps) == 0)
        empty_set_rate = empty_count / n_test

        X_scaled = self.scaler.transform(X_test)
        y_pred = self.classifier.predict(X_scaled)
        classifier_accuracy = np.mean(y_pred == y_test)

        metrics = {
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'median_set_size': median_set_size,
            'singleton_rate': singleton_rate,
            'singleton_accuracy': singleton_accuracy,
            'empty_set_rate': empty_set_rate,
            'classifier_accuracy': classifier_accuracy,
            'set_sizes': set_sizes,
            'n_labeled_cal': self._n_labeled,
            'n_nnm_scores': self._n_unlabeled,
        }

        if verbose:
            print("\n" + "=" * 50)
            print(f"SemiCP EVALUATION (score_fn={self.score_fn})")
            print("=" * 50)
            print(f"Test examples:                 {n_test}")
            print(f"Target coverage (1-alpha):     {1 - self.alpha:.3f}")
            print(f"Achieved coverage:             {coverage:.3f}")
            print(f"Average set size:              {avg_set_size:.2f}")
            print(f"Median set size:               {median_set_size:.1f}")
            print(f"Singleton rate:                {singleton_rate:.3f}")
            print(f"Empty set rate:                {empty_set_rate:.3f}")
            print(f"Classifier accuracy (top-1):   {classifier_accuracy:.3f}")
            print(f"Labeled cal scores:            {self._n_labeled}")
            print(f"NNM augmented scores:          {self._n_unlabeled}")
            print("=" * 50)

        return metrics


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

        prediction_sets = []
        set_sizes = []
        all_p_values = [] if return_p_values else None
        empty_count = 0

        iterator = tqdm(range(n_test), desc="CV+") if verbose else range(n_test)

        for i in iterator:
            x_test = X_test[i]
            pred_set = []
            p_vals = {}

            # Compute test score under each fold's NCM (shared across labels? no - depends on y)
            for y_candidate in self.classes:
                # K test scores, one per fold NCM
                n_folds = self._actual_folds
                test_scores_per_fold = np.zeros(n_folds)
                for k in range(n_folds):
                    test_scores_per_fold[k] = self.fold_ncms[k].score_x_cv(
                        x_test, int(y_candidate)
                    )

                # Vectorised CV+ comparison:
                # For each cal point i in fold k(i), compare R_i >= R_test^{k(i)}
                test_score_per_cal = test_scores_per_fold[self.fold_assignments]
                n_greater = np.sum(self.cal_scores >= test_score_per_cal)

                p_value = (n_greater + 1) / (n_cal + 1)
                p_vals[int(y_candidate)] = p_value

                if p_value > self.alpha:
                    pred_set.append(int(y_candidate))

            if len(pred_set) == 0:
                empty_count += 1
                if empty_count <= 3 and verbose:
                    best_p = max(p_vals.values())
                    best_class = max(p_vals, key=p_vals.get)
                    print(f"\nWarning: Empty prediction set for test example {i}")
                    print(f"  Best p-value: {best_p:.4f} (class {best_class}), alpha: {self.alpha}")

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
            'prediction_time': prediction_time
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
               reg: float = 1e-4) -> NonconformityMeasure:
    """
    Factory function to create NCM instances.

    Args:
        ncm_type: One of 'softmax', 'mahal_nn_ratio', 'whitened_geodesic',
                  'geodesic_topk_mean', 'geodesic_topk_asym'
        k: Number of neighbors for top-k NCMs
        reg: Variance regularisation for Mahalanobis NCMs
    """
    if ncm_type == "softmax":
        return SoftmaxNonconformity()
    elif ncm_type == "mahal_nn_ratio":
        return MahalNNRatio(reg=reg)
    elif ncm_type == "whitened_geodesic":
        return WhitenedGeodesicNNRatio(reg=reg)
    elif ncm_type == "geodesic_topk_mean":
        # Symmetric: topk on both same and other
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=True, topk_other=True)
    elif ncm_type == "geodesic_topk_asym":
        # Asymmetric: 1-NN for same (tight numerator), topk for other (Trojan Horse fix)
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=False, topk_other=True)
    elif ncm_type == "unwhitened_topk_mean":
        # Ablation: symmetric topk WITHOUT whitening
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=True, topk_other=True, whiten=False)
    elif ncm_type == "unwhitened_topk_asym":
        # Ablation: asymmetric topk WITHOUT whitening
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=False, topk_other=True, whiten=False)
    elif ncm_type == "geodesic_topk":
        # Numerator-only: mean of top-k same-class geodesic distances, no ratio
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=True, numerator_only=True)
    elif ncm_type == "geodesic_1nn":
        # Numerator-only: 1-NN same-class geodesic distance, no ratio
        return GeodesicTopKMeanNCM(reg=reg, k=k if k != 5 else None,
                                   topk_same=False, numerator_only=True)
    elif ncm_type == "rbf_density":
        # Gaussian-kernel density NCM. Bandwidth auto from cal median heuristic.
        return RBFDensityNCM()
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
    
    # Generate random permutation
    indices = np.random.permutation(n)
    
    # Calculate split points
    n_train = int(n * train_ratio)
    n_cal = int(n * cal_ratio)
    
    train_idx = indices[:n_train]
    cal_idx = indices[n_train:n_train + n_cal]
    test_idx = indices[n_train + n_cal:]
    
    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_cal = embeddings[cal_idx]
    y_cal = labels[cal_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]
    
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
    
    # Generate random permutation
    indices = np.random.permutation(n)
    
    # Calculate split point
    n_cal = int(n * cal_ratio)
    
    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]
    
    X_cal = embeddings[cal_idx]
    y_cal = labels[cal_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]
    
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
