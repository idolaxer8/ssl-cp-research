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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import logsumexp as scipy_logsumexp



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


class KNNNonconformity(NonconformityMeasure):
    """
    k-Nearest Neighbors nonconformity measure (Cherubin et al. 2021).

    Nonconformity score = min distance to own class / min distance to other class

    Higher score = far from own class and/or close to other classes = more nonconforming.
    This ratio-based approach is more discriminative than distance alone.
    """
    
    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        self.k = k
        self.metric = metric
        self.X_cal = None
        self.y_cal = None
        self.classes = None
        # index helper
        self.class_indices = {}
        # cached baseline calibration scores (alpha0)
        self.alpha0 = None
        
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Store calibration data and build per-class index mapping."""
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(y_cal)
        # build per-class indices
        self.class_indices = {cls: np.where(y_cal == cls)[0] for cls in self.classes}
        # compute baseline calibration scores (alpha0)
        self.alpha0 = self.get_calibration_scores()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity score using min-distance ratio:
        A((x,y)) = min_{i: yi=y} d(x, xi) / min_{i: yi!=y} d(x, xi)
        """
        eps = 1e-8
        scores = np.zeros(len(X))
        for i, (x, label) in enumerate(zip(X, y)):
            # distances to same and other classes
            same_idx = self.class_indices.get(label, np.array([], dtype=int))
            other_idx = np.where(self.y_cal != label)[0]
            if len(same_idx) == 0 or len(other_idx) == 0:
                scores[i] = np.inf
                continue
            d_same = euclidean_distances([x], self.X_cal[same_idx]).flatten()
            d_other = euclidean_distances([x], self.X_cal[other_idx]).flatten()
            scores[i] = float(np.min(d_same)) / max(float(np.min(d_other)), eps)
        return scores

    def get_calibration_scores(self) -> np.ndarray:
        """
        Baseline calibration scores alpha0 in original order.
        For each calibration xi with label yi:
        alpha0_i = min_{j: yj=yi, j!=i} d(xi, xj) / min_{j: yj!=yi} d(xi, xj)
        """
        eps = 1e-8
        if self.X_cal is None:
            raise ValueError("Must call fit() first")
        n_cal = len(self.X_cal)
        alpha0 = np.zeros(n_cal, dtype=float)
        for i in range(n_cal):
            yi = self.y_cal[i]
            same_idx = self.class_indices.get(yi, np.array([], dtype=int))
            # exclude self from same-class set
            same_idx = same_idx[same_idx != i]
            other_idx = np.where(self.y_cal != yi)[0]
            if len(same_idx) == 0 or len(other_idx) == 0:
                alpha0[i] = np.inf
                continue
            d_same = euclidean_distances(self.X_cal[i:i+1], self.X_cal[same_idx]).flatten()
            d_other = euclidean_distances(self.X_cal[i:i+1], self.X_cal[other_idx]).flatten()
            alpha0[i] = float(np.min(d_same)) / max(float(np.min(d_other)), eps)
        return alpha0

    def score_x(self, x: np.ndarray, y: int) -> float:
        """Score for a single test (x,y) using min-distance ratio."""
        eps = 1e-8
        same_idx = self.class_indices.get(y, np.array([], dtype=int))
        other_idx = np.where(self.y_cal != y)[0]
        if len(same_idx) == 0 or len(other_idx) == 0:
            return float('inf')
        d_same = euclidean_distances([x], self.X_cal[same_idx]).flatten()
        d_other = euclidean_distances([x], self.X_cal[other_idx]).flatten()
        return float(np.min(d_same)) / max(float(np.min(d_other)), eps)

    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        """
        Naive Full CP update: recompute alpha_i' for all calibration points assuming (x,y) is added.
        No optimizations; we recompute min distances directly.
        """
        eps = 1e-8
        if self.X_cal is None:
            raise ValueError("Must call fit() first")
        n_cal = len(self.X_cal)
        updated = np.zeros(n_cal, dtype=float)
        # Precompute distances from x to all calibration points
        d_x_to_all = euclidean_distances([x], self.X_cal).flatten()
        for i in range(n_cal):
            yi = self.y_cal[i]
            same_idx = self.class_indices.get(yi, np.array([], dtype=int))
            # exclude self from same-class set
            same_idx = same_idx[same_idx != i]
            other_idx = np.where(self.y_cal != yi)[0]
            # base sets
            if len(other_idx) == 0:
                updated[i] = np.inf
                continue
            # min same, possibly influenced by (x,y) if y==yi
            if len(same_idx) == 0:
                min_same = d_x_to_all[i] if y == yi else float('inf')
            else:
                d_same = euclidean_distances(self.X_cal[i:i+1], self.X_cal[same_idx]).flatten()
                min_same = float(np.min(d_same))
                if y == yi:
                    min_same = min(min_same, d_x_to_all[i])
            # min other, possibly influenced by (x,y) if y!=yi
            d_other = euclidean_distances(self.X_cal[i:i+1], self.X_cal[other_idx]).flatten()
            min_other = float(np.min(d_other))
            if y != yi:
                min_other = min(min_other, d_x_to_all[i])
            updated[i] = min_same / max(min_other, eps)
        return updated


class SimplifiedKNNNonconformity(NonconformityMeasure):
    """
    Simplified k-NN nonconformity from Cherubin et al. 2021 (Fast version).
    
    Optimization: Precompute k nearest distances for all calibration points.
    Nonconformity = sum of distances to k nearest neighbors of same class.
    
    This is much faster than regular k-NN because:
    - Distances are precomputed during fit()
    - Only need to compute distances from test point to calibration points
    """
    
    def __init__(self, k: int = 5):
        self.k = k
        # calibration data
        self.X_cal = None         # (n_cal, d)
        self.y_cal = None         # (n_cal,)
        self.classes = None
        # per-class arrays
        self.class_indices = {}   # class -> array of global calibration indices
        self.distances = {}       # class -> (n_class, k_actual) distances (sorted asc)
        self.temporary_scores = {}# class -> (n_class,) sum of k distances
        self.alpha0 = None        # (n_cal,) baseline calibration scores aligned with X_cal order
      
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(y_cal)
        n_cal = len(X_cal)

        # build per-class data and store index mapping
        for cls in self.classes:
            idx = np.where(y_cal == cls)[0]           # global indices in calibration array
            self.class_indices[cls] = idx
            X_cls = X_cal[idx]
            if len(X_cls) == 0:
                raise ValueError(f"Label {cls} not present")

            # pairwise distances within class
            D = euclidean_distances(X_cls, X_cls)
            D.sort(axis=1)
            # exclude self (first column = 0)
            k_actual = min(self.k, max(0, len(X_cls) - 1))
            if k_actual == 0:
                # no neighbors available -> distances array is empty (shape n_class x 0)
                self.distances[cls] = np.zeros((len(X_cls), 0))
                self.temporary_scores[cls] = np.zeros(len(X_cls))
            else:
                # keep k_actual nearest (excluding self)
                self.distances[cls] = D[:, 1:k_actual+1]    # shape (n_class, k_actual)
                self.temporary_scores[cls] = np.sum(self.distances[cls], axis=1)

        # build alpha0 aligned with original calibration order
        alpha0 = np.zeros(n_cal, dtype=float)
        for cls in self.classes:
            idx = self.class_indices[cls]
            alpha0[idx] = self.temporary_scores[cls]
        self.alpha0 = alpha0
    
    def get_calibration_scores(self) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()
    
    def score_x(self, x: np.ndarray, y: int) -> float:
        """Score for a single test (x,y) — only test score (fast)."""
        if y not in self.class_indices:
            return np.inf
        idx = self.class_indices[y]
        X_same = self.X_cal[idx]
        dists = euclidean_distances([x], X_same).flatten()
        k_actual = min(self.k, len(dists))
        if k_actual == 0:
            return 0.0
        # sum of k smallest distances
        ksmall = np.partition(dists, k_actual-1)[:k_actual]
        return float(np.sum(ksmall))

    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        """
        Return updated calibration scores (length n_cal) after hypothetically adding (x,y).
        Only calibration examples of class `y` can change.
        """
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        n_cal = len(self.X_cal)
        updated = self.alpha0.copy()   # baseline
            
        if y not in self.class_indices:
            # unknown label: no changes to calibration scores
            return updated

        # compute distances from x to every calibration example of class y
        idx = self.class_indices[y]
        X_same = self.X_cal[idx]
        dists = euclidean_distances([x], X_same).flatten()

        k_actual = min(self.k, max(0, X_same.shape[0] - 1))
        if k_actual == 0:
            # no k-neighbors to update for this class
            return updated

        # per-class stored arrays
        kdist_class = self.distances[y]        # shape (n_class, k_actual)
        tmp_scores = self.temporary_scores[y]  # shape (n_class,)

        # iterate through same-class calibration examples in their per-class order
        for local_i, global_i in enumerate(idx):
            dist_x = dists[local_i]
            # if kdist_class may have fewer than k entries (k_actual < self.k)
            if kdist_class.shape[1] == 0:
                # nothing to do
                continue
            kth = kdist_class[local_i, -1]   # largest of the k stored distances
            if dist_x < kth:
                # replace kth with dist_x
                updated[global_i] = tmp_scores[local_i] - kth + dist_x
            else:
                # unchanged (already copied from alpha0)
                pass

        return updated


class CentroidNonconformity(NonconformityMeasure):
    """
    Centroid Distance (Prototypical) nonconformity measure.
    
    Nonconformity score = distance from point to its class centroid.
    α_i = d(x_i, μ_y) where μ_y is the mean of all calibration examples with label y.
    
    This is a global measure (vs local k-NN) and is computationally efficient.
    """
    
    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
        self.X_cal = None
        self.y_cal = None
        self.classes = None
        self.class_indices = {}
        self.centroids = {}      # class -> centroid vector
        self.class_counts = {}   # class -> number of examples
        self.alpha0 = None
    
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Compute class centroids from calibration data."""
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(y_cal)
        
        # Compute centroids for each class
        for cls in self.classes:
            idx = np.where(y_cal == cls)[0]
            self.class_indices[cls] = idx
            self.class_counts[cls] = len(idx)
            self.centroids[cls] = X_cal[idx].mean(axis=0)
        
        # Compute baseline calibration scores
        self.alpha0 = self.get_calibration_scores()
    
    def get_calibration_scores(self) -> np.ndarray:
        """Compute α_i = d(x_i, μ_{y_i}) for all calibration points."""
        if self.X_cal is None:
            raise ValueError("Must call fit() first")
        
        n_cal = len(self.X_cal)
        alpha0 = np.zeros(n_cal, dtype=float)
        
        for i in range(n_cal):
            yi = self.y_cal[i]
            centroid = self.centroids[yi]
            alpha0[i] = np.linalg.norm(self.X_cal[i] - centroid)
        
        return alpha0
    
    def score_x(self, x: np.ndarray, y: int) -> float:
        """Score for a single test point: d(x, μ_y)."""
        if y not in self.centroids:
            return float('inf')
        return float(np.linalg.norm(x - self.centroids[y]))
    
    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        """
        Compute updated calibration scores after hypothetically adding (x, y).
        
        When (x, y) is added:
        - Centroid μ'_y = (n_y * μ_y + x) / (n_y + 1)
        - All calibration points of class y get new scores with μ'_y
        - Other classes unchanged
        """
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        
        updated = self.alpha0.copy()
        
        if y not in self.centroids:
            return updated
        
        # Compute new centroid for class y
        n_y = self.class_counts[y]
        old_centroid = self.centroids[y]
        new_centroid = (n_y * old_centroid + x) / (n_y + 1)
        
        # Update scores for all calibration points of class y
        idx = self.class_indices[y]
        for i in idx:
            updated[i] = np.linalg.norm(self.X_cal[i] - new_centroid)
        
        return updated


class RelativeCentroidNonconformity(NonconformityMeasure):
    """
    Relative Centroid Distance nonconformity measure.
    
    More robust to OOD data than plain centroid distance because it asks:
    "Is this point closer to my class than to the nearest enemy class?"
    
    Nonconformity score:
        S(x, y) = d(x, μ_y) - min_{y' ≠ y} d(x, μ_{y'})
    
    Interpretation (higher = more nonconforming, as required by CP):
    - Positive score: d_own > d_other → farther from own class than nearest enemy (nonconforming)
    - Zero: equidistant to own class and nearest enemy class
    - Negative score: d_own < d_other → closer to own class than nearest enemy (conforming)
    
    This is more discriminative than absolute centroid distance for OOD detection.
    """
    
    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
        self.X_cal = None
        self.y_cal = None
        self.classes = None
        self.class_indices = {}
        self.centroids = {}      # class -> centroid vector
        self.class_counts = {}   # class -> number of examples
        self.alpha0 = None
        # Precomputed for vectorized operations
        self._centroid_matrix = None  # (n_classes, d)
        self._class_list = None       # list of class labels in order
    
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Compute class centroids from calibration data."""
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(y_cal)
        
        # Compute centroids for each class
        for cls in self.classes:
            idx = np.where(y_cal == cls)[0]
            self.class_indices[cls] = idx
            self.class_counts[cls] = len(idx)
            self.centroids[cls] = X_cal[idx].mean(axis=0)
        
        # Build centroid matrix for vectorized operations
        self._class_list = list(self.classes)
        self._centroid_matrix = np.array([self.centroids[c] for c in self._class_list])
        
        # Compute baseline calibration scores
        self.alpha0 = self._compute_scores_vectorized(X_cal, y_cal, self._centroid_matrix)
    
    def _compute_scores_vectorized(self, X: np.ndarray, y: np.ndarray, centroid_matrix: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of S(x, y) = d(x, μ_y) - min_{y' ≠ y} d(x, μ_{y'}) for all points.
        
        Args:
            X: (n, d) feature matrix
            y: (n,) labels
            centroid_matrix: (n_classes, d) centroid matrix
        
        Returns:
            (n,) scores
        """
        n = len(X)
        n_classes = len(self._class_list)
        
        # Compute all pairwise distances: (n, n_classes)
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c.T
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
        C_sq = np.sum(centroid_matrix ** 2, axis=1, keepdims=True).T  # (1, n_classes)
        dists_sq = X_sq + C_sq - 2 * X @ centroid_matrix.T  # (n, n_classes)
        dists_sq = np.maximum(dists_sq, 0)  # numerical safety
        dists = np.sqrt(dists_sq)  # (n, n_classes)
        
        # Map labels to indices in centroid_matrix
        class_to_idx = {c: i for i, c in enumerate(self._class_list)}
        y_idx = np.array([class_to_idx[yi] for yi in y])
        
        # d_own: distance to own class centroid
        d_own = dists[np.arange(n), y_idx]  # (n,)
        
        # d_other_min: min distance to any other class centroid
        # Set own-class distance to inf so it's not selected as min
        dists_masked = dists.copy()
        dists_masked[np.arange(n), y_idx] = np.inf
        d_other_min = np.min(dists_masked, axis=1)  # (n,)
        
        return d_own - d_other_min
    
    def get_calibration_scores(self) -> np.ndarray:
        """Return precomputed calibration scores."""
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()
    
    def score_x(self, x: np.ndarray, y: int) -> float:
        """Score for a single test point."""
        if y not in self.centroids:
            return float('inf')
        
        d_own = np.linalg.norm(x - self.centroids[y])
        
        # Find minimum distance to other class centroids
        d_others = [np.linalg.norm(x - c) for cls, c in self.centroids.items() if cls != y]
        
        if len(d_others) == 0:
            return d_own  # No other classes, fall back to absolute distance
        
        return d_own - min(d_others)
    
    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        """
        Compute updated calibration scores after hypothetically adding (x, y).
        
        Vectorized implementation for Full CP efficiency.
        
        When (x, y) is added:
        - Centroid μ'_y = (n_y * μ_y + x) / (n_y + 1)
        - ALL calibration points need score updates because:
          - Points of class y: their d(x_i, μ_y) changes
          - Points of other classes: their min distance to "enemy" classes may change
            if class y was their nearest enemy
        """
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        
        if y not in self.centroids:
            return self.alpha0.copy()
        
        # Compute new centroid for class y
        n_y = self.class_counts[y]
        old_centroid = self.centroids[y]
        new_centroid = (n_y * old_centroid + x) / (n_y + 1)
        
        # Build updated centroid matrix (only one centroid changes)
        y_idx = self._class_list.index(y)
        updated_centroid_matrix = self._centroid_matrix.copy()
        updated_centroid_matrix[y_idx] = new_centroid
        
        # Vectorized recomputation for ALL calibration points
        return self._compute_scores_vectorized(self.X_cal, self.y_cal, updated_centroid_matrix)


class RidgeRegressionNonconformity(NonconformityMeasure):
    """
    Ridge Regression (LS-SVM) in Primal Form optimized for Full CP.
    
    Based on Cherubin et al. 2021 exact CP optimization, adapted for:
    1. Primal form: inverts (d×d) matrix instead of (N×N) - efficient for d << N
    2. Multi-class: One-vs-Rest with {-1, +1} encoding in a single matrix operation
    3. Sherman-Morrison: O(d²) instant updates for Full CP
    4. Fast re-scoring: O(N×d) instead of O(N×d×n_classes) for many-class problems
    
    Nonconformity score: s(x, y) = -f_y(x)
    where f_y(x) is the Ridge regression prediction for class y.
    
    Interpretation:
    - Negative score: model confidently predicts class y (conforming)
    - Positive score: model does NOT predict class y (nonconforming)
    
    Reference: https://arxiv.org/pdf/2102.03236
    
    Optimization for many classes (e.g., CIFAR-100):
    Instead of recomputing W_new and scoring all classes, we use:
        W_new = W + outer(u, delta)
        where delta = (y_vec - u @ XtY) / denom
    
    For re-scoring calibration points, we only compute the true-class prediction:
        score_i_new = alpha0_i - (x_i @ u) * delta[y_cal_indices[i]]
    
    This reduces complexity from O(N × d × n_classes) to O(N × d).
    """
    
    def __init__(self, lambda_reg: float = 1.0):
        """
        Args:
            lambda_reg: Regularization parameter (larger = more regularization)
        """
        self.lambda_reg = lambda_reg
        self.X_cal = None
        self.y_cal = None
        self.classes = None
        self.n_classes = None
        
        # Precomputed mappings
        self.class_to_idx = None
        self.y_cal_indices = None  # Indices of true classes for calibration points
        
        # Core matrices for Ridge Regression
        self.inv_cov = None   # (d, d) - Inverse of (X^T X + λI)
        self.XtY = None       # (d, n_classes) - Cross-correlation X^T Y
        
        # Cached calibration scores
        self.alpha0 = None
        
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Fit Ridge Regression on calibration set.
        
        Computes and caches:
        - Inverse covariance matrix (X^T X + λI)^{-1}
        - Cross-correlation X^T Y
        - Calibration scores
        """
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(y_cal)
        self.n_classes = len(self.classes)
        N, d = X_cal.shape
        
        # Precompute class mappings
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.y_cal_indices = np.array([self.class_to_idx[y] for y in y_cal])
        
        # 1. One-Hot Encode Targets with {-1, +1} scheme
        # Shape: (N, n_classes)
        Y_onehot = -1.0 * np.ones((N, self.n_classes))
        Y_onehot[np.arange(N), self.y_cal_indices] = 1.0
            
        # 2. Compute Covariance: A = X^T X + λI
        # Shape: (d, d)
        A = X_cal.T @ X_cal
        A[np.diag_indices_from(A)] += self.lambda_reg
        
        # 3. Compute Inverse: A^{-1} (expensive step, done once)
        self.inv_cov = np.linalg.inv(A)
        
        # 4. Compute Cross-Correlation: B = X^T Y
        self.XtY = X_cal.T @ Y_onehot
        
        # 5. Cache calibration scores using optimized method
        self.alpha0 = self._compute_calibration_scores_optimized()
    
    def _compute_calibration_scores_optimized(self) -> np.ndarray:
        """
        Compute scores for all calibration points efficiently.
        Only computes the prediction for the TRUE class of each point.
        
        Complexity: O(N × d) instead of O(N × d × n_classes)
        """
        # W = inv_cov @ XtY, shape (d, n_classes)
        W = self.inv_cov @ self.XtY
        
        # For each calibration point i, we only need W[:, y_cal_indices[i]]
        # Using advanced indexing: W[:, y_cal_indices] has shape (d, N)
        # Then element-wise multiply with X_cal.T and sum
        W_true_classes = W[:, self.y_cal_indices]  # (d, N)
        
        # true_class_preds[i] = X_cal[i] @ W[:, y_cal_indices[i]]
        # = sum_j X_cal[i, j] * W[j, y_cal_indices[i]]
        true_class_preds = np.sum(self.X_cal * W_true_classes.T, axis=1)  # (N,)
        
        return -true_class_preds
    
    def get_calibration_scores(self) -> np.ndarray:
        """Return cached calibration scores."""
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()
    
    def score_x(self, x: np.ndarray, y: int) -> float:
        """
        Score for test point (x, y) AFTER hypothetically adding it to calibration set.
        
        Optimized: Only computes the single column W_new[:, y] needed.
        
        Derivation:
            W_new = W + outer(u, delta), where delta = (y_vec - u @ XtY) / denom
            score = -x @ W_new[:, y] = -x @ W[:, y] - (x @ u) * delta[y]
        
        Since we don't cache W, we compute W[:, y] = inv_cov @ XtY[:, y] on the fly.
        Complexity: O(d²) instead of O(d² × n_classes)
        """
        if y not in self.class_to_idx:
            return float('inf')
        
        cls_idx = self.class_to_idx[y]
        
        # Sherman-Morrison quantities
        u = self.inv_cov @ x  # (d,)
        denom = 1.0 + x @ u
        
        if abs(denom) < 1e-10:
            # Fallback: use original model
            W_y = self.inv_cov @ self.XtY[:, cls_idx]
            return -float(x @ W_y)
        
        # Original weight for class y: W[:, y] = inv_cov @ XtY[:, y]
        W_y = self.inv_cov @ self.XtY[:, cls_idx]  # (d,)
        orig_pred = x @ W_y
        
        # Delta for class y
        # y_vec has +1 at cls_idx, -1 elsewhere
        # delta[cls_idx] = (1.0 - (u @ XtY)[cls_idx]) / denom
        u_dot_XtY_y = u @ self.XtY[:, cls_idx]  # scalar
        delta_y = (1.0 - u_dot_XtY_y) / denom
        
        # Updated prediction
        pred_new = orig_pred + (x @ u) * delta_y
        
        return -float(pred_new)

    def score_x_cv(self, x: np.ndarray, y: int) -> float:
        """
        Score for CV+/Jackknife+: use the existing model WITHOUT Sherman-Morrison update.

        Returns -f_y(x) = -(x @ W[:, y]) where W = inv_cov @ XtY.
        Complexity: O(d²)
        """
        if y not in self.class_to_idx:
            return float('inf')
        cls_idx = self.class_to_idx[y]
        W_y = self.inv_cov @ self.XtY[:, cls_idx]
        return -float(x @ W_y)

    def updated_calibration_scores_for(self, x: np.ndarray, y: int) -> np.ndarray:
        """
        Compute updated calibration scores after hypothetically adding (x, y).
        
        OPTIMIZED VERSION: O(N × d) instead of O(N × d × n_classes)
        
        Key insight: We only need the prediction for each point's TRUE class.
        Using Sherman-Morrison, the weight update is:
            W_new = W + outer(u, delta)
            where u = inv_cov @ x, delta = (y_vec - u @ XtY) / denom
        
        For calibration point i with true class c_i:
            score_i_new = -x_i @ W_new[:, c_i]
                        = -x_i @ W[:, c_i] - (x_i @ u) * delta[c_i]
                        = alpha0[i] - (x_i @ u) * delta[c_i]
        
        This avoids computing the full W_new matrix!
        """
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        
        if y not in self.class_to_idx:
            return np.inf * np.ones(len(self.X_cal))
        
        cls_idx = self.class_to_idx[y]
        
        # Sherman-Morrison quantities
        u = self.inv_cov @ x  # (d,)
        denom = 1.0 + x @ u
        
        if abs(denom) < 1e-10:
            # Numerical issue: return original scores
            return self.alpha0.copy()
        
        # Compute delta for ALL classes (needed because calibration points have various classes)
        # y_vec: +1 at cls_idx, -1 elsewhere
        y_target_vec = -1.0 * np.ones(self.n_classes)
        y_target_vec[cls_idx] = 1.0
        
        u_dot_XtY = u @ self.XtY  # (n_classes,) - O(d × n_classes)
        delta = (y_target_vec - u_dot_XtY) / denom  # (n_classes,)
        
        # Compute X_cal @ u: dot product of each calibration point with u
        dots = self.X_cal @ u  # (N,) - O(N × d)
        
        # Get the delta value for each calibration point's TRUE class
        delta_per_point = delta[self.y_cal_indices]  # (N,)
        
        # Updated scores: alpha0[i] - dots[i] * delta[y_cal_indices[i]]
        return self.alpha0 - dots * delta_per_point


class FastNNRatio(NonconformityMeasure):
    """
    Optimized Nearest Neighbor Ratio NCM using Lookup Tables.
    
    Designed to work well in the few-shot regime (as few as 1 example per class)
    by using a contrastive ratio approach.
    
    Nonconformity score:
        α_i = min_{j: y_j = y_i, j ≠ i} d(x_i, x_j) / min_{k: y_k ≠ y_i} d(x_i, x_k)
    
    Interpretation (higher = more nonconforming):
    - Ratio > 1: closer to enemy class than own class (nonconforming)
    - Ratio < 1: closer to own class than enemy class (conforming)
    - Ratio = 1: equidistant
    
    The ratio automatically handles density variations - if a region is sparse,
    both numerator and denominator grow, but the ratio remains meaningful.
    
    Optimization:
    - Precompute lookup tables for min distances during fit() - O(N²) once
    - Update scores in O(N) using vectorized comparisons against lookup tables
    - No neighbor re-sorting needed at test time
    """
    
    def __init__(self):
        # Calibration data
        self.X_cal = None
        self.y_cal = None

        # LOOKUP TABLES (The Optimization)
        # These store the current minimum distance for every sample
        self.lookup_same = None   # shape (n_cal,): dist to nearest neighbor of SAME class
        self.lookup_other = None  # shape (n_cal,): dist to nearest neighbor of DIFF class

        # Baseline alpha scores
        self.alpha0 = None

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Build the Lookup Tables using vectorized operations.

        Complexity: O(N²) - done once during calibration.
        """
        self.X_cal = X_cal
        self.y_cal = y_cal
        n_cal = len(X_cal)

        # 1. Compute full distance matrix once
        D = euclidean_distances(X_cal, X_cal)
        np.fill_diagonal(D, np.inf)  # Ignore self-matches

        # 2. Initialize Lookup Tables
        self.lookup_same = np.full(n_cal, np.inf)
        self.lookup_other = np.full(n_cal, np.inf)

        # 3. Fill Tables using vectorized operations per class
        classes = np.unique(y_cal)
        for c in classes:
            # Indices for this class and other classes
            idx_same = np.where(y_cal == c)[0]
            idx_other = np.where(y_cal != c)[0]

            if len(idx_same) > 0:
                # Update 'same' lookup: min distance to other samples of SAME class
                if len(idx_same) > 1:
                    # Use np.ix_ for efficient sub-matrix indexing
                    self.lookup_same[idx_same] = np.min(D[np.ix_(idx_same, idx_same)], axis=1)
                else:
                    # Singleton class: no same-class neighbor available
                    self.lookup_same[idx_same] = np.inf

                # Update 'other' lookup: min distance to samples of OTHER classes
                if len(idx_other) > 0:
                    self.lookup_other[idx_same] = np.min(D[np.ix_(idx_same, idx_other)], axis=1)

        # 4. Compute Baseline Scores (alpha0)
        self.alpha0 = self.lookup_same / (self.lookup_other + 1e-8)

        # Replace infs (singleton classes) with a high penalty score
        self.alpha0[np.isinf(self.alpha0)] = 1e9
    
    def get_calibration_scores(self) -> np.ndarray:
        """Return precomputed calibration scores."""
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()
    
    def score_x(self, x: np.ndarray, y: int, precomputed_dists: np.ndarray = None) -> float:
        """
        Calculate score for the test point x itself.
        
        Args:
            x: Test point features
            y: Candidate label
            precomputed_dists: Optional precomputed distances from x to all calibration points.
                               If provided, avoids redundant distance computation.
        
        Complexity: O(N), or O(1) masking if precomputed_dists provided
        """
        # Use precomputed distances if available, otherwise compute
        if precomputed_dists is not None:
            dists = precomputed_dists
        else:
            dists = euclidean_distances([x], self.X_cal).flatten()
        
        # 1. Find nearest SAME class neighbor in calibration set
        mask_same = (self.y_cal == y)
        if np.any(mask_same):
            dist_same = np.min(dists[mask_same])
        else:
            return 1e9  # Penalty: proposed class y has no examples

        # 2. Find nearest OTHER class neighbor
        mask_other = (self.y_cal != y)
        if np.any(mask_other):
            dist_other = np.min(dists[mask_other])
        else:
            dist_other = 1e-8  # Should not happen if >1 class

        return dist_same / (dist_other + 1e-8)
    
    def score_x_cv(self, x: np.ndarray, y: int) -> float:
        """Score for CV+/Jackknife+: ratio using current calibration (no hypothetical addition)."""
        return self.score_x(x, y)

    def updated_calibration_scores_for(self, x: np.ndarray, y: int, precomputed_dists: np.ndarray = None) -> np.ndarray:
        """
        Update scores of EXISTING calibration points after hypothetically adding (x, y).

        The Optimization:
        We do NOT recalculate nearest neighbors or re-sort.
        We simply compare dist(cal_i, x) against lookup_table[i].

        Args:
            x: Test point features
            y: Candidate label
            precomputed_dists: Optional precomputed distances from x to all calibration points.
                               If provided, avoids redundant distance computation.

        Complexity: O(N) - fully vectorized
        """
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")

        # Start with baseline scores
        updated_scores = self.alpha0.copy()

        # Use precomputed distances if available, otherwise compute
        if precomputed_dists is not None:
            dists_x = precomputed_dists
        else:
            dists_x = euclidean_distances([x], self.X_cal).flatten()

        # CASE A: Update points belonging to the SAME class (y)
        # For these points, x is a new candidate for "Same Class Neighbor"
        idx_same = np.where(self.y_cal == y)[0]
        if len(idx_same) > 0:
            new_same = np.minimum(self.lookup_same[idx_same], dists_x[idx_same])
            updated_scores[idx_same] = new_same / (self.lookup_other[idx_same] + 1e-8)

        # CASE B: Update points belonging to OTHER classes (!= y)
        # For these points, x is a new candidate for "Other Class Neighbor"
        idx_other = np.where(self.y_cal != y)[0]
        if len(idx_other) > 0:
            new_other = np.minimum(self.lookup_other[idx_other], dists_x[idx_other])
            updated_scores[idx_other] = self.lookup_same[idx_other] / (new_other + 1e-8)

        return updated_scores


class HypersphericalNNRatio(NonconformityMeasure):
    """
    Geodesic Nearest-Neighbor Ratio NCM for hyperspherical embeddings.

    SSL models (DINOv2, etc.) produce L2-normalized embeddings on the unit
    hypersphere S^{d-1}.  This NCM operates in the true angular / geodesic
    space instead of treating the manifold as Euclidean.

    Nonconformity score:
        alpha_i = arccos(max_same_sim_i) / (arccos(max_other_sim_i) + eps)

    where similarities are cosine similarities and arccos converts them to
    geodesic distances on the unit sphere.

    Key insight: The geodesic ratio arccos(s_same)/arccos(s_other) is NOT
    monotonically equivalent to the Euclidean ratio sqrt((1-s_same)/(1-s_other)).
    The arccos nonlinearity amplifies differences in high-similarity regions
    (near class centroids), making it more discriminative for tightly clustered
    SSL embeddings.

    Works with FullConformalPredictor out of the box: precomputed Euclidean
    distances are converted internally to cosine similarities via the exact
    identity  sim = 1 - d^2/2  (valid for unit vectors).
    """

    def __init__(self):
        self.X_cal = None
        self.y_cal = None

        # Lookup tables: max cosine similarity to same / other class
        self.lookup_same_sim = None   # (n_cal,)
        self.lookup_other_sim = None  # (n_cal,)

        self.alpha0 = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dists_to_sims(dists: np.ndarray) -> np.ndarray:
        """Convert Euclidean distances to cosine similarities (exact for unit vecs)."""
        return 1.0 - dists ** 2 / 2.0

    @staticmethod
    def _geodesic_ratio(same_sim: np.ndarray, other_sim: np.ndarray) -> np.ndarray:
        """Compute arccos(same) / (arccos(other) + eps)."""
        eps = 1e-8
        same_clipped = np.clip(same_sim, -1.0, 1.0)
        other_clipped = np.clip(other_sim, -1.0, 1.0)
        return np.arccos(same_clipped) / (np.arccos(other_clipped) + eps)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        self.X_cal = X_cal
        self.y_cal = y_cal
        n_cal = len(X_cal)

        # Cosine similarity matrix (assumes L2-normalised rows)
        S = X_cal @ X_cal.T
        np.fill_diagonal(S, -np.inf)  # exclude self

        self.lookup_same_sim = np.full(n_cal, -1.0)
        self.lookup_other_sim = np.full(n_cal, -1.0)

        classes = np.unique(y_cal)
        for c in classes:
            idx_same = np.where(y_cal == c)[0]
            idx_other = np.where(y_cal != c)[0]

            if len(idx_same) > 1:
                self.lookup_same_sim[idx_same] = np.max(
                    S[np.ix_(idx_same, idx_same)], axis=1
                )
            # else: singleton -> stays at -1.0 => arccos(-1)=pi (max penalty)

            if len(idx_other) > 0:
                self.lookup_other_sim[idx_same] = np.max(
                    S[np.ix_(idx_same, idx_other)], axis=1
                )

        self.alpha0 = self._geodesic_ratio(self.lookup_same_sim,
                                            self.lookup_other_sim)
        # Cap infinities from singleton classes
        self.alpha0[~np.isfinite(self.alpha0)] = 1e9

    # ------------------------------------------------------------------
    # calibration scores
    # ------------------------------------------------------------------
    def get_calibration_scores(self) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        return self.alpha0.copy()

    # ------------------------------------------------------------------
    # score a single test point
    # ------------------------------------------------------------------
    def score_x(self, x: np.ndarray, y: int,
                precomputed_dists: np.ndarray = None) -> float:
        if precomputed_dists is not None:
            sims = self._dists_to_sims(precomputed_dists)
        else:
            sims = (x @ self.X_cal.T)  # cosine similarity (unit vecs)

        mask_same = (self.y_cal == y)
        mask_other = (self.y_cal != y)

        if not np.any(mask_same):
            return 1e9

        max_same_sim = np.max(sims[mask_same])
        if not np.any(mask_other):
            max_other_sim = -1.0
        else:
            max_other_sim = np.max(sims[mask_other])

        return float(self._geodesic_ratio(
            np.array([max_same_sim]), np.array([max_other_sim])
        )[0])

    def score_x_cv(self, x: np.ndarray, y: int) -> float:
        """Score for CV+/Jackknife+: geodesic ratio using current calibration."""
        return self.score_x(x, y)

    # ------------------------------------------------------------------
    # updated calibration scores after hypothetical addition of (x, y)
    # ------------------------------------------------------------------
    def updated_calibration_scores_for(self, x: np.ndarray, y: int,
                                       precomputed_dists: np.ndarray = None) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")

        if precomputed_dists is not None:
            sims_x = self._dists_to_sims(precomputed_dists)
        else:
            sims_x = (x @ self.X_cal.T)

        # Start from lookup tables
        updated_same = self.lookup_same_sim.copy()
        updated_other = self.lookup_other_sim.copy()

        # Points of SAME class as y: x is a new same-class candidate
        idx_same = np.where(self.y_cal == y)[0]
        if len(idx_same) > 0:
            updated_same[idx_same] = np.maximum(
                self.lookup_same_sim[idx_same], sims_x[idx_same]
            )

        # Points of OTHER class: x is a new other-class candidate
        idx_other = np.where(self.y_cal != y)[0]
        if len(idx_other) > 0:
            updated_other[idx_other] = np.maximum(
                self.lookup_other_sim[idx_other], sims_x[idx_other]
            )

        scores = self._geodesic_ratio(updated_same, updated_other)
        scores[~np.isfinite(scores)] = 1e9
        return scores


class MahalNNRatio(NonconformityMeasure):
    """
    Point-to-Point Mahalanobis Nearest-Neighbor Ratio NCM.

    Identical in structure to FastNNRatio but replaces Euclidean distance with
    Mahalanobis distance using a pooled within-class diagonal covariance (LDA-style).

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

    Full CP safety (exchangeability):
    - No centroid is ever computed or shifted.
    - update(x*, y*) only adjusts lookup_same / lookup_other by comparing the new
      Mahalanobis distances against existing min-lookup tables — exactly like FastNNRatio.
    - Adding (x*, y*) changes distances to x* but NOT the whitening metric (pooled_var
      is fixed from fit()).  All n+1 scores share the same metric => exchangeable.
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

    def _compute_mahal_dists_from_x(self, x: np.ndarray,
                                     precomputed_dists=None) -> np.ndarray:
        """Mahalanobis distances from x to all cal points, with per-x caching.

        Uses x.ctypes.data (stable pointer into X_test's buffer) as cache key,
        so the O(n*d) computation runs once per test point across all y-candidates.
        Using id(precomputed_dists) is NOT safe — CPython can reuse the same memory
        address for different temporary arrays within the same predict() call.
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

        # 4. Build lookup tables exactly like FastNNRatio
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

    def score_x(self, x: np.ndarray, y: int,
                precomputed_dists: np.ndarray = None, **kwargs) -> float:
        dists = self._compute_mahal_dists_from_x(x, precomputed_dists)
        mask_same  = (self.y_cal == y)
        mask_other = (self.y_cal != y)
        if not np.any(mask_same):
            return 1e9
        d_same  = np.min(dists[mask_same])
        d_other = np.min(dists[mask_other]) if np.any(mask_other) else 1e-8
        return float(d_same / (d_other + 1e-8))

    def updated_calibration_scores_for(self, x: np.ndarray, y: int,
                                       precomputed_dists: np.ndarray = None,
                                       **kwargs) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        dists_x = self._compute_mahal_dists_from_x(x, precomputed_dists)
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

    Full CP safety:
    - Whitening is fixed at fit() time (pooled_var does not change with x*).
    - After whitening + renorm, points live on a new unit sphere; Full CP update
      is identical to HypersphericalNNRatio: only lookup_same_sim / lookup_other_sim
      change by a max() comparison — no centroid shift.
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

    def _get_sims(self, x: np.ndarray, precomputed_dists=None) -> np.ndarray:
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

        # 4. Build lookup tables (same structure as HypersphericalNNRatio)
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

    def score_x(self, x: np.ndarray, y: int,
                precomputed_dists: np.ndarray = None, **kwargs) -> float:
        sims = self._get_sims(x, precomputed_dists)
        mask_same  = (self.y_cal == y)
        if not np.any(mask_same):
            return 1e9
        max_same  = float(np.max(sims[mask_same]))
        max_other = float(np.max(sims[~mask_same])) if np.any(~mask_same) else -1.0
        return float(self._geodesic_ratio(np.array([max_same]), np.array([max_other]))[0])

    def updated_calibration_scores_for(self, x: np.ndarray, y: int,
                                       precomputed_dists: np.ndarray = None,
                                       **kwargs) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        sims_x = self._get_sims(x, precomputed_dists)

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

    def score_x(self, x: np.ndarray, y: int, **kwargs) -> float:
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

    def updated_calibration_scores_for(self, x: np.ndarray, y: int,
                                        **kwargs) -> np.ndarray:
        if self.alpha0 is None:
            raise ValueError("Must call fit() first")
        sims_x = self._get_sims(x)
        k      = self.k

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
            return v if v is not None else np.full(len(self.y_cal), fallback)

        scores = self._score_ratio(
            _arr(upd_sum_same,  0.0), _arr(upd_k_same,  1.0), _arr(upd_max_same,  -1.0),
            _arr(upd_sum_other, 0.0), _arr(upd_k_other, 1.0), _arr(upd_max_other, -1.0),
        )
        scores[~np.isfinite(scores)] = 1e9
        return scores


class TemperedDensityRatioNCM(NonconformityMeasure):
    """Non-parametric log-density-ratio NCM on the unit sphere.

    S(x, y) = log KDE_all(x) - log KDE_y(x)

    where  KDE_c(x)   = sum_{x_j in C_c} exp(sim(x, x_j) / tau)
    and    KDE_all(x) = sum_c KDE_c(x)  (sum over ALL calibration points).

    Equivalent to -log p(y|x) under a von Mises-Fisher mixture on the sphere:
    low score  = x is well explained by class y relative to all other classes.

    Key properties:
    - tau -> 0  : recovers 1-NN ratio (same limit as geodesic_topk_asym at k=1)
    - tau -> inf: uniform kernel, no class discrimination
    - tau ~ 0.1 : standard in contrastive learning; good default for DINOv2

    Transductive FCP update is O(n_cal) via np.logaddexp — no re-fit needed.
    """

    def __init__(self, tau: float = 0.1):
        self.tau = tau
        self.X_norm = None       # (n, d) L2-normalised calibration embeddings
        self.y_cal = None        # (n,) integer labels
        self._lse_same = None    # (n,) logsumexp(sim(x_i, C_{y_i}) / tau) at fit time
        self._lse_all = None     # (n,) logsumexp(sim(x_i, X_cal)  / tau) at fit time
        self._cache_key = None
        self._cache_sims_tau = None  # (n,) cached sim(x*, X_norm) / tau

    # ------------------------------------------------------------------
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        norms = np.linalg.norm(X_cal, axis=1, keepdims=True)
        self.X_norm = X_cal / np.maximum(norms, 1e-10)
        self.y_cal = y_cal

        # Full pairwise similarity matrix / tau  — shape (n, n)
        S = (self.X_norm @ self.X_norm.T) / self.tau   # (n, n)
        np.fill_diagonal(S, -np.inf)                    # exclude self-similarity

        n = len(y_cal)
        self._lse_all  = scipy_logsumexp(S, axis=1)    # (n,)
        self._lse_same = np.empty(n)
        for i in range(n):
            self._lse_same[i] = scipy_logsumexp(S[i, y_cal == y_cal[i]])

    def get_calibration_scores(self) -> np.ndarray:
        return self._lse_all - self._lse_same

    # ------------------------------------------------------------------
    def _sims_tau(self, x: np.ndarray) -> np.ndarray:
        """Cached (n,) vector of sim(x, X_cal) / tau."""
        key = x.ctypes.data
        if key == self._cache_key:
            return self._cache_sims_tau
        x_norm = x / max(float(np.linalg.norm(x)), 1e-10)
        sims = (self.X_norm @ x_norm) / self.tau
        self._cache_key = key
        self._cache_sims_tau = sims
        return sims

    # ------------------------------------------------------------------
    def score_x(self, x: np.ndarray, y: int, precomputed_dists=None) -> float:
        """Transductive score: x scores against existing calibration members only (no self)."""
        sims = self._sims_tau(x)
        same_mask = self.y_cal == y
        if same_mask.sum() == 0:
            return 1e9
        lse_same_x = scipy_logsumexp(sims[same_mask])
        lse_all_x  = scipy_logsumexp(sims)
        return float(lse_all_x - lse_same_x)

    def score_x_cv(self, x: np.ndarray, y: int) -> float:
        """CV+: no transduction — x is NOT added to the calibration set."""
        sims = self._sims_tau(x)
        same_mask = self.y_cal == y
        if same_mask.sum() == 0:
            return 1e9
        return float(scipy_logsumexp(sims) - scipy_logsumexp(sims[same_mask]))

    def updated_calibration_scores_for(self, x: np.ndarray, y: int,
                                        precomputed_dists=None) -> np.ndarray:
        """O(n) update via logaddexp — no re-fit."""
        sims = self._sims_tau(x)                             # (n,)

        # Every cal point gains exp(sims[i]) in its denominator (all-class pool)
        lse_all_upd = np.logaddexp(self._lse_all, sims)     # (n,)

        # Only same-class cal points gain exp(sims[i]) in their numerator
        lse_same_upd = self._lse_same.copy()
        same_mask = self.y_cal == y
        lse_same_upd[same_mask] = np.logaddexp(
            self._lse_same[same_mask], sims[same_mask]
        )

        return lse_all_upd - lse_same_upd


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

    def score_x(self, x: np.ndarray, y: int, **kwargs) -> float:
        """Score a single point.  Ignores precomputed_dists (not applicable)."""
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

    def updated_calibration_scores_for(self, x: np.ndarray, y: int, **kwargs) -> np.ndarray:
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
        # Try new API first (SimplifiedKNN), fallback to legacy (KNN)
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
        verbose: bool = True
    ) -> Dict:
        """
        Compute prediction sets for test examples using Full CP.
        
        Args:
            X_test: Test features (n_test, d)
            return_p_values: If True, return p-values for all classes
            verbose: Show progress bar
            
        Returns:
            Dictionary with:
            - 'prediction_sets': List of prediction sets (lists of labels)
            - 'set_sizes': Size of each prediction set
            - 'p_values': (optional) p-values for each (test_idx, class) pair
            - 'prediction_time': Total prediction time in seconds
        """
        if self.cal_scores is None:
            raise ValueError("Must call calibrate() before predict()")
        
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
            p_vals = {}  # Always compute for debugging

            # Precompute distances once per test point for NCMs that support it (100x speedup)
            precomputed_dists = None
            if hasattr(self.ncm, 'X_cal') and self.ncm.X_cal is not None:
                precomputed_dists = euclidean_distances([x_test], self.ncm.X_cal).flatten()

            # For each candidate label, compute p-value
            for y_candidate in self.classes:
                yc = int(y_candidate)

                # Compute base scores
                if precomputed_dists is not None:
                    try:
                        test_score = self.ncm.score_x(x_test, yc, precomputed_dists=precomputed_dists)
                        updated_scores = self.ncm.updated_calibration_scores_for(x_test, yc, precomputed_dists=precomputed_dists)
                    except TypeError:
                        test_score = self.ncm.score_x(x_test, yc)
                        updated_scores = self.ncm.updated_calibration_scores_for(x_test, yc)
                else:
                    test_score = self.ncm.score_x(x_test, yc)
                    updated_scores = self.ncm.updated_calibration_scores_for(x_test, yc)

                n_greater = np.sum(updated_scores >= test_score)
                p_value = (n_greater + 1) / (n_cal + 1)

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
                         (e.g. lambda: SimplifiedKNNNonconformity(k=5))
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


def create_ncm(ncm_type: str, k: int = 5, lambda_reg: float = 1.0,
               reg: float = 1e-4, tau: float = 0.1) -> NonconformityMeasure:
    """
    Factory function to create NCM instances.

    Args:
        ncm_type: One of 'knn', 'simplified_knn', 'centroid',
                  'relative_centroid', 'ridge', 'nn_ratio', 'geodesic_nn_ratio',
                  'softmax', 'mahal_nn_ratio', 'whitened_geodesic'
        k: Number of neighbors (for knn, simplified_knn)
        lambda_reg: Regularization parameter (for ridge)
        reg: Variance regularisation for Mahalanobis NCMs
    """
    if ncm_type == "knn":
        return KNNNonconformity(k=k, metric='euclidean')
    elif ncm_type == "simplified_knn":
        return SimplifiedKNNNonconformity(k=k)
    elif ncm_type == "centroid":
        return CentroidNonconformity()
    elif ncm_type == "relative_centroid":
        return RelativeCentroidNonconformity()
    elif ncm_type == "ridge":
        return RidgeRegressionNonconformity(lambda_reg=lambda_reg)
    elif ncm_type == "nn_ratio":
        return FastNNRatio()
    elif ncm_type == "geodesic_nn_ratio":
        return HypersphericalNNRatio()
    elif ncm_type == "softmax":
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
    elif ncm_type == "tempered_density_ratio":
        return TemperedDensityRatioNCM(tau=tau)
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
