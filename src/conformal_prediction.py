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


class KNNNonconformity(NonconformityMeasure):
    """
    k-Nearest Neighbors nonconformity measure (Cherubin et al. 2021).
    
    Nonconformity score = sum(k distances to own class) / sum(k distances to other classes)
    
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

class FullConformalPredictor:
    """
    Full Conformal Prediction with optimizations from Cherubin et al. 2021.
    
    For each test example and each candidate label:
    1. Compute nonconformity score assuming that label
    2. Compare with calibration scores
    3. Compute p-value: fraction of calibration examples with score >= test score
    4. Include label in prediction set if p-value > significance level α
    
    Optimizations:
    - Vectorized operations where possible
    - Caching of calibration scores
    - Early stopping for prediction sets
    """


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


class FullConformalPredictor:
    
    def __init__(
        self,
        nonconformity_measure: NonconformityMeasure,
        alpha: float = 0.1
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
        
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Calibrate the conformal predictor on calibration set.
        
        Args:
            X_cal: Calibration features (n_cal, d)
            y_cal: Calibration labels (n_cal,)
        """
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(y_cal)
        
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
        
        print(f"Calibrated with {len(X_cal)} examples, {len(self.classes)} classes")
        print(f"Calibration scores - min: {self.cal_scores.min():.4f}, max: {self.cal_scores.max():.4f}, mean: {self.cal_scores.mean():.4f}, std: {self.cal_scores.std():.4f}")
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
            
            # For each candidate label, compute p-value
            for y_candidate in self.classes:
                # Exact Full CP path using new NCM API
                test_score = self.ncm.score_x(x_test, int(y_candidate))
                updated_scores = self.ncm.updated_calibration_scores_for(x_test, int(y_candidate))
                n_greater = np.sum(updated_scores >= test_score)
                p_value = (n_greater + 1) / (n_cal + 1)
                
                p_vals[int(y_candidate)] = p_value
                
                # Include in prediction set if p-value > alpha
                if p_value > self.alpha:
                    pred_set.append(int(y_candidate))
            
            # Debug: warn if empty
            if len(pred_set) == 0:
                empty_count += 1
                if empty_count <= 3 and verbose:  # Show first 3 warnings
                    best_p = max(p_vals.values())
                    best_class = max(p_vals, key=p_vals.get)
                    print(f"\nWarning: Empty prediction set for test example {i}")
                    print(f"  Best p-value: {best_p:.4f} (class {best_class}), alpha: {self.alpha}")
                    # Compute test scores for debugging using new API
                    test_scores_all = [self.ncm.score_x(x_test, int(c)) for c in self.classes]
                    print(f"  Test scores range: [{min(test_scores_all):.4f}, {max(test_scores_all):.4f}]")
                    print(f"  Cal scores range: [{self.cal_scores.min():.4f}, {self.cal_scores.max():.4f}]")
            
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


def train_val_test_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.0,  # Kept for backward compatibility, but not used
    cal_ratio: float = 0.5,
    test_ratio: float = 0.5,
    random_state: int = 42
) -> Tuple:
    """
    DEPRECATED: Use cal_test_split() instead.
    
    Full CP doesn't use a training set, only calibration and test.
    This function is kept for backward compatibility but returns empty train sets.
    """
    X_cal, y_cal, X_test, y_test = cal_test_split(
        embeddings, labels, cal_ratio=cal_ratio, random_state=random_state
    )
    
    # Return empty train sets for compatibility
    X_train = np.array([])
    y_train = np.array([])
    
    return X_train, y_train, X_cal, y_cal, X_test, y_test
