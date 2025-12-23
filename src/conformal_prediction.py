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
        
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Store calibration data for nonconformity computation."""
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(y_cal)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity score following Cherubin et al.
        
        For each point:
        1. Compute distances to all calibration points
        2. Find k smallest distances to same class
        3. Find k smallest distances to other classes
        4. Return ratio: sum(dist_same) / max(sum(dist_other), 0.1)
        """
        
        scores = np.zeros(len(X))
        
        for i, (x, label) in enumerate(zip(X, y)):
            # Compute distances to all calibration points
            distances = euclidean_distances([x], self.X_cal).flatten()
            
            # Get distances to same class and other classes
            same_class_mask = self.y_cal == label
            other_class_mask = self.y_cal != label
            
            dist_same = distances[same_class_mask]
            dist_other = distances[other_class_mask]
            
            # Get k smallest distances (best_k)
            k_same = min(self.k, len(dist_same))
            k_other = min(self.k, len(dist_other))
            
            if k_same > 0:
                kdist_same = np.partition(dist_same, k_same-1)[:k_same]
            else:
                kdist_same = np.array([1e10])
            
            if k_other > 0:
                kdist_other = np.partition(dist_other, k_other-1)[:k_other]
            else:
                kdist_other = np.array([0.1])
            
            # Nonconformity = sum(dist to same) / sum(dist to other)
            # Higher = more nonconforming (far from own class, close to others)
            scores[i] = np.sum(kdist_same) / max(np.sum(kdist_other), 0.1)
        
        return scores


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
                # Try new API (SimplifiedKNN with exact Full CP updates)
                try:
                    test_score = self.ncm.score_x(x_test, int(y_candidate))
                    updated_scores = self.ncm.updated_calibration_scores_for(x_test, int(y_candidate))
                    # p-value using updated calibration scores
                    n_greater = np.sum(updated_scores >= test_score)
                    p_value = (n_greater + 1) / (n_cal + 1)
                except (AttributeError, NotImplementedError):
                    # Fallback for legacy NCMs (KNN) - use fixed calibration scores
                    try:
                        test_score = self.ncm.score(
                            x_test.reshape(1, -1),
                            np.array([y_candidate]),
                            only_score_x=True
                        )[0]
                    except TypeError:
                        test_score = self.ncm.score(
                            x_test.reshape(1, -1),
                            np.array([y_candidate])
                        )[0]
                    # Use fixed calibration scores (not exact Full CP)
                    n_greater = np.sum(self.cal_scores >= test_score)
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
                    # Compute test scores for debugging
                    try:
                        test_scores_all = [self.ncm.score_x(x_test, int(c)) for c in self.classes]
                    except (AttributeError, NotImplementedError):
                        test_scores_all = [self.ncm.score(x_test.reshape(1, -1), np.array([c]))[0] for c in self.classes]
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
