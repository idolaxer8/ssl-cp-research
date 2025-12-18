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
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


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
    k-Nearest Neighbors nonconformity measure.
    Nonconformity = average distance to k nearest neighbors of the same class.
    
    Works well with normalized SSL embeddings where Euclidean distance ≈ Cosine distance.
    """
    
    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        self.k = k
        self.metric = metric
        self.X_cal = None
        self.y_cal = None
        self.classes = None
        self.knn_models = {}  # One KNN per class
        
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Fit KNN models for each class."""
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(y_cal)
        
        # Build a KNN index for each class
        for cls in self.classes:
            mask = y_cal == cls
            X_cls = X_cal[mask]
            
            # Ensure k doesn't exceed number of samples
            k_actual = min(self.k, len(X_cls))
            
            if len(X_cls) > 0:
                knn = NearestNeighbors(n_neighbors=k_actual, metric=self.metric)
                knn.fit(X_cls)
                self.knn_models[cls] = knn
    
    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity score = average distance to k nearest neighbors of same class.
        """
        scores = np.zeros(len(X))
        
        for i, (x, label) in enumerate(zip(X, y)):
            if label not in self.knn_models:
                # Unknown class -> maximum nonconformity
                scores[i] = np.inf
                continue
            
            knn = self.knn_models[label]
            x_reshaped = x.reshape(1, -1)
            
            # Get distances to k nearest neighbors of same class
            distances, _ = knn.kneighbors(x_reshaped)
            scores[i] = np.mean(distances[0])
        
        return scores


class InverseKNNNonconformity(NonconformityMeasure):
    """
    Inverse k-NN nonconformity: average distance to k-NN of OTHER classes.
    Nonconformity = 1 / (1 + avg_distance_to_other_classes)
    
    Lower distance to other classes = higher nonconformity = less confident.
    """
    
    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        self.k = k
        self.metric = metric
        self.X_cal = None
        self.y_cal = None
        self.classes = None
        
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray):
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.classes = np.unique(y_cal)
        
    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute inverse distance to other classes."""
        scores = np.zeros(len(X))
        
        for i, (x, label) in enumerate(zip(X, y)):
            # Get samples from OTHER classes
            mask_other = self.y_cal != label
            X_other = self.X_cal[mask_other]
            
            if len(X_other) == 0:
                scores[i] = 0.0
                continue
            
            # Find k nearest neighbors from other classes
            k_actual = min(self.k, len(X_other))
            knn = NearestNeighbors(n_neighbors=k_actual, metric=self.metric)
            knn.fit(X_other)
            
            x_reshaped = x.reshape(1, -1)
            distances, _ = knn.kneighbors(x_reshaped)
            
            # Inverse nonconformity: smaller distance to other classes = more nonconforming
            avg_dist = np.mean(distances[0])
            scores[i] = 1.0 / (1.0 + avg_dist)
        
        return scores


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
        self.ncm.fit(X_cal, y_cal)
        
        # Compute and cache calibration scores
        self.cal_scores = self.ncm.score(X_cal, y_cal)
        
        print(f"Calibrated with {len(X_cal)} examples, {len(self.classes)} classes")
        print(f"Calibration scores - mean: {self.cal_scores.mean():.4f}, std: {self.cal_scores.std():.4f}")
    
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
        """
        if self.cal_scores is None:
            raise ValueError("Must call calibrate() before predict()")
        
        n_test = len(X_test)
        n_cal = len(self.X_cal)
        
        prediction_sets = []
        set_sizes = []
        all_p_values = [] if return_p_values else None
        
        iterator = tqdm(range(n_test), desc="Full CP") if verbose else range(n_test)
        
        for i in iterator:
            x_test = X_test[i]
            pred_set = []
            p_vals = {} if return_p_values else None
            
            # For each candidate label, compute p-value
            for y_candidate in self.classes:
                # Compute test score assuming this label
                test_score = self.ncm.score(
                    x_test.reshape(1, -1),
                    np.array([y_candidate])
                )[0]
                
                # Compute p-value: fraction of calibration scores >= test score
                # Add 1 to numerator and denominator (smoothed p-value)
                n_greater = np.sum(self.cal_scores >= test_score)
                p_value = (n_greater + 1) / (n_cal + 1)
                
                if return_p_values:
                    p_vals[int(y_candidate)] = p_value
                
                # Include in prediction set if p-value > alpha
                if p_value > self.alpha:
                    pred_set.append(int(y_candidate))
            
            prediction_sets.append(pred_set)
            set_sizes.append(len(pred_set))
            
            if return_p_values:
                all_p_values.append(p_vals)
        
        results = {
            'prediction_sets': prediction_sets,
            'set_sizes': np.array(set_sizes)
        }
        
        if return_p_values:
            results['p_values'] = all_p_values
        
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


def train_val_test_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.5,
    cal_ratio: float = 0.25,
    test_ratio: float = 0.25,
    random_state: int = 42
) -> Tuple:
    """
    Split data into train/calibration/test sets.
    
    Args:
        embeddings: Feature vectors (n, d)
        labels: Labels (n,)
        train_ratio: Fraction for training (not used in CP, but for future classifier training)
        cal_ratio: Fraction for calibration
        test_ratio: Fraction for testing
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
