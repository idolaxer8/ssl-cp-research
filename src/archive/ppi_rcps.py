"""
PPI-RCPS: Prediction-Powered Inference for Risk-Controlling Prediction Sets.

ARCHIVED 2026-05-18 — Not relevant to current research line.

Originally added as a stronger semi-supervised baseline (Einbinder et al. 2024,
arXiv:2412.11174). The intended use was lambda-tuning for MS-CS, but the project
has moved on and FCP+PCA already dominates SemiCP empirically across all our
benchmarks (see findings.md §11). Archived here for historical reference.

To re-activate: copy `PPIRCPS` and `clopper_pearson_ucb` back into
`src/conformal_prediction.py` (clopper_pearson_ucb belongs near other utility
functions around line 1015; PPIRCPS belongs near `SemiCP`/`CrossValidationPlus`).
The `compute_cp_scores` function it depends on remains in conformal_prediction.py.

Reference: Einbinder, B.-S., Ringel, L., & Romano, Y. (2024). Semi-supervised
risk control via prediction-powered inference. arXiv:2412.11174
"""

import os
import sys
import numpy as np
from typing import Dict

# PPIRCPS depends on compute_cp_scores from conformal_prediction.py (still in src/).
# This archive lives in src/archive/, so add the parent (src/) to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conformal_prediction import compute_cp_scores  # type: ignore


def clopper_pearson_ucb(k: int, m: int, delta: float) -> float:
    """Clopper-Pearson upper confidence bound on binomial proportion.
    Given k events in m trials, returns UCB at confidence 1-delta."""
    from scipy.stats import beta as beta_dist
    if m <= 0:
        return 1.0
    if k >= m:
        return 1.0
    if k < 0:
        return 0.0
    return float(beta_dist.ppf(1 - delta, k + 1, m - k))


class PPIRCPS:
    """
    Prediction-Powered Inference for Risk-Controlling Prediction Sets (PPI-RCPS).

    Uses unlabeled data to tighten threshold selection via Prediction-Powered
    Inference with Clopper-Pearson UCBs.

    For each candidate threshold q, computes a UCB on miscoverage that combines:
    1. Unlabeled risk estimate R_U(q) using pseudo-labels (low variance, biased)
    2. Labeled rectifying correction R_rect(q) that removes bias (unbiased, higher var)

    Selects the smallest q whose UCB stays below alpha.

    Reference: Einbinder et al. (2024/2025), arXiv 2412.11174
    """

    def __init__(self, alpha: float = 0.1, score_fn: str = "THR",
                 delta: float = 0.1, delta1_frac: float = 0.1,
                 max_iter: int = 1000, k_reg: int = 5, lambda_raps: float = 0.01):
        self.alpha = alpha
        self.score_fn = score_fn
        self.delta = delta
        self.delta1_frac = delta1_frac
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
        Calibrate using PPI grid search (Algorithm 2 from Einbinder et al.).

        Args:
            X_cal: Calibration features (n, d)
            y_cal: Calibration labels (n,)
            X_unlabeled: Unlabeled features (N, d)
            all_classes: Full label space for prediction sets
        """
        if self.classifier is None:
            raise ValueError("Must call fit() before calibrate()")

        self._all_classes = all_classes

        delta1 = self.delta * self.delta1_frac
        delta2 = self.delta * (1 - self.delta1_frac)

        X_cal_scaled = self.scaler.transform(X_cal)
        probs_cal = self.classifier.predict_proba(X_cal_scaled)
        y_cal_idx = np.array([self._class_to_idx.get(y, 0) for y in y_cal])
        n = len(y_cal)
        self._n_labeled = n

        # Labeled true scores: S(x_i, y_i)
        true_scores_cal = compute_cp_scores(
            probs_cal, y_cal_idx, self.score_fn, self.k_reg, self.lambda_raps)
        # Labeled pseudo-scores: S(x_i, y_hat_i)
        y_hat_cal = np.argmax(probs_cal, axis=1)
        pseudo_scores_cal = compute_cp_scores(
            probs_cal, y_hat_cal, self.score_fn, self.k_reg, self.lambda_raps)

        if X_unlabeled is not None and len(X_unlabeled) > 0:
            X_u_scaled = self.scaler.transform(X_unlabeled)
            probs_u = self.classifier.predict_proba(X_u_scaled)
            y_hat_u = np.argmax(probs_u, axis=1)
            pseudo_scores_u = compute_cp_scores(
                probs_u, y_hat_u, self.score_fn, self.k_reg, self.lambda_raps)
            N = len(X_unlabeled)
            self._n_unlabeled = N
        else:
            pseudo_scores_u = np.array([])
            N = 0
            self._n_unlabeled = 0

        # Build grid of candidate thresholds
        all_score_vals = np.concatenate([true_scores_cal, pseudo_scores_u]) \
            if N > 0 else true_scores_cal.copy()
        Q_grid = np.unique(all_score_vals)
        # Supplement with uniform grid if too sparse
        if len(Q_grid) < 100:
            lo, hi = Q_grid.min(), Q_grid.max()
            Q_grid = np.unique(np.concatenate([Q_grid, np.linspace(lo, hi, 200)]))
        Q_grid = np.sort(Q_grid)[::-1]  # descending: large q first (safe)

        # Grid search: large q -> small q
        q_hat = Q_grid[0]  # safest default
        if N > 0:
            for q in Q_grid:
                L_u = (pseudo_scores_u > q)
                L_cal = (true_scores_cal > q)
                L_cal_hat = (pseudo_scores_cal > q)

                # Clipped rectifier: max(0, L_i - L_tilde_i)
                rect = np.maximum(0, L_cal.astype(int) - L_cal_hat.astype(int))

                UCB_U = clopper_pearson_ucb(int(L_u.sum()), N, delta1)
                UCB_rect = clopper_pearson_ucb(int(rect.sum()), n, delta2)
                UCB = UCB_U + UCB_rect

                if UCB < self.alpha:
                    q_hat = q  # keep tightening
                else:
                    break
        else:
            # Fallback: standard split CP quantile from labeled cal only
            quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            quantile_level = min(quantile_level, 1.0)
            q_hat = np.quantile(true_scores_cal, quantile_level)

        self.q_hat = q_hat

        print(f"PPI-RCPS calibrated: {n} labeled + {N} unlabeled, "
              f"score_fn={self.score_fn}, q_hat={self.q_hat:.4f}")

    def predict(self, X_test: np.ndarray) -> Dict:
        """Compute prediction sets for test examples."""
        if self.q_hat is None:
            raise ValueError("Must call calibrate() before predict()")

        X_scaled = self.scaler.transform(X_test)
        probs = self.classifier.predict_proba(X_scaled)
        n_test, K = probs.shape

        if self._all_classes is not None:
            candidate_classes = np.unique(self._all_classes)
        else:
            candidate_classes = self._clf_classes

        clf_to_col = {c: i for i, c in enumerate(self._clf_classes)}

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
                sorted_col = np.argsort(-probs[i])
                cumsum = 0.0
                pred_set = []
                for j, col_idx in enumerate(sorted_col):
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
            'n_unlabeled': self._n_unlabeled,
        }

        if verbose:
            print("\n" + "=" * 50)
            print(f"PPI-RCPS EVALUATION (score_fn={self.score_fn})")
            print("=" * 50)
            print(f"Test examples:                 {n_test}")
            print(f"Target coverage (1-alpha):     {1 - self.alpha:.3f}")
            print(f"Achieved coverage:             {coverage:.3f}")
            print(f"Average set size:              {avg_set_size:.2f}")
            print(f"Median set size:               {median_set_size:.1f}")
            print(f"Singleton rate:                {singleton_rate:.3f}")
            print(f"Empty set rate:                {empty_set_rate:.3f}")
            print(f"Classifier accuracy (top-1):   {classifier_accuracy:.3f}")
            print(f"Labeled cal:                   {self._n_labeled}")
            print(f"Unlabeled:                     {self._n_unlabeled}")
            print("=" * 50)

        return metrics
