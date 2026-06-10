"""
Inductive (split) conformal prediction baselines.

Moved verbatim out of conformal_prediction.py (2026-06-10) to keep that module
focused on the transductive FCP core (NCMs + FullConformalPredictor + CV+).
conformal_prediction re-exports every public name here, so
`from conformal_prediction import SoftmaxSplitCP, SemiCP, ...` keeps working.

Contents:
- compute_cp_scores / compute_cp_sets -- THR / APS / RAPS score functions
- SoftmaxSplitCP -- split CP with a logistic-regression softmax head (THR)
- SemiCP -- semi-supervised CP with NNM augmentation (Zhou et al. 2025)
- ClusteredSplitCP -- Clustered CP baseline (Ding et al. 2023)
"""

from typing import Dict, Optional

import numpy as np


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
    n = probs.shape[0]
    scores = np.zeros(n)

    if score_fn == "THR":
        for i in range(n):
            scores[i] = 1.0 - probs[i, y_indices[i]]
        return scores

    if score_fn in ("APS", "RAPS"):
        for i in range(n):
            sorted_idx = np.argsort(-probs[i])
            # rank of y in the descending order (1-based)
            rank = int(np.where(sorted_idx == y_indices[i])[0][0]) + 1
            # APS score = cumulative probability up to and including y's position
            cumsum = float(probs[i, sorted_idx[:rank]].sum())
            scores[i] = cumsum
            if score_fn == "RAPS":
                scores[i] += lambda_raps * max(0, rank - k_reg)
        return scores

    raise ValueError(f"Unknown score_fn: {score_fn}. Use 'THR', 'APS', or 'RAPS'.")


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
        return prediction_sets

    if score_fn in ("APS", "RAPS"):
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
        return prediction_sets

    raise ValueError(f"Unknown score_fn: {score_fn}")


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

        self.classifier = LogisticRegression(
            max_iter=self.max_iter, solver='lbfgs', random_state=42,
        )
        self.classifier.fit(X_train_scaled, y_train)

        print(f"Trained softmax classifier on {len(X_train)} examples, "
              f"{len(self.classes)} classes")


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
        probs = self.classifier.predict_proba(X_cal_scaled)

        # Score: s_i = 1 - p(y_true | x_i); unknown classes get max nonconformity (1.0).
        class_to_idx = {c: i for i, c in enumerate(self.classifier.classes_)}
        self.cal_scores = np.array([
            1.0 - prob[class_to_idx[y_true]] if y_true in class_to_idx else 1.0
            for prob, y_true in zip(probs, y_cal)
        ])

        # Finite-sample-corrected quantile level for split CP.
        n_cal = len(y_cal)
        quantile_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
        self.q_hat = np.quantile(self.cal_scores, quantile_level)

        s = self.cal_scores
        print(f"Calibrated on {n_cal} examples")
        print(f"Calibration scores - min: {s.min():.4f}, max: {s.max():.4f}, "
              f"mean: {s.mean():.4f}, std: {s.std():.4f}")
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

        empty_set_rate = sum(1 for ps in pred_sets if len(ps) == 0) / n_test

        X_test_scaled = self.scaler.transform(X_test)
        classifier_accuracy = float(np.mean(self.classifier.predict(X_test_scaled) == y_test))

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
            print("\n" + "=" * 50)
            print("SOFTMAX SPLIT CP EVALUATION")
            print("=" * 50)
            print(f"Test examples:                 {n_test}")
            print(f"Target coverage (1-alpha):     {1 - self.alpha:.3f}")
            print(f"Achieved coverage:             {coverage:.3f}")
            print(f"Average set size:              {avg_set_size:.2f}")
            print(f"Median set size:               {median_set_size:.1f}")
            print(f"Singleton rate:                {singleton_rate:.3f}")
            print(f"Singleton accuracy:            {singleton_accuracy:.3f}")
            print(f"Empty set rate:                {empty_set_rate:.3f}")
            print(f"Classifier accuracy (top-1):   {classifier_accuracy:.3f}")
            print("=" * 50)

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


class ClusteredSplitCP:
    """
    Clustered Conformal Prediction (Ding, Tibshirani & Ramdas, 2023).

    Class-conditional split CP for many-class settings. Classes are clustered
    by their calibration-score distribution signatures (k-means on per-class
    score quantiles); a separate threshold q_hat is fit per cluster, so the
    coverage guarantee is approximately class-conditional rather than only
    marginal. Rare classes (n_y < n_thresh) are pooled into a single "null
    cluster" to avoid degenerate quantile estimates.

    Uses THR scores (s = 1 - p(y|x)), matching Ding et al.'s main
    classification setup. Classifier is logistic regression to keep parity
    with SoftmaxSplitCP / SemiCP baselines in this codebase.

    Reference: Ding, Tibshirani & Ramdas (2023),
        "Class-Conditional Conformal Prediction with Many Classes", NeurIPS.
        arXiv:2306.09335. Public repo: tiffanyding/class-conditional-conformal.

    Note: at very small cal-per-class (e.g. cal=400 with K=100 -> 4/class),
    most classes fall into the rare null cluster and ClusterCP effectively
    degenerates to plain Split CP. `fraction_rare` is logged so this collapse
    is visible.
    """

    def __init__(self, alpha: float = 0.1, n_clusters: int = 5,
                 n_quantile_levels: int = 5, n_thresh: Optional[int] = None,
                 quantile_levels: Optional[np.ndarray] = None,
                 random_state: int = 42, max_iter: int = 1000):
        """
        Args:
            alpha: target miscoverage
            n_clusters: target number of clusters for k-means on signatures
            n_quantile_levels: number of quantile levels in each per-class
                score signature (only used when quantile_levels is None)
            n_thresh: minimum cal samples per class to qualify for clustering.
                If None, uses ceil((n_quantile_levels + 1) / alpha)
                (matches Ding et al.'s public repo default).
            quantile_levels: explicit quantile levels (e.g. [0.5, 0.6, ...]).
                If None, evenly spaced in [0.5, 0.95].
            random_state: seed for KMeans
            max_iter: logistic regression max iterations
        """
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.n_quantile_levels = n_quantile_levels
        self.n_thresh = n_thresh
        if quantile_levels is None:
            self.quantile_levels = np.linspace(0.5, 0.95, n_quantile_levels)
        else:
            self.quantile_levels = np.asarray(quantile_levels)
            self.n_quantile_levels = len(self.quantile_levels)
        self.random_state = random_state
        self.max_iter = max_iter

        self.classifier = None
        self.scaler = None
        self.cluster_of = None
        self.q_hat_by_cluster = None
        self._all_classes = None
        self._fraction_rare = None
        self._effective_n_clusters = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train logistic regression classifier (matches SoftmaxSplitCP.fit)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        self.classifier = LogisticRegression(
            max_iter=self.max_iter, solver='lbfgs', random_state=42,
        )
        self.classifier.fit(X_scaled, y_train)
        self._clf_classes = self.classifier.classes_
        self._class_to_col = {c: i for i, c in enumerate(self._clf_classes)}

        print(f"ClusteredSplitCP: trained classifier on {len(X_train)} examples, "
              f"{len(self._clf_classes)} classes")

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                  all_classes: np.ndarray = None):
        """Score cal points, cluster classes by score signature, fit per-cluster q_hat."""
        from sklearn.cluster import KMeans

        if self.classifier is None:
            raise ValueError("Must call fit() before calibrate()")

        self._all_classes = (np.unique(all_classes) if all_classes is not None
                             else np.unique(y_cal))

        X_cal_scaled = self.scaler.transform(X_cal)
        probs_cal = self.classifier.predict_proba(X_cal_scaled)
        cal_scores = np.empty(len(y_cal), dtype=float)
        for i, y_true in enumerate(y_cal):
            if y_true in self._class_to_col:
                cal_scores[i] = 1.0 - probs_cal[i, self._class_to_col[y_true]]
            else:
                cal_scores[i] = 1.0

        scores_by_class = {int(c): cal_scores[y_cal == c] for c in self._all_classes}

        if self.n_thresh is None:
            import math as _math
            self.n_thresh = int(_math.ceil((self.n_quantile_levels + 1) / self.alpha))

        eligible = [int(c) for c in self._all_classes
                    if len(scores_by_class[int(c)]) >= self.n_thresh]
        rare = [int(c) for c in self._all_classes
                if len(scores_by_class[int(c)]) < self.n_thresh]
        self._fraction_rare = len(rare) / len(self._all_classes)

        cluster_of = {int(c): -1 for c in rare}
        if len(eligible) >= 2:
            n_clusters = min(self.n_clusters, len(eligible))
            sigs = np.stack([
                np.quantile(scores_by_class[c], self.quantile_levels)
                for c in eligible
            ])
            km = KMeans(n_clusters=n_clusters, n_init=10,
                        random_state=self.random_state)
            km.fit(sigs)
            for c, lbl in zip(eligible, km.labels_):
                cluster_of[c] = int(lbl)
            self._effective_n_clusters = n_clusters
        elif len(eligible) == 1:
            cluster_of[eligible[0]] = 0
            self._effective_n_clusters = 1
        else:
            self._effective_n_clusters = 0

        cluster_ids = sorted(set(cluster_of.values()))
        q_hat_by_cluster = {}
        cluster_pool_sizes = {}
        for g in cluster_ids:
            classes_in_g = [c for c in cluster_of if cluster_of[c] == g]
            pool = np.concatenate([scores_by_class[c] for c in classes_in_g])
            n_g = len(pool)
            cluster_pool_sizes[g] = n_g
            if n_g == 0:
                q_hat_by_cluster[g] = np.inf
                continue
            level = min(np.ceil((n_g + 1) * (1 - self.alpha)) / n_g, 1.0)
            q_hat_by_cluster[g] = float(np.quantile(pool, level))

        self.cluster_of = cluster_of
        self.q_hat_by_cluster = q_hat_by_cluster

        print(f"ClusteredSplitCP calibrated: {len(eligible)}/{len(self._all_classes)} "
              f"classes eligible (n_thresh={self.n_thresh}), "
              f"{len(rare)} rare ({self._fraction_rare:.1%}) pooled into null cluster")
        print(f"  effective clusters: {self._effective_n_clusters} + "
              f"{1 if rare else 0} null. Pool sizes: "
              f"{ {g: cluster_pool_sizes[g] for g in cluster_ids} }")
        print(f"  q_hat per cluster: "
              f"{ {g: round(q_hat_by_cluster[g], 4) for g in cluster_ids} }")

    def predict(self, X_test: np.ndarray) -> Dict:
        """Include class c in C(x) iff (1 - p(c|x)) <= q_hat[cluster_of[c]]."""
        if self.q_hat_by_cluster is None:
            raise ValueError("Must call calibrate() before predict()")

        X_scaled = self.scaler.transform(X_test)
        probs = self.classifier.predict_proba(X_scaled)
        n_test = len(X_test)

        candidate_classes = np.unique(self._all_classes)
        prediction_sets = []
        for i in range(n_test):
            pred_set = []
            for c in candidate_classes:
                ci = int(c)
                p = probs[i, self._class_to_col[ci]] if ci in self._class_to_col else 0.0
                score = 1.0 - p
                g = self.cluster_of.get(ci, -1)
                if g not in self.q_hat_by_cluster:
                    q = max(self.q_hat_by_cluster.values()) if self.q_hat_by_cluster else 0.0
                else:
                    q = self.q_hat_by_cluster[g]
                if score <= q:
                    pred_set.append(ci)
            prediction_sets.append(pred_set)

        return {'prediction_sets': prediction_sets}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 verbose: bool = True) -> Dict:
        """Compute marginal coverage / set size / singleton metrics."""
        predictions = self.predict(X_test)
        pred_sets = predictions['prediction_sets']
        n_test = len(y_test)

        covered = sum(1 for i, ps in enumerate(pred_sets) if y_test[i] in ps)
        coverage = covered / n_test
        set_sizes = [len(ps) for ps in pred_sets]
        avg_set_size = float(np.mean(set_sizes))
        median_set_size = float(np.median(set_sizes))
        singleton_count = sum(1 for ps in pred_sets if len(ps) == 1)
        singleton_rate = singleton_count / n_test
        singleton_correct = sum(1 for i, ps in enumerate(pred_sets)
                                if len(ps) == 1 and y_test[i] in ps)
        singleton_accuracy = (singleton_correct / singleton_count
                              if singleton_count > 0 else 0.0)
        empty_set_rate = sum(1 for ps in pred_sets if len(ps) == 0) / n_test

        X_scaled = self.scaler.transform(X_test)
        classifier_accuracy = float(np.mean(
            self.classifier.predict(X_scaled) == y_test))

        metrics = {
            'coverage': coverage,
            'avg_set_size': avg_set_size,
            'median_set_size': median_set_size,
            'singleton_rate': singleton_rate,
            'singleton_accuracy': singleton_accuracy,
            'empty_set_rate': empty_set_rate,
            'classifier_accuracy': classifier_accuracy,
            'set_sizes': set_sizes,
            'fraction_rare': self._fraction_rare,
            'effective_n_clusters': self._effective_n_clusters,
        }

        if verbose:
            print("\n" + "=" * 50)
            print("CLUSTERED SPLIT CP EVALUATION")
            print("=" * 50)
            print(f"Test examples:                 {n_test}")
            print(f"Target coverage (1-alpha):     {1 - self.alpha:.3f}")
            print(f"Achieved (marginal) coverage:  {coverage:.3f}")
            print(f"Average set size:              {avg_set_size:.2f}")
            print(f"Median set size:               {median_set_size:.1f}")
            print(f"Singleton rate:                {singleton_rate:.3f}")
            print(f"Fraction rare classes:         {self._fraction_rare:.3f}")
            print(f"Effective n_clusters:          {self._effective_n_clusters}")
            print("=" * 50)

        return metrics
