"""
MA-CS (Model-Agnostic Class Similarity) experiment: binary superclass indicator penalty.
s_lambda(x, y) = s(x, y) + lambda * I{g(y) != g(y_hat(x))}

Reference: Fargion, Dabah & Tirer (2025), Section 4.
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.metrics import euclidean_distances
from conformal_prediction import (
    FullConformalPredictor, CrossValidationPlusPredictor,
    SoftmaxSplitCP, create_ncm,
)

# CIFAR-100 ImageFolder label -> coarse (superclass) label
# 20 superclasses, 5 fine classes each
CIFAR100_FINE_TO_COARSE = {
    0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,
    10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7,
    19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15,
    28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16,
    37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13,
    46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2,
    55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12,
    64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0,
    73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19,
    82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18,
    91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13,
}


def get_superclass(y):
    """Map fine label to coarse label."""
    return CIFAR100_FINE_TO_COARSE[int(y)]


def run_fcp_with_macs(X_cal, y_cal, X_test, y_test, all_classes, ncm_name,
                      alpha, lam):
    """Run FCP with MA-CS binary penalty applied post-hoc."""
    ncm = create_ncm(ncm_name, k=5)
    cp = FullConformalPredictor(ncm, alpha=alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)

    # --- Compute LOO 1-NN penalties for calibration points ---
    n_cal = len(X_cal)
    D_cal = euclidean_distances(X_cal, X_cal)
    np.fill_diagonal(D_cal, np.inf)
    loo_nn_idx = np.argmin(D_cal, axis=1)
    y_hat_loo = y_cal[loo_nn_idx]

    # Binary penalty: lambda * I{g(y_i) != g(y_hat_i)}
    cal_penalty = np.zeros(n_cal)
    for i in range(n_cal):
        if get_superclass(y_cal[i]) != get_superclass(y_hat_loo[i]):
            cal_penalty[i] = lam

    # Add penalty to cached calibration scores
    base_cal_scores = cp.cal_scores.copy()
    cp.cal_scores = base_cal_scores + cal_penalty

    # --- Predict with MA-CS penalty on test scores ---
    n_test = len(X_test)
    prediction_sets = []

    # Precompute test-to-cal distances
    D_test_cal = euclidean_distances(X_test, X_cal)

    for i in range(n_test):
        x_test = X_test[i]
        dists = D_test_cal[i]

        # y_hat for test point = 1-NN
        nn_idx = np.argmin(dists)
        y_hat_test = int(y_cal[nn_idx])
        g_hat = get_superclass(y_hat_test)

        pred_set = []
        for yc in all_classes:
            yc = int(yc)
            # Base test score
            test_score = cp.ncm.score_x(x_test, yc, precomputed_dists=dists)
            # MA-CS penalty
            if get_superclass(yc) != g_hat:
                test_score += lam

            # Updated cal scores (base + cal_penalty already applied)
            updated_scores = cp.ncm.updated_calibration_scores_for(
                x_test, yc, precomputed_dists=dists) + cal_penalty

            # p-value
            n_greater = np.sum(updated_scores >= test_score)
            p_value = (n_greater + 1) / (n_cal + 1)

            if p_value > alpha:
                pred_set.append(yc)

        prediction_sets.append(pred_set)

    # Compute metrics
    coverage = np.mean([y_test[i] in prediction_sets[i]
                        for i in range(n_test)])
    avg_size = np.mean([len(s) for s in prediction_sets])
    return coverage, avg_size


def stratified_split(X, y, cal_size, test_size, all_classes, rng):
    """Stratified pool -> cal + test split."""
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


def main():
    # Load CIFAR-100
    data = torch.load("output/embeddings_cifar100.pt", map_location="cpu",
                      weights_only=False)
    X = data["embeddings"].numpy()
    y = data["labels"].numpy()
    all_classes = np.unique(y)
    print(f"CIFAR-100: {X.shape}, {len(all_classes)} classes, "
          f"20 superclasses (5 fine per coarse)")

    NCM = "geodesic_topk_mean"
    ALPHA = 0.1
    CAL_SIZES = [300, 400, 600, 800]
    TEST_SIZE = 300
    N_TRIALS = 3
    SEED = 42
    LAMBDAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1]

    print(f"NCM: {NCM}, alpha={ALPHA}, trials={N_TRIALS}")
    print(f"Lambda sweep: {LAMBDAS}")
    print()

    for cal_size in CAL_SIZES:
        print(f"{'='*70}")
        print(f"cal_size={cal_size}")
        print(f"{'='*70}")

        for lam in LAMBDAS:
            covs, szs = [], []
            t0 = time.time()
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(SEED + trial * 1000)
                X_cal, y_cal, X_test, y_test = stratified_split(
                    X, y, cal_size, TEST_SIZE, all_classes, rng)

                if lam == 0.0:
                    # Plain FCP (no penalty)
                    ncm = create_ncm(NCM, k=5)
                    cp = FullConformalPredictor(ncm, alpha=ALPHA)
                    cp.calibrate(X_cal, y_cal, all_classes=all_classes)
                    m = cp.evaluate(X_test, y_test, verbose=False)
                    covs.append(m["coverage"])
                    szs.append(m["avg_set_size"])
                else:
                    cov, sz = run_fcp_with_macs(
                        X_cal, y_cal, X_test, y_test, all_classes,
                        NCM, ALPHA, lam)
                    covs.append(cov)
                    szs.append(sz)

            elapsed = time.time() - t0
            cov_mean = np.mean(covs)
            sz_mean = np.mean(szs)
            valid = "OK" if cov_mean >= 0.89 else "!!"
            label = f"lambda={lam:.2f}" if lam > 0 else "No penalty"
            print(f"  {label:15s}  cov={cov_mean:.3f} {valid}  "
                  f"sz={sz_mean:.2f}  ({elapsed:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
