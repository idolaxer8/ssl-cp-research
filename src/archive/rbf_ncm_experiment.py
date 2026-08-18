"""RBF density NCM iteration: validity + comparison vs PCA+geodesic baseline.

Pipeline:
1. Smoke: tiny synthetic check that NCM math is sane.
2. Validity: FCP coverage on CIFAR-100 with rbf_density across 3 seeds.
3. Comparison at cal in {400, 600, 800}, 3 seeds, CIFAR-100:
     baseline       PCA-128 + geodesic_topk_mean
     ablation A     PCA-32  + geodesic_topk_mean    (control for dim)
     ablation B     AE-32   + geodesic_topk_mean    (the current "AE no-help")
     ablation C     PCA-128 + rbf_density           (control for NCM)
     candidate D    PCA-32  + rbf_density           (linear features + nonlinear NCM)
     candidate E    AE-32   + rbf_density           (nonlinear + nonlinear)
"""
# ARCHIVE-CANDIDATE (review 2026-06-24): HOLD for P1.5 (RBF NCM multi-dataset confirmation); archive if P1.5 is dropped.
# If unused by the review date, move to src/archive/ -- watch list: src/archive/README.md.


import sys
import time
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, "src")
from conformal_prediction import (
    RBFDensityNCM, GeodesicTopKMeanNCM, FullConformalPredictor,
    stratified_cal_test_split,
)
from autoencoder_utils import EmbeddingAutoencoder


# ------------------------------------------------------------------
# Smoke test on synthetic data
# ------------------------------------------------------------------
def smoke():
    print("=" * 60)
    print("Smoke: RBFDensityNCM on synthetic 3-class Gaussians")
    print("=" * 60)
    rng = np.random.default_rng(0)
    d = 8
    n_per_class = 50
    centers = np.array([[3, 0] + [0]*(d-2), [-3, 0] + [0]*(d-2), [0, 3] + [0]*(d-2)])
    X, y = [], []
    for c in range(3):
        Xc = rng.normal(centers[c], 0.5, (n_per_class, d))
        X.append(Xc); y.append(np.full(n_per_class, c))
    X = np.vstack(X); y = np.concatenate(y).astype(np.int64)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n_cal = 100
    X_cal, y_cal = X[:n_cal], y[:n_cal]
    X_te, y_te = X[n_cal:], y[n_cal:]

    # approved: cal-fit RBF bandwidth (non-exchangeable smoke test)
    ncm = RBFDensityNCM(allow_nonexchangeable=True)
    fcp = FullConformalPredictor(ncm, alpha=0.1)
    fcp.calibrate(X_cal, y_cal, all_classes=np.arange(3))
    res = fcp.predict(X_te, verbose=False)
    cov = float(np.mean([y_te[i] in res['prediction_sets'][i] for i in range(len(y_te))]))
    sz = float(np.mean(res['set_sizes']))
    print(f"  sigma chosen: {ncm.sigma:.3f}")
    print(f"  coverage: {cov:.3f} (target >=0.90), avg set size: {sz:.3f}")
    assert cov >= 0.85, f"Validity FAIL: coverage {cov:.3f}"
    print("  Smoke PASSED.")


# ------------------------------------------------------------------
# Data utilities
# ------------------------------------------------------------------
def load_cifar100():
    cal = torch.load("output/embeddings_cifar100.pt", weights_only=True)
    test = torch.load("output/embeddings_cifar100_test.pt", weights_only=True)
    unl = torch.load("output/embeddings_cifar100_unlabeled.pt", weights_only=True)
    # We use cal+test as the labeled pool and split it; unlabeled is for PCA/AE fit
    X = np.vstack([cal["embeddings"].numpy(), test["embeddings"].numpy()]).astype(np.float64)
    y = np.concatenate([cal["labels"].numpy(), test["labels"].numpy()]).astype(np.int64)
    X_unl = unl["embeddings"].numpy().astype(np.float64)
    return X, y, X_unl


def fit_pca(X_unl, dim):
    pca = PCA(n_components=dim, random_state=42).fit(X_unl)
    return pca


def fit_ae(X_unl, dim, seed=42):
    ae = EmbeddingAutoencoder(bottleneck_dim=dim, hidden_dim=512,
                               epochs=150, patience=15, seed=seed)
    ae.fit(X_unl.astype(np.float32))
    return ae


# ------------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------------
def run_one(X_cal_red, y_cal, X_te_red, y_te, ncm_name, alpha=0.1, all_classes=None):
    if ncm_name == "geodesic_topk_mean":
        # whiten=True preserves the historical (cal-fit) behavior after the
        # default flip to whiten=False; approved as non-exchangeable.
        ncm = GeodesicTopKMeanNCM(topk_same=True, topk_other=True,
                                  whiten=True, allow_nonexchangeable=True)
    elif ncm_name == "rbf_density":
        # Tuned: scale=0.2 ratio=True from rbf_bandwidth_sweep
        ncm = RBFDensityNCM(bandwidth_rule="median", sigma_scale=0.2, ratio=True,
                            allow_nonexchangeable=True)
    else:
        raise ValueError(ncm_name)
    fcp = FullConformalPredictor(ncm, alpha=alpha)
    fcp.calibrate(X_cal_red, y_cal, all_classes=all_classes)
    t0 = time.time()
    res = fcp.predict(X_te_red, verbose=False)
    elapsed = time.time() - t0
    cov = float(np.mean([y_te[i] in res['prediction_sets'][i] for i in range(len(y_te))]))
    sz = float(np.mean(res['set_sizes']))
    return cov, sz, elapsed


def experiment(cal_sizes=(400, 600, 800), n_trials=3, n_test=1000,
               pca_dims=(32, 128), ae_dim=32, alpha=0.1):
    X, y, X_unl = load_cifar100()
    classes = np.unique(y)
    print(f"Data: cal-pool {len(X)}, test-pool implicit, unlabeled {len(X_unl)}, K={len(classes)}")

    print("Fitting PCA(128) on unlabeled pool...")
    pcas = {d: fit_pca(X_unl, d) for d in pca_dims}
    print("Fitting AE-32 on unlabeled pool...")
    ae = fit_ae(X_unl, ae_dim)

    configs = [
        ("PCA-128 + geodesic", 128, "pca", "geodesic_topk_mean"),
        ("PCA-32  + geodesic", 32,  "pca", "geodesic_topk_mean"),
        ("AE-32   + geodesic", ae_dim, "ae", "geodesic_topk_mean"),
        ("PCA-128 + rbf",      128, "pca", "rbf_density"),
        ("PCA-32  + rbf",      32,  "pca", "rbf_density"),
        ("AE-32   + rbf",      ae_dim, "ae", "rbf_density"),
    ]

    results = {name: {"cov": {c: [] for c in cal_sizes},
                       "sz":  {c: [] for c in cal_sizes},
                       "t":   {c: [] for c in cal_sizes}} for name, *_ in configs}

    for trial in range(n_trials):
        seed = 100 + trial
        print(f"\n--- Trial {trial+1}/{n_trials} (seed {seed}) ---")
        for cal_size in cal_sizes:
            print(f"  cal={cal_size}")
            # Build split (uses same seed for fairness across NCMs)
            X_cal_full, y_cal, X_te_full, y_te = stratified_cal_test_split(
                X, y, cal_size=cal_size, test_size=n_test,
                balanced=True, random_state=seed)
            for name, dim, kind, ncm_name in configs:
                if kind == "pca":
                    proj = pcas[dim].transform(X_cal_full), pcas[dim].transform(X_te_full)
                else:
                    proj = ae.transform(X_cal_full.astype(np.float32)), \
                           ae.transform(X_te_full.astype(np.float32))
                X_cal_red, X_te_red = proj
                cov, sz, elapsed = run_one(
                    X_cal_red.astype(np.float64), y_cal,
                    X_te_red.astype(np.float64), y_te,
                    ncm_name, alpha=alpha, all_classes=classes)
                results[name]["cov"][cal_size].append(cov)
                results[name]["sz"][cal_size].append(sz)
                results[name]["t"][cal_size].append(elapsed)
                print(f"    {name:25s}  cov={cov:.3f}  sz={sz:6.2f}  t={elapsed:5.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY (mean over {n_trials} trials, alpha={alpha}, target cov >= {1-alpha:.2f})")
    print("=" * 70)
    print(f"{'config':28s}" + "".join(f"  cal={c}: cov/sz   " for c in cal_sizes))
    for name in [c[0] for c in configs]:
        line = f"{name:28s}"
        for c in cal_sizes:
            cov_m = np.mean(results[name]["cov"][c])
            sz_m = np.mean(results[name]["sz"][c])
            line += f"  {cov_m:.3f}/{sz_m:6.2f}    "
        print(line)
    return results


if __name__ == "__main__":
    smoke()
    print()
    experiment()
