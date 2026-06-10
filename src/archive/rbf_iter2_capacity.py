"""Iter 2: push AE bottleneck higher (64, 128) + per-dim bandwidth retune.

If AE-128 + RBF beats PCA-128 + geodesic, the alignment hypothesis is fully
proven and we have a paper-quality result.
"""
import sys, time
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, "src")
from conformal_prediction import (
    RBFDensityNCM, GeodesicTopKMeanNCM, FullConformalPredictor,
    stratified_cal_test_split,
)
from autoencoder_utils import EmbeddingAutoencoder


def load():
    cal = torch.load("output/embeddings_cifar100.pt", weights_only=True)
    test = torch.load("output/embeddings_cifar100_test.pt", weights_only=True)
    unl = torch.load("output/embeddings_cifar100_unlabeled.pt", weights_only=True)
    X = np.vstack([cal["embeddings"].numpy(), test["embeddings"].numpy()]).astype(np.float64)
    y = np.concatenate([cal["labels"].numpy(), test["labels"].numpy()]).astype(np.int64)
    Xu = unl["embeddings"].numpy().astype(np.float64)
    return X, y, Xu


def quick_run(Xc, yc, Xt, yt, ncm, classes, alpha=0.1):
    fcp = FullConformalPredictor(ncm, alpha=alpha)
    fcp.calibrate(Xc, yc, all_classes=classes)
    t0 = time.time()
    res = fcp.predict(Xt, verbose=False)
    elapsed = time.time() - t0
    cov = float(np.mean([yt[i] in res['prediction_sets'][i] for i in range(len(yt))]))
    sz = float(np.mean(res['set_sizes']))
    return cov, sz, elapsed


def tune_sigma(Xc, yc, Xt, yt, classes, scales=(0.1, 0.15, 0.2, 0.3, 0.4, 0.5)):
    """One-trial sigma_scale tune. Returns best scale by sz subject to cov>=0.88."""
    best = (None, np.inf, None, None)
    for s in scales:
        ncm = RBFDensityNCM(bandwidth_rule="median", sigma_scale=s, ratio=True)
        cov, sz, t = quick_run(Xc, yc, Xt, yt, ncm, classes)
        if cov >= 0.88 and sz < best[1]:
            best = (s, sz, cov, t)
    return best


def main():
    X, y, Xu = load()
    classes = np.unique(y)

    print("Pre-fitting reducers on the 10K unlabeled pool...")
    pca128 = PCA(n_components=128, random_state=42).fit(Xu)
    ae64 = EmbeddingAutoencoder(bottleneck_dim=64, hidden_dim=512,
                                  epochs=150, patience=15, seed=42)
    ae64.fit(Xu.astype(np.float32))
    ae128 = EmbeddingAutoencoder(bottleneck_dim=128, hidden_dim=512,
                                   epochs=150, patience=15, seed=42)
    ae128.fit(Xu.astype(np.float32))

    # Tune sigma_scale at cal=800 (target regime) for AE-64, AE-128, PCA-128
    print("\n=== Bandwidth re-tune at cal=800 (single trial seed=200) ===")
    Xc_full, yc, Xt_full, yt = stratified_cal_test_split(
        X, y, cal_size=800, test_size=500, balanced=True, random_state=200)

    configs_to_tune = {
        "AE-64": (ae64.transform(Xc_full.astype(np.float32)),
                   ae64.transform(Xt_full.astype(np.float32))),
        "AE-128": (ae128.transform(Xc_full.astype(np.float32)),
                    ae128.transform(Xt_full.astype(np.float32))),
        "PCA-128": (pca128.transform(Xc_full), pca128.transform(Xt_full)),
    }
    tuned = {}
    for name, (Xc, Xt) in configs_to_tune.items():
        Xc = Xc.astype(np.float64); Xt = Xt.astype(np.float64)
        best = tune_sigma(Xc, yc, Xt, yt, classes)
        print(f"  {name:8s} best scale={best[0]} sz={best[1]:.2f} cov={best[2]:.3f}")
        tuned[name] = best[0]

    # 3-trial comparison at cal=800
    print("\n=== 3-trial comparison at cal=800 (n_test=1000) ===")
    results = {}
    cfg_list = [
        ("PCA-128 + geodesic", "pca128", "geodesic"),
        ("AE-64   + geodesic", "ae64",   "geodesic"),
        ("AE-128  + geodesic", "ae128",  "geodesic"),
        ("PCA-128 + rbf",      "pca128", "rbf"),
        ("AE-64   + rbf",      "ae64",   "rbf"),
        ("AE-128  + rbf",      "ae128",  "rbf"),
    ]
    for name, *_ in cfg_list:
        results[name] = {"cov": [], "sz": [], "t": []}

    for trial in range(3):
        seed = 300 + trial
        print(f"\n-- Trial {trial+1}/3 (seed {seed}) --")
        Xc_full, yc, Xt_full, yt = stratified_cal_test_split(
            X, y, cal_size=800, test_size=1000, balanced=True, random_state=seed)
        proj_pca = (pca128.transform(Xc_full), pca128.transform(Xt_full))
        proj_ae64 = (ae64.transform(Xc_full.astype(np.float32)),
                     ae64.transform(Xt_full.astype(np.float32)))
        proj_ae128 = (ae128.transform(Xc_full.astype(np.float32)),
                      ae128.transform(Xt_full.astype(np.float32)))
        proj_map = {"pca128": proj_pca, "ae64": proj_ae64, "ae128": proj_ae128}
        for name, key, kind in cfg_list:
            Xc, Xt = proj_map[key]
            Xc = Xc.astype(np.float64); Xt = Xt.astype(np.float64)
            if kind == "geodesic":
                ncm = GeodesicTopKMeanNCM(topk_same=True, topk_other=True)
            else:
                # Use tuned scale (fallback 0.2)
                tkey = {"pca128": "PCA-128", "ae64": "AE-64", "ae128": "AE-128"}[key]
                scale = tuned.get(tkey, 0.2) or 0.2
                ncm = RBFDensityNCM(bandwidth_rule="median",
                                     sigma_scale=scale, ratio=True)
            cov, sz, t = quick_run(Xc, yc, Xt, yt, ncm, classes)
            results[name]["cov"].append(cov)
            results[name]["sz"].append(sz)
            results[name]["t"].append(t)
            print(f"  {name:22s} cov={cov:.3f} sz={sz:5.2f} t={t:5.1f}s")

    print("\n" + "=" * 60)
    print(f"SUMMARY cal=800, mean over 3 trials")
    print("=" * 60)
    print(f"{'config':25s} | cov   | sz   ")
    for name, *_ in cfg_list:
        cov_m = np.mean(results[name]["cov"])
        sz_m = np.mean(results[name]["sz"])
        print(f"{name:25s} | {cov_m:.3f} | {sz_m:5.2f}")


if __name__ == "__main__":
    main()
