"""Iter 3: whitened RBF NCM. Combines pooled-within-class whitening (geodesic's
structural advantage) with the RBF kernel's nonlinearity. Predicted to close the
8% gap left after iter1/2.

Configs:
  PCA-128 + geodesic                  baseline
  AE-32   + whitened-rbf              candidate
  AE-128  + whitened-rbf              candidate
  PCA-128 + whitened-rbf              control (whitening + RBF on linear features)
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


def tune_sigma(Xc, yc, Xt, yt, classes, whiten,
               scales=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)):
    best = (None, np.inf, None)
    for s in scales:
        ncm = RBFDensityNCM(bandwidth_rule="median", sigma_scale=s,
                              ratio=True, whiten=whiten)
        cov, sz, t = quick_run(Xc, yc, Xt, yt, ncm, classes)
        if cov >= 0.88 and sz < best[1]:
            best = (s, sz, cov)
    return best


def main():
    X, y, Xu = load()
    classes = np.unique(y)

    print("Pre-fitting reducers on 10K unlabeled...")
    pca128 = PCA(n_components=128, random_state=42).fit(Xu)
    ae32 = EmbeddingAutoencoder(bottleneck_dim=32, hidden_dim=512,
                                  epochs=150, patience=15, seed=42)
    ae32.fit(Xu.astype(np.float32))
    ae128 = EmbeddingAutoencoder(bottleneck_dim=128, hidden_dim=512,
                                   epochs=150, patience=15, seed=42)
    ae128.fit(Xu.astype(np.float32))

    # Bandwidth re-tune at cal=800 (whitened space has different scale!)
    print("\n=== Bandwidth re-tune for whitened RBF at cal=800 (seed=200) ===")
    Xc_full, yc, Xt_full, yt = stratified_cal_test_split(
        X, y, cal_size=800, test_size=500, balanced=True, random_state=200)

    configs_to_tune = {
        "AE-32":   (ae32.transform(Xc_full.astype(np.float32)),
                     ae32.transform(Xt_full.astype(np.float32))),
        "AE-128":  (ae128.transform(Xc_full.astype(np.float32)),
                     ae128.transform(Xt_full.astype(np.float32))),
        "PCA-128": (pca128.transform(Xc_full), pca128.transform(Xt_full)),
    }
    tuned = {}
    for name, (Xc, Xt) in configs_to_tune.items():
        Xc = Xc.astype(np.float64); Xt = Xt.astype(np.float64)
        best = tune_sigma(Xc, yc, Xt, yt, classes, whiten=True)
        print(f"  {name:8s} best scale={best[0]} sz={best[1]:.2f} cov={best[2]:.3f}")
        tuned[name] = best[0]

    # 3-trial comparison
    print("\n=== 3-trial comparison at cal=800 (n_test=1000) ===")
    cfg_list = [
        ("PCA-128 + geodesic (BASE)", "pca128", "geodesic", None),
        ("PCA-128 + whitened-rbf",    "pca128", "wrbf",     "PCA-128"),
        ("AE-32   + whitened-rbf",    "ae32",   "wrbf",     "AE-32"),
        ("AE-128  + whitened-rbf",    "ae128",  "wrbf",     "AE-128"),
    ]
    results = {n: {"cov": [], "sz": [], "t": []} for n, *_ in cfg_list}

    for trial in range(3):
        seed = 400 + trial
        print(f"\n-- Trial {trial+1}/3 (seed {seed}) --")
        Xc_full, yc, Xt_full, yt = stratified_cal_test_split(
            X, y, cal_size=800, test_size=1000, balanced=True, random_state=seed)
        proj_map = {
            "pca128": (pca128.transform(Xc_full), pca128.transform(Xt_full)),
            "ae32":   (ae32.transform(Xc_full.astype(np.float32)),
                        ae32.transform(Xt_full.astype(np.float32))),
            "ae128":  (ae128.transform(Xc_full.astype(np.float32)),
                        ae128.transform(Xt_full.astype(np.float32))),
        }
        for name, key, kind, tname in cfg_list:
            Xc, Xt = proj_map[key]
            Xc = Xc.astype(np.float64); Xt = Xt.astype(np.float64)
            if kind == "geodesic":
                ncm = GeodesicTopKMeanNCM(topk_same=True, topk_other=True)
            else:
                scale = tuned.get(tname, 0.2) or 0.2
                ncm = RBFDensityNCM(bandwidth_rule="median",
                                     sigma_scale=scale, ratio=True, whiten=True)
            cov, sz, t = quick_run(Xc, yc, Xt, yt, ncm, classes)
            results[name]["cov"].append(cov)
            results[name]["sz"].append(sz)
            results[name]["t"].append(t)
            print(f"  {name:30s} cov={cov:.3f} sz={sz:5.2f} t={t:5.1f}s")

    print("\n" + "=" * 60)
    print(f"SUMMARY cal=800, mean over 3 trials")
    print("=" * 60)
    print(f"{'config':32s} | cov   | sz   ")
    for name, *_ in cfg_list:
        cov_m = np.mean(results[name]["cov"])
        sz_m = np.mean(results[name]["sz"])
        print(f"{name:32s} | {cov_m:.3f} | {sz_m:5.2f}")


if __name__ == "__main__":
    main()
