"""Sweep RBF bandwidth rules on PCA-32 / AE-32 to find a working sigma."""
import sys, time
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, "src")
from conformal_prediction import (
    RBFDensityNCM, FullConformalPredictor, stratified_cal_test_split,
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


def run_one(X_cal_r, y_cal, X_te_r, y_te, rule, scale, ratio, classes):
    ncm = RBFDensityNCM(bandwidth_rule=rule, sigma_scale=scale, ratio=ratio)
    fcp = FullConformalPredictor(ncm, alpha=0.1)
    fcp.calibrate(X_cal_r, y_cal, all_classes=classes)
    t0 = time.time()
    res = fcp.predict(X_te_r, verbose=False)
    elapsed = time.time() - t0
    cov = float(np.mean([y_te[i] in res['prediction_sets'][i] for i in range(len(y_te))]))
    sz = float(np.mean(res['set_sizes']))
    return cov, sz, ncm.sigma, elapsed


def main():
    X, y, Xu = load()
    classes = np.unique(y)

    pca32 = PCA(n_components=32, random_state=42).fit(Xu)
    print("Fitting AE-32...")
    ae32 = EmbeddingAutoencoder(bottleneck_dim=32, hidden_dim=512,
                                 epochs=120, patience=15, seed=42)
    ae32.fit(Xu.astype(np.float32))

    X_cal, y_cal, X_te, y_te = stratified_cal_test_split(
        X, y, cal_size=600, test_size=200, balanced=True, random_state=100)

    for label, Xc, Xt in [
        ("PCA-32", pca32.transform(X_cal), pca32.transform(X_te)),
        ("AE-32",  ae32.transform(X_cal.astype(np.float32)),
                   ae32.transform(X_te.astype(np.float32))),
    ]:
        Xc = Xc.astype(np.float64); Xt = Xt.astype(np.float64)
        print(f"\n=== {label} | cal=600 test=200 ===")
        header = f"{'config':50s} | cov   | sz    | sigma | t"
        print(header)
        print("-" * len(header))
        for rule in ["median", "knn", "within_class_median"]:
            for scale in [1.0, 0.5, 0.3, 0.2, 0.1]:
                for ratio in [False, True]:
                    cov, sz, sg, t = run_one(Xc, y_cal, Xt, y_te,
                                              rule, scale, ratio, classes)
                    name = f"rule={rule}  scale={scale}  ratio={ratio}"
                    print(f"{name:50s} | {cov:.3f} | {sz:5.2f} | {sg:5.2f} | {t:.1f}s")


if __name__ == "__main__":
    main()
