"""Iter 4: 5-trial confirmation of AE-128 + whitened-RBF vs PCA-128 + geodesic
across cal in {400, 600, 800}. Locks in the iter3 marginal win or reveals it
was noise.

Bandwidth: scale=0.2 (winner of iter3 tune for AE-128 + whitening).
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


def main():
    X, y, Xu = load()
    classes = np.unique(y)

    print("Pre-fitting PCA-128 and AE-128 on 10K unlabeled...")
    pca128 = PCA(n_components=128, random_state=42).fit(Xu)
    ae128 = EmbeddingAutoencoder(bottleneck_dim=128, hidden_dim=512,
                                   epochs=150, patience=15, seed=42)
    ae128.fit(Xu.astype(np.float32))

    cal_sizes = (400, 600, 800)
    n_trials = 5
    n_test = 1000

    cfg = [
        ("PCA-128 + geodesic (BASE)", "pca", "geo", None),
        ("AE-128  + wRBF (scale=0.20)", "ae", "wrbf", 0.20),
        ("AE-128  + wRBF (scale=0.15)", "ae", "wrbf", 0.15),
        ("AE-128  + wRBF (scale=0.30)", "ae", "wrbf", 0.30),
    ]
    results = {name: {c: {"cov": [], "sz": [], "t": []} for c in cal_sizes}
                for name, *_ in cfg}

    for trial in range(n_trials):
        seed = 500 + trial
        print(f"\n--- Trial {trial+1}/{n_trials} (seed {seed}) ---")
        for cal_size in cal_sizes:
            Xc_full, yc, Xt_full, yt = stratified_cal_test_split(
                X, y, cal_size=cal_size, test_size=n_test,
                balanced=True, random_state=seed)
            proj_pca = (pca128.transform(Xc_full), pca128.transform(Xt_full))
            proj_ae = (ae128.transform(Xc_full.astype(np.float32)),
                        ae128.transform(Xt_full.astype(np.float32)))
            for name, kind, ncm_kind, scale in cfg:
                Xc, Xt = proj_pca if kind == "pca" else proj_ae
                Xc = Xc.astype(np.float64); Xt = Xt.astype(np.float64)
                if ncm_kind == "geo":
                    ncm = GeodesicTopKMeanNCM(topk_same=True, topk_other=True)
                else:
                    ncm = RBFDensityNCM(bandwidth_rule="median",
                                         sigma_scale=scale, ratio=True, whiten=True)
                cov, sz, t = quick_run(Xc, yc, Xt, yt, ncm, classes)
                results[name][cal_size]["cov"].append(cov)
                results[name][cal_size]["sz"].append(sz)
                results[name][cal_size]["t"].append(t)
                print(f"  cal={cal_size} {name:30s} cov={cov:.3f} sz={sz:5.2f} t={t:4.1f}s")

    # Summary with stderr
    print("\n" + "=" * 75)
    print(f"SUMMARY (mean ± std/sqrt({n_trials}), {n_trials} trials, n_test={n_test})")
    print("=" * 75)
    header = f"{'config':30s} | "
    for c in cal_sizes:
        header += f"cal={c}: cov | sz ±se      "
    print(header)
    for name, *_ in cfg:
        line = f"{name:30s} | "
        for c in cal_sizes:
            covs = results[name][c]["cov"]; szs = results[name][c]["sz"]
            cov_m = np.mean(covs)
            sz_m = np.mean(szs); sz_se = np.std(szs, ddof=1) / np.sqrt(n_trials)
            line += f"{cov_m:.3f}    | {sz_m:5.2f}±{sz_se:.2f}  "
        print(line)


if __name__ == "__main__":
    main()
