"""Iter 5: ASYMMETRIC whitened RBF (1-NN num + soft denom + ratio + whiten).

Mirrors the geodesic_topk_asym pattern (which gave +39% over geodesic_topk_mean
in the linear case). Goal: AE-128 + this NCM beats PCA-128+geodesic_topk_mean
ACROSS ALL cal sizes (cal=400, 600, 800), not just at cal=600.

Workflow:
1. Bandwidth tune at cal=800 (multi-seed average so we don't overfit one split).
2. 5-trial confirm at cal=400, 600, 800.
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


def tune_sigma_multiseed(X, y, classes, transformer, kind, cal_size=800,
                           test_size=500, seeds=(200, 201, 202),
                           scales=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40),
                           ncm_kwargs=None):
    """Bandwidth tune averaged over multiple seeds."""
    ncm_kwargs = ncm_kwargs or {}
    seed_scales = {s: [] for s in scales}
    for seed in seeds:
        Xc_full, yc, Xt_full, yt = stratified_cal_test_split(
            X, y, cal_size=cal_size, test_size=test_size,
            balanced=True, random_state=seed)
        if kind == "ae":
            Xc = transformer.transform(Xc_full.astype(np.float32)).astype(np.float64)
            Xt = transformer.transform(Xt_full.astype(np.float32)).astype(np.float64)
        else:
            Xc = transformer.transform(Xc_full).astype(np.float64)
            Xt = transformer.transform(Xt_full).astype(np.float64)
        for s in scales:
            ncm = RBFDensityNCM(bandwidth_rule="median", sigma_scale=s, **ncm_kwargs)
            cov, sz, _ = quick_run(Xc, yc, Xt, yt, ncm, classes)
            if cov >= 0.88:
                seed_scales[s].append(sz)
    best_s, best_sz = None, np.inf
    print(f"    Tune (mean over {len(seeds)} seeds, cal=800, test={test_size}):")
    for s, szs in seed_scales.items():
        if len(szs) == len(seeds):
            m = np.mean(szs); std = np.std(szs, ddof=1) if len(szs) > 1 else 0
            print(f"      scale={s:.2f}: sz={m:.3f}±{std:.3f} ({len(szs)} valid)")
            if m < best_sz:
                best_s, best_sz = s, m
    return best_s


def main():
    X, y, Xu = load()
    classes = np.unique(y)

    print("Pre-fitting PCA-128 and AE-128 on 10K unlabeled...")
    pca128 = PCA(n_components=128, random_state=42).fit(Xu)
    ae128 = EmbeddingAutoencoder(bottleneck_dim=128, hidden_dim=512,
                                   epochs=150, patience=15, seed=42)
    ae128.fit(Xu.astype(np.float32))

    # Bandwidth tune for AE-128 + whitened asym RBF (multi-seed!)
    print("\n=== Bandwidth tune: AE-128 + whitened asym RBF ===")
    best_ae = tune_sigma_multiseed(
        X, y, classes, ae128, "ae",
        ncm_kwargs=dict(ratio=True, whiten=True, same_agg="max"))
    print(f"  >> best scale for AE-128 wAsymRBF: {best_ae}")

    print("\n=== Bandwidth tune: PCA-128 + whitened asym RBF (control) ===")
    best_pca = tune_sigma_multiseed(
        X, y, classes, pca128, "pca",
        ncm_kwargs=dict(ratio=True, whiten=True, same_agg="max"))
    print(f"  >> best scale for PCA-128 wAsymRBF: {best_pca}")

    # 5-trial comparison at cal=400, 600, 800
    print("\n=== 5-trial confirmation ===")
    cal_sizes = (400, 600, 800)
    n_trials = 5
    n_test = 1000

    cfg = [
        ("PCA-128 + geodesic_topk_mean (BASE)", "pca", "geo", None),
        ("AE-128  + wAsymRBF", "ae", "wrbf_asym", best_ae),
        ("PCA-128 + wAsymRBF", "pca", "wrbf_asym", best_pca),
        ("AE-128  + wSymRBF (prev)",  "ae", "wrbf_sym", 0.20),
    ]
    results = {n: {c: {"cov": [], "sz": []} for c in cal_sizes} for n, *_ in cfg}

    for trial in range(n_trials):
        seed = 600 + trial
        print(f"\n-- Trial {trial+1}/{n_trials} (seed {seed}) --")
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
                elif ncm_kind == "wrbf_asym":
                    ncm = RBFDensityNCM(bandwidth_rule="median",
                                         sigma_scale=scale, ratio=True,
                                         whiten=True, same_agg="max")
                elif ncm_kind == "wrbf_sym":
                    ncm = RBFDensityNCM(bandwidth_rule="median",
                                         sigma_scale=scale, ratio=True,
                                         whiten=True, same_agg="sum")
                cov, sz, t = quick_run(Xc, yc, Xt, yt, ncm, classes)
                results[name][cal_size]["cov"].append(cov)
                results[name][cal_size]["sz"].append(sz)
                print(f"  cal={cal_size} {name:42s} cov={cov:.3f} sz={sz:5.2f} t={t:4.1f}s")

    print("\n" + "=" * 90)
    print(f"SUMMARY 5 trials, target cov >= 0.90")
    print("=" * 90)
    print(f"{'config':42s} | " + "".join(f"cal={c}: cov | sz±se        " for c in cal_sizes))
    for name, *_ in cfg:
        line = f"{name:42s} | "
        for c in cal_sizes:
            covs = results[name][c]["cov"]; szs = results[name][c]["sz"]
            cov_m = np.mean(covs)
            sz_m = np.mean(szs); sz_se = np.std(szs, ddof=1) / np.sqrt(n_trials)
            line += f"{cov_m:.3f}    | {sz_m:5.2f}±{sz_se:.2f}    "
        print(line)


if __name__ == "__main__":
    main()
