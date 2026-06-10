"""Iter 6: tighten the cal=800 result with multi-seed bandwidth tune + 10-trial
paired confirmation. Goal: demonstrate AE-128 + wSymRBF beats BASE at both
cal=600 and cal=800 with statistical significance.

Paired analysis (same seed for both configs) cuts the trial-to-trial variance
that confounded iter4 vs iter5.
"""
import sys, time
import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel

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


def tune_sym_multi(X, y, classes, ae,
                     cal_size, seeds=(700, 701, 702, 703),
                     scales=(0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30)):
    print(f"  Bandwidth tune for AE-128 + wSymRBF at cal={cal_size}:")
    seed_scales = {s: [] for s in scales}
    for seed in seeds:
        Xc_full, yc, Xt_full, yt = stratified_cal_test_split(
            X, y, cal_size=cal_size, test_size=500,
            balanced=True, random_state=seed)
        Xc = ae.transform(Xc_full.astype(np.float32)).astype(np.float64)
        Xt = ae.transform(Xt_full.astype(np.float32)).astype(np.float64)
        for s in scales:
            ncm = RBFDensityNCM(bandwidth_rule="median", sigma_scale=s,
                                  ratio=True, whiten=True, same_agg="sum")
            cov, sz, _ = quick_run(Xc, yc, Xt, yt, ncm, classes)
            if cov >= 0.88:
                seed_scales[s].append(sz)
    best_s, best_sz = None, np.inf
    for s, szs in seed_scales.items():
        if len(szs) >= len(seeds) - 1:
            m = np.mean(szs)
            print(f"    scale={s:.2f}: sz={m:.3f} (n={len(szs)})")
            if m < best_sz:
                best_s, best_sz = s, m
    print(f"    >> best: scale={best_s}")
    return best_s


def main():
    X, y, Xu = load()
    classes = np.unique(y)

    print("Pre-fitting PCA-128 + AE-128 on 10K unlabeled...")
    pca128 = PCA(n_components=128, random_state=42).fit(Xu)
    ae128 = EmbeddingAutoencoder(bottleneck_dim=128, hidden_dim=512,
                                   epochs=150, patience=15, seed=42)
    ae128.fit(Xu.astype(np.float32))

    # Per-cal bandwidth tune
    print("\n=== Per-cal-size bandwidth tune (4 seeds each) ===")
    tuned = {}
    for cal_size in [600, 800]:
        tuned[cal_size] = tune_sym_multi(X, y, classes, ae128, cal_size)

    # 10-trial paired confirmation at cal=600 and cal=800
    print("\n=== 10-trial PAIRED confirmation ===")
    cal_sizes = (600, 800)
    n_trials = 10
    n_test = 1000

    results = {c: {"base_sz": [], "ae_sz": [], "base_cov": [], "ae_cov": []}
                for c in cal_sizes}

    for trial in range(n_trials):
        seed = 800 + trial
        print(f"\n-- Trial {trial+1}/{n_trials} (seed {seed}) --")
        for cal_size in cal_sizes:
            Xc_full, yc, Xt_full, yt = stratified_cal_test_split(
                X, y, cal_size=cal_size, test_size=n_test,
                balanced=True, random_state=seed)
            Xc_pca = pca128.transform(Xc_full).astype(np.float64)
            Xt_pca = pca128.transform(Xt_full).astype(np.float64)
            Xc_ae = ae128.transform(Xc_full.astype(np.float32)).astype(np.float64)
            Xt_ae = ae128.transform(Xt_full.astype(np.float32)).astype(np.float64)

            base_ncm = GeodesicTopKMeanNCM(topk_same=True, topk_other=True)
            cov_b, sz_b, _ = quick_run(Xc_pca, yc, Xt_pca, yt, base_ncm, classes)

            ae_ncm = RBFDensityNCM(bandwidth_rule="median",
                                     sigma_scale=tuned[cal_size], ratio=True,
                                     whiten=True, same_agg="sum")
            cov_a, sz_a, _ = quick_run(Xc_ae, yc, Xt_ae, yt, ae_ncm, classes)

            results[cal_size]["base_sz"].append(sz_b)
            results[cal_size]["ae_sz"].append(sz_a)
            results[cal_size]["base_cov"].append(cov_b)
            results[cal_size]["ae_cov"].append(cov_a)
            print(f"  cal={cal_size}: BASE sz={sz_b:.2f} cov={cov_b:.3f} | "
                  f"AE sz={sz_a:.2f} cov={cov_a:.3f} | diff={sz_a-sz_b:+.3f}")

    # Summary with paired t-test
    print("\n" + "=" * 75)
    print(f"PAIRED SUMMARY ({n_trials} trials, same seed splits per pair)")
    print("=" * 75)
    for c in cal_sizes:
        bs = np.array(results[c]["base_sz"])
        ascr = np.array(results[c]["ae_sz"])
        bc = np.array(results[c]["base_cov"])
        ac = np.array(results[c]["ae_cov"])
        diffs = ascr - bs
        t_stat, p_val = ttest_rel(ascr, bs)
        print(f"\n  cal={c} (sigma_scale={tuned[c]})")
        print(f"    BASE  sz: {bs.mean():.3f} ± {bs.std(ddof=1)/np.sqrt(n_trials):.3f}  cov={bc.mean():.3f}")
        print(f"    AE+rbf sz: {ascr.mean():.3f} ± {ascr.std(ddof=1)/np.sqrt(n_trials):.3f}  cov={ac.mean():.3f}")
        print(f"    Paired diff (AE - BASE): {diffs.mean():+.3f} ± {diffs.std(ddof=1)/np.sqrt(n_trials):.3f}")
        print(f"    Paired t-test: t={t_stat:.3f}, p={p_val:.4f}, "
              f"{'AE WINS' if (p_val < 0.05 and diffs.mean() < 0) else 'BASE WINS' if (p_val < 0.05) else 'TIE'}")


if __name__ == "__main__":
    main()
