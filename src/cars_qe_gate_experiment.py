"""The registered stanford_cars discriminating experiment for Theorem D3.

docs/dwt_denoise_theorem.md Section 6 registered this cell in advance:
cars has measured h_w = 0.461 -- BELOW the folklore homophily gate (~0.7,
predicts qe harms) but ABOVE the D3 gate (h* = 0.292, predicts qe GAINS,
d'-ratio 1.52). qe was never run on cars (its embeddings live in the
layers-file format that kept it out of the pool-repr-menu round). One run
decides theorem vs folklore.

Arms (same prototype-cosine split-CP scoring everywhere, LOO cal quantile):
  raw    L2-normed final-layer embeddings (context)
  wt     champion W/T stage: pool-fit PCA-128 + cluster-whiten, NO qe
  qe_wt  full DWT: qe denoise (k=10, a=3, pool) -> same W/T stage

Also measured: the empirical margin-level d'-ratio of qe in raw space
(fixed nearest-prototype pair axes, before vs after smoothing) -- the
direct check of D3's predicted 1.52 -- and kNN label homophily.

Splits: balanced cal + balanced test (repo default), paired across arms
(identical indices per trial). Deterministic given --seed.

Usage (from repo root; worktree needs the main repo's output dir):
    python src/cars_qe_gate_experiment.py \
        --emb_dir <...>/output/from_cluster/embeddings \
        --out_dir <...>/output/cars_qe_gate
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conformal_prediction import stratified_cal_test_split       # noqa: E402
from exchangeable_features import UnlabeledTransform             # noqa: E402
from dwt_score_histograms import (                               # noqa: E402
    l2n, qe_smooth, knn_homophily, scores_and_qhat)

# K=196: budgets are 2/4/8 shots per class (cal=200 would be 1 shot/class,
# where the LOO prototype degenerates -- scores collapse to -cos(x,x))
CAL_SIZES = [392, 784, 1568]
TEST_SIZE = 2000
N_TRIALS = 20
PCA_DIM = 128
N_CLUSTERS = 20

# D3 predictions for cars (docs/dwt_denoise_theorem.md Section 6, measured
# constants h_w=.461, beta=.827, k_eff=9.83, kappa=.585 -> h*=.292)
PREDICTED_DPRIME_RATIO = 1.52
H_STAR = 0.292
H_W = 0.461


def dprime_pair_ratios(X_raw, X_smooth, y, min_pts=5):
    """Empirical D3 check: for each class y, take its nearest-prototype
    partner c (cosine of raw class means), fix the raw pair axis
    v = (mu_y - mu_c)/||.||, and compute d' of the projections <x, v> for
    class-y vs class-c points before and after qe. Returns per-class ratios
    smoothed/raw (the quantity D3 predicts = 1.52 on cars)."""
    Xr = l2n(X_raw)
    Xs = l2n(X_smooth)
    classes = np.unique(y)
    mus = np.stack([Xr[y == c].mean(axis=0) for c in classes])
    cosm = l2n(mus) @ l2n(mus).T
    np.fill_diagonal(cosm, -np.inf)
    partner = classes[np.argmax(cosm, axis=1)]

    ratios, raw_dps, sm_dps = [], [], []
    for i, c_y in enumerate(classes):
        c_c = partner[i]
        v = mus[i] - mus[classes == c_c][0]
        v = v / (np.linalg.norm(v) + 1e-12)
        m_y, m_c = (y == c_y), (y == c_c)
        if m_y.sum() < min_pts or m_c.sum() < min_pts:
            continue
        out = []
        for Z in (Xr, Xs):
            gy, gc = Z[m_y] @ v, Z[m_c] @ v
            sd = np.sqrt(0.5 * (gy.var(ddof=1) + gc.var(ddof=1)))
            out.append((gy.mean() - gc.mean()) / (sd + 1e-12))
        raw_dps.append(out[0])
        sm_dps.append(out[1])
        ratios.append(out[1] / out[0])
    return np.array(ratios), np.array(raw_dps), np.array(sm_dps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--out_dir", default="output/cars_qe_gate")
    ap.add_argument("--n_trials", type=int, default=N_TRIALS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    lab = torch.load(os.path.join(
        args.emb_dir, "embeddings_stanford_cars_layers.pt"),
        map_location="cpu", weights_only=False)
    unl = torch.load(os.path.join(
        args.emb_dir, "embeddings_stanford_cars_unlabeled_layers.pt"),
        map_location="cpu", weights_only=False)
    X = lab["final"].numpy().astype(np.float64)
    y = lab["labels"].numpy().astype(int)
    U = unl["final"].numpy().astype(np.float64)
    print(f"labeled {X.shape}, pool {U.shape}, K={len(np.unique(y))}",
          flush=True)

    h = knn_homophily(X, y)
    print(f"labeled kNN homophily h(k=10) = {h:.3f}", flush=True)

    # ---- arm representations, computed once (smoothing is deterministic) ----
    print("preparing arms ...", flush=True)
    Z_arm = {"raw": l2n(X)}
    tr = UnlabeledTransform(pca_dim=PCA_DIM, whiten="cluster",
                            n_clusters=N_CLUSTERS,
                            random_state=args.seed).fit(U)
    Z_arm["wt"] = tr.transform(X)

    U_s = qe_smooth(U, U, exclude_self=True)
    X_s = qe_smooth(X, U)
    tr_s = UnlabeledTransform(pca_dim=PCA_DIM, whiten="cluster",
                              n_clusters=N_CLUSTERS,
                              random_state=args.seed).fit(U_s)
    Z_arm["qe_wt"] = tr_s.transform(X_s)

    # ---- D3's own quantity: margin d'-ratio in raw space ----
    ratios, raw_dps, sm_dps = dprime_pair_ratios(X, X_s, y)
    print(f"empirical d'-ratio: mean {ratios.mean():.3f}  "
          f"median {np.median(ratios):.3f}  "
          f"frac>1 {(ratios > 1).mean():.2f}  (predicted {PREDICTED_DPRIME_RATIO})",
          flush=True)

    # ---- CP trials: paired balanced splits across arms ----
    idx_all = np.arange(len(y))[:, None].astype(np.float64)
    results = {arm: {c: {"cov": [], "size": []} for c in CAL_SIZES}
               for arm in Z_arm}
    for cal_size in CAL_SIZES:
        for t in range(args.n_trials):
            ic, yc, it, yt = stratified_cal_test_split(
                idx_all, y, cal_size=cal_size, test_size=TEST_SIZE,
                random_state=args.seed + 1000 * t + cal_size)
            ic = ic[:, 0].astype(int)
            it = it[:, 0].astype(int)
            for arm, Z in Z_arm.items():
                st = scores_and_qhat(Z[ic], yc, Z[it], yt)
                results[arm][cal_size]["cov"].append(st["coverage"])
                results[arm][cal_size]["size"].append(st["avg_size"])
        line = f"cal={cal_size}: "
        for arm in ["raw", "wt", "qe_wt"]:
            s = results[arm][cal_size]
            line += (f"{arm} sz {np.mean(s['size']):.2f} "
                     f"cov {np.mean(s['cov']):.3f} | ")
        print(line, flush=True)

    # ---- summarize ----
    out = {
        "dataset": "stanford_cars", "alpha": 0.1,
        "n_trials": args.n_trials, "cal_sizes": CAL_SIZES,
        "test_size": TEST_SIZE, "pca_dim": PCA_DIM,
        "homophily_labeled_k10": float(h),
        "registered_prediction": {
            "h_w": H_W, "h_star": H_STAR,
            "predicted_dprime_ratio": PREDICTED_DPRIME_RATIO,
            "folklore_gate": 0.7,
            "d3_predicts": "gain", "folklore_predicts": "harm"},
        "dprime_empirical": {
            "ratios": [float(r) for r in ratios],
            "mean_ratio": float(ratios.mean()),
            "median_ratio": float(np.median(ratios)),
            "frac_pairs_improved": float((ratios > 1).mean()),
            "n_pairs": int(len(ratios)),
            "mean_raw_dprime": float(raw_dps.mean()),
            "mean_smoothed_dprime": float(sm_dps.mean())},
        "arms": {},
    }
    for arm in Z_arm:
        out["arms"][arm] = {}
        for cal_size in CAL_SIZES:
            s = results[arm][cal_size]
            out["arms"][arm][str(cal_size)] = {
                "coverage_mean": float(np.mean(s["cov"])),
                "coverage_sd": float(np.std(s["cov"])),
                "size_mean": float(np.mean(s["size"])),
                "size_sd": float(np.std(s["size"])),
                "sizes": [float(v) for v in s["size"]],
                "coverages": [float(v) for v in s["cov"]]}
    # paired qe-vs-no-qe deltas (the verdict)
    out["verdict"] = {}
    for cal_size in CAL_SIZES:
        d = (np.array(results["qe_wt"][cal_size]["size"])
             - np.array(results["wt"][cal_size]["size"]))
        rel = d / np.array(results["wt"][cal_size]["size"])
        out["verdict"][str(cal_size)] = {
            "paired_delta_mean": float(d.mean()),
            "paired_delta_se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "relative_change_mean": float(rel.mean()),
            "qe_gains": bool(d.mean() < 0)}

    path = os.path.join(args.out_dir, "cars_qe_gate.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("arms",)}, indent=2))
    print(f"results -> {path}")


if __name__ == "__main__":
    main()
