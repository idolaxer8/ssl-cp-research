"""Goal 5 (ICLR 2027 run-up): does the DWT label-free gate generalize ACROSS
SSL backbones?

The DWT deploy rule (docs/dwt_denoise_theorem.md) was fit on DINOv2 features:
qe pool-neighbor denoise HELPS when the neighborhood is homophilous and HARMS
below the break-even homophily h*. The paper claim we need for a backbone
table is NOT "which backbone wins" -- it is "the same label-free dials
(homophily h, participation ratio PR, D3 d'-ratio) still pick the right
transform/gate on backbones whose geometry is nothing like DINOv2's".

This script runs, for each (backbone x dataset) cell:

  DIALS (regime fingerprint, measured on labeled data + labeled pool carve):
    h        raw kNN label homophily h(k=10)            (regime pole indicator)
    PR       participation ratio of the raw pool cov     (LABEL-FREE)
    h_w      qe-weighted neighborhood homophily          (D3 gate input)
    h_star   D3 break-even homophily                     (D3 gate threshold)
    d'ratio  D3 predicted standardized-margin change     (>1 gain / <1 harm)

  ARMS (identical prototype-cosine split-CP scoring, LOO cal quantile; arms
  precomputed on the FULL set then split by shared indices -> paired trials):
    raw      L2-normed final-layer embeddings            (context)
    wt       champion W/T: pool-fit PCA-128 + cluster-whiten, NO qe
    qe_wt    full DWT: qe denoise (k=10,a=3,pool) -> same W/T stage

  VERDICT per cell:
    actual   sign of paired (qe_wt - wt) set-size delta at the largest cal
    D3       predicts gain iff d'ratio > 1 (equivalently h_w >= h_star)
    folklore predicts gain iff h >= 0.7
    -> we tabulate whether each gate's prediction MATCHES the actual sign,
       per backbone, so the table shows the D3 dial transfers off DINOv2.

Self-contained: only depends on conformal_prediction.stratified_cal_test_split
and exchangeable_features.UnlabeledTransform (both on main). qe_smooth /
knn_homophily / scores_and_qhat / the D3 constant measurement are inlined
verbatim from the theory-dwt worktree (dwt_score_histograms.py,
dwt_gate_constants.py) so numbers are comparable to the DINOv2 runs.

Embedding file naming (output/from_cluster/embeddings/):
    dinov2 : embeddings_<ds>.pt            embeddings_<ds>_unlabeled.pt
    mae    : embeddings_<ds>_mae-base.pt   embeddings_<ds>_unlabeled_mae-base.pt
    clip   : embeddings_<ds>_clip-base.pt  embeddings_<ds>_unlabeled_clip-base.pt

Usage (from repo root, on the cluster):
    python src/backbone_dwt_experiment.py \
        --emb_dir output/from_cluster/embeddings \
        --out_dir output/backbone_dwt \
        --backbones dinov2 mae clip \
        --datasets cifar100 aircraft
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from conformal_prediction import stratified_cal_test_split   # noqa: E402
from exchangeable_features import UnlabeledTransform          # noqa: E402

ALPHA = 0.1
QE_K = 10
QE_A = 3.0
PCA_DIM = 128
N_CLUSTERS = 20
CAL_SIZES = [200, 400, 800]      # 2 / 4 / 8 shots per class (K=100 poles)
TEST_SIZE = 2000
N_TRIALS = 20
FOLKLORE_GATE = 0.7

# backbone -> embedding filename suffix (before ".pt")
BACKBONE_SUFFIX = {
    "dinov2": "",
    "mae": "_mae-base",
    "clip": "_clip-base",
}


# --------------------------------------------------------------------------
# inlined helpers (verbatim from dwt_score_histograms.py)
# --------------------------------------------------------------------------
def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def qe_smooth(X, pool, k=QE_K, a=QE_A, exclude_self=False):
    """Classic alpha-QE against a frozen L2-normed pool:
    T(x) = L2norm(x + sum_{i in NN_k(x)} max(cos(x,u_i),0)^a * u_i)."""
    P = l2n(pool.astype(np.float64))
    Xn = l2n(X.astype(np.float64))
    out = np.empty_like(Xn)
    for i0 in range(0, len(Xn), 1024):
        ch = Xn[i0:i0 + 1024]
        S = ch @ P.T
        if exclude_self:
            rows = np.arange(len(ch))
            S[rows, i0 + rows] = -np.inf
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        s = np.take_along_axis(S, idx, axis=1)
        w = np.clip(s, 0.0, None) ** a
        v = ch + (w[:, :, None] * P[idx]).sum(axis=1)
        out[i0:i0 + 1024] = l2n(v)
    return out


def knn_homophily(X, y, k=QE_K):
    """h(k): mean fraction of the k nearest OTHER labeled points sharing the
    ego's label (kNN label homophily diagnostic, raw space)."""
    Xn = l2n(X.astype(np.float64))
    same = 0.0
    n = len(Xn)
    for i0 in range(0, n, 1024):
        ch = Xn[i0:i0 + 1024]
        S = ch @ Xn.T
        rows = np.arange(len(ch))
        S[rows, i0 + rows] = -np.inf
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        same += (y[idx] == y[i0:i0 + 1024, None]).mean(axis=1).sum()
    return same / n


def participation_ratio(U):
    """LABEL-FREE dial: PR = (sum lambda)^2 / sum lambda^2 of the raw pool
    covariance eigenvalues -- effective dimensionality of the feature cloud.
    Low PR (aircraft-like) => signal in a low-variance tail => truncation
    load-bearing; high PR (cifar-like) => broad spectrum."""
    Un = l2n(U.astype(np.float64))
    Uc = Un - Un.mean(axis=0, keepdims=True)
    # singular values of centered data; lambda_i = s_i^2 / (n-1)
    s = np.linalg.svd(Uc, full_matrices=False, compute_uv=False)
    lam = (s ** 2) / (len(Uc) - 1)
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-30))


def scores_and_qhat(Z_cal, y_cal, Z_test, y_test):
    """Prototype-cosine nonconformity scores + split-CP quantile (LOO cal)."""
    K = int(max(y_cal.max(), y_test.max())) + 1
    protos = np.zeros((K, Z_cal.shape[1]))
    for c in range(K):
        m = y_cal == c
        if m.any():
            protos[c] = Z_cal[m].mean(axis=0)
    protos = l2n(protos)
    s_test = -(l2n(Z_test) @ protos.T)    # (n_test, K)

    n = len(y_cal)
    Zc = l2n(Z_cal)
    cal_true = np.empty(n)
    for c in np.unique(y_cal):
        m = y_cal == c
        n_c = int(m.sum())
        cls = Zc[m]
        if n_c < 2:
            loo = np.repeat(l2n(cls.sum(0, keepdims=True)), n_c, axis=0)
        else:
            loo = l2n((cls.sum(axis=0, keepdims=True) - cls) / (n_c - 1))
        cal_true[m] = -np.einsum("ij,ij->i", cls, loo)
    m_idx = int(np.ceil((1 - ALPHA) * (n + 1))) - 1
    q_hat = np.sort(cal_true)[min(m_idx, n - 1)]

    nt = len(y_test)
    test_true = s_test[np.arange(nt), y_test]
    coverage = float((test_true <= q_hat).mean())
    sizes = (s_test <= q_hat).sum(axis=1)
    return dict(q_hat=float(q_hat), coverage=coverage,
                avg_size=float(sizes.mean()))


# --------------------------------------------------------------------------
# D3 gate constants (verbatim math from dwt_gate_constants.py)
# --------------------------------------------------------------------------
def measure_gate_constants(X, y, U, yu, k=QE_K, a=QE_A, n_egos=4000, seed=0):
    """Return the D3 gate constants: h_w, beta, k_eff, kappa, rho, h_star,
    d_prime_ratio, plus sigma_v / sigma_tot / axis SNR."""
    X, U = l2n(X.astype(np.float64)), l2n(U.astype(np.float64))
    K = int(max(y.max(), yu.max())) + 1
    mu = np.stack([X[y == c].mean(axis=0) if (y == c).any()
                   else np.zeros(X.shape[1]) for c in range(K)])

    D2 = ((mu[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D2, np.inf)
    cstar = D2.argmin(axis=1)
    delta_pair = np.sqrt(D2[np.arange(K), cstar])
    v = (mu - mu[cstar]) / (delta_pair[:, None] + 1e-12)

    sig_v2, sig_tot2, nres = 0.0, 0.0, 0
    for c in range(K):
        m = y == c
        if not m.any():
            continue
        E = X[m] - mu[c]
        sig_v2 += ((E @ v[c]) ** 2).sum()
        sig_tot2 += (E ** 2).sum()
        nres += len(E)
    sigma_v = float(np.sqrt(sig_v2 / nres))
    sigma_tot = float(np.sqrt(sig_tot2 / nres))

    rng = np.random.default_rng(seed)
    idx_egos = rng.permutation(len(X))[:n_egos]

    h_w_all, h_unw_all, beta_all, keff_all = [], [], [], []
    kappa_num, kappa_den = 0.0, 0.0
    for i0 in range(0, len(idx_egos), 512):
        ids = idx_egos[i0:i0 + 512]
        ch, yc = X[ids], y[ids]
        S = ch @ U.T
        nn = np.argpartition(-S, k, axis=1)[:, :k]
        s = np.take_along_axis(S, nn, axis=1)
        w = np.clip(s, 0.0, None) ** a
        W = w.sum(axis=1)
        ok = W > 1e-12
        beta_all.append(W[ok] / (1.0 + W[ok]))
        keff_all.append(W[ok] ** 2 / (w[ok] ** 2).sum(axis=1))
        cn = yu[nn]
        same = cn == yc[:, None]
        h_w = np.where(ok, (w * same).sum(axis=1) / np.maximum(W, 1e-12), np.nan)
        h_w_all.append(h_w[ok])
        h_unw_all.append(same.mean(axis=1))
        proj = ((mu[yc[:, None]] - mu[cn]) * v[yc][:, None, :]).sum(-1)
        drift = (w * (~same) * proj).sum(axis=1) / np.maximum(W, 1e-12)
        imp = np.where(ok, (w * (~same)).sum(axis=1) / np.maximum(W, 1e-12), 0.0)
        den = imp * delta_pair[yc]
        m = ok & (imp > 0.02)
        kappa_num += drift[m].sum()
        kappa_den += den[m].sum()

    h_w = float(np.concatenate(h_w_all).mean())
    h_unw = float(np.concatenate(h_unw_all).mean())
    beta = float(np.concatenate(beta_all).mean())
    keff = float(np.concatenate(keff_all).mean())
    kappa = float(kappa_num / kappa_den) if kappa_den > 0 else float("nan")
    rho = float(np.sqrt((1 - beta) ** 2 + beta ** 2 / keff))
    hstar = 1 - (1 - rho) / (2 * beta * kappa)
    dratio = (1 - 2 * beta * kappa * (1 - h_w)) / rho
    return dict(h_w=h_w, h_unweighted=h_unw, beta_hat=beta, k_eff=keff,
                kappa=kappa, rho=rho, h_star=float(hstar),
                d_prime_ratio=float(dratio), sigma_v=sigma_v,
                sigma_tot=sigma_tot, delta_pair_mean=float(delta_pair.mean()),
                axis_snr=float(delta_pair.mean() / (sigma_v + 1e-12)))


# --------------------------------------------------------------------------
# per-cell runner
# --------------------------------------------------------------------------
def load_cell(emb_dir, ds, backbone):
    suf = BACKBONE_SUFFIX[backbone]
    lab_f = os.path.join(emb_dir, f"embeddings_{ds}{suf}.pt")
    unl_f = os.path.join(emb_dir, f"embeddings_{ds}_unlabeled{suf}.pt")
    lab = torch.load(lab_f, map_location="cpu", weights_only=False)
    unl = torch.load(unl_f, map_location="cpu", weights_only=False)
    X = lab["embeddings"].numpy().astype(np.float64)
    y = lab["labels"].numpy().astype(int)
    U = unl["embeddings"].numpy().astype(np.float64)
    yu = unl["labels"].numpy().astype(int) if "labels" in unl else y[:len(U)] * 0
    return X, y, U, yu


def run_cell(ds, backbone, emb_dir, n_trials, seed):
    X, y, U, yu = load_cell(emb_dir, ds, backbone)
    K = len(np.unique(y))
    print(f"[{backbone}/{ds}] labeled {X.shape}, pool {U.shape}, K={K}",
          flush=True)

    # ---- dials ----
    h = knn_homophily(X, y)
    pr = participation_ratio(U)
    gc = measure_gate_constants(X, y, U, yu, seed=seed)
    print(f"[{backbone}/{ds}] h={h:.3f} PR={pr:.1f} h_w={gc['h_w']:.3f} "
          f"h*={gc['h_star']:.3f} d'ratio={gc['d_prime_ratio']:.3f} "
          f"axis_snr={gc['axis_snr']:.2f}", flush=True)

    # ---- arms precomputed on FULL set (paired splits via shared indices) ----
    Z_arm = {"raw": l2n(X)}
    tr = UnlabeledTransform(pca_dim=PCA_DIM, whiten="cluster",
                            n_clusters=N_CLUSTERS, random_state=seed).fit(U)
    Z_arm["wt"] = tr.transform(X)

    U_s = qe_smooth(U, U, exclude_self=True)
    X_s = qe_smooth(X, U)
    tr_s = UnlabeledTransform(pca_dim=PCA_DIM, whiten="cluster",
                              n_clusters=N_CLUSTERS, random_state=seed).fit(U_s)
    Z_arm["qe_wt"] = tr_s.transform(X_s)

    idx_all = np.arange(len(y))[:, None].astype(np.float64)
    results = {arm: {c: {"cov": [], "size": []} for c in CAL_SIZES}
               for arm in Z_arm}
    for cal_size in CAL_SIZES:
        for t in range(n_trials):
            ic, yc, it, yt = stratified_cal_test_split(
                idx_all, y, cal_size=cal_size, test_size=TEST_SIZE,
                random_state=seed + 1000 * t + cal_size)
            ic = ic[:, 0].astype(int)
            it = it[:, 0].astype(int)
            for arm, Z in Z_arm.items():
                st = scores_and_qhat(Z[ic], yc, Z[it], yt)
                results[arm][cal_size]["cov"].append(st["coverage"])
                results[arm][cal_size]["size"].append(st["avg_size"])
        line = f"[{backbone}/{ds}] cal={cal_size}: "
        for arm in ["raw", "wt", "qe_wt"]:
            s = results[arm][cal_size]
            line += (f"{arm} sz {np.mean(s['size']):.2f} "
                     f"cov {np.mean(s['cov']):.3f} | ")
        print(line, flush=True)

    # ---- assemble output ----
    out = {"dataset": ds, "backbone": backbone, "K": K,
           "n_labeled": int(len(X)), "n_pool": int(len(U)),
           "alpha": ALPHA, "cal_sizes": CAL_SIZES, "test_size": TEST_SIZE,
           "n_trials": n_trials,
           "dials": {"h_knn_k10": float(h), "participation_ratio": pr, **gc},
           "arms": {}}
    for arm in Z_arm:
        out["arms"][arm] = {}
        for cal_size in CAL_SIZES:
            s = results[arm][cal_size]
            out["arms"][arm][str(cal_size)] = {
                "coverage_mean": float(np.mean(s["cov"])),
                "size_mean": float(np.mean(s["size"])),
                "size_sd": float(np.std(s["size"])),
                "sizes": [float(v) for v in s["size"]],
                "coverages": [float(v) for v in s["cov"]]}

    # ---- paired qe verdict + gate-prediction check ----
    out["verdict"] = {}
    for cal_size in CAL_SIZES:
        d = (np.array(results["qe_wt"][cal_size]["size"])
             - np.array(results["wt"][cal_size]["size"]))
        rel = d / np.array(results["wt"][cal_size]["size"])
        out["verdict"][str(cal_size)] = {
            "paired_delta_mean": float(d.mean()),
            "paired_delta_se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "relative_change_mean": float(rel.mean()),
            "qe_gains_actual": bool(d.mean() < 0)}

    largest = str(max(CAL_SIZES))
    actual_gain = out["verdict"][largest]["qe_gains_actual"]
    d3_pred_gain = bool(gc["d_prime_ratio"] > 1.0)
    hw_pred_gain = bool(gc["h_w"] >= gc["h_star"])
    folk_pred_gain = bool(h >= FOLKLORE_GATE)
    out["gate_check"] = {
        "decided_at_cal": int(largest),
        "actual_qe_gains": actual_gain,
        "d3_dratio_predicts_gain": d3_pred_gain,
        "d3_hw_vs_hstar_predicts_gain": hw_pred_gain,
        "folklore_h07_predicts_gain": folk_pred_gain,
        "d3_dratio_correct": bool(d3_pred_gain == actual_gain),
        "d3_hw_correct": bool(hw_pred_gain == actual_gain),
        "folklore_correct": bool(folk_pred_gain == actual_gain)}
    print(f"[{backbone}/{ds}] VERDICT@cal{largest}: actual_gain={actual_gain} "
          f"| D3(d'ratio)->{d3_pred_gain} {'OK' if out['gate_check']['d3_dratio_correct'] else 'MISS'} "
          f"| folklore(h>=.7)->{folk_pred_gain} {'OK' if out['gate_check']['folklore_correct'] else 'MISS'}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--out_dir", default="output/backbone_dwt")
    ap.add_argument("--backbones", nargs="+", default=["dinov2", "mae", "clip"])
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "aircraft"])
    ap.add_argument("--n_trials", type=int, default=N_TRIALS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_cells = []
    for backbone in args.backbones:
        for ds in args.datasets:
            try:
                cell = run_cell(ds, backbone, args.emb_dir,
                                args.n_trials, args.seed)
            except FileNotFoundError as e:
                print(f"[SKIP] {backbone}/{ds}: {e}", flush=True)
                continue
            all_cells.append(cell)
            with open(os.path.join(
                    args.out_dir, f"cell_{backbone}_{ds}.json"), "w") as f:
                json.dump(cell, f, indent=2)

    # ---- master table ----
    table = {"alpha": ALPHA, "cal_sizes": CAL_SIZES, "test_size": TEST_SIZE,
             "n_trials": args.n_trials, "folklore_gate": FOLKLORE_GATE,
             "cells": all_cells}
    with open(os.path.join(args.out_dir, "backbone_dwt_table.json"), "w") as f:
        json.dump(table, f, indent=2)

    # ---- console summary table ----
    print("\n" + "=" * 100)
    print("BACKBONE x DATASET  DWT gate table  (verdict decided at largest cal)")
    print("=" * 100)
    hdr = (f"{'backbone':8s} {'dataset':10s} {'h':>5s} {'PR':>6s} "
           f"{'h_w':>5s} {'h*':>6s} {'d`rat':>6s} | "
           f"{'wt@800':>7s} {'qe@800':>7s} {'rel%':>6s} {'act':>4s} "
           f"{'D3':>4s} {'folk':>5s}")
    print(hdr)
    print("-" * 100)
    for c in all_cells:
        g = c["dials"]
        largest = str(max(CAL_SIZES))
        wt = c["arms"]["wt"][largest]["size_mean"]
        qe = c["arms"]["qe_wt"][largest]["size_mean"]
        rel = c["verdict"][largest]["relative_change_mean"] * 100
        gk = c["gate_check"]
        print(f"{c['backbone']:8s} {c['dataset']:10s} "
              f"{g['h_knn_k10']:5.2f} {g['participation_ratio']:6.1f} "
              f"{g['h_w']:5.2f} {g['h_star']:6.2f} {g['d_prime_ratio']:6.2f} | "
              f"{wt:7.2f} {qe:7.2f} {rel:+6.1f} "
              f"{'gain' if gk['actual_qe_gains'] else 'harm':>4s} "
              f"{'OK' if gk['d3_dratio_correct'] else 'MISS':>4s} "
              f"{'OK' if gk['folklore_correct'] else 'MISS':>5s}")
    print("=" * 100)
    print(f"\nsaved -> {os.path.join(args.out_dir, 'backbone_dwt_table.json')}")


if __name__ == "__main__":
    main()
