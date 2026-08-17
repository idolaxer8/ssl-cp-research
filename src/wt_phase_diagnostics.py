"""W/T phase diagnostics: the per-phase d' instrument for lemmas W1/T1.

Companion to src/dwt_gate_constants.py (D phase). For each dataset we build
the pool-fit transforms of the deployed menu and measure, per nearest-
prototype class pair, the discriminability d' in each space -- the same
currency as Theorem D3 -- plus the specific quantities the W1/T1 lemmas
predict (docs/dwt_wt_lemmas.md):

  W1: d' after whitening vs the label-oracle Mahalanobis bound
      sqrt(delta' Sigma_w^{-1} delta)  (Cauchy-Schwarz maximum; W1a).
      Pool-fit (pseudo-cluster) whitening should approach the oracle on
      high-homophily data and lag on low-h data (impurity clause W1c).
  T1: subspace alignment a_m = ||P_m delta||^2 / ||delta||^2 of pair axes
      with the top-m pool PCs (T1a: d'_m/d' = sqrt(a_m) in the whitened
      metric), the Chang discriminant-contribution tail fraction (T1
      failure certificate), and the finite-shot alignment factor
      A_m/(A_m + 2m/s) (T1b: the estimation saving that makes truncation
      profitable at all).

Spaces (all transforms fit on the unlabeled pool only):
  raw        L2-normed embeddings (768)
  wdiag      per-dim within-k-means-cluster inverse-std (deployed engine)
  wlw        full-rank Ledoit-Wolf within-cluster whitening (champion on
             aircraft: lw_cluster768)
  t128       PCA-128, no whitening
  t128w      PCA-128 + per-dim cluster whitening (champion on separable data)

Output: output/dwt_theory/wt_phase_diagnostics.json + printed table.
Run:    python src/wt_phase_diagnostics.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from measure_dprime_all import load                                # noqa: E402

DATASETS = ["cifar100", "miniimagenet", "cifar10", "stanford_cars", "aircraft"]
N_CLUSTERS = 20
SEED = 42
M_GRID = [32, 128, 256]
SHOT_GRID = [2, 8]


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def nearest_pairs(X, y):
    """Nearest-prototype partner per class from raw class means (fixed across
    spaces so every space is scored on the SAME confusable pairs)."""
    classes = np.unique(y)
    mus = np.stack([X[y == c].mean(axis=0) for c in classes])
    cosm = l2n(mus) @ l2n(mus).T
    np.fill_diagonal(cosm, -np.inf)
    partner = classes[np.argmax(cosm, axis=1)]
    return classes, partner


def dprime_in_space(Z, y, classes, partner, normalize, min_pts=5):
    """Per-class pair d' with axis from THIS space's class means."""
    Zs = l2n(Z) if normalize else Z
    mus = np.stack([Zs[y == c].mean(axis=0) for c in classes])
    dps = []
    for i, c_y in enumerate(classes):
        c_c = partner[i]
        v = mus[i] - mus[classes == c_c][0]
        v = v / (np.linalg.norm(v) + 1e-12)
        m_y, m_c = (y == c_y), (y == c_c)
        if m_y.sum() < min_pts or m_c.sum() < min_pts:
            continue
        gy, gc = Zs[m_y] @ v, Zs[m_c] @ v
        sd = np.sqrt(0.5 * (gy.var(ddof=1) + gc.var(ddof=1)))
        dps.append((gy.mean() - gc.mean()) / (sd + 1e-12))
    return np.array(dps)


def within_residuals(X, labels):
    resid = X.copy()
    for c in np.unique(labels):
        m = labels == c
        if m.any():
            resid[m] -= X[m].mean(axis=0)
    return resid


def lw_inv_sqrt(resid):
    """Ledoit-Wolf covariance of residuals -> Sigma^{-1/2} (symmetric)."""
    lw = LedoitWolf().fit(resid)
    w, V = np.linalg.eigh(lw.covariance_)
    w = np.maximum(w, 1e-10)
    return V @ np.diag(w ** -0.5) @ V.T


def build_spaces(X, U):
    """Fit the transform menu on the pool U; return dict name -> (X_t, U_t)."""
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10).fit(U)
    lab_u = km.labels_

    # diagonal within-cluster inverse std (deployed engine)
    resid = within_residuals(U, lab_u)
    var = (resid ** 2).mean(axis=0)
    inv_std = 1.0 / np.sqrt(var + max(1e-4, 0.01 * float(np.median(var))))

    # full-rank LW within-cluster whitening
    W_lw = lw_inv_sqrt(resid)

    # PCA-128 (pool-fit) + cluster whitening in PCA space
    pca = PCA(n_components=128, random_state=SEED).fit(U)
    Up = pca.transform(U)
    km_p = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10).fit(Up)
    resid_p = within_residuals(Up, km_p.labels_)
    var_p = (resid_p ** 2).mean(axis=0)
    inv_std_p = 1.0 / np.sqrt(
        var_p + max(1e-4, 0.01 * float(np.median(var_p))))

    Xp = pca.transform(X)
    return {
        "raw": X,
        "wdiag": X * inv_std,
        "wlw": X @ W_lw,
        "t128": Xp,
        "t128w": Xp * inv_std_p,
    }


def oracle_bound(X, y, classes, partner):
    """W1a label-oracle: sqrt(delta' Sigma_w^{-1} delta) per pair, with
    Sigma_w = LW estimate of the labeled within-class covariance, vs the
    raw-axis d' (unnormalized) it upper-bounds."""
    resid = within_residuals(X, y)
    lw = LedoitWolf().fit(resid)
    w, V = np.linalg.eigh(lw.covariance_)
    w = np.maximum(w, 1e-10)
    inv_sqrt = V @ np.diag(w ** -0.5) @ V.T
    mus = np.stack([X[y == c].mean(axis=0) for c in classes])
    bounds = []
    for i in range(len(classes)):
        delta = mus[i] - mus[classes == partner[i]][0]
        bounds.append(np.linalg.norm(inv_sqrt @ delta))
    return np.array(bounds)


def alignment_and_chang(X, U, y, classes, partner):
    """T1a alignment a_m of pair axes with top-m POOL PCs (raw space) and the
    Chang discriminant tail fraction over pool PCs."""
    Uc = U - U.mean(axis=0)
    cov = (Uc.T @ Uc) / len(U)
    lam, V = np.linalg.eigh(cov)          # ascending
    lam, V = lam[::-1], V[:, ::-1]        # descending
    mus = np.stack([X[y == c].mean(axis=0) for c in classes])

    a_m = {m: [] for m in M_GRID}
    chang_tail = []
    for i in range(len(classes)):
        delta = mus[i] - mus[classes == partner[i]][0]
        proj2 = (V.T @ delta) ** 2                       # per-PC energy
        tot = proj2.sum() + 1e-24
        for m in M_GRID:
            a_m[m].append(proj2[:m].sum() / tot)
        contrib = proj2 / np.maximum(lam, 1e-10)         # Chang functional
        chang_tail.append(contrib[128:].sum() / (contrib.sum() + 1e-24))

    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    c = X.shape[1] / len(U)
    bulk_edge = (1 + np.sqrt(c)) ** 2 * float(np.median(lam))
    n_spikes = int((lam > bulk_edge).sum())
    return ({m: float(np.mean(v)) for m, v in a_m.items()},
            float(np.mean(chang_tail)), pr, n_spikes)


def finite_shot_check(X, y, classes, partner, rng, n_rep=20):
    """T1b: measured cos^2(v_hat, v) of s-shot estimated pair axes vs the
    predicted alignment factor A_m/(A_m + 2m/s) in the deployed t128 space
    is checked in the doc analytically; here we measure cos^2 at full dim
    (m=768) and m=128 via pool PCA done by caller -- kept raw-space simple:
    axis stability under s-shot prototypes."""
    mus = np.stack([X[y == c].mean(axis=0) for c in classes])
    out = {}
    for s in SHOT_GRID:
        cs = []
        for i in range(len(classes)):
            c_y, c_c = classes[i], partner[i]
            iy, ic = np.where(y == c_y)[0], np.where(y == c_c)[0]
            if len(iy) < s or len(ic) < s:
                continue
            v = mus[i] - mus[classes == c_c][0]
            v = v / (np.linalg.norm(v) + 1e-12)
            for _ in range(n_rep):
                vh = (X[rng.choice(iy, s, replace=False)].mean(axis=0)
                      - X[rng.choice(ic, s, replace=False)].mean(axis=0))
                vh = vh / (np.linalg.norm(vh) + 1e-12)
                cs.append(float((vh @ v) ** 2))
        out[s] = float(np.mean(cs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--out_dir", default="output/dwt_theory")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    results = {}
    for ds in args.datasets:
        X, y, U = load(ds, args.emb_dir)
        X, U = l2n(X), l2n(U)
        classes, partner = nearest_pairs(X, y)
        spaces = build_spaces(X, U)

        row = {"spaces": {}}
        base = None
        for name, Z in spaces.items():
            dps_n = dprime_in_space(Z, y, classes, partner, normalize=True)
            dps_u = dprime_in_space(Z, y, classes, partner, normalize=False)
            if name == "raw":
                base = dps_n
            row["spaces"][name] = {
                "dprime_mean": float(dps_n.mean()),
                "dprime_mean_unnorm": float(dps_u.mean()),
                "ratio_vs_raw": float((dps_n / base).mean()),
                "frac_improved": float((dps_n / base > 1).mean()),
            }

        bounds = oracle_bound(X, y, classes, partner)
        dps_raw_u = dprime_in_space(X, y, classes, partner, normalize=False)
        dps_wlw_u = dprime_in_space(spaces["wlw"], y, classes, partner,
                                    normalize=False)
        row["w1"] = {
            "oracle_mahal_mean": float(bounds.mean()),
            "raw_unnorm_mean": float(dps_raw_u.mean()),
            "wlw_unnorm_mean": float(dps_wlw_u.mean()),
            "wlw_frac_of_oracle": float((dps_wlw_u / bounds).mean()),
            "bound_violations": int((dps_wlw_u > bounds * (1 + 1e-6)).sum()),
        }

        a_m, chang_tail, pr, n_spikes = alignment_and_chang(
            X, U, y, classes, partner)
        row["t1"] = {
            "alignment_a_m": a_m,
            "pred_t128_ratio_sqrt_a128": float(np.sqrt(a_m[128])),
            "chang_tail_frac_beyond_128": chang_tail,
            "pool_pr": pr, "n_spikes_crude": n_spikes,
        }
        row["t1b_axis_cos2"] = finite_shot_check(
            X, y, classes, partner, rng)

        results[ds] = row
        s = row["spaces"]
        print(f"{ds:14} d'raw {s['raw']['dprime_mean']:5.2f}  "
              f"wdiag x{s['wdiag']['ratio_vs_raw']:4.2f}  "
              f"wlw x{s['wlw']['ratio_vs_raw']:4.2f}  "
              f"t128 x{s['t128']['ratio_vs_raw']:4.2f}  "
              f"t128w x{s['t128w']['ratio_vs_raw']:4.2f}  "
              f"| a128 {a_m[128]:.2f} sqrt {np.sqrt(a_m[128]):.2f}  "
              f"changTail {chang_tail:.2f}  PR {pr:5.1f}  "
              f"wlw/oracle {row['w1']['wlw_frac_of_oracle']:.2f}",
              flush=True)

    path = os.path.join(args.out_dir, "wt_phase_diagnostics.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
