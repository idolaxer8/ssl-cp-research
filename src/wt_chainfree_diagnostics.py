"""Chain-free W/T diagnostics: do W/T improve D1's own inputs?

Re-anchoring (2026-08-18): with D3's quantitative layer unsupported and D1
the base lemma, W/T are justified WITHOUT the d' -> Lemma B -> Prop C chain,
as services to the two primitive objects every downstream step consumes:

  NEIGHBORHOODS (W's target): pool-kNN label homophily h_w is D1's only
    dataset-dependent input -- the impurity budget is (1-h_w)*Delta_F.
    Claim W-N: computing neighborhoods in the whitened metric raises h_w
    (raw cosine proximity is dominated by shared-field nuisance directions;
    whitening downweights them), hence shrinks D1's budget and raises the
    D1 certification rate -- an assumption-free, per-point justification.
  ANCHORS (T's target): the estimated prototype mu_hat is the only
    estimated object in the deployed score. Claim T-A: truncation cuts the
    s-shot anchor error (retained noise trace m/s vs d/s) at the measured
    alignment cost a_m ~ 1 -- a D1-shaped statement ("estimated anchor gets
    closer to the true anchor, with an exact rate").

Registered predictions (BEFORE the run):
  (P-N) h_w(wlw) > h_w(raw), largest gain on low-h data (aircraft/cars);
        wdiag ~ no change (no rotation); D1 certification rate moves with
        h_w.
  (P-A) s-shot relative anchor error: t128/t128w < raw; wlw >= raw (full-
        rank whitening upweights noisy-estimate directions). The W-vs-T
        division of labor: W serves neighborhoods, T serves anchors;
        t128w serves both.

Per space (all pool-fit; L2-normalized after transform):
  h_w (weighted, k=10, a=3, mirroring dwt_gate_constants), D1 certification
  rate (sum_u (w/W) r_u + (1-h_w)*Delta_F < eps, all quantities in-space),
  and s-shot anchor error ||mu_hat - mu|| / delta_pair for s in {2, 8}.

Output: output/dwt_theory/wt_chainfree_diagnostics.json + printed table.
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from dwt_gate_constants import DATASETS, load_pt, l2n            # noqa: E402
from wt_phase_diagnostics import within_residuals, lw_inv_sqrt   # noqa: E402

N_CLUSTERS = 20
SEED = 42
K_NN = 10
A_W = 3.0
N_EGOS = 4000
SHOT_GRID = [2, 8]
N_REP = 20


def build_spaces_xu(X, U):
    """Pool-fit transform menu applied to BOTH labeled X and pool U."""
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10).fit(U)
    resid = within_residuals(U, km.labels_)
    var = (resid ** 2).mean(axis=0)
    inv_std = 1.0 / np.sqrt(var + max(1e-4, 0.01 * float(np.median(var))))
    W_lw = lw_inv_sqrt(resid)

    pca = PCA(n_components=128, random_state=SEED).fit(U)
    Up = pca.transform(U)
    km_p = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10).fit(Up)
    resid_p = within_residuals(Up, km_p.labels_)
    var_p = (resid_p ** 2).mean(axis=0)
    inv_std_p = 1.0 / np.sqrt(
        var_p + max(1e-4, 0.01 * float(np.median(var_p))))

    Xp = pca.transform(X)
    return {
        "raw":   (X,            U),
        "wdiag": (X * inv_std,  U * inv_std),
        "wlw":   (X @ W_lw,     U @ W_lw),
        "t128":  (Xp,           Up),
        "t128w": (Xp * inv_std_p, Up * inv_std_p),
    }


def neighborhood_stats(Z, y, Zu, yu, rng):
    """h_w and D1 certification rate in this space (L2-normed)."""
    Zn, Zun = l2n(Z), l2n(Zu)
    K = int(max(y.max(), yu.max())) + 1
    mu = np.stack([Zn[y == c].mean(axis=0) for c in range(K)])
    MD = np.sqrt(((mu[:, None, :] - mu[None, :, :]) ** 2).sum(-1))
    ru_all = np.linalg.norm(Zun - mu[yu], axis=1)      # pool own-anchor error

    idx = rng.permutation(len(Zn))[:N_EGOS]
    hs, certs = [], []
    for i0 in range(0, len(idx), 512):
        ids = idx[i0:i0 + 512]
        ch, yc = Zn[ids], y[ids]
        S = ch @ Zun.T
        nn = np.argpartition(-S, K_NN, axis=1)[:, :K_NN]
        s = np.take_along_axis(S, nn, axis=1)
        w = np.clip(s, 0.0, None) ** A_W
        Wt = w.sum(axis=1)
        ok = Wt > 1e-12
        cn = yu[nn]
        same = cn == yc[:, None]
        h_w = (w * same).sum(axis=1) / np.maximum(Wt, 1e-12)
        # D1 condition, all in-space
        r_term = (w * ru_all[nn]).sum(axis=1) / np.maximum(Wt, 1e-12)
        dF = np.where(~same, MD[yc[:, None], cn], 0.0).max(axis=1)
        eps = np.linalg.norm(ch - mu[yc], axis=1)
        cert = (r_term + (1 - h_w) * dF) < eps
        hs.append(h_w[ok])
        certs.append(cert[ok])
    return float(np.concatenate(hs).mean()), float(np.concatenate(certs).mean())


def anchor_error(Z, y, rng):
    """s-shot prototype error relative to the nearest-pair separation."""
    Zn = l2n(Z)
    K = int(y.max()) + 1
    mu = np.stack([Zn[y == c].mean(axis=0) for c in range(K)])
    MD = np.sqrt(((mu[:, None, :] - mu[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(MD, np.inf)
    dpair = MD.min(axis=1)
    out = {}
    for s in SHOT_GRID:
        errs = []
        for c in range(K):
            ic = np.where(y == c)[0]
            if len(ic) < s:
                continue
            for _ in range(N_REP):
                mh = Zn[rng.choice(ic, s, replace=False)].mean(axis=0)
                errs.append(np.linalg.norm(mh - mu[c]) / dpair[c])
        out[s] = float(np.mean(errs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default=os.path.join(
        "output", "from_cluster", "embeddings"))
    ap.add_argument("--out_dir", default=os.path.join("output", "dwt_theory"))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for ds in args.datasets:
        lab_f, pool_f = DATASETS[ds]
        X, y = load_pt(os.path.join(args.emb_dir, lab_f))
        U, yu = load_pt(os.path.join(args.emb_dir, pool_f))
        X, U = l2n(X.astype(np.float64)), l2n(U.astype(np.float64))
        spaces = build_spaces_xu(X, U)

        row = {}
        for name, (Zx, Zu) in spaces.items():
            rng = np.random.default_rng(SEED)
            h_w, cert = neighborhood_stats(Zx, y, Zu, yu, rng)
            aerr = anchor_error(Zx, y, rng)
            row[name] = {"h_w": h_w, "d1_cert_rate": cert,
                         "anchor_err_s2": aerr[2], "anchor_err_s8": aerr[8]}
        results[ds] = row
        r = row
        print(f"{ds:14}  h_w   raw {r['raw']['h_w']:.3f}  wdiag "
              f"{r['wdiag']['h_w']:.3f}  wlw {r['wlw']['h_w']:.3f}  t128 "
              f"{r['t128']['h_w']:.3f}  t128w {r['t128w']['h_w']:.3f}",
              flush=True)
        print(f"{'':14}  cert  raw {r['raw']['d1_cert_rate']:.3f}  wdiag "
              f"{r['wdiag']['d1_cert_rate']:.3f}  wlw "
              f"{r['wlw']['d1_cert_rate']:.3f}  t128 "
              f"{r['t128']['d1_cert_rate']:.3f}  t128w "
              f"{r['t128w']['d1_cert_rate']:.3f}", flush=True)
        print(f"{'':14}  aerr2 raw {r['raw']['anchor_err_s2']:.3f}  wdiag "
              f"{r['wdiag']['anchor_err_s2']:.3f}  wlw "
              f"{r['wlw']['anchor_err_s2']:.3f}  t128 "
              f"{r['t128']['anchor_err_s2']:.3f}  t128w "
              f"{r['t128w']['anchor_err_s2']:.3f}", flush=True)

    path = os.path.join(args.out_dir, "wt_chainfree_diagnostics.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
