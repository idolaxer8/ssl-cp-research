"""Why does the (I)-model overpredict the qe d'-ratio on every dataset?

The D3 ratio factors exactly:  ratio = S / N  with
  S = signal shrink  = (mean pair separation after) / (before)
      model: S_pred = 1 - 2*beta*kappa*(1-h)
  N = noise shrink   = (within-class sd along the pair axis after)/(before)
      model: N_pred = rho = sqrt((1-beta)^2 + beta^2/k_eff)

and the denominator's exact empirical decomposition is
  N_meas^2 = (1-beta)^2 + beta^2 * Var(nu_v)/sigma_v^2
             + 2*beta*(1-beta) * Cov(e_v, nu_v)/sigma_v^2
where nu is the weighted pool-neighbor mean and e the ego's own error.
Under (I): Var(nu_v)/sigma_v^2 = 1/k_eff and the covariance is ZERO.

This script measures, per dataset and per nearest-prototype class pair:
  - S_meas, N_meas (so which side of the ratio carries the overprediction)
  - the three denominator components (so V1's two faces are separated:
    ego-neighbor covariance vs neighbor-mean variance floor -- the
    mean-shift/local-structure effect that k-averaging cannot remove)
  - V1 retained-own-noise coefficient  <xi_same, e>/||e||^2
  - V2 alignment  corr(e_v, dbar_v)  (error-aligned foreign selection)
using the POOL LABELS available in the carve-out files (diagnostic only --
never used by the deployed pipeline).

Writes output/cars_qe_gate/dprime_overprediction.json (+ figure via
src/plot_dprime_overprediction.py).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dwt_score_histograms import l2n                              # noqa: E402
from measure_dprime_all import load, PREDICTED, VERDICT           # noqa: E402

QE_K = 10
QE_A = 3.0

# gate constants (docs/dwt_denoise_theorem.md Section 6): h_w, beta, k_eff, kappa
CONSTANTS = {
    "cifar100": (0.809, 0.615, 9.38, 0.629),
    "miniimagenet": (0.919, 0.693, 9.30, 0.717),
    "cifar10": (0.969, 0.644, 9.58, 0.755),
    "stanford_cars": (0.461, 0.827, 9.83, 0.585),
    "aircraft": (0.258, 0.862, 9.87, 0.615),
}


def per_ego_scalars(X, y, U, yU, mus, classes, partner_idx, max_per_class):
    """One pass over egos: for each ego compute the scalars the decomposition
    needs, all projected on the ego's own nearest-prototype pair axis v.
    Returns dict of per-ego arrays."""
    P = l2n(U)
    Xn = l2n(X)
    cls_index = {c: i for i, c in enumerate(classes)}

    # subsample egos per class for tractability (mini has 500/class)
    keep = []
    rng = np.random.default_rng(0)
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    Xk, yk = Xn[keep], y[keep]

    # per-class axis v = (mu_y - mu_c)/||.||, partner = nearest prototype
    axes = np.zeros_like(mus)
    for i in range(len(classes)):
        d = mus[i] - mus[partner_idx[i]]
        axes[i] = d / (np.linalg.norm(d) + 1e-12)

    n = len(Xk)
    out = {k: np.empty(n) for k in
           ["e_v", "nu_v", "h_w", "beta", "k_eff",
            "v1_coef", "dbar_v", "drift_v"]}
    for i0 in range(0, n, 512):
        ch = Xk[i0:i0 + 512]
        ych = yk[i0:i0 + 512]
        S = ch @ P.T
        idx = np.argpartition(-S, QE_K, axis=1)[:, :QE_K]
        s = np.take_along_axis(S, idx, axis=1)
        w = np.clip(s, 0.0, None) ** QE_A
        W = w.sum(axis=1) + 1e-12
        wn = w / W[:, None]
        for j in range(len(ch)):
            ci = cls_index[ych[j]]
            v = axes[ci]
            mu_y = mus[ci]
            e = ch[j] - mu_y
            nb = P[idx[j]]                       # (k, d) neighbor vectors
            nb_lab = yU[idx[j]]
            same = nb_lab == ych[j]
            nu = wn[j] @ nb                      # weighted neighbor mean
            # same-class weighted neighbor noise (vs the EGO's anchor)
            xi_same = (wn[j][same, None] * (nb[same] - mu_y)).sum(axis=0)
            # foreign drift: anchor part (dbar) and full realized part
            if (~same).any():
                mu_nb = mus[[cls_index[c] for c in nb_lab[~same]]]
                dbar = (wn[j][~same, None] * (mu_nb - mu_y)).sum(axis=0)
                drift = (wn[j][~same, None] * (nb[~same] - mu_y)).sum(axis=0)
            else:
                dbar = np.zeros_like(mu_y)
                drift = np.zeros_like(mu_y)
            k = i0 + j
            out["e_v"][k] = e @ v
            out["nu_v"][k] = (nu - mu_y) @ v
            out["h_w"][k] = wn[j][same].sum()
            out["beta"][k] = W[j] / (1 + W[j])
            out["k_eff"][k] = 1.0 / (wn[j] ** 2).sum()
            out["v1_coef"][k] = (xi_same @ e) / (e @ e + 1e-12)
            out["dbar_v"][k] = dbar @ v
            out["drift_v"][k] = drift @ v
    out["y"] = yk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--out_dir", default="output/cars_qe_gate")
    ap.add_argument("--datasets", nargs="+", default=list(CONSTANTS))
    ap.add_argument("--max_per_class", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for ds in args.datasets:
        X, y, U = load(ds, args.emb_dir)
        import torch  # pool labels live in the same carve-out files
        if ds == "stanford_cars":
            yU = torch.load(os.path.join(
                args.emb_dir, "embeddings_stanford_cars_unlabeled_layers.pt"),
                map_location="cpu", weights_only=False)["labels"].numpy().astype(int)
        else:
            yU = torch.load(os.path.join(
                args.emb_dir, f"embeddings_{ds}_unlabeled.pt"),
                map_location="cpu", weights_only=False)["labels"].numpy().astype(int)

        Xn = l2n(X)
        classes = np.unique(y)
        mus = np.stack([Xn[y == c].mean(axis=0) for c in classes])
        cosm = l2n(mus) @ l2n(mus).T
        np.fill_diagonal(cosm, -np.inf)
        partner_idx = np.argmax(cosm, axis=1)

        eg = per_ego_scalars(X, y, U, yU, mus, classes, partner_idx,
                             args.max_per_class)

        h_tab, b_tab, k_tab, kap_tab = CONSTANTS[ds]
        beta = float(np.mean(eg["beta"]))
        k_eff = float(np.mean(eg["k_eff"]))
        rho_pred = float(np.sqrt((1 - beta) ** 2 + beta ** 2 / k_eff))
        S_pred = float(1 - 2 * b_tab * kap_tab * (1 - h_tab))

        # ---- numerator: mean separation shrink, per class pair ----
        # raw sep along v: E[e_v|y] - E[e_v|c] + Delta_pair ~ measured directly
        # smoothed ego (projected): (1-beta)*x_v + beta*(mu_y + nu_res)_v; use
        # per-ego beta for exactness.
        sep_ratio, sd_ratio = [], []
        var_nu_ratio, cov_ratio = [], []
        for i, c_y in enumerate(classes):
            m_y = eg["y"] == c_y
            m_c = eg["y"] == classes[partner_idx[i]]
            if m_y.sum() < 5 or m_c.sum() < 5:
                continue
            # raw projections relative to own anchors cancel in separation;
            # reconstruct absolute positions on the shared axis of class y
            v_sep_raw, v_sep_sm, sds_raw, sds_sm = [], [], [], []
            for m, ci in ((m_y, i), (m_c, partner_idx[i])):
                # both classes projected on class-y's axis for a shared frame
                v = mus[i] - mus[partner_idx[i]]
                v = v / (np.linalg.norm(v) + 1e-12)
                mu_own = mus[ci] @ v
                # e_v/nu_v were computed on the ego's OWN axis; recompute cheap
                # proxy: for the pair frame use e_v of y-egos as-is and flip
                # sign for partner egos (axes are near-antiparallel for mutual
                # nearest pairs; exact only for mutual partners).
                sgn = 1.0 if ci == i else -1.0
                g_raw = mu_own + sgn * eg["e_v"][m]
                g_sm = (mu_own + sgn * ((1 - eg["beta"][m]) * eg["e_v"][m]
                                        + eg["beta"][m] * eg["nu_v"][m]))
                v_sep_raw.append(g_raw.mean())
                v_sep_sm.append(g_sm.mean())
                sds_raw.append(g_raw.std(ddof=1))
                sds_sm.append(g_sm.std(ddof=1))
            sep_r = (v_sep_raw[0] - v_sep_raw[1])
            sep_s = (v_sep_sm[0] - v_sep_sm[1])
            if sep_r > 1e-9:
                sep_ratio.append(sep_s / sep_r)
            sd_r = np.sqrt(0.5 * (sds_raw[0] ** 2 + sds_raw[1] ** 2))
            sd_s = np.sqrt(0.5 * (sds_sm[0] ** 2 + sds_sm[1] ** 2))
            sd_ratio.append(sd_s / sd_r)
            # denominator components on class y only (own axis, exact)
            ev, nv = eg["e_v"][m_y], eg["nu_v"][m_y]
            if ev.var() > 1e-12:
                var_nu_ratio.append(nv.var(ddof=1) / ev.var(ddof=1))
                cov_ratio.append(np.cov(ev, nv)[0, 1] / ev.var(ddof=1))

        S_meas = float(np.mean(sep_ratio))
        N_meas = float(np.mean(sd_ratio))
        vnu = float(np.mean(var_nu_ratio))       # model: 1/k_eff
        cev = float(np.mean(cov_ratio))          # model: 0
        # reconstructed N^2 from components (sanity identity)
        N2_recon = ((1 - beta) ** 2 + beta ** 2 * vnu
                    + 2 * beta * (1 - beta) * cev)

        results[ds] = {
            "predicted_ratio": PREDICTED[ds], "verdict": VERDICT[ds],
            "beta": beta, "k_eff": k_eff,
            "S_pred": S_pred, "S_meas": S_meas,
            "N_pred": rho_pred, "N_meas": N_meas,
            "ratio_meas_SN": S_meas / N_meas,
            "var_nu_over_sigma2": vnu, "var_nu_model": 1.0 / k_eff,
            "cov_e_nu_over_sigma2": cev,
            "N2_reconstructed": float(N2_recon), "N2_direct": N_meas ** 2,
            "v1_retained_noise_coef": float(np.mean(eg["v1_coef"])),
            "v2_corr_e_dbar": float(np.corrcoef(
                eg["e_v"], eg["dbar_v"])[0, 1]),
            "v2_corr_e_drift": float(np.corrcoef(
                eg["e_v"], eg["drift_v"])[0, 1]),
            "h_w_check": float(np.mean(eg["h_w"])),
        }
        r = results[ds]
        print(f"{ds:14} S {r['S_pred']:.2f}->{r['S_meas']:.2f}  "
              f"N {r['N_pred']:.2f}->{r['N_meas']:.2f}  "
              f"Var(nu)/s2 {vnu:.2f} (model {1/k_eff:.2f})  "
              f"Cov/s2 {cev:.2f}  v1 {r['v1_retained_noise_coef']:.2f}  "
              f"v2 {r['v2_corr_e_dbar']:.2f}", flush=True)

    path = os.path.join(args.out_dir, "dprime_overprediction.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
