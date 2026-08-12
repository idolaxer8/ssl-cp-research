"""Measure the constants of Theorem D3 (docs/dwt_denoise_theorem.md) per dataset.

For the deployed classic alpha-QE map T(x) = L2norm(x + sum_{u in NN_k(x)}
max(cos(x,u),0)^a * u) acting on raw L2-normed embeddings, measure per ego:

  beta_hat = W/(1+W)          implied convex smoothing weight (self-weight 1)
  k_eff    = W^2 / sum w^2    Kish effective neighbor count
  h_w      = same-class weight fraction of the neighborhood (weighted homophily)
  kappa    = D(x) / ((1-h_w) * Delta_pair(y))   drift concentration, where
             D(x) = sum_{u foreign} (w_u/W) <mu_y - mu_{c(u)}, v_y> is the
             realized drift down the nearest-confusable-pair axis
             v_y = (mu_y - mu_{c*})/Delta_pair, c* = argmin_c ||mu_c - mu_y||.

Then the scale-free gate of Theorem D3:

  rho  = sqrt((1-beta)^2 + beta^2/k_eff)        margin-noise shrink factor
  h*   = 1 - (1-rho) / (2*beta*kappa)           break-even homophily
  d'ratio = (1 - 2*beta*kappa*(1-h_w)) / rho    predicted standardized-margin
                                                 change (>1 gain, <1 harm)

Also reports sigma_v (within-class noise s.d. along the pair axis), sigma_tot
(full residual norm scale) and Delta_pair, for the norm-vs-margin discussion.
All quantities are diagnostics measured on labeled data + the labeled pool
carve-outs; the deployed gate itself must remain label-free (open blocker).
"""

import argparse
import json
import os

import numpy as np
import torch

EMB_DIR = os.path.join("output", "from_cluster", "embeddings")

DATASETS = {
    "cifar100": ("embeddings_cifar100.pt", "embeddings_cifar100_unlabeled.pt"),
    "miniimagenet": ("embeddings_miniimagenet.pt", "embeddings_miniimagenet_unlabeled.pt"),
    "aircraft": ("embeddings_aircraft.pt", "embeddings_aircraft_unlabeled.pt"),
    "stanford_cars": ("embeddings_stanford_cars_layers.pt", "embeddings_stanford_cars_unlabeled_layers.pt"),
    "cifar10": ("embeddings_cifar10.pt", "embeddings_cifar10_unlabeled.pt"),
}


def load_pt(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    emb = d["embeddings"] if "embeddings" in d else d["final"]
    return emb.float().numpy(), d["labels"].numpy()


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def measure(ds, emb_dir, k=10, a=3.0, n_egos=4000, seed=0):
    lab_f, pool_f = DATASETS[ds]
    X, y = load_pt(os.path.join(emb_dir, lab_f))
    U, yu = load_pt(os.path.join(emb_dir, pool_f))
    X, U = l2n(X.astype(np.float64)), l2n(U.astype(np.float64))

    K = int(max(y.max(), yu.max())) + 1
    mu = np.stack([X[y == c].mean(axis=0) for c in range(K)])

    # nearest-confusable pair axis per class
    D2 = ((mu[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D2, np.inf)
    cstar = D2.argmin(axis=1)
    delta_pair = np.sqrt(D2[np.arange(K), cstar])
    v = (mu - mu[cstar]) / delta_pair[:, None]

    # within-class noise: along the pair axis and in full norm
    sig_v2, sig_tot2, nres = 0.0, 0.0, 0
    for c in range(K):
        E = X[y == c] - mu[c]
        sig_v2 += ((E @ v[c]) ** 2).sum()
        sig_tot2 += (E ** 2).sum()
        nres += len(E)
    sigma_v = float(np.sqrt(sig_v2 / nres))
    sigma_tot = float(np.sqrt(sig_tot2 / nres))

    rng = np.random.default_rng(seed)
    idx_egos = rng.permutation(len(X))[:n_egos]

    h_w_all, h_unw_all, beta_all, keff_all = [], [], [], []
    kappa_num, kappa_den = 0.0, 0.0  # ratio-of-means (stabler than mean-of-ratios)
    kappa_pt = []
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
        cn = yu[nn]                        # neighbor labels
        same = cn == yc[:, None]
        h_w = np.where(ok, (w * same).sum(axis=1) / np.maximum(W, 1e-12), np.nan)
        h_w_all.append(h_w[ok])
        h_unw_all.append(same.mean(axis=1))
        # drift down the pair axis: sum_{foreign} (w/W) <mu_y - mu_cn, v_y>
        proj = ((mu[yc[:, None]] - mu[cn]) * v[yc][:, None, :]).sum(-1)
        drift = (w * (~same) * proj).sum(axis=1) / np.maximum(W, 1e-12)
        imp = np.where(ok, (w * (~same)).sum(axis=1) / np.maximum(W, 1e-12), 0.0)
        den = imp * delta_pair[yc]
        m = ok & (imp > 0.02)
        kappa_num += drift[m].sum()
        kappa_den += den[m].sum()
        kappa_pt.append(np.where(den > 1e-12, drift / den, np.nan)[m])

    h_w = float(np.concatenate(h_w_all).mean())
    h_unw = float(np.concatenate(h_unw_all).mean())
    beta = float(np.concatenate(beta_all).mean())
    keff = float(np.concatenate(keff_all).mean())
    kappa = float(kappa_num / kappa_den) if kappa_den > 0 else float("nan")
    kappa_med = float(np.nanmedian(np.concatenate(kappa_pt))) if kappa_pt else float("nan")

    rho = float(np.sqrt((1 - beta) ** 2 + beta ** 2 / keff))
    hstar = 1 - (1 - rho) / (2 * beta * kappa)
    dratio = (1 - 2 * beta * kappa * (1 - h_w)) / rho

    return dict(dataset=ds, K=K, n_labeled=len(X), n_pool=len(U),
                h_w=h_w, h_unweighted=h_unw, beta_hat=beta, k_eff=keff,
                kappa=kappa, kappa_median=kappa_med, rho=rho, h_star=hstar,
                d_prime_ratio=dratio, sigma_v=sigma_v, sigma_tot=sigma_tot,
                delta_pair_mean=float(delta_pair.mean()),
                axis_snr=float(delta_pair.mean() / sigma_v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default=EMB_DIR)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--a", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join("output", "dwt_theory", "gate_constants.json"))
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        r = measure(ds, args.emb_dir, k=args.k, a=args.a)
        rows.append(r)
        print(f"{ds:14s} h_w={r['h_w']:.3f} (unw {r['h_unweighted']:.3f}) "
              f"beta={r['beta_hat']:.3f} k_eff={r['k_eff']:.2f} kappa={r['kappa']:.3f} "
              f"rho={r['rho']:.3f} h*={r['h_star']:.3f} d'ratio={r['d_prime_ratio']:.3f} "
              f"axis_snr={r['axis_snr']:.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
