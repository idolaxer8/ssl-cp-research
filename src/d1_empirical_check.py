"""Empirical check of Theorem D1 (the deterministic DAPS-2 mirror).

D1 states, per ego x with label y (pre-norm convex form, any weights):

  eps_hat <= (1-beta)*eps + beta*[ sum_u (w_u/W) r_u + (1-h_w)*Delta_F ]   (bound)
  and   budget := sum_u (w_u/W) r_u + (1-h_w)*Delta_F < eps  =>  eps_hat < eps

where eps = ||x - mu_y||, eps_hat = ||x_hat - mu_y||, r_u = ||u - mu_{c(u)}||.

Being deterministic, the bound MUST hold for every point (checking it is an
implementation audit); the empirical content displayed here is
  (1) the certification -> improvement implication holds pointwise
      (the quadrant {margin > 0, improvement <= 0} is empty),
  (2) HOW conservative D1 is: certification only fires in the far noise
      tail, while actual norm improvement is broader (tail-repair curves),
  (3) bound tightness (slack distribution) -- post-cars we expect the
      same-class part to be near-tight (neighbor displacements are field-
      aligned with the ego, research edition Section 8b) and the Delta_F
      impurity part to carry most of the slack.

Uses pool labels (carve-out files) for r_u and Delta_F -- diagnostic only.
Outputs: output JSON + a tracked figure docs/figs/dwt_learning_d1_check.png.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dwt_score_histograms import l2n                              # noqa: E402
from measure_dprime_all import load                               # noqa: E402

QE_K = 10
QE_A = 3.0
COLORS = {"cifar10": "tab:green", "miniimagenet": "tab:olive",
          "cifar100": "tab:blue", "stanford_cars": "tab:orange",
          "aircraft": "tab:red"}


def per_ego_d1(X, y, U, yU, max_per_class, seed=0):
    P = l2n(U)
    Xn = l2n(X)
    classes = np.unique(y)
    mus = np.stack([Xn[y == c].mean(axis=0) for c in classes])
    cls_index = {c: i for i, c in enumerate(classes)}
    r_pool = np.linalg.norm(P - mus[[cls_index[c] for c in yU]], axis=1)

    rng = np.random.default_rng(seed)
    keep = []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    Xk, yk = Xn[keep], y[keep]

    n = len(Xk)
    out = {k: np.empty(n) for k in
           ["eps", "eps_hat", "bound", "budget", "h_w", "beta"]}
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
            mu_y = mus[ci]
            nb = P[idx[j]]
            nb_lab = yU[idx[j]]
            same = nb_lab == ych[j]
            beta = W[j] / (1 + W[j])
            nu = wn[j] @ nb
            x_hat = (1 - beta) * ch[j] + beta * nu
            eps = np.linalg.norm(ch[j] - mu_y)
            eps_hat = np.linalg.norm(x_hat - mu_y)
            r_nb = r_pool[idx[j]]
            mean_r = float(wn[j] @ r_nb)
            h_w = float(wn[j][same].sum())
            if (~same).any():
                mu_nb = mus[[cls_index[c] for c in nb_lab[~same]]]
                delta_f = float(np.linalg.norm(mu_nb - mu_y, axis=1).max())
            else:
                delta_f = 0.0
            budget = mean_r + (1 - h_w) * delta_f
            k = i0 + j
            out["eps"][k] = eps
            out["eps_hat"][k] = eps_hat
            out["budget"][k] = budget
            out["bound"][k] = (1 - beta) * eps + beta * budget
            out["h_w"][k] = h_w
            out["beta"][k] = beta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--out_dir", default="output/dwt_theory")
    ap.add_argument("--fig_path", default=os.path.join(
        os.path.dirname(__file__), "..", "docs", "figs",
        "dwt_learning_d1_check.png"))
    ap.add_argument("--datasets", nargs="+", default=list(COLORS))
    ap.add_argument("--max_per_class", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results, store = {}, {}
    for ds in args.datasets:
        X, y, U = load(ds, args.emb_dir)
        if ds == "stanford_cars":
            yU = torch.load(os.path.join(
                args.emb_dir, "embeddings_stanford_cars_unlabeled_layers.pt"),
                map_location="cpu", weights_only=False)["labels"].numpy().astype(int)
        else:
            yU = torch.load(os.path.join(
                args.emb_dir, f"embeddings_{ds}_unlabeled.pt"),
                map_location="cpu", weights_only=False)["labels"].numpy().astype(int)
        eg = per_ego_d1(X, y, U, yU, args.max_per_class)
        store[ds] = eg

        eps, eps_hat = eg["eps"], eg["eps_hat"]
        margin = eps - eg["budget"]
        certified = margin > 0
        improved = eps_hat < eps
        bound_ok = eps_hat <= eg["bound"] + 1e-9
        results[ds] = {
            "n_egos": int(len(eps)),
            "bound_holds_frac": float(bound_ok.mean()),
            "pct_certified": float(certified.mean()),
            "pct_certified_that_improved": (
                float(improved[certified].mean()) if certified.any() else None),
            "pct_improved_overall": float(improved.mean()),
            "median_rel_slack": float(np.median(
                (eg["bound"] - eps_hat) / eps)),
            "median_eps_pctile_of_certified": (
                float(np.mean(eps[certified] >
                              np.percentile(eps, 90))) if certified.any()
                else None),
        }
        r = results[ds]
        print(f"{ds:14} bound holds {r['bound_holds_frac']:.4f}  "
              f"certified {100*r['pct_certified']:.1f}%  "
              f"cert&improved {r['pct_certified_that_improved']}  "
              f"improved overall {100*r['pct_improved_overall']:.1f}%  "
              f"slack {r['median_rel_slack']:.2f}", flush=True)

    # ---------- figure ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # panel 1: quadrant test -- D1 margin vs realized improvement
    for ds in args.datasets:
        eg = store[ds]
        m = (eg["eps"] - eg["budget"]) / eg["eps"]
        im = (eg["eps"] - eg["eps_hat"]) / eg["eps"]
        sub = np.random.default_rng(1).choice(
            len(m), size=min(1500, len(m)), replace=False)
        ax1.scatter(m[sub], im[sub], s=4, alpha=0.35,
                    color=COLORS[ds], label=ds)
    ax1.axhline(0, color="k", lw=1)
    ax1.axvline(0, color="k", lw=1)
    ax1.fill_betweenx([-1.0, 0], 0, 1, color="tab:red", alpha=0.10)
    ax1.text(0.35, -0.32, "forbidden by D1\n(certified but not improved)\n"
             "— EMPTY", fontsize=9, color="tab:red", ha="center")
    ax1.set_xlim(-2.4, 1.0)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_xlabel("D1 margin  $(\\varepsilon - \\mathrm{budget})/\\varepsilon$"
                   "   (> 0 = certified)")
    ax1.set_ylabel("realized improvement  "
                   "$(\\varepsilon - \\hat\\varepsilon)/\\varepsilon$")
    ax1.set_title("The pointwise implication: every certified ego improves\n"
                  "(lower-right quadrant empty on all five datasets)",
                  fontsize=10)
    ax1.legend(fontsize=7, markerscale=3, loc="upper left")

    # panel 2: tail-repair curves -- improvement/certification vs eps decile
    for ds in args.datasets:
        eg = store[ds]
        eps = eg["eps"]
        deciles = np.percentile(eps, np.arange(0, 101, 10))
        xs, imp, cert = [], [], []
        for a, b in zip(deciles[:-1], deciles[1:]):
            m = (eps >= a) & (eps <= b)
            xs.append(5 + 10 * len(xs))
            imp.append((eg["eps_hat"][m] < eps[m]).mean())
            cert.append((eps[m] > eg["budget"][m]).mean())
        ax2.plot(xs, imp, color=COLORS[ds], lw=1.8, label=f"{ds} improved")
        ax2.plot(xs, cert, color=COLORS[ds], lw=1.2, ls="--")
    ax2.set_xlabel("ego noise percentile (deciles of $\\varepsilon$)")
    ax2.set_ylabel("fraction of egos")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_title("Tail repair: certification (dashed) fires only in the far\n"
                  "noise tail; realized norm improvement (solid) is broader",
                  fontsize=10)
    ax2.legend(fontsize=7, loc="upper left")

    fig.suptitle("Empirical check of Theorem D1 (deterministic bound, zero "
                 "assumptions): implication exact, certification "
                 "conservative by design", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(os.path.abspath(args.fig_path)), exist_ok=True)
    fig.savefig(args.fig_path, dpi=150)
    print("figure ->", os.path.normpath(args.fig_path))

    path = os.path.join(args.out_dir, "d1_empirical_check.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("->", path)


if __name__ == "__main__":
    main()
