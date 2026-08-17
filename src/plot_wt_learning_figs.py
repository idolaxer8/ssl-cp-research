"""Figures for docs/dwt_wt_learning_edition.md (lemmas W1/T1).

Everything is closed-form / hardcoded-measured -- no data files, no
randomness. The 2-D toy matches Section 4 of the learning edition
(Sigma = diag(4, 1/4), delta = (1.2, 0.9)); the measured panels hardcode
output/dwt_theory/wt_phase_diagnostics.json (run 2026-08-16).

Outputs docs/figs/dwt_wt_learning_*.png (tracked). Linear axes throughout.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figs")

# ---- Section-4 toy -----------------------------------------------------
SIGMA = np.diag([4.0, 0.25])          # nuisance variance along x1
DELTA = np.array([1.2, 0.9])          # pair separation, mixed alignment

# ---- measured diagnostics (wt_phase_diagnostics.json, 2026-08-16) ------
DS = ["cifar100", "miniimagenet", "cifar10", "stanford_cars", "aircraft"]
RATIOS = {  # ds -> (wdiag, wlw, t128, t128w) ratio vs raw, normalized d'
    "cifar100":      (1.00, 1.03, 1.00, 1.00),
    "miniimagenet":  (1.00, 0.91, 1.17, 1.11),
    "cifar10":       (1.00, 1.05, 1.09, 1.07),
    "stanford_cars": (1.01, 1.87, 0.89, 1.34),
    "aircraft":      (1.03, 3.13, 0.89, 1.77),
}
A128 = {"cifar100": 0.82, "miniimagenet": 0.93, "cifar10": 0.99,
        "stanford_cars": 0.85, "aircraft": 0.87}
CHANG_TAIL = {"cifar100": 0.45, "miniimagenet": 0.24, "cifar10": 0.15,
              "stanford_cars": 0.79, "aircraft": 0.83}
PR = {"cifar100": 243, "miniimagenet": 255, "cifar10": 119,
      "stanford_cars": 24, "aircraft": 16}
SHORT = {"cifar100": "c100", "miniimagenet": "mini", "cifar10": "c10",
         "stanford_cars": "cars", "aircraft": "airc"}


def ellipse(mu, cov, nsig=1.0, n=200):
    t = np.linspace(0, 2 * math.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    w, V = np.linalg.eigh(cov)
    pts = V @ np.diag(np.sqrt(w)) @ circ * nsig
    return mu[0] + pts[0], mu[1] + pts[1]


def dprime(M, Sigma, delta):
    num = float(delta @ M @ delta)
    den = float(delta @ M @ Sigma @ M @ delta)
    return num / math.sqrt(den)


def fig_whiten_toy():
    """W1a: raw vs whitened 2-D toy -- the metric rotation is the win."""
    Sig_inv = np.linalg.inv(SIGMA)
    Sig_mh = np.diag(1.0 / np.sqrt(np.diag(SIGMA)))   # Sigma^{-1/2}
    mu_p, mu_m = DELTA / 2, -DELTA / 2
    d_raw = dprime(np.eye(2), SIGMA, DELTA)
    d_max = math.sqrt(float(DELTA @ Sig_inv @ DELTA))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    for mu, c in ((mu_p, "tab:green"), (mu_m, "tab:red")):
        for ns in (1, 2):
            ex, ey = ellipse(mu, SIGMA, nsig=ns * 0.5)
            ax.plot(ex, ey, color=c, alpha=0.9 if ns == 1 else 0.35)
        ax.plot(*mu, "o", color=c, ms=8)
    v = DELTA / np.linalg.norm(DELTA)
    ax.annotate("", xytext=(-1.6 * v[0], -1.6 * v[1]),
                xy=(1.6 * v[0], 1.6 * v[1]),
                arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax.set_title(f"raw space: pair axis fights the nuisance\n"
                 f"$d'(I) = {d_raw:.2f}$")
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)

    ax = axes[1]
    mu_pw, mu_mw = Sig_mh @ mu_p, Sig_mh @ mu_m
    dw = Sig_mh @ DELTA
    for mu, c in ((mu_pw, "tab:green"), (mu_mw, "tab:red")):
        for ns in (1, 2):
            ex, ey = ellipse(mu, np.eye(2), nsig=ns * 0.5)
            ax.plot(ex, ey, color=c, alpha=0.9 if ns == 1 else 0.35)
        ax.plot(*mu, "o", color=c, ms=8)
    vw = dw / np.linalg.norm(dw)
    ax.annotate("", xytext=(-1.6 * vw[0], -1.6 * vw[1]),
                xy=(1.6 * vw[0], 1.6 * vw[1]),
                arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax.annotate("", xytext=(-1.6 * v[0], -1.6 * v[1]),
                xy=(1.6 * v[0], 1.6 * v[1]),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="gray",
                                linestyle=":"))
    ax.set_title(f"whitened space: noise is round, axis ROTATES\n"
                 f"$d'(\\Sigma^{{-1}}) = {d_max:.2f}$  "
                 f"(x{d_max / d_raw:.1f}, no labels needed: W1b)")
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    fig.suptitle("W1a in one picture: whitening = make the noise round, "
                 "then the honest axis appears", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dwt_wt_learning_whiten_toy.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_chang():
    """T1: signal ENERGY vs DISCRIMINANT contribution per PC (toy) and the
    measured top-128 split per dataset."""
    j = np.arange(1, 11)
    lam = 10 * 0.55 ** j + 0.05
    # signal energy: mostly top-spectrum, small tail component
    energy = np.array([.30, .22, .15, .09, .06, .05, .04, .04, .03, .02])
    contrib = energy / lam
    contrib = contrib / contrib.sum()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    w = 0.38
    ax.bar(j - w / 2, energy, w, label=r"energy $(u_j^\top\delta)^2$ (norm.)",
           color="tab:blue")
    ax.bar(j + w / 2, contrib, w,
           label=r"discriminant $(u_j^\top\delta)^2/\lambda_j$ (norm.)",
           color="tab:orange")
    ax.set_xlabel("PC index $j$ (variance-sorted)")
    ax.set_title("Chang 1983 toy: the axis ENERGY sits in top PCs,\n"
                 "the DISCRIMINANT sits in the tail ($1/\\lambda$ weighting)")
    ax.legend(); ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    x = np.arange(len(DS))
    ax.bar(x - w / 2, [A128[d] for d in DS], w,
           label="energy kept by PC-128 ($a_{128}$)", color="tab:blue")
    ax.bar(x + w / 2, [1 - CHANG_TAIL[d] for d in DS], w,
           label="discriminant kept by PC-128", color="tab:orange")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(x, [SHORT[d] for d in DS])
    ax.set_ylim(0, 1.05)
    ax.set_title("measured: the two currencies agree on easy data,\n"
                 "split on fine-grained (cars/aircraft keep ~0.2 of "
                 "discriminant)")
    ax.legend(loc="lower left"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dwt_wt_learning_chang.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_t1b():
    """T1b: effective d' vs kept dimension m, for several shot counts."""
    K, A_inf = 100, 16.0
    m = np.linspace(4, 768, 500)
    A_m = A_inf * (1 - np.exp(-m / K))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for s, c in ((2, "tab:red"), (8, "tab:orange"), (32, "tab:blue")):
        d_eff = A_m / np.sqrt(A_m + 2 * m / s)
        ax.plot(m, d_eff, color=c, label=f"$s={s}$ shots/class")
        i = int(np.argmax(d_eff))
        ax.plot(m[i], d_eff[i], "o", color=c, ms=7)
    ax.plot(m, np.sqrt(A_m), "k--", lw=1.2,
            label=r"$s=\infty$ (population: $\sqrt{A_m}$, monotone)")
    ax.axvline(128, color="gray", lw=0.8, linestyle=":")
    ax.text(133, 0.4, "m=128", color="gray")
    ax.set_xlabel("kept dimension $m$")
    ax.set_ylabel(r"$d'_{\rm eff}(m,s) = A_m/\sqrt{A_m + 2m/s}$")
    ax.set_title("T1b: truncation's real gain is finite-shot — the optimum\n"
                 "sits near $m \\approx K$ at small $s$ and vanishes as "
                 "$s \\to \\infty$")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dwt_wt_learning_t1b.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_phase_map():
    """Measured d'-ratios per space per dataset (the W/T phase map)."""
    labels = ["wdiag", "wlw (full-rank)", "t128", "t128w"]
    colors = ["tab:gray", "tab:purple", "tab:blue", "tab:cyan"]
    x = np.arange(len(DS))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, (lab, c) in enumerate(zip(labels, colors)):
        vals = [RATIOS[d][i] for d in DS]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c)
    ax.axhline(1.0, color="k", lw=0.9)
    ax.set_xticks(x, [f"{SHORT[d]}\nPR {PR[d]}" for d in DS])
    ax.set_ylabel("pair $d'$ ratio vs raw (normalized)")
    ax.set_title("the measured W/T phase map: full-rank whitening is the\n"
                 "fine-grained lever (off-diagonal); nothing moves "
                 "population $d'$ on easy data")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dwt_wt_learning_phase_map.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_whiten_toy()
    fig_chang()
    fig_t1b()
    fig_phase_map()
    print("wrote 4 figs ->", OUT_DIR)
