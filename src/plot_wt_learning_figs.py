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


def fig_metric_family():
    """The estimator-family dial on the napkin toy: every W/T menu member
    estimates the same W1a-optimal metric Sigma^{-1}; shrinkage slides
    along one curve, truncation jumps off it."""
    c = float(np.trace(SIGMA)) / 2          # trace-preserving shrink target
    ts = np.linspace(0, 1, 300)
    dps = []
    for t in ts:
        M = np.linalg.inv((1 - t) * SIGMA + t * c * np.eye(2))
        dps.append(dprime(M, SIGMA, DELTA))
    dps = np.array(dps)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(ts, dps, color="tab:purple", lw=2,
            label=r"shrunk metric $M_t = ((1{-}t)\Sigma + t\,\bar\lambda I)^{-1}$")
    ax.plot(0, dps[0], "o", color="tab:purple", ms=8)
    ax.annotate(f"$t=0$: oracle whitening\n$d' = {dps[0]:.2f}$ (W1a max)",
                (0, dps[0]), xytext=(0.06, 1.78), fontsize=9)
    ax.plot(1, dps[-1], "o", color="tab:gray", ms=8)
    ax.annotate(f"$t=1$: identity metric\n$d' = {dps[-1]:.2f}$ (raw)",
                (1, dps[-1]), xytext=(0.68, 1.05), fontsize=9)
    # Ledoit-Wolf: picks t from the data, more samples -> smaller t
    ax.annotate("Ledoit-Wolf / CNAPS blend:\npick $t$ from sample size\n"
                "(more shots $\\to$ smaller $t$)",
                (0.4, float(np.interp(0.4, ts, dps))),
                xytext=(0.36, 1.52), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    # truncation to the top-variance PC: off the curve
    d_trunc = dprime(np.diag([1.0, 0.0]), SIGMA, DELTA)
    ax.plot(0.5, d_trunc, "X", color="tab:red", ms=11)
    ax.annotate("rank-1 truncation (keep top-variance PC):\n"
                f"$d' = {d_trunc:.2f}$ — OFF the curve (Chang: kept the\n"
                "loud direction, dropped the discriminant)",
                (0.5, d_trunc), xytext=(0.13, 0.62), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ax.set_xlabel("shrinkage intensity $t$  (regularization toward "
                  "$\\bar\\lambda I$)")
    ax.set_ylabel("pair $d'$ achieved on the napkin toy")
    ax.set_ylim(0.4, 2.05)
    ax.set_title("one family: every menu member estimates the SAME metric "
                 "$\\Sigma^{-1}$;\nshrinkage slides along the curve — "
                 "blind truncation can jump off it")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dwt_wt_learning_metric_family.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# chain-free diagnostics (wt_chainfree_diagnostics.json, 2026-08-18)
CERT = {  # D1 certification rate per space
    "cifar100":      (0.417, 0.414, 0.338, 0.466, 0.455),
    "miniimagenet":  (0.638, 0.635, 0.543, 0.637, 0.605),
    "cifar10":       (0.637, 0.636, 0.768, 0.559, 0.592),
    "stanford_cars": (0.077, 0.077, 0.116, 0.068, 0.091),
    "aircraft":      (0.027, 0.029, 0.150, 0.023, 0.026),
}
AERR2 = {  # 2-shot relative anchor error per space
    "cifar100":      (1.15, 1.16, 1.72, 0.82, 0.85),
    "miniimagenet":  (0.86, 0.86, 1.50, 0.53, 0.57),
    "cifar10":       (1.21, 1.21, 1.53, 0.90, 0.96),
    "stanford_cars": (1.52, 1.52, 1.71, 1.51, 1.39),
    "aircraft":      (2.67, 2.65, 2.51, 2.72, 2.22),
}
SPACE_LABELS = ["raw", "wdiag", "wlw", "t128", "t128w"]
SPACE_COLORS = ["tab:gray", "silver", "tab:purple", "tab:blue", "tab:cyan"]


def fig_chainfree():
    """The chain-free grounding: W extends D1's certification reach,
    T shrinks anchor error."""
    x = np.arange(len(DS))
    w = 0.16
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for i, (lab, c) in enumerate(zip(SPACE_LABELS, SPACE_COLORS)):
        ax.bar(x + (i - 2) * w, [CERT[d][i] for d in DS], w, label=lab,
               color=c)
    ax.set_xticks(x, [SHORT[d] for d in DS])
    ax.set_ylabel("D1 certification rate (higher = better)")
    ax.set_title("W extends the BASE LEMMA's reach where h is low:\n"
                 "aircraft $.027 \\to .150$ ($\\times 5.5$) under full-rank "
                 "whitening")
    ax.legend(ncol=2); ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    for i, (lab, c) in enumerate(zip(SPACE_LABELS, SPACE_COLORS)):
        ax.bar(x + (i - 2) * w, [AERR2[d][i] for d in DS], w, label=lab,
               color=c)
    ax.set_xticks(x, [SHORT[d] for d in DS])
    ax.set_ylabel("2-shot anchor error / pair separation (lower = better)")
    ax.set_title("T shrinks the ESTIMATED-ANCHOR error (m/s vs d/s):\n"
                 "truncation wins everywhere, full-rank W pays for its "
                 "rotation")
    ax.legend(ncol=2); ax.grid(alpha=0.25, axis="y")
    fig.suptitle("the chain-free grounding: W serves neighborhoods "
                 "(D1's reach), T serves anchors", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dwt_wt_learning_chainfree.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_whiten_toy()
    fig_chang()
    fig_t1b()
    fig_phase_map()
    fig_metric_family()
    fig_chainfree()
    print("wrote 6 figs ->", OUT_DIR)
