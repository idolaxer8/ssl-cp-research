"""Figures for docs/dwt_d3_learning_edition.md.

Everything is computed from closed-form D2/D3 formulas -- no data files, no
randomness. The toy universe matches Section 3 of the learning edition
(anchors at +-1, sigma_v = 0.5, k = 4 uniform neighbors, beta = 0.8,
kappa = 1); the regime map uses the measured gate constants of
output/dwt_theory/gate_constants.json (hardcoded below, same values as the
table in Section 8 / dwt_denoise_theorem.md Section 6).

Outputs docs/figs/dwt_learning_*.png (tracked; the learning edition embeds
them). Linear axes throughout.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figs")

# Toy universe (Section 3)
DELTA_PAIR = 2.0
SIGMA_V = 0.5
K_EFF_TOY = 4.0
BETA_TOY = 0.8
KAPPA_TOY = 1.0

# Measured gate constants (Section 8): h_w, beta, k_eff, kappa, tested verdict.
# stanford_cars was the registered out-of-sample cell; the run (2026-08-13,
# src/cars_qe_gate_experiment.py) came back HARM (+11/28/40% at 2/4/8
# shots/class; measured d'-ratio 0.735 vs (I)-model 1.52) -- the one cell the
# (I)-model mis-sorts; its marker keeps the star shape to stay identifiable.
DATASETS = {
    "cifar100": (0.809, 0.615, 9.38, 0.629, "gain"),
    "miniimagenet": (0.919, 0.693, 9.30, 0.717, "gain"),
    "cifar10": (0.969, 0.644, 9.58, 0.755, "gain"),
    "stanford_cars": (0.461, 0.827, 9.83, 0.585, "harm"),
    "aircraft": (0.258, 0.862, 9.87, 0.615, "harm"),
}
VERDICT_COLOR = {"gain": "tab:green", "harm": "tab:red"}
STAR = {"stanford_cars"}  # the registered-prediction cell


def rho(beta, k_eff):
    return math.sqrt((1 - beta) ** 2 + beta**2 / k_eff)


def dprime_ratio(h, beta, k_eff, kappa):
    return (1 - 2 * beta * kappa * (1 - h)) / rho(beta, k_eff)


def h_star(beta, k_eff, kappa):
    return 1 - (1 - rho(beta, k_eff)) / (2 * beta * kappa)


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def fig_battle():
    """Three-panel toy: raw vs smoothed at high/low homophily."""
    x = np.linspace(-2.6, 2.6, 800)
    r = rho(BETA_TOY, K_EFF_TOY)  # 0.447
    panels = []
    for title, h in [("raw embedding", None), ("smoothed, h = 3/4", 0.75), ("smoothed, h = 1/2", 0.50)]:
        if h is None:
            mu, sd = DELTA_PAIR / 2, SIGMA_V
        else:
            mu = (DELTA_PAIR / 2) * (1 - 2 * BETA_TOY * KAPPA_TOY * (1 - h))
            sd = r * SIGMA_V
        panels.append((title, mu, sd))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (title, mu, sd) in zip(axes, panels):
        py = gauss(x, mu, sd)
        pc = gauss(x, -mu, sd)
        ax.plot(x, py, color="tab:green", lw=2, label="class $y$")
        ax.plot(x, pc, color="tab:red", lw=2, label="class $c$")
        ax.fill_between(x, np.minimum(py, pc), color="0.55", alpha=0.55, label="confusable overlap")
        for m, c in [(mu, "tab:green"), (-mu, "tab:red")]:
            ax.axvline(m, color=c, ls="--", lw=1, alpha=0.7)
        dp = 2 * mu / sd
        ax.set_title(f"{title}\n$d' = {dp:.2f}$   (pair acc. {100 * phi(dp / 2):.1f}%)")
        ax.set_xlabel("position on the pair axis  $g(x) = \\langle x, v\\rangle$")
        ax.set_yticks([])
        ax.set_xlim(-2.6, 2.6)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "The battle D3 scores: smoothing shrinks the noise the SAME way in both panels "
        f"($\\rho = {r:.3f}$); homophily decides what happens to the signal "
        f"(toy: $\\beta={BETA_TOY}$, $k_{{\\rm eff}}={K_EFF_TOY:.0f}$, $\\kappa={KAPPA_TOY:.0f}$)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, "dwt_learning_battle.png"


def fig_d2_vs_d3():
    """Norm-level vs margin-level benefit as a function of beta (Corollary C2)."""
    betas = np.linspace(1e-3, 0.999, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, h in zip(axes, [0.75, 0.40]):
        dbar = (1 - h) * KAPPA_TOY * DELTA_PAIR  # 1-D toy: all drift on the axis
        R = 1 / K_EFF_TOY + (dbar / SIGMA_V) ** 2
        norm_benefit = 1 / np.sqrt((1 - betas) ** 2 + betas**2 * R)
        marg_benefit = np.array([dprime_ratio(h, b, K_EFF_TOY, KAPPA_TOY) for b in betas])
        ax.plot(betas, norm_benefit, color="tab:blue", lw=2, label="D2 norm level:  $\\sigma_{\\rm raw}/\\sigma_{\\rm new}$")
        ax.plot(betas, marg_benefit, color="tab:purple", lw=2, label="D3 margin level:  $d'_{\\rm new}/d'_{\\rm raw}$")
        ax.axhline(1.0, color="k", lw=1)
        ax.axvline(BETA_TOY, color="0.5", ls=":", lw=1.5)
        ax.text(BETA_TOY - 0.02, -0.38, "deployed $\\beta$", fontsize=8, color="0.35", ha="right")
        ax.fill_between(betas, 1.0, np.maximum(marg_benefit, 1.0), color="tab:purple", alpha=0.12)
        ax.set_xlabel("smoothing strength $\\beta$")
        ax.set_title(f"homophily $h = {h}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 1.6)
    axes[0].set_ylabel("benefit factor  (> 1 = improvement)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].annotate(
        "norm level promises\na small-$\\beta$ rescue...",
        xy=(0.14, 1.08), xytext=(0.28, 1.32), fontsize=9, color="tab:blue",
        arrowprops=dict(arrowstyle="->", color="tab:blue"),
    )
    axes[1].annotate(
        "...margin level: below the gate,\nNO $\\beta$ helps (harm not tunable)",
        xy=(0.70, 0.33), xytext=(0.16, -0.10), fontsize=9, color="tab:purple",
        arrowprops=dict(arrowstyle="->", color="tab:purple"),
    )
    fig.suptitle(
        "Why the theorem must live at the margin level: D2 and D3 genuinely disagree at low homophily "
        f"(toy: $k_{{\\rm eff}}={K_EFF_TOY:.0f}$, $\\kappa={KAPPA_TOY:.0f}$) — and the data sides with D3",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, "dwt_learning_d2_vs_d3.png"


def fig_regime_map():
    """Two views of the gate on the 5 measured datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # Panel 1: d'-ratio vs homophily; one thin law-curve per dataset's constants
    hs = np.linspace(0, 1, 300)
    label_at_1 = {  # explicit label positions (data coords) to avoid collisions
        "cifar100": (0.68, 1.70),
        "miniimagenet": (0.66, 2.55),
        "cifar10": (0.86, 2.05),
        "stanford_cars": (0.33, 1.60),
        "aircraft": (0.20, 0.45),
    }
    for name, (h_w, beta, k_eff, kappa, verdict) in DATASETS.items():
        curve = [dprime_ratio(h, beta, k_eff, kappa) for h in hs]
        ax1.plot(hs, curve, color="0.75", lw=1, zorder=1)
        y = dprime_ratio(h_w, beta, k_eff, kappa)
        ax1.scatter([h_w], [y], s=90 if name in STAR else 70,
                    color=VERDICT_COLOR[verdict], zorder=3,
                    marker="*" if name in STAR else "o",
                    edgecolor="k", linewidth=0.6)
        leader = dict(arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7)) if name in ("miniimagenet", "cifar10") else {}
        ax1.annotate(name, (h_w, y), xytext=label_at_1[name], fontsize=8, **leader)
    ax1.annotate(
        "cars ran 2026-08-13: qe HARMED (+11-40%)\n"
        "measured $d'$-ratio 0.73, not 1.52 —\n"
        "the (I)-model's one mis-sorted cell (V1/V2)",
        xy=(0.461, 1.52), xytext=(0.06, 2.35), fontsize=8, color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.9))
    ax1.axhline(1.0, color="k", lw=1)
    ax1.text(0.02, 1.03, "help", fontsize=8, color="0.3")
    ax1.text(0.02, 0.90, "harm", fontsize=8, color="0.3")
    ax1.axvline(0.7, color="tab:red", ls="--", lw=1.2)
    ax1.text(0.705, 2.7, "folklore gate 0.7\n(wrong operator)", fontsize=8, color="tab:red")
    ax1.set_xlabel("measured weighted homophily $h_w$")
    ax1.set_ylabel("(I)-model predicted $d'$-ratio")
    ax1.set_title("(I)-model law curves. The registered cars cell ($\\star$) came back\nHARM — the empirical gate sits between $h$ = 0.46 and 0.81")
    ax1.set_xlim(0, 1)

    # Panel 2: the (beta, h) plane with the h*(beta) gate frontier
    betas = np.linspace(0.5, 0.95, 300)
    k_typ, kap_typ = 9.6, 0.62
    frontier = np.array([h_star(b, k_typ, kap_typ) for b in betas])
    ax2.plot(betas, frontier, color="k", lw=2, label="gate $h^{\\ast}(\\beta)$  ($\\kappa=0.62$, $k_{\\rm eff}=9.6$)")
    ax2.fill_between(betas, frontier, 1.0, color="tab:green", alpha=0.10)
    ax2.fill_between(betas, 0.0, frontier, color="tab:red", alpha=0.10)
    ax2.text(0.515, 0.55, "smoothing helps ($h > h^{\\ast}$)", fontsize=9, color="tab:green")
    ax2.text(0.62, 0.16, "smoothing harms — at every $\\beta$", fontsize=9, color="tab:red")
    label_at_2 = {  # explicit label positions (data coords) to avoid collisions
        "cifar100": (0.545, 0.775),
        "miniimagenet": (0.705, 0.885),
        "cifar10": (0.555, 0.935),
        "stanford_cars": (0.755, 0.395),
        "aircraft": (0.795, 0.295),
    }
    for name, (h_w, beta, k_eff, kappa, verdict) in DATASETS.items():
        ax2.scatter([beta], [h_w], s=90 if name in STAR else 70,
                    color=VERDICT_COLOR[verdict],
                    marker="*" if name in STAR else "o",
                    edgecolor="k", linewidth=0.6, zorder=3)
        ax2.annotate(name, (beta, h_w), xytext=label_at_2[name], fontsize=8)
    ax2.axhline(0.7, color="tab:red", ls="--", lw=1.2)
    ax2.text(0.51, 0.715, "folklore 0.7", fontsize=8, color="tab:red")
    ax2.annotate("cars: HARM despite sitting above the\n(I)-model frontier — the frontier is a FLOOR;\nthe true gate lies in (0.46, 0.81)",
                 xy=(0.827, 0.461), xytext=(0.512, 0.30), fontsize=8,
                 color="tab:red",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.9))
    ax2.set_xlabel("self-tuned smoothing strength $\\beta = W/(1+W)$")
    ax2.set_ylabel("measured homophily $h_w$")
    ax2.set_title("The (I)-model frontier is a floor, not the boundary (§8/§9):\ncars confirms V1/V2 push the true gate up")
    ax2.set_xlim(0.5, 0.95)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    return fig, "dwt_learning_regime_map.png"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for builder in [fig_battle, fig_d2_vs_d3, fig_regime_map]:
        fig, name = builder()
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("saved", os.path.normpath(path))


if __name__ == "__main__":
    main()
