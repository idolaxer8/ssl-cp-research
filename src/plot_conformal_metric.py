"""Plots for the conformal-metric-learning (G1) experiments.
Linear y-axes throughout (user preference; no log scales)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_landscape(out, path):
    """Phase-1 go/no-go scatter: pool rehearsal objective vs TRUE FCP size."""
    probes = out["probes"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for wh, col in [("cluster", "tab:blue"), ("lw_cluster", "tab:red")]:
        ps = [p for p in probes if p["whiten"] == wh]
        if not ps:
            continue
        ax.errorbar([p["rehearsal_sz"] for p in ps],
                    [p["true_sz"] for p in ps],
                    yerr=[p["true_sz_se"] for p in ps],
                    fmt="o", color=col, label=f"whiten={wh}", alpha=0.75)
    for p in probes:
        if p.get("tag"):
            ax.annotate(p["tag"], (p["rehearsal_sz"], p["true_sz"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("pool rehearsal set size (surrogate, half-B)")
    ax.set_ylabel(f"true FCP set size (cal={out['gt_cal']}, geodesic)")
    ax.set_title(f"{out['dataset']}: Spearman={out['spearman']:+.2f}, "
                 f"argmin-regret={out['argmin_regret']:+.1%} "
                 f"-> {out['gate']}  (PR={out['pool_pr']:.0f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved -> {path}")


def plot_spectrum(report, ds, path):
    """Mechanism figure: learned s_j vs eigen-index, with the d'=128/512
    reference gates."""
    s = np.asarray(report["s_final"])
    s1 = np.asarray(report.get("s_rung1", s))
    j = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(j, s1 / max(s1.max(), 1e-12), label="s rung-1 (normalized)",
            color="tab:blue")
    if "s_rung2" in report:
        s2 = np.asarray(report["s_rung2"])
        ax.plot(j, s2 / max(s2.max(), 1e-12), label="s rung-2 (normalized)",
                color="tab:green", alpha=0.8)
    for d, col in [(128, "gray"), (512, "silver")]:
        ax.axvline(d, ls="--", color=col, lw=1)
        ax.text(d, 1.02, f"d'={d}", fontsize=8, ha="center", color=col)
    w = report["rung1"]["winner"]
    ax.set_xlabel("pool eigen-index j")
    ax.set_ylabel("learned scale s_j (max-normalized)")
    ax.set_title(f"{ds}: learned spectral filter -- "
                 f"j0={w['j0']:.0f}, w={w['w']:.1f}, gamma={w['gamma']:+.2f}, "
                 f"whiten={report['whiten_final']}, "
                 f"eff_dim={report['s_eff_dim']:.0f}")
    ax.set_ylim(-0.02, 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved -> {path}")


def plot_benchmark(out, path):
    """Set size + coverage vs cal per arm, one column per NCM, balanced split
    (mirror of the transform_control figure)."""
    rows = out["rows"]
    ncms = sorted({r["ncm"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, len(arms)))
    fig, axes = plt.subplots(2, len(ncms), figsize=(6.5 * len(ncms), 8),
                             squeeze=False)
    for jcol, nm in enumerate(ncms):
        for a, col in zip(arms, colors):
            rs = sorted([r for r in rows if r["arm"] == a and r["ncm"] == nm
                         and r["split"] == "balanced_both"],
                        key=lambda r: r["cal"])
            if not rs:
                continue
            cals = [r["cal"] for r in rs]
            axes[0][jcol].errorbar(cals, [r["sz"] for r in rs],
                                   yerr=[r["sz_se"] for r in rs],
                                   marker="o", label=a, color=col)
            axes[1][jcol].plot(cals, [r["cov"] for r in rs], marker="o",
                               label=a, color=col)
        axes[0][jcol].set_title(f"{nm} -- set size (balanced_both)")
        axes[0][jcol].set_xlabel("cal size")
        axes[0][jcol].set_ylabel("avg set size")
        axes[0][jcol].legend(fontsize=7)
        axes[1][jcol].axhline(1 - out["alpha"], ls="--", c="gray")
        axes[1][jcol].set_title(f"{nm} -- coverage")
        axes[1][jcol].set_xlabel("cal size")
        axes[1][jcol].set_ylim(0.85, 1.0)
    fig.suptitle(f"G1 benchmark -- {out['dataset']} "
                 f"({out['n_trials']} trials, PR={out['pool_pr']:.0f}, "
                 f"all arms pool-fit/exchangeable)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved -> {path}")
