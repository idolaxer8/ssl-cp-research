"""Appendix D heatmap figure from config_sweep_heatmaps.py output.

2x3 panels: row 1 = kappa x gamma, row 2 = d' x C (in multiples of K),
one column per dataset. Cell = mean filled set size sz1 (FRCP,
alpha=0.1, shots from the sweep config); the deployed cell is boxed.
Values annotated so the color scale is a guide, not the data.

Usage: python src/plot_config_heatmaps.py \
    --in_dir output/config_heatmaps_0907 --out_dir docs/paper/iclr2027/figs
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = ["cifar10", "cifar100", "eurosat"]
DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "eurosat": "EuroSAT"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def panel(ax, rows, grid, p1n, p1v, p2n, p2v, deploy, xlab, ylab):
    M = np.full((len(p2v), len(p1v)), np.nan)
    for r in rows:
        if r["grid"] != grid:
            continue
        i = p2v.index(r[p2n])
        j = p1v.index(r[p1n])
        M[i, j] = r["sz1"]
    im = ax.imshow(M, cmap="viridis_r", aspect="auto", origin="lower")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                vmin, vmax = np.nanmin(M), np.nanmax(M)
                frac = 0.5 if vmax == vmin else (M[i, j] - vmin) / (vmax - vmin)
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if frac > 0.55 else "black")
    di, dj = p2v.index(deploy[p2n]), p1v.index(deploy[p1n])
    ax.add_patch(plt.Rectangle((dj - 0.5, di - 0.5), 1, 1, fill=False,
                               edgecolor="red", lw=1.8))
    ax.set_xticks(range(len(p1v)))
    ax.set_xticklabels([f"{v:g}" for v in p1v])
    ax.set_yticks(range(len(p2v)))
    ax.set_yticklabels([f"{v:g}" for v in p2v])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="output/config_heatmaps_0907")
    ap.add_argument("--out_dir", default="output/config_heatmaps_0907")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(5.8, 4.0))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.10,
                        hspace=0.45, wspace=0.34)
    stats = {}
    for j, ds in enumerate(DS):
        d = json.load(open(os.path.join(args.in_dir, f"results_{ds}.json")))
        rows, dep = d["rows"], d["deploy"]
        kg = sorted({r["kappa"] for r in rows if r["grid"] == "kg"})
        gg = sorted({r["gamma"] for r in rows if r["grid"] == "kg"})
        panel(axes[0][j], rows, "kg", "kappa", kg, "gamma", gg, dep,
              r"$\kappa$", r"$\gamma$")
        axes[0][j].set_title(DS_LABEL[ds], pad=3)
        dg = sorted({r["dprime"] for r in rows if r["grid"] == "dc"})
        cg = sorted({r["cmult"] for r in rows if r["grid"] == "dc"})
        panel(axes[1][j], rows, "dc", "dprime", dg, "cmult", cg, dep,
              r"$d'$", r"$C / K$")
        sz1 = [r["sz1"] for r in rows]
        dep_row = [r for r in rows if r["grid"] == "kg"
                   and r["kappa"] == dep["kappa"]
                   and r["gamma"] == dep["gamma"]][0]
        stats[ds] = dict(min=min(sz1), max=max(sz1),
                         deployed=dep_row["sz1"],
                         n_cells=len(sz1))
    stem = os.path.join(args.out_dir, "fig_config_heatmaps")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=250, bbox_inches="tight")
    with open(os.path.join(args.out_dir, "heatmap_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"saved {stem}.pdf/.png")
    for ds, s in stats.items():
        print(f"  {ds}: deployed sz1={s['deployed']:.2f}, grid range "
              f"[{s['min']:.2f}, {s['max']:.2f}] over {s['n_cells']} cells")


if __name__ == "__main__":
    main()
