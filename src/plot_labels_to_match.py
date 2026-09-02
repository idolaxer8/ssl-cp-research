"""Budget-equivalence figure: labels the strongest baseline needs to
match FRCP.

Treats the total label budget as the scarce resource (the project's
standing framing): for every budget s given to FRCP, how many labels per
class does the strongest baseline (best of split CP / CV+ / SemiCP at
their best score x train fraction, per cell) need before its mean set
size drops to FRCP's? Read off the baseline size-vs-shots curve by
linear interpolation; points ABOVE the dashed grid-limit line mean the
baseline does not get there anywhere on the measured grid (>14
labels/class, i.e. >7x at s=2).

One panel per backbone (DINOv2 ViT-B from output/headline, CLIP ViT-B
from output/backbone_headline/clip-base), one curve per headline dataset,
gray diagonal = parity (baseline as label-efficient as FRCP).

Usage (from repo root):
    python src/plot_labels_to_match.py
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = ["cifar10", "cifar100", "miniimagenet", "eurosat"]
DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "miniimagenet": "miniImageNet", "eurosat": "EuroSAT"}
COLOR = {"cifar10": "#56B4E9", "cifar100": "#0072B2",
         "miniimagenet": "#009E73", "eurosat": "#E69F00"}
MARKER = {"cifar10": "o", "cifar100": "s", "miniimagenet": "^",
          "eurosat": "D"}
BASELINES = ["cvplus", "splitcp", "semicp"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def best_row(rows, arm, s, alpha):
    c = [x for x in rows if x["arm"] == arm and x["shots"] == s
         and x["alpha"] == alpha]
    return min(c, key=lambda x: x["sz"]) if c else None


def curves(rows, alpha):
    grid = sorted({x["shots"] for x in rows})
    frcp = {s: best_row(rows, "frozen", s, alpha)["sz"] for s in grid}
    base = {s: min(best_row(rows, a, s, alpha)["sz"] for a in BASELINES)
            for s in grid}
    return grid, frcp, base


def shots_to_match(grid, base, target):
    """First crossing of the baseline size-vs-shots curve below target;
    None if the curve never reaches it on the grid."""
    if base[grid[0]] <= target:
        return float(grid[0])           # matched already at the smallest s
    for sa, sb in zip(grid, grid[1:]):
        if base[sa] > target >= base[sb]:
            frac = (base[sa] - target) / (base[sa] - base[sb])
            return float(sa + frac * (sb - sa))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline_dir", default="output/headline")
    ap.add_argument("--clipb_dir",
                    default="output/backbone_headline/clip-base")
    ap.add_argument("--out_dir", default="output/headline/plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    backbones = [("DINOv2 ViT-B", args.headline_dir),
                 ("CLIP ViT-B", args.clipb_dir)]
    for alpha, atag in ((0.1, "a01"), (0.05, "a005")):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
        fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.18,
                            wspace=0.16)
        stats = {"alpha": alpha, "cells": []}
        for ax, (bb, res_dir) in zip(axes, backbones):
            grid_max = 0
            for ds in DS:
                with open(os.path.join(res_dir,
                                       f"results_{ds}.json")) as f:
                    rows = json.load(f)["rows"]
                grid, frcp, base = curves(rows, alpha)
                grid_max = max(grid_max, grid[-1])
                y_cens = grid[-1] * 1.13
                xs, ys, cens = [], [], []
                for s in grid:
                    m = shots_to_match(grid, base, frcp[s])
                    xs.append(s)
                    ys.append(m if m is not None else y_cens)
                    cens.append(m is None)
                    stats["cells"].append(dict(
                        backbone=bb, ds=ds, shots=s,
                        frcp_sz=round(frcp[s], 3),
                        base_sz_here=round(base[s], 3),
                        base_sz_at_max=round(base[grid[-1]], 3),
                        matched_shots=None if m is None else round(m, 2)))
                ax.plot(xs, ys, color=COLOR[ds], lw=1.3, zorder=3,
                        label=DS_LABEL[ds] if bb == backbones[0][0]
                        else None)
                for x, y, c in zip(xs, ys, cens):
                    ax.plot([x], [y], marker=MARKER[ds], ms=4.2,
                            mfc="white" if c else COLOR[ds],
                            mec=COLOR[ds], mew=1.0, ls="none", zorder=4)
            lim = grid_max * 1.22
            ax.plot([0, lim], [0, lim], color="0.6", lw=0.8,
                    ls=(0, (4, 3)), zorder=1)
            ax.axhline(grid_max, color="0.35", lw=0.7, ls=":", zorder=2)
            ax.text(0.98, grid_max / lim + 0.015, "measured-grid limit",
                    transform=ax.transAxes, fontsize=6.2, color="0.35",
                    ha="right", va="bottom")
            ax.set_xlim(0, grid_max + 1)
            ax.set_ylim(0, lim)
            ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
            ax.set_yticks([2, 4, 6, 8, 10, 12, 14])
            ax.set_title(bb, loc="left", pad=3)
            ax.set_xlabel("labels per class given to FRCP")
            ax.grid(color="#000000", alpha=0.07, lw=0.6)
        axes[0].set_ylabel("labels per class the\nstrongest baseline needs")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, 1.0), columnspacing=1.4,
                   handlelength=1.6, handletextpad=0.5)

        stem = os.path.join(args.out_dir, f"fig_labels_to_match_{atag}")
        fig.savefig(stem + ".pdf", bbox_inches="tight")
        fig.savefig(stem + ".png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        n_cens = sum(1 for c in stats["cells"]
                     if c["matched_shots"] is None)
        caption = (
            "Label-budget equivalence: labels per class the strongest "
            "baseline (best of split CP, CV+, SemiCP at their best score "
            "and train fraction per cell) needs before its mean set size "
            "matches FRCP's, per backbone and dataset "
            f"($\\alpha={alpha:g}$, 50 trials, balanced split; read off "
            "the baseline size-vs-budget curve by linear interpolation). "
            "The gray diagonal is parity; open markers above the dotted "
            "line are budgets where no measured baseline budget "
            f"($\\leq${grid_max} labels/class) reaches FRCP's set size "
            f"({n_cens} of {len(stats['cells'])} cells).")
        with open(stem + "_caption.txt", "w", encoding="utf-8") as f:
            f.write(caption + "\n")
        with open(os.path.join(args.out_dir,
                               f"labels_to_match_{atag}.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[{atag}] saved {stem}.pdf  censored {n_cens}/"
              f"{len(stats['cells'])}")


if __name__ == "__main__":
    main()
