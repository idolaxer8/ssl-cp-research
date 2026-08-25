"""Plots for the goal-5 backbone DWT table (src/backbone_dwt_experiment.py).

Reads backbone_dwt_table.json and produces three figures:

  A_setsize_vs_cal.png   pipeline robustness: |C| vs cal per backbone, wt
                         (solid) vs qe_wt (dashed), faceted by dataset. Shows
                         all backbones reach valid tight-ish sets.
  B_qe_relchange.png     qe effect: paired (qe_wt-wt)/wt (%) vs cal per cell,
                         filled marker = significant (|z|>=2). Shows the qe
                         sign is cal- and backbone-dependent.
  C_gate_dials.png       THE money plot: qe rel% vs the two gate dials at a
                         fixed cal. Left = participation ratio (LABEL-FREE):
                         cleanly separates harm (low PR) from gain. Right =
                         kNN homophily: does NOT separate (the DINOv2 gate
                         fails to transfer -- same h, opposite qe sign).

Linear y-axes throughout (repo plot convention). Usage:
    python src/plot_backbone_dwt.py \
        --table output/from_cluster/backbone_dwt_v2/backbone_dwt_table.json \
        --out_dir output/from_cluster/backbone_dwt_v2/plots
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKBONE_COLOR = {
    "dinov2": "#1f77b4", "dinov3": "#2ca02c",
    "clip": "#ff7f0e", "clip-large": "#d62728",
    "dinov3-512": "#9467bd", "mae": "#7f7f7f",
}
DS_MARKER = {"cifar100": "o", "aircraft": "s"}


def relpct(cell, L):
    return cell["verdict"][str(L)]["relative_change_mean"] * 100


def is_sig(cell, L):
    v = cell["verdict"][str(L)]
    return v["sig_gain"] or v["sig_harm"]


def fig_setsize(cells, cals, out_dir):
    datasets = sorted({c["dataset"] for c in cells})
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(6.2 * len(datasets), 5.2), squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        for c in [c for c in cells if c["dataset"] == ds]:
            col = BACKBONE_COLOR.get(c["backbone"], "#333333")
            wt = [c["arms"]["wt"][str(L)]["size_mean"] for L in cals]
            qe = [c["arms"]["qe_wt"][str(L)]["size_mean"] for L in cals]
            ax.plot(cals, wt, "-o", color=col, lw=2, label=f"{c['backbone']} (W/T)")
            ax.plot(cals, qe, "--x", color=col, lw=1.6, alpha=0.8,
                    label=f"{c['backbone']} (qe->W/T)")
        ax.set_title(f"{ds} — set size vs calibration size", fontsize=11)
        ax.set_xlabel("calibration size"); ax.set_ylabel("avg set size |C|")
        ax.set_xticks(cals); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Pipeline robustness: valid tight-ish sets across backbones "
                 "(solid = W/T champion, dashed = + qe smoothing)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out_dir, "A_setsize_vs_cal.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_qe_relchange(cells, cals, out_dir):
    datasets = sorted({c["dataset"] for c in cells})
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(6.2 * len(datasets), 5.2), squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        for c in [c for c in cells if c["dataset"] == ds]:
            col = BACKBONE_COLOR.get(c["backbone"], "#333333")
            ys = [relpct(c, L) for L in cals]
            ax.plot(cals, ys, "-", color=col, lw=1.8, label=c["backbone"])
            for L, y in zip(cals, ys):
                filled = is_sig(c, L)
                ax.plot(L, y, "o", color=col, ms=9,
                        markerfacecolor=col if filled else "white",
                        markeredgecolor=col, mew=1.6)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(f"{ds} — qe effect on set size", fontsize=11)
        ax.set_xlabel("calibration size")
        ax.set_ylabel("qe change  (qe_wt - W/T)/W/T  [%]   (<0 = qe helps)")
        ax.set_xticks(cals); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("qe smoothing effect is cal- and backbone-dependent "
                 "(filled marker = significant, |z|>=2)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out_dir, "B_qe_relchange.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_gate_dials(cells, out_dir, cal):
    """qe rel% vs (participation ratio | homophily) at a fixed cal."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    dials = [("participation_ratio", "participation ratio PR  (LABEL-FREE)"),
             ("h_knn_k10", "kNN homophily h(k=10)  (needs labels)")]
    for ax, (key, xlabel) in zip(axes, dials):
        xs, ys = [], []
        for c in cells:
            x = c["dials"][key]; y = relpct(c, cal)
            xs.append(x); ys.append(y)
            col = BACKBONE_COLOR.get(c["backbone"], "#333333")
            mk = DS_MARKER.get(c["dataset"], "^")
            filled = is_sig(c, cal)
            ax.scatter([x], [y], s=140, marker=mk,
                       facecolor=col if filled else "white",
                       edgecolor=col, linewidths=1.8, zorder=3)
            ax.annotate(f"{c['backbone']}/{c['dataset'][:4]}", (x, y),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="k", lw=1)
        # shade the qe-harm zone (rel% > 0)
        ax.axhspan(0, max(ys) * 1.15 + 1, color="#d62728", alpha=0.05)
        ax.set_xlabel(xlabel); ax.set_ylabel(f"qe rel% at cal={cal}  (<0 helps)")
        ax.grid(alpha=0.3)
    axes[0].set_title("PR SEPARATES: qe harms only at low PR (<~20)", fontsize=11)
    axes[1].set_title("homophily does NOT: same h, opposite qe sign\n"
                      "(dinov2/air h~.30 harms vs clip/air h~.31 helps)",
                      fontsize=11)
    # legend proxies
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=BACKBONE_COLOR.get(b, "#333"),
                          markeredgecolor=BACKBONE_COLOR.get(b, "#333"),
                          markersize=9, label=b)
               for b in sorted({c["backbone"] for c in cells})]
    handles += [plt.Line2D([0], [0], marker=DS_MARKER[d], color="k",
                           markerfacecolor="white", markersize=9,
                           label=d, linestyle="none")
                for d in sorted({c["dataset"] for c in cells}) if d in DS_MARKER]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle("Which dial gates qe across backbones? "
                 "(filled = significant qe effect; red band = qe harms)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(out_dir, "C_gate_dials.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(
        "output", "from_cluster", "backbone_dwt_v2", "backbone_dwt_table.json"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dial_cal", type=int, default=400,
                    help="cal size for the dial scatter (fig C)")
    args = ap.parse_args()
    t = json.load(open(args.table))
    cells = t["cells"]; cals = t["cal_sizes"]
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.table), "plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(cells)} cells, cals={cals}, dial_cal={args.dial_cal}")
    fig_setsize(cells, cals, out_dir)
    fig_qe_relchange(cells, cals, out_dir)
    fig_gate_dials(cells, out_dir, args.dial_cal)
    print("all plots ->", out_dir)


if __name__ == "__main__":
    main()
