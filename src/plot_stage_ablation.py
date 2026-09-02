"""Paper-grade stage-ablation figure from the 09-01 rescoped run.

Data: output/pipeline_ablation/results_<ds>_stageabl.json (arms raw / T /
W / D / TW / TWD; prototype-softmax NCM with per-arm auto temperature;
exact full CP, 20 trials, alpha 0.1). The D arm letter in the data is the
smoothing stage; all display text uses S (paper naming rule).

Figure: 2x2 dataset panels, upset-style. Top of each panel: grouped bars
(one group per stage combo, one bar per labeled budget) of mean set size
with 1.96 SE whiskers; bottom: the stage on/off dot matrix. Bars above
the panel cap (degenerate arms) are clipped and annotated with their
value. Shows both that every stage is needed AND the order story: W alone
floors the temperature fit and inflates sets; T first stabilizes it.

Usage (from repo root):
    python src/plot_stage_ablation.py
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = ["cifar10", "cifar100", "miniimagenet", "eurosat"]
DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "miniimagenet": "miniImageNet", "eurosat": "EuroSAT"}
# display order: raw, single stages in pipeline order, then the chain
COMBOS = [("raw", "raw"), ("T", "T"), ("W", "W"), ("D", "S"),
          ("TW", "T+W"), ("TWD", "full")]
STAGES = [("T", "T (truncate)"), ("W", "W (whiten)"), ("D", "S (smooth)")]
SHOT_COLOR = {2: "#9ECAE1", 4: "#4292C6", 8: "#08519C"}
CAP_FACTOR = 3.0        # panel cap = CAP_FACTOR * worst full-pipeline bar

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def get(rows, arm, shots):
    r = [x for x in rows if x["arm"] == arm and x["shots"] == shots]
    return r[0] if r else None


def panel(ax_bar, ax_mat, res, shots_sel, stats):
    rows = res["rows"]
    n_sh = len(shots_sel)
    width = 0.8 / n_sh
    full_max = max(get(rows, "TWD", s)["sz"] for s in shots_sel)
    vmax = max(x["sz"] for x in rows if x["shots"] in shots_sel)
    cap = min(vmax * 1.02, CAP_FACTOR * full_max)
    for si, s in enumerate(shots_sel):
        xs, ys, es, clipped = [], [], [], []
        for i, (arm, _) in enumerate(COMBOS):
            r = get(rows, arm, s)
            if r is None:
                continue
            x = i + (si - (n_sh - 1) / 2) * width
            v = r["sz"]
            xs.append(x)
            ys.append(min(v, cap))
            es.append(1.96 * r["sz_se"] if v <= cap else 0.0)
            if v > cap:
                clipped.append((x, v))
            stats.append(dict(dataset=res["dataset"], arm=arm, shots=s,
                              sz=round(v, 3), cov=round(r["cov"], 4)))
        ax_bar.bar(xs, ys, width * 0.9, yerr=es, capsize=1.2,
                   error_kw=dict(elinewidth=0.6),
                   color=SHOT_COLOR[s], edgecolor="white", linewidth=0.4,
                   label=f"{s} labels/class")
        for x, v in clipped:
            ax_bar.annotate(f"{v:.1f}", xy=(x, cap), xytext=(0, -1),
                            textcoords="offset points", fontsize=6,
                            rotation=90, ha="center", va="top",
                            color="#333333")
    ax_bar.set_ylim(0, cap * 1.04)
    ax_bar.set_xlim(-0.55, len(COMBOS) - 0.45)
    ax_bar.set_title(f"{DS_LABEL[res['dataset']]}   (K = {res['K']})",
                     loc="left", pad=3)
    plt.setp(ax_bar.get_xticklabels(), visible=False)
    ax_bar.tick_params(axis="x", length=0)
    ax_bar.grid(axis="y", color="#000000", alpha=0.08, lw=0.6)
    # stage on/off matrix, upset style: active dots joined by a bar
    for i, (arm, _) in enumerate(COMBOS):
        on_rows = [ri for ri, (letter, _) in enumerate(STAGES)
                   if letter in arm and arm != "raw"]
        ys_all = [2 - ri for ri in range(len(STAGES))]
        ax_mat.scatter([i] * len(ys_all), ys_all, s=26, c="#E3E3E3",
                       zorder=2)
        if on_rows:
            ys_on = [2 - ri for ri in on_rows]
            ax_mat.plot([i, i], [min(ys_on), max(ys_on)], color="#333333",
                        lw=1.4, zorder=3, solid_capstyle="round")
            ax_mat.scatter([i] * len(ys_on), ys_on, s=26, c="#333333",
                           zorder=4)
    ax_mat.set_xlim(-0.55, len(COMBOS) - 0.45)
    ax_mat.set_ylim(-0.5, 2.5)
    ax_mat.set_xticks(range(len(COMBOS)))
    ax_mat.set_xticklabels([lab for _, lab in COMBOS], fontsize=7)
    for sp in ("top", "right", "left", "bottom"):
        ax_mat.spines[sp].set_visible(False)
    ax_mat.tick_params(length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="output/pipeline_ablation")
    ap.add_argument("--out_dir", default="output/pipeline_ablation/plots")
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4, 8])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for ds in DS:
        with open(os.path.join(args.data_dir,
                               f"results_{ds}_stageabl.json")) as f:
            results.append(json.load(f))

    fig = plt.figure(figsize=(5.5, 4.6))
    outer = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24,
                             left=0.09, right=0.99, top=0.90, bottom=0.07)
    stats = []
    for j, res in enumerate(results):
        inner = outer[j // 2, j % 2].subgridspec(
            2, 1, height_ratios=[3.1, 1.0], hspace=0.06)
        ax_bar = fig.add_subplot(inner[0])
        ax_mat = fig.add_subplot(inner[1], sharex=ax_bar)
        panel(ax_bar, ax_mat, res, args.shots, stats)
        if j % 2 == 0:
            ax_bar.set_ylabel("mean set size")
            ax_mat.set_yticks([2, 1, 0])
            ax_mat.set_yticklabels([lab for _, lab in STAGES], fontsize=6.8)
        else:
            ax_mat.set_yticks([])
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.97), columnspacing=1.6,
               handlelength=1.2, handletextpad=0.5)

    stem = os.path.join(args.out_dir, "fig_stage_ablation")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # headline numbers for the caption + a machine-readable dump
    covs = [c["cov"] for c in stats]
    full_wins = sum(
        1 for ds in DS for s in args.shots
        if min(c["sz"] for c in stats
               if c["dataset"] == ds and c["shots"] == s)
        >= next(c["sz"] for c in stats if c["dataset"] == ds
                and c["shots"] == s and c["arm"] == "TWD") - 5e-3)
    caption = (
        "Stage ablation of the frozen refinement on the four headline "
        "datasets (DINOv2 ViT-B, prototype-softmax score, exact full CP, "
        "20 trials, $\\alpha=0.1$): mean prediction-set size for every "
        "on/off combination of truncation (T), whitening (W) and "
        "smoothing (S), at 2/4/8 labels per class. Bars above the panel "
        "cap are clipped and annotated. The full pipeline (T+W+S) gives "
        f"the smallest sets in {full_wins}/{len(DS) * len(args.shots)} "
        "dataset\\,$\\times$\\,budget cells; W alone floors the "
        "temperature fit and inflates sets, T first stabilizes it "
        "(the T$\\to$W order story of Sec. 4). Every arm is exact full "
        f"CP (coverage {min(covs):.2f}--{max(covs):.2f}).")
    with open(stem + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    with open(os.path.join(args.out_dir, "stage_ablation_stats.json"),
              "w") as f:
        json.dump({"cells": stats, "full_wins": full_wins,
                   "cov_range": [min(covs), max(covs)]}, f, indent=2)
    print(f"saved {stem}.pdf/.png  full wins {full_wins}/"
          f"{len(DS) * len(args.shots)}  cov {min(covs):.3f}-{max(covs):.3f}")


if __name__ == "__main__":
    main()
