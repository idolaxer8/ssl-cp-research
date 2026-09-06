"""Paper-grade headline figures from headline_experiment.py results.

One figure per (dataset, alpha): panel (a) mean set size vs shots (linear
y, clipped -- collapsed baselines drawn as open markers at the cap with
their true value annotated), panel (b) marginal coverage vs shots with the
1-alpha target line. Baselines follow the table convention: best cell over
score x train_frac per shots. Error bars: +-1.96 * SE over trials.

Usage:
    python src/plot_headline.py --dataset cifar100 \
        --results output/headline/results_cifar100.json \
        --out_dir output/headline/plots
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito (colorblind-safe); ours = black, emphasized
STYLE = {
    "ours":    dict(color="#000000", marker="o", ls="-",  lw=1.8, ms=4.5,
                    zorder=5, label="FRCP"),
    "cvplus":  dict(color="#0072B2", marker="s", ls="--", lw=1.2, ms=3.8,
                    zorder=4, label="CV+"),
    "splitcp": dict(color="#E69F00", marker="^", ls="-.", lw=1.2, ms=4.2,
                    zorder=3, label="Split CP (best)"),
    "semicp":  dict(color="#009E73", marker="D", ls=":",  lw=1.2, ms=3.5,
                    zorder=3, label="SemiCP (best)"),
}
DS_LABEL = {"cifar100": "CIFAR-100", "miniimagenet": "miniImageNet",
            "cub200": "CUB-200", "food101": "Food-101",
            "aircraft": "FGVC-Aircraft", "stanford_cars": "Stanford Cars"}
# frozen arm per dataset regime (repaired NCM on fine-grained)
FROZEN_ARM = {"aircraft": "frozen_unwhitened_topk_asym",
              "stanford_cars": "frozen_unwhitened_topk_asym"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def series(rows, arm, alpha):
    """(shots, sz, sz_se, cov) best-over-score-x-frac per shots."""
    out = []
    for s in sorted({x["shots"] for x in rows}):
        cand = [x for x in rows if x["arm"] == arm and x["shots"] == s
                and x["alpha"] == alpha]
        if not cand:
            continue
        b = min(cand, key=lambda x: x["sz"])
        out.append((s, b["sz"], b["sz_se"], b["cov"]))
    return np.array(out)


def make_fig(res, alpha, cap, out_base):
    ds = res["dataset"]
    rows = res["rows"]
    frozen_arm = FROZEN_ARM.get(ds, "frozen")
    arms = [("ours", frozen_arm), ("cvplus", "cvplus"),
            ("splitcp", "splitcp"), ("semicp", "semicp")]

    # Width matches the ICLR text block (5.5 in), so the figure is
    # included at \linewidth with NO downscaling and the 8/7 pt type
    # below lands on the page at its stated size. At the old 6.8 in it
    # was shrunk to ~77%, which is what made the labels look soft.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5.5, 3.0), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.09)

    clipped_by_x = {}                       # x -> [(value, color)] for stacks
    cov_seen = []                           # coverage of the PLOTTED series only
    for key, arm in arms:
        S = series(rows, arm, alpha)
        if not len(S):
            continue
        st = dict(STYLE[key])
        label = st.pop("label")
        shots, sz, se, cov = S.T
        clipped = sz > cap
        # size panel: line + filled markers through unclipped points only
        ax1.errorbar(shots[~clipped], sz[~clipped],
                     yerr=1.96 * se[~clipped], capsize=1.5,
                     elinewidth=0.7, label=label, **st)
        for x, v in zip(shots[clipped], sz[clipped]):
            ax1.plot([x], [cap], marker=st["marker"], ms=st["ms"] + 0.5,
                     mfc="white", mec=st["color"], mew=1.0, ls="none",
                     zorder=st["zorder"])
            clipped_by_x.setdefault(x, []).append((v, st["color"]))
        # coverage panel
        st2 = {k: v for k, v in st.items()}
        ax2.plot(shots, cov, marker=st2.pop("marker"), **st2)
        cov_seen.extend(cov.tolist())

    # off-scale values: per-x colored stacks just under the cap
    for x, vals in clipped_by_x.items():
        for i, (v, color) in enumerate(sorted(vals, reverse=True)):
            ax1.annotate(f"↑{v:.0f}", xy=(x, cap),
                         xytext=(2.5, -3 - 8.5 * i),
                         textcoords="offset points",
                         fontsize=6.5, color=color, ha="left", va="top",
                         annotation_clip=False)

    ax1.set_xlabel("labels per class (shots)")
    ax1.set_ylabel("mean prediction-set size")
    ax1.set_ylim(0, cap * 1.02)
    # (a) and (b) get identical treatment. The dataset, alpha and K
    # were only on (a) and unbalanced the two panels, so they move to
    # the caption, which already carries them.
    ax1.set_title("(a)", loc="left", pad=3)
    ax2.set_title("(b)", loc="left", pad=3)

    ax2.axhline(1 - alpha, color="0.45", lw=0.8, ls=(0, (4, 3)), zorder=1)
    # Left edge, just above the line: the curves all sit high at the
    # smallest budgets, so this corner is the only empty one. (The
    # right edge is where every method converges onto the target.)
    ax2.text(0.015, 1 - alpha + 0.0012, f"target {1 - alpha:g}",
             transform=ax2.get_yaxis_transform(), fontsize=6,
             color="0.35", va="bottom", ha="left")
    ax2.set_xlabel("labels per class (shots)")
    ax2.set_ylabel("marginal coverage")
    # Fit to the data rather than running to 1.0. With a fixed
    # (1-alpha-0.03, 1.004) window the curves sat in a middle band with
    # dead space above and below, so panel (b) read as vertically
    # offset from (a), which fills its box. The target line stays in
    # view either way.
    # cov_seen, not every row: `rows` also holds the score x train_frac
    # arms that lose the best-per-cell selection, some of which sit far
    # below target and would drag the window back open.
    lo = min([1 - alpha] + cov_seen) - 0.006
    hi = max([1 - alpha] + cov_seen) + 0.006
    ax2.set_ylim(lo, hi)

    for ax in (ax1, ax2):
        ax.set_xticks(sorted({x["shots"] for x in rows}))

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4,
               columnspacing=1.4, handlelength=1.9, handletextpad=0.5,
               borderaxespad=0.0)

    # No bbox_inches="tight": it re-crops to the legend row and pulls
    # the axes off-centre. constrained_layout already reserves the
    # space, so the saved canvas is exactly figsize.
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=400)
    plt.close(fig)

    cap_txt = (
        f"{DS_LABEL.get(ds, ds)} headline at alpha = {alpha:g} (K = "
        f"{res['K']}, {res['config']['n_trials']} trials). (a) Mean "
        f"prediction-set size vs. labeled budget (shots per class; "
        f"calibration size = shots x K). Split CP and SemiCP are shown at "
        f"their best score (THR/APS/RAPS) and train fraction per budget; "
        f"open markers at the axis cap denote clipped values (annotated). "
        f"Error bars: +-1.96 SE over trials. (b) Marginal coverage; dashed "
        f"line = 1 - alpha target. All methods use the same backbone "
        f"embeddings, splits, and label budgets; the unlabeled pool is "
        f"available to ours (transform fitting) and SemiCP (score "
        f"augmentation).")
    with open(out_base + "_caption.txt", "w") as f:
        f.write(cap_txt + "\n")
    print(f"saved {out_base}.pdf/.png/_caption.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--results", default=None)
    ap.add_argument("--out_dir", default="output/headline/plots")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.05])
    ap.add_argument("--caps", type=float, nargs="+", default=None,
                    help="size-axis cap per alpha (default 12)")
    args = ap.parse_args()

    path = args.results or f"output/headline/results_{args.dataset}.json"
    res = json.load(open(path))
    os.makedirs(args.out_dir, exist_ok=True)
    caps = args.caps or [12.0] * len(args.alphas)
    for alpha, cap in zip(args.alphas, caps):
        a_tag = f"{alpha:g}".replace(".", "")
        make_fig(res, alpha, cap,
                 os.path.join(args.out_dir,
                              f"fig_headline_{args.dataset}_a{a_tag}"))


if __name__ == "__main__":
    main()
