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
                    zorder=5, label="Ours (frozen transform + full CP)"),
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
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
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

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.8, 2.55), gridspec_kw={"wspace": 0.30})

    clipped_by_x = {}                       # x -> [(value, color)] for stacks
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

    # off-scale values: per-x colored stacks just under the cap
    for x, vals in clipped_by_x.items():
        for i, (v, color) in enumerate(sorted(vals, reverse=True)):
            ax1.text(x + 0.25, cap * (0.955 - 0.075 * i),
                     f"↑{v:.0f}", fontsize=6.5, color=color,
                     ha="left", va="top")

    ax1.set_xlabel("labels per class (shots)")
    ax1.set_ylabel("mean prediction-set size")
    ax1.set_ylim(0, cap * 1.02)
    ax1.set_title(f"(a)  {DS_LABEL.get(ds, ds)}   "
                  fr"($\alpha$ = {alpha:g}, K = {res['K']})",
                  loc="left", pad=4)
    ax2.set_title("(b)", loc="left", pad=4)

    ax2.axhline(1 - alpha, color="0.45", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax2.text(min(x["shots"] for x in rows), 1 - alpha + 0.001,
             f"target {1 - alpha:g}", fontsize=6, color="0.35",
             va="bottom", ha="left")
    ax2.set_xlabel("labels per class (shots)")
    ax2.set_ylabel("marginal coverage")
    ax2.set_ylim(1 - alpha - 0.03, 1.004)

    for ax in (ax1, ax2):
        ax.set_xticks(sorted({x["shots"] for x in rows}))

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.99), columnspacing=1.3,
               handlelength=2.0, handletextpad=0.5)

    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", bbox_inches="tight", dpi=300)
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
