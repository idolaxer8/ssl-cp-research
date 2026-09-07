"""Instructor-facing comparison pack (user call 2026-09-07).

Two self-contained comparisons, assembled from EXISTING 50-trial runs
(no new experiments):

1. Encoder scaling: FRCP and the best trained-head baseline, base vs
   large encoder, per family (DINOv2 ViT-B vs ViT-L, CLIP ViT-B vs
   ViT-L) on the three headline datasets.
   Sources: output/headline (dinov2-base),
   output/backbone_headline/{dinov2-large,clip-base,clip-large}.

2. Non-empty-sets convention: the plain sets against the top-1-filled
   sets (every empty set receives the top-ranked label), paired within
   one run (same trials and splits; arms record both conventions).
   Source: output/temperature/nonempty (50 trials, pool-margin T,
   shrinkage off = the deployed no-shrinkage config, audited within
   2 SE of the shipped rows).

Outputs -> --out_dir (default output/instructor_comparisons_0907):
    fig_encoder_scaling.png/.pdf     2x3 panels, set size vs budget
    table_encoder_scaling.md/.tex    cells-won + FRCP sizes
    fig_nonempty_cifar10.png/.pdf    empty rates + plain-vs-filled sizes
    table_nonempty.md/.tex           sz->sz1 (empty%) and cov->cov1
    README.md                        takeaways + provenance

Usage (from repo root, absolute dirs work too):
    python src/make_instructor_comparisons.py
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = ["cifar10", "cifar100", "eurosat"]
DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "eurosat": "EuroSAT"}
BASELINES = ["splitcp", "cvplus", "semicp"]
SHOTS = [2, 4, 8, 12]
ALPHA = 0.1

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def load(res_dir, ds):
    p = os.path.join(res_dir, f"results_{ds}.json")
    return json.load(open(p))["rows"] if os.path.exists(p) else None


def best(rows, arm, s, a):
    c = [x for x in rows if x["arm"] == arm and x["shots"] == s
         and x["alpha"] == a]
    return min(c, key=lambda x: x["sz"]) if c else None


def series(rows, arm, a):
    """(shots, sz) over SHOTS for one arm; arm='best_bl' = min baseline."""
    xs, ys = [], []
    for s in SHOTS:
        if arm == "best_bl":
            cand = [best(rows, b, s, a) for b in BASELINES]
            cand = [c for c in cand if c]
            r = min(cand, key=lambda x: x["sz"]) if cand else None
        else:
            r = best(rows, arm, s, a)
        if r:
            xs.append(s)
            ys.append(r["sz"])
    return xs, ys


def wins(rows, frozen_arm, a, shots):
    w, n = 0, 0
    for s in shots:
        fro = best(rows, frozen_arm, s, a)
        if fro is None:
            continue
        n += 1
        bl = min((b for b in (best(rows, x, s, a) for x in BASELINES) if b),
                 key=lambda r: r["sz"])
        w += fro["sz"] <= bl["sz"] + 5e-3
    return w, n


# ---------------------------------------------------------------- part 1
def encoder_scaling(dirs, out_dir):
    FAMS = [("DINOv2", "dinov2-base", "dinov2-large"),
            ("CLIP", "clip-base", "clip-large")]
    C_BASE, C_LARGE = "#1F77B4", "#D62728"

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.2))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.09,
                        hspace=0.42, wspace=0.24)
    for i, (fam, tb, tl) in enumerate(FAMS):
        for j, ds in enumerate(DS):
            ax = axes[i][j]
            rb, rl = load(dirs[tb], ds), load(dirs[tl], ds)
            panel_vals = []
            for rows, col, tag in ((rb, C_BASE, "base"),
                                   (rl, C_LARGE, "large")):
                if rows is None:
                    continue
                xf, yf = series(rows, "frozen", ALPHA)
                xb, yb = series(rows, "best_bl", ALPHA)
                panel_vals += yf
                ax.plot(xf, yf, "-o", color=col, ms=3.5, lw=1.5,
                        label=f"FRCP ({tag})")
                ax.plot(xb, yb, ":s", color=col, ms=3, lw=1.2, alpha=0.75,
                        label=f"best baseline ({tag})")
            cap = 1.8 * max(panel_vals)
            # annotate clipped baseline points at the cap
            for line in ax.get_lines():
                yd = np.asarray(line.get_ydata(), dtype=float)
                if (yd > cap).any():
                    xd = np.asarray(line.get_xdata(), dtype=float)
                    for x, y in zip(xd[yd > cap], yd[yd > cap]):
                        ax.annotate(f"{y:.0f}", (x, cap), fontsize=7,
                                    ha="center", va="bottom",
                                    color=line.get_color())
                    yd = np.minimum(yd, cap)
                    line.set_ydata(yd)
            ax.set_ylim(0, cap * 1.12)
            ax.set_xticks(SHOTS)
            ax.set_title(f"{fam} on {DS_LABEL[ds]}", pad=3)
            if j == 0:
                ax.set_ylabel("mean set size")
            if i == 1:
                ax.set_xlabel("labels per class")
            ax.grid(axis="y", color="k", alpha=0.08, lw=0.6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Does a larger encoder strengthen FRCP? "
                 r"(mean set size, $\alpha=0.1$, 50 trials)",
                 y=1.0, fontsize=11)
    stem = os.path.join(out_dir, "fig_encoder_scaling")
    fig.savefig(stem + ".png", dpi=250, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.png/.pdf")

    # win-count + landing table (md + tex)
    ENC = [("DINOv2 ViT-B (paper Table 2)", "dinov2-base"),
           ("DINOv2 ViT-L", "dinov2-large"),
           ("CLIP ViT-B", "clip-base"),
           ("CLIP ViT-L", "clip-large")]
    md = ["| encoder | " + " | ".join(DS_LABEL[d] for d in DS)
          + " | total (a=0.10) | total (a=0.05) | FRCP size s=2 "
            "(c10 / c100 / euro) |",
          "|---" * 7 + "|"]
    tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
           "encoder & " + " & ".join(DS_LABEL[d] for d in DS)
           + r" & total ($\alpha{=}.1$) & total ($\alpha{=}.05$)"
             r" & FRCP $|C|$ at $s{=}2$\\", r"\midrule"]
    for name, tag in ENC:
        per, t1w, t1n, t2w, t2n, s2 = [], 0, 0, 0, 0, []
        for ds in DS:
            rows = load(dirs[tag], ds)
            w, n = wins(rows, "frozen", 0.1, SHOTS)
            per.append(f"{w}/{n}")
            t1w, t1n = t1w + w, t1n + n
            w5, n5 = wins(rows, "frozen", 0.05, SHOTS)
            t2w, t2n = t2w + w5, t2n + n5
            s2.append(f"{best(rows, 'frozen', 2, 0.1)['sz']:.2f}")
        md.append(f"| {name} | " + " | ".join(per)
                  + f" | **{t1w}/{t1n}** | {t2w}/{t2n} | "
                  + " / ".join(s2) + " |")
        tex.append(f"{name} & " + " & ".join(per)
                   + rf" & \textbf{{{t1w}/{t1n}}} & {t2w}/{t2n} & "
                   + " / ".join(s2) + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "table_encoder_scaling.md"), "w",
              encoding="utf-8") as f:
        f.write("FRCP cells won against the best trained-head baseline "
                "(per dataset, budgets {2,4,8,12}, 50 trials; CLIP ViT-B "
                "has no 12-shot run, so its cells are /3).\n\n"
                + "\n".join(md) + "\n")
    with open(os.path.join(out_dir, "table_encoder_scaling.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("saved table_encoder_scaling.md/.tex")


# ---------------------------------------------------------------- part 2
def nonempty(ne_dir, out_dir):
    ARMS = [("splitcp", "Split CP"), ("cvplus", "CV+"),
            ("semicp", "SemiCP"), ("frozen_Tpool_shrzero", "FRCP (ours)")]
    NE_SHOTS = [2, 4, 8, 14]      # this run predates the 12-shot cap
    COLORS = {"splitcp": "#E69F00", "cvplus": "#0072B2",
              "semicp": "#009E73", "frozen_Tpool_shrzero": "#000000"}

    # figure: cifar10 is where empty sets actually occur
    rows10 = load(ne_dir, "cifar10")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.16,
                        wspace=0.28)
    for arm, lbl in ARMS:
        e = [100 * best(rows10, arm, s, ALPHA)["empty"] for s in NE_SHOTS]
        ax1.plot(NE_SHOTS, e, "-o", ms=3.5, lw=1.5, color=COLORS[arm],
                 label=lbl)
    ax1.set_xlabel("labels per class")
    ax1.set_ylabel("empty-set rate (%)")
    ax1.set_xticks(NE_SHOTS)
    ax1.set_title("(a) every method abstains on CIFAR-10")
    ax1.grid(axis="y", color="k", alpha=0.08, lw=0.6)
    for arm, lbl in [("frozen_Tpool_shrzero", "FRCP"), ("cvplus", "CV+")]:
        r = [best(rows10, arm, s, ALPHA) for s in NE_SHOTS]
        ax2.plot(NE_SHOTS, [x["sz"] for x in r], "-o", ms=3.5, lw=1.5,
                 color=COLORS[arm], label=f"{lbl} plain")
        ax2.plot(NE_SHOTS, [x["sz1"] for x in r], "--s", ms=3, lw=1.3,
                 color=COLORS[arm], alpha=0.7, label=f"{lbl} filled")
    ax2.axhline(1.0, color="k", lw=0.7, alpha=0.4)
    ax2.set_xlabel("labels per class")
    ax2.set_ylabel("mean set size")
    ax2.set_xticks(NE_SHOTS)
    ax2.set_title("(b) filling the empties: sizes barely move")
    ax2.grid(axis="y", color="k", alpha=0.08, lw=0.6)
    ax1.legend(ncol=2, loc="upper left")
    ax2.legend(ncol=2, loc="upper right")
    fig.suptitle("Non-empty-sets convention on CIFAR-10 "
                 r"($\alpha=0.1$, 50 trials, paired within run)",
                 y=0.97, fontsize=11)
    stem = os.path.join(out_dir, "fig_nonempty_cifar10")
    fig.savefig(stem + ".png", dpi=250, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.png/.pdf")

    # table: sz -> sz1 (empty %) and cov -> cov1, all three datasets
    md = ["Plain vs top-1-filled sets, paired within one 50-trial run "
          "(same trials and splits; baselines at their best score and "
          "train fraction by plain size). Cell: size plain -> filled "
          "(empty-set rate). Coverage rows beneath each block.\n"]
    hdr = "| dataset | method | " + " | ".join(
        f"{s} shots" for s in NE_SHOTS) + " |"
    md += [hdr, "|---" * (len(NE_SHOTS) + 2) + "|"]
    tex = [r"\begin{tabular}{ll" + "c" * len(NE_SHOTS) + "}", r"\toprule",
           "dataset & method & " + " & ".join(
               f"{s} shots" for s in NE_SHOTS) + r"\\"]
    for ds in DS:
        rows = load(ne_dir, ds)
        tex.append(r"\midrule")
        for k, (arm, lbl) in enumerate(ARMS):
            cs_md, cs_tx, cv_md, cv_tx = [], [], [], []
            for s in NE_SHOTS:
                b = best(rows, arm, s, ALPHA)
                cs_md.append(f"{b['sz']:.2f} -> {b['sz1']:.2f} "
                             f"({100 * b['empty']:.0f}%)")
                cs_tx.append(rf"{b['sz']:.2f}$\to${b['sz1']:.2f} "
                             rf"{{\scriptsize({100 * b['empty']:.0f}\%)}}")
                cv_md.append(f"{b['cov']:.3f} -> {b['cov1']:.3f}")
                cv_tx.append(rf"{{\scriptsize {b['cov']:.3f}$\to$"
                             rf"{b['cov1']:.3f}}}")
            dsl = DS_LABEL[ds] if k == 0 else ""
            md.append(f"| {dsl} | {lbl} | " + " | ".join(cs_md) + " |")
            md.append(f"|  | *coverage* | " + " | ".join(cv_md) + " |")
            head = (rf"\multirow{{{2 * len(ARMS)}}}{{*}}{{{DS_LABEL[ds]}}}"
                    if k == 0 else "")
            tex.append(f"{head} & {lbl} & " + " & ".join(cs_tx) + r"\\")
            tex.append(" & " + " & ".join([""] + cv_tx) + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "table_nonempty.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "table_nonempty.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print("saved table_nonempty.md/.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline_dir", default="output/headline")
    ap.add_argument("--backbone_dir", default="output/backbone_headline")
    ap.add_argument("--nonempty_dir", default="output/temperature/nonempty")
    ap.add_argument("--out_dir",
                    default="output/instructor_comparisons_0907")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dirs = {"dinov2-base": args.headline_dir,
            "dinov2-large": os.path.join(args.backbone_dir, "dinov2-large"),
            "clip-base": os.path.join(args.backbone_dir, "clip-base"),
            "clip-large": os.path.join(args.backbone_dir, "clip-large")}
    encoder_scaling(dirs, args.out_dir)
    nonempty(args.nonempty_dir, args.out_dir)


if __name__ == "__main__":
    main()
