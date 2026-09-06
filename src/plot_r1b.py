"""Paper-grade outputs for the R1b champion decomposition ablation.

From r1_headline_experiment.py results (arms raw/wt/qe_wt x NCMs x shots):
  fig_r1b_stages.{pdf,png}   stage decomposition (champion NCM), one panel
                             per dataset: raw -> +T,W -> +D(=frozen)
  fig_r1b_ncm.{pdf,png}      NCM ablation on the full pipeline (qe_wt):
                             prototype_softmax vs prototype_cosine vs
                             unwhitened_topk_asym (D1 softmax column)
  table_r1b.tex              booktabs sizes table, arms x NCMs at
                             --table_shots, per dataset; bold best per
                             column; appendix companion for split=random
  *_caption.txt              ready captions

Usage:
    python src/plot_r1b.py \
        --results output/r1_headline/results_cifar100.json \
                  output/r1_headline/results_miniimagenet.json \
        --out_dir output/r1_headline/plots
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS_LABEL = {"cifar100": "CIFAR-100", "miniimagenet": "miniImageNet",
            "cub200": "CUB-200", "food101": "Food-101",
            "aircraft": "FGVC-Aircraft", "stanford_cars": "Stanford Cars"}
ARM_STYLE = {   # progressive refinement: light -> full pipeline (black)
    "raw":   dict(color="#9A9A9A", marker="^", ls=":",  lw=1.2, ms=4.0,
                  zorder=3, label="raw embedding"),
    "wt":    dict(color="#0072B2", marker="s", ls="--", lw=1.3, ms=3.8,
                  zorder=4, label="+ T, W (truncate, whiten)"),
    "qe_wt": dict(color="#000000", marker="o", ls="-",  lw=1.8, ms=4.5,
                  zorder=5, label="+ S (smoothing) = full pipeline"),
}
NCM_STYLE = {
    "prototype_softmax":    dict(color="#000000", marker="o", ls="-",
                                 lw=1.8, ms=4.5, zorder=5,
                                 label="prototype-softmax"),
    "prototype_cosine":     dict(color="#D55E00", marker="s", ls="--",
                                 lw=1.3, ms=3.8, zorder=4,
                                 label="prototype-cosine (softmax-free)"),
    "unwhitened_topk_asym": dict(color="#009E73", marker="D", ls=":",
                                 lw=1.3, ms=3.5, zorder=3,
                                 label="geodesic top-k (asym)"),
}
NCM_TEX = {"prototype_softmax": "prototype-softmax",
           "prototype_cosine": "prototype-cosine",
           "unwhitened_topk_asym": "geodesic top-$k$"}
ARM_TEX = {"raw": "raw", "wt": "$+$T,W", "qe_wt": "$+$T,W,D (full)"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def get(rows, **kw):
    out = [x for x in rows if all(x.get(k) == v for k, v in kw.items())]
    return out[0] if len(out) == 1 else out


def curve_fig(results, split, curves, sel_fixed, out_base, cap, caption):
    """Generic small-multiples: one panel per dataset, one curve per
    `curves` item (dict of style + selector)."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.35 * n, 2.5),
                             gridspec_kw={"wspace": 0.28}, squeeze=False)
    for j, res in enumerate(results):
        ax = axes[0][j]
        rows = [x for x in res["rows"] if x["split"] == split]
        shots_all = sorted({x["shots"] for x in rows})
        clip_notes = []
        y_top = 0.0                       # auto-fit when nothing clips
        for sel, st in curves:
            st = dict(st)
            label = st.pop("label")
            pts = [(s, get(rows, shots=s, **{**sel_fixed, **sel}))
                   for s in shots_all]
            pts = [(s, b) for s, b in pts if isinstance(b, dict)]
            if not pts:
                continue
            sh = np.array([p[0] for p in pts], dtype=float)
            sz = np.array([p[1]["sz"] for p in pts])
            se = np.array([p[1]["sz_se"] for p in pts])
            m = sz > cap
            if (~m).any():
                y_top = max(y_top, float((sz[~m] + 1.96 * se[~m]).max()))
            ax.errorbar(sh[~m], sz[~m], yerr=1.96 * se[~m], capsize=1.5,
                        elinewidth=0.7, label=label if j == 0 else None,
                        **st)
            for x, v in zip(sh[m], sz[m]):
                ax.plot([x], [cap], marker=st["marker"], ms=st["ms"] + 0.5,
                        mfc="white", mec=st["color"], mew=1.0, ls="none",
                        zorder=st["zorder"])
                clip_notes.append((x, v, st["color"]))
        for x, vals in {x: [(v, c) for xx, v, c in clip_notes if xx == x]
                        for x in {c[0] for c in clip_notes}}.items():
            for i, (v, c) in enumerate(sorted(vals, reverse=True)):
                ax.text(x + 0.25, cap * (0.955 - 0.075 * i), f"↑{v:.0f}",
                        fontsize=6.5, color=c, ha="left", va="top")
        tag = chr(ord("a") + j)
        ax.set_title(f"({tag})  {DS_LABEL.get(res['dataset'], res['dataset'])}"
                     f"   ($\\alpha$ = {res['config']['alpha']:g})",
                     loc="left", pad=4)
        ax.set_xlabel("labels per class (shots)")
        if j == 0:
            ax.set_ylabel("mean prediction-set size")
        ax.set_xticks(shots_all)
        ax.set_ylim(0, cap * 1.02 if clip_notes else y_top * 1.08)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.985), columnspacing=1.3,
               handlelength=2.0, handletextpad=0.5)
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    with open(out_base + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    print(f"saved {out_base}.pdf/.png/_caption.txt")


def tex_table(results, split, shots_sel, out_path, label, caption):
    ncms = ["prototype_softmax", "prototype_cosine", "unwhitened_topk_asym"]
    arms = ["raw", "wt", "qe_wt"]
    cols = "ll" + "c" * (len(shots_sel) * len(results))
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule"]
    span = " & ".join(
        rf"\multicolumn{{{len(shots_sel)}}}{{c}}"
        rf"{{{DS_LABEL.get(r['dataset'], r['dataset'])}}}"
        for r in results)
    lines.append(rf" & & {span}\\")
    sh = " & ".join(str(s) for _ in results for s in shots_sel)
    lines.append(rf"NCM & transform & {sh}\\")
    lines.append(r"\midrule")
    # best (smallest) per column across all ncm x arm rows
    best = {}
    for r in results:
        rows = [x for x in r["rows"] if x["split"] == split]
        for s in shots_sel:
            vals = [get(rows, arm=a, ncm=n, shots=s)
                    for a in arms for n in ncms]
            vals = [v["sz"] for v in vals if isinstance(v, dict)]
            best[(r["dataset"], s)] = min(vals) if vals else None
    for n in ncms:
        for i, a in enumerate(arms):
            cells = []
            for r in results:
                rows = [x for x in r["rows"] if x["split"] == split]
                for s in shots_sel:
                    b = get(rows, arm=a, ncm=n, shots=s)
                    if not isinstance(b, dict):
                        cells.append("--")
                        continue
                    txt = f"{b['sz']:.2f}$\\pm${b['sz_se']:.2f}"
                    # bold best incl. ties at display precision
                    if b["sz"] <= best[(r["dataset"], s)] + 5e-3:
                        txt = rf"\textbf{{{txt}}}"
                    cells.append(txt)
            head = NCM_TEX[n] if i == 0 else ""
            lines.append(f"{head} & {ARM_TEX[a]} & " + " & ".join(cells)
                         + r"\\")
        if n != ncms[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {out_path}")



def _panel(ax, res, split, curves, sel_fixed, cap, ylabel):
    """Draw one ablation panel. Shared by the combined figure."""
    rows = [x for x in res["rows"] if x["split"] == split]
    shots_all = sorted({x["shots"] for x in rows})
    clip_notes, y_top = [], 0.0
    for sel, st in curves:
        st = dict(st)
        label = st.pop("label")
        pts = [(s, get(rows, shots=s, **{**sel_fixed, **sel})) for s in shots_all]
        pts = [(s, b) for s, b in pts if isinstance(b, dict)]
        if not pts:
            continue
        sh = np.array([p[0] for p in pts], dtype=float)
        sz = np.array([p[1]["sz"] for p in pts])
        se = np.array([p[1]["sz_se"] for p in pts])
        m = sz > cap
        if (~m).any():
            y_top = max(y_top, float((sz[~m] + 1.96 * se[~m]).max()))
        ax.errorbar(sh[~m], sz[~m], yerr=1.96 * se[~m], capsize=1.5,
                    elinewidth=0.7, label=label, **st)
        for x, v in zip(sh[m], sz[m]):
            ax.plot([x], [cap], marker=st["marker"], ms=st["ms"] + 0.5,
                    mfc="white", mec=st["color"], mew=1.0, ls="none",
                    zorder=st["zorder"])
            clip_notes.append((x, v, st["color"]))
    for x in {c[0] for c in clip_notes}:
        vals = sorted([(v, c) for xx, v, c in clip_notes if xx == x], reverse=True)
        for i, (v, c) in enumerate(vals):
            ax.annotate(f"\u2191{v:.0f}", xy=(x, cap), xytext=(2.5, -3 - 8.5 * i),
                        textcoords="offset points", fontsize=6.5, color=c,
                        ha="left", va="top", annotation_clip=False)
    ax.set_xlabel("labels per class (shots)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(shots_all)
    # Just enough headroom for the in-axes legend, which sits upper
    # right where the curves have already collapsed. The earlier 1.30
    # spent a third of each panel on empty space, which is what made
    # the plots read small.
    ax.set_ylim(0, (cap * 1.02 if clip_notes else y_top * 1.08) * 1.16)


def combined_fig(res, split, champion_ncm, out_base, cap, caption):
    """Both ablation panels in ONE float.

    They used to be two separately saved PDFs placed side by side in the
    LaTeX. Each was cropped with bbox_inches="tight" to its own legend
    row, and the two legends differ in width, so the files came out
    4.82x2.96 in and 5.38x2.96 in. Scaled to the same \linewidth
    fraction, unequal aspect ratios mean unequal heights, and the panels
    do not line up. One figure, one crop, no mismatch possible.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.5),
                                   layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.07)
    _panel(ax1, res, split,
           [({"arm": a}, ARM_STYLE[a]) for a in ("raw", "wt", "qe_wt")],
           {"ncm": champion_ncm}, cap, "mean prediction-set size")
    _panel(ax2, res, split,
           [({"ncm": n}, NCM_STYLE[n]) for n in NCM_STYLE],
           {"arm": "qe_wt"}, cap, None)
    # Per-panel legends: the two panels compare different things, so a
    # shared figure legend cannot serve both.
    for ax, tag in ((ax1, "a"), (ax2, "b")):
        ax.set_title(f"({tag})", loc="left", pad=3)
        ax.legend(loc="upper right", frameon=False, borderaxespad=0.3,
                  handlelength=1.8, handletextpad=0.5, labelspacing=0.35)
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=400)
    plt.close(fig)
    with open(out_base + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    print(f"saved {out_base}.pdf/.png/_caption.txt")


MECH_DS = ["cifar10", "cifar100", "miniimagenet", "eurosat"]
MECH_DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
                 "miniimagenet": "miniImageNet", "eurosat": "EuroSAT",
                 "aircraft": "FGVC-Aircraft"}


def mech_panel(ax, h_cert_path):
    """Panel (b) of the ablation figure (2026-08-31, replaces the parked
    score-family panel): weighted pool homophily h_w raw -> whitened per
    dataset (dumbbell), Lemma 1 certified fraction annotated per row.
    Data: the deterministic h-cert diagnostic (h_cert_roster.json)."""
    d = json.load(open(h_cert_path))["results"]
    ys = np.arange(len(MECH_DS))[::-1]
    for y, ds in zip(ys, MECH_DS):
        r, w = d[ds]["raw"], d[ds]["whitened"]
        ax.plot([r["h_w_mean"], w["h_w_mean"]], [y, y], color="#BBBBBB",
                lw=2.0, zorder=2, solid_capstyle="round")
        ax.plot([r["h_w_mean"]], [y], marker="o", ms=6.0, mfc="white",
                mec="#888888", mew=1.3, ls="none", zorder=3,
                label="raw" if ds == MECH_DS[0] else None)
        ax.plot([w["h_w_mean"]], [y], marker="o", ms=6.2, mfc="#0072B2",
                mec="white", mew=0.8, ls="none", zorder=4,
                label="whitened" if ds == MECH_DS[0] else None)
        ax.annotate(f"cert. {100*r['certified_frac']:.0f}"
                    f"$\\to${100*w['certified_frac']:.0f}%",
                    xy=(1.03, y), xycoords=("axes fraction", "data"),
                    fontsize=8, color="#666666", ha="left", va="center",
                    annotation_clip=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([MECH_DS_LABEL[ds] for ds in MECH_DS], fontsize=8.5)
    # 2026-09-02: with FGVC-Aircraft out of the roster every remaining
    # dataset sits above h_w = 0.8, so the old 0-1 axis collapsed the
    # dumbbells into dots. Range is now data-driven with a little padding.
    lo = min(min(d[ds]["raw"]["h_w_mean"], d[ds]["whitened"]["h_w_mean"])
             for ds in MECH_DS)
    ax.set_xlim(max(0.0, lo - 0.03), 1.005)
    ticks = [x / 100 for x in range(0, 101, 5)]
    ticks = [x for x in ticks if x >= max(0.0, lo - 0.03)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.2f}".rstrip("0").rstrip(".")
                        if i % 2 == 0 else ""
                        for i, x in enumerate(ticks)])
    ax.set_xlabel("pool-neighborhood homophily $h_w$")
    ax.set_ylim(-0.6, len(MECH_DS) - 0.4)
    ax.grid(axis="x", color="#000000", alpha=0.10, lw=0.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=False,
              handletextpad=0.4, borderaxespad=0.2, labelspacing=0.4,
              fontsize=8.5)


def combined_mech_fig(res, split, champion_ncm, h_cert_path, out_base, cap,
                      caption):
    """Paper Figure 3 as of 2026-08-31 (user call): panel (a) stage
    ablation as before, panel (b) = mech_panel. The old score-family
    panel is PARKED (its presentation is an open question); fig_r1b_ncm_*
    still carries that material."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.2),
                                   layout="constrained",
                                   width_ratios=[1.0, 1.12])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.16)
    # Short legend labels: at half width the full stage names overflow
    # the axes; the caption spells the stages out.
    short = {"raw": "raw", "wt": "+ T, W", "qe_wt": "+ S (full)"}
    arm_st = {a: dict(ARM_STYLE[a], label=short[a]) for a in ARM_STYLE}
    _panel(ax1, res, split,
           [({"arm": a}, arm_st[a]) for a in ("raw", "wt", "qe_wt")],
           {"ncm": champion_ncm}, cap, "mean prediction-set size")
    ax1.legend(loc="upper right", frameon=False, borderaxespad=0.3,
               handlelength=1.8, handletextpad=0.5, labelspacing=0.35)
    mech_panel(ax2, h_cert_path)
    for ax, tag in ((ax1, "a"), (ax2, "b")):
        ax.set_title(f"({tag})", loc="left", pad=3)
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png", dpi=400)
    plt.close(fig)
    with open(out_base + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    print(f"saved {out_base}.pdf/.png/_caption.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out_dir", default="output/r1_headline/plots")
    ap.add_argument("--split", default="balanced_both")
    ap.add_argument("--champion_ncm", default="prototype_softmax")
    ap.add_argument("--cap", type=float, default=13.0)
    ap.add_argument("--table_shots", type=int, nargs="+", default=[2, 6, 14])
    ap.add_argument("--h_cert", default="output/dwt_theory/h_cert_roster.json")
    args = ap.parse_args()

    results = [json.load(open(p)) for p in args.results]
    os.makedirs(args.out_dir, exist_ok=True)
    trials = results[0]["config"]["n_trials"]
    tag = "_".join(r["dataset"] for r in results)

    curve_fig(
        results, args.split,
        [({"arm": a}, ARM_STYLE[a]) for a in ("raw", "wt", "qe_wt")],
        {"ncm": args.champion_ncm},
        os.path.join(args.out_dir, f"fig_r1b_stages_{tag}"), args.cap,
        ("Champion-pipeline decomposition (" + args.champion_ncm.replace("_", "-")
         + f" NCM, balanced split, {trials} trials). Mean prediction-set "
         "size vs. labeled budget for the raw embedding, after truncation "
         "and whitening (T, W), and after pool-neighbor smoothing (D; the "
         "full frozen pipeline). Transforms are fit on the unlabeled pool "
         "only; every arm is exact full CP. Error bars: +-1.96 SE over "
         "trials; open markers at the axis cap denote clipped values "
         "(annotated). All arms are valid (coverage 0.90-0.97, tightening "
         "toward target down the pipeline)."))
    curve_fig(
        results, args.split,
        [({"ncm": n}, NCM_STYLE[n]) for n in NCM_STYLE],
        {"arm": "qe_wt"},
        os.path.join(args.out_dir, f"fig_r1b_ncm_{tag}"), args.cap,
        (f"NCM ablation on the full frozen pipeline (balanced split, "
         f"{trials} trials): the champion prototype-softmax score vs. its "
         "softmax-free counterpart (prototype-cosine) and the geodesic "
         "top-k score. Removing the softmax normalizer costs 24-35% set "
         "size on separable data. Error bars: +-1.96 SE over trials."))
    combined_mech_fig(
        results[0], args.split, args.champion_ncm, args.h_cert,
        os.path.join(args.out_dir, "fig_ablation_mech"), args.cap,
        (f"Refinement ablation and mechanism ({results[0]['dataset']} stages, "
         "all-roster h-cert panel; see the paper caption)."))
    combined_fig(
        results[0], args.split, args.champion_ncm,
        os.path.join(args.out_dir, f"fig_r1b_combined_{results[0]['dataset']}"),
        args.cap,
        (f"Refinement ablation on {DS_LABEL.get(results[0]['dataset'])} at "
         f"alpha = {results[0]['config']['alpha']:g} ({trials} trials). "
         "(a) mean set size for the raw embedding, after truncation and "
         "whitening, and after smoothing (the full pipeline), under the "
         "prototype softmax score. (b) the full pipeline under the three "
         "training-free scores. Error bars: +-1.96 SE over trials."))

    tex_table(results, args.split, args.table_shots,
              os.path.join(args.out_dir, f"table_r1b_{tag}.tex"),
              f"tab:r1b-decomposition-{tag}",
              "Champion decomposition: mean prediction-set size "
              "($\\pm$ SE, " + f"{trials} trials, balanced split, "
              "$\\alpha=0.1$) for each transform stage $\\times$ NCM. "
              "Bold: best per budget. Every arm is exact full CP with "
              "pool-fit transforms.")
    # appendix companion: random split (exact-validity protocol)
    if any(x["split"] == "random" for r in results for x in r["rows"]):
        tex_table(results, "random", args.table_shots,
                  os.path.join(args.out_dir, f"table_r1b_random_{tag}.tex"),
                  f"tab:r1b-decomposition-random-{tag}",
                  "Champion decomposition under the fully random split "
                  "(exact exchangeability; appendix companion to the "
                  "balanced-split table). Prototype-softmax rows below "
                  "8 shots are omitted (degenerate probe corner, see "
                  "text).")
    print("done")


if __name__ == "__main__":
    main()
