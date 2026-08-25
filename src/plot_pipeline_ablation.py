"""
Pipeline stage ablation figure: the 2^3 grid over {D smoothing, W whiten,
T truncate} of the adopted pipeline, one panel per dataset, upset-style
(bars on top, stage on/off dot matrix below).

Stage mapping per combo (per-dataset d'):
  ---      raw768                 D--     qe_raw768
  -W-      lw_cluster768          DW-     qe_lw768
  --T      pca_d' (raw-variance   D-T     qe_pca_d'   <- truncation without W
           ranking by construction)
  -WT      ldapool_d'             DWT     qe_ldapool_d'   (the pipeline)

Usage:
python src/plot_pipeline_ablation.py --data_dir output/pool_repr_menu/stage_ablation
"""
import os, json, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMBOS = [("---", "raw768"), ("D--", "qe_raw768"),
          ("-W-", "lw_cluster768"), ("--T", "pca{d}"),
          ("DW-", "qe_lw768"), ("D-T", "qe_pca{d}"),
          ("-WT", "ldapool{d}"), ("DWT", "qe_ldapool{d}")]
DS = [("cifar100", 192, "prototype_softmax"),
      ("cub200", 512, "prototype_softmax"),
      ("aircraft", 512, "unwhitened_topk_mean")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="output/pool_repr_menu/stage_ablation")
    ap.add_argument("--ncm_override", nargs="*", default=[],
                    help="dataset=ncm pairs to override the plotted NCM")
    args = ap.parse_args()
    ncm_of = {ds: ncm for ds, _, ncm in DS}
    for kv in args.ncm_override:
        k, v = kv.split("=")
        ncm_of[k] = v

    fig = plt.figure(figsize=(6.2 * len(DS), 6.2))
    gs = fig.add_gridspec(2, len(DS), height_ratios=[3.2, 1], hspace=0.06)
    for j, (ds, d, _) in enumerate(DS):
        ncm = ncm_of[ds]
        with open(os.path.join(args.data_dir, f"results_{ds}_stageabl.json")) as f:
            rows = json.load(f)["rows"]
        cals = sorted({r["cal"] for r in rows})
        ax = fig.add_subplot(gs[0, j])
        axm = fig.add_subplot(gs[1, j], sharex=ax)
        width = 0.8 / len(cals)
        vals_all = []
        for ci, cal in enumerate(cals):
            xs, ys, es = [], [], []
            for i, (tag, arm_t) in enumerate(COMBOS):
                arm = arm_t.format(d=d)
                r = next((r for r in rows if r["arm"] == arm
                          and r["ncm"] == ncm and r["cal"] == cal), None)
                if r is None:
                    continue
                xs.append(i + (ci - (len(cals) - 1) / 2) * width)
                ys.append(r["sz"]); es.append(r["sz_se"])
            vals_all += ys
            ax.bar(xs, ys, width * 0.92, yerr=es, capsize=1.5,
                   label=f"cal {cal}",
                   color=plt.cm.viridis(0.15 + 0.3 * ci))
        # clip axis to keep the pipeline end readable; annotate clipped bars
        base = min(vals_all)
        ylim = min(max(vals_all), base * 12) * 1.15
        for p in ax.patches:
            if p.get_height() > ylim:
                ax.annotate(f"{p.get_height():.0f}^",
                            (p.get_x() + p.get_width() / 2, ylim * 0.97),
                            ha="center", va="top", fontsize=6.5, rotation=90)
                p.set_height(ylim * 0.985)
        ax.set_ylim(0, ylim)
        ax.set_title(f"{ds} ({ncm.split('_')[0]}, d'={d})", fontsize=11)
        ax.set_ylabel("mean set size" if j == 0 else "")
        ax.legend(fontsize=8)
        plt.setp(ax.get_xticklabels(), visible=False)
        # stage matrix
        for i, (tag, _) in enumerate(COMBOS):
            for si, (sy, name) in enumerate(zip([2, 1, 0],
                                                ["D smoothing", "W whiten",
                                                 "T truncate"])):
                on = tag[si] != "-"
                axm.scatter(i, sy, s=90, c="#333333" if on else "#dddddd",
                            zorder=3)
        axm.set_yticks([2, 1, 0])
        axm.set_yticklabels(["D smoothing", "W whiten", "T truncate"],
                            fontsize=8)
        axm.set_xticks(range(len(COMBOS)))
        axm.set_xticklabels([t for t, _ in COMBOS], fontsize=8)
        axm.set_ylim(-0.6, 2.6)
        axm.set_xlim(-0.6, len(COMBOS) - 0.4)
        for s in ("top", "right", "left", "bottom"):
            axm.spines[s].set_visible(False)
        axm.tick_params(length=0)
    fig.suptitle("Stage ablation of the adopted pipeline "
                 "T(x) = E^T W0 (D(x) - mu) — all 2^3 stage combinations "
                 "(balanced, 10 trials; bars clipped ^)", fontsize=12)
    p = os.path.join(args.data_dir, "pipeline_stage_ablation.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("Saved ->", p)


if __name__ == "__main__":
    main()
