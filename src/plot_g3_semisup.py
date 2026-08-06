"""
G3 analysis + depth-axis figure (docs/g3_semisup_baseline_plan.md sec 8).

Reads results_{dataset}{tag}.json from g3_semisup_experiment.py and produces:
  - a verdict table per dataset (arm A oracle vs champion; arm B vs champion
    by the 2-SE rule) printed to stdout
  - g3_depth_axis{tag}.png: one panel per dataset, set size vs cal for
    raw / SCP-THR (arm A r=0, best ratio) / arm A best (best ratio+round) /
    poolmlp_raw / poolmlp_qe / champion. Linear y-axis; series that collapse
    off-scale are clipped to the top edge and annotated with their value.

Usage:
python src/plot_g3_semisup.py --results_dir output/pool_repr_menu/g3_semisup \
    --datasets cifar100 aircraft cub200 --tag pilot
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FCP_SERIES = [("raw", "raw embeddings", "#888888", "o"),
              ("poolmlp_raw", "pool-MLP (B-raw)", "#1f77b4", "s"),
              ("poolmlp_qe", "pool-MLP (B-qe)", "#17becf", "D"),
              ("champion", "closed-form champion", "#d62728", "*")]


def best_fcp_cell(rows, arm, cal, split="balanced_both"):
    """Best NCM for (arm, cal): (sz, se, ncm) or None."""
    cand = [r for r in rows if r["arm"] == arm and r["cal"] == cal
            and r["split"] == split]
    if not cand:
        return None
    r = min(cand, key=lambda r: r["sz"])
    return r["sz"], r["sz_se"], r["ncm"]


def arm_a_cells(rows, cal):
    """(scp_thr_refresh, arm_a_oracle) for a cal: each (sz, se, label)."""
    r0 = [r for r in rows if r["cal"] == cal and r["round"] == 0]
    rall = [r for r in rows if r["cal"] == cal]
    if not r0:
        return None, None
    b0 = min(r0, key=lambda r: r["sz"])
    ba = min(rall, key=lambda r: r["sz"])
    return ((b0["sz"], b0["sz_se"], f"ratio {b0['ratio']:.2f}"),
            (ba["sz"], ba["sz_se"],
             f"ratio {ba['ratio']:.2f}, r={ba['round']}"))


def verdict(sz_a, se_a, sz_b, se_b):
    """2-SE rule: 'win' if a beats b, 'loss' if b beats a, else 'tie'."""
    d = sz_a - sz_b
    lim = 2 * max(se_a, se_b, 1e-9)
    return "loss" if d > lim else ("win" if d < -lim else "tie")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="output/pool_repr_menu/g3_semisup")
    ap.add_argument("--datasets", nargs="+", default=["cifar100"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    data = {}
    for ds in args.datasets:
        p = os.path.join(args.results_dir, f"results_{ds}{tag}.json")
        if not os.path.exists(p):
            p = os.path.join(args.results_dir, f"results_{ds}.json")
        with open(p) as f:
            data[ds] = json.load(f)

    fig, axes = plt.subplots(1, len(args.datasets),
                             figsize=(5.2 * len(args.datasets), 4.4),
                             squeeze=False)
    for ax, ds in zip(axes[0], args.datasets):
        d = data[ds]
        frows, srows = d["fcp_rows"], d["selftrain_rows"]
        cals = sorted({r["cal"] for r in frows
                       if r["split"] == "balanced_both"})
        print(f"\n=== {ds} ===")
        # verdict table
        for cal in cals:
            ch = best_fcp_cell(frows, "champion", cal)
            print(f"cal={cal}:")
            if ch:
                print(f"  champion         {ch[0]:7.2f}+-{ch[1]:.2f} ({ch[2]})")
            scp, ora = arm_a_cells(srows, cal)
            if scp:
                print(f"  SCP-THR (r=0)    {scp[0]:7.2f}+-{scp[1]:.2f} "
                      f"({scp[2]})")
            if ora and ch:
                print(f"  arm A oracle     {ora[0]:7.2f}+-{ora[1]:.2f} "
                      f"({ora[2]}) -> {verdict(ora[0], ora[1], *ch[:2])} "
                      f"vs champion; collapse(>2x)="
                      f"{'YES' if ora[0] > 2 * ch[0] else 'no'}")
            for arm in ("poolmlp_raw", "poolmlp_qe"):
                b = best_fcp_cell(frows, arm, cal)
                if b and ch:
                    print(f"  {arm:16s} {b[0]:7.2f}+-{b[1]:.2f} ({b[2]}) "
                          f"-> {verdict(b[0], b[1], *ch[:2])} vs champion")

        # figure: linear axis, clip off-scale series to top edge + label
        fcp_max = max(best_fcp_cell(frows, a, c)[0]
                      for a, *_ in FCP_SERIES for c in cals
                      if best_fcp_cell(frows, a, c))
        ylim = fcp_max * 1.6
        for arm, label, color, marker in FCP_SERIES:
            ys, es = [], []
            for cal in cals:
                b = best_fcp_cell(frows, arm, cal)
                ys.append(b[0] if b else np.nan)
                es.append(b[1] if b else 0)
            ax.errorbar(cals, ys, yerr=es, label=label, color=color,
                        marker=marker, ms=6, capsize=3)
        for src, label, color in ((0, "SCP-THR (split, r=0)", "#9467bd"),
                                  (1, "self-train oracle (arm A)", "#e377c2")):
            ys = []
            for cal in cals:
                cells = arm_a_cells(srows, cal)
                ys.append(cells[src][0] if cells[src] else np.nan)
            clipped = [min(v, ylim * 0.97) if np.isfinite(v) else v
                       for v in ys]
            ax.plot(cals, clipped, label=label, color=color, marker="v",
                    ms=6, ls="--")
            for x, v, c in zip(cals, ys, clipped):
                if np.isfinite(v) and v > ylim * 0.97:
                    ax.annotate(f"{v:.0f}", (x, c), textcoords="offset points",
                                xytext=(0, -12), ha="center", fontsize=8,
                                color=color)
        ax.set_ylim(0, ylim)
        ax.set_xlabel("calibration size")
        ax.set_ylabel("mean set size")
        ax.set_title(ds)
        ax.set_xticks(cals)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("The depth axis: label-dependent split CP vs pool-only "
                 "representations under exact FCP", fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.results_dir, f"g3_depth_axis{tag}.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
