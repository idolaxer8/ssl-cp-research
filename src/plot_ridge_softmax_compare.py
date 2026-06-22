"""
Plot ridge_softmax (FCA-inspired) vs the geodesic NCMs (topk_asym, topk_mean) on
CIFAR-100, reading the results_<tag>.json files written by
exchangeable_fcp_experiment.py. Produces a 2x2 figure (rows = split, cols =
set size | coverage) and saves it to the comparison output dir.

Run:  python src/plot_ridge_softmax_compare.py --dir output/ridge_softmax_compare
"""
import os
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tag -> (ncm label, split). geo_mean/_asym are the EXCHANGEABLE (pool-whitened)
# counterparts of geodesic_topk_mean / _asym -- the NCM adds no cal-fit whitening.
TAGS = {
    "ridge_bal":   ("ridge_softmax (FCA)", "balanced_both"),
    "ridge_rand":  ("ridge_softmax (FCA)", "random"),
    "uwasym_bal":  ("geodesic_topk_asym",  "balanced_both"),
    "uwasym_rand": ("geodesic_topk_asym",  "random"),
    "uwmean_bal":  ("geodesic_topk_mean",  "balanced_both"),
    "uwmean_rand": ("geodesic_topk_mean",  "random"),
}
COLORS = {"ridge_softmax (FCA)": "#d62728",
          "geodesic_topk_asym": "#1f77b4",
          "geodesic_topk_mean": "#2ca02c"}
MARKERS = {"ridge_softmax (FCA)": "o",
           "geodesic_topk_asym": "s",
           "geodesic_topk_mean": "^"}
SPLITS = ["balanced_both", "random"]
SPLIT_TITLE = {"balanced_both": "balanced_both (default, conservative)",
               "random": "random (exact-validity)"}


def load(d):
    """Return {(ncm, split): [(cal, cov, sz), ...]} from results_<tag>.json."""
    out = {}
    for tag, (ncm, split) in TAGS.items():
        p = os.path.join(d, f"results_{tag}.json")
        if not os.path.exists(p):
            print(f"  (missing: {p})")
            continue
        with open(p) as f:
            data = json.load(f)
        pts = [(r["cal"], r["cov"], r["sz"]) for r in data["rows"]
               if r.get("lam", 0.0) == 0.0]
        out[(ncm, split)] = sorted(pts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/ridge_softmax_compare")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=None, help="output PNG (default <dir>/ridge_vs_geodesic.png)")
    args = ap.parse_args()

    series = load(args.dir)
    if not series:
        raise SystemExit(f"No results_*.json found in {args.dir}")
    ncms = [n for n in COLORS if any(k[0] == n for k in series)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row, split in enumerate(SPLITS):
        ax_sz, ax_cov = axes[row]
        for ncm in ncms:
            pts = series.get((ncm, split))
            if not pts:
                continue
            cals = [p[0] for p in pts]
            covs = [p[1] for p in pts]
            szs = [p[2] for p in pts]
            kw = dict(color=COLORS[ncm], marker=MARKERS[ncm], lw=2, ms=8)
            ax_sz.plot(cals, szs, **kw, label=ncm)
            ax_cov.plot(cals, covs, **kw, label=ncm)
            for c, s in zip(cals, szs):
                ax_sz.annotate(f"{s:.2f}", (c, s), textcoords="offset points",
                               xytext=(0, 7), ha="center", fontsize=8,
                               color=COLORS[ncm])
        ax_cov.axhline(1 - args.alpha, color="k", ls=":", lw=1, label="target 1-alpha")
        ax_sz.set_ylabel(f"{SPLIT_TITLE[split]}\n\navg set size")
        ax_cov.set_ylabel("marginal coverage")
        ax_sz.grid(alpha=0.3); ax_cov.grid(alpha=0.3)
        ax_cov.set_ylim(0.86, 0.96)
        if row == 0:
            ax_sz.set_title("Set size (lower = better)")
            ax_cov.set_title("Coverage (target = 0.90)")
            ax_sz.legend(fontsize=8, loc="best")
            ax_cov.legend(fontsize=8, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("calibration size")
    axes[1, 0].set_xticks(sorted({p[0] for v in series.values() for p in v}))

    fig.suptitle("Ridge-Softmax (FCA-inspired) vs geodesic NCMs\n"
                 "CIFAR-100, K=100, DINOv2 + PCA-128 + cluster-whiten, Full CP, "
                 "alpha=0.1, 5 trials", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = args.out or os.path.join(args.dir, "ridge_vs_geodesic.png")
    fig.savefig(out, dpi=150)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
