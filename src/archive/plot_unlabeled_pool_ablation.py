"""Plot the unlabeled-pool ablation produced by exchangeable_fcp_experiment.py.

Isolates the effect of the *unlabeled pool* in our best exchangeable FCP
combination. Two runs (both random-split => exactly exchangeable, so coverage is
valid at every cal size) give a clean 3-way decomposition of where the pool
helps:

    without pool   FCP (plain)            -- no unlabeled pool at all (identity)
    with pool      PCA+whiten (lam=0)     -- pool used for features only
    with pool      PCA+whiten+MS-CS       -- pool used for features + MS-CS penalty

Left panel: marginal coverage vs cal (all curves hug the 1-alpha target =>
removing/adding the pool never breaks the guarantee). Right panel: avg set size
vs cal (the pool's payoff -- monotonically tighter sets as the pool feeds more
of the pipeline).

Usage:  python src/plot_unlabeled_pool_ablation.py [results_dir]
        (default output/unlabeled_pool_ablation)
"""
# ARCHIVE-CANDIDATE (review 2026-08-24): plots only the settled §4e pool-ablation results; archive together with pool_ablation_hightrial.py.
import sys, os, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (tag, method-substring) -> (legend label, color, linestyle)
STYLE = [
    ("without_pool",          "FCP",   "without pool · FCP (plain)",           "#888888", "--"),
    ("without_pool_centroid", "MS-CS", "without pool · MS-CS (cal centroids)",  "#d62728", ":"),
    ("with_pool",             "FCP",   "with pool · PCA+whiten (lam=0)",        "#1f77b4", "-"),
    ("with_pool",             "MS-CS", "with pool · PCA+whiten+MS-CS",          "#2ca02c", "-"),
]


def load(results_dir):
    rows, alpha = {}, 0.1
    for fp in sorted(glob.glob(os.path.join(results_dir, "results_*.json"))):
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        alpha = d.get("config", {}).get("alpha", 0.1)
        for r in d["rows"]:
            rows[(d["tag"], r["method"], r["cal"])] = (r["cov"], r["sz"])
    return rows, alpha


def series_for(rows, tag, method_sub):
    pts = sorted((cal, v) for (t, m, cal), v in rows.items()
                 if t == tag and method_sub in m)
    cals = [c for c, _ in pts]
    cov = [v[0] for _, v in pts]
    sz = [v[1] for _, v in pts]
    return cals, cov, sz


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "output/unlabeled_pool_ablation"
    rows, alpha = load(rdir)
    if not rows:
        print(f"No results_*.json in {rdir}"); return

    fig, (axc, axs) = plt.subplots(1, 2, figsize=(13, 5.2))
    for tag, msub, label, color, ls in STYLE:
        cals, cov, sz = series_for(rows, tag, msub)
        if not cals:
            continue
        axc.plot(cals, cov, marker="o", color=color, ls=ls, lw=2, label=label)
        axs.plot(cals, sz, marker="o", color=color, ls=ls, lw=2, label=label)

    axc.axhline(1 - alpha, ls=":", c="k", lw=1.2, label=f"target {1 - alpha:.2f}")
    axc.set(xlabel="calibration size", ylabel="marginal coverage",
            title="Coverage — exchangeable with OR without the pool")
    axc.legend(fontsize=8, loc="lower right"); axc.grid(alpha=.3)
    axs.set(xlabel="calibration size", ylabel="avg prediction-set size",
            title="Set size — the unlabeled pool's payoff")
    axs.legend(fontsize=8); axs.grid(alpha=.3)

    # annotate the pool's set-size reduction at the largest shared cal
    np_cals, _, np_sz = series_for(rows, "without_pool", "FCP")
    wp_cals, _, wp_sz = series_for(rows, "with_pool", "MS-CS")
    if np_cals and wp_cals:
        c = max(set(np_cals) & set(wp_cals))
        s0 = np_sz[np_cals.index(c)]; s1 = wp_sz[wp_cals.index(c)]
        red = 100 * (1 - s1 / s0)
        axs.annotate(f"cal={c}: {s0:.2f} -> {s1:.2f}  ({red:.0f}% smaller)",
                     xy=(c, s1), xytext=(0.40, 0.82), textcoords="axes fraction",
                     fontsize=9, arrowprops=dict(arrowstyle="->", color="#2ca02c"))

    fig.suptitle("Isolating the unlabeled pool — exchangeable Full CP "
                 "(CIFAR-100, DINOv2)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(rdir, "unlabeled_pool_ablation.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
