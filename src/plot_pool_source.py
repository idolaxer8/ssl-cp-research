"""Replot pool_source_comparison.py results WITHOUT a log axis.

The original right panel plotted absolute set size on a log scale (sz spans
~1.7 at cal=800 to ~31 at cal=200, so a linear absolute axis buries the
large-cal differences). This replaces it with a log-free, more informative
2-panel view:

  left   coverage vs cal (linear)
  right  set-size REDUCTION vs the no-pool baseline (%) vs cal (linear) --
         bounded, directly comparable across cal sizes, and it answers the
         actual question (how much each unlabeled-data source shrinks sets).

Absolute set sizes stay in the README table. Reads/writes by absolute path into
the main checkout's output/.
"""
import os, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = r"C:\Users\IDO\Desktop\Ido_student\Msc\ssl-cp-research"
RES = os.path.join(REPO, "output/pool_source_comparison")

STYLE = {"no-pool": ("#888888", "--", "o"), "transductive": ("#ff7f0e", "-", "s"),
         "pool-matched": ("#1f77b4", "-", "^"), "pool-10k": ("#2ca02c", "-", "o")}


def main():
    d = json.load(open(os.path.join(RES, "results.json")))
    arms = d["arms"]; alpha = d["config"]["alpha"]
    cals = [r["cal"] for r in arms["no-pool"]["rows"]]
    base = {r["cal"]: r["sz"] for r in arms["no-pool"]["rows"]}  # no-pool baseline per cal

    fig, (axc, axr) = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, dd in arms.items():
        c, ls, mk = STYLE[name]
        cov = [r["cov"] for r in dd["rows"]]; cse = [r["cov_se"] for r in dd["rows"]]
        axc.errorbar(cals, cov, yerr=cse, marker=mk, color=c, ls=ls, lw=2, capsize=3, label=name)
        if name == "no-pool":
            continue  # baseline => 0% by definition, skip on reduction panel
        red = [100 * (1 - r["sz"] / base[r["cal"]]) for r in dd["rows"]]
        axr.plot(cals, red, marker=mk, color=c, ls=ls, lw=2, label=name)
        for cal, rr, r in zip(cals, red, dd["rows"]):
            axr.annotate(f"{r['sz']:.2f}", (cal, rr), textcoords="offset points",
                         xytext=(0, 6), fontsize=7, color=c, ha="center")

    axc.axhline(1 - alpha, ls=":", c="k", lw=1.2, label=f"target {1-alpha:.2f}")
    axc.set(xlabel="calibration size", ylabel="marginal coverage",
            title="Coverage -- all arms exactly exchangeable")
    axc.legend(fontsize=8, loc="lower right"); axc.grid(alpha=.3)
    axr.set(xlabel="calibration size", ylabel="set-size reduction vs no-pool (%)",
            title="Set-size reduction (linear) -- labels show absolute sz")
    axr.legend(fontsize=8, loc="upper right"); axr.grid(alpha=.3)
    axr.set_ylim(0, 55)
    fig.suptitle("Transform fit source: separate pool vs the test set itself "
                 "(CIFAR-100, DINOv2; 30 trials, test=1000)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(RES, "pool_source_comparison.png")
    fig.savefig(p, dpi=140)
    print("saved", p)


if __name__ == "__main__":
    main()
