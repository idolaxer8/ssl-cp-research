"""Plot exchangeable-FCP results saved by exchangeable_fcp_experiment.py.

Two panels vs calibration size: marginal coverage (with the 1-alpha target) and
average set size. One line per (run-tag, method), e.g. "degraded · FCP",
"full · FCP", "full · MS-CS lam=0.05" -- so the figure shows the guarantee
(coverage ~ target at every cal) and the efficiency (PCA+whiten+MS-CS shrinking
sets) at a glance.

Usage:  python src/plot_exchangeable.py [results_dir]   (default output/exchangeable_fcp)
"""
import sys, os, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(results_dir):
    series, alpha = {}, 0.1
    for fp in sorted(glob.glob(os.path.join(results_dir, "results_*.json"))):
        with open(fp) as f:
            d = json.load(f)
        alpha = d.get("config", {}).get("alpha", 0.1)
        for r in d["rows"]:
            series.setdefault(f"{d['tag']} · {r['method']}", {})[r["cal"]] = (r["cov"], r["sz"])
    return series, alpha


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "output/exchangeable_fcp"
    series, alpha = load(results_dir)
    if not series:
        print(f"No results_*.json in {results_dir}"); return

    # order: degraded first, then full FCP, then full MS-CS (ascending lambda)
    names = sorted(series, key=lambda n: ("full" in n, "MS-CS" in n, n))
    fig, (axc, axs) = plt.subplots(1, 2, figsize=(13, 5))
    for name in names:
        cals = sorted(series[name])
        cov = [series[name][c][0] for c in cals]
        sz = [series[name][c][1] for c in cals]
        ls = "--" if name.startswith("degraded") else "-"
        axc.plot(cals, cov, marker="o", ls=ls, label=name)
        axs.plot(cals, sz, marker="o", ls=ls, label=name)

    axc.axhline(1 - alpha, ls=":", c="k", lw=1.2, label=f"target {1 - alpha:.2f}")
    axc.set(xlabel="calibration size", ylabel="marginal coverage",
            title="Coverage — exchangeable ⇒ valid at every cal")
    axc.legend(fontsize=8); axc.grid(alpha=.3)
    axs.set(xlabel="calibration size", ylabel="avg prediction-set size",
            title="Set size — PCA+whiten+MS-CS (all from unlabeled)")
    axs.legend(fontsize=8); axs.grid(alpha=.3)
    fig.suptitle("Exchangeable Full Conformal Prediction (CIFAR-100, DINOv2)")
    fig.tight_layout()

    out = os.path.join(results_dir, "exchangeable_fcp.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
