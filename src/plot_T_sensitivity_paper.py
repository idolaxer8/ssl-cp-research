"""Appendix D temperature-sensitivity figure (3-dataset roster).

Adapted 2026-09-07 from the temperature-audit branch's
plot_T_sensitivity.py (R2 figure): mean FRCP set size under a grid of
FIXED temperatures (log x, linear y per the standing preference), one
panel per headline dataset, shots {2, 8} at alpha=0.1, 20 trials. The
deployed pool-resolved T is the dashed vertical line.

Reads output/temperature/sweep/<ds>/T<i>/headline_checkpoint_<ds>.json
and output/temperature/resolver_check.json (dinov2 rows, T_pool).

Usage: python src/plot_T_sensitivity_paper.py \
    --sweep_dir output/temperature/sweep \
    --resolver output/temperature/resolver_check.json \
    --out_dir output/temperature/sweep
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = ["cifar10", "cifar100", "eurosat"]
DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "eurosat": "EuroSAT"}
SHOTS = [2, 8]
AK = "0.1"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default="output/temperature/sweep")
    ap.add_argument("--resolver",
                    default="output/temperature/resolver_check.json")
    ap.add_argument("--out_dir", default="output/temperature/sweep")
    args = ap.parse_args()
    resolver = {r["dataset"]: r for r in json.load(open(args.resolver))
                if r["encoder"] == "dinov2"}

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.1))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.80, bottom=0.20,
                        wspace=0.28)
    stats = {}
    for ax, ds in zip(axes, DS):
        for shots, marker, col in zip(SHOTS, ["o", "s"],
                                      ["#1F77B4", "#D62728"]):
            Ts, mu, se = [], [], []
            for i in range(9):
                p = os.path.join(args.sweep_dir, ds, f"T{i}",
                                 f"headline_checkpoint_{ds}.json")
                if not os.path.exists(p):
                    continue
                ck = json.load(open(p))
                cell = ck["cells"].get(f"frozen|balanced_both|{shots}")
                if not cell or not cell["trials"]:
                    continue
                sz = np.array([t[AK]["sz"] for t in cell["trials"]
                               if AK in t])
                Ts.append(float(cell["T"]))
                mu.append(sz.mean())
                se.append(sz.std() / np.sqrt(len(sz)))
            o = np.argsort(Ts)
            Ts, mu, se = np.array(Ts)[o], np.array(mu)[o], np.array(se)[o]
            ax.errorbar(Ts, mu, yerr=se, marker=marker, ms=3, lw=1.2,
                        capsize=2, color=col, label=f"{shots} shots")
            stats.setdefault(ds, {})[shots] = {
                "T": Ts.tolist(), "sz": mu.tolist()}
        T_pool = resolver[ds]["T_pool"]
        ax.axvline(T_pool, color="k", ls="--", lw=1.0,
                   label="deployed $T$ (pool)")
        stats[ds]["T_pool"] = T_pool
        ax.set_xscale("log")
        ax.set_title(f"{DS_LABEL[ds]}", pad=3)
        ax.set_xlabel("fixed temperature $T$")
    axes[0].set_ylabel("mean set size")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.99))
    stem = os.path.join(args.out_dir, "fig_T_sensitivity")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=250, bbox_inches="tight")
    with open(os.path.join(args.out_dir, "T_sensitivity_stats.json"),
              "w") as f:
        json.dump(stats, f, indent=2)
    print(f"saved {stem}.pdf/.png")


if __name__ == "__main__":
    main()
