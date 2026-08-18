"""
Recap figures for the qe x SNAPS stacking grid (docs/pool_repr_menu_plan.md
sec 7; JSONs in output/pool_repr_menu/snaps_stack/).

Figure 1  the 2x2 corners: base / best-SNAPS / qe / qe+best-SNAPS set sizes,
          grouped bars per cal, one panel per dataset (linear y).
Figure 2  the mechanism: (a) SNAPS marginal gain raw vs post-qe (the
          cannibalization collapse), (b) neighbor purity raw vs qe space
          (the homophily lift), (c) eta* raw vs post-qe (the shrink-to-zero
          signature).

Usage:
python src/plot_qe_snaps_stack.py --data_dir output/pool_repr_menu/snaps_stack
"""
import os, json, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = [("cifar100_p128", "CIFAR-100 (pca128_cw)"),
            ("mini_p128", "miniImageNet (pca128_cw)"),
            ("cub200_p128", "CUB-200 (pca128_cw)"),
            ("cub200_p512", "CUB-200 (pca512_cw, champion)")]
CORNERS = ["base", "SNAPS", "qe", "qe+SNAPS"]
COLORS = ["#9e9e9e", "#5b8db8", "#d1862c", "#7a5aa0"]


def load(data_dir, ds):
    out = {}
    for pre in ("none", "qe"):
        with open(os.path.join(data_dir, f"results_{ds}_{pre}.json")) as f:
            out[pre] = json.load(f)["rows"]
    return out


def cells_for(rows_by_pre, cal):
    """corner -> (sz, sz_se, eta*, purity_of_best_k) for one cal."""
    out = {}
    for pre, base_lbl, snaps_lbl in (("none", "base", "SNAPS"),
                                     ("qe", "qe", "qe+SNAPS")):
        rows = [r for r in rows_by_pre[pre] if r["cal"] == cal]
        b = next(r for r in rows if r["eta"] == 0.0)
        s = min((r for r in rows if r["eta"] > 0), key=lambda r: r["sz"])
        out[base_lbl] = dict(sz=b["sz"], se=b["sz_se"], eta=0.0, pur=np.nan)
        out[snaps_lbl] = dict(sz=s["sz"], se=s["sz_se"], eta=s["eta"],
                              pur=s["purity"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="output/pool_repr_menu/snaps_stack")
    args = ap.parse_args()

    data = {ds: load(args.data_dir, ds) for ds, _ in DATASETS}

    # ---------------------------------------------------------- figure 1
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.6 * len(DATASETS), 4.4))
    for ax, (ds, title) in zip(axes, DATASETS):
        cals = sorted({r["cal"] for r in data[ds]["none"]})
        width = 0.2
        xs = np.arange(len(cals))
        for j, corner in enumerate(CORNERS):
            vals = [cells_for(data[ds], c)[corner] for c in cals]
            ax.bar(xs + (j - 1.5) * width, [v["sz"] for v in vals], width,
                   yerr=[v["se"] for v in vals], capsize=2,
                   label=corner, color=COLORS[j])
            for x, v in zip(xs, vals):
                ax.text(x + (j - 1.5) * width, v["sz"] + v["se"] + 0.03,
                        f"{v['sz']:.2f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(c) for c in cals])
        ax.set_xlabel("cal size")
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.18)
    axes[0].set_ylabel("mean set size")
    axes[0].legend(fontsize=8)
    fig.suptitle("qe x SNAPS stacking — 2x2 corners (20 trials, balanced, "
                 "prototype+poolT base; SNAPS = best eta x k arm)", fontsize=11)
    fig.tight_layout()
    p1 = os.path.join(args.data_dir, "stack_corners.png")
    fig.savefig(p1, dpi=150)
    print(f"Saved -> {p1}")

    # ---------------------------------------------------------- figure 2
    fig, (axg, axp, axe) = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) SNAPS marginal gain, raw vs post-qe, at each cal
    labels, g_raw, g_qe = [], [], []
    for ds, title in DATASETS:
        cals = sorted({r["cal"] for r in data[ds]["none"]})
        for cal in cals:
            c = cells_for(data[ds], cal)
            labels.append(f"{title.split(' ')[0]}\n{cal}")
            g_raw.append(100 * (c["SNAPS"]["sz"] - c["base"]["sz"]) / c["base"]["sz"])
            g_qe.append(100 * (c["qe+SNAPS"]["sz"] - c["qe"]["sz"]) / c["qe"]["sz"])
    xs = np.arange(len(labels))
    axg.bar(xs - 0.2, g_raw, 0.4, label="SNAPS gain on RAW base", color="#5b8db8")
    axg.bar(xs + 0.2, g_qe, 0.4, label="SNAPS gain on QE base", color="#7a5aa0")
    axg.axhline(0, color="k", lw=0.8)
    axg.set_xticks(xs)
    axg.set_xticklabels(labels, fontsize=6, rotation=90)
    axg.set_ylabel("SNAPS marginal set-size change (%)")
    axg.set_title("(a) the cannibalization collapse")
    axg.legend(fontsize=8)

    # (b) neighbor purity (k=10 / k=20), raw vs qe space
    labels, pr10, pq10, pr20, pq20 = [], [], [], [], []
    for ds, title in DATASETS:
        cals = sorted({r["cal"] for r in data[ds]["none"]})
        cal = cals[0]
        row = {}
        for pre in ("none", "qe"):
            rows = [r for r in data[ds][pre] if r["cal"] == cal and r["eta"] > 0]
            for k in (10, 20):
                ks = [r["purity"] for r in rows if r.get("k") == k]
                row[(pre, k)] = float(np.nanmean(ks)) if ks else np.nan
        labels.append(title.split(" ")[0] + ("(512)" if "512" in ds else ""))
        pr10.append(row[("none", 10)]); pq10.append(row[("qe", 10)])
        pr20.append(row[("none", 20)]); pq20.append(row[("qe", 20)])
    xs = np.arange(len(labels))
    axp.bar(xs - 0.3, pr10, 0.2, label="raw, k=10", color="#5b8db8")
    axp.bar(xs - 0.1, pq10, 0.2, label="qe, k=10", color="#d1862c")
    axp.bar(xs + 0.1, pr20, 0.2, label="raw, k=20", color="#a8c4dd", hatch="//")
    axp.bar(xs + 0.3, pq20, 0.2, label="qe, k=20", color="#eec99a", hatch="//")
    axp.axhline(0.7, color="r", ls="--", lw=0.8)
    axp.text(len(labels) - 0.55, 0.705, "SNAPS break-even", color="r", fontsize=7)
    axp.set_xticks(xs)
    axp.set_xticklabels(labels, fontsize=8)
    axp.set_ylabel("kNN neighbor label purity")
    axp.set_ylim(0.5, 1.0)
    axp.set_title("(b) qe raises graph homophily")
    axp.legend(fontsize=7)

    # (c) eta* raw vs post-qe per dataset x cal
    labels, e_raw, e_qe = [], [], []
    for ds, title in DATASETS:
        cals = sorted({r["cal"] for r in data[ds]["none"]})
        for cal in cals:
            c = cells_for(data[ds], cal)
            labels.append(f"{title.split(' ')[0]}\n{cal}")
            e_raw.append(c["SNAPS"]["eta"])
            e_qe.append(c["qe+SNAPS"]["eta"])
    xs = np.arange(len(labels))
    axe.bar(xs - 0.2, e_raw, 0.4, label="eta* on RAW base", color="#5b8db8")
    axe.bar(xs + 0.2, e_qe, 0.4, label="eta* on QE base", color="#7a5aa0")
    axe.set_xticks(xs)
    axe.set_xticklabels(labels, fontsize=6, rotation=90)
    axe.set_ylabel("best SNAPS mixing weight eta*")
    axe.set_ylim(0, 0.8)
    axe.set_title("(c) eta* shrinks once qe is under it")
    axe.legend(fontsize=8)

    fig.suptitle("qe x SNAPS mechanism: purity rises, but the correction has "
                 "nothing left to harvest", fontsize=11)
    fig.tight_layout()
    p2 = os.path.join(args.data_dir, "stack_mechanism.png")
    fig.savefig(p2, dpi=150)
    print(f"Saved -> {p2}")


if __name__ == "__main__":
    main()
