"""
Champion comparison figures: the adopted pool-fit pipeline (qe -> ldapool)
vs the previous PCA+whitening menu, with and without the SNAPS score
correction, plus raw-embedding context.

Fig A  champion_lines.png    set size vs cal, 4 series x 4 datasets
Fig B  champion_gains.png    % change vs the old menu champion
Fig C  champion_covgap.png   class-conditional CovGap, old vs new

Sources (all 10-20 trial balanced runs under output/pool_repr_menu/):
  old menu / raw768     results_{ds}.json (+ miniimagenet_confirm)
  old + SNAPS           snaps_stack/results_*_none.json (best eta x k arm;
                        aircraft: measured champion-base harm +13..37%
                        [2026-07-28 stage-2] -> annotation, no curve)
  new champion          ldapool/ JSONs (cifar qe_ldapool192; CUB
                        qe_ldapool512@C300 best of classic/safe per cal;
                        aircraft qe_ldapool512 safe; mini qe_ldapool128)

Usage: python src/plot_champion_comparison.py --base output/pool_repr_menu
"""
import os, json, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows_of(path):
    with open(path) as f:
        return json.load(f)["rows"]


def tc_series(rows, arm, ncm, key="sz"):
    out = {}
    for r in rows:
        if r["arm"] == arm and r["ncm"] == ncm:
            out[r["cal"]] = (r[key], r.get(key + "_se", 0.0))
    return out


def snaps_best(rows):
    out = {}
    for cal in sorted({r["cal"] for r in rows}):
        arms = [r for r in rows if r["cal"] == cal and r["eta"] > 0]
        best = min(arms, key=lambda r: r["sz"])
        out[cal] = (best["sz"], best["sz_se"])
    return out


def build(base):
    P, G = "prototype_softmax", "unwhitened_topk_mean"
    ds = {}
    r1c = rows_of(f"{base}/results_cifar100.json")
    r1b = rows_of(f"{base}/results_cub200.json")
    r1a = rows_of(f"{base}/results_aircraft.json")
    rmc = rows_of(f"{base}/results_miniimagenet_confirm.json")

    def best_of(series_list, names=None):
        """Per-cal min across series; if arm names given, also return the
        winning-arm label string (so the plot can say WHICH menu arm the
        'old' curve is in each panel — e.g. aircraft = full-768 LW, not
        PCA truncation)."""
        out, won = {}, {}
        for i, s in enumerate(series_list):
            for cal, v in s.items():
                if cal not in out or v[0] < out[cal][0]:
                    out[cal] = v
                    won[cal] = names[i] if names else None
        if names is None:
            return out
        uniq = sorted(set(won.values()))
        if len(uniq) == 1:
            label = uniq[0]
        else:
            label = ", ".join(f"{a}@{c}" for c, a in sorted(won.items()))
        return out, label

    MENU = ("pca128_cw", "pca512_cw", "lw_cluster768")
    old_c, lbl_c = best_of([tc_series(r1c, a, P) for a in MENU], MENU)
    old_b, lbl_b = best_of([tc_series(r1b, a, P) for a in MENU], MENU)
    old_a, lbl_a = best_of([tc_series(r1a, a, G) for a in MENU], MENU)
    ds["CIFAR-100"] = dict(
        ncm="prototype", cals=[200, 400, 800],
        raw=tc_series(r1c, "raw768", P),
        old=old_c, old_label=lbl_c,
        snaps=snaps_best(rows_of(f"{base}/snaps_stack/results_cifar100_p128_none.json")),
        new=tc_series(rows_of(f"{base}/ldapool/results_cifar100_dscan.json"),
                      "qe_ldapool192", P),
        old_cg=best_of([tc_series(r1c, "pca128_cw", P, "covgap")]),
        new_cg=tc_series(rows_of(f"{base}/ldapool/results_cifar100_dscan.json"),
                         "qe_ldapool192", P, "covgap"))
    ds["CUB-200"] = dict(
        ncm="prototype", cals=[400, 800, 1600],
        raw=tc_series(r1b, "raw768", P),
        old=old_b, old_label=lbl_b,
        snaps=best_of([snaps_best(rows_of(f"{base}/snaps_stack/results_cub200_p128_none.json")),
                       snaps_best(rows_of(f"{base}/snaps_stack/results_cub200_p512_none.json"))]),
        new=best_of([tc_series(rows_of(f"{base}/ldapool/results_cub200_qec300.json"),
                               "qe_ldapool512", P),
                     tc_series(rows_of(f"{base}/ldapool/results_cub200_safeqec300.json"),
                               "qe_ldapool512", P)]),
        old_cg=tc_series(r1b, "pca512_cw", P, "covgap"),
        new_cg=tc_series(rows_of(f"{base}/ldapool/results_cub200_safeqec300.json"),
                         "qe_ldapool512", P, "covgap"))
    ds["Aircraft"] = dict(
        ncm="geodesic", cals=[200, 400, 800],
        raw=tc_series(r1a, "raw768", G),
        old=old_a, old_label=lbl_a + " (full 768-d, no PCA cut)"
            if lbl_a == "lw_cluster768" else lbl_a,
        snaps=None,   # champion-base harm +13..37% (2026-07-28), gated off
        new=tc_series(rows_of(f"{base}/ldapool/results_aircraft_safeqe.json"),
                      "qe_ldapool512", G),
        old_cg=tc_series(r1a, "lw_cluster768", G, "covgap"),
        new_cg=tc_series(rows_of(f"{base}/ldapool/results_aircraft_safeqe.json"),
                         "qe_ldapool512", G, "covgap"))
    ds["miniImageNet"] = dict(
        ncm="prototype", cals=[200, 400, 800],
        raw=tc_series(rmc, "raw768", P),
        old=tc_series(rmc, "pca128_cw", P), old_label="pca128_cw",
        snaps=snaps_best(rows_of(f"{base}/snaps_stack/results_mini_p128_none.json")),
        new=tc_series(rows_of(f"{base}/ldapool/results_miniimagenet.json"),
                      "qe_ldapool128", P),
        old_cg=tc_series(rmc, "pca128_cw", P, "covgap"),
        new_cg=tc_series(rows_of(f"{base}/ldapool/results_miniimagenet.json"),
                         "qe_ldapool128", P, "covgap"))
    return ds


STYLE = {"raw": dict(color="#9e9e9e", ls=":", marker="s", lw=1.5,
                     label="raw embeddings (no pool repr.)"),
         "old": dict(color="#5b8db8", ls="-", marker="o", lw=1.8,
                     label="old pool-fit menu, per-cell best arm\n"
                           "(PCA+whiten OR full-768 LW; see panel note)"),
         "snaps": dict(color="#2f5d82", ls="--", marker="^", lw=1.8,
                       label="old + SNAPS score corr. (best eta x k)"),
         "new": dict(color="#d1862c", ls="-", marker="D", lw=2.6,
                     label="NEW: qe -> discriminant (one construction)")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output/pool_repr_menu")
    args = ap.parse_args()
    ds = build(args.base)
    outd = os.path.join(args.base, "champion")
    os.makedirs(outd, exist_ok=True)

    # ---------------- Fig A: size vs cal ----------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.6))
    for ax, (name, d) in zip(axes, ds.items()):
        cals = d["cals"]
        nonraw_max = max(v[0] for k in ("old", "snaps", "new")
                         if d.get(k) for v in d[k].values())
        ylim = nonraw_max * 1.22
        for key in ("raw", "old", "snaps", "new"):
            s = d.get(key)
            if not s:
                continue
            xs = [c for c in cals if c in s]
            ys = [min(s[c][0], ylim * 0.985) for c in xs]
            ax.errorbar(xs, ys, yerr=[s[c][1] for c in xs],
                        capsize=2, **STYLE[key])
            for c, yplot in zip(xs, ys):
                v = s[c][0]
                clipped = v > ylim * 0.985
                ax.annotate(f"{v:.2f}" + ("^" if clipped else ""),
                            (c, yplot), textcoords="offset points",
                            xytext=(0, 6 if key != "old" else -13),
                            fontsize=7, ha="center",
                            color=STYLE[key]["color"])
        if d["snaps"] is None:
            ax.text(0.5, 0.9, "SNAPS on champion base:\n+13..37% (harm, gated off)",
                    transform=ax.transAxes, ha="center", fontsize=8,
                    color=STYLE["snaps"]["color"], style="italic")
        ax.text(0.02, 0.02, f"old menu arm: {d['old_label']}",
                transform=ax.transAxes, fontsize=7.5,
                color=STYLE["old"]["color"], style="italic")
        ax.set_ylim(0, ylim)
        ax.set_xticks(cals)
        ax.set_xlabel("cal size")
        ax.set_title(f"{name} ({d['ncm']})", fontsize=11)
    axes[0].set_ylabel("mean prediction-set size")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Pool-fit representation champion vs the old menu and the score-level "
                 "correction (balanced, 10-20 trials; raw values clipped ^)",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(outd, "champion_lines.png")
    fig.savefig(p, dpi=150); print("Saved ->", p)

    # ---------------- Fig B: % change vs old champion ---------------------
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    labels, g_sn, g_new = [], [], []
    for name, d in ds.items():
        for c in d["cals"]:
            labels.append(f"{name}\n{c}")
            g_new.append(100 * (d["new"][c][0] - d["old"][c][0]) / d["old"][c][0])
            g_sn.append(100 * (d["snaps"][c][0] - d["old"][c][0]) / d["old"][c][0]
                        if d["snaps"] else np.nan)
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, g_sn, 0.4, color=STYLE["snaps"]["color"],
           label="old + SNAPS vs old")
    ax.bar(xs + 0.2, g_new, 0.4, color=STYLE["new"]["color"],
           label="NEW champion vs old")
    for x in xs[np.isnan(g_sn)]:
        ax.text(x - 0.2, 1.5, "harm\n(gated)", ha="center", fontsize=7,
                color=STYLE["snaps"]["color"], style="italic")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("set-size change vs old champion (%)")
    ax.set_title("Both levers vs the old PCA+whiten champion (negative = smaller sets)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(outd, "champion_gains.png")
    fig.savefig(p, dpi=150); print("Saved ->", p)

    # ---------------- Fig C: CovGap ---------------------------------------
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    labels, cg_old, cg_new = [], [], []
    for name, d in ds.items():
        for c in d["cals"]:
            if c in d["old_cg"] and c in d["new_cg"]:
                labels.append(f"{name}\n{c}")
                cg_old.append(d["old_cg"][c][0])
                cg_new.append(d["new_cg"][c][0])
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, cg_old, 0.4, color=STYLE["old"]["color"], label="old champion")
    ax.bar(xs + 0.2, cg_new, 0.4, color=STYLE["new"]["color"], label="NEW champion")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("class-conditional CovGap (pp)")
    ax.set_title("Conditional-coverage check: CovGap ties-or-improves on CIFAR, "
                 "mild cost (+0.5-1.2pp) on aircraft/mini/CUB@1600")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(outd, "champion_covgap.png")
    fig.savefig(p, dpi=150); print("Saved ->", p)


if __name__ == "__main__":
    main()
