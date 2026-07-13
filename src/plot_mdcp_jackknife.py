"""
Figure for the authors'-JK+ vs our-MDCP comparison
(src/mdcp_jackknife_compare.py results JSON, incl. pilot-D fcp overlays).

Layout 2x3:
  (0,0)/(1,0) mean set size vs cal (balanced / random), log y
  (0,1)/(1,1) coverage vs cal, target 0.9 + JK+ 1-2alpha floor 0.8
  (0,2)      measured JK+ wall time vs m (both splits) + O(m(m+n)K) guide
  (1,2)      analytic JK+ memory map: 9*m*n_test*K bytes vs m for several
             n_test, 16GB laptop / 48GB cluster ceilings, authors' regime
             (m=9000, n=6000) and our grid marked

python src/plot_mdcp_jackknife.py --results <...>/mdcp_jackknife_compare_results.json
"""
import os, json, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    ("raw1_proto128", "1-D split CP (proto@pca128)", "0.5", "--", "o"),
    ("dratio2_proto", "OURS split-style pool D-ratio (pilot B)", "tab:blue", "-", "s"),
    ("jk2", "THEIRS Jackknife+ multi-score, H=2 (Alg B.1)", "tab:red", "-", "^"),
    ("jk1", "THEIRS Jackknife+, H=1", "tab:orange", "--", "v"),
]
FCP_ARM = ("fcp_bag", "OURS full-CP MDCP bag (pilot D, exact)", "tab:green", "-", "D")


def series(results, split, cals, arm):
    xs, mu, se = [], [], []
    for c in cals:
        key = f"{split}_cal{c}"
        if key in results and arm in results[key]:
            xs.append(c)
            mu.append(results[key][arm]["size"])
            se.append(results[key][arm]["size_se"])
    return xs, mu, se


def series_cov(results, split, cals, arm):
    xs, mu, se = [], [], []
    for c in cals:
        key = f"{split}_cal{c}"
        if key in results and arm in results[key]:
            xs.append(c)
            mu.append(results[key][arm]["cov"])
            se.append(results[key][arm]["cov_se"])
    return xs, mu, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    res = data["results"]
    cfg = data["config"]
    cals = sorted(cfg["cal_sizes"])
    splits = cfg["splits"]
    fcp = data.get("fcp_overlays", {})

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, split in enumerate(splits):
        ax_s, ax_c = axes[row, 0], axes[row, 1]
        for arm, label, color, ls, mk in ARMS:
            xs, mu, se = series(res, split, cals, arm)
            ax_s.errorbar(xs, mu, yerr=se, color=color, ls=ls, marker=mk,
                          capsize=3, label=label)
            xs, mu, se = series_cov(res, split, cals, arm)
            ax_c.errorbar(xs, mu, yerr=se, color=color, ls=ls, marker=mk,
                          capsize=3, label=label)
        if split in fcp:
            arm, label, color, ls, mk = FCP_ARM
            xs, mu, se, cv, cvse = [], [], [], [], []
            for c in sorted(fcp[split], key=lambda x: int(x)):
                d = fcp[split][c].get(arm)
                if d:
                    xs.append(int(c)); mu.append(d["size"]); se.append(d["size_se"])
                    cv.append(d["cov"]); cvse.append(d["cov_se"])
            ax_s.errorbar(xs, mu, yerr=se, color=color, ls=ls, marker=mk,
                          capsize=3, label=label)
            ax_c.errorbar(xs, cv, yerr=cvse, color=color, ls=ls, marker=mk,
                          capsize=3, label=label)
        ax_s.set_yscale("log")
        ax_s.set_xlabel("cal size m"); ax_s.set_ylabel("mean set size (log)")
        ax_s.set_title(f"{split}: set size")
        ax_s.set_xticks(cals); ax_s.grid(alpha=0.3)
        ax_c.axhline(0.9, color="k", lw=0.8)
        ax_c.axhline(0.8, color="tab:red", lw=0.8, ls=":",
                     label="JK+ theoretical floor (1-2a)")
        ax_c.set_xlabel("cal size m"); ax_c.set_ylabel("coverage")
        ax_c.set_title(f"{split}: coverage (target 0.90)")
        ax_c.set_xticks(cals); ax_c.grid(alpha=0.3)
        if row == 0:
            ax_c.legend(fontsize=7, loc="lower right")

    # --- feasibility: measured time ---
    ax_t = axes[0, 2]
    for split, color in zip(splits, ("tab:blue", "tab:purple")):
        xs, secs = [], []
        for c in cals:
            key = f"{split}_cal{c}"
            if key in res and "_jk2_meta" in res[key]:
                xs.append(c); secs.append(res[key]["_jk2_meta"]["sec_mean"])
        ax_t.plot(xs, secs, "o-", color=color, label=f"jk2 measured ({split})")
    if xs and secs:
        n_cap = cfg.get("test_cap", 500)
        guide = [secs[-1] * (m * (m + n_cap)) / (xs[-1] * (xs[-1] + n_cap))
                 for m in xs]
        ax_t.plot(xs, guide, "k:", label="O(m(m+n)K) guide")
    ax_t.set_xlabel("cal size m"); ax_t.set_ylabel("JK+ seconds / trial")
    ax_t.set_title(f"JK+ wall time (n_test={cfg.get('test_cap')}, K=100, CPU)")
    ax_t.grid(alpha=0.3); ax_t.legend(fontsize=8)

    # --- feasibility: analytic memory map ---
    ax_m = axes[1, 2]
    K = 100
    m_grid = np.logspace(2, 4.2, 100)
    for n_test, color in ((500, "tab:blue"), (1000, "tab:cyan"),
                          (6000, "tab:orange"), (10000, "tab:red")):
        gb = 9.0 * m_grid * n_test * K / 1e9   # int64 attribution + bool votes
        ax_m.plot(m_grid, gb, color=color, label=f"n_test={n_test}")
    ax_m.axhline(16, color="k", ls="--", lw=0.8)
    ax_m.text(120, 17, "16GB laptop", fontsize=7)
    ax_m.axhline(48, color="k", ls=":", lw=0.8)
    ax_m.text(120, 52, "48GB cluster", fontsize=7)
    ax_m.plot([9000], [9.0 * 9000 * 6000 * K / 1e9], "r*", ms=14,
              label="authors' regime (m=9000, n=6000)")
    ax_m.plot([200, 400, 800], [9.0 * m * 500 * K / 1e9 for m in (200, 400, 800)],
              "ks", ms=5, label="this run")
    ax_m.set_xscale("log"); ax_m.set_yscale("log")
    ax_m.set_xlabel("cal size m"); ax_m.set_ylabel("attribution memory (GB)")
    ax_m.set_title("JK+ memory = 9*m*n_test*K bytes (K=100)")
    ax_m.grid(alpha=0.3, which="both"); ax_m.legend(fontsize=7, loc="upper left")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    for h, l in zip(h2, l2):
        if l not in labels:
            handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               frameon=False)
    fig.suptitle(
        "Authors' Jackknife+ Multi-Score CP (verbatim Alg B.1) vs our MDCP arms -- "
        f"CIFAR-100, dims {' x '.join('proto:' + v for v in cfg['dims_views'])}, "
        f"{cfg['n_trials']} trials, alpha={cfg['alpha']} "
        "(note: JK+ force-includes one label in empty sets)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out = args.out or os.path.join(os.path.dirname(args.results),
                                   "fig_jackknife_compare.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
