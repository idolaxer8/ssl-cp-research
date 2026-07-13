"""
Figure for the authors'-JK+ vs our-MDCP comparison
(src/mdcp_jackknife_compare.py results JSON, incl. pilot-D fcp overlays).

Layout (rows = splits) x 3:
  col 0  mean set size vs cal, log y
  col 1  coverage vs cal, target 0.9 + JK+ 1-2alpha floor 0.8
  col 2  row 0: measured wall time per arm (ours dratio2 vs theirs jk2 vs
                shared dim-score cost; optional full-CP probe hline via
                --fcp_sec_per_trial)
         row 1 (if present): analytic JK+ memory map, 16GB/48GB ceilings,
                authors' regime (m=9000, n=6000) marked

python src/plot_mdcp_jackknife.py --results <...>/mdcp_jackknife_compare_results.json
"""
import os, json, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLES = {
    "dratio2": ("OURS split-style pool D-ratio (pilot B)", "tab:blue", "-", "s"),
    "jk2": ("THEIRS Jackknife+ multi-score (Alg B.1)", "tab:red", "-", "^"),
    "jk1": ("THEIRS Jackknife+, H=1", "tab:orange", "--", "v"),
    "jk2_rank": ("THEIRS JK+ in rank space (control)", "tab:purple", ":", "^"),
    "fcp_bag": ("OURS full-CP MDCP bag (pilot D, exact)", "tab:green", "-", "D"),
}
RAW_COLORS = ["0.45", "0.7"]


def series(results, split, cals, arm, field=0):
    xs, mu, se = [], [], []
    key_mu, key_se = ("size", "size_se") if field == 0 else ("cov", "cov_se")
    for c in cals:
        key = f"{split}_cal{c}"
        if key in results and arm in results[key]:
            xs.append(c)
            mu.append(results[key][arm][key_mu])
            se.append(results[key][arm][key_se])
    return xs, mu, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fcp_sec_per_trial", type=float, default=None,
                    help="measured full-CP seconds per trial (probe) for the "
                         "time panel annotation")
    ap.add_argument("--no_mem_panel", action="store_true")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    res = data["results"]
    cfg = data["config"]
    cals = sorted(cfg["cal_sizes"])
    splits = cfg["splits"]
    dim_labels = data.get("dim_labels", [])
    fcp = data.get("fcp_overlays", {})

    arm_order = [f"raw1_{l}" for l in dim_labels] + \
                ["dratio2", "jk2", "jk1", "jk2_rank"]
    present = [a for a in arm_order
               if any(a in res[k] for k in res)]

    n_rows = len(splits)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.7 * n_rows),
                             squeeze=False)
    for row, split in enumerate(splits):
        ax_s, ax_c = axes[row, 0], axes[row, 1]
        raw_i = 0
        for arm in present:
            if arm.startswith("raw1_"):
                label = f"1-D split CP ({arm[5:]})"
                color, ls, mk = RAW_COLORS[min(raw_i, 1)], "--", "o"
                raw_i += 1
            else:
                label, color, ls, mk = STYLES[arm]
            xs, mu, se = series(res, split, cals, arm, 0)
            ax_s.errorbar(xs, mu, yerr=se, color=color, ls=ls, marker=mk,
                          capsize=3, label=label)
            xs, mu, se = series(res, split, cals, arm, 1)
            ax_c.errorbar(xs, mu, yerr=se, color=color, ls=ls, marker=mk,
                          capsize=3, label=label)
        if split in fcp:
            label, color, ls, mk = STYLES["fcp_bag"]
            xs, mu, se, cv, cvse = [], [], [], [], []
            for c in sorted(fcp[split], key=lambda x: int(x)):
                d = fcp[split][c].get("fcp_bag")
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

    # --- wall time per arm ---
    ax_t = axes[0, 2]
    for arm, color, label in (("scores", "0.6", "shared dim scores (all arms)"),
                              ("dratio2", "tab:blue", "OURS dratio2 adds"),
                              ("jk2", "tab:red", "THEIRS jk2 adds"),
                              ("jk2_rank", "tab:purple", "THEIRS jk2 (rank space)")):
        xs, secs = [], []
        for c in cals:
            key = f"{splits[0]}_cal{c}"
            if key in res and arm in res[key].get("_arm_sec", {}):
                xs.append(c); secs.append(res[key]["_arm_sec"][arm])
        if xs:
            ax_t.plot(xs, secs, "o-", color=color, label=label)
    if args.fcp_sec_per_trial:
        ax_t.axhline(args.fcp_sec_per_trial, color="tab:green", ls="--", lw=1)
        ax_t.text(cals[0], args.fcp_sec_per_trial * 1.15,
                  f"OURS full CP: ~{args.fcp_sec_per_trial:.0f}s/trial "
                  f"(measured probe, {cfg.get('test_cap')} test pts)",
                  fontsize=7, color="tab:green")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("cal size m"); ax_t.set_ylabel("seconds / trial (log)")
    ax_t.set_title(f"wall time per arm ({splits[0]}, "
                   f"n_test={cfg.get('test_cap')}, CPU)")
    ax_t.grid(alpha=0.3, which="both"); ax_t.legend(fontsize=7)

    # --- analytic JK+ memory map ---
    if n_rows > 1 and not args.no_mem_panel:
        ax_m = axes[1, 2]
        K = 100
        m_grid = np.logspace(2, 4.2, 100)
        for n_test, color in ((500, "tab:blue"), (1000, "tab:cyan"),
                              (6000, "tab:orange"), (10000, "tab:red")):
            gb = 9.0 * m_grid * n_test * K / 1e9
            ax_m.plot(m_grid, gb, color=color, label=f"n_test={n_test}")
        ax_m.axhline(16, color="k", ls="--", lw=0.8)
        ax_m.text(120, 17, "16GB laptop", fontsize=7)
        ax_m.axhline(48, color="k", ls=":", lw=0.8)
        ax_m.text(120, 52, "48GB cluster", fontsize=7)
        ax_m.plot([9000], [9.0 * 9000 * 6000 * K / 1e9], "r*", ms=14,
                  label="authors' regime (m=9000, n=6000)")
        ax_m.plot(cals, [9.0 * m * cfg.get("test_cap", 500) * K / 1e9
                         for m in cals], "ks", ms=5, label="this run")
        ax_m.set_xscale("log"); ax_m.set_yscale("log")
        ax_m.set_xlabel("cal size m"); ax_m.set_ylabel("attribution memory (GB)")
        ax_m.set_title("JK+ memory = 9*m*n_test*K bytes (K=100)")
        ax_m.grid(alpha=0.3, which="both"); ax_m.legend(fontsize=7, loc="upper left")
    elif n_rows == 1:
        pass

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               frameon=False)
    ds = os.path.basename(cfg["embeddings_path"]).replace("embeddings_", "") \
        .replace(".pt", "")
    fig.suptitle(
        "Authors' Jackknife+ Multi-Score CP (verbatim Alg B.1) vs our MDCP arms -- "
        f"{ds}, dims {' x '.join(cfg['dims'])}, {cfg['n_trials']} trials, "
        f"alpha={cfg['alpha']} (JK+ force-includes one label in empty sets)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0.12 / n_rows, 1, 0.95))
    out = args.out or os.path.join(os.path.dirname(args.results),
                                   "fig_jackknife_compare.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
