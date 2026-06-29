"""
"Selling" figures for the two novelty directions, built from the pilot JSONs.
Run ONLY after the confirmations pass. Conclusion-titled, annotated, intuitive.

  Fig 1  reliability_sell.png   -- Direction A (PAC reliability)
  Fig 2  geometry_sell.png      -- Direction B (geometry-conditional coverage)

Reads output/reliability/<ds>/reliability_cal<c>.json and
output/geometric_coverage/<ds>/geometric_coverage.json for ds in the datasets
that exist. Plain-text math labels (repo convention).
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_FULL = "#1b7837"   # full CP = green (good)
C_FULL2 = "#5aae61"
C_SCP = "#c2334d"    # split CP = red (bad)
C_RAND = "#3b6fb6"   # exact random arm = blue
ALPHA = 0.1
CALS = [200, 400, 800]
CLASS_COVGAP = 7.68  # project's class-conditional CovGap @cal800 (findings 2b)


def beta_sd(n):
    return np.sqrt(ALPHA * (1 - ALPHA) / (n + 2))


def load_rel(ds, root):
    out = {}
    for c in CALS:
        p = os.path.join(root, "reliability", ds, f"reliability_cal{c}.json")
        if os.path.exists(p):
            out[c] = json.load(open(p))
    return out


def load_geo(ds, root):
    p = os.path.join(root, "geometric_coverage", ds, "geometric_coverage.json")
    return json.load(open(p)) if os.path.exists(p) else None


# ----------------------------- FIG 1: reliability -------------------------
def fig_reliability(datasets, root, out_dir):
    data = {ds: load_rel(ds, root) for ds in datasets}
    data = {ds: d for ds, d in data.items() if d}
    nds = len(data)
    fig, axes = plt.subplots(nds, 3, figsize=(15, 4.4 * nds), squeeze=False)
    for r, (ds, rel) in enumerate(data.items()):
        cals = [c for c in CALS if c in rel]
        # ---- panel 0: coverage SD vs cal (reliability variance) ----
        ax = axes[r][0]
        sd_proto = [rel[c]["summary"]["full_proto_bal"]["coverage"]["sd"] for c in cals]
        fk_scp = [rel[c]["summary"]["scp_thr_bal"]["frac_K"]["mean"] for c in cals]
        sd_scp = [rel[c]["summary"]["scp_thr_bal"]["coverage"]["sd"] for c in cals]
        ax.plot(cals, sd_proto, "o-", c=C_FULL, lw=2.4, ms=8, label="Full CP (prototype)")
        # SCP: only the non-degenerate points (degenerate => constant size-K, SD trivially ~0)
        ok = [(c, s) for c, s, fk in zip(cals, sd_scp, fk_scp) if fk < 0.3]
        deg = [c for c, fk in zip(cals, fk_scp) if fk >= 0.3]
        if ok:
            ax.plot([c for c, _ in ok], [s for _, s in ok], "s-", c=C_SCP, lw=2.4, ms=8,
                    label="Split CP-THR (matched B)")
        for c in deg:
            ax.scatter([c], [0.0008], marker="x", c=C_SCP, s=70, zorder=5)
        if deg:
            ax.text(deg[0], 0.0018, "Split CP degenerate\n(sets = K, SD trivial)",
                    fontsize=7, c=C_SCP)
        ax.plot(cals, [beta_sd(c) for c in cals], "--", c=C_FULL, lw=1.2, alpha=.7,
                label="Beta(n=B) ideal [all B -> rank]")
        ax.plot(cals, [beta_sd(c / 2) for c in cals], "--", c=C_SCP, lw=1.2, alpha=.7,
                label="Beta(n=B/2) [split spends half]")
        ax.set_ylim(bottom=0)
        ax.set_xticks(cals); ax.set_xlabel("label budget B"); ax.set_ylabel("coverage SD over draws")
        ax.set_title("Full CP's coverage is tighter\n(rides the all-B Beta rate)", fontsize=11)
        ax.legend(fontsize=7.5); ax.grid(alpha=.25)
        # ---- panel 1: set-size volatility vs cal (the collapse) ----
        ax = axes[r][1]
        for arm, c_, lab in [("full_proto_bal", C_FULL, "Full CP (prototype)"),
                             ("scp_thr_bal", C_SCP, "Split CP-THR (matched B)")]:
            med, lo, hi = [], [], []
            for c in cals:
                a = np.array(rel[c]["raw"][arm]["mean_sz"])
                med.append(np.median(a)); lo.append(np.percentile(a, 10)); hi.append(np.percentile(a, 90))
            ax.plot(cals, med, "o-", c=c_, lw=2.4, ms=8, label=lab)
            ax.fill_between(cals, lo, hi, color=c_, alpha=.18)
        ax.axhline(100, ls=":", c="grey", lw=1); ax.text(cals[0], 104, "K=100 (useless)", fontsize=7, c="grey")
        ax.set_yscale("log"); ax.set_xticks(cals); ax.set_xlabel("label budget B")
        ax.set_ylabel("mean set size (median, p10-p90)")
        # annotate SCP collapse
        fk = rel[cals[0]]["summary"]["scp_thr_bal"]["frac_K"]["mean"]
        ax.annotate(f"Split CP collapses:\n{fk*100:.0f}% of sets = all K",
                    xy=(cals[0], rel[cals[0]]["raw"]["scp_thr_bal"]["mean_sz"][0]),
                    xytext=(cals[0] + 60, 30), fontsize=8, c=C_SCP,
                    arrowprops=dict(arrowstyle="->", color=C_SCP))
        ax.set_title("Split CP collapses at few-shot;\nFull CP stays tight", fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=.25, which="both")
        # ---- panel 2: PAC size-cost (price of a 95%-reliable 0.90) ----
        ax = axes[r][2]
        cbig = cals[-1]
        names, prices, base = [], [], []
        for arm, lab, c_ in [("full_proto_bal", "Full CP", C_FULL),
                             ("scp_thr_bal", "Split CP-THR", C_SCP)]:
            pac = rel[cbig]["summary"][arm]["pac"].get("delta_0.05", {})
            if pac.get("pac_price") is not None:
                names.append(lab); prices.append(pac["pac_price"]); base.append(pac["size_at_0.10"])
        x = np.arange(len(names))
        bars = ax.bar(x, base, color=[C_FULL, C_SCP][:len(names)], alpha=.45, label="size @ nominal 0.90")
        ax.bar(x, prices, bottom=base, color=[C_FULL, C_SCP][:len(names)], alpha=.95,
               label="extra size for 95%-reliable 0.90")
        for i, (b, p) in enumerate(zip(base, prices)):
            ax.text(i, b + p + .15, f"+{p:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("mean set size")
        ax.set_title(f"Price of a PAC guarantee (B={cbig}):\nFull CP buys reliability ~cheaply", fontsize=11)
        ax.legend(fontsize=7.5)
        axes[r][0].annotate(ds, xy=(-0.22, 0.5), xycoords="axes fraction", rotation=90,
                            fontsize=13, fontweight="bold", va="center")
    fig.suptitle("At a fixed label budget, Full CP is RELIABLE; budget-splitting Split CP is not",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    p = os.path.join(out_dir, "reliability_sell.png")
    fig.savefig(p, dpi=140); plt.close(fig); print(f"[fig] {p}")


# ----------------------------- FIG 2: geometry ----------------------------
def _rows(geo, key, cov):
    m = geo["results"][key]["by_covariate"][cov]
    xs = [r["stratum"] for r in m["rows"] if r["cov"] is not None]
    ys = [r["cov"] for r in m["rows"] if r["cov"] is not None]
    return xs, ys, m


def fig_geometry(datasets, root, out_dir):
    geos = {ds: load_geo(ds, root) for ds in datasets}
    geos = {ds: g for ds, g in geos.items() if g}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    cmap = {"cifar100": "#1b7837", "miniimagenet": "#762a83"}
    cbig = 800
    # panel 0: per-stratum coverage by DENSITY (the hero)
    ax = axes[0]
    for ds, g in geos.items():
        c_ = cmap.get(ds, "#444")
        xs, ys, m = _rows(g, f"{cbig}_balanced_prototype", "density")
        ax.plot(xs, ys, "o-", c=c_, lw=2.6, ms=8, label=f"{ds} balanced (gap {m['covgap_geo']:.1f}pp)")
        if f"{cbig}_random_geodesic" in g["results"]:
            xs2, ys2, m2 = _rows(g, f"{cbig}_random_geodesic", "density")
            ax.plot(xs2, ys2, "s--", c=c_, lw=1.6, ms=6, alpha=.6,
                    label=f"{ds} random/exact (gap {m2['covgap_geo']:.1f}pp)")
    ax.axhline(1 - ALPHA, ls="--", c="k", lw=1.2); ax.axhspan(0.6, 1 - ALPHA, color="red", alpha=.05)
    ax.text(0.05, 1 - ALPHA + .004, "target 0.90", fontsize=8)
    ax.set_xlabel("local density stratum  (dense -> sparse)"); ax.set_ylabel("coverage")
    ax.set_title("Full CP under-covers in SPARSE regions\n(monotone, even on the exact split)", fontsize=11)
    ax.legend(fontsize=7.5); ax.grid(alpha=.25)
    # panel 1: per-stratum coverage by LID (cross-covariate confirm)
    ax = axes[1]
    for ds, g in geos.items():
        c_ = cmap.get(ds, "#444")
        xs, ys, m = _rows(g, f"{cbig}_balanced_prototype", "lid")
        ax.plot(xs, ys, "o-", c=c_, lw=2.6, ms=8, label=f"{ds} (gap {m['covgap_geo']:.1f}pp)")
    ax.axhline(1 - ALPHA, ls="--", c="k", lw=1.2); ax.axhspan(0.6, 1 - ALPHA, color="red", alpha=.05)
    ax.set_xlabel("local intrinsic dimension stratum  (low -> high)"); ax.set_ylabel("coverage")
    ax.set_title("Same story by LID:\nharder (high-LID) points under-cover", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    # panel 2: worst-stratum coverage vs cal (tail risk grows)
    ax = axes[2]
    for ds, g in geos.items():
        c_ = cmap.get(ds, "#444")
        cs, ws, gaps = [], [], []
        for c in CALS:
            key = f"{c}_balanced_prototype"
            if key in g["results"]:
                m = g["results"][key]["by_covariate"]["density"]
                cs.append(c); ws.append(m["worst"]); gaps.append(m["covgap_geo"])
        if cs:
            ax.plot(cs, ws, "o-", c=c_, lw=2.6, ms=8, label=f"{ds} worst-stratum cov")
    ax.axhline(1 - ALPHA, ls="--", c="k", lw=1.2)
    ax.set_xticks(CALS); ax.set_xlabel("label budget B"); ax.set_ylabel("worst-stratum coverage")
    ax.set_title("Tail miscoverage persists as sets tighten\n(marginal stays ~0.90)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.suptitle("Marginal coverage hides it: Full CP coverage is NOT uniform across the SSL embedding "
                 "(label-free geometry predicts where it fails)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, "geometry_sell.png")
    fig.savefig(p, dpi=140); plt.close(fig); print(f"[fig] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "miniimagenet"])
    ap.add_argument("--out_dir", default="output/novelty_selling")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    fig_reliability(a.datasets, a.root, a.out_dir)
    fig_geometry(a.datasets, a.root, a.out_dir)


if __name__ == "__main__":
    main()
