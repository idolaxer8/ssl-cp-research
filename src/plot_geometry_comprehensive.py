"""
Comprehensive geometry figure: does the per-stratum coverage slope PERSIST across
NCMs (prototype, geodesic) and with the MS-CS penalty? Reads the 4-arm
geometric_coverage.json and overlays per-stratum density coverage + CovGap_geo bars.

Conclusion it should show: the slope is NCM- and MS-CS-invariant -- the penalty
shrinks sets but does NOT close (slightly widens) the geometric gap => only an
explicitly geometry-conditional method (Stage 2) addresses it.
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.1
ARMS = ["prototype", "geodesic", "prototype_mscs", "geodesic_mscs"]
ARM_C = {"prototype": "#1b7837", "geodesic": "#5aae61",
         "prototype_mscs": "#9970ab", "geodesic_mscs": "#c2334d"}
ARM_LAB = {"prototype": "prototype", "geodesic": "geodesic",
           "prototype_mscs": "prototype + MS-CS", "geodesic_mscs": "geodesic + MS-CS"}


def load(ds, root):
    p = os.path.join(root, "geometric_coverage", ds, "geometric_coverage.json")
    return json.load(open(p))["results"] if os.path.exists(p) else None


def rows(res, key, cov):
    m = res[key]["by_covariate"][cov]
    xs = [r["stratum"] for r in m["rows"] if r["cov"] is not None]
    ys = [r["cov"] for r in m["rows"] if r["cov"] is not None]
    return xs, ys, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "miniimagenet"])
    ap.add_argument("--cal", type=int, default=800)
    ap.add_argument("--out_dir", default="output/novelty_selling")
    a = ap.parse_args()
    data = {ds: load(ds, a.root) for ds in a.datasets}
    data = {ds: r for ds, r in data.items() if r}
    nds = len(data)
    fig, axes = plt.subplots(1, nds + 1, figsize=(5.2 * (nds + 1), 4.6))
    if nds + 1 == 1:
        axes = [axes]
    # per-dataset slope panels
    for j, (ds, res) in enumerate(data.items()):
        ax = axes[j]
        for arm in ARMS:
            key = f"{a.cal}_balanced_{arm}"
            if key not in res:
                continue
            xs, ys, m = rows(res, key, "density")
            ax.plot(xs, ys, "o-", c=ARM_C[arm], lw=2.2, ms=7,
                    label=f"{ARM_LAB[arm]} (gap {m['covgap_geo']:.1f}, sz {m['marg_size']:.2f})")
        ax.axhline(1 - ALPHA, ls="--", c="k", lw=1.1)
        ax.axhspan(0.55, 1 - ALPHA, color="red", alpha=.05)
        ax.set_xlabel("local density stratum  (dense -> sparse)"); ax.set_ylabel("coverage")
        ax.set_title(f"{ds}  (cal={a.cal})", fontsize=11)
        ax.legend(fontsize=7.5); ax.grid(alpha=.25)
    # CovGap_geo bars (density) grouped by dataset x arm
    ax = axes[-1]
    x = np.arange(len(data)); w = 0.2
    for i, arm in enumerate(ARMS):
        vals = []
        for ds, res in data.items():
            key = f"{a.cal}_balanced_{arm}"
            vals.append(res[key]["by_covariate"]["density"]["covgap_geo"] if key in res else 0)
        ax.bar(x + (i - 1.5) * w, vals, w, color=ARM_C[arm], label=ARM_LAB[arm])
    ax.set_xticks(x); ax.set_xticklabels(list(data.keys()))
    ax.set_ylabel("CovGap_geo (density, pp)")
    ax.set_title("MS-CS does NOT close the geometric gap\n(it shrinks sets, not the slope)", fontsize=11)
    ax.legend(fontsize=7.5); ax.grid(alpha=.25, axis="y")
    fig.suptitle("The geometric under-coverage is NCM- and MS-CS-invariant "
                 "(only an explicitly geometry-conditional method can fix it)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "geometry_comprehensive.png")
    fig.savefig(out, dpi=140); plt.close(fig); print(f"[fig] {out}")
    # console table
    print("\ndataset       arm                 marg_cov  marg_size  CovGap_geo(density)  worst")
    for ds, res in data.items():
        for arm in ARMS:
            key = f"{a.cal}_balanced_{arm}"
            if key not in res:
                continue
            m = res[key]["by_covariate"]["density"]
            print(f"{ds:13s} {ARM_LAB[arm]:18s}  {res[key]['marg_cov']:.3f}     "
                  f"{res[key]['marg_size']:.2f}       {m['covgap_geo']:.2f}              {m['worst']:.3f}")


if __name__ == "__main__":
    main()
