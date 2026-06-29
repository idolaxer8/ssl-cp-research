"""
Robustness "selling" figure for geometry-conditional CP: across alpha / G /
covariate, global FCP's geometric CovGap varies widely (5-20pp) while
Mondrian-geometry flattens it to <~2pp. Reads the sweep JSONs.
"""
import os, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def label_from(cfg, fname):
    a = cfg["alpha"]; G = cfg["n_strata"]; cov = cfg["covariate"]
    return f"a{a}\nG{G}\n{cov[:3]}"


def collect(ddir, split="balanced"):
    rows = []
    for p in sorted(glob.glob(os.path.join(ddir, "stage0_*.json"))):
        d = json.load(open(p)); cfg = d["config"]; res = d["results"]
        if split not in res or "mondrian_geo" not in res[split]:
            continue
        g = res[split]["global_fcp"]; m = res[split]["mondrian_geo"]
        rows.append(dict(label=label_from(cfg, p),
                         a=cfg["alpha"], G=cfg["n_strata"], cov=cfg["covariate"],
                         g_gap=g["covgap_geo"], m_gap=m["covgap_geo"],
                         g_worst=g["worst"], m_worst=m["worst"],
                         g_sz=g["marg_size"], m_sz=m["marg_size"], m_marg=m["marg_cov"]))
    # order: alpha sweep (G5,density), G sweep (a0.1,density), covariate
    def key(r):
        return (0 if r["cov"] == "density" else 1, r["G"], r["a"])
    return sorted(rows, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/geometric_conditional_sweep")
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "miniimagenet"])
    ap.add_argument("--out", default="output/novelty_selling/geometric_robustness.png")
    a = ap.parse_args()
    data = {ds: collect(os.path.join(a.root, ds)) for ds in a.datasets}
    data = {ds: r for ds, r in data.items() if r}
    nds = len(data)
    fig, axes = plt.subplots(nds, 2, figsize=(13, 4.4 * nds), squeeze=False)
    for r, (ds, rows) in enumerate(data.items()):
        labels = [x["label"] for x in rows]
        xs = np.arange(len(rows)); w = 0.38
        # CovGap_geo before/after
        ax = axes[r][0]
        ax.bar(xs - w / 2, [x["g_gap"] for x in rows], w, color="#c2334d", label="global FCP")
        ax.bar(xs + w / 2, [x["m_gap"] for x in rows], w, color="#1b7837", label="Mondrian-geo")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("CovGap_geo (pp)"); ax.set_title(f"{ds}: geometric CovGap")
        ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
        # worst-stratum before/after with target line
        ax = axes[r][1]
        ax.bar(xs - w / 2, [x["g_worst"] for x in rows], w, color="#c2334d", label="global FCP")
        ax.bar(xs + w / 2, [x["m_worst"] for x in rows], w, color="#1b7837", label="Mondrian-geo")
        for i, x in enumerate(rows):
            ax.plot([i - 0.5, i + 0.5], [1 - x["a"]] * 2, "k--", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("worst-stratum coverage"); ax.set_ylim(0, 1.0)
        ax.set_title(f"{ds}: worst-stratum (dashed=1-alpha)")
        ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
    fig.suptitle("Mondrian-geometry flattens the geometric gap across alpha / G / covariate "
                 "(worst-stratum -> ~1-alpha everywhere)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140); plt.close(fig); print(f"[fig] {a.out}")


if __name__ == "__main__":
    main()
