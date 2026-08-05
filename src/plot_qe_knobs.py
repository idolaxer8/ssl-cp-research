"""
qe knob-sweep heatmaps: set size over k x alpha per (dataset, cal), with the
incumbent (k=10, alpha=3) and per-panel best marked, and the no-qe incumbent
as a reference contour value in the title.

Usage:
python src/plot_qe_knobs.py --data_dir output/pool_repr_menu/qe_knobs
"""
import os, json, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS, ALPHAS = [5, 10, 20, 50], [0, 1, 3, 5]   # alpha=0 = plain AQE control
                                             # (k=50/a=0 not run -> blank)
REF = {"cifar100": {200: 4.17, 400: 1.64, 800: 1.30},
       "cub200": {400: 3.65, 800: 1.73, 1600: 1.31},
       "aircraft": {200: 29.01, 400: 24.61, 800: 22.03}}
NCM = {"cifar100": "prototype", "cub200": "prototype", "aircraft": "geodesic"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="output/pool_repr_menu/qe_knobs")
    ap.add_argument("--clip_pct", type=float, default=35,
                    help="clip heat color range this %% above the panel best "
                         "(keeps degenerate cells from washing out the map).")
    args = ap.parse_args()

    datasets = list(REF)
    n_cals = 3
    fig, axes = plt.subplots(len(datasets), n_cals,
                             figsize=(3.6 * n_cals, 3.2 * len(datasets)))
    for i, ds in enumerate(datasets):
        grid = {}
        for k in KS:
            for a in ALPHAS:
                p = os.path.join(args.data_dir, f"results_{ds}_qek{k}a{a}.json")
                if not os.path.exists(p):
                    continue
                for r in json.load(open(p))["rows"]:
                    grid[(k, a, r["cal"])] = r["sz"]
        cals = sorted({c for (_, _, c) in grid})
        for j, cal in enumerate(cals):
            ax = axes[i][j]
            M = np.array([[grid.get((k, a, cal), np.nan) for a in ALPHAS]
                          for k in KS])
            best = np.nanmin(M)
            vmax = best * (1 + args.clip_pct / 100)
            im = ax.imshow(np.ma.masked_invalid(np.minimum(M, vmax)),
                           cmap="viridis_r", vmin=best, vmax=vmax,
                           aspect="auto")
            for r in range(len(KS)):
                for c in range(len(ALPHAS)):
                    v = M[r, c]
                    if np.isnan(v):
                        ax.text(c, r, "-", ha="center", va="center",
                                fontsize=8, color="gray")
                        continue
                    txt = f"{v:.2f}" if v < 10 else f"{v:.0f}"
                    bb, bc = None, ("w" if v > (best + vmax) / 2 else "k")
                    if (KS[r], ALPHAS[c]) == (10, 3):
                        bb = dict(boxstyle="round,pad=0.15", fc="none",
                                  ec="red", lw=1.2)
                    if v == best:
                        txt += "*"
                    ax.text(c, r, txt, ha="center", va="center",
                            fontsize=8, color=bc, bbox=bb)
            ax.set_xticks(range(len(ALPHAS)))
            ax.set_xticklabels([f"a={a}" for a in ALPHAS], fontsize=8)
            ax.set_yticks(range(len(KS)))
            ax.set_yticklabels([f"k={k}" for k in KS], fontsize=8)
            ref = REF[ds][cal]
            ax.set_title(f"{ds} ({NCM[ds]}) cal={cal}\nno-qe incumbent {ref:.2f}",
                         fontsize=9)
    fig.suptitle("qe knob sweep — mean set size over k x alpha "
                 "(red box = pre-registered (10,3); * = panel best; "
                 "color clipped near best)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(args.data_dir, "qe_knob_heatmaps.png")
    fig.savefig(p, dpi=150)
    print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
