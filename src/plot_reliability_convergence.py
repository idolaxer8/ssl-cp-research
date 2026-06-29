"""
Reliability CONVERGENCE figure: coverage-SD vs label budget B over an extended
range, showing (i) Full CP riding the Beta(n=B) floor, (ii) Split CP-THR's
degenerate small-cal region (sets = K) and where it "turns on", and (iii) where
its coverage SD converges to the Beta(n=B/2) floor.

Reads output/reliability/<ds>/reliability_cal<c>.json for the cal sizes present.
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.1
C_FULL = "#1b7837"; C_GEO = "#5aae61"; C_SCP = "#c2334d"


def beta_sd(n):
    return np.sqrt(ALPHA * (1 - ALPHA) / (n + 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--root", default="output")
    ap.add_argument("--cals", type=int, nargs="+",
                    default=[200, 400, 800, 1200, 1600, 2400, 3200])
    ap.add_argument("--out_dir", default="output/novelty_selling")
    a = ap.parse_args()
    cals, sd_p, sd_g, sd_s, fk_s, sz_s = [], [], [], [], [], []
    for c in a.cals:
        p = os.path.join(a.root, "reliability", a.dataset, f"reliability_cal{c}.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))["summary"]
        cals.append(c)
        sd_p.append(s["full_proto_bal"]["coverage"]["sd"])
        sd_g.append(s["full_geo_bal"]["coverage"]["sd"])
        sd_s.append(s["scp_thr_bal"]["coverage"]["sd"])
        fk_s.append(s["scp_thr_bal"]["frac_K"]["mean"])
        sz_s.append(s["scp_thr_bal"]["mean_size"]["mean"])
    cals = np.array(cals)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # ---- panel 0: coverage SD vs B, with both Beta floors ----
    grid = np.linspace(cals.min(), cals.max(), 100)
    ax[0].plot(grid, [beta_sd(n) for n in grid], "--", c=C_FULL, lw=1.4, alpha=.8,
               label="Beta(n=B) floor  [Full CP: all B -> rank]")
    ax[0].plot(grid, [beta_sd(n / 2) for n in grid], "--", c=C_SCP, lw=1.4, alpha=.8,
               label="Beta(n=B/2) floor [Split: half on head]")
    ax[0].plot(cals, sd_p, "o-", c=C_FULL, lw=2.4, ms=8, label="Full CP (prototype)")
    ax[0].plot(cals, sd_g, "^-", c=C_GEO, lw=1.8, ms=7, label="Full CP (geodesic)")
    # SCP: split degenerate (frac_K>=0.3) vs usable
    deg = np.array(fk_s) >= 0.3
    ax[0].plot(cals[~deg], np.array(sd_s)[~deg], "s-", c=C_SCP, lw=2.4, ms=8,
               label="Split CP-THR (usable)")
    if deg.any():
        ax[0].scatter(cals[deg], np.array(sd_s)[deg], marker="x", c=C_SCP, s=80, zorder=5)
        xmax = cals[deg].max()
        ax[0].axvspan(cals.min() - 30, xmax + (cals[~deg].min() - xmax) / 2 if (~deg).any() else xmax,
                      color=C_SCP, alpha=.06)
        ax[0].text(cals[deg][0], max(sd_s) * 0.92,
                   "Split CP degenerate\n(sets = K; SD meaningless)", fontsize=8, c=C_SCP)
    ax[0].set_xscale("log"); ax[0].set_xticks(cals); ax[0].set_xticklabels(cals)
    ax[0].set_xlabel("label budget B (log)"); ax[0].set_ylabel("coverage SD over draws")
    ax[0].set_ylim(bottom=0)
    ax[0].set_title(f"{a.dataset}: Full CP rides the all-B reliability floor;\n"
                    "Split CP only 'turns on' at larger B and rides the B/2 floor", fontsize=11)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, which="both")

    # ---- panel 1: Split CP mean set size vs B (where it becomes usable) ----
    ax[1].plot(cals, sz_s, "s-", c=C_SCP, lw=2.4, ms=8, label="Split CP-THR mean set size")
    ax[1].axhline(100, ls=":", c="grey"); ax[1].text(cals[0], 104, "K=100", fontsize=8, c="grey")
    for c, fk, z in zip(cals, fk_s, sz_s):
        if fk >= 0.3:
            ax[1].annotate(f"{fk*100:.0f}% =K", (c, z), fontsize=7, c=C_SCP,
                           xytext=(0, 6), textcoords="offset points", ha="center")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xticks(cals)
    ax[1].set_xticklabels(cals); ax[1].set_xlabel("label budget B (log)")
    ax[1].set_ylabel("Split CP mean set size (log)")
    ax[1].set_title("Split CP's sets collapse to K at few-shot,\nonly shrinking once B/2 trains a usable head", fontsize=11)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.25, which="both")

    fig.suptitle("Where does Split CP 'converge'? Only after the budget can afford a usable inner classifier",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"reliability_convergence_{a.dataset}.png")
    fig.savefig(out, dpi=140); plt.close(fig); print(f"[fig] {out}")
    # console table
    print("\nB     SD_full_proto  SD_full_geo  SD_scp   Beta(B)  Beta(B/2)  SCP_frac_K  SCP_size")
    for i, c in enumerate(cals):
        print(f"{c:5d}  {sd_p[i]:.4f}        {sd_g[i]:.4f}      {sd_s[i]:.4f}  "
              f"{beta_sd(c):.4f}   {beta_sd(c/2):.4f}    {fk_s[i]:.2f}        {sz_s[i]:.2f}")


if __name__ == "__main__":
    main()
