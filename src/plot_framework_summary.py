"""One-figure summary of the pool-selects-the-pipeline idea (Direction 2).

Left: the decision flow — unlabeled pool -> spectrum diagnostics -> transform
family (+ the T-sweep addendum: the same regime flag governs the softmax
temperature). Right: the evidence — regret of the label-free framework pick vs
accuracy-based selection, per dataset ordered by PR (geodesic NCM ground
truth, temperature-free). High-level by design; in-development pieces (mid-PR
d' ranking) are marked open, not detailed.

Output -> <main repo>/output/transform_selection_pilot/fig_framework_summary.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

MAIN = r"C:\Users\IDO\Desktop\Ido_student\Msc\ssl-cp-research"

fig = plt.figure(figsize=(13.2, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.14)

# ---------------------------------------------------------------- left: schema
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")


def box(x, y, w, h, text, fc, fontsize=9, ec="0.3"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3)


def arrow(x0, y0, x1, y1, label=None, dx=0.15):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, color="0.25", lw=1.3,
                                 zorder=1))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, label, fontsize=8,
                ha="left", va="center", color="0.25")


box(2.6, 8.7, 4.8, 1.05,
    "UNLABELED POOL\n(zero labels spent)", "#eaf2f8", fontsize=9.5)
box(2.6, 6.7, 4.8, 1.15,
    "pool covariance spectrum\nPR = effective # of variance directions\n"
    "(+ top-band share)", "#fef9e7")
arrow(5.0, 8.7, 5.0, 7.95)
arrow(4.2, 6.6, 2.6, 5.55, "PR large\n(spread spectrum)", dx=-2.5)
arrow(5.8, 6.6, 7.4, 5.55, "PR small\n(collapsed)", dx=0.1)
box(0.3, 4.0, 4.5, 1.55,
    "TRUNCATE: PCA-d' + cluster whiten\nd' ranked label-free by the\n"
    "margin near-tie statistic (0% regret)", "#d5f5e3")
box(5.3, 4.0, 4.5, 1.55,
    "DON'T TRUNCATE: full-rank\nLedoit-Wolf cluster whitening\n"
    "(signal hides in the variance tail)", "#fdebd0")
arrow(2.55, 3.95, 4.2, 3.0)
arrow(7.55, 3.95, 5.9, 3.0)
box(1.6, 1.8, 6.8, 1.1,
    "Full CP with the selected metric — coverage is EXACT either way:\n"
    "selection never touches cal/test labels ⇒ exchangeability untouched",
    "#eaecee", fontsize=9)
ax.text(5.0, 0.95,
        "T-sweep addendum (parallel session): the SAME regime flag governs the\n"
        "prototype-softmax temperature — auto-fit T under-fits exactly on "
        "collapsed spectra;\nuniversal T ≈ 0.06 fixes it (aircraft proto "
        "17.3 → 13.2). One pool diagnostic, two knobs.",
        ha="center", va="center", fontsize=8.3, style="italic", color="0.25",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f8f4fc", ec="#8e44ad",
                  lw=0.9))
ax.set_title("The idea: the unlabeled pool configures the CP pipeline\n"
             "before a single label is spent", fontsize=11.5)

# ------------------------------------------------------------- right: evidence
ax2 = fig.add_subplot(gs[0, 1])
ds = ["miniImageNet\nPR 255", "CIFAR-100\nPR 243", "CIFAR-10\nPR 119",
      "CUB-200\nPR 58", "Aircraft\nPR 16"]
frame = [0.0, 0.0, 0.9, 15.5, 0.0]        # framework pick regret, %
acc = [0.0, 0.0, 0.9, np.nan, 89.4]       # accuracy-selection regret, %
x = np.arange(len(ds))
w = 0.36
b1 = ax2.bar(x - w / 2, frame, w, color="#27ae60", alpha=0.9,
             label="this framework (zero labels)")
b1[3].set_hatch("//")
b1[3].set_alpha(0.45)
b2 = ax2.bar(x + w / 2, acc, w, color="#c0392b", alpha=0.9,
             label="accuracy-based selection (needs labels)")
for i, v in enumerate(frame):
    ax2.text(x[i] - w / 2, v + 1.5,
             "open" if i == 3 else f"{v:.0f}%", ha="center", fontsize=9,
             color="0.2")
for i, v in enumerate(acc):
    if np.isfinite(v):
        ax2.text(x[i] + w / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9,
                 color="0.2")
ax2.set_xticks(x)
ax2.set_xticklabels(ds, fontsize=9)
ax2.set_ylabel("set-size regret of the selected transform\n"
               "(% above the best possible arm, geodesic NCM)")
ax2.set_ylim(0, 100)
ax2.grid(axis="y", alpha=0.25)
ax2.legend(fontsize=9, loc="upper left", framealpha=0.95)
ax2.set_title("The evidence: near-optimal picks with zero labels;\n"
              "accuracy-based selection fails where it matters most",
              fontsize=11)
ax2.annotate("mid-PR ranking\nstill in development",
             xy=(3 - w / 2, 15.5), xytext=(2.15, 42), fontsize=8.5,
             ha="center", color="0.35",
             arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9))

out = os.path.join(MAIN, "output", "transform_selection_pilot",
                   "fig_framework_summary.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
print("saved ->", out)
