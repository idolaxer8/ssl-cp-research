"""Pilot B (2-D multi-resolution purity) vs SINGLE-dimension methods,
CIFAR-100, both splits. Sources: cifar100_multiview (pair + its raw dims) and
cifar100_gt (raw geodesic baseline; GT-dosed arm for the random split).

Run: python src/plot_mdcp_pilotb_vs_1d.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:/Users/IDO/Desktop/Ido_student/Msc/ssl-cp-research/output/mdcp_pool_pilot"
CALS = [200, 400, 800]

MV = json.load(open(f"{BASE}/cifar100_multiview/mdcp_pool_pilot_results.json"))["results"]
GT = json.load(open(f"{BASE}/cifar100_gt/mdcp_pool_pilot_results.json"))["results"]

ARMS = [  # (source, arm, style) -- 1-D baselines dashed, pilot-B solid
    (GT, "raw1_geo", dict(color="#888888", ls="--", marker="D",
                          label="1-D geodesic asym (old default NCM)")),
    (MV, "raw1_proto_pca128_cw", dict(color="#444444", ls="--", marker="v",
     label="1-D prototype @ pca128 (previous champion pipeline)")),
    (MV, "raw1_proto_pca32_cw", dict(color="#bbbbbb", ls="--", marker="s",
     label="1-D prototype @ pca32 (coarse view alone)")),
    (MV, "dratio2_proto", dict(color="#d62728", ls="-", marker="o",
     label="pilot B: 2-D purity, proto128 x proto32 (deployable)")),
    (MV, "dratio2_proto_gt", dict(color="#ff7f0e", ls="-", marker="o",
     label="pilot B + GT corner aug (random-split medicine)")),
    (MV, "dratio2_oracle", dict(color="#2ca02c", ls=":", marker="d",
     label="pilot B, oracle pool labels (ceiling)")),
]

fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), sharex=True,
                         gridspec_kw=dict(height_ratios=[3, 1]))
for col, split in enumerate(("balanced_both", "random")):
    ax, axc = axes[0, col], axes[1, col]
    for src, arm, st in ARMS:
        try:
            sz = [src[f"{split}_cal{c}"][arm]["size"] for c in CALS]
            se = [src[f"{split}_cal{c}"][arm]["size_se"] for c in CALS]
            cv = [src[f"{split}_cal{c}"][arm]["cov"] for c in CALS]
            ce = [src[f"{split}_cal{c}"][arm]["cov_se"] for c in CALS]
        except KeyError:
            continue
        ax.errorbar(CALS, sz, yerr=se, capsize=3, **st)
        axc.errorbar(CALS, cv, yerr=ce, capsize=3, **{**st, "label": None})
    ax.set_yscale("log")
    ax.set_title(f"CIFAR-100, {split} split (20 trials)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    axc.axhline(0.9, color="k", lw=1, ls="--")
    axc.set_xlabel("calibration size", fontsize=11)
    axc.set_xticks(CALS)
    axc.grid(alpha=0.3)
axes[0, 0].set_ylabel("mean set size (log)", fontsize=12)
axes[1, 0].set_ylabel("coverage", fontsize=10)
axes[0, 0].legend(fontsize=8, loc="upper right")
fig.suptitle("Pilot B (2-D score-space purity) vs single-dimension methods",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p = f"{BASE}/figs/fig_pilotb_vs_1d_cifar100.png"
os.makedirs(os.path.dirname(p), exist_ok=True)
fig.savefig(p, dpi=150)
print("saved", p)
