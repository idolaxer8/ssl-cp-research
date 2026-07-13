"""Consolidated Aircraft combination comparison (balanced, 20 trials):
solo dimensions vs every gate-passing pair vs the screened 3-D.

Run: python src/plot_aircraft_combos.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:/Users/IDO/Desktop/Ido_student/Msc/ssl-cp-research/output/mdcp_pool_pilot"
CALS = [200, 400, 800]


def get(run, key, arm):
    d = json.load(open(f"{BASE}/{run}/mdcp_pool_pilot_results.json"))["results"]
    r = d[key][arm]
    return r["size"], r["size_se"], r["cov"]


ARMS = [  # (label, run, arm, color, hatch)
    ("geo@lw768 solo",            "aircraft_lw_proto2d", "raw1_geo_final__lw768",          "#888888", ""),
    ("geo@pca512 solo",           "air_pca512_proto",    "raw1_geo_final__pca512_cw",      "#bbbbbb", ""),
    ("proto@pca128 solo",         "aircraft_lw_proto2d", "raw1_proto_final__pca128_cw",    "#dddddd", ""),
    ("PAIR lw x proto (.61)",     "aircraft_lw_proto2d", "dratio2_proto",                  "#d62728", ""),
    ("PAIR pca512 x proto (.58)", "air_pca512_proto",    "dratio2_proto",                  "#ff7f0e", ""),
    ("PAIR lw x pca512 (.71)",    "air_lw_pca512",       "dratio2_proto",                  "#1f77b4", ""),
    ("3-D all three",             "air_3d_strong",       "dratio2_proto",                  "#2ca02c", "//"),
]

fig, ax = plt.subplots(figsize=(12, 6))
width = 0.11
xs = np.arange(len(CALS))
print(f"{'arm':<28}" + "".join(f"{c:>16}" for c in CALS))
for j, (label, run, arm, color, hatch) in enumerate(ARMS):
    vals, ses, covs = [], [], []
    for cal in CALS:
        try:
            s, se, cv = get(run, f"balanced_both_cal{cal}", arm)
        except (FileNotFoundError, KeyError):
            s, se, cv = np.nan, 0, np.nan
        vals.append(s); ses.append(se); covs.append(cv)
    ax.bar(xs + (j - len(ARMS) / 2) * width, vals, width, yerr=ses, capsize=2,
           label=label, color=color, hatch=hatch, edgecolor="black", lw=0.4)
    print(f"{label:<28}" + "".join(f"{v:>9.2f}({c:.2f})" for v, c in zip(vals, covs)))
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([f"cal={c}" for c in CALS], fontsize=12)
ax.set_ylabel("mean set size (log)", fontsize=12)
ax.set_title("Aircraft, balanced (20 trials): solo dims vs screened combinations\n"
             "(labels show anchor-A false-corr; all arms coverage-valid)",
             fontsize=12)
ax.grid(alpha=0.3, axis="y", which="both")
ax.legend(fontsize=9, ncol=2)
fig.tight_layout()
p = f"{BASE}/figs/fig_aircraft_combos.png"
os.makedirs(os.path.dirname(p), exist_ok=True)
fig.savefig(p, dpi=150)
print("saved", p)
