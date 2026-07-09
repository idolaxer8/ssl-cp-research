"""Summary figure for pilot D (full-CP MDCP), balanced split: full-CP arms
(cluster ladder, 20 trials x 150 test) vs pilot-B references (split-style 2-D
purity + raw 1-D prototype, 20-trial multiview grid). Local replication
points overlaid where available.

Run: python src/plot_mdcp_fullcp.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:/Users/IDO/Desktop/Ido_student/Msc/ssl-cp-research/output"
CLUSTER = f"{BASE}/from_cluster/mdcp_pool_pilot"
LOCAL = f"{BASE}/mdcp_pool_pilot"
OUT = f"{BASE}/mdcp_pool_pilot/figs"

CALS = [200, 400, 800]


def load(path, key):
    d = json.load(open(path))["results"][key]
    return d


def series(arm, src):
    sz, sze, cv, cve = [], [], [], []
    for cal in CALS:
        r = src(cal)[arm]
        sz.append(r["size"]); sze.append(r["size_se"])
        cv.append(r["cov"]); cve.append(r["cov_se"])
    return sz, sze, cv, cve


def main():
    cluster_src = lambda cal: load(
        f"{CLUSTER}/full_cp_bal{cal}/mdcp_full_cp_results.json",
        f"balanced_both_cal{cal}")
    pilotb = lambda cal: load(
        f"{LOCAL}/cifar100_multiview/mdcp_pool_pilot_results.json",
        f"balanced_both_cal{cal}")

    fig, (ax, axc) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    STYLES = [
        ("fcp_pool", cluster_src, dict(color="#d62728", marker="o", ls="-",
         label="FULL CP, ratio purity, pool-false (exact)")),
        ("fcp_bag", cluster_src, dict(color="#ff7f0e", marker="s", ls="-",
         label="FULL CP, ratio purity, bag-false (exact)")),
        ("fcp_count", cluster_src, dict(color="#8c564b", marker="d", ls=":",
         label="FULL CP, count purity (instructor-faithful, exact)")),
        ("dratio2_proto", pilotb, dict(color="#1f77b4", marker="o", ls="--",
         label="pilot B split-style 2-D purity (empirical validity)")),
        ("raw1_proto_pca128_cw", pilotb, dict(color="#888888", marker="v",
         ls="--", label="raw 1-D prototype (SCP)")),
    ]
    for arm, src, st in STYLES:
        sz, sze, cv, cve = series(arm, src)
        ax.errorbar(CALS, sz, yerr=sze, capsize=3, **st)
        axc.errorbar(CALS, cv, yerr=cve, capsize=3, **{**st, "label": None})

    # local 4GB replication points (bal-200/400) for the exact arms
    loc200 = load(f"{LOCAL}/full_cp_cal200/mdcp_full_cp_results.json",
                  "balanced_both_cal200")
    ax.scatter([200], [loc200["fcp_bag"]["size"]], marker="x", s=90,
               c="#ff7f0e", zorder=5, label="local-GPU replication (bag)")

    ax.set_yscale("log")
    ax.set_ylabel("mean set size (log)", fontsize=12)
    ax.set_title("Full-CP MDCP vs split-style pilot B -- CIFAR-100, balanced, "
                 "multi-resolution pair\n(cluster 20 trials x 150 test; "
                 "exact arms carry the 1-alpha guarantee)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)
    axc.axhline(0.9, color="k", lw=1, ls="--")
    axc.set_ylabel("coverage", fontsize=11)
    axc.set_xlabel("calibration size", fontsize=12)
    axc.grid(alpha=0.3)
    axc.set_xticks(CALS)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/fig_fullcp_balanced.png"
    fig.savefig(p, dpi=150)
    print("saved", p)


if __name__ == "__main__":
    main()
