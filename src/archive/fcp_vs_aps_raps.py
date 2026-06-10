"""
FCP (geodesic_topk_mean) vs SCP-APS vs SCP-RAPS baseline comparison.
No unlabeled data -- pure labeled-budget comparison.

Usage:
    python src/fcp_vs_aps_raps.py
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conformal_prediction import FullConformalPredictor, SemiCP, create_ncm

# Config
ALPHA = 0.1
SEED = 42
N_TRIALS = 5
CAL_SIZES = [300, 400, 600, 800, 1000]
TEST_SIZE = 500
SPLIT_TRAIN_RATIO = 0.5
NCM = "geodesic_topk_mean"
EMB_PATH = "output/embeddings_cifar100.pt"
TEST_PATH = "output/embeddings_cifar100_test.pt"
OUT_DIR = Path("output/fcp_vs_aps_raps")


def stratified_cal(X, y, all_classes, cal_size, rng):
    """Stratified sample of cal_size from (X, y) with >=1 per class."""
    n_classes = len(all_classes)
    first = np.array([rng.choice(np.where(y == c)[0], 1, replace=False)[0]
                      for c in all_classes])
    rest = np.setdiff1d(np.arange(len(X)), first)
    n_extra = cal_size - n_classes
    if n_extra > 0:
        extra = rng.choice(rest, min(n_extra, len(rest)), replace=False)
        idx = np.concatenate([first, extra])
    else:
        idx = first[:cal_size]
    idx = rng.permutation(idx)
    return X[idx], y[idx]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
    X_labeled = data["embeddings"].numpy()
    y_labeled = data["labels"].numpy()
    all_classes = np.unique(y_labeled)
    n_classes = len(all_classes)

    tdata = torch.load(TEST_PATH, map_location="cpu", weights_only=False)
    X_test_full = tdata["embeddings"].numpy()
    y_test_full = tdata["labels"].numpy()

    # Subsample test set to TEST_SIZE (stratified)
    rng_test = np.random.default_rng(SEED)
    test_per_class = TEST_SIZE // n_classes
    test_idx = []
    for c in all_classes:
        c_idx = np.where(y_test_full == c)[0]
        chosen = rng_test.choice(c_idx, min(test_per_class, len(c_idx)), replace=False)
        test_idx.append(chosen)
    test_idx = np.concatenate(test_idx)
    X_test = X_test_full[test_idx]
    y_test = y_test_full[test_idx]
    print(f"Labeled: {X_labeled.shape}, Test: {X_test.shape} (from {X_test_full.shape[0]}), Classes: {n_classes}")

    methods = ["FCP", "SCP-THR", "SCP-APS", "SCP-RAPS"]
    # {cal_size -> {method -> [trial_dicts]}}
    results = {cs: {m: [] for m in methods} for cs in CAL_SIZES}

    for cs in CAL_SIZES:
        t0 = time.time()
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(SEED + trial * 1000)
            X_cal, y_cal = stratified_cal(X_labeled, y_labeled, all_classes, cs, rng)

            # --- FCP: full budget as calibration ---
            # approved: legacy whitened-NCM experiment (cal-fit whitening, non-exchangeable)
            t1 = time.time()
            ncm = create_ncm(NCM, k=5, allow_nonexchangeable=True)
            cp = FullConformalPredictor(ncm, alpha=ALPHA)
            cp.calibrate(X_cal, y_cal, all_classes=all_classes)
            m = cp.evaluate(X_test, y_test, verbose=False)
            results[cs]["FCP"].append({
                "coverage": m["coverage"], "avg_set_size": m["avg_set_size"],
                "time": time.time() - t1})

            # --- SCP methods: split budget into train + cal ---
            n_train = int(SPLIT_TRAIN_RATIO * cs)
            X_tr, y_tr = X_cal[:n_train], y_cal[:n_train]
            X_cs, y_cs = X_cal[n_train:], y_cal[n_train:]

            for score_fn, name in [("THR", "SCP-THR"), ("APS", "SCP-APS"), ("RAPS", "SCP-RAPS")]:
                t1 = time.time()
                scp = SemiCP(alpha=ALPHA, score_fn=score_fn)
                scp.fit(X_tr, y_tr)
                scp.calibrate(X_cs, y_cs, all_classes=all_classes)
                m = scp.evaluate(X_test, y_test, verbose=False)
                results[cs][name].append({
                    "coverage": m["coverage"], "avg_set_size": m["avg_set_size"],
                    "time": time.time() - t1})

        elapsed = time.time() - t0
        print(f"\ncal={cs} ({elapsed:.1f}s, {N_TRIALS} trials)")
        for m_name in methods:
            vals = results[cs][m_name]
            covs = [v["coverage"] for v in vals]
            szs = [v["avg_set_size"] for v in vals]
            flag = "OK" if np.mean(covs) >= 0.89 else "!!"
            print(f"  {m_name:10s}  cov={np.mean(covs):.3f}+/-{np.std(covs):.3f} {flag}  "
                  f"sz={np.mean(szs):.2f}+/-{np.std(szs):.2f}")

    # --- Save JSON ---
    def jsonify(o):
        if isinstance(o, dict):
            return {str(k): jsonify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonify(i) for i in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(jsonify(results), f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'results.json'}")

    # --- Plot ---
    styles = {
        "FCP":      {"color": "#1f77b4", "ls": "-",  "marker": "o", "lw": 2.5},
        "SCP-THR":  {"color": "#2ca02c", "ls": ":",  "marker": "^", "lw": 1.5},
        "SCP-APS":  {"color": "#9467bd", "ls": "--", "marker": "v", "lw": 2.0},
        "SCP-RAPS": {"color": "#ff7f0e", "ls": "--", "marker": "D", "lw": 2.0},
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"CIFAR-100: FCP ({NCM}) vs APS/RAPS Baselines  (alpha={ALPHA})",
                 fontsize=13, fontweight="bold")

    for m_name in methods:
        s = styles[m_name]
        xs, mu_sz, sd_sz, mu_cov, sd_cov = [], [], [], [], []
        for cs in CAL_SIZES:
            vals = results[cs][m_name]
            covs = [v["coverage"] for v in vals]
            szs = [v["avg_set_size"] for v in vals]
            xs.append(cs)
            mu_cov.append(np.mean(covs)); sd_cov.append(np.std(covs))
            mu_sz.append(np.mean(szs)); sd_sz.append(np.std(szs))

        xs = np.array(xs)
        mu_sz = np.array(mu_sz); sd_sz = np.array(sd_sz)
        mu_cov = np.array(mu_cov); sd_cov = np.array(sd_cov)

        label = f"{m_name} (n_cal={'{cal}' if m_name == 'FCP' else '{cal/2}'})"
        if m_name == "FCP":
            label = f"FCP ({NCM}, n_cal=budget)"
        else:
            label = f"{m_name} (n_train=n_cal=budget/2)"

        axes[0].plot(xs, mu_sz, color=s["color"], ls=s["ls"], marker=s["marker"],
                     lw=s["lw"], ms=6, label=label)
        axes[0].fill_between(xs, mu_sz - sd_sz, mu_sz + sd_sz, alpha=0.12, color=s["color"])

        axes[1].plot(xs, mu_cov * 100, color=s["color"], ls=s["ls"], marker=s["marker"],
                     lw=s["lw"], ms=6, label=label)
        axes[1].fill_between(xs, (mu_cov - sd_cov) * 100, (mu_cov + sd_cov) * 100,
                             alpha=0.12, color=s["color"])

    axes[1].axhline(90, color="black", ls="--", lw=1.2, label="Target 90%")

    axes[0].set_ylabel("Avg Prediction Set Size", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Coverage (%)", fontsize=11)
    axes[1].set_xlabel("Labeled Budget", fontsize=11)
    axes[1].set_ylim(70, 102)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = OUT_DIR / "fcp_vs_aps_raps.png"
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")

    # --- Text summary ---
    lines = [f"CIFAR-100: FCP ({NCM}) vs APS/RAPS  (alpha={ALPHA}, {N_TRIALS} trials)",
             "=" * 70,
             f"{'Cal':>6}  {'Method':>10}  {'Coverage':>18}  {'Set Size':>18}",
             "-" * 60]
    for cs in CAL_SIZES:
        first = True
        for m_name in methods:
            vals = results[cs][m_name]
            covs = [v["coverage"] for v in vals]
            szs = [v["avg_set_size"] for v in vals]
            prefix = f"{cs:>6}" if first else " " * 6
            lines.append(f"{prefix}  {m_name:>10}  "
                         f"{np.mean(covs)*100:5.1f}%+/-{np.std(covs)*100:4.1f}%  "
                         f"{np.mean(szs):7.2f}+/-{np.std(szs):5.2f}")
            first = False
        lines.append("-" * 60)

    txt = "\n".join(lines) + "\n"
    txt_path = OUT_DIR / "summary.txt"
    Path(txt_path).write_text(txt)
    print(f"Saved: {txt_path}")
    print("\n" + txt)


if __name__ == "__main__":
    main()
