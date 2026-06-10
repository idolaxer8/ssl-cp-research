"""
MS-CS Exchangeability Experiment — exact (fully exchangeable) vs O(1/n) (frozen).

Settles open question #2 in docs/theory.md: for the lead pipeline
  DINOv2 -> PCA-128 -> whitened geodesic_topk_mean -> FCP + MS-CS penalty
is there a *measurable* difference between

  (A) MS-CS frozen   — M and y_hat fixed at their calibration-only values
                       (default code path; O(1/n) non-exchangeable approximation),
  (B) MS-CS exact    — M and y_hat recomputed for each augmented bag
                       (run_fcp_with_mscs(exchangeable=True); the bag-symmetric
                       penalty that makes Theorem 1 hold *exactly*).

If (A) ≈ (B) at n = 2K..8K we can state Theorem 1 as effectively exact for the
headline pipeline; if they diverge we quantify the slack. The interesting axes:

  - marginal coverage          (the core validity question)
  - CovGap / worst-class / frac-undercovered   (Ding et al. 2023, class-conditional)
  - mean prediction-set size   (efficiency cost of the penalty path)
  - wall-clock runtime         (cost of exactness)
  - head-to-head set agreement (identical-set %, mean |Δsize|, Δcoverage, symdiff)

The two MS-CS modes run on the SAME backend (CPU / vectorized NumPy) so the
runtime ratio is a fair "cost of exactness". The repo's GPU fast-path
(FullConformalPredictor.predict(device='cuda')) is plain FCP with NO penalty, so
it does not apply to the MS-CS penalty path; --device only accelerates the
optional no-penalty FCP+PCA anchor. CovGap/coverage/size are device-independent.

Reuses per_class_metrics / aggregate_method from conditional_coverage_experiment
(identical CovGap definition to findings §2b) and dataset/split helpers from
semicp_experiment. The exchangeable inner loop in run_fcp_with_mscs was
vectorized (bit-identical) so this is tractable on CPU; verify_vectorization()
checks that transformation at startup.

Usage
-----
Cluster (full, CIFAR-100 is the discriminating benchmark, ~1-1.5 h):
    python src/mscs_exchangeability_experiment.py \
        --datasets cifar100 --cal_sizes 200 400 600 800 --n_trials 5 --test_size 2000

Add miniImageNet:
    python src/mscs_exchangeability_experiment.py --datasets cifar100 miniimagenet ...

Laptop smoke test (synthetic data, no .pt files needed, ~30 s):
    python src/mscs_exchangeability_experiment.py --smoke

Run from a different checkout (point at an existing output/ dir):
    python src/mscs_exchangeability_experiment.py --embeddings_root /path/to/repo
"""
# ARCHIVE-CANDIDATE (review 2026-06-24): question settled (findings sec 4b: exact vs frozen penalty ~identical).
# If unused by the review date, move to src/archive/ -- watch list: src/archive/README.md.


import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_prediction import FullConformalPredictor, create_ncm
from mscs_unlabeled_experiment import (
    build_cluster_similarity_matrix,
    run_fcp_with_mscs,
)
from semicp_experiment import (
    DATASETS,
    balanced_sample,
    stratified_split,
    _jsonify,
)
# Identical CovGap definition to findings §2b (single source of truth).
from conditional_coverage_experiment import per_class_metrics, aggregate_method


# ---------------------------------------------------------------------------
# Defaults (matched to findings §1/§4 lead pipeline)
# ---------------------------------------------------------------------------
ALPHA = 0.1
SEED = 42
NCM_FCP = "geodesic_topk_mean"
MSCS_LAMBDA = 0.05
MSCS_N_CLUSTERS = 20
MSCS_TAU_MULT = 0.5

MODE_FROZEN = "MS-CS frozen (O(1/n))"
MODE_EXACT = "MS-CS exact (exchangeable)"
MODE_ANCHOR = "FCP+PCA (no penalty)"

MODE_STYLE = {
    MODE_FROZEN: {"color": "#1f77b4", "ls": "-",  "marker": "o"},
    MODE_EXACT:  {"color": "#d62728", "ls": "--", "marker": "s"},
    MODE_ANCHOR: {"color": "#7f7f7f", "ls": ":",  "marker": "^"},
}


# ---------------------------------------------------------------------------
# Correctness gate for the vectorized exchangeable inner loop
# ---------------------------------------------------------------------------

def verify_vectorization(seed=0, n_cal=200, K=30, reps=20, tol=0.0):
    """Assert the vectorized cal-penalty gather equals the original per-j loop.

    Mirrors the transformation applied to run_fcp_with_mscs's exchangeable path:
        loop:  cal_penalty[j] = lam*(1 - M[cal_class_idx[j], yc if dists[j]<loo[j]
                                                              else yhat_loo_idx[j]])
        vec :  yhat = where(dists<loo, yc, yhat_loo_idx);  lam*(1 - M[cal_class_idx, yhat])
    Raises AssertionError if they ever differ. Pure-numpy, instant.
    """
    rng = np.random.default_rng(seed)
    max_abs = 0.0
    for _ in range(reps):
        M = rng.random((K, K))
        lam = float(rng.uniform(0.01, 0.2))
        yc_idx = int(rng.integers(0, K))
        cal_class_idx = rng.integers(0, K, size=n_cal)
        yhat_loo_idx = rng.integers(0, K, size=n_cal)
        dists = rng.random(n_cal)
        loo = rng.random(n_cal)

        # original per-j loop
        ref = np.empty(n_cal)
        for j in range(n_cal):
            yhat_j = yc_idx if dists[j] < loo[j] else yhat_loo_idx[j]
            ref[j] = lam * (1.0 - M[cal_class_idx[j], yhat_j])

        # vectorized
        yhat = np.where(dists < loo, yc_idx, yhat_loo_idx)
        vec = lam * (1.0 - M[cal_class_idx, yhat])

        max_abs = max(max_abs, float(np.abs(ref - vec).max()))
    assert max_abs <= tol, f"vectorization mismatch: max|Δ|={max_abs}"
    return max_abs


# ---------------------------------------------------------------------------
# Head-to-head set agreement between two prediction-set lists
# ---------------------------------------------------------------------------

def set_agreement(sets_frozen, sets_exact, y_test):
    """Per-test-point agreement stats between the frozen and exact prediction sets."""
    n = len(sets_frozen)
    y_test = np.asarray(y_test)
    a = [set(s) for s in sets_frozen]
    b = [set(s) for s in sets_exact]
    identical = np.mean([a[i] == b[i] for i in range(n)])
    abs_size_delta = np.mean([abs(len(a[i]) - len(b[i])) for i in range(n)])
    signed_size_delta = np.mean([len(b[i]) - len(a[i]) for i in range(n)])  # exact - frozen
    symdiff = np.mean([len(a[i] ^ b[i]) for i in range(n)])
    cov_frozen = np.mean([int(y_test[i]) in a[i] for i in range(n)])
    cov_exact = np.mean([int(y_test[i]) in b[i] for i in range(n)])
    # disagreement that flips the true-label decision (the validity-relevant part)
    flips_true = np.mean([(int(y_test[i]) in a[i]) != (int(y_test[i]) in b[i])
                          for i in range(n)])
    return {
        "identical_set_frac": float(identical),
        "mean_abs_size_delta": float(abs_size_delta),
        "mean_signed_size_delta": float(signed_size_delta),  # exact - frozen
        "mean_symdiff": float(symdiff),
        "coverage_frozen": float(cov_frozen),
        "coverage_exact": float(cov_exact),
        "coverage_delta": float(cov_exact - cov_frozen),  # exact - frozen
        "frac_truelabel_flips": float(flips_true),
    }


def aggregate_agreement(trials):
    """Mean+std over trials of each agreement scalar."""
    if not trials:
        return {}
    keys = trials[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([t[k] for t in trials], dtype=float)
        agg[f"{k}_mean"] = float(np.nanmean(vals))
        agg[f"{k}_std"] = float(np.nanstd(vals))
    return agg


# ---------------------------------------------------------------------------
# Per-trial: build M in PCA space, run both modes, collect metrics + agreement
# ---------------------------------------------------------------------------

def run_modes(X_cal_pca, y_cal, X_test_pca, y_test, X_unlabeled_pca, all_classes,
              alpha, ncm, lam, n_clusters, tau_mult, do_exact=True,
              do_anchor=True, anchor_device="cpu"):
    """Run frozen + exact MS-CS (and optional no-penalty anchor) on one trial.

    All inputs are already PCA-projected (MS-CS clusters live in PCA space,
    findings §4). Returns ({mode: metrics_dict}, {"frozen": sets, "exact": sets}).
    """
    out = {}
    sets = {}

    # Build similarity matrix M (+ artifacts for the exact path) from the
    # unlabeled pool and the current calibration set, in PCA space.
    (M, c2c, eff_tau, med_d2, cls_centroids, cls_counts,
     clust_centroids, clust_dists) = build_cluster_similarity_matrix(
        X_unlabeled_pca, X_cal_pca, y_cal, all_classes, n_clusters,
        tau=-tau_mult)  # negative tau => effective_tau = tau_mult * median_d^2

    # (A) Frozen / O(1/n): M and y_hat fixed at cal-only values.
    # approved: this branch deliberately studies the non-exchangeable mode
    # (and the whitened NCM is itself cal-fit non-exchangeable).
    t0 = time.time()
    _, _, sets_frozen = run_fcp_with_mscs(
        X_cal_pca, y_cal, X_test_pca, y_test, all_classes, ncm, alpha, lam, M,
        exchangeable=False, return_sets=True, allow_nonexchangeable=True)
    t_frozen = time.time() - t0
    m = per_class_metrics(sets_frozen, y_test, all_classes, alpha)
    m["time"] = t_frozen
    out[MODE_FROZEN] = m
    sets["frozen"] = sets_frozen

    # (B) Exact / exchangeable: M and y_hat recomputed per augmented bag.
    if do_exact:
        t0 = time.time()
        # MS-CS ŷ/M are bag-symmetric here; the cal-fit whitened NCM is the only
        # remaining O(1/n) break, approved below.
        _, _, sets_exact = run_fcp_with_mscs(
            X_cal_pca, y_cal, X_test_pca, y_test, all_classes, ncm, alpha, lam, M,
            exchangeable=True, return_sets=True,
            class_centroids=cls_centroids, class_counts=cls_counts,
            class_to_cluster=c2c, cluster_centroids=clust_centroids,
            cluster_dists=clust_dists, effective_tau=eff_tau,
            allow_nonexchangeable=True)
        t_exact = time.time() - t0
        m = per_class_metrics(sets_exact, y_test, all_classes, alpha)
        m["time"] = t_exact
        out[MODE_EXACT] = m
        sets["exact"] = sets_exact

    # Anchor: plain FCP+PCA, no penalty (context; GPU-accelerable).
    if do_anchor:
        t0 = time.time()
        ncm_obj = create_ncm(ncm, k=5, allow_nonexchangeable=True)
        cp = FullConformalPredictor(ncm_obj, alpha=alpha)
        cp.calibrate(X_cal_pca, y_cal, all_classes=all_classes)
        res = cp.predict(X_test_pca, verbose=False, device=anchor_device)
        sets_anchor = res["prediction_sets"]
        t_anchor = time.time() - t0
        m = per_class_metrics(sets_anchor, y_test, all_classes, alpha)
        m["time"] = t_anchor
        out[MODE_ANCHOR] = m

    return out, sets


# ---------------------------------------------------------------------------
# Dataset preparation (PCA on unlabeled pool; external-test or carve fallback)
# Mirrors conditional_coverage_experiment.main() so results stay comparable.
# ---------------------------------------------------------------------------

def _resolve(path, root):
    if path is None:
        return None
    return os.path.join(root, path) if root else path


def prepare_dataset(ds_name, cal_sizes, test_size, pca_dim_override, seed, root=""):
    """Load embeddings, carve/locate unlabeled pool, fit PCA, project test+pool.

    Returns dict or None if the labeled file is missing.
    """
    cfg = DATASETS[ds_name]
    labeled_path = _resolve(cfg["labeled"], root)
    if not Path(labeled_path).exists():
        print(f"[SKIP] {ds_name}: {labeled_path} not found")
        return None

    import torch
    data = torch.load(labeled_path, map_location="cpu", weights_only=False)
    X_labeled = data["embeddings"].numpy()
    y_labeled = data["labels"].numpy()
    all_classes = np.unique(y_labeled)
    n_classes = len(all_classes)
    print(f"  Labeled: {X_labeled.shape}, {n_classes} classes")

    # External test set (stratified subsample), if configured & present.
    X_test_ext = y_test_ext = None
    test_path = _resolve(cfg.get("test"), root)
    if test_path and Path(test_path).exists():
        tdata = torch.load(test_path, map_location="cpu", weights_only=False)
        Xtf = tdata["embeddings"].numpy()
        ytf = tdata["labels"].numpy()
        rng_t = np.random.default_rng(seed)
        per_class = max(1, test_size // n_classes)
        idx = []
        for c in all_classes:
            c_idx = np.where(ytf == c)[0]
            idx.append(rng_t.choice(c_idx, min(per_class, len(c_idx)), replace=False))
        idx = np.concatenate(idx)
        X_test_ext, y_test_ext = Xtf[idx], ytf[idx]
        print(f"  External test: {X_test_ext.shape} ({per_class}/class)")

    # External or carved unlabeled pool (disjoint from cal/test).
    X_unlabeled = None
    X_lab_rem, y_lab_rem = X_labeled, y_labeled
    unlabeled_path = _resolve(cfg.get("unlabeled"), root)
    if unlabeled_path and Path(unlabeled_path).exists():
        udata = torch.load(unlabeled_path, map_location="cpu", weights_only=False)
        X_unlabeled = udata["embeddings"].numpy()
        print(f"  External unlabeled: {X_unlabeled.shape}")
    else:
        if cfg.get("unlabeled") and not (unlabeled_path and Path(unlabeled_path).exists()):
            print(f"  [INFO] {ds_name}: unlabeled file missing, carving from labeled.")
        if cfg.get("unlabeled_carve_per_class"):
            carve_per_class = cfg["unlabeled_carve_per_class"]
        else:
            max_cal = max(cal_sizes)
            need = max(1, max_cal // n_classes) + max(1, test_size // n_classes) + 5
            avail = int(min(np.bincount(y_labeled.astype(int))))
            carve_per_class = max(5, min(50, avail - need))
            print(f"  [INFO] auto-carve {carve_per_class}/class "
                  f"(avail {avail}, reserved {need})")
        rng_c = np.random.default_rng(seed)
        carve_idx, remain_idx = [], []
        for c in all_classes:
            c_idx = rng_c.permutation(np.where(y_labeled == c)[0])
            n_carve = min(carve_per_class, len(c_idx) - 5)
            carve_idx.extend(c_idx[:n_carve])
            remain_idx.extend(c_idx[n_carve:])
        carve_idx, remain_idx = np.array(carve_idx), np.array(remain_idx)
        X_unlabeled = X_labeled[carve_idx]
        X_lab_rem, y_lab_rem = X_labeled[remain_idx], y_labeled[remain_idx]
        print(f"  Carved unlabeled: {X_unlabeled.shape}, remaining: {X_lab_rem.shape}")

    # Fit PCA on the unlabeled pool only (Prop. 2: unsupervised wrt cal/test).
    pca_dim = pca_dim_override or cfg.get("pca_dim", 128)
    pca_dim = min(pca_dim, X_unlabeled.shape[1], X_unlabeled.shape[0])
    pca = PCA(n_components=pca_dim)
    pca.fit(X_unlabeled)
    X_unlabeled_pca = pca.transform(X_unlabeled)
    X_test_pca_ext = pca.transform(X_test_ext) if X_test_ext is not None else None
    print(f"  PCA: {X_unlabeled.shape[1]}d -> {pca_dim}d "
          f"(EV {pca.explained_variance_ratio_.sum():.3f})")

    return {
        "all_classes": all_classes, "n_classes": n_classes,
        "X_lab_rem": X_lab_rem, "y_lab_rem": y_lab_rem,
        "X_test_ext": X_test_ext, "y_test_ext": y_test_ext,
        "X_test_pca_ext": X_test_pca_ext,
        "X_unlabeled_pca": X_unlabeled_pca, "pca": pca, "pca_dim": pca_dim,
    }


def make_synthetic(seed=0, n_classes=20, n_per_class=60, d=64, sep=2.5):
    """Separable Gaussian blobs for the --smoke path (no .pt files needed)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=sep, size=(n_classes, d))
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.normal(scale=1.0, size=(n_per_class, d)))
        y.append(np.full(n_per_class, c))
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dataset(by_cal, agree_by_cal, out_path, ds_name, alpha, modes):
    """2x3 panel: coverage, set size, CovGap, worst-class, runtime, agreement."""
    cals = sorted(by_cal.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    target = 1 - alpha

    def series(metric):
        return {m: ([cs for cs in cals if by_cal[cs].get(m, {}).get("n_trials_valid", 0)],
                    [by_cal[cs][m][f"{metric}_mean"] for cs in cals
                     if by_cal[cs].get(m, {}).get("n_trials_valid", 0)],
                    [by_cal[cs][m][f"{metric}_std"] for cs in cals
                     if by_cal[cs].get(m, {}).get("n_trials_valid", 0)])
                for m in modes}

    def draw(ax, metric, title, ylabel, hline=None, logy=False):
        for m in modes:
            xs, ys, es = series(metric)[m]
            if not xs:
                continue
            st = MODE_STYLE.get(m, {})
            ax.errorbar(xs, ys, yerr=es, label=m, capsize=3,
                        color=st.get("color"), ls=st.get("ls", "-"),
                        marker=st.get("marker", "o"), lw=1.8)
        if hline is not None:
            ax.axhline(hline, color="black", ls="--", lw=1.0, alpha=0.7)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Calibration size n")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    draw(axes[0, 0], "marginal_coverage", "Marginal coverage", "coverage", hline=target)
    draw(axes[0, 1], "marginal_set_size", "Mean set size", "size")
    draw(axes[0, 2], "cov_gap_mean", f"CovGap (pp, lower=better)", "CovGap (pp)")
    draw(axes[1, 0], "worst_class_coverage", "Worst-class coverage", "coverage", hline=target)
    draw(axes[1, 1], "time", "Runtime per (mode, trial)", "seconds", logy=True)

    # Agreement panel (frozen vs exact)
    ax = axes[1, 2]
    xs = [cs for cs in cals if agree_by_cal.get(cs)]
    if xs:
        ident = [agree_by_cal[cs]["identical_set_frac_mean"] for cs in xs]
        symd = [agree_by_cal[cs]["mean_symdiff_mean"] for cs in xs]
        flips = [agree_by_cal[cs]["frac_truelabel_flips_mean"] for cs in xs]
        ax.plot(xs, ident, "-o", color="#2ca02c", label="identical-set frac")
        ax.plot(xs, flips, "-s", color="#9467bd", label="true-label flip frac")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction")
        ax2 = ax.twinx()
        ax2.plot(xs, symd, ":^", color="#ff7f0e", label="mean symdiff")
        ax2.set_ylabel("mean |symmetric difference|")
        ln1, lb1 = ax.get_legend_handles_labels()
        ln2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(ln1 + ln2, lb1 + lb2, fontsize=8, loc="center right")
    ax.set_title("Exact vs frozen agreement")
    ax.set_xlabel("Calibration size n")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{ds_name.upper()} — MS-CS exact (exchangeable) vs frozen (O(1/n)), "
                 f"target {target:.0%}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MS-CS exact vs frozen (O(1/n)) exchangeability test")
    ap.add_argument("--datasets", nargs="+", default=["cifar100"],
                    choices=list(DATASETS.keys()))
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 600, 800])
    ap.add_argument("--test_size", type=int, default=2000,
                    help="Stratified test points. K=100 default 2000 -> 20/class (CovGap).")
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--ncm", type=str, default=NCM_FCP)
    ap.add_argument("--lam", type=float, default=MSCS_LAMBDA, help="MS-CS penalty weight lambda")
    ap.add_argument("--n_clusters", type=int, default=MSCS_N_CLUSTERS)
    ap.add_argument("--tau_mult", type=float, default=MSCS_TAU_MULT,
                    help="effective_tau = tau_mult * median squared inter-cluster dist")
    ap.add_argument("--pca_dim", type=int, default=None, help="Override dataset's PCA dim")
    ap.add_argument("--no_anchor", action="store_true", help="Skip the no-penalty FCP+PCA anchor")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="Device for the no-penalty anchor only (MS-CS modes are CPU).")
    ap.add_argument("--embeddings_root", type=str, default="",
                    help="Prefix for DATASETS paths (run from another checkout).")
    ap.add_argument("--output_dir", type=str, default="output/mscs_exchangeability")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="Synthetic-data end-to-end smoke test (no .pt files needed).")
    args = ap.parse_args()

    # Windows consoles default to a legacy codepage; force UTF-8 so prints don't
    # crash on non-ASCII (and degrade gracefully if reconfigure is unavailable).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # --- correctness gate for the vectorized exchangeable inner loop ---
    max_abs = verify_vectorization()
    print(f"[selftest] vectorized cal-penalty == per-j loop  (max abs diff={max_abs:g})  OK")

    modes = [MODE_FROZEN, MODE_EXACT] + ([] if args.no_anchor else [MODE_ANCHOR])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []

    # ---- SMOKE: synthetic data, tiny scale, full pipeline ----
    if args.smoke:
        print("\n=== SMOKE (synthetic blobs) ===")
        X, y = make_synthetic(seed=args.seed, n_classes=20, n_per_class=70, d=64)
        Xu, _ = make_synthetic(seed=args.seed + 1, n_classes=20, n_per_class=40, d=64)
        all_classes = np.unique(y)
        pca = PCA(n_components=32).fit(Xu)
        Xu_pca = pca.transform(Xu)
        rng = np.random.default_rng(args.seed)
        Xc, yc, Xt, yt, _ = stratified_split(X, y, all_classes, 200, 400, rng)
        out, sets = run_modes(
            pca.transform(Xc), yc, pca.transform(Xt), yt, Xu_pca, all_classes,
            args.alpha, args.ncm, args.lam, args.n_clusters, args.tau_mult,
            do_exact=True, do_anchor=not args.no_anchor, anchor_device="cpu")
        ag = set_agreement(sets["frozen"], sets["exact"], yt)
        print(f"  {'mode':<28} {'cov':>6} {'size':>6} {'CovGap':>7} {'time':>7}")
        for m in modes:
            d = out[m]
            print(f"  {m:<28} {d['marginal_coverage']:>6.3f} "
                  f"{d['marginal_set_size']:>6.2f} {d['cov_gap_mean']:>7.2f} "
                  f"{d['time']:>7.2f}s")
        print(f"  agreement: identical-set={ag['identical_set_frac']:.3f}  "
              f"mean|d_size|={ag['mean_abs_size_delta']:.3f}  "
              f"d_cov(exact-frozen)={ag['coverage_delta']:+.3f}  "
              f"true-label flips={ag['frac_truelabel_flips']:.3f}")
        print("\nSmoke OK.")
        return

    print(f"\nConfig: ncm={args.ncm} lambda={args.lam} n_clusters={args.n_clusters} "
          f"tau_mult={args.tau_mult} alpha={args.alpha} trials={args.n_trials}")
    print("NOTE: both MS-CS modes run on CPU (vectorized NumPy); --device affects "
          "only the no-penalty anchor.\n")

    for ds_name in args.datasets:
        print(f"{'='*72}\nDataset: {ds_name.upper()}\n{'='*72}")
        prep = prepare_dataset(ds_name, args.cal_sizes, args.test_size,
                               args.pca_dim, args.seed, root=args.embeddings_root)
        if prep is None:
            continue
        all_classes = prep["all_classes"]
        pca = prep["pca"]

        by_cal = {}        # cal -> {mode -> agg}
        agree_by_cal = {}  # cal -> agg agreement
        raw = {}           # cal -> {mode -> [trial dicts]}, plus "_agreement"

        for cs in args.cal_sizes:
            print(f"\n--- cal={cs} ---")
            trial_metrics = {m: [] for m in modes}
            trial_agree = []
            t0_cs = time.time()

            for trial in range(args.n_trials):
                rng = np.random.default_rng(args.seed + trial * 1000)

                if prep["X_test_ext"] is not None:
                    X_test_pca = prep["X_test_pca_ext"]
                    y_test = prep["y_test_ext"]
                    cal_idx = balanced_sample(prep["X_lab_rem"], prep["y_lab_rem"],
                                              all_classes, cs, rng)
                    X_cal_pca = pca.transform(prep["X_lab_rem"][cal_idx])
                    y_cal = prep["y_lab_rem"][cal_idx]
                else:
                    Xc, y_cal, Xt, y_test, _ = stratified_split(
                        prep["X_lab_rem"], prep["y_lab_rem"], all_classes,
                        cs, args.test_size, rng)
                    X_cal_pca = pca.transform(Xc)
                    X_test_pca = pca.transform(Xt)

                out, sets = run_modes(
                    X_cal_pca, y_cal, X_test_pca, y_test, prep["X_unlabeled_pca"],
                    all_classes, args.alpha, args.ncm, args.lam, args.n_clusters,
                    args.tau_mult, do_exact=True, do_anchor=not args.no_anchor,
                    anchor_device=args.device)

                for m in modes:
                    trial_metrics[m].append(out[m])
                trial_agree.append(set_agreement(sets["frozen"], sets["exact"], y_test))
                print(f"  trial {trial+1}/{args.n_trials} done "
                      f"({time.time() - t0_cs:.1f}s elapsed)")

            by_cal[cs] = {m: aggregate_method(trial_metrics[m]) for m in modes}
            agree_by_cal[cs] = aggregate_agreement(trial_agree)
            raw[cs] = {m: trial_metrics[m] for m in modes}
            raw[cs]["_agreement"] = trial_agree

            # --- print comparison table ---
            print(f"\n  cal={cs} summary ({time.time() - t0_cs:.1f}s)")
            print(f"  {'Method':<28} {'Cov':>6} {'Size':>6} {'CovGap':>7} "
                  f"{'Worst':>6} {'Frac<':>6} {'Time(s)':>8}")
            for m in modes:
                a = by_cal[cs][m]
                if a.get("n_trials_valid", 0) == 0:
                    print(f"  {m:<28}  <no valid trials>")
                    continue
                print(f"  {m:<28} {a['marginal_coverage_mean']:>6.3f} "
                      f"{a['marginal_set_size_mean']:>6.2f} "
                      f"{a['cov_gap_mean_mean']:>7.2f} "
                      f"{a['worst_class_coverage_mean']:>6.2f} "
                      f"{a['frac_undercovered_mean']:>6.2f} "
                      f"{a['time_mean']:>8.2f}")
                csv_rows.append({
                    "dataset": ds_name, "cal_size": cs, "mode": m,
                    "marginal_coverage": a["marginal_coverage_mean"],
                    "marginal_coverage_std": a["marginal_coverage_std"],
                    "set_size": a["marginal_set_size_mean"],
                    "cov_gap_mean": a["cov_gap_mean_mean"],
                    "worst_class_coverage": a["worst_class_coverage_mean"],
                    "frac_undercovered": a["frac_undercovered_mean"],
                    "time_sec": a["time_mean"], "n_trials": a["n_trials_valid"],
                })
            ag = agree_by_cal[cs]
            print(f"  exact vs frozen: identical-set={ag['identical_set_frac_mean']:.3f}"
                  f"+/-{ag['identical_set_frac_std']:.3f}  "
                  f"mean|d_size|={ag['mean_abs_size_delta_mean']:.3f}  "
                  f"d_size(exact-frozen)={ag['mean_signed_size_delta_mean']:+.3f}  "
                  f"d_cov={ag['coverage_delta_mean']:+.4f}  "
                  f"true-label flips={ag['frac_truelabel_flips_mean']:.4f}")

        # plots + JSON
        plot_dataset(by_cal, agree_by_cal, out_dir / f"{ds_name}_exact_vs_frozen.png",
                     ds_name, args.alpha, modes)
        print(f"\n  Saved plot: {ds_name}_exact_vs_frozen.png")

        with open(out_dir / f"{ds_name}_mscs_exchangeability.json", "w") as f:
            json.dump(_jsonify({
                "aggregated": by_cal, "agreement": agree_by_cal, "raw": raw,
                "config": {
                    "alpha": args.alpha, "n_trials": args.n_trials, "ncm": args.ncm,
                    "lambda": args.lam, "n_clusters": args.n_clusters,
                    "tau_mult": args.tau_mult, "pca_dim": prep["pca_dim"],
                    "test_size": args.test_size, "n_classes": int(prep["n_classes"]),
                },
            }), f, indent=2)
        print(f"  Saved JSON: {ds_name}_mscs_exchangeability.json")

    if csv_rows:
        csv_path = out_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nSaved summary CSV: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
