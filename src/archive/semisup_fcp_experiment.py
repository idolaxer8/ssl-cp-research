"""
Semi-Supervised Full Conformal Prediction Experiment.

Two augmentation modes for using unlabeled data with FCP:

1. Full augmentation (--mode full): augments both same-class and other-class
   neighbor pools. Simple but breaks exchangeability at O(1/n) due to feedback
   loop: cal points influence pseudo-labels which serve as their own neighbors.
   Causes undercoverage that worsens with N.

2. Denominator-only augmentation (--mode denom_only): augments only the
   other-class (denominator) neighbor pool. Same-class distances are computed
   from real cal data only. Preserves exchangeability because:
   - Cal LOO scores: numerator unchanged from baseline (no feedback loop)
   - Denominator enriched equally for cal and test
   - Removing one cal point barely affects other-class pseudo-labels: O(1/n^2)

Design:
  1. Pseudo-label unlabeled pool via 1-NN to calibration data
  2. Mode-dependent NCM fitting:
     - full: fit on augmented pool [real cal | pseudo-labeled]
     - denom_only: fit_with_background(X_cal, y_cal, X_bg, y_bg)
  3. Run FCP prediction loop
  4. P-values computed over REAL calibration points only

References:
  - Zhou et al. (2026) "SemiCP" -- NNM score for semi-supervised CP (CVPR)
  - Bozorgtabar et al. (2026) "LATA" -- kNN graph smoothing (arXiv:2602.17535)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm
from conformal_prediction import (
    create_ncm, stratified_cal_test_split, FullConformalPredictor,
)


# ---------------------------------------------------------------------------
# Pseudo-labeling
# ---------------------------------------------------------------------------

def pseudo_label_1nn(X_unl_wn, X_cal_wn, y_cal):
    """
    Assign pseudo-labels to unlabeled points via 1-NN in whitened space.

    Returns:
        y_pseudo : (N_unl,) pseudo-labels
        nn_sims  : (N_unl,) cosine similarity to nearest cal point (confidence)
    """
    sims = X_unl_wn @ X_cal_wn.T          # (N_unl, n_cal)
    nn_idx = np.argmax(sims, axis=1)       # (N_unl,)
    nn_sims = sims[np.arange(len(X_unl_wn)), nn_idx]
    return y_cal[nn_idx], nn_sims


# ---------------------------------------------------------------------------
# Semi-supervised FCP prediction loop
# ---------------------------------------------------------------------------

def semisup_fcp_predict(ncm, n_cal_real, X_test, y_test, all_classes, alpha,
                        desc="SemiSup FCP"):
    """
    FCP prediction using an NCM fitted on an augmented pool.

    P-values are computed over the first `n_cal_real` calibration scores only.
    The remaining (pseudo-labeled) points enrich the neighbor pool but do NOT
    contribute to the p-value denominator.
    """
    n_test = len(X_test)
    prediction_sets = []

    iterator = tqdm(range(n_test), desc=desc)
    for i in iterator:
        x_test = X_test[i]
        pred_set = []
        for y_c in all_classes:
            yc = int(y_c)
            test_score = ncm.score_x(x_test, yc)
            updated_scores = ncm.updated_calibration_scores_for(x_test, yc)

            # Only count real calibration points for p-value
            real_scores = updated_scores[:n_cal_real]
            n_geq = np.sum(real_scores >= test_score)
            p_value = (n_geq + 1) / (n_cal_real + 1)

            if p_value > alpha:
                pred_set.append(yc)
        prediction_sets.append(pred_set)

    set_sizes = np.array([len(s) for s in prediction_sets])
    coverage = np.mean([y_test[i] in prediction_sets[i]
                        for i in range(n_test)])
    return {
        "coverage": float(coverage),
        "avg_set_size": float(np.mean(set_sizes)),
        "median_set_size": float(np.median(set_sizes)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Semi-supervised FCP: pseudo-labeled augmentation")
    p.add_argument("--cal_embeddings", default="output/embeddings_cifar100.pt")
    p.add_argument("--test_embeddings", default="output/embeddings_cifar100_test.pt")
    p.add_argument("--unlabeled_embeddings",
                   default="output/embeddings_cifar100_unlabeled.pt")
    p.add_argument("--ncm", default="geodesic_topk_mean")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--cal_sizes", type=int, nargs="+", default=[400, 600])
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--n_pseudo", type=int, nargs="+", default=[500, 1000, 2000],
                   help="Number of pseudo-labeled points to test")
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--confidence_threshold", type=float, default=0.0,
                   help="Min cosine sim for pseudo-label inclusion (0=all)")
    p.add_argument("--use_true_labels", action="store_true",
                   help="Oracle: use true labels instead of pseudo-labels (diagnostic)")
    p.add_argument("--mode", choices=["full", "denom_only", "whiten_only"],
                   default="denom_only",
                   help="full: augment both pools. denom_only: augment other-class only. "
                        "whiten_only: use unlabeled data only for better whitening (default: denom_only)")
    args = p.parse_args()

    # ---- Load data ----
    data_cal = torch.load(args.cal_embeddings, map_location="cpu",
                          weights_only=False)
    X_cal_pool = data_cal["embeddings"].numpy()
    y_cal_pool = data_cal["labels"].numpy()
    all_classes = np.unique(y_cal_pool)
    n_classes = len(all_classes)
    print(f"Cal pool: {X_cal_pool.shape}, {n_classes} classes")

    data_test = torch.load(args.test_embeddings, map_location="cpu",
                           weights_only=False)
    X_test_pool = data_test["embeddings"].numpy()
    y_test_pool = data_test["labels"].numpy()
    print(f"Test pool: {X_test_pool.shape}")

    data_unl = torch.load(args.unlabeled_embeddings, map_location="cpu",
                          weights_only=False)
    X_unl_all = data_unl["embeddings"].numpy()
    y_unl_true = (data_unl["labels"].numpy()
                  if "labels" in data_unl else None)
    print(f"Unlabeled pool: {X_unl_all.shape}")

    print(f"\nNCM: {args.ncm}, alpha={args.alpha}, mode={args.mode}, "
          f"trials={args.n_trials}")
    print(f"Cal sizes: {args.cal_sizes}, n_pseudo sweep: {args.n_pseudo}")

    # ---- Run experiments ----
    for cal_size in args.cal_sizes:
        print(f"\n{'='*70}")
        print(f"cal_size = {cal_size}  ({cal_size // n_classes}/class)")
        print(f"{'='*70}")

        # Collect results: n_pseudo -> list of trial dicts
        all_results = {0: []}  # 0 = baseline
        for np_ in args.n_pseudo:
            all_results[np_] = []

        for trial in range(args.n_trials):
            seed = args.seed + trial * 1000
            rng = np.random.default_rng(seed)

            # Balanced cal split from cal pool
            X_cal, y_cal, _, _ = stratified_cal_test_split(
                X_cal_pool, y_cal_pool,
                cal_size=cal_size, test_size=0,
                balanced=True, random_state=seed)

            # Test from separate test pool (balanced)
            test_per_class = args.test_size // n_classes
            test_idx = []
            for c in all_classes:
                avail = rng.permutation(np.where(y_test_pool == c)[0])
                test_idx.append(avail[:test_per_class])
            test_idx = np.concatenate(test_idx)
            X_test = X_test_pool[test_idx]
            y_test = y_test_pool[test_idx]

            # ---- Baseline (no pseudo) ----
            t0 = time.time()
            ncm_base = create_ncm(args.ncm, k=5)
            ncm_base.fit(X_cal, y_cal)
            m_base = semisup_fcp_predict(
                ncm_base, len(X_cal), X_test, y_test,
                all_classes, args.alpha,
                desc=f"Base t{trial}")
            m_base["time"] = time.time() - t0
            all_results[0].append(m_base)

            # Pseudo-label once for this trial (using baseline NCM whitening)
            X_unl_wn = ncm_base._whiten_normalize(X_unl_all)
            y_pseudo_all, nn_sims_all = pseudo_label_1nn(
                X_unl_wn, ncm_base.X_cal_wn, y_cal)

            if trial == 0 and y_unl_true is not None:
                acc = np.mean(y_pseudo_all == y_unl_true)
                print(f"  Pseudo-label accuracy (full pool): {acc:.3f}")

            # ---- Semi-sup at each n_pseudo ----
            for np_ in args.n_pseudo:
                n_use = min(np_, len(X_unl_all))
                # Deterministic subsample of unlabeled pool
                unl_idx = rng.permutation(len(X_unl_all))[:n_use]
                X_unl = X_unl_all[unl_idx]

                # Use true labels (oracle) or pseudo-labels
                if args.use_true_labels and y_unl_true is not None:
                    y_pseudo = y_unl_true[unl_idx]
                else:
                    y_pseudo = y_pseudo_all[unl_idx]

                # Confidence filter
                if args.confidence_threshold > 0:
                    keep = nn_sims_all[unl_idx] >= args.confidence_threshold
                    X_unl = X_unl[keep]
                    y_pseudo = y_pseudo[keep]

                t0 = time.time()
                ncm_semi = create_ncm(args.ncm, k=5)

                if args.mode == "denom_only":
                    ncm_semi.fit_with_background(X_cal, y_cal, X_unl, y_pseudo)
                elif args.mode == "whiten_only":
                    # Pre-compute whitening from augmented pool, fit NCM on cal only
                    X_all = np.concatenate([X_cal, X_unl], axis=0)
                    y_all = np.concatenate([y_cal, y_pseudo])
                    residuals = X_all.copy()
                    for c in np.unique(y_all):
                        idx = np.where(y_all == c)[0]
                        residuals[idx] -= X_all[idx].mean(axis=0)
                    pooled_var = (residuals ** 2).mean(axis=0)
                    adaptive_reg = max(1e-4, 0.01 * float(np.median(pooled_var)))
                    ncm_semi.inv_std = 1.0 / np.sqrt(pooled_var + adaptive_reg)
                    ncm_semi.whiten = False
                    ncm_semi.fit(X_cal, y_cal)
                else:
                    X_aug = np.concatenate([X_cal, X_unl], axis=0)
                    y_aug = np.concatenate([y_cal, y_pseudo])
                    ncm_semi.fit(X_aug, y_aug)

                m_semi = semisup_fcp_predict(
                    ncm_semi, len(X_cal), X_test, y_test,
                    all_classes, args.alpha,
                    desc=f"Semi({np_}) t{trial}")
                m_semi["time"] = time.time() - t0
                m_semi["n_pseudo_used"] = len(X_unl)
                all_results[np_].append(m_semi)

        # ---- Print results table ----
        print(f"\n  Results (cal={cal_size}, {args.n_trials} trials, "
              f"mode={args.mode}):\n")
        print(f"  {'Method':<20s}  {'Coverage':>14s}  {'Set Size':>14s}  {'Time':>6s}")
        print(f"  {'-'*20}  {'-'*14}  {'-'*14}  {'-'*6}")

        baseline_sz = np.mean([r["avg_set_size"] for r in all_results[0]])

        for np_ in sorted(all_results.keys()):
            trials = all_results[np_]
            covs = [r["coverage"] for r in trials]
            szs = [r["avg_set_size"] for r in trials]
            times = [r["time"] for r in trials]
            label = "Baseline" if np_ == 0 else f"SemiSup(N={np_})"
            valid = "OK" if np.mean(covs) >= 0.89 else "!!"

            delta = ""
            if np_ > 0:
                pct = (baseline_sz - np.mean(szs)) / baseline_sz * 100
                delta = f" ({pct:+.1f}%)"

            print(f"  {label:<20s}  "
                  f"{np.mean(covs):.3f}+/-{np.std(covs):.3f} {valid}  "
                  f"{np.mean(szs):.2f}+/-{np.std(szs):.2f}{delta:<8s}  "
                  f"{np.mean(times):5.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
