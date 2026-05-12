"""
Split Audit Experiment: Compare balanced vs unbalanced calibration splits,
and test with proper train/test separation (cal from train embeddings, test from test embeddings).

Compares FCP (geodesic_topk_mean) vs CV+ vs SCP under:
  (A) Old approach: random split from single pool (train embeddings only)
  (B) Balanced split from single pool
  (C) Proper separation: cal from train embeddings, test from test embeddings (balanced)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from conformal_prediction import (
    FullConformalPredictor, CrossValidationPlusPredictor, SoftmaxSplitCP,
    create_ncm, cal_test_split, stratified_cal_test_split,
)


def run_all_methods(X_cal, y_cal, X_test, y_test, all_classes, ncm_name, alpha,
                    n_folds=5):
    """Run FCP, CV+, and SCP on given cal/test split. Returns dict of metrics."""
    results = {}

    # --- FCP ---
    t0 = time.time()
    ncm = create_ncm(ncm_name, k=5)
    cp = FullConformalPredictor(ncm, alpha=alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)
    m = cp.evaluate(X_test, y_test, verbose=False)
    results["FCP"] = {"cov": m["coverage"], "sz": m["avg_set_size"],
                      "time": time.time() - t0}

    # --- CV+ ---
    t0 = time.time()
    ncm_cv_factory = lambda: create_ncm("geodesic_nn_ratio", k=5)
    cp_cv = CrossValidationPlusPredictor(ncm_cv_factory, alpha=alpha, n_folds=n_folds)
    cp_cv.calibrate(X_cal, y_cal, all_classes=all_classes)
    m_cv = cp_cv.evaluate(X_test, y_test, verbose=False)
    results["CV+"] = {"cov": m_cv["coverage"], "sz": m_cv["avg_set_size"],
                      "time": time.time() - t0}

    # --- SCP ---
    # SCP needs a train set; split cal into train(50%) + cal_scp(50%)
    n_cal = len(X_cal)
    n_train_scp = n_cal // 2
    X_train_scp, y_train_scp = X_cal[:n_train_scp], y_cal[:n_train_scp]
    X_cal_scp, y_cal_scp = X_cal[n_train_scp:], y_cal[n_train_scp:]

    t0 = time.time()
    cp_scp = SoftmaxSplitCP(alpha=alpha)
    cp_scp.fit(X_train_scp, y_train_scp)
    cp_scp.calibrate(X_cal_scp, y_cal_scp)
    m_scp = cp_scp.evaluate(X_test, y_test, verbose=False)
    results["SCP"] = {"cov": m_scp["coverage"], "sz": m_scp["avg_set_size"],
                      "time": time.time() - t0}

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split audit: balanced vs unbalanced vs train/test separation")
    parser.add_argument("--train_embeddings", type=str, default="output/embeddings_cifar100.pt",
                       help="Train pool embeddings")
    parser.add_argument("--test_embeddings", type=str, default="output/embeddings_cifar100_test.pt",
                       help="Test pool embeddings (separate split)")
    parser.add_argument("--ncm", type=str, default="geodesic_topk_mean")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cal_sizes", type=int, nargs="+", default=[300, 400, 600])
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load train pool
    data_train = torch.load(args.train_embeddings, map_location="cpu", weights_only=False)
    X_train_pool = data_train["embeddings"].numpy()
    y_train_pool = data_train["labels"].numpy()
    all_classes = np.unique(y_train_pool)
    n_classes = len(all_classes)
    print(f"Train pool: {X_train_pool.shape}, {n_classes} classes")

    # Load test pool (separate split)
    has_test_split = os.path.exists(args.test_embeddings)
    if has_test_split:
        data_test = torch.load(args.test_embeddings, map_location="cpu", weights_only=False)
        X_test_pool = data_test["embeddings"].numpy()
        y_test_pool = data_test["labels"].numpy()
        print(f"Test pool (separate): {X_test_pool.shape}, {len(np.unique(y_test_pool))} classes")
    else:
        print(f"WARNING: {args.test_embeddings} not found. Skipping mode (C).")

    print(f"\nNCM: {args.ncm}, alpha={args.alpha}, trials={args.n_trials}")
    print(f"Cal sizes: {args.cal_sizes}, test_size={args.test_size}")
    print()

    # Results storage
    all_results = {}

    for cal_size in args.cal_sizes:
        print(f"{'='*70}")
        print(f"cal_size={cal_size}")
        print(f"{'='*70}")
        all_results[cal_size] = {"A": [], "B": [], "C": []}

        for trial in range(args.n_trials):
            seed = args.seed + trial * 1000
            rng = np.random.default_rng(seed)

            # --- Mode A: Unbalanced stratified split from single pool (old approach) ---
            X_cal_a, y_cal_a, X_test_a, y_test_a = stratified_cal_test_split(
                X_train_pool, y_train_pool,
                cal_size=cal_size, test_size=args.test_size,
                balanced=False, random_state=seed)

            res_a = run_all_methods(X_cal_a, y_cal_a, X_test_a, y_test_a,
                                   all_classes, args.ncm, args.alpha)
            all_results[cal_size]["A"].append(res_a)

            # --- Mode B: Balanced stratified split from single pool ---
            X_cal_b, y_cal_b, X_test_b, y_test_b = stratified_cal_test_split(
                X_train_pool, y_train_pool,
                cal_size=cal_size, test_size=args.test_size,
                balanced=True, random_state=seed)

            res_b = run_all_methods(X_cal_b, y_cal_b, X_test_b, y_test_b,
                                   all_classes, args.ncm, args.alpha)
            all_results[cal_size]["B"].append(res_b)

            # --- Mode C: Proper train/test separation (balanced cal from train, test from test split) ---
            if has_test_split:
                # Cal from train pool (balanced)
                cal_per_class = cal_size // n_classes
                remainder = cal_size - cal_per_class * n_classes
                cal_idx_c = []
                class_indices = {c: np.where(y_train_pool == c)[0] for c in all_classes}
                for ci, c in enumerate(all_classes):
                    avail = rng.permutation(class_indices[c])
                    take = cal_per_class + (1 if ci < remainder else 0)
                    cal_idx_c.append(avail[:take])
                cal_idx_c = np.concatenate(cal_idx_c)
                cal_idx_c = rng.permutation(cal_idx_c)
                X_cal_c, y_cal_c = X_train_pool[cal_idx_c], y_train_pool[cal_idx_c]

                # Test from test pool (stratified sample)
                test_per_class = args.test_size // n_classes
                test_idx_c = []
                test_class_indices = {c: np.where(y_test_pool == c)[0] for c in all_classes}
                for c in all_classes:
                    avail = rng.permutation(test_class_indices[c])
                    test_idx_c.append(avail[:test_per_class])
                test_idx_c = np.concatenate(test_idx_c)
                X_test_c, y_test_c = X_test_pool[test_idx_c], y_test_pool[test_idx_c]

                res_c = run_all_methods(X_cal_c, y_cal_c, X_test_c, y_test_c,
                                       all_classes, args.ncm, args.alpha)
                all_results[cal_size]["C"].append(res_c)

        # Print results for this cal_size
        for mode, label in [("A", "Random split (single pool)"),
                            ("B", "Balanced split (single pool)"),
                            ("C", "Balanced cal(train) + test(test split)")]:
            trials = all_results[cal_size][mode]
            if not trials:
                continue
            print(f"\n  [{mode}] {label}:")
            for method in ["FCP", "CV+", "SCP"]:
                covs = [t[method]["cov"] for t in trials]
                szs = [t[method]["sz"] for t in trials]
                times = [t[method]["time"] for t in trials]
                valid = "OK" if np.mean(covs) >= 0.89 else "!!"
                print(f"    {method:4s}  cov={np.mean(covs):.3f}+/-{np.std(covs):.3f} {valid}  "
                      f"sz={np.mean(szs):.2f}+/-{np.std(szs):.2f}  "
                      f"({np.mean(times):.1f}s)")

    print("\n" + "="*70)
    print("SUMMARY: Does balanced/separated split change results?")
    print("="*70)
    for cal_size in args.cal_sizes:
        print(f"\ncal={cal_size}  |  FCP set size comparison:")
        for mode, label in [("A", "Random"), ("B", "Balanced"), ("C", "Separated")]:
            trials = all_results[cal_size][mode]
            if not trials:
                continue
            sz_fcp = np.mean([t["FCP"]["sz"] for t in trials])
            cov_fcp = np.mean([t["FCP"]["cov"] for t in trials])
            print(f"  {mode} ({label:10s}): cov={cov_fcp:.3f}  sz={sz_fcp:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
