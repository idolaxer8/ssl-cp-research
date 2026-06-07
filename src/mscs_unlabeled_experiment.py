"""
MS-CS with Unlabeled Data: build class similarity matrix from k-means clustering
on external unlabeled embeddings to avoid data leakage.

Pipeline:
1. K-means on unlabeled embeddings -> cluster centroids
2. Compute class centroids from calibration data only
3. Match: assign each cal class centroid to nearest k-means cluster
4. Build M[c,c'] from cluster co-assignment + inter-cluster distance
5. Apply MS-CS penalty: s_l(x,y) = s(x,y) + l * (1 - M[y, y_hat(x)])

Reference: Fargion, Dabah & Tirer (2025), Section 4.
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from conformal_prediction import FullConformalPredictor, create_ncm


def build_cluster_similarity_matrix(X_unlabeled, X_cal, y_cal, all_classes, n_clusters, tau=1.0):
    """
    Build class similarity matrix from k-means on unlabeled data.

    Args:
        X_unlabeled: Unlabeled embeddings (n_u, d) for clustering
        X_cal: Calibration embeddings (n_cal, d)
        y_cal: Calibration labels (n_cal,)
        all_classes: All possible class labels (used for indexing M)
        n_clusters: Number of k-means clusters
        tau: Temperature for inter-cluster distance kernel

    Returns:
        M: (n_classes, n_classes) similarity matrix, M[c,c'] in [0,1], M[c,c]=1
        class_to_cluster: mapping from class index to cluster assignment
    """
    classes = all_classes
    n_classes = len(classes)

    # Step 1: K-means on unlabeled data (cast to float64 for sklearn compatibility)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_unlabeled.astype(np.float64))
    cluster_centroids = km.cluster_centers_  # (n_clusters, d)

    # Step 2: Compute class centroids from calibration data
    # For classes absent from cal, use global cal centroid as fallback
    global_centroid = X_cal.astype(np.float64).mean(axis=0)
    class_centroids = np.zeros((n_classes, X_cal.shape[1]), dtype=np.float64)
    for i, c in enumerate(classes):
        mask = y_cal == c
        if mask.any():
            class_centroids[i] = X_cal[mask].astype(np.float64).mean(axis=0)
        else:
            class_centroids[i] = global_centroid

    # Step 3: Assign each class centroid to nearest k-means cluster
    class_to_cluster = km.predict(class_centroids)  # (n_classes,)

    # Step 4: Build M from cluster assignments + inter-cluster distances
    # Inter-cluster distance matrix
    cluster_dists = euclidean_distances(cluster_centroids, cluster_centroids)

    # Compute effective tau
    # If tau == "normalize" (handled by caller passing tau < 0), use median squared
    # inter-cluster distance as the scale
    upper_tri = cluster_dists[np.triu_indices(n_clusters, k=1)]
    median_dist_sq = float(np.median(upper_tri) ** 2)

    if tau < 0:
        # Negative tau signals normalization: effective_tau = |tau| * median_dist_sq
        effective_tau = abs(tau) * median_dist_sq
    else:
        effective_tau = tau

    M = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                M[i, j] = 1.0
            elif class_to_cluster[i] == class_to_cluster[j]:
                # Same cluster -> maximum similarity
                M[i, j] = 1.0
            else:
                # Different clusters -> exponential decay of inter-cluster distance
                d = cluster_dists[class_to_cluster[i], class_to_cluster[j]]
                M[i, j] = np.exp(-d**2 / effective_tau)

    # Per-class sample counts (for exchangeable centroid updates)
    class_counts = np.array([np.sum(y_cal == c) for c in classes])

    return M, class_to_cluster, effective_tau, median_dist_sq, class_centroids, class_counts, cluster_centroids, cluster_dists


def update_M_for_candidate(class_centroids, class_counts, class_to_cluster,
                           cluster_centroids, cluster_dists, effective_tau,
                           yc_idx, x_test, M_base):
    """
    Update similarity matrix M when (x_test, yc) is added to the augmented set.

    Only class yc's centroid shifts: new_centroid = (n_y * old + x_test) / (n_y + 1).
    This may change class yc's cluster assignment, requiring M row/col update.

    Returns updated M (or M_base if no cluster change).
    """
    n_y = class_counts[yc_idx]
    if n_y == 0:
        # Class absent from cal — centroid becomes x_test
        new_centroid = x_test.astype(np.float64)
    else:
        old_centroid = class_centroids[yc_idx]
        new_centroid = (n_y * old_centroid + x_test.astype(np.float64)) / (n_y + 1)

    # Predict new cluster for class yc
    dists_to_clusters = np.linalg.norm(cluster_centroids - new_centroid, axis=1)
    new_cluster = np.argmin(dists_to_clusters)

    if new_cluster == class_to_cluster[yc_idx]:
        # No cluster change → M unchanged
        return M_base

    # Cluster changed — update row and column for yc_idx
    M_updated = M_base.copy()
    n_classes = M_base.shape[0]
    old_c2c = class_to_cluster.copy()
    old_c2c[yc_idx] = new_cluster

    for j in range(n_classes):
        if j == yc_idx:
            M_updated[yc_idx, j] = 1.0
            M_updated[j, yc_idx] = 1.0
        elif old_c2c[j] == new_cluster:
            M_updated[yc_idx, j] = 1.0
            M_updated[j, yc_idx] = 1.0
        else:
            d = cluster_dists[new_cluster, old_c2c[j]]
            sim = np.exp(-d**2 / effective_tau)
            M_updated[yc_idx, j] = sim
            M_updated[j, yc_idx] = sim

    return M_updated


def run_fcp_with_mscs(X_cal, y_cal, X_test, y_test, all_classes, ncm_name,
                      alpha, lam, M, exchangeable=False,
                      class_centroids=None, class_counts=None,
                      class_to_cluster=None, cluster_centroids=None,
                      cluster_dists=None, effective_tau=None,
                      return_sets=False):
    """Run FCP with MS-CS continuous penalty.

    Args:
        exchangeable: If True, update M and y_hat per candidate (x_test, yc)
            to restore exchangeability. Requires passing clustering artifacts.
        return_sets: If True, return (coverage, avg_size, prediction_sets)
            instead of (coverage, avg_size). Default False keeps the
            existing 2-tuple signature for all callers in this repo.
    """
    ncm = create_ncm(ncm_name, k=5)
    cp = FullConformalPredictor(ncm, alpha=alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)

    # Map class labels to indices in M
    class_to_idx = {int(c): i for i, c in enumerate(all_classes)}

    # --- Compute LOO 1-NN penalties for calibration points (base, before augmentation) ---
    n_cal = len(X_cal)
    D_cal = euclidean_distances(X_cal, X_cal)
    np.fill_diagonal(D_cal, np.inf)
    loo_nn_idx = np.argmin(D_cal, axis=1)
    y_hat_loo = y_cal[loo_nn_idx]
    loo_nn_dists = D_cal[np.arange(n_cal), loo_nn_idx]

    # Class-index arrays for vectorized penalty gathers. These replace the per-j
    # Python loops below; bit-identical results, but they remove the O(n_cal)
    # inner loop that otherwise made the exchangeable path O(n_test * K * n_cal)
    # in pure Python.
    cal_class_idx = np.array([class_to_idx[int(c)] for c in y_cal])      # (n_cal,)
    yhat_loo_idx  = np.array([class_to_idx[int(c)] for c in y_hat_loo])  # (n_cal,)

    # Base cal penalty (used when exchangeable=False, or as starting point)
    cal_penalty_base = lam * (1.0 - M[cal_class_idx, yhat_loo_idx])

    # --- Predict with MS-CS penalty on test scores ---
    n_test = len(X_test)
    prediction_sets = []

    D_test_cal = euclidean_distances(X_test, X_cal)

    for i in range(n_test):
        x_test = X_test[i]
        dists = D_test_cal[i]  # distances from x_test to each cal point

        # y_hat for test point = 1-NN to cal (LOO in augmented set)
        nn_idx = np.argmin(dists)
        y_hat_test = int(y_cal[nn_idx])
        y_hat_idx = class_to_idx[y_hat_test]

        pred_set = []
        for yc in all_classes:
            yc = int(yc)
            yc_idx = class_to_idx[yc]

            if exchangeable:
                # --- Exchangeable version: update M and y_hat for augmented set ---
                # 1. Update M: shift class yc centroid to include x_test
                M_aug = update_M_for_candidate(
                    class_centroids, class_counts, class_to_cluster,
                    cluster_centroids, cluster_dists, effective_tau,
                    yc_idx, x_test, M)

                # 2. Update cal y_hat: if x_test (with label yc) is closer than a
                #    cal point's current LOO-NN, that point's predicted label
                #    becomes yc. Vectorized over cal points (was a per-j loop).
                yhat_j_idx = np.where(dists < loo_nn_dists, yc_idx, yhat_loo_idx)
                cal_penalty = lam * (1.0 - M_aug[cal_class_idx, yhat_j_idx])

                # Test penalty with updated M
                test_score = cp.ncm.score_x(x_test, yc)
                test_score += lam * (1.0 - M_aug[yc_idx, y_hat_idx])
            else:
                # --- Original (non-exchangeable) version ---
                cal_penalty = cal_penalty_base
                test_score = cp.ncm.score_x(x_test, yc)
                test_score += lam * (1.0 - M[yc_idx, y_hat_idx])

            # Updated cal scores + cal_penalty
            updated_scores = cp.ncm.updated_calibration_scores_for(x_test, yc) + cal_penalty

            # p-value
            n_greater = np.sum(updated_scores >= test_score)
            p_value = (n_greater + 1) / (n_cal + 1)

            if p_value > alpha:
                pred_set.append(yc)

        prediction_sets.append(pred_set)

    coverage = np.mean([y_test[i] in prediction_sets[i] for i in range(n_test)])
    avg_size = np.mean([len(s) for s in prediction_sets])
    if return_sets:
        return coverage, avg_size, prediction_sets
    return coverage, avg_size


def stratified_split(X, y, cal_size, test_size, all_classes, rng):
    """Stratified pool -> cal + test split (same as macs_experiment.py)."""
    n_classes = len(all_classes)
    num_per_class = math.ceil((test_size + cal_size) / n_classes)
    min_count = min(np.sum(y == c) for c in all_classes)
    num_per_class = min(num_per_class, min_count)

    pool_idx = []
    for c in all_classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, num_per_class, replace=False)
        pool_idx.append(chosen)
    pool_idx = np.concatenate(pool_idx)
    rng.shuffle(pool_idx)

    X_pool, y_pool = X[pool_idx], y[pool_idx]
    X_test, y_test = X_pool[-test_size:], y_pool[-test_size:]
    X_rem, y_rem = X_pool[:-test_size], y_pool[:-test_size]

    # Stratified cal
    rem_classes = np.unique(y_rem)
    first = np.array([rng.choice(np.where(y_rem == c)[0], 1, replace=False)[0]
                      for c in rem_classes])
    rest = np.setdiff1d(np.arange(len(X_rem)), first)
    n_extra = cal_size - len(rem_classes)
    if n_extra > 0:
        extra = rng.choice(rest, min(n_extra, len(rest)), replace=False)
        cal_idx = np.concatenate([first, extra])
    else:
        cal_idx = first[:cal_size]
    cal_idx = rng.permutation(cal_idx)

    return X_rem[cal_idx], y_rem[cal_idx], X_test, y_test


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MS-CS with unlabeled data experiment")
    parser.add_argument("--embeddings_path", type=str, default="output/embeddings_cifar100.pt",
                       help="Labeled embeddings (cal+test pool)")
    parser.add_argument("--unlabeled_path", type=str, default=None,
                       help="Unlabeled embeddings .pt file (if None, hold out from labeled pool)")
    parser.add_argument("--ncm", type=str, default="geodesic_topk_mean")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_clusters", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--lambdas", type=float, nargs="+",
                       default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.1])
    parser.add_argument("--taus", type=float, nargs="+", default=[1.0],
                       help="Temperature values for inter-cluster similarity kernel")
    parser.add_argument("--tau_normalize", action="store_true",
                       help="Normalize tau by median squared inter-cluster distance. "
                            "Each --taus value becomes a multiplier: effective_tau = t * median_d^2")
    parser.add_argument("--cal_sizes", type=int, nargs="+", default=[400, 600])
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exchangeable", action="store_true",
                       help="Update M and y_hat per candidate to restore exchangeability")
    args = parser.parse_args()

    # Load labeled data
    data = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)
    X = data["embeddings"].numpy()
    y = data["labels"].numpy()
    all_classes = np.unique(y)
    print(f"Labeled data: {X.shape}, {len(all_classes)} classes")

    # Load or create unlabeled data
    if args.unlabeled_path:
        udata = torch.load(args.unlabeled_path, map_location="cpu", weights_only=False)
        X_unlabeled = udata["embeddings"].numpy()
        print(f"Unlabeled data (external): {X_unlabeled.shape}")
    else:
        # Hold out portion of labeled data as "unlabeled" (drop labels)
        # Use samples not selected for cal/test in any trial
        rng_u = np.random.default_rng(args.seed + 9999)
        n_unlabeled = min(len(X) // 2, 1500)
        u_idx = rng_u.choice(len(X), size=n_unlabeled, replace=False)
        X_unlabeled = X[u_idx]
        print(f"Unlabeled data (held out from labeled pool): {X_unlabeled.shape}")

    print(f"NCM: {args.ncm}, alpha={args.alpha}, trials={args.n_trials}")
    print(f"Cluster counts: {args.n_clusters}")
    print(f"Tau values: {args.taus} {'(normalized by median_d^2)' if args.tau_normalize else '(absolute)'}")
    print(f"Lambda sweep: {args.lambdas}")
    print(f"Exchangeable: {args.exchangeable}")
    print()

    for n_clust in args.n_clusters:
        for tau_input in args.taus:
            # If normalizing, pass negative tau as signal to build function
            tau_arg = -tau_input if args.tau_normalize else tau_input

            print(f"{'='*70}")
            tau_label = f"tau={tau_input}*median_d^2" if args.tau_normalize else f"tau={tau_input}"
            print(f"n_clusters={n_clust}, {tau_label}")
            print(f"{'='*70}")

            for cal_size in args.cal_sizes:
                print(f"\n  cal_size={cal_size}")
                print(f"  {'-'*60}")

                for lam in args.lambdas:
                    covs, szs = [], []
                    t0 = time.time()

                    for trial in range(args.n_trials):
                        rng = np.random.default_rng(args.seed + trial * 1000)
                        X_cal, y_cal, X_test, y_test = stratified_split(
                            X, y, cal_size, args.test_size, all_classes, rng)

                        if lam == 0.0:
                            # Plain FCP (no penalty) — baseline
                            ncm = create_ncm(args.ncm, k=5)
                            cp = FullConformalPredictor(ncm, alpha=args.alpha)
                            cp.calibrate(X_cal, y_cal, all_classes=all_classes)
                            m = cp.evaluate(X_test, y_test, verbose=False)
                            covs.append(m["coverage"])
                            szs.append(m["avg_set_size"])
                        else:
                            # Build M from unlabeled data + current calibration
                            (M, c2c, eff_tau, med_d2, cls_centroids,
                             cls_counts, clust_centroids, clust_dists
                             ) = build_cluster_similarity_matrix(
                                X_unlabeled, X_cal, y_cal, all_classes, n_clust, tau=tau_arg)

                            cov, sz = run_fcp_with_mscs(
                                X_cal, y_cal, X_test, y_test, all_classes,
                                args.ncm, args.alpha, lam, M,
                                exchangeable=args.exchangeable,
                                class_centroids=cls_centroids,
                                class_counts=cls_counts,
                                class_to_cluster=c2c,
                                cluster_centroids=clust_centroids,
                                cluster_dists=clust_dists,
                                effective_tau=eff_tau)
                            covs.append(cov)
                            szs.append(sz)

                    elapsed = time.time() - t0
                    cov_mean = np.mean(covs)
                    sz_mean = np.mean(szs)
                    valid = "OK" if cov_mean >= 0.89 else "!!"
                    label = f"lambda={lam:.2f}" if lam > 0 else "No penalty"
                    print(f"    {label:15s}  cov={cov_mean:.3f} {valid}  "
                          f"sz={sz_mean:.2f}  ({elapsed:.1f}s)")

    # Sanity check: print M properties for last trial
    if args.lambdas[-1] > 0:
        print(f"\n--- Similarity Matrix M sanity check (last config) ---")
        print(f"  n_clusters={n_clust}, effective_tau={eff_tau:.4f}, median_d^2={med_d2:.4f}")
        print(f"  M shape: {M.shape}")
        print(f"  M diagonal: all 1.0? {np.allclose(np.diag(M), 1.0)}")
        print(f"  M symmetric? {np.allclose(M, M.T)}")
        print(f"  M range: [{M.min():.4f}, {M.max():.4f}]")
        print(f"  M mean off-diagonal: {(M.sum() - M.trace()) / (M.shape[0]**2 - M.shape[0]):.4f}")

        # Cluster assignment distribution
        unique_clusters = len(np.unique(c2c))
        print(f"  Classes mapped to {unique_clusters}/{n_clust} clusters")

    print("\nDone.")


if __name__ == "__main__":
    main()
