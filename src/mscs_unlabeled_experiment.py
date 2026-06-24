"""
MS-CS class-similarity matrix M for the FCP penalty s_l(x,y)=s(x,y)+l*(1-M[y,y_hat]).

Two ways to build M (`--similarity`):
  cluster  (default): k-means on EXTERNAL unlabeled embeddings -> cluster
           centroids; assign each cal class-centroid to its nearest cluster;
           M[c,c'] from cluster co-assignment + inter-cluster distance.
  centroid : cal-only baseline, M[c,c']=exp(-||mu_c-mu_c'||^2/tau) from class
           centroids estimated on the calibration set. Uses NO unlabeled data,
           so running both under identical splits measures how much the
           unlabeled pool actually contributes.

Leakage note: when no --unlabeled_path is given, the unlabeled pool for the
cluster mode is carved as a DISJOINT, stratified holdout from the labeled data
(stratified_holdout_unlabeled), so M never sees the cal/test points. The earlier
fallback sampled from the full labeled set and overlapped cal/test.

Reference: Fargion, Dabah & Tirer (2025), Section 4.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from conformal_prediction import (
    FullConformalPredictor, create_ncm, warn_nonexchangeable,
    stratified_pool_split as stratified_split,
)


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


def build_centroid_similarity_matrix(X_cal, y_cal, all_classes, tau=1.0):
    """Cal-only class-similarity matrix from pairwise class-centroid distances.

    Unlabeled-data-free baseline for MS-CS: M[c,c'] = exp(-||mu_c - mu_c'||^2 / tau)
    where mu_c is class c's centroid computed from the CALIBRATION set only.
    No k-means, no unlabeled pool. The scale `tau` follows the same convention as
    build_cluster_similarity_matrix: a negative value signals normalization, i.e.
    effective_tau = |tau| * median squared off-diagonal centroid distance.

    This isolates the contribution of the unlabeled data: compare a run with this
    matrix against build_cluster_similarity_matrix under identical splits/lambda.

    Returns (cluster-only artifacts are simply not produced):
        M, effective_tau, median_dist_sq, class_centroids, class_counts
    """
    classes = all_classes
    n_classes = len(classes)

    # Class centroids from calibration data; absent classes -> global cal centroid.
    global_centroid = X_cal.astype(np.float64).mean(axis=0)
    class_centroids = np.zeros((n_classes, X_cal.shape[1]), dtype=np.float64)
    for i, c in enumerate(classes):
        mask = y_cal == c
        class_centroids[i] = (X_cal[mask].astype(np.float64).mean(axis=0)
                              if mask.any() else global_centroid)

    # Pairwise centroid distances + scale (median squared off-diagonal distance).
    centroid_dists = euclidean_distances(class_centroids, class_centroids)
    upper_tri = centroid_dists[np.triu_indices(n_classes, k=1)]
    median_dist_sq = float(np.median(upper_tri) ** 2)

    if tau < 0:
        effective_tau = abs(tau) * median_dist_sq
    else:
        effective_tau = tau
    if effective_tau <= 0:  # degenerate (near-coincident centroids): avoid /0
        effective_tau = 1.0

    M = np.exp(-centroid_dists**2 / effective_tau)
    np.fill_diagonal(M, 1.0)

    class_counts = np.array([np.sum(y_cal == c) for c in classes])
    return M, effective_tau, median_dist_sq, class_centroids, class_counts


def update_centroid_M_for_candidate(class_centroids, class_counts, effective_tau,
                                    yc_idx, x_test, M_base):
    """Exchangeable update of the centroid-distance M when (x_test, yc) is added.

    Adding (x_test, yc) to the augmented set shifts only class yc's centroid, so
    only row/col yc_idx of M changes: M[yc, j] = exp(-||mu_yc' - mu_j||^2 / tau).
    """
    n_y = class_counts[yc_idx]
    if n_y == 0:
        new_centroid = x_test.astype(np.float64)
    else:
        new_centroid = (n_y * class_centroids[yc_idx]
                        + x_test.astype(np.float64)) / (n_y + 1)

    dists = np.linalg.norm(class_centroids - new_centroid, axis=1)
    sims = np.exp(-dists**2 / effective_tau)
    M_updated = M_base.copy()
    M_updated[yc_idx, :] = sims
    M_updated[:, yc_idx] = sims
    M_updated[yc_idx, yc_idx] = 1.0
    return M_updated


def run_fcp_with_mscs(X_cal, y_cal, X_test, y_test, all_classes, ncm_name,
                      alpha, lam, M, exchangeable=False,
                      class_centroids=None, class_counts=None,
                      class_to_cluster=None, cluster_centroids=None,
                      cluster_dists=None, effective_tau=None,
                      return_sets=False, update_M_fn=None, yhat_mode="ncm",
                      allow_nonexchangeable=False, device="cpu", gpu_batch_size=128,
                      temperature=None, logit="cosine"):
    """Run FCP with MS-CS continuous penalty.

    Args:
        exchangeable: If True, update M and y_hat per candidate (x_test, yc)
            to restore exchangeability. Requires passing clustering artifacts.
        return_sets: If True, return (coverage, avg_size, prediction_sets)
            instead of (coverage, avg_size). Default False keeps the
            existing 2-tuple signature for all callers in this repo.
        update_M_fn: Optional callable (yc_idx, x_test, M_base) -> M for the
            exchangeable M update. When None (default), the built-in cluster
            update (update_M_for_candidate) is used, preserving existing
            callers. Pass update_centroid_M_for_candidate (curried) to run the
            cal-only centroid similarity matrix in exchangeable mode.
        yhat_mode: How to pick the suspected class ŷ(x) in the penalty
            lam*(1 - M[y, ŷ(x)]).
            "ncm" (default, variant A): ŷ = argmax_c top-k mean similarity to
                class c in the NCM's whitened space (the NCM numerator's own
                class prediction). More informative than 1-NN and reuses the
                NCM's neighbour computation — no extra distance matrices.
                Requires a GeodesicTopKMeanNCM; falls back to "1nn" otherwise.
            "1nn": legacy raw-Euclidean LOO 1-NN (kept for A/B comparison).
        allow_nonexchangeable: Approve (and silence the validity warning for) the
            non-exchangeable components: a whitened cal-fit NCM (ncm_name) and the
            fixed-ŷ/M penalty used when exchangeable=False. exchangeable=True with
            an unwhitened NCM (from an unlabeled-pool transform) is exact and needs
            no approval.
    """
    # The exchangeable=False MS-CS penalty freezes ŷ/M at fit() time rather than
    # recomputing them on the augmented bag per candidate -> O(1/n) break.
    if not exchangeable:
        warn_nonexchangeable(
            "MS-CS fixed-ŷ/M penalty (exchangeable=False)",
            "Pass exchangeable=True (bag-symmetric ŷ/M update) for an exact "
            "guarantee.",
            order="O(1/n)", allow=allow_nonexchangeable)
    ncm = create_ncm(ncm_name, k=5, allow_nonexchangeable=allow_nonexchangeable,
                     temperature=temperature, logit=logit)  # softmax NCMs: fixed T
    cp = FullConformalPredictor(ncm, alpha=alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)

    # GPU/torch fast path (set-parity with the loop below; ~7-30x faster). Covers
    # the canonical config (GeodesicTopKMeanNCM topk_same+topk_other, yhat_mode
    # "ncm", cluster-M built-in update or frozen penalty); falls back to the CPU
    # loop for anything else.
    if device == "cuda":
        try:
            from mscs_gpu import (run_mscs_torch, run_prototype_mscs_torch,
                                  MSCSGpuUnsupported)
            from conformal_prediction import PrototypeSoftmaxNCM
            gpu_fn = (run_prototype_mscs_torch
                      if isinstance(cp.ncm, PrototypeSoftmaxNCM) else run_mscs_torch)
            return gpu_fn(
                cp, X_cal, y_cal, X_test, y_test, all_classes, alpha, lam, M,
                exchangeable=exchangeable, yhat_mode=yhat_mode, update_M_fn=update_M_fn,
                class_centroids=class_centroids, class_counts=class_counts,
                class_to_cluster=class_to_cluster, cluster_centroids=cluster_centroids,
                cluster_dists=cluster_dists, effective_tau=effective_tau,
                device="cuda", batch_size=gpu_batch_size, return_sets=return_sets)
        except MSCSGpuUnsupported:
            pass  # unsupported config -> CPU loop below

    # Map class labels to indices in M (dict + vectorized lookup array;
    # labels are small non-negative ints, so an array map is exact and fast).
    class_to_idx = {int(c): i for i, c in enumerate(all_classes)}
    _maxlbl = int(max(int(c) for c in all_classes))
    lbl2col = np.zeros(_maxlbl + 1, dtype=np.int64)
    for c in all_classes:
        lbl2col[int(c)] = class_to_idx[int(c)]
    cal_row_idx = lbl2col[y_cal.astype(np.int64)]   # M row (true label) per cal pt

    # --- y_hat machinery: NCM-consistent (variant A) or legacy raw 1-NN ---
    n_cal = len(X_cal)
    use_ncm_yhat = (yhat_mode == "ncm" and hasattr(cp.ncm, "predict_class"))
    if use_ncm_yhat:
        cp.ncm._ensure_cal_yhat()
        cal_y_hat = cp.ncm.cal_y_hat            # (n_cal,) NCM-predicted class per cal pt
        D_test_cal = None                       # no raw distance matrices needed
        loo_nn_dists = None
    else:
        # Legacy raw-Euclidean LOO 1-NN over cal
        D_cal = euclidean_distances(X_cal, X_cal)
        np.fill_diagonal(D_cal, np.inf)
        loo_nn_idx = np.argmin(D_cal, axis=1)
        cal_y_hat = y_cal[loo_nn_idx]
        loo_nn_dists = D_cal[np.arange(n_cal), loo_nn_idx]
        D_test_cal = euclidean_distances(X_test, X_cal)

    # Base cal penalty (used when exchangeable=False, or as starting point)
    cal_penalty_base = lam * (1.0 - M[cal_row_idx, lbl2col[cal_y_hat.astype(np.int64)]])

    # --- Predict with MS-CS penalty on test scores ---
    n_test = len(X_test)
    prediction_sets = []

    for i in range(n_test):
        x_test = X_test[i]

        # y_hat for the test point
        if use_ncm_yhat:
            dists = None
            y_hat_test = cp.ncm.predict_class(x_test)   # variant A: NCM argmax
        else:
            dists = D_test_cal[i]                        # legacy raw 1-NN
            y_hat_test = int(y_cal[int(np.argmin(dists))])
        y_hat_idx = class_to_idx[y_hat_test]

        pred_set = []
        for yc in all_classes:
            yc = int(yc)
            yc_idx = class_to_idx[yc]

            if exchangeable:
                # --- Exchangeable version: update M and y_hat for augmented set ---
                # 1. Update M: shift class yc centroid to include x_test
                if update_M_fn is not None:
                    M_aug = update_M_fn(yc_idx, x_test, M)
                else:
                    M_aug = update_M_for_candidate(
                        class_centroids, class_counts, class_to_cluster,
                        cluster_centroids, cluster_dists, effective_tau,
                        yc_idx, x_test, M)

                # 2. Update cal y_hat for the augmented set {cal} u {(x_test, yc)}
                if use_ncm_yhat:
                    # variant A: x_test (label yc) can only raise each cal point's
                    # top-k sim to class yc; O(n_cal) exact, vectorized update.
                    cal_yhat_aug = cp.ncm.predict_class_augmented_cal(x_test, yc)
                    cal_penalty = lam * (1.0 - M_aug[
                        cal_row_idx, lbl2col[cal_yhat_aug.astype(np.int64)]])
                else:
                    # legacy raw 1-NN: x_test becomes cal_j's NN iff it is closer
                    cal_penalty = cal_penalty_base.copy()
                    for j in range(n_cal):
                        if dists[j] < loo_nn_dists[j]:
                            y_hat_j_idx = yc_idx
                        else:
                            y_hat_j_idx = class_to_idx[int(cal_y_hat[j])]
                        cj_idx = class_to_idx[int(y_cal[j])]
                        cal_penalty[j] = lam * (1.0 - M_aug[cj_idx, y_hat_j_idx])

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


def stratified_holdout_unlabeled(X, y, all_classes, n_unlabeled, rng):
    """Carve a DISJOINT, stratified unlabeled pool from the labeled data.

    Returns (X_pool, y_pool, X_unlabeled) where X_unlabeled shares NO index with
    X_pool, so a similarity matrix built on X_unlabeled never sees the cal/test
    points later drawn from X_pool. This fixes the prior fallback, which sampled
    the "unlabeled" pool from the full labeled set and thus overlapped cal/test
    (leakage into M). Stratified per the repo-wide sampling rule: ~equal samples
    held out per class, always leaving >=1 labeled sample per class.
    """
    n_classes = len(all_classes)
    per_class = max(1, n_unlabeled // n_classes)
    unl_idx = []
    for c in all_classes:
        c_idx = np.where(y == c)[0]
        take = min(per_class, len(c_idx) - 1)  # leave >=1 labeled per class
        if take > 0:
            unl_idx.append(rng.choice(c_idx, take, replace=False))
    unl_idx = np.concatenate(unl_idx)
    keep_mask = np.ones(len(X), dtype=bool)
    keep_mask[unl_idx] = False
    return X[keep_mask], y[keep_mask], X[unl_idx]


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
    parser.add_argument("--similarity", type=str, default="cluster",
                       choices=["cluster", "centroid"],
                       help="M construction: 'cluster' = k-means on unlabeled data "
                            "(default); 'centroid' = cal-only class-centroid "
                            "distances (NO unlabeled data). Run both under "
                            "identical splits to measure the unlabeled-data "
                            "contribution.")
    parser.add_argument("--yhat_mode", type=str, default="ncm",
                       choices=["ncm", "1nn"],
                       help="Suspected-class ŷ for the MS-CS penalty: 'ncm' "
                            "(variant A, argmax top-k NCM similarity; default) "
                            "or '1nn' (legacy raw-Euclidean 1-NN).")
    args = parser.parse_args()

    # Load labeled data
    data = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)
    X = data["embeddings"].numpy()
    y = data["labels"].numpy()
    all_classes = np.unique(y)
    print(f"Labeled data: {X.shape}, {len(all_classes)} classes")

    # Load or create unlabeled data. X_pool/y_pool is the ONLY data the trial
    # splits draw cal/test from — kept disjoint from X_unlabeled to avoid leakage.
    if args.unlabeled_path:
        udata = torch.load(args.unlabeled_path, map_location="cpu", weights_only=False)
        X_unlabeled = udata["embeddings"].numpy()
        X_pool, y_pool = X, y
        print(f"Unlabeled data (external): {X_unlabeled.shape}")
    elif args.similarity == "centroid":
        # Centroid mode uses no unlabeled data at all.
        X_unlabeled = None
        X_pool, y_pool = X, y
        print("Unlabeled data: none (centroid similarity uses cal centroids only)")
    else:
        # Carve a DISJOINT, stratified unlabeled pool from the labeled data so the
        # similarity matrix never sees cal/test points (fixes the old leakage).
        rng_u = np.random.default_rng(args.seed + 9999)
        n_unlabeled = min(len(X) // 2, 1500)
        X_pool, y_pool, X_unlabeled = stratified_holdout_unlabeled(
            X, y, all_classes, n_unlabeled, rng_u)
        print(f"Unlabeled data (disjoint stratified holdout): {X_unlabeled.shape}; "
              f"labeled pool for cal/test: {X_pool.shape}")

    print(f"NCM: {args.ncm}, alpha={args.alpha}, trials={args.n_trials}")
    print(f"Cluster counts: {args.n_clusters}")
    print(f"Tau values: {args.taus} {'(normalized by median_d^2)' if args.tau_normalize else '(absolute)'}")
    print(f"Lambda sweep: {args.lambdas}")
    print(f"Exchangeable: {args.exchangeable}")
    print(f"Similarity: {args.similarity}  |  y_hat mode: {args.yhat_mode}")
    print()

    for n_clust in args.n_clusters:
        # Centroid M ignores n_clusters; run it once (first value) to avoid repeats.
        if args.similarity == "centroid" and n_clust != args.n_clusters[0]:
            continue
        for tau_input in args.taus:
            # If normalizing, pass negative tau as signal to build function
            tau_arg = -tau_input if args.tau_normalize else tau_input

            print(f"{'='*70}")
            tau_label = f"tau={tau_input}*median_d^2" if args.tau_normalize else f"tau={tau_input}"
            if args.similarity == "centroid":
                mode_label = "centroid M (cal-only, no unlabeled)"
            else:
                mode_label = f"cluster M (k-means unlabeled), n_clusters={n_clust}"
            print(f"{mode_label}, {tau_label}")
            print(f"{'='*70}")

            for cal_size in args.cal_sizes:
                print(f"\n  cal_size={cal_size}")
                print(f"  {'-'*60}")

                for lam in args.lambdas:
                    covs, szs = [], []
                    t0 = time.time()

                    for trial in range(args.n_trials):
                        rng = np.random.default_rng(args.seed + trial * 1000)
                        # Identical splits across similarity modes (same seed),
                        # so cluster-vs-centroid is a controlled comparison.
                        X_cal, y_cal, X_test, y_test = stratified_split(
                            X_pool, y_pool, cal_size, args.test_size, all_classes, rng)

                        if lam == 0.0:
                            # Plain FCP (no penalty) — baseline
                            # approved: legacy whitened-NCM experiment (non-exchangeable)
                            ncm = create_ncm(args.ncm, k=5, allow_nonexchangeable=True)
                            cp = FullConformalPredictor(ncm, alpha=args.alpha)
                            cp.calibrate(X_cal, y_cal, all_classes=all_classes)
                            m = cp.evaluate(X_test, y_test, verbose=False)
                            covs.append(m["coverage"])
                            szs.append(m["avg_set_size"])
                        elif args.similarity == "centroid":
                            # Cal-only centroid-distance M (NO unlabeled data)
                            (M, eff_tau, med_d2, cls_centroids, cls_counts
                             ) = build_centroid_similarity_matrix(
                                X_cal, y_cal, all_classes, tau=tau_arg)
                            c2c = None  # not used in centroid mode
                            update_fn = None
                            if args.exchangeable:
                                update_fn = (lambda yc_idx, x_test, M_base:
                                             update_centroid_M_for_candidate(
                                                 cls_centroids, cls_counts, eff_tau,
                                                 yc_idx, x_test, M_base))
                            cov, sz = run_fcp_with_mscs(
                                X_cal, y_cal, X_test, y_test, all_classes,
                                args.ncm, args.alpha, lam, M,
                                exchangeable=args.exchangeable,
                                class_centroids=cls_centroids,
                                class_counts=cls_counts,
                                effective_tau=eff_tau,
                                update_M_fn=update_fn,
                                yhat_mode=args.yhat_mode,
                                allow_nonexchangeable=True)
                            covs.append(cov)
                            szs.append(sz)
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
                                effective_tau=eff_tau,
                                yhat_mode=args.yhat_mode,
                                allow_nonexchangeable=True)
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
        print(f"  similarity={args.similarity}, effective_tau={eff_tau:.4f}, median_d^2={med_d2:.4f}")
        print(f"  M shape: {M.shape}")
        print(f"  M diagonal: all 1.0? {np.allclose(np.diag(M), 1.0)}")
        print(f"  M symmetric? {np.allclose(M, M.T)}")
        print(f"  M range: [{M.min():.4f}, {M.max():.4f}]")
        print(f"  M mean off-diagonal: {(M.sum() - M.trace()) / (M.shape[0]**2 - M.shape[0]):.4f}")

        # Cluster assignment distribution (cluster mode only)
        if args.similarity == "cluster" and c2c is not None:
            unique_clusters = len(np.unique(c2c))
            print(f"  Classes mapped to {unique_clusters}/{n_clust} clusters")

    print("\nDone.")


if __name__ == "__main__":
    main()
