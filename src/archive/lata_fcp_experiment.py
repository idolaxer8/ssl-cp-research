"""
LATA-style Graph-Smoothed Full Conformal Prediction.

Uses unlabeled data as geometric bridges in a kNN graph to smooth
nonconformity scores, improving discrimination at small calibration sizes.

Design (exchangeability-preserving):
  1. Build kNN graph on ALL embeddings: cal ∪ unlabeled ∪ test (batch mode)
     - Graph is entirely LABEL-FREE (cosine similarity on embeddings)
     - Graph is built ONCE before the FCP loop
  2. Standard FCP computes raw scores (NCM fitted on cal only)
  3. Scores are smoothed via iterative diffusion on the graph:
     - Scored nodes (cal + current test) are anchored to raw scores
     - Unlabeled nodes propagate information as geometric bridges
     - After T iterations, scored nodes receive smoothed values
  4. P-values computed on smoothed scores

Exchangeability argument:
  - The graph G is a deterministic function of embeddings (no labels)
  - The smoothing operator W(G) is a fixed matrix, applied identically
    to all scored nodes (cal and test)
  - Therefore s_smooth = f(s_raw, W) is still symmetric in the
    exchangeable bag {z_1,...,z_n, z_{n+1}}
  - Same order of approximation as whitening: O(1/n)

References:
  - Bozorgtabar et al. (2026) "LATA" -- kNN graph smoothing (arXiv:2602.17535)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
from conformal_prediction import (
    create_ncm, stratified_cal_test_split,
)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_knn_graph(X_all, k_graph=10):
    """
    Build a symmetric, row-normalized kNN graph on embeddings.

    Args:
        X_all: (N, d) all embeddings (cal + unlabeled + test)
        k_graph: number of nearest neighbors

    Returns:
        W: (N, N) sparse row-normalized adjacency matrix
    """
    # kneighbors_graph with cosine metric returns distances (1 - cosine_sim)
    G = kneighbors_graph(X_all, n_neighbors=k_graph, mode='distance',
                         metric='cosine', n_jobs=-1)
    # Convert distances to similarities
    G.data = np.maximum(1.0 - G.data, 0.0)
    # Symmetrize
    G = (G + G.T) / 2
    G.eliminate_zeros()
    # Remove self-loops
    G.setdiag(0)
    G.eliminate_zeros()
    # Row-normalize
    row_sums = np.array(G.sum(axis=1)).flatten()
    D_inv = sp.diags(1.0 / np.maximum(row_sums, 1e-10))
    W = D_inv @ G
    return W.tocsr()


# ---------------------------------------------------------------------------
# Score smoothing via graph diffusion
# ---------------------------------------------------------------------------

def smooth_scores(W, raw_scores, scored_indices, gamma, T):
    """
    Smooth scores via iterative diffusion on the graph.

    Scored nodes are anchored to their raw scores with weight (1-gamma).
    Unlabeled nodes freely propagate information as geometric bridges.

    Args:
        W: (N, N) sparse row-normalized adjacency
        raw_scores: dict mapping node_index -> raw_score (only scored nodes)
        scored_indices: array of indices that have scores
        gamma: smoothing strength (0 = no smoothing, 1 = full graph average)
        T: number of diffusion steps

    Returns:
        smoothed: dict mapping node_index -> smoothed_score
    """
    n_total = W.shape[0]
    s = np.zeros(n_total)
    s0 = np.zeros(n_total)

    scored_mask = np.zeros(n_total, dtype=bool)
    for idx in scored_indices:
        s[idx] = raw_scores[idx]
        s0[idx] = raw_scores[idx]
        scored_mask[idx] = True

    for t in range(T):
        s_new = W @ s
        # Scored nodes: anchor to raw + blend with diffused
        s[scored_mask] = (1 - gamma) * s0[scored_mask] + gamma * s_new[scored_mask]
        # Unlabeled nodes: freely adopt diffused values
        s[~scored_mask] = s_new[~scored_mask]

    return {idx: s[idx] for idx in scored_indices}


# ---------------------------------------------------------------------------
# LATA-smoothed FCP prediction
# ---------------------------------------------------------------------------

def lata_fcp_predict(ncm, W, n_cal, test_node_indices, X_test, y_test,
                     all_classes, alpha, gamma, T, desc="LATA FCP"):
    """
    FCP prediction with LATA-style graph smoothing.

    Args:
        ncm: fitted NCM (on cal data only)
        W: (N, N) sparse kNN graph adjacency (cal + unlabeled + test)
        n_cal: number of calibration points (first n_cal nodes in graph)
        test_node_indices: array mapping test[i] -> graph node index
        X_test, y_test: test data
        all_classes: candidate labels
        alpha: significance level
        gamma: smoothing strength
        T: diffusion steps
    """
    n_test = len(X_test)
    prediction_sets = []
    cal_indices = np.arange(n_cal)

    iterator = tqdm(range(n_test), desc=desc)
    for i in iterator:
        x_test = X_test[i]
        test_idx = test_node_indices[i]
        pred_set = []

        for y_c in all_classes:
            yc = int(y_c)

            # Standard FCP raw scores
            test_score = ncm.score_x(x_test, yc)
            updated_cal_scores = ncm.updated_calibration_scores_for(x_test, yc)

            # Build raw score map for smoothing
            raw_scores = {}
            for j in range(n_cal):
                raw_scores[j] = updated_cal_scores[j]
            raw_scores[test_idx] = test_score

            scored_idx = np.append(cal_indices, test_idx)

            # Smooth
            smoothed = smooth_scores(W, raw_scores, scored_idx, gamma, T)

            # P-value on smoothed scores
            smoothed_cal = np.array([smoothed[j] for j in range(n_cal)])
            smoothed_test = smoothed[test_idx]

            n_geq = np.sum(smoothed_cal >= smoothed_test)
            p_value = (n_geq + 1) / (n_cal + 1)

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
        description="LATA-style graph-smoothed FCP")
    p.add_argument("--cal_embeddings", default="output/embeddings_cifar100.pt")
    p.add_argument("--test_embeddings", default="output/embeddings_cifar100_test.pt")
    p.add_argument("--unlabeled_embeddings",
                   default="output/embeddings_cifar100_unlabeled.pt")
    p.add_argument("--ncm", default="geodesic_topk_mean")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--cal_sizes", type=int, nargs="+", default=[600])
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--k_graph", type=int, nargs="+", default=[10],
                   help="kNN graph neighbor counts to sweep")
    p.add_argument("--gamma", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                   help="Smoothing strength (0=none, 1=full graph)")
    p.add_argument("--T", type=int, default=3,
                   help="Number of diffusion steps")
    p.add_argument("--n_trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
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
    print(f"Unlabeled pool: {X_unl_all.shape}")

    print(f"\nNCM: {args.ncm}, alpha={args.alpha}, T={args.T}, "
          f"trials={args.n_trials}")
    print(f"k_graph sweep: {args.k_graph}, gamma sweep: {args.gamma}")

    # ---- Run experiments ----
    for cal_size in args.cal_sizes:
        print(f"\n{'='*70}")
        print(f"cal_size = {cal_size}  ({cal_size // n_classes}/class)")
        print(f"{'='*70}")

        # Results: (k_graph, gamma) -> list of trial metrics
        all_results = {}
        all_results[(0, 0.0)] = []  # baseline

        for trial in range(args.n_trials):
            seed = args.seed + trial * 1000
            rng = np.random.default_rng(seed)

            # Balanced cal split
            X_cal, y_cal, _, _ = stratified_cal_test_split(
                X_cal_pool, y_cal_pool,
                cal_size=cal_size, test_size=0,
                balanced=True, random_state=seed)

            # Test from separate pool (balanced)
            test_per_class = args.test_size // n_classes
            test_idx = []
            for c in all_classes:
                avail = rng.permutation(np.where(y_test_pool == c)[0])
                test_idx.append(avail[:test_per_class])
            test_idx = np.concatenate(test_idx)
            X_test = X_test_pool[test_idx]
            y_test = y_test_pool[test_idx]

            n_cal = len(X_cal)
            n_unl = len(X_unl_all)
            n_test = len(X_test)

            # ---- Fit NCM on cal only ----
            ncm = create_ncm(args.ncm, k=5)
            ncm.fit(X_cal, y_cal)

            # ---- Baseline FCP (no smoothing) ----
            t0 = time.time()
            m_base = lata_fcp_predict(
                ncm, sp.eye(n_cal + n_unl + n_test, format='csr'),
                n_cal,
                np.arange(n_cal + n_unl, n_cal + n_unl + n_test),
                X_test, y_test, all_classes, args.alpha,
                gamma=0.0, T=0, desc=f"Base t{trial}")
            m_base["time"] = time.time() - t0
            all_results[(0, 0.0)].append(m_base)

            # ---- Build graph (once per trial, includes test) ----
            for k_graph in args.k_graph:
                X_all = np.concatenate([X_cal, X_unl_all, X_test], axis=0)
                print(f"  Building kNN graph (k={k_graph}, "
                      f"N={len(X_all)})...", end=" ", flush=True)
                t_graph = time.time()
                W = build_knn_graph(X_all, k_graph=k_graph)
                print(f"{time.time() - t_graph:.1f}s, "
                      f"nnz={W.nnz}")

                test_node_indices = np.arange(n_cal + n_unl,
                                              n_cal + n_unl + n_test)

                # ---- Sweep gamma ----
                for gamma in args.gamma:
                    key = (k_graph, gamma)
                    if key not in all_results:
                        all_results[key] = []

                    t0 = time.time()
                    m_lata = lata_fcp_predict(
                        ncm, W, n_cal, test_node_indices,
                        X_test, y_test, all_classes, args.alpha,
                        gamma=gamma, T=args.T,
                        desc=f"LATA(k={k_graph},g={gamma}) t{trial}")
                    m_lata["time"] = time.time() - t0
                    all_results[key].append(m_lata)

        # ---- Print results table ----
        print(f"\n  Results (cal={cal_size}, {args.n_trials} trials, "
              f"T={args.T}):\n")
        print(f"  {'Method':<30s}  {'Coverage':>14s}  {'Set Size':>14s}  "
              f"{'Time':>6s}")
        print(f"  {'-'*30}  {'-'*14}  {'-'*14}  {'-'*6}")

        baseline_sz = np.mean([r["avg_set_size"]
                               for r in all_results[(0, 0.0)]])

        for key in sorted(all_results.keys()):
            trials = all_results[key]
            if not trials:
                continue
            covs = [r["coverage"] for r in trials]
            szs = [r["avg_set_size"] for r in trials]
            times = [r["time"] for r in trials]

            if key == (0, 0.0):
                label = "Baseline (no smoothing)"
            else:
                label = f"LATA(k={key[0]}, g={key[1]:.1f})"
            valid = "OK" if np.mean(covs) >= 0.89 else "!!"

            delta = ""
            if key != (0, 0.0):
                pct = (baseline_sz - np.mean(szs)) / baseline_sz * 100
                delta = f" ({pct:+.1f}%)"

            print(f"  {label:<30s}  "
                  f"{np.mean(covs):.3f}+/-{np.std(covs):.3f} {valid}  "
                  f"{np.mean(szs):.2f}+/-{np.std(szs):.2f}{delta:<8s}  "
                  f"{np.mean(times):5.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
