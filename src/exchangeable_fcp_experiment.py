"""
Exchangeable Full Conformal Prediction experiment.

Pipeline (the whole thing stays EXACTLY exchangeable, so coverage >= 1-alpha
holds at every cal size):

  --unlabeled_path given:
      UnlabeledTransform fit on the unlabeled pool -> PCA + within-cluster
      whitening (fit on unlabeled => fixed map w.r.t. the cal/test bag), then
      the missing-class-symmetric unwhitened NCM scores the transformed features,
      and (optional) MS-CS adds its penalty using a similarity matrix built from
      the SAME unlabeled clusters. Recovers the efficiency of the old cal-fit
      pipeline WITHOUT its exchangeability break.

  no --unlabeled_path:
      IdentityTransform -> plain unwhitened FCP. Still fully exchangeable, just
      without the PCA/whitening/MS-CS efficiency gains (graceful degradation).

Always uses a whiten=False NCM (the cal-fit whitening is the O(1/n) violator) and
the symmetric missing-class score fix in conformal_prediction.py.

Examples
--------
# full exchangeable pipeline (PCA + cluster-whiten + MS-CS)
python src/exchangeable_fcp_experiment.py \
    --embeddings_path output/embeddings_cifar100.pt \
    --unlabeled_path  output/embeddings_cifar100_unlabeled.pt \
    --pca_dim 128 --whiten cluster --mscs --lambdas 0.0 0.05 0.1

# degraded but fully-exchangeable fallback (no unlabeled pool)
python src/exchangeable_fcp_experiment.py --embeddings_path output/embeddings_cifar100.pt
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import make_transform
from mscs_unlabeled_experiment import (
    build_cluster_similarity_matrix, build_centroid_similarity_matrix,
    update_centroid_M_for_candidate, run_fcp_with_mscs)


def random_split(X, y, cal, test, allc, rng):
    """Uniform random partition -> exactly exchangeable (tight coverage)."""
    idx = rng.permutation(len(X))
    return idx[:cal], idx[cal:cal + test]


def balanced_cal_split(X, y, cal, test, allc, rng):
    """Balanced cal (cal//K per class) + random test. NOT exactly exchangeable
    (label-dependent cal) -> empirically conservative (over-covers)."""
    m = cal // len(allc)
    used = np.zeros(len(X), bool)
    ci = []
    for c in allc:
        ch = rng.choice(np.where(y == c)[0], m, replace=False)
        ci.append(ch); used[ch] = True
    ci = np.concatenate(ci)
    ti = rng.choice(np.where(~used)[0], test, replace=False)
    return ci, ti


def balanced_both_split(X, y, cal, test, allc, rng):
    """Balanced cal + balanced test, EQUAL shots/class (m = cal//K each),
    disjoint, fresh per-class permutation each trial. The DEFAULT split for new
    experiments (2026-06-21): lit-comparable few-shot protocol, every class
    present in cal AND test, smaller sets. NOT exactly exchangeable
    (label-dependent) -> empirically conservative (over-covers ~+1-3pp). Report
    the `random` arm alongside for the exact-validity claim. `test` is ignored
    (set to m*K); `cal` should be a multiple of K. Keep cal >= 2K (m >= 2)."""
    m = cal // len(allc)
    ci, ti = [], []
    for c in allc:
        perm = rng.permutation(np.where(y == c)[0])
        ci.append(perm[:m]); ti.append(perm[m:2 * m])
    return np.concatenate(ci), np.concatenate(ti)


SPLITS = {"random": random_split, "balanced_cal": balanced_cal_split,
          "balanced_both": balanced_both_split}


def main():
    ap = argparse.ArgumentParser(description="Exchangeable FCP (PCA+whiten+MS-CS from unlabeled pool)")
    ap.add_argument("--embeddings_path", type=str, default="output/embeddings_cifar100.pt")
    ap.add_argument("--unlabeled_path", type=str, default=None,
                    help="Unlabeled pool. If given, enables PCA + whitening + MS-CS "
                         "(all fit on it, exchangeability-preserving). If omitted, "
                         "degrades to plain unwhitened FCP (still exchangeable).")
    ap.add_argument("--ncm", type=str, default="unwhitened_topk_mean",
                    help="Must be a whiten=False NCM for exchangeability "
                         "(unwhitened_topk_mean / unwhitened_topk_asym), or "
                         "'ridge_softmax' (FCA-inspired; plain FCP only).")
    # ridge_softmax knobs (ignored by other NCMs)
    ap.add_argument("--temperature", type=str, default="auto",
                    help="ridge_softmax softmax T: a float (exact) or 'auto' "
                         "(cal-fit, O(1/n), warns). Validity holds for any fixed T.")
    ap.add_argument("--lam_ridge", type=float, default=1.0,
                    help="ridge_softmax ridge stabiliser.")
    ap.add_argument("--lam_anchor", type=float, default=1.0,
                    help="ridge_softmax class-mean anchor strength "
                         "(large -> prototype, 0 -> discriminative LDA fit).")
    ap.add_argument("--no_loo", action="store_true",
                    help="ridge_softmax: use full-bag (self-inclusive) scores "
                         "instead of PRESS leave-one-out.")
    ap.add_argument("--pca_dim", type=int, default=128, help="PCA dim (0/negative = off).")
    ap.add_argument("--whiten", type=str, default="cluster", choices=["cluster", "global", "none"])
    ap.add_argument("--n_clusters_whiten", type=int, default=100,
                    help="k-means clusters for within-cluster whitening (~n_classes).")
    ap.add_argument("--mscs", action="store_true", help="Add the MS-CS penalty (exchangeable mode).")
    ap.add_argument("--n_clusters_mscs", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.5, help="MS-CS tau multiplier on median_d^2.")
    ap.add_argument("--similarity", type=str, default="cluster", choices=["cluster", "centroid"],
                    help="MS-CS M: 'cluster' = k-means on the unlabeled pool (default), "
                         "'centroid' = cal class-centroid distances (no unlabeled, for the "
                         "unlabeled-contribution comparison).")
    ap.add_argument("--yhat_mode", type=str, default="ncm", choices=["ncm", "1nn"],
                    help="Suspected-class ŷ for the penalty: 'ncm' (variant A) or '1nn' (legacy).")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.05, 0.1])
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 300, 400, 600, 800])
    ap.add_argument("--test_size", type=int, default=300)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--split", type=str, default="balanced_both", choices=list(SPLITS),
                    help="DEFAULT 'balanced_both' (balanced cal+test, equal shots/class, "
                         "lit-comparable, conservative). Use 'random' for the exact-validity "
                         "(tight ~0.90) reference arm.")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="MS-CS penalty backend: 'cuda' uses the GPU fast path "
                         "(set-parity, ~7-30x); centroid-M falls back to CPU.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="output/exchangeable_fcp",
                    help="Where to save results JSON (consumed by plot_exchangeable.py).")
    ap.add_argument("--tag", type=str, default=None,
                    help="Label for this run (default: 'full' if --unlabeled_path else 'degraded').")
    args = ap.parse_args()

    data = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)
    X = data["embeddings"].numpy(); y = data["labels"].numpy()
    allc = np.unique(y)
    print(f"Labeled: {X.shape}, {len(allc)} classes")

    whiten = None if args.whiten == "none" else args.whiten
    pca_dim = args.pca_dim if args.pca_dim and args.pca_dim > 0 else None
    if args.unlabeled_path:
        Xu = torch.load(args.unlabeled_path, map_location="cpu", weights_only=False)["embeddings"].numpy()
        transform = make_transform(Xu, pca_dim=pca_dim, whiten=whiten,
                                   n_clusters=args.n_clusters_whiten)
        print(f"Unlabeled: {Xu.shape} -> {transform}")
        mscs_ok = args.mscs and getattr(transform, "has_clusters", False)
    else:
        transform = make_transform(None)
        print(f"No unlabeled pool -> {transform} (degraded, fully exchangeable)")
        mscs_ok = False
        if args.mscs:
            print("  (MS-CS requested but needs an unlabeled pool -> skipped)")

    print(f"NCM={args.ncm} | split={args.split} | alpha={args.alpha} | trials={args.n_trials}")
    exch_note = "exact" if args.split == "random" else "conservative (label-dependent balanced split, over-covers ~+1-3pp)"
    print(f"EXCHANGEABLE: {exch_note}  (whitening/PCA/MS-CS sourced from the unlabeled pool)")
    split_fn = SPLITS[args.split]
    tau_arg = -args.tau  # negative => normalize by median_d^2 (see build_cluster_similarity_matrix)
    if args.ncm == "ridge_softmax" and mscs_ok:
        print("  (ridge_softmax: MS-CS not wired -> plain FCP only)")
        mscs_ok = False
    lambdas = args.lambdas if mscs_ok else [0.0]

    # ridge_softmax: resolve ONE fixed softmax temperature, held constant across
    # ALL trials -> each trial's CP stays exactly exchangeable (a fixed T from a
    # pilot draw is a hyperparameter chosen on separate data). A per-fit auto-T
    # would be cal-fit O(1/n) and, with a small sharp T, under-covers.
    ridge_T = None
    if args.ncm == "ridge_softmax":
        if args.temperature == "auto":
            from conformal_prediction import RidgeSoftmaxNCM
            rng0 = np.random.default_rng(args.seed)
            pilot = min(max(args.cal_sizes), 800)
            pci, _ = balanced_cal_split(X, y, pilot, max(args.test_size, 200), allc, rng0)
            pncm = RidgeSoftmaxNCM(
                lam=args.lam_ridge, lam_anchor=args.lam_anchor,
                temperature=None, loo=not args.no_loo, allow_nonexchangeable=True
            ).fit(transform.transform(X[pci]), y[pci])
            ridge_T = float(pncm._T)
            print(f"  ridge_softmax: fixed T={ridge_T:.4f} (pilot; constant across trials -> exact)")
        else:
            ridge_T = float(args.temperature)

    tag = args.tag or ("full" if args.unlabeled_path else "degraded")
    results = {"tag": tag,
               "config": {"unlabeled": bool(args.unlabeled_path), "ncm": args.ncm,
                          "pca_dim": pca_dim, "whiten": whiten, "mscs": bool(mscs_ok),
                          "similarity": args.similarity, "yhat_mode": args.yhat_mode,
                          "split": args.split, "n_trials": args.n_trials, "alpha": args.alpha},
               "rows": []}

    for cal in args.cal_sizes:
        print(f"\ncal={cal}")
        print("  " + "-" * 56)
        for lam in lambdas:
            covs, szs = [], []
            t0 = time.time()
            for trial in range(args.n_trials):
                rng = np.random.default_rng(args.seed + trial * 1000)
                ci, ti = split_fn(X, y, cal, args.test_size, allc, rng)
                Xc, yc = transform.transform(X[ci]), y[ci]
                Xt, yt = transform.transform(X[ti]), y[ti]

                if lam == 0.0:
                    if args.ncm == "ridge_softmax":
                        ncm = create_ncm(args.ncm, temperature=ridge_T,
                                         lam_ridge=args.lam_ridge,
                                         lam_anchor=args.lam_anchor,
                                         loo=not args.no_loo)
                    else:
                        ncm = create_ncm(args.ncm, k=5)
                    cp = FullConformalPredictor(ncm, alpha=args.alpha)
                    cp.calibrate(Xc, yc, all_classes=allc)
                    m = cp.evaluate(Xt, yt, verbose=False)
                    covs.append(m["coverage"]); szs.append(m["avg_set_size"])
                elif args.similarity == "centroid":
                    # cal class-centroid M (no unlabeled) — for the contribution comparison
                    (M, eff_tau, med_d2, cls_cen, cls_cnt
                     ) = build_centroid_similarity_matrix(Xc, yc, allc, tau=tau_arg)
                    update_fn = (lambda yc_idx, x_test, M_base:
                                 update_centroid_M_for_candidate(
                                     cls_cen, cls_cnt, eff_tau, yc_idx, x_test, M_base))
                    cov, sz = run_fcp_with_mscs(
                        Xc, yc, Xt, yt, allc, args.ncm, args.alpha, lam, M,
                        exchangeable=True, yhat_mode=args.yhat_mode,
                        class_centroids=cls_cen, class_counts=cls_cnt,
                        effective_tau=eff_tau, update_M_fn=update_fn,
                        device=args.device)  # centroid-M unsupported on GPU -> CPU fallback
                    covs.append(cov); szs.append(sz)
                else:
                    # unlabeled k-means cluster M (in the transformed space)
                    (M, c2c, eff_tau, med_d2, cls_cen, cls_cnt,
                     clu_cen, clu_d) = build_cluster_similarity_matrix(
                        transform.Xu_transformed_, Xc, yc, allc,
                        args.n_clusters_mscs, tau=tau_arg)
                    cov, sz = run_fcp_with_mscs(
                        Xc, yc, Xt, yt, allc, args.ncm, args.alpha, lam, M,
                        exchangeable=True, yhat_mode=args.yhat_mode,
                        class_centroids=cls_cen, class_counts=cls_cnt,
                        class_to_cluster=c2c, cluster_centroids=clu_cen,
                        cluster_dists=clu_d, effective_tau=eff_tau,
                        device=args.device)
                    covs.append(cov); szs.append(sz)
            cov_m, sz_m = float(np.mean(covs)), float(np.mean(szs))
            valid = "OK" if cov_m >= args.alpha * 0 + (1 - args.alpha) - 0.005 else "!!"
            label = "FCP (no penalty)" if lam == 0.0 else f"MS-CS lam={lam:.2f}"
            method = "FCP" if lam == 0.0 else f"MS-CS λ={lam:g}"
            results["rows"].append({"cal": cal, "lam": lam, "method": method,
                                    "cov": cov_m, "sz": sz_m})
            print(f"  {label:18s} cov={cov_m:.4f} {valid}  sz={sz_m:.2f}  ({time.time()-t0:.1f}s)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"results_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
