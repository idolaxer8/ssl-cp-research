"""
MS-CS penalty effect on prototype_softmax (vanilla FCA) and the geodesic NCMs,
under exchangeable Full CP.

Question this answers: on a HARD dataset (e.g. FGVC-Aircraft) where prototype_softmax
already gives the smallest sets but those sets are still large (sz ~ 13-21 / 100),
does the MS-CS cluster-similarity penalty shrink them further while keeping coverage?

Pipeline (exactly exchangeable):
    DINOv2 -> pool-fit PCA + within-cluster whitening (UnlabeledTransform, fit on the
    unlabeled pool) -> NCM -> Full CP -> MS-CS penalty s_l(x,y)=s(x,y)+l*(1-M[y,yhat]),
    with M built from k-means on the SAME unlabeled pool and updated per candidate
    (exchangeable=True). Softmax NCMs use ONE pilot-fixed temperature held constant
    across all trials (a fixed hyperparameter -> each trial's CP stays exactly
    exchangeable; an auto/per-fit T would be cal-fit O(1/n)).

For each (cal, ncm, lambda): marginal coverage, avg set size, and the % set-size
reduction vs lambda=0 (same NCM, same splits). lambda=0 is the no-penalty Full-CP
baseline. Validity holds at every lambda; this measures efficiency.

Expects output/embeddings_<ds>.pt and output/embeddings_<ds>_unlabeled.pt
(keys {"embeddings","labels"} for the labeled file).

Examples
--------
# Aircraft, prototype_softmax lambda sweep (cluster, GPU):
python src/mscs_softmax_experiment.py --dataset aircraft --device cuda \
    --ncms prototype_softmax --cal_sizes 400 800 1200 --lambdas 0.0 0.05 0.1 0.2 0.4

# also run the geodesic NCMs under the same penalty, for comparison:
python src/mscs_softmax_experiment.py --dataset aircraft \
    --ncms prototype_softmax unwhitened_topk_asym unwhitened_topk_mean
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import FullConformalPredictor, create_ncm, PrototypeSoftmaxNCM
from exchangeable_features import make_transform
from mscs_unlabeled_experiment import (
    build_cluster_similarity_matrix, run_fcp_with_mscs,
    build_centroid_similarity_matrix, update_centroid_M_for_candidate,
    build_prototype_similarity_matrix, update_prototype_M_for_candidate,
    build_centered_cosine_similarity_matrix, update_centered_cosine_M_for_candidate,
)

SOFTMAX_NCMS = {"prototype_softmax"}   # ridge_softmax MS-CS is not wired -> excluded


def build_M_and_run_kwargs(similarity, Xu_t, Xc, yc, allc, args, tau_arg):
    """Build the MS-CS similarity matrix M + the run_fcp_with_mscs kwargs for the
    chosen --similarity mode. All three modes are exactly exchangeable here
    (per-candidate M update); they differ only in how class similarity is defined:
      cluster   : k-means on the unlabeled pool (Fargion-style; needs Xu_t).
      centroid  : cal class-centroid Gaussian kernel (cal-only, no pool).
      prototype : cosine between class-mean prototypes (softmax-NATIVE; the
                  prototype NCM's own logit geometry, M = Gram of its prototypes).
      centered_cosine : FAITHFUL Fargion MS (their §5): cosine of the CENTERED
                  class means, cos(h_c-h_G, h_c'-h_G), h_c from cal (score-aligned),
                  h_G = prepped-pool global mean (needs Xu_t; bag-independent -> the
                  update stays exactly exchangeable).
    Returns (M, run_kwargs) where run_kwargs feeds run_fcp_with_mscs(... **kwargs).
    """
    if similarity == "cluster":
        (M, c2c, eff_tau, med_d2, cc, cn, uc, ud) = build_cluster_similarity_matrix(
            Xu_t, Xc, yc, allc, args.n_clusters_mscs, tau=tau_arg)
        return M, dict(similarity="cluster", class_centroids=cc, class_counts=cn,
                       class_to_cluster=c2c, cluster_centroids=uc, cluster_dists=ud,
                       effective_tau=eff_tau)
    if similarity == "centroid":
        (M, eff_tau, med_d2, cc, cn) = build_centroid_similarity_matrix(
            Xc, yc, allc, tau=tau_arg)
        upd = (lambda yc_idx, x, Mb: update_centroid_M_for_candidate(
            cc, cn, eff_tau, yc_idx, x, Mb))
        return M, dict(similarity="centroid", class_centroids=cc, class_counts=cn,
                       effective_tau=eff_tau, update_M_fn=upd)
    if similarity == "prototype":
        cosine = (args.logit == "cosine")
        M, csum, cnt, Pn = build_prototype_similarity_matrix(
            Xc, yc, allc, cosine=cosine)
        upd = (lambda yc_idx, x, Mb: update_prototype_M_for_candidate(
            csum, cnt, Pn, cosine, yc_idx, x, Mb))
        return M, dict(similarity="prototype", update_M_fn=upd)
    if similarity == "centered_cosine":
        cosine = (args.logit == "cosine")
        clipneg = getattr(args, "clip_negative", False)
        M, csum, cnt, Vu, hG = build_centered_cosine_similarity_matrix(
            Xc, yc, allc, Xu_t, cosine=cosine, clip_negative=clipneg)
        upd = (lambda yc_idx, x, Mb: update_centered_cosine_M_for_candidate(
            csum, cnt, Vu, hG, cosine, yc_idx, x, Mb, clip_negative=clipneg))
        return M, dict(similarity="centered_cosine", update_M_fn=upd)
    raise ValueError(f"unknown similarity '{similarity}'")


def balanced_split(y, allc, m_cal, m_test, rng):
    """Balanced cal (m_cal/class) + balanced test (m_test/class), disjoint."""
    ci, ti = [], []
    for c in allc:
        perm = rng.permutation(np.where(y == c)[0])
        ci.append(perm[:m_cal])
        ti.append(perm[m_cal:m_cal + m_test])
    return np.concatenate(ci), np.concatenate(ti)


def load_dataset(data_dir, ds):
    lab = os.path.join(data_dir, f"embeddings_{ds}.pt")
    unl = os.path.join(data_dir, f"embeddings_{ds}_unlabeled.pt")
    if not os.path.exists(lab):
        return None
    d = torch.load(lab, map_location="cpu", weights_only=False)
    X, y = d["embeddings"].numpy(), d["labels"].numpy()
    if not os.path.exists(unl):
        return None
    Xu = torch.load(unl, map_location="cpu", weights_only=False)["embeddings"].numpy()
    return X, y, Xu


def pilot_temperature(transform, X, y, allc, args):
    """ONE pilot-fixed prototype_softmax T, constant across trials -> exchangeable.
    Piloted from a stable larger-m balanced draw with a fixed seed. A float
    --proto_temperature overrides the pilot."""
    if str(args.proto_temperature) != "auto":
        return float(args.proto_temperature)
    K = len(allc)
    m = max(4, min(max(args.cal_sizes), 800) // K)
    rng = np.random.default_rng(args.seed)
    ci, _ = balanced_split(y, allc, m, 1, rng)
    p = PrototypeSoftmaxNCM(temperature=None, logit=args.logit,
                            allow_nonexchangeable=True).fit(transform.transform(X[ci]), y[ci])
    return float(p._T)


def main():
    ap = argparse.ArgumentParser(
        description="MS-CS penalty (lambda) sweep on prototype_softmax + geodesic NCMs "
                    "under exchangeable Full CP.")
    ap.add_argument("--dataset", default="aircraft")
    ap.add_argument("--data_dir", default="output",
                    help="dir with embeddings_<ds>.pt and _unlabeled.pt")
    ap.add_argument("--ncms", nargs="+", default=["prototype_softmax"],
                    help="NCMs to sweep (prototype_softmax / unwhitened_topk_asym / "
                         "unwhitened_topk_mean).")
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[400, 600, 800, 1000, 1200])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--test_per_class", type=int, default=10)
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--whiten", default="cluster", choices=["cluster", "global", "none"])
    ap.add_argument("--n_clusters_whiten", type=int, default=100)
    ap.add_argument("--n_clusters_mscs", type=int, default=20)
    ap.add_argument("--similarity", default="cluster",
                    choices=["cluster", "centroid", "prototype", "centered_cosine"],
                    help="MS-CS class-similarity M: 'cluster' (k-means on the "
                         "unlabeled pool, default), 'centroid' (cal class-centroid "
                         "Gaussian kernel, no pool), 'prototype' (softmax-native "
                         "cosine between class-mean prototypes; M = the prototype "
                         "NCM's own logit Gram, label-free, exactly exchangeable), "
                         "or 'centered_cosine' (FAITHFUL Fargion MS §5: cosine of "
                         "CENTERED class means cos(h_c-h_G, h_c'-h_G); h_G = "
                         "prepped-pool global mean, needs the unlabeled pool).")
    ap.add_argument("--clip_negative", action="store_true",
                    help="centered_cosine/prototype: fold anti-correlated classes "
                         "(M<0) to 0 (ablation; default off = signed [-1,1], faithful).")
    ap.add_argument("--tau", type=float, default=0.5, help="MS-CS tau multiplier on median_d^2.")
    ap.add_argument("--yhat_mode", default="ncm", choices=["ncm", "1nn"])
    ap.add_argument("--logit", default="cosine", choices=["cosine", "dot"])
    ap.add_argument("--proto_temperature", default="auto",
                    help="prototype_softmax T: a float (exact) or 'auto' (pilot-fixed).")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--output_dir", default="output/mscs_softmax")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but CUDA unavailable -> cpu")
        args.device = "cpu"

    loaded = load_dataset(args.data_dir, args.dataset)
    if loaded is None:
        print(f"[error] need {args.data_dir}/embeddings_{args.dataset}.pt and "
              f"_unlabeled.pt")
        sys.exit(1)
    X, y, Xu = loaded
    allc = np.unique(y)
    K = len(allc)
    print(f"{args.dataset}: X={X.shape}, K={K}, unlabeled={Xu.shape}, device={args.device}")

    whiten = None if args.whiten == "none" else args.whiten
    transform = make_transform(Xu, pca_dim=args.pca_dim, whiten=whiten,
                               n_clusters=args.n_clusters_whiten)
    print(f"transform: {transform}")

    T_map = {}
    for nm in args.ncms:
        if nm in SOFTMAX_NCMS:
            T_map[nm] = pilot_temperature(transform, X, y, allc, args)
            print(f"  {nm} fixed T = {T_map[nm]:.4f}")

    tau_arg = -args.tau  # negative -> normalize by median_d^2 (build_cluster_similarity_matrix)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for cal in args.cal_sizes:
        m = cal // K
        if m < 2:
            print(f"[skip] cal={cal}: m={m} < 2 (need cal >= 2K={2*K})")
            continue
        print(f"\ncal={cal} (m={m}/class)")
        for ncm_name in args.ncms:
            is_soft = ncm_name in SOFTMAX_NCMS
            base_sz = None
            for lam in args.lambdas:
                covs, szs = [], []
                t0 = time.time()
                for t in range(args.n_trials):
                    rng = np.random.default_rng(args.seed + 1000 * t)
                    ci, ti = balanced_split(y, allc, m, args.test_per_class, rng)
                    Xc, yc = transform.transform(X[ci]), y[ci]
                    Xt, yt = transform.transform(X[ti]), y[ti]
                    if lam == 0.0:
                        ncm = (create_ncm(ncm_name, temperature=T_map[ncm_name], logit=args.logit)
                               if is_soft else create_ncm(ncm_name, k=5))
                        cp = FullConformalPredictor(ncm, alpha=args.alpha)
                        cp.calibrate(Xc, yc, all_classes=allc)
                        # GPU fast path for the no-penalty baseline (prototype_softmax +
                        # geodesics both have one); compute cov/size manually like
                        # fca_family. Falls back to CPU if the device path is unsupported.
                        try:
                            res = cp.predict(Xt, verbose=False, device=args.device)
                        except (RuntimeError, ValueError):
                            res = cp.predict(Xt, verbose=False, device="cpu")
                        psets = res["prediction_sets"]
                        covs.append(float(np.mean([yt[i] in psets[i] for i in range(len(yt))])))
                        szs.append(float(np.mean([len(s) for s in psets])))
                    else:
                        M, run_kw = build_M_and_run_kwargs(
                            args.similarity, transform.Xu_transformed_, Xc, yc,
                            allc, args, tau_arg)
                        soft_kw = (dict(temperature=T_map[ncm_name], logit=args.logit)
                                   if is_soft else {})
                        cov, sz = run_fcp_with_mscs(
                            Xc, yc, Xt, yt, allc, ncm_name, args.alpha, lam, M,
                            exchangeable=True, yhat_mode=args.yhat_mode,
                            device=args.device, **run_kw, **soft_kw)
                        covs.append(cov); szs.append(sz)
                cov_m, sz_m = float(np.mean(covs)), float(np.mean(szs))
                if lam == 0.0:
                    base_sz = sz_m
                red = (100.0 * (1.0 - sz_m / base_sz)) if base_sz else 0.0
                rows.append({"dataset": args.dataset, "ncm": ncm_name, "cal": cal,
                             "lam": lam, "cov": cov_m, "sz": sz_m,
                             "sz_reduction_pct": red, "n_trials": args.n_trials})
                print(f"  {ncm_name:20s} lam={lam:<5} cov={cov_m:.4f} sz={sz_m:7.3f} "
                      f"red={red:5.1f}%  ({time.time()-t0:.1f}s)")

    # tag non-default similarity so prototype/centroid runs don't clobber the
    # existing cluster-M results_<ds>.json
    sim_tag = "" if args.similarity == "cluster" else f"_{args.similarity}"
    out = os.path.join(args.output_dir, f"results_{args.dataset}{sim_tag}.json")
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "config": vars(args), "rows": rows}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
