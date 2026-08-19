"""
W-phase covariance-source ladder + pool-abundance sweep.

Paper positioning (2026-08-19): whitening = canonical label-free metric choice
(Mahalanobis w.r.t. pool covariance); Sigma_t = Sigma_w + Sigma_b, so total
whitening is NOT the optimal discriminative metric; its value is that a global
geometry is estimable from an ABUNDANT UNLABELED pool when class-specific
geometry cannot be estimated from a few labels; set-size gain is an efficiency
hypothesis to test. This script supplies the two missing set-size experiments:

MODE ladder -- one pipeline, only the covariance SOURCE of the full-rank ZCA
whitener varies (matched LW shrinkage estimator, matched pool-mean centering,
matched NCM/split/trials):

    raw         identity (no whitening)               -- baseline
    shot_lw     within-class cov from s=cal/K labeled -- "few labels cannot
                shots/class DISJOINT from cal+test       estimate class-specific
                (per-trial refit; still exchangeable)    geometry" demonstrated
    cluster_lw  within-KMEANS-cluster cov from the    -- deployed label-free
                unlabeled pool (= lw_cluster768 arm)     within-proxy
    total_lw    total cov from the unlabeled pool     -- deployed label-free
                (= lw_global768 arm)                     canonical Sigma_t
    oracle_lw   within-class cov from the pool using  -- label-information
                its TRUE labels (pool independent of     ceiling; exchangeable
                cal/test -> still exactly exchangeable)  but not label-free

MODE pool_sweep -- total_lw and cluster_lw refit on subsampled pools of size
n_pool (3 draws, trial t uses draw t%3), fixed cal; logs LW shrinkage as the
estimability readout. raw included as the flat reference.

All arms are fixed maps w.r.t. the cal/test bag (pool-fit, pool-label-fit, or
fit on a labeled split disjoint from cal+test), so every arm is exactly
exchangeable -- efficiency comparison only.

Usage (local, from repo root; data_dir may be an absolute path):
python src/w_covariance_ladder_experiment.py --mode ladder --dataset cifar100 \
    --data_dir output/from_cluster/embeddings --device cuda
python src/w_covariance_ladder_experiment.py --mode pool_sweep --dataset cifar100 \
    --data_dir output/from_cluster/embeddings --device cuda
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import UnlabeledTransform, IdentityTransform


class ZCAWhitenTransform:
    """Full-rank ZCA whitener from an LW-shrunk covariance of GIVEN residuals,
    mirroring UnlabeledTransform's lw_* construction exactly (pool-mean
    centering, LedoitWolf(assume_centered=True), symmetric inverse sqrt)."""

    def fit(self, resid, center):
        lw = LedoitWolf(assume_centered=True).fit(np.asarray(resid, np.float64))
        self.lw_shrinkage_ = float(lw.shrinkage_)
        evals, evecs = np.linalg.eigh(lw.covariance_)
        evals = np.maximum(evals, 1e-12)
        self.W_ = evecs @ np.diag(evals ** -0.5) @ evecs.T
        self.center_ = np.asarray(center, np.float64)
        return self

    def transform(self, X):
        return (np.asarray(X, np.float64) - self.center_) @ self.W_


def within_class_residuals(X, y):
    X = np.asarray(X, np.float64)
    resid = X.copy()
    for c in np.unique(y):
        m = y == c
        resid[m] -= X[m].mean(axis=0)
    return resid


def load_embeddings(data_dir, dataset):
    """Plain finals {embeddings,labels}; falls back to the _layers.pt 'final'
    key (stanford_cars has no plain finals)."""
    def _one(tag):
        plain = os.path.join(data_dir, f"embeddings_{dataset}{tag}.pt")
        if os.path.exists(plain):
            d = torch.load(plain, map_location="cpu", weights_only=False)
            return d["embeddings"].numpy(), d["labels"].numpy()
        layers = os.path.join(data_dir, f"embeddings_{dataset}{tag}_layers.pt")
        d = torch.load(layers, map_location="cpu", weights_only=False)
        return d["final"].numpy(), d["labels"].numpy()
    X, y = _one("")
    Xu, yu = _one("_unlabeled")
    return X, y, Xu, yu


def balanced_split_with_shots(y, allc, m_cal, m_test, n_shots, rng):
    """Cal/test slices identical to transform_control_experiment.balanced_split
    under the same rng; shots come from the NEXT slice of the same permutation
    (disjoint from cal+test by construction)."""
    ci, ti, si = [], [], []
    for c in allc:
        perm = rng.permutation(np.where(y == c)[0])
        ci.append(perm[:m_cal])
        ti.append(perm[m_cal:m_cal + m_test])
        si.append(perm[m_cal + m_test:m_cal + m_test + n_shots])
    return np.concatenate(ci), np.concatenate(ti), np.concatenate(si)


def run_cell(tf_for_trial, X, y, allc, cal, m_cal, m_test, nm, args,
             shot_fit=False):
    """One (arm, cal, ncm) cell: n_trials FCP runs. tf_for_trial(t, si) returns
    the fitted transform for trial t (si = shot indices, used by shot_lw)."""
    K = len(allc)
    cls_to_j = {int(c): j for j, c in enumerate(allc)}
    covs, szs, shrinks = [], [], []
    pooled_cov, pooled_tot = np.zeros(K), np.zeros(K)
    n_fallback = 0
    for t in range(args.n_trials):
        rng = np.random.default_rng(args.seed + 1000 * t)
        ci, ti, si = balanced_split_with_shots(
            y, allc, m_cal, m_test, m_cal if shot_fit else 0, rng)
        tf = tf_for_trial(t, si)
        if getattr(tf, "lw_shrinkage_", None) is not None:
            shrinks.append(tf.lw_shrinkage_)
        Xc, yc = tf.transform(X[ci]), y[ci]
        Xt, yt = tf.transform(X[ti]), y[ti]
        ncm = create_ncm(nm, k=5)
        cp = FullConformalPredictor(ncm, alpha=args.alpha)
        cp.calibrate(Xc, yc, all_classes=allc)
        try:
            res = cp.predict(Xt, verbose=False, device=args.device)
        except (RuntimeError, ValueError):
            res = cp.predict(Xt, verbose=False, device="cpu")
            n_fallback += 1
        psets = res["prediction_sets"]
        covered = np.array([yt[i] in psets[i] for i in range(len(yt))])
        covs.append(float(covered.mean()))
        szs.append(float(np.mean([len(s) for s in psets])))
        for c in allc:
            msk = yt == c
            if msk.any():
                j = cls_to_j[int(c)]
                pooled_cov[j] += float(covered[msk].sum())
                pooled_tot[j] += int(msk.sum())
    valid = pooled_tot > 0
    pcov = pooled_cov[valid] / pooled_tot[valid]
    row = {"cal": cal, "ncm": nm,
           "cov": float(np.mean(covs)), "cov_sd": float(np.std(covs)),
           "sz": float(np.mean(szs)), "sz_sd": float(np.std(szs)),
           "sz_se": float(np.std(szs) / np.sqrt(len(szs))),
           "covgap": float(100 * np.mean(np.abs(pcov - (1 - args.alpha)))),
           "n_trials": args.n_trials, "n_cpu_fallback": n_fallback}
    if shrinks:
        row["lw_shrinkage"] = float(np.mean(shrinks))
    return row


def main():
    ap = argparse.ArgumentParser(description="W covariance-source ladder / pool sweep (all arms exchangeable)")
    ap.add_argument("--mode", choices=["ladder", "pool_sweep"], default="ladder")
    ap.add_argument("--data_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="ladder: subset of raw/shot_lw/cluster_lw/total_lw/"
                         "oracle_lw; pool_sweep: subset of raw/cluster_lw/total_lw")
    ap.add_argument("--ncms", nargs="+",
                    default=["unwhitened_topk_asym", "unwhitened_topk_mean"])
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=None,
                    help="default 200/400/800 (2/4/8 per class scaled by K)")
    ap.add_argument("--pool_sizes", type=int, nargs="+",
                    default=[500, 1000, 2000, 5000, 10000])
    ap.add_argument("--pool_cal", type=int, default=None,
                    help="pool_sweep cal size (default 4 shots/class)")
    ap.add_argument("--pool_draws", type=int, default=3)
    ap.add_argument("--test_per_class", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_clusters_whiten", type=int, default=100)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="output/w_ladder")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu")
        args.device = "cpu"

    X, y, Xu, yu = load_embeddings(args.data_dir, args.dataset)
    allc = np.unique(y)
    K = len(allc)
    if args.cal_sizes is None:
        args.cal_sizes = [2 * K, 4 * K, 8 * K]
    m_test = args.test_per_class
    center = Xu.mean(axis=0)
    print(f"{args.dataset}: X={X.shape} K={K} pool={Xu.shape} "
          f"(pool labels: {len(np.unique(yu))} classes) | device={args.device}")

    rows = []
    if args.mode == "ladder":
        arms = args.arms or ["raw", "shot_lw", "cluster_lw", "total_lw",
                             "oracle_lw"]
        fixed = {}
        for arm in arms:
            t0 = time.time()
            if arm == "raw":
                fixed[arm] = IdentityTransform().fit()
            elif arm == "cluster_lw":
                fixed[arm] = UnlabeledTransform(
                    pca_dim=None, whiten="lw_cluster", projection=None,
                    n_clusters=args.n_clusters_whiten).fit(Xu)
            elif arm == "total_lw":
                fixed[arm] = UnlabeledTransform(
                    pca_dim=None, whiten="lw_global", projection=None).fit(Xu)
            elif arm == "oracle_lw":
                fixed[arm] = ZCAWhitenTransform().fit(
                    within_class_residuals(Xu, yu), center)
            if arm in fixed:
                sh = getattr(fixed[arm], "lw_shrinkage_", None)
                print(f"fit {arm} [{time.time()-t0:.0f}s]"
                      + (f" lw_shrinkage={sh:.4f}" if sh is not None else ""))

        for arm in arms:
            print(f"\n=== arm {arm} ===")
            for cal in args.cal_sizes:
                m_cal = cal // K
                if m_cal < 2:
                    continue
                if arm == "shot_lw":
                    def tf_for_trial(t, si):
                        return ZCAWhitenTransform().fit(
                            within_class_residuals(X[si], y[si]), center)
                else:
                    def tf_for_trial(t, si, _tf=fixed[arm]):
                        return _tf
                for nm in args.ncms:
                    t0 = time.time()
                    row = run_cell(tf_for_trial, X, y, allc, cal, m_cal,
                                   m_test, nm, args, shot_fit=(arm == "shot_lw"))
                    row["arm"] = arm
                    rows.append(row)
                    print(f"  cal={cal:4d} {nm:22s} cov={row['cov']:.4f} "
                          f"sz={row['sz']:7.2f}+-{row['sz_se']:.2f} "
                          f"covgap={row['covgap']:5.2f}pp"
                          + (f" shrink={row['lw_shrinkage']:.3f}"
                             if "lw_shrinkage" in row else "")
                          + f" ({time.time()-t0:.0f}s)")
            if args.device == "cuda":
                torch.cuda.empty_cache()

    else:  # pool_sweep
        arms = args.arms or ["raw", "cluster_lw", "total_lw"]
        cal = args.pool_cal or 4 * K
        m_cal = cal // K
        pool_sizes = [n for n in args.pool_sizes if n <= len(Xu)]
        if len(Xu) not in pool_sizes:
            pool_sizes.append(len(Xu))
        for arm in arms:
            print(f"\n=== arm {arm} ===")
            for n_pool in (pool_sizes if arm != "raw" else [len(Xu)]):
                if arm == "raw":
                    tfs = [IdentityTransform().fit()]
                else:
                    tfs = []
                    for r in range(args.pool_draws):
                        rng = np.random.default_rng(args.seed + 100000 * r
                                                    + n_pool)
                        idx = rng.choice(len(Xu), n_pool, replace=False)
                        if arm == "total_lw":
                            tfs.append(UnlabeledTransform(
                                pca_dim=None, whiten="lw_global",
                                projection=None).fit(Xu[idx]))
                        else:
                            tfs.append(UnlabeledTransform(
                                pca_dim=None, whiten="lw_cluster",
                                projection=None,
                                n_clusters=args.n_clusters_whiten).fit(Xu[idx]))

                def tf_for_trial(t, si, _tfs=tfs):
                    return _tfs[t % len(_tfs)]
                for nm in args.ncms:
                    t0 = time.time()
                    row = run_cell(tf_for_trial, X, y, allc, cal, m_cal,
                                   m_test, nm, args)
                    row["arm"], row["n_pool"] = arm, n_pool
                    row["lw_shrinkage_draws"] = [
                        getattr(tf, "lw_shrinkage_", None) for tf in tfs]
                    rows.append(row)
                    print(f"  n_pool={n_pool:5d} cal={cal} {nm:22s} "
                          f"cov={row['cov']:.4f} sz={row['sz']:7.2f}"
                          f"+-{row['sz_se']:.2f}"
                          + (f" shrink={row['lw_shrinkage_draws'][0]:.3f}"
                             if row['lw_shrinkage_draws'][0] is not None else "")
                          + f" ({time.time()-t0:.0f}s)")
            if args.device == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    out = {"config": vars(args), "dataset": args.dataset, "K": K, "rows": rows}
    out_json = os.path.join(args.output_dir,
                            f"{args.mode}_{args.dataset}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
