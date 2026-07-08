"""
ICA / negentropy-selection pilot (TAFSSL-inspired, arXiv:2003.06670).

TAFSSL's insight: under the per-dim mixture model
    F_d ~ rho_n*N(mu_n,sigma_n) + rho_s*N(mu_s,sigma_s),
a signal-carrying direction is a Gaussian MIXTURE, hence non-Gaussian,
REGARDLESS of its variance. So rank directions by NON-GAUSSIANITY
(negentropy) instead of variance -- on collapsed-spectrum data (aircraft)
this could find the low-variance tail signal label-free, giving the regime a
COMPACT subspace (prototype-friendly) instead of full-rank whitening.

Arms (all pool-fit fixed maps -> exactly exchangeable):
  negent128     full pool eigenbasis -> standardize dims -> keep top-128 by
                negentropy (logcosh approx), selected from ANYWHERE in the
                spectrum. Reports which spectral bands get selected -- the
                label-free counterpart of spectral_band_diagnostic.py.
  negent128_cw  + within-cluster diagonal whitening in the selected subspace.
  ica128_lw     LW-cluster whitening first, then FastICA-128 on the whitened
                pool (closest to TAFSSL's actual recipe on top of our metric).
  pca128_cw / lw_cluster768   references on identical splits.

Usage:
python src/ica_negent_pilot.py --dataset aircraft --k_cw 20 --k_lw 10 \
    --data_dir output/from_cluster --output_dir output/pca_pilots/ica_negent
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA

from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import UnlabeledTransform
from ae_lowpr_pilot import Chain, resolve_T
from transform_control_experiment import balanced_split

BAND_EDGES = [0, 16, 128, 512, 10 ** 9]


def logcosh(x):
    return np.logaddexp(x, -x) - np.log(2.0)


class NegentropySelect:
    """Full pool eigenbasis -> per-dim standardization -> keep the top-d dims
    by a negentropy approximation J(y) = (E[logcosh(y)] - E[logcosh(nu)])^2,
    nu ~ N(0,1). Selection can pick LOW-variance directions (unlike PCA)."""

    def __init__(self, d=128, random_state=42):
        self.d = d
        self.random_state = random_state

    def fit(self, Xu):
        X = np.asarray(Xu, dtype=np.float64)
        self.pca_ = PCA(n_components=min(X.shape),
                        random_state=self.random_state).fit(X)
        Z = self.pca_.transform(X)
        self.sd_ = Z.std(axis=0) + 1e-12
        Zs = Z / self.sd_
        rng = np.random.default_rng(self.random_state)
        gauss_ref = float(np.mean(logcosh(rng.standard_normal(200_000))))
        J = (logcosh(Zs).mean(axis=0) - gauss_ref) ** 2
        self.keep_ = np.argsort(J)[::-1][: self.d].copy()
        self.negentropy_ = J
        # where in the spectrum did the selection land?
        ranks = np.sort(self.keep_)
        self.band_counts_ = {
            f"{lo + 1}-{hi if hi < 10**9 else len(J)}": int(
                ((ranks >= lo) & (ranks < hi)).sum())
            for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])
            if lo < len(J)}
        self.median_rank_ = int(np.median(ranks)) + 1
        return self

    def transform(self, X):
        Z = self.pca_.transform(np.asarray(X, dtype=np.float64)) / self.sd_
        return Z[:, self.keep_]

    def __repr__(self):
        return (f"NegentropySelect(d={self.d}, median_spectral_rank="
                f"{self.median_rank_}, bands={self.band_counts_})")


class LWThenICA:
    """LW-cluster whitening -> FastICA to d components (pool-fit, seeded)."""

    def __init__(self, d=128, n_clusters=10, random_state=42):
        self.d, self.n_clusters, self.random_state = d, n_clusters, random_state

    def fit(self, Xu):
        self.lw_ = UnlabeledTransform(pca_dim=None, whiten="lw_cluster",
                                      n_clusters=self.n_clusters).fit(Xu)
        Zw = self.lw_.Xu_transformed_
        self.ica_ = FastICA(n_components=self.d, random_state=self.random_state,
                            max_iter=1000, tol=1e-3).fit(Zw)
        return self

    def transform(self, X):
        return self.ica_.transform(self.lw_.transform(X)).astype(np.float64)

    def __repr__(self):
        it = getattr(self.ica_, "n_iter_", "?")
        return f"LWThenICA(d={self.d}, k={self.n_clusters}, ica_iters={it})"


def main():
    ap = argparse.ArgumentParser(description="Negentropy/ICA subspace pilot (TAFSSL-inspired)")
    ap.add_argument("--data_dir", default="output/from_cluster")
    ap.add_argument("--dataset", default="aircraft")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--k_cw", type=int, default=20,
                    help="clusters for diagonal cluster-whitening arms (incl. pca128_cw ref)")
    ap.add_argument("--k_lw", type=int, default=10,
                    help="clusters for LW-cluster whitening (ica arm + lw ref)")
    ap.add_argument("--ncms", nargs="+",
                    default=["unwhitened_topk_mean", "prototype_softmax"])
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[400, 800])
    ap.add_argument("--test_per_class", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="output/pca_pilots/ica_negent")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    d = torch.load(os.path.join(args.data_dir, f"embeddings_{args.dataset}.pt"),
                   map_location="cpu", weights_only=False)
    X, y = d["embeddings"].numpy(), d["labels"].numpy()
    Xu = torch.load(os.path.join(args.data_dir,
                                 f"embeddings_{args.dataset}_unlabeled.pt"),
                    map_location="cpu", weights_only=False)["embeddings"].numpy()
    allc = np.unique(y)
    K = len(allc)
    print(f"{args.dataset}: X={X.shape} K={K} pool={Xu.shape} | "
          f"k_cw={args.k_cw} k_lw={args.k_lw}")

    arms = {
        "negent128":    Chain(NegentropySelect(args.d, args.seed)),
        "negent128_cw": Chain(NegentropySelect(args.d, args.seed),
                              UnlabeledTransform(pca_dim=None, whiten="cluster",
                                                 n_clusters=args.k_cw)),
        "ica128_lw":    Chain(LWThenICA(args.d, args.k_lw, args.seed)),
        "pca128_cw":    Chain(UnlabeledTransform(pca_dim=128, whiten="cluster",
                                                 n_clusters=args.k_cw)),
        "lw_cluster768": Chain(UnlabeledTransform(pca_dim=None,
                                                  whiten="lw_cluster",
                                                  n_clusters=args.k_lw)),
    }

    rows = []
    for name, tf in arms.items():
        t0 = time.time()
        tf.fit(Xu)
        print(f"\n=== {name}: {tf}  [fit {time.time()-t0:.0f}s] ===")
        T = (resolve_T(tf, X, y, allc, args.seed, max(args.cal_sizes))
             if "prototype_softmax" in args.ncms else None)
        for cal in args.cal_sizes:
            m_cal = cal // K
            for nm in args.ncms:
                covs, szs = [], []
                t0 = time.time()
                for t in range(args.n_trials):
                    rng = np.random.default_rng(args.seed + 1000 * t)
                    ci, ti = balanced_split(y, allc, m_cal, args.test_per_class, rng)
                    Xc, Xt = tf.transform(X[ci]), tf.transform(X[ti])
                    yc, yt = y[ci], y[ti]
                    ncm = (create_ncm(nm, temperature=T, logit="cosine")
                           if nm == "prototype_softmax" else create_ncm(nm, k=5))
                    cp = FullConformalPredictor(ncm, alpha=args.alpha)
                    cp.calibrate(Xc, yc, all_classes=allc)
                    try:
                        res = cp.predict(Xt, verbose=False, device=args.device)
                    except (RuntimeError, ValueError):
                        res = cp.predict(Xt, verbose=False, device="cpu")
                    psets = res["prediction_sets"]
                    covs.append(float(np.mean([yt[i] in psets[i]
                                               for i in range(len(yt))])))
                    szs.append(float(np.mean([len(s) for s in psets])))
                row = {"arm": name, "cal": cal, "ncm": nm,
                       "cov": float(np.mean(covs)), "sz": float(np.mean(szs)),
                       "sz_se": float(np.std(szs) / np.sqrt(len(szs))),
                       "n_trials": args.n_trials}
                first = tf.steps[0]
                if isinstance(first, NegentropySelect):
                    row["band_counts"] = first.band_counts_
                    row["median_spectral_rank"] = first.median_rank_
                rows.append(row)
                print(f"  cal={cal:4d} {nm:22s} cov={row['cov']:.4f} "
                      f"sz={row['sz']:6.2f}+-{row['sz_se']:.2f} "
                      f"({time.time()-t0:.0f}s)")
        if args.device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"results_{args.dataset}.json")
    with open(out_json, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
