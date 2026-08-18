"""
AE-on-low-PR pilot: can an autoencoder beat the LW-cluster metric on
collapsed-spectrum (low participation-ratio) data, i.e. aircraft?

Hypothesis under test (user, 2026-07-06): PCA fails on aircraft because the
class signal hides in the low-variance tail; maybe a NONLINEAR bottleneck
captures what linear truncation cannot.

Sharp predictions stated up front (from the alignment hypothesis + the
variance-driven nature of reconstruction):
  ae128      vanilla AE-128 on the raw pool. MSE reconstruction is
             variance-driven, so its capacity goes to the 75%-variance
             nuisance bulk -> should be ~ a PCA-128 no-op, possibly worse.
  ae128_cw   AE latent + within-cluster diagonal whitening (mirror of
             pca128_cw) -> same story, mildly rescaled.
  lw10_ae128 LW-cluster whitening FIRST (equalize the spectrum so
             reconstruction no longer favors nuisance), THEN AE-128 on the
             whitened pool. The only arm where the AE sees tail signal at
             fair scale -> the live candidate.
References recomputed on identical splits: pca128_cw (k=20), lw_cluster768
(k=10, the current aircraft-best for the geodesic NCM).

Every step is fit on the unlabeled pool only (AE included) -> all arms are
fixed maps w.r.t. cal/test, exactly exchangeable.

Usage:
python src/ae_lowpr_pilot.py --data_dir output/from_cluster \
    --output_dir output/pca_pilots/ae_lowpr --device cuda
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import (FullConformalPredictor, create_ncm,
                                  PrototypeSoftmaxNCM)
from exchangeable_features import UnlabeledTransform
from autoencoder_utils import EmbeddingAutoencoder
from transform_control_experiment import balanced_split


class AEStep:
    """Pool-fit autoencoder bottleneck (centers by the pool mean first)."""

    def __init__(self, dim=128, hidden=512, epochs=150, patience=15, seed=42):
        self.dim, self.hidden = dim, hidden
        self.epochs, self.patience, self.seed = epochs, patience, seed

    def fit(self, Xu):
        self.center_ = Xu.mean(axis=0)
        self.ae_ = EmbeddingAutoencoder(
            bottleneck_dim=self.dim, hidden_dim=self.hidden, epochs=self.epochs,
            patience=self.patience, seed=self.seed).fit(Xu - self.center_)
        return self

    def transform(self, X):
        return self.ae_.transform(np.asarray(X) - self.center_).astype(np.float64)

    def __repr__(self):
        return f"AEStep(dim={self.dim}, hidden={self.hidden})"


class Chain:
    """Sequential pool-fit steps; each fits on the previous step's pool output."""

    def __init__(self, *steps):
        self.steps = steps

    def fit(self, Xu):
        Z = Xu
        for s in self.steps:
            s.fit(Z)
            Z = s.transform(Z)
        self.Xu_transformed_ = Z
        return self

    def transform(self, X):
        for s in self.steps:
            X = s.transform(X)
        return X

    def __repr__(self):
        return " -> ".join(repr(s) for s in self.steps)


def resolve_T(tf, X, y, allc, seed, cal_max=800):
    """Pilot-fixed prototype-softmax temperature (per arm; fixed across trials)."""
    m = max(4, cal_max // len(allc))
    rng = np.random.default_rng(seed)
    ci, _ = balanced_split(y, allc, m, 1, rng)
    p = PrototypeSoftmaxNCM(temperature=None, logit="cosine",
                            allow_nonexchangeable=True).fit(tf.transform(X[ci]), y[ci])
    return float(p._T)


def main():
    ap = argparse.ArgumentParser(description="AE vs LW-cluster on low-PR (aircraft)")
    ap.add_argument("--data_dir", default="output/from_cluster")
    ap.add_argument("--dataset", default="aircraft")
    ap.add_argument("--ncms", nargs="+",
                    default=["unwhitened_topk_mean", "prototype_softmax"])
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[400, 800])
    ap.add_argument("--test_per_class", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--ae_dim", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="output/pca_pilots/ae_lowpr")
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
    print(f"{args.dataset}: X={X.shape} K={K} pool={Xu.shape}")

    def ut(**kw):
        return UnlabeledTransform(**kw)

    arms = {
        "ae128":        Chain(AEStep(args.ae_dim, seed=args.seed)),
        "ae128_cw":     Chain(AEStep(args.ae_dim, seed=args.seed),
                              ut(pca_dim=None, whiten="cluster", n_clusters=20)),
        "lw10_ae128":   Chain(ut(pca_dim=None, whiten="lw_cluster", n_clusters=10),
                              AEStep(args.ae_dim, seed=args.seed)),
        "pca128_cw":    Chain(ut(pca_dim=128, whiten="cluster", n_clusters=20)),
        "lw_cluster768": Chain(ut(pca_dim=None, whiten="lw_cluster", n_clusters=10)),
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
                    cov = np.mean([yt[i] in psets[i] for i in range(len(yt))])
                    covs.append(float(cov))
                    szs.append(float(np.mean([len(s) for s in psets])))
                row = {"arm": name, "cal": cal, "ncm": nm,
                       "cov": float(np.mean(covs)),
                       "sz": float(np.mean(szs)),
                       "sz_se": float(np.std(szs) / np.sqrt(len(szs))),
                       "n_trials": args.n_trials}
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
