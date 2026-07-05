"""
Embedding-geometry diagnostic: WHY does the pool-fit transform shrink sets?

Measures, per feature space (same arms as transform_control_experiment.py),
the high-dimensional pathologies the metric-learning framing says the
transform repairs, plus the discriminative signal the NCM actually consumes:

  hub_skew      skewness of the k-occurrence distribution N_k (k=10) in the
                cosine-kNN graph (hubness; Radovanovic et al., JMLR 2010).
                High skew = hub points corrupt kNN.
  anisotropy    mean pairwise cosine similarity of L2-normalised features
                (Ethayarajh 2019). High = narrow-cone geometry.
  rel_contrast  mean over queries of (d_max - d_min)/d_min on cosine distance
                (Beyer et al. 1999). Low = distance concentration.
  part_ratio    participation ratio of the covariance spectrum,
                (sum lambda)^2 / sum lambda^2 (effective dimensionality).
  class_ratio   mean between-class / mean within-class cosine distance on
                LABELED data (labels used for DIAGNOSTICS only, never in the
                pipeline). Higher = more class contrast for a distance NCM.
  nn1_acc       1-NN accuracy (cosine) on the labeled subsample -- the crudest
                proxy for the geodesic NCM's numerator quality.

All spaces are L2-normalised AFTER the transform, mirroring what the geodesic /
prototype NCMs do internally.

Usage:
python src/embedding_geometry_diagnostic.py \
    --data_dir output/from_cluster --output_dir output/pca_pilots/geometry
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from scipy.stats import skew

from exchangeable_features import UnlabeledTransform, IdentityTransform
from transform_control_experiment import ARMS


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def cosine_dist_matrix(Z):
    S = np.clip(Z @ Z.T, -1.0, 1.0)
    return 1.0 - S


def geometry_metrics(Z, k=10):
    """Z: L2-normalised (N, d). Returns hubness / anisotropy / concentration."""
    N = len(Z)
    D = cosine_dist_matrix(Z)
    np.fill_diagonal(D, np.inf)

    # k-occurrence: how often each point appears in another's kNN list
    knn = np.argpartition(D, k, axis=1)[:, :k]
    Nk = np.bincount(knn.ravel(), minlength=N)
    hub_skew = float(skew(Nk))
    hub_max = int(Nk.max())

    # relative contrast (finite entries only)
    dmin = D.min(axis=1)
    Df = np.where(np.isinf(D), -np.inf, D)
    dmax = Df.max(axis=1)
    rel_contrast = float(np.mean((dmax - dmin) / np.maximum(dmin, 1e-12)))

    # anisotropy: mean pairwise cosine similarity (off-diagonal)
    S = 1.0 - np.where(np.isinf(D), 1.0, D)  # diag -> 0 contribution handled below
    aniso = float((S.sum() - np.trace(S)) / (N * (N - 1)))

    # participation ratio of the (transformed-space) spectrum
    Zc = Z - Z.mean(axis=0)
    ev = np.linalg.svd(Zc, compute_uv=False) ** 2
    part_ratio = float(ev.sum() ** 2 / (ev ** 2).sum())

    return dict(hub_skew=hub_skew, hub_max=hub_max, rel_contrast=rel_contrast,
                anisotropy=aniso, part_ratio=part_ratio)


def class_metrics(Z, y, rng, max_between=200_000):
    """Labeled diagnostics: between/within cosine-distance ratio + 1-NN acc."""
    D = cosine_dist_matrix(Z)
    same = (y[:, None] == y[None, :])
    iu = np.triu_indices(len(Z), 1)
    d_flat, same_flat = D[iu], same[iu]
    within = float(d_flat[same_flat].mean())
    between_vals = d_flat[~same_flat]
    if len(between_vals) > max_between:
        between_vals = rng.choice(between_vals, max_between, replace=False)
    between = float(between_vals.mean())

    np.fill_diagonal(D, np.inf)
    nn1 = y[np.argmin(D, axis=1)]
    return dict(within_dist=within, between_dist=between,
                class_ratio=between / max(within, 1e-12),
                nn1_acc=float((nn1 == y).mean()))


def main():
    ap = argparse.ArgumentParser(description="Hubness/anisotropy/contrast per transform arm")
    ap.add_argument("--data_dir", default="output/from_cluster")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--n_pool", type=int, default=4000,
                    help="pool subsample for unlabeled geometry metrics")
    ap.add_argument("--n_labeled", type=int, default=3000,
                    help="labeled subsample for class-contrast metrics")
    ap.add_argument("--k", type=int, default=10, help="k for k-occurrence hubness")
    ap.add_argument("--n_clusters_whiten", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="output/pca_pilots/geometry")
    args = ap.parse_args()

    lab = os.path.join(args.data_dir, f"embeddings_{args.dataset}.pt")
    unl = os.path.join(args.data_dir, f"embeddings_{args.dataset}_unlabeled.pt")
    d = torch.load(lab, map_location="cpu", weights_only=False)
    X, y = d["embeddings"].numpy(), d["labels"].numpy()
    Xu = torch.load(unl, map_location="cpu", weights_only=False)["embeddings"].numpy()

    rng = np.random.default_rng(args.seed)
    pool_idx = rng.choice(len(Xu), min(args.n_pool, len(Xu)), replace=False)
    lab_idx = rng.choice(len(X), min(args.n_labeled, len(X)), replace=False)
    print(f"{args.dataset}: pool subsample {len(pool_idx)}, "
          f"labeled subsample {len(lab_idx)}")

    results = []
    for arm in args.arms:
        spec = ARMS[arm]
        if spec is None:
            tf = IdentityTransform().fit()
        else:
            tf = UnlabeledTransform(n_clusters=args.n_clusters_whiten,
                                    rp_seed=args.seed + 7919, **spec).fit(Xu)
        Zu = l2n(tf.transform(Xu[pool_idx]))
        Zl = l2n(tf.transform(X[lab_idx]))
        row = {"arm": arm, **geometry_metrics(Zu, k=args.k),
               **class_metrics(Zl, y[lab_idx], rng)}
        if getattr(tf, "lw_shrinkage_", None) is not None:
            row["lw_shrinkage"] = tf.lw_shrinkage_
        results.append(row)
        print(f"  {arm:14s} hub_skew={row['hub_skew']:6.2f} "
              f"aniso={row['anisotropy']:.4f} RC={row['rel_contrast']:7.2f} "
              f"PR={row['part_ratio']:6.1f} class_ratio={row['class_ratio']:.3f} "
              f"1nn={row['nn1_acc']:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"geometry_{args.dataset}.json")
    with open(out_json, "w") as f:
        json.dump({"config": vars(args), "rows": results}, f, indent=2)
    print(f"Saved -> {out_json}")

    # figure: one bar panel per metric, arms in canonical order
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("hub_skew", "hubness (k-occ skew, lower=better)"),
               ("anisotropy", "anisotropy (mean pair cos, lower=better)"),
               ("rel_contrast", "relative contrast (higher=better)"),
               ("part_ratio", "participation ratio"),
               ("class_ratio", "between/within dist ratio (higher=better)"),
               ("nn1_acc", "1-NN accuracy (higher=better)")]
    arms = [r["arm"] for r in results]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (m, title) in zip(axes.ravel(), metrics):
        vals = [r[m] for r in results]
        bars = ax.bar(range(len(arms)), vals,
                      color=["#d62728" if a == "pca128_cw" else "#1f77b4"
                             for a in arms])
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    fig.suptitle(f"Embedding-geometry diagnostics — {args.dataset} "
                 f"(pool n={len(pool_idx)}, cosine space, post-L2-norm)")
    fig.tight_layout()
    p = os.path.join(args.output_dir, f"geometry_{args.dataset}.png")
    fig.savefig(p, dpi=140)
    print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
