"""
Spectral-band diagnostic: WHERE along the pool spectrum does class signal live?

Fits full PCA on the unlabeled pool, splits the eigenbasis into bands
(top-16 | 17-128 | 129-512 | 513-768 by default), then measures, on a labeled
subsample PROJECTED ONTO EACH BAND ALONE (L2-normalised, mirroring the NCMs):

    class_ratio   mean between-class / mean within-class cosine distance
    nn1_acc       1-NN accuracy (cosine)
    var_share     fraction of pool variance the band carries

Labels are used for DIAGNOSTICS only -- nothing here feeds the pipeline.
This is the mechanism figure behind the per-regime transform heuristic:
separable backbones concentrate class signal in the top band (truncate there);
collapsed-spectrum data (aircraft) spreads it into the tail (whiten, don't
truncate).

Usage:
python src/spectral_band_diagnostic.py --data_dir output/from_cluster \
    --datasets cifar100 aircraft miniimagenet \
    --output_dir output/pca_pilots/geometry
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.decomposition import PCA

from embedding_geometry_diagnostic import l2n, class_metrics


def main():
    ap = argparse.ArgumentParser(description="Class-signal location along the pool spectrum")
    ap.add_argument("--data_dir", default="output/from_cluster")
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "aircraft"])
    ap.add_argument("--bands", type=int, nargs="+", default=[16, 128, 512],
                    help="band edges; bands are [0:e1], [e1:e2], ..., [eN:D].")
    ap.add_argument("--n_labeled", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="output/pca_pilots/geometry")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}
    for ds in args.datasets:
        lab = os.path.join(args.data_dir, f"embeddings_{ds}.pt")
        unl = os.path.join(args.data_dir, f"embeddings_{ds}_unlabeled.pt")
        if not (os.path.exists(lab) and os.path.exists(unl)):
            print(f"[skip] {ds}: missing embeddings in {args.data_dir}")
            continue
        d = torch.load(lab, map_location="cpu", weights_only=False)
        X, y = d["embeddings"].numpy(), d["labels"].numpy()
        Xu = torch.load(unl, map_location="cpu",
                        weights_only=False)["embeddings"].numpy()

        D = X.shape[1]
        pca = PCA(n_components=min(len(Xu), D)).fit(Xu)
        n_comp = pca.components_.shape[0]
        edges = [0] + [e for e in args.bands if e < n_comp] + [n_comp]

        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X), min(args.n_labeled, len(X)), replace=False)
        Z_full = (X[idx] - pca.mean_) @ pca.components_.T  # rotate to eigenbasis
        var = pca.explained_variance_

        rows = []
        print(f"\n=== {ds}: pool spectrum bands (n_comp={n_comp}) ===")
        for lo, hi in zip(edges[:-1], edges[1:]):
            Zb = l2n(Z_full[:, lo:hi])
            cm = class_metrics(Zb, y[idx], rng)
            row = {"band": f"{lo + 1}-{hi}", "lo": lo, "hi": hi,
                   "var_share": float(var[lo:hi].sum() / var.sum()), **cm}
            rows.append(row)
            print(f"  dims {row['band']:>9s}: var={row['var_share']:6.1%} "
                  f"class_ratio={row['class_ratio']:.3f} "
                  f"1nn={row['nn1_acc']:.3f}")
        # full space reference
        cm = class_metrics(l2n(Z_full), y[idx], rng)
        rows.append({"band": "full", "lo": 0, "hi": n_comp, "var_share": 1.0, **cm})
        print(f"  {'full':>14s}: var=100.0% class_ratio={cm['class_ratio']:.3f} "
              f"1nn={cm['nn1_acc']:.3f}")
        all_results[ds] = rows

    out_json = os.path.join(args.output_dir, "spectral_bands.json")
    with open(out_json, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)
    print(f"\nSaved -> {out_json}")

    # figure: per dataset, 1nn_acc + class_ratio per band
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(all_results)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7), squeeze=False)
    for j, (ds, rows) in enumerate(all_results.items()):
        bands = [r["band"] for r in rows]
        for i, (m, ttl) in enumerate([("nn1_acc", "1-NN accuracy"),
                                      ("class_ratio", "between/within ratio")]):
            ax = axes[i][j]
            bars = ax.bar(range(len(bands)), [r[m] for r in rows],
                          color=["#1f77b4"] * (len(bands) - 1) + ["#d62728"])
            ax.set_xticks(range(len(bands)))
            ax.set_xticklabels(bands, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{ds} — {ttl} per spectral band", fontsize=10)
            ax.bar_label(bars, fmt="%.2f", fontsize=7)
    fig.suptitle("Where does class signal live along the pool spectrum? "
                 "(bands of the pool PCA eigenbasis, L2-normalised)")
    fig.tight_layout()
    p = os.path.join(args.output_dir, "spectral_bands.png")
    fig.savefig(p, dpi=140)
    print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
