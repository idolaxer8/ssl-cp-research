"""
Dimension-pair screen for MDCP-pool (direction 2: multi-VIEW dimensions from
the SAME backbone -- different pool-fit feature transforms of DINOv2, zero new
extraction).

For each candidate (ncm, view) dimension it reports, against the anchor
dimension (prototype LAC on pca128_cw = our best balanced 1-D score):
  - Spearman correlation on TRUE-label scores and on FALSE-label scores
    (false-corr is the one that predicts 2-D purity gains: CIFAR-100
    geo-vs-proto same-view false-corr 0.92 -> zero gain; Aircraft 0.59 ->
    -7..11%)
  - the dimension's SOLO 1-D SCP mean set size on the same split (a candidate
    that is garbage alone can only contribute auto-selection, not gains)

GO gate for the 20-trial grid: false-corr well below ~0.8 AND solo size
within ~2x of the anchor's.

Run:
python src/mdcp_dim_screen.py --embeddings_path <...>.pt --unlabeled_path <...>.pt
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from scipy.stats import spearmanr

from conformal_prediction import PrototypeSoftmaxNCM
from mdcp_pool_pilot import (geodesic_asym_scores, prototype_lac_scores,
                             balanced_both_split, scp_eval, build_view_feats)


def pilot_T(Z, y, classes, seed=0):
    rng = np.random.default_rng(seed)
    ci, _ = balanced_both_split(y, classes, 8 * len(classes), rng)
    p = PrototypeSoftmaxNCM(temperature=None, logit="cosine",
                            allow_nonexchangeable=True)
    p.fit(Z[ci], y[ci])
    return float(p._T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_path", required=True)
    ap.add_argument("--unlabeled_path", required=True)
    ap.add_argument("--cal", type=int, default=800)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor", default="proto:pca128_cw")
    ap.add_argument("--cands", nargs="+",
                    default=["geo:pca128_cw", "proto:raw768", "geo:raw768",
                             "proto:cw768", "geo:cw768", "proto:pca32_cw",
                             "geo:pca32_cw"],
                    help="candidate dims 'ncm:view' to screen vs the anchor")
    args = ap.parse_args()

    d = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)
    X, y = d["embeddings"].numpy().astype(np.float64), d["labels"].numpy()
    Xu = torch.load(args.unlabeled_path, map_location="cpu",
                    weights_only=False)["embeddings"].numpy().astype(np.float64)
    classes = np.unique(y)
    K = len(classes)

    specs = [tuple(args.anchor.split(":"))] + [tuple(c.split(":")) for c in args.cands]
    print("building views...")
    views = build_view_feats(X, Xu, [v for _, v in specs])

    rng = np.random.default_rng(args.seed + args.cal)
    ci, ti = balanced_both_split(y, classes, args.cal, rng)
    yc, yt = y[ci], y[ti]
    y_test_col = np.searchsorted(classes, yt)
    k_geo = max(1, min(5, len(yc) // K))

    # dimension scores for every requested (ncm, view)
    dims = {}
    T_cache = {}
    for ncm, vname in specs:
        if (ncm, vname) in dims:
            continue
        Zv = views[vname][0]
        Zc, Zt = Zv[ci], Zv[ti]
        if ncm == "proto":
            if vname not in T_cache:
                T_cache[vname] = pilot_T(Zv, y, classes, args.seed)
            T = T_cache[vname]
            dims[(ncm, vname)] = dict(
                cal=prototype_lac_scores(Zc, Zc, yc, classes, T, loo=True),
                test=prototype_lac_scores(Zt, Zc, yc, classes, T), T=T)
        else:
            dims[(ncm, vname)] = dict(
                cal=geodesic_asym_scores(Zc, Zc, yc, classes, k_geo, self_is_cal=True),
                test=geodesic_asym_scores(Zt, Zc, yc, classes, k_geo))

    # solo 1-D SCP size per dimension (single split, indicative)
    rows_c = np.arange(len(yc)); col_c = np.searchsorted(classes, yc)
    solo = {}
    for key, dd in dims.items():
        cov, sz, _, _ = scp_eval(dd["cal"][rows_c, col_c], dd["test"], y_test_col,
                                 args.alpha)
        solo[key] = (cov, sz)

    # Spearman vs the anchor, on test scores split into true/false
    anchor = tuple(args.anchor.split(":"))
    mask_true = np.zeros_like(dims[anchor]["test"], dtype=bool)
    mask_true[np.arange(len(yt)), y_test_col] = True
    A = dims[anchor]["test"]

    print(f"\nanchor = {anchor}  solo: cov {solo[anchor][0]:.3f}  sz {solo[anchor][1]:.2f}"
          f"  (cal={args.cal}, balanced_both, single split)\n")
    print(f"{'candidate dim':<24}{'corr TRUE':>10}{'corr FALSE':>11}{'solo cov':>10}{'solo sz':>9}")
    for key, dd in sorted(dims.items()):
        if key == anchor:
            continue
        B = dd["test"]
        rt = spearmanr(A[mask_true], B[mask_true]).statistic
        rf = spearmanr(A[~mask_true], B[~mask_true]).statistic
        print(f"{str(key):<24}{rt:>10.3f}{rf:>11.3f}{solo[key][0]:>10.3f}{solo[key][1]:>9.2f}")


if __name__ == "__main__":
    main()
