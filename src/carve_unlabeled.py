"""Carve a DISJOINT, stratified unlabeled pool out of a single labeled embeddings
file, producing the labeled / unlabeled .pt pair the exchangeable pipeline expects
(embeddings_<ds>.pt + embeddings_<ds>_unlabeled.pt).

Datasets like Flowers-102 / CUB-200 are extracted as ONE file (download_datasets.py
merges all splits), so they have no natural train/test unlabeled pool the way
Aircraft / mini-ImageNet do. This holds out `--unlabeled_per_class` samples per
class for the unlabeled pool (PCA / whitening / MS-CS fit there) and leaves the
rest as the labeled cal/test pool -- the two never share an index, so the pool-fit
transform never sees a cal/test point (exactly exchangeable).

Example (run from repo root, after extracting the full set to *_all.pt):
    python src/extract_features.py --data_dir data/flowers102 \
        --output_name embeddings_flowers102_all.pt --model dinov2-base
    python src/carve_unlabeled.py \
        --in output/embeddings_flowers102_all.pt \
        --out_labeled output/embeddings_flowers102.pt \
        --out_unlabeled output/embeddings_flowers102_unlabeled.pt \
        --unlabeled_per_class 30
"""
import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="Input embeddings .pt with keys {embeddings, labels}.")
    ap.add_argument("--out_labeled", required=True,
                    help="Output labeled .pt (reduced cal/test pool).")
    ap.add_argument("--out_unlabeled", required=True,
                    help="Output unlabeled .pt (disjoint stratified holdout).")
    ap.add_argument("--unlabeled_per_class", type=int, default=30,
                    help="Samples held out per class for the unlabeled pool. "
                         "Always leaves >=1 labeled sample per class.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = torch.load(args.inp, map_location="cpu", weights_only=False)
    X = d["embeddings"]
    y = d["labels"]
    Xn = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
    yn = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    classes = np.unique(yn)
    rng = np.random.default_rng(args.seed)

    unl_idx = []
    for c in classes:
        c_idx = np.where(yn == c)[0]
        take = min(args.unlabeled_per_class, len(c_idx) - 1)  # leave >=1 labeled
        if take > 0:
            unl_idx.append(rng.choice(c_idx, take, replace=False))
    unl_idx = np.concatenate(unl_idx)
    keep = np.ones(len(Xn), dtype=bool)
    keep[unl_idx] = False

    X_lab, y_lab = X[torch.as_tensor(keep)], y[torch.as_tensor(keep)]
    X_unl = X[torch.as_tensor(unl_idx)]

    os.makedirs(os.path.dirname(args.out_labeled) or ".", exist_ok=True)
    torch.save({"embeddings": X_lab, "labels": y_lab}, args.out_labeled)
    # unlabeled file: labels kept too (harmless; loaders read only "embeddings")
    torch.save({"embeddings": X_unl, "labels": y[torch.as_tensor(unl_idx)]},
               args.out_unlabeled)

    lc = len(classes)
    print(f"input : {tuple(Xn.shape)}, {lc} classes")
    print(f"labeled  -> {args.out_labeled}   {tuple(X_lab.shape)} "
          f"(~{len(X_lab)//lc}/class)")
    print(f"unlabeled-> {args.out_unlabeled} {tuple(X_unl.shape)} "
          f"(~{len(X_unl)//lc}/class), DISJOINT from labeled")


if __name__ == "__main__":
    main()
