"""Food-101 for the R1 headline (3rd well-separated high-K dataset,
user call 2026-08-24): download, carve a class-balanced labeled subset +
disjoint unlabeled pool, extract dinov2-base embeddings at 336px (the
local-4GB convention, same caveat as cub200 vs the matched-518 cluster
finals), and save

    <emb_dir>/embeddings_food101.pt            {embeddings, labels}
    <emb_dir>/embeddings_food101_unlabeled.pt  {embeddings, labels}

Carve: per class, the first --n_labeled shuffled train images -> labeled,
the next --n_pool -> pool (seeded, disjoint by construction; test images
come from the labeled file's per-class remainder at split time, exactly as
in the other headline datasets).

Usage (from repo root):
    python src/prepare_food101.py            # ~5GB download on first run
"""
import argparse, json, os, shutil, subprocess, sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/food101_raw")
    ap.add_argument("--subset_dir", default="data/food101_subset")
    ap.add_argument("--emb_dir", default="output/local_embeddings")
    ap.add_argument("--n_labeled", type=int, default=25,
                    help="per class; shots<=20 with test_per_class=5")
    ap.add_argument("--n_pool", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input_size", type=int, default=336)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--model", default="dinov2-base")
    args = ap.parse_args()

    from torchvision.datasets import Food101
    Food101(root=args.root, split="train", download=True)   # fetch + verify
    base = os.path.join(args.root, "food-101")
    with open(os.path.join(base, "meta", "train.json")) as f:
        per_class = json.load(f)                 # class -> ["class/id", ...]
    print(f"food101: {len(per_class)} classes, "
          f"{sum(map(len, per_class.values()))} train images")

    rng = np.random.default_rng(args.seed)
    need = args.n_labeled + args.n_pool
    for part in ("labeled", "pool"):
        shutil.rmtree(os.path.join(args.subset_dir, part),
                      ignore_errors=True)
    n_copied = 0
    for cls, rels in sorted(per_class.items()):
        take = rng.permutation(len(rels))[:need]
        for j, idx in enumerate(take):
            part = "labeled" if j < args.n_labeled else "pool"
            dst_dir = os.path.join(args.subset_dir, part, cls)
            os.makedirs(dst_dir, exist_ok=True)
            src = os.path.join(base, "images", rels[idx] + ".jpg")
            shutil.copy(src, dst_dir)
            n_copied += 1
    print(f"copied {n_copied} images -> {args.subset_dir}")

    os.makedirs(args.emb_dir, exist_ok=True)
    for part, out_name in (("labeled", "embeddings_food101.pt"),
                           ("pool", "embeddings_food101_unlabeled.pt")):
        cmd = [sys.executable, os.path.join(SRC, "extract_features.py"),
               "--data_dir", os.path.join(args.subset_dir, part),
               "--output_name", out_name,
               "--model", args.model,
               "--input_size", str(args.input_size),
               "--batch_size", str(args.batch_size)]
        print("::", " ".join(cmd))
        subprocess.run(cmd, check=True)
        src_pt = os.path.join("output", out_name)
        dst_pt = os.path.join(args.emb_dir, out_name)
        if os.path.abspath(src_pt) != os.path.abspath(dst_pt):
            shutil.move(src_pt, dst_pt)
        print(f"saved {dst_pt}")


if __name__ == "__main__":
    main()
