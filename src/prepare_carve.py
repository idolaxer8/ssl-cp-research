"""Generic ImageFolder carve + extraction for headline datasets.

Per class: seeded shuffle, first --n_labeled images -> labeled subset,
next up to --n_pool -> unlabeled pool (disjoint by construction). Both
subsets are extracted with extract_features.py and saved as

    <emb_dir>/embeddings_<name>.pt            {embeddings, labels}
    <emb_dir>/embeddings_<name>_unlabeled.pt  {embeddings, labels}

(the filenames headline_experiment.py expects). Generalizes
prepare_food101.py to any ImageFolder root (eurosat, caltech101, stl10,
...).

Usage:
    python src/prepare_carve.py --src data/eurosat --name eurosat \
        --n_labeled 30 --n_pool 1000 --input_size 224
"""
import argparse, os, shutil, subprocess, sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ImageFolder root")
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset_dir", default=None)
    ap.add_argument("--emb_dir", default="output/local_embeddings")
    ap.add_argument("--n_labeled", type=int, required=True)
    ap.add_argument("--n_pool", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input_size", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--model", default="dinov2-base")
    args = ap.parse_args()
    subset = args.subset_dir or f"data/{args.name}_subset"

    classes = sorted(d for d in os.listdir(args.src)
                     if os.path.isdir(os.path.join(args.src, d)))
    print(f"{args.name}: {len(classes)} classes in {args.src}")
    rng = np.random.default_rng(args.seed)
    for part in ("labeled", "pool"):
        shutil.rmtree(os.path.join(subset, part), ignore_errors=True)
    n_lab = n_pool = 0
    for cls in classes:
        files = sorted(f for f in os.listdir(os.path.join(args.src, cls))
                       if os.path.splitext(f)[1].lower() in IMG_EXT)
        if len(files) < args.n_labeled + 1:
            raise SystemExit(f"class {cls}: only {len(files)} images "
                             f"(< n_labeled+1)")
        take = rng.permutation(len(files))
        sel = {"labeled": take[:args.n_labeled],
               "pool": take[args.n_labeled:args.n_labeled + args.n_pool]}
        for part, idxs in sel.items():
            dst = os.path.join(subset, part, cls)
            os.makedirs(dst, exist_ok=True)
            for i in idxs:
                shutil.copy(os.path.join(args.src, cls, files[i]), dst)
        n_lab += len(sel["labeled"])
        n_pool += len(sel["pool"])
    print(f"carved {n_lab} labeled + {n_pool} pool -> {subset}")

    os.makedirs(args.emb_dir, exist_ok=True)
    for part, out_name in (
            ("labeled", f"embeddings_{args.name}.pt"),
            ("pool", f"embeddings_{args.name}_unlabeled.pt")):
        cmd = [sys.executable, os.path.join(SRC, "extract_features.py"),
               "--data_dir", os.path.join(subset, part),
               "--output_name", out_name, "--model", args.model,
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
