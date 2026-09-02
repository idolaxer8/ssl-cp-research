"""Apply the existing cub200 labeled/pool carve to a fresh 518px
extraction (resolution-confound test, user call 2026-08-25).

The 336px carve (output/pca_pilots/heldout_data/) is a row PARTITION of
output/embeddings_cub200_all.pt (5977 + 4000 = 9977 rows, verified), and
extract_features.py preserves ImageFolder order, so row i of any re-
extraction corresponds to row i of the 336px all-file. We recover the
carve indices by exact float row-matching (consume-once multimap, robust
to duplicate images) and apply them to the 518px all-file.

Outputs (same filenames the headline driver expects via --cub_dir):
    <out_dir>/embeddings_cub200.pt            {embeddings, labels}
    <out_dir>/embeddings_cub200_unlabeled.pt  {embeddings, labels}
"""
import argparse, os
from collections import defaultdict

import numpy as np
import torch


def row_index_map(A):
    m = defaultdict(list)
    A = np.ascontiguousarray(A)
    for i in range(len(A)):
        m[A[i].tobytes()].append(i)
    return m


def match_rows(part, m):
    idx = []
    for r in np.ascontiguousarray(part):
        lst = m[r.tobytes()]
        if not lst:
            raise SystemExit("carve row not found in all-file (or reused) "
                             "-- carve is not a row partition?")
        idx.append(lst.pop(0))          # consume-once: duplicates safe
    return np.array(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all336", default="output/embeddings_cub200_all.pt")
    ap.add_argument("--carve_dir", default="output/pca_pilots/heldout_data")
    ap.add_argument("--all518", required=True,
                    help="fresh 518px extraction of data/cub200 (all rows)")
    ap.add_argument("--out_dir", default="output/local_embeddings518")
    args = ap.parse_args()

    ld = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    all336 = ld(args.all336)
    lab336 = ld(os.path.join(args.carve_dir, "embeddings_cub200.pt"))
    unl336 = ld(os.path.join(args.carve_dir,
                             "embeddings_cub200_unlabeled.pt"))
    all518 = ld(args.all518)
    A = all336["embeddings"].numpy()
    assert len(A) == len(all518["embeddings"]), \
        f"row mismatch: {len(A)} vs {len(all518['embeddings'])}"

    m = row_index_map(A)
    li = match_rows(lab336["embeddings"].numpy(), m)
    ui = match_rows(unl336["embeddings"].numpy(), m)
    assert len(set(li) | set(ui)) == len(A), "carve does not cover all rows"
    # label agreement: same image -> same label in both extractions
    yl = all336["labels"].numpy()
    assert (yl[li] == lab336["labels"].numpy()).all()
    assert (yl[ui] == unl336["labels"].numpy()).all()
    assert (all518["labels"].numpy() == yl).all(), \
        "518 extraction has different ImageFolder label order"

    os.makedirs(args.out_dir, exist_ok=True)
    E = all518["embeddings"]
    torch.save({"embeddings": E[li], "labels": all518["labels"][li]},
               os.path.join(args.out_dir, "embeddings_cub200.pt"))
    torch.save({"embeddings": E[ui], "labels": all518["labels"][ui]},
               os.path.join(args.out_dir, "embeddings_cub200_unlabeled.pt"))
    print(f"carve recovered exactly: labeled {len(li)}, pool {len(ui)} "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
