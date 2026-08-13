"""Empirical margin d'-ratio of qe on all five theorem datasets.

Complement to src/dwt_gate_constants.py: that script PREDICTS the d'-ratio
from the (I)-model composition constants (h_w, beta, k_eff, kappa); this one
MEASURES it directly -- for each class, fix the raw nearest-prototype pair
axis v = (mu_y - mu_c)/||.||, project class-y and class-c points onto v
before and after qe smoothing (k=10, a=3, pool), and take the ratio of the
standardized separations. Selection effects (V1)/(V2), which the (I) model
idealizes away, are fully present in the measurement, so
predicted-vs-measured localizes the idealization error.

Writes output/cars_qe_gate/dprime_predicted_vs_measured.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dwt_score_histograms import qe_smooth                       # noqa: E402
from cars_qe_gate_experiment import dprime_pair_ratios           # noqa: E402

# (I)-model predictions from docs/dwt_denoise_theorem.md Section 6
PREDICTED = {
    "cifar100": 1.96, "miniimagenet": 2.41, "cifar10": 2.35,
    "stanford_cars": 1.52, "aircraft": 0.70,
}
# observed qe verdict in the champion pipeline (menu round 1 / this run)
VERDICT = {
    "cifar100": "gain", "miniimagenet": "gain", "cifar10": "gain",
    "stanford_cars": "harm", "aircraft": "harm",
}


def load(name, emb_dir):
    if name == "stanford_cars":
        lab = torch.load(os.path.join(
            emb_dir, "embeddings_stanford_cars_layers.pt"),
            map_location="cpu", weights_only=False)
        unl = torch.load(os.path.join(
            emb_dir, "embeddings_stanford_cars_unlabeled_layers.pt"),
            map_location="cpu", weights_only=False)
        return (lab["final"].numpy().astype(np.float64),
                lab["labels"].numpy().astype(int),
                unl["final"].numpy().astype(np.float64))
    lab = torch.load(os.path.join(emb_dir, f"embeddings_{name}.pt"),
                     map_location="cpu", weights_only=False)
    unl = torch.load(os.path.join(emb_dir, f"embeddings_{name}_unlabeled.pt"),
                     map_location="cpu", weights_only=False)
    return (lab["embeddings"].numpy().astype(np.float64),
            lab["labels"].numpy().astype(int),
            unl["embeddings"].numpy().astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--out_dir", default="output/cars_qe_gate")
    ap.add_argument("--datasets", nargs="+", default=list(PREDICTED))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    out = {}
    for ds in args.datasets:
        X, y, U = load(ds, args.emb_dir)
        X_s = qe_smooth(X, U)
        ratios, raw_dps, sm_dps = dprime_pair_ratios(X, X_s, y)
        out[ds] = {
            "predicted_ratio": PREDICTED[ds],
            "measured_mean_ratio": float(ratios.mean()),
            "measured_median_ratio": float(np.median(ratios)),
            "frac_pairs_improved": float((ratios > 1).mean()),
            "n_pairs": int(len(ratios)),
            "mean_raw_dprime": float(raw_dps.mean()),
            "mean_smoothed_dprime": float(sm_dps.mean()),
            "verdict": VERDICT[ds],
        }
        print(f"{ds:14} predicted {PREDICTED[ds]:.2f}  "
              f"measured {ratios.mean():.3f}  "
              f"frac>1 {(ratios > 1).mean():.2f}  "
              f"raw d' {raw_dps.mean():.2f}", flush=True)

    path = os.path.join(args.out_dir, "dprime_predicted_vs_measured.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
