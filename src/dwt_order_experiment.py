"""Stage-order experiment: qe in RAW space (deployed D->W) vs qe in the
WHITENED space (proposed W->D).

Motivation (user, 2026-08-18): the chain-free diagnostic showed whitening
raises pool-kNN homophily exactly where qe currently harms (aircraft
h_w .261->.337, cars .467->.545) and multiplies D1's certification rate
by 5.5 on aircraft. D1's reach is gated by homophily, and homophily is a
property of the METRIC the neighborhoods are computed in -- so if W
manufactures D's precondition, the coherent sequential order is W->D->T,
not the deployed D->W->T. This script runs the cheap decisive test:

  arm RAW->QE : qe_smooth in raw space, margin d'-ratio measured in raw
                space (reproduces the known 5/5 sign predictor:
                1.05/1.18/1.21/0.74/0.64).
  arm WLW->QE : full-rank LW pool whitening FIRST, then qe_smooth with
                neighborhoods in the whitened metric, d'-ratio measured
                in the whitened space.
  arm T128W->QE: deployed separable-champion transform first, then qe.

Registered predictions (BEFORE the run):
  (P-O1) measured d'-ratio of qe is HIGHER in the whitened space than raw
         on aircraft/cars (homophily up => less foreign drift).
  (P-O2) on separable datasets the order is ~neutral (h_w already high;
         wlw even dips it slightly).
  (open) whether cars/aircraft CROSS 1.0 (qe flips to helpful post-W) is
         left open -- h_w(wlw)=.545 on cars sits mid-bracket; a crossing
         would make the order swap a deployable discovery and warrant a
         full CP run.

Output: output/dwt_theory/dwt_order_experiment.json + printed table.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dwt_gate_constants import DATASETS, load_pt, l2n            # noqa: E402
from dwt_score_histograms import qe_smooth                       # noqa: E402
from cars_qe_gate_experiment import dprime_pair_ratios           # noqa: E402
from wt_chainfree_diagnostics import build_spaces_xu             # noqa: E402

RAW_DRATIO = {  # measure_dprime_all.py, 2026-08-13 (sign predicts 5/5)
    "cifar100": 1.05, "miniimagenet": 1.18, "cifar10": 1.21,
    "stanford_cars": 0.74, "aircraft": 0.64,
}


def qe_arm(Zx, Zu, y):
    """qe with neighborhoods in THIS space; d'-ratio measured in-space."""
    Zxn, Zun = l2n(Zx), l2n(Zu)
    Zs = qe_smooth(Zxn, Zun)
    ratios, raw_dps, sm_dps = dprime_pair_ratios(Zxn, Zs, y)
    return {
        "mean_ratio": float(ratios.mean()),
        "median_ratio": float(np.median(ratios)),
        "frac_pairs_improved": float((ratios > 1).mean()),
        "mean_raw_dprime": float(raw_dps.mean()),
        "mean_smoothed_dprime": float(sm_dps.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default=os.path.join(
        "output", "from_cluster", "embeddings"))
    ap.add_argument("--out_dir", default=os.path.join("output", "dwt_theory"))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for ds in args.datasets:
        lab_f, pool_f = DATASETS[ds]
        X, y = load_pt(os.path.join(args.emb_dir, lab_f))
        U, _ = load_pt(os.path.join(args.emb_dir, pool_f))
        X, U = l2n(X.astype(np.float64)), l2n(U.astype(np.float64))
        spaces = build_spaces_xu(X, U)

        row = {"raw_qe": qe_arm(*spaces["raw"], y=y),
               "wlw_qe": qe_arm(*spaces["wlw"], y=y),
               "t128w_qe": qe_arm(*spaces["t128w"], y=y),
               "raw_reference_2026_08_13": RAW_DRATIO[ds]}
        results[ds] = row
        print(f"{ds:14} qe d'-ratio:  raw {row['raw_qe']['mean_ratio']:.3f} "
              f"(frac>1 {row['raw_qe']['frac_pairs_improved']:.2f})   "
              f"wlw {row['wlw_qe']['mean_ratio']:.3f} "
              f"(frac>1 {row['wlw_qe']['frac_pairs_improved']:.2f})   "
              f"t128w {row['t128w_qe']['mean_ratio']:.3f} "
              f"(frac>1 {row['t128w_qe']['frac_pairs_improved']:.2f})",
              flush=True)

    path = os.path.join(args.out_dir, "dwt_order_experiment.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
