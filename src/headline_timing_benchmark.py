"""Per-arm compute-time benchmark for the R1 headline comparison.

Times each arm STANDALONE per trial, on the exact headline protocol
(balanced split, shots x K cal, 5*K test, trial seeds seed+1000*t):

    frozen   transform-apply (cal+test) + full-CP calibrate + predict
             (GPU fast path, p-values for both alphas -- one pass, as in
             the headline driver); the one-off pool transform fit and the
             prototype-softmax T pilot are reported separately
    splitcp  probe fit (train_frac 0.5) + THR/APS/RAPS scoring, quantiles
             (both alphas) + set construction
    cvplus   min(5, shots) stratified fold probe fits + vectorized CV+
             p-values + sets
    semicp   probe fit + pool scoring + NNM bias correction + merged
             quantiles + sets (THR/APS/RAPS)

Probe fits are re-timed inside each arm that needs one (standalone
deployment cost, not the shared-fit optimization the driver uses).

Run on an OTHERWISE IDLE machine -- concurrent GPU/CPU load distorts it.

Usage:
    python src/headline_timing_benchmark.py --dataset cifar100 \
        --shots 2 8 14 --n_reps 5
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import FullConformalPredictor
from headline_experiment import (REGIME, build_frozen_transform,
                                 load_dataset, cvplus_pvalues, l2n)
from r1_headline_experiment import balanced_split, resolve_softmax_T, \
    make_ncm
from g3_semisup_experiment import fit_probe, full_probs
import semicp_port as sp

ALPHAS = (0.1, 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--cub_dir", default="output/pca_pilots/heldout_data")
    ap.add_argument("--dataset", default="cifar100", choices=list(REGIME))
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 8, 14])
    ap.add_argument("--n_reps", type=int, default=5)
    ap.add_argument("--test_per_class", type=int, default=5)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--frozen_ncm", default=None,
                    help="default: prototype_softmax on separable, "
                         "unwhitened_topk_asym on fine-grained (the "
                         "headline-table arms)")
    ap.add_argument("--proto_temperature", default="auto")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    sep = REGIME[args.dataset]["kind"] == "separable"
    fro_ncm = args.frozen_ncm or ("prototype_softmax" if sep
                                  else "unwhitened_topk_asym")

    X, y, Xu = load_dataset(args)
    allc = np.unique(y)
    K = len(allc)
    col = {int(c): j for j, c in enumerate(allc)}
    Zl, Zu = l2n(X), l2n(Xu)

    one_off = {}
    t0 = time.perf_counter()
    tf = build_frozen_transform(args.dataset, Xu)
    one_off["transform_fit_s"] = time.perf_counter() - t0
    T_frozen = None
    if fro_ncm == "prototype_softmax":
        t0 = time.perf_counter()
        T_frozen = resolve_softmax_T(tf, X, y, allc, args)
        one_off["T_pilot_s"] = time.perf_counter() - t0
    print(f"{args.dataset}: K={K} pool={len(Xu)} device={args.device} "
          f"frozen_ncm={fro_ncm} | one-off: "
          + ", ".join(f"{k}={v:.1f}s" for k, v in one_off.items()))

    rows = []
    for shots in args.shots:
        for rep in range(args.n_reps):
            rng = np.random.default_rng(args.seed + 1000 * rep)
            ci, ti = balanced_split(y, allc, shots, args.test_per_class,
                                    rng)
            yci_idx = np.array([col[int(c)] for c in y[ci]])

            # frozen (GPU warm-up rep not excluded; reps are averaged)
            t0 = time.perf_counter()
            ncm = make_ncm(fro_ncm, T_frozen)
            cp = FullConformalPredictor(ncm, alpha=min(ALPHAS))
            cp.calibrate(tf.transform(X[ci]), y[ci], all_classes=allc)
            res = cp.predict(tf.transform(X[ti]), verbose=False,
                             device=args.device, return_p_values=True)
            for a in ALPHAS:
                _ = [{c for c, p in pv.items() if p > a}
                     for pv in res["p_values"]]
            rows.append(dict(arm="frozen", shots=shots, rep=rep,
                             seconds=time.perf_counter() - t0))

            # cvplus
            t0 = time.perf_counter()
            pv = cvplus_pvalues(Zl[ci], y[ci], Zl[ti], allc, args.lam,
                                args.cv_folds)
            for a in ALPHAS:
                _ = [set(np.flatnonzero(pv[i] > a).tolist())
                     for i in range(len(ti))]
            rows.append(dict(arm="cvplus", shots=shots, rep=rep,
                             seconds=time.perf_counter() - t0))

            # splitcp (probe fit + all three scores, both alphas)
            rng2 = np.random.default_rng(args.seed + 1000 * rep + 17)
            perm = rng2.permutation(ci)
            n_tr = int(round(args.train_frac * len(ci)))
            tr, ca = perm[:n_tr], perm[n_tr:]
            yca_idx = np.array([col[int(c)] for c in y[ca]])
            t0 = time.perf_counter()
            clf = fit_probe(Zl[tr], y[tr], lam=args.lam)
            P_ca, P_ti = full_probs(clf, Zl[ca], allc), \
                full_probs(clf, Zl[ti], allc)
            for sf in sp.SCORE_FNS:
                for a in ALPHAS:
                    q = sp.splitcp_qhat(P_ca, yca_idx, a, sf)
                    _ = sp.predict_sets(P_ti, q, sf)
            t_split = time.perf_counter() - t0
            rows.append(dict(arm="splitcp", shots=shots, rep=rep,
                             seconds=t_split))

            # semicp (own probe fit + pool scoring + NNM, all scores)
            t0 = time.perf_counter()
            clf = fit_probe(Zl[tr], y[tr], lam=args.lam)
            P_ca, P_ti = full_probs(clf, Zl[ca], allc), \
                full_probs(clf, Zl[ti], allc)
            P_un = full_probs(clf, Zu, allc)
            for sf in sp.SCORE_FNS:
                s_lab = sp.official_scores_true(P_ca, yca_idx, sf)
                s_unl = sp.uns_adjusted_scores(P_un, P_ca, yca_idx, sf)
                merged = np.concatenate([s_lab, s_unl])
                for a in ALPHAS:
                    q = sp.official_qhat(merged, a)
                    _ = sp.predict_sets(P_ti, q, sf)
            rows.append(dict(arm="semicp", shots=shots, rep=rep,
                             seconds=time.perf_counter() - t0))
        done = [r for r in rows if r["shots"] == shots]
        msg = "  ".join(
            f"{arm}={np.mean([r['seconds'] for r in done if r['arm'] == arm]):6.2f}s"
            for arm in ("frozen", "cvplus", "splitcp", "semicp"))
        print(f"shots={shots:2d} (mean/trial over {args.n_reps} reps): {msg}")

    out = args.out or f"output/headline/timing_{args.dataset}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "K": K, "device": args.device,
                   "frozen_ncm": fro_ncm, "n_test": len(allc) * args.test_per_class,
                   "pool": len(Xu), "config": vars(args),
                   "one_off": one_off, "rows": rows}, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
