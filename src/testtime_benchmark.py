"""Test-time-only runtime benchmark (final-results round, 2026-09-07).

Times ONLY what happens when a new test batch arrives, with everything
else prepared beforehand and excluded from the timer:

    prepared (untimed, recorded as calib rows for the caption):
        pool-transform fit + temperature (one-off per dataset),
        FRCP calibrate, probe fit + calibration probs + quantiles
        (splitcp), + pool scoring + merged quantiles (semicp),
        per-fold probe fits + sorted fold calibration scores (cvplus)
    timed test phase per arm:
        splitcp      probe probs on the test batch + set construction
                     (all three scores, both alphas)
        semicp       identical test phase (different scalar quantiles;
                     its pool/NNM cost is calibration-side)
        cvplus       per-fold probe probs on the test batch +
                     searchsorted p-values + sets
        frozen       transform-apply(test) + full-CP GPU sweep
                     (predict(device='cuda'), the closed-form batched
                     path) + sets
        frozen_naive transform-apply(test) + predict(device='cpu'):
                     the plain per-test-point, per-candidate
                     transductive loop, no vectorization -- the naive
                     full CP everyone fears; fewer reps (--n_naive_reps)

The CV+ fit/test split is a mechanical two-phase copy of
headline_experiment.cvplus_pvalues (fold cap, stratified when possible,
p = (1 + #{R_j >= T}) / (n+1)); the math is untouched. On rep 0 the
GPU and naive p-values are asserted equal (the GPU path is bit-exact),
so the naive row times the SAME computation.

Timing is invariant to the temperature source and the shrinkage setting
(T is a scalar divisor, shrinkage only changes matrix entries), so this
benchmark uses the deployed transform + pilot T of this branch.

Run on an OTHERWISE IDLE machine.

Usage:
    python src/testtime_benchmark.py --dataset cifar100 \
        --shots 2 4 6 8 --n_reps 5 --n_naive_reps 3
Output: output/headline/timing_testtime_<dataset>.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, KFold

from conformal_prediction import FullConformalPredictor
from headline_experiment import REGIME, build_frozen_transform, \
    load_dataset, l2n
from r1_headline_experiment import balanced_split, resolve_softmax_T, \
    make_ncm
from g3_semisup_experiment import fit_probe, full_probs
import semicp_port as sp

ALPHAS = (0.1, 0.05)


def cvplus_fit(Z_ci, y_ci, allc, lam, n_folds):
    """Fit phase of headline_experiment.cvplus_pvalues: fold probes +
    per-fold sorted calibration scores. Same splitter/seeds/math."""
    col = {int(c): j for j, c in enumerate(allc)}
    y_idx = np.array([col[int(c)] for c in y_ci])
    n = len(y_ci)
    min_cls = int(np.unique(y_ci, return_counts=True)[1].min())
    folds = max(2, min(n_folds, min_cls))
    if min_cls < folds:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True,
                                   random_state=42)
    clfs, Rs_sorted = [], []
    for tr_i, va_i in splitter.split(Z_ci, y_ci):
        clf = fit_probe(Z_ci[tr_i], y_ci[tr_i], lam=lam)
        P_va = full_probs(clf, Z_ci[va_i], allc)
        Rs_sorted.append(np.sort(
            1.0 - P_va[np.arange(len(va_i)), y_idx[va_i]]))
        clfs.append(clf)
    return clfs, Rs_sorted, n


def cvplus_test(clfs, Rs_sorted, n, Z_ti, allc):
    """Test phase: per-fold test probs + searchsorted accumulation."""
    n_ge = None
    for clf, Rs in zip(clfs, Rs_sorted):
        S_test = 1.0 - full_probs(clf, Z_ti, allc)
        add = len(Rs) - np.searchsorted(Rs, S_test, side="left")
        n_ge = add if n_ge is None else n_ge + add
    return (n_ge + 1.0) / (n + 1.0)


def sets_from_pvalues(res, alphas):
    for a in alphas:
        _ = [{c for c, p in pv.items() if p > a}
             for pv in res["p_values"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--cub_dir", default="output/pca_pilots/heldout_data")
    ap.add_argument("--dataset", default="cifar100", choices=list(REGIME))
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--n_reps", type=int, default=5)
    ap.add_argument("--n_naive_reps", type=int, default=3,
                    help="reps for the frozen_naive (CPU loop) arm")
    ap.add_argument("--test_per_class", type=int, default=5)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--proto_temperature", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("cuda required for the frozen (ours) arm")
    X, y, Xu = load_dataset(args)
    allc = np.unique(y)
    K = len(allc)
    col = {int(c): j for j, c in enumerate(allc)}
    Zl, Zu = l2n(X), l2n(Xu)

    one_off = {}
    t0 = time.perf_counter()
    tf = build_frozen_transform(args.dataset, Xu)
    one_off["transform_fit_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    T_frozen = resolve_softmax_T(tf, X, y, allc, args)
    one_off["T_pilot_s"] = time.perf_counter() - t0
    print(f"{args.dataset}: K={K} pool={len(Xu)} "
          f"n_test={K * args.test_per_class} | one-off: "
          + ", ".join(f"{k}={v:.1f}s" for k, v in one_off.items()))

    rows = []
    checked = False
    for shots in args.shots:
        for rep in range(args.n_reps):
            rng = np.random.default_rng(args.seed + 1000 * rep)
            ci, ti = balanced_split(y, allc, shots, args.test_per_class,
                                    rng)

            # ---------- prepare (untimed test-wise; timed as calib) ----
            t0 = time.perf_counter()
            ncm = make_ncm("prototype_softmax", T_frozen)
            cp = FullConformalPredictor(ncm, alpha=min(ALPHAS))
            cp.calibrate(tf.transform(X[ci]), y[ci], all_classes=allc)
            rows.append(dict(arm="frozen", phase="calib", shots=shots,
                             rep=rep, seconds=time.perf_counter() - t0))

            rng2 = np.random.default_rng(args.seed + 1000 * rep + 17)
            perm = rng2.permutation(ci)
            n_tr = int(round(args.train_frac * len(ci)))
            tr, ca = perm[:n_tr], perm[n_tr:]
            yca_idx = np.array([col[int(c)] for c in y[ca]])

            t0 = time.perf_counter()
            clf = fit_probe(Zl[tr], y[tr], lam=args.lam)
            P_ca = full_probs(clf, Zl[ca], allc)
            q_split = {(sf, a): sp.splitcp_qhat(P_ca, yca_idx, a, sf)
                       for sf in sp.SCORE_FNS for a in ALPHAS}
            rows.append(dict(arm="splitcp", phase="calib", shots=shots,
                             rep=rep, seconds=time.perf_counter() - t0))

            t0 = time.perf_counter()
            P_un = full_probs(clf, Zu, allc)
            q_semi = {}
            for sf in sp.SCORE_FNS:
                s_lab = sp.official_scores_true(P_ca, yca_idx, sf)
                s_unl = sp.uns_adjusted_scores(P_un, P_ca, yca_idx, sf)
                merged = np.concatenate([s_lab, s_unl])
                for a in ALPHAS:
                    q_semi[(sf, a)] = sp.official_qhat(merged, a)
            rows.append(dict(arm="semicp", phase="calib", shots=shots,
                             rep=rep, seconds=time.perf_counter() - t0))

            t0 = time.perf_counter()
            clfs, Rs_sorted, n_cv = cvplus_fit(Zl[ci], y[ci], allc,
                                               args.lam, args.cv_folds)
            rows.append(dict(arm="cvplus", phase="calib", shots=shots,
                             rep=rep, seconds=time.perf_counter() - t0))

            # GPU warm-up (rep 0 only, untimed)
            if rep == 0:
                _ = cp.predict(tf.transform(X[ti]), verbose=False,
                               device="cuda", return_p_values=True)

            # ---------- timed test phases -----------------------------
            t0 = time.perf_counter()
            P_ti = full_probs(clf, Zl[ti], allc)
            for sf in sp.SCORE_FNS:
                for a in ALPHAS:
                    _ = sp.predict_sets(P_ti, q_split[(sf, a)], sf)
            rows.append(dict(arm="splitcp", phase="test", shots=shots,
                             rep=rep, seconds=time.perf_counter() - t0))

            t0 = time.perf_counter()
            P_ti2 = full_probs(clf, Zl[ti], allc)
            for sf in sp.SCORE_FNS:
                for a in ALPHAS:
                    _ = sp.predict_sets(P_ti2, q_semi[(sf, a)], sf)
            rows.append(dict(arm="semicp", phase="test", shots=shots,
                             rep=rep, seconds=time.perf_counter() - t0))

            t0 = time.perf_counter()
            pv = cvplus_test(clfs, Rs_sorted, n_cv, Zl[ti], allc)
            for a in ALPHAS:
                _ = [set(np.flatnonzero(pv[i] > a).tolist())
                     for i in range(len(ti))]
            rows.append(dict(arm="cvplus", phase="test", shots=shots,
                             rep=rep, seconds=time.perf_counter() - t0))

            t0 = time.perf_counter()
            res = cp.predict(tf.transform(X[ti]), verbose=False,
                             device="cuda", return_p_values=True)
            sets_from_pvalues(res, ALPHAS)
            rows.append(dict(arm="frozen", phase="test", shots=shots,
                             rep=rep, seconds=time.perf_counter() - t0))

        # naive pass LAST, after every timed arm of every rep: its
        # minutes-long CPU saturation would otherwise contaminate the
        # measurements that follow it (observed 09-07: frozen medians
        # inflate ~3x when naive reps are interleaved)
        for rep in range(args.n_naive_reps):
            rng = np.random.default_rng(args.seed + 1000 * rep)
            ci, ti = balanced_split(y, allc, shots, args.test_per_class,
                                    rng)
            ncm = make_ncm("prototype_softmax", T_frozen)
            cp = FullConformalPredictor(ncm, alpha=min(ALPHAS))
            cp.calibrate(tf.transform(X[ci]), y[ci], all_classes=allc)
            t0 = time.perf_counter()
            res = cp.predict(tf.transform(X[ti]), verbose=False,
                             device="cpu", return_p_values=True)
            sets_from_pvalues(res, ALPHAS)
            rows.append(dict(arm="frozen_naive", phase="test",
                             shots=shots, rep=rep,
                             seconds=time.perf_counter() - t0))
            if not checked:
                res_g = cp.predict(tf.transform(X[ti]), verbose=False,
                                   device="cuda", return_p_values=True)
                for pg, pc in zip(res_g["p_values"], res["p_values"]):
                    for c in pg:
                        assert abs(pg[c] - pc[c]) < 1e-12, \
                            (shots, c, pg[c], pc[c])
                checked = True
                print(f"  [check] naive(CPU) == ours(GPU) p-values "
                      f"at shots={shots}: PASS")

        done = [r for r in rows if r["shots"] == shots
                and r["phase"] == "test"]
        msg = "  ".join(
            f"{arm}={np.median([r['seconds'] for r in done if r['arm'] == arm]):7.3f}s"
            for arm in ("splitcp", "cvplus", "semicp", "frozen",
                        "frozen_naive"))
        print(f"shots={shots:2d} (median test-phase): {msg}", flush=True)

    out = args.out or f"output/headline/timing_testtime_{args.dataset}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "K": K, "device": "cuda",
                   "n_test": K * args.test_per_class, "pool": len(Xu),
                   "config": vars(args), "one_off": one_off,
                   "rows": rows}, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
