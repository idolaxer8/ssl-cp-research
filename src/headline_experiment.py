"""
R1 headline v2 (user spec 2026-08-22): FROZEN champion pipeline vs published
baselines, shots-indexed, dual alpha. Supersedes r1_headline_experiment.py as
the headline driver (that script remains the champion-decomposition ablation).

Arms
    frozen   full (transductive) CP + prototype_softmax on the frozen T->W->D
             transform (freeze 2026-08-20):
               separable   (cifar100, miniimagenet, cub200):
                   pca128 -> full-matrix LW cluster whiten -> alpha-QE post
                   (qe_pca128_lwcw, qe_stage='post', qe defaults k=10 a=3)
               fine-grained (aircraft, stanford_cars):
                   full-rank LW cluster whiten, NO PCA, qe gated OFF
    splitcp  split CP with a trained softmax probe; official THR/APS/RAPS
             scores, quantile and set rule via semicp_port (boundary class
             EXCLUDED; RAPS penalty=0.01 kreg=2 -- paper defaults)
    cvplus   CV+ (Barber et al. 2021) on the same probe family, THR score,
             min(--cv_folds, shots) stratified folds, ALL labels used (no
             train/cal split -- CV+'s selling point); vectorized port of
             conformal_prediction.CrossValidationPlusPredictor
    semicp   SemiCP (Zhou et al., arXiv 2505.21147, CVPR 2026): split CP
             whose quantile is stabilized by NNM bias-corrected pseudo-scores
             from the unlabeled pool; official-code-faithful (semicp_port)

Protocol: balanced cal/test split (--test_per_class per class), shots in
2..14 labels/class (cal = shots*K), --n_trials trials seeded seed+1000*t
(identical splits across arms -> paired comparison), alphas {0.1, 0.05}
evaluated from ONE scoring pass per trial (FCP/CV+ p-values; per-alpha
q_hat for splitcp/semicp). Metrics per cell x alpha: coverage, mean set
size, class-conditional CovGap (pooled over trials, percentage points).

Fairness notes (documented adaptations vs the original papers):
  * baselines run on L2-normalized RAW backbone embeddings (the published
    methods as-is); only the frozen arm uses the pool-fit transform. The
    unlabeled pool is available to every method that can use one (semicp).
  * one shared probe family (StandardScaler + multinomial logistic
    regression, lam=1e-2, g3 convention). SemiCP's paper rides a frozen
    FULLY-trained classifier -- impossible at our label budgets; the
    few-shot probe is the documented adaptation (their fig. 11 shows NNM
    quality degrades with classifier accuracy).
  * splitcp/semicp split the labeled budget train/cal at train_frac in
    {0.25, 0.5, 0.75}; all fracs are logged, the paper table reports the
    best per cell (strongest-baseline convention).

CHECKPOINT/RESUME: same scheme as r1_headline_experiment.py -- after every
cell-trial the state is written atomically to
<output_dir>/headline_checkpoint_<dataset>.json; re-running the same command
resumes bit-identically (trial t seeded seed + 1000*t).

Usage (one dataset per invocation; see cluster/run_headline.sh):
    python src/headline_experiment.py --dataset cifar100 --device cuda \
        --data_dir output/from_cluster/embeddings --n_trials 50
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import FullConformalPredictor
from exchangeable_features import UnlabeledTransform
from r1_headline_experiment import (balanced_split, random_split,
                                    pool_participation_ratio,
                                    resolve_softmax_T, make_ncm)
from g3_semisup_experiment import fit_probe, full_probs
import semicp_port as sp

# per-dataset frozen-pipeline config (freeze 2026-08-20; cub200 added to the
# separable group by user call 2026-08-22 -- n_clusters extrapolated to K)
REGIME = {
    "cifar100":      dict(kind="separable",    n_clusters=100),
    "miniimagenet":  dict(kind="separable",    n_clusters=100),
    "cub200":        dict(kind="separable",    n_clusters=200),
    "food101":       dict(kind="separable",    n_clusters=101),
    "aircraft":      dict(kind="fine_grained", n_clusters=10),
    "stanford_cars": dict(kind="fine_grained", n_clusters=10),
}
ARM_NAMES = ["frozen", "splitcp", "cvplus", "semicp"]


def build_frozen_transform(ds, Xu, qe="champion"):
    """qe='champion' follows the per-regime freeze (on for separable, off
    for fine-grained); 'on'/'off' force it (W->D ablations)."""
    r = REGIME[ds]
    sep = r["kind"] == "separable"
    kw = dict(whiten="lw_cluster", n_clusters=r["n_clusters"])
    kw.update(dict(pca_dim=128, projection="pca") if sep
              else dict(pca_dim=None, projection=None))
    if sep if qe == "champion" else qe == "on":
        kw.update(pre="qe", qe_stage="post")
    return UnlabeledTransform(**kw).fit(Xu)


def _emb(d, keys=("embeddings", "final")):
    for k in keys:
        if k in d:
            return d[k].numpy()
    raise KeyError(f"none of {keys} in {list(d)}")


def load_dataset(args):
    ds = args.dataset
    if ds == "stanford_cars":
        lab = os.path.join(args.data_dir, "embeddings_stanford_cars_layers.pt")
        unl = os.path.join(args.data_dir,
                           "embeddings_stanford_cars_unlabeled_layers.pt")
    elif ds == "cub200":
        lab = os.path.join(args.cub_dir, "embeddings_cub200.pt")
        unl = os.path.join(args.cub_dir, "embeddings_cub200_unlabeled.pt")
    else:
        lab = os.path.join(args.data_dir, f"embeddings_{ds}.pt")
        unl = os.path.join(args.data_dir, f"embeddings_{ds}_unlabeled.pt")
    dl = torch.load(lab, map_location="cpu", weights_only=False)
    X, y = _emb(dl), dl["labels"].numpy()
    Xu = _emb(torch.load(unl, map_location="cpu", weights_only=False))
    return X, y, Xu


def l2n(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def cvplus_pvalues(Z_ci, y_ci, Z_ti, allc, lam, n_folds):
    """Vectorized CV+ p-values, THR score, same math as
    conformal_prediction.CrossValidationPlusPredictor (fold cap, stratified
    when possible, p = (1 + #{R_j >= R_test^{k(j)}}) / (n+1))."""
    from sklearn.model_selection import StratifiedKFold, KFold
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
    cal_scores = np.empty(n)
    fold_assign = np.empty(n, dtype=int)
    S_test_by_fold = []
    for k, (tr_i, va_i) in enumerate(splitter.split(Z_ci, y_ci)):
        clf = fit_probe(Z_ci[tr_i], y_ci[tr_i], lam=lam)
        P_va = full_probs(clf, Z_ci[va_i], allc)
        cal_scores[va_i] = 1.0 - P_va[np.arange(len(va_i)), y_idx[va_i]]
        S_test_by_fold.append(1.0 - full_probs(clf, Z_ti, allc))
        fold_assign[va_i] = k
    n_ge = np.zeros(S_test_by_fold[0].shape)
    for k in range(folds):
        Rs = np.sort(cal_scores[fold_assign == k])
        # #{R_j >= T} = m_k - #{R_j < T}
        n_ge += len(Rs) - np.searchsorted(Rs, S_test_by_fold[k], side="left")
    return (n_ge + 1.0) / (n + 1.0)


# ---------------- checkpoint (same scheme as r1_headline_experiment) -------

def ckpt_path(args):
    return os.path.join(args.output_dir,
                        f"headline_checkpoint_{args.dataset}.json")


def load_ckpt(args, config_now):
    p = ckpt_path(args)
    if not os.path.exists(p):
        return {"config": config_now, "dataset": args.dataset,
                "done": False, "cells": {}}
    with open(p) as f:
        ck = json.load(f)
    keys = ["seed", "alphas", "test_per_class", "train_fracs", "lam",
            "cv_folds", "raps_penalty", "raps_kreg", "proto_temperature"]
    old, new = ck.get("config", {}), config_now
    diff = [k for k in keys if old.get(k) != new.get(k)]
    if diff:
        raise SystemExit(f"[ckpt] config mismatch on {diff} vs {p}; "
                         f"move/delete the checkpoint or match the flags.")
    n_done = sum(len(c["trials"]) for c in ck["cells"].values())
    print(f"[ckpt] resuming from {p}: {len(ck['cells'])} cells, "
          f"{n_done} cell-trials already done")
    return ck


def save_ckpt(ck, args):
    p = ckpt_path(args)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ck, f)
    os.replace(tmp, p)


# ---------------- cell bookkeeping ----------------

def get_cell(ck, key, meta, K, alpha_keys):
    cell = ck["cells"].get(key)
    if cell is None:
        cell = dict(meta)
        cell.update({"trials": [], "pooled_tot": [0] * K,
                     "pooled_cov": {ak: [0.0] * K for ak in alpha_keys}})
        ck["cells"][key] = cell
    return cell


def record_trial(cell, yt_idx, sets_by_alpha, K):
    """sets_by_alpha: alpha_key -> list of python sets of label INDICES."""
    entry = {}
    for ak, sets in sets_by_alpha.items():
        covered = np.array([yt_idx[i] in sets[i] for i in range(len(yt_idx))])
        sizes = np.array([len(s) for s in sets])
        for j in range(K):
            msk = yt_idx == j
            if msk.any():
                cell["pooled_cov"][ak][j] += float(covered[msk].sum())
        entry[ak] = {"cov": float(covered.mean()),
                     "sz": float(sizes.mean())}
    for j in range(K):                      # test set identical across alphas
        cell["pooled_tot"][j] += int((yt_idx == j).sum())
    cell["trials"].append(entry)


def port_sets(P_test, q_hat, sf, args):
    return [set(s.tolist()) for s in
            sp.predict_sets(P_test, q_hat, sf,
                            penalty=args.raps_penalty, kreg=args.raps_kreg)]


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Headline: frozen champion vs SplitCP/CV+/SemiCP")
    ap.add_argument("--data_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--cub_dir", default="output/pca_pilots/heldout_data",
                    help="dir holding embeddings_cub200{,_unlabeled}.pt")
    ap.add_argument("--dataset", default="cifar100", choices=list(REGIME))
    ap.add_argument("--arms", nargs="+", default=ARM_NAMES, choices=ARM_NAMES)
    ap.add_argument("--scores", nargs="+", default=list(sp.SCORE_FNS),
                    choices=list(sp.SCORE_FNS),
                    help="score functions for splitcp + semicp")
    ap.add_argument("--split", default="balanced_both",
                    choices=["balanced_both", "random"])
    ap.add_argument("--shots", type=int, nargs="+",
                    default=[2, 4, 6, 8, 10, 12, 14])
    ap.add_argument("--test_per_class", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.05])
    ap.add_argument("--train_fracs", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75],
                    help="probe train fraction of the labeled budget")
    ap.add_argument("--lam", type=float, default=1e-2,
                    help="probe ridge strength (g3 convention)")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--raps_penalty", type=float, default=sp.RAPS_PENALTY)
    ap.add_argument("--raps_kreg", type=int, default=sp.RAPS_KREG)
    ap.add_argument("--frozen_qe", default="champion",
                    choices=["champion", "on", "off"],
                    help="override the per-regime qe gate for the frozen "
                         "arm (W->D ablations); non-champion choices get "
                         "tagged cell keys")
    ap.add_argument("--frozen_ncm", default="prototype_softmax",
                    choices=["prototype_softmax", "prototype_cosine",
                             "unwhitened_topk_asym"],
                    help="NCM for the frozen arm; non-default choices get "
                         "their own cell keys (repair/ablation passes). "
                         "KNOWN ISSUE: prototype_softmax auto-T collapses "
                         "(T~0.002 -> one-hot scores -> full sets) on the "
                         "fine-grained full-rank whitened geometry "
                         "(aircraft/cars).")
    ap.add_argument("--proto_temperature", default="auto")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="output/headline")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu")
        args.device = "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    alpha_keys = [f"{a:g}" for a in args.alphas]

    X, y, Xu = load_dataset(args)
    allc = np.unique(y)
    K = len(allc)
    col = {int(c): j for j, c in enumerate(allc)}
    pool_pr = pool_participation_ratio(Xu, seed=args.seed)
    max_shots_avail = min(np.bincount(np.searchsorted(allc, y))) \
        - args.test_per_class
    print(f"{args.dataset}: X={X.shape} K={K} pool={Xu.shape} "
          f"PR={pool_pr:.1f} regime={REGIME[args.dataset]['kind']} | "
          f"shots avail <= {max_shots_avail} | device={args.device}")

    config_now = {k: (sorted(v) if isinstance(v, list) else v)
                  for k, v in vars(args).items()}
    ck = load_ckpt(args, config_now)
    ck["pool_participation_ratio"] = pool_pr
    ck["K"] = K
    ck["regime"] = REGIME[args.dataset]["kind"]

    Zl, Zu = l2n(X), l2n(Xu)
    tf = T_frozen = None
    # non-default frozen NCMs get their own cell keys / arm names so repair
    # passes coexist with the original cells in one checkpoint
    fro_arm = ("frozen" if args.frozen_ncm == "prototype_softmax"
               else f"frozen_{args.frozen_ncm}")
    if args.frozen_qe != "champion":
        fro_arm += f"_qe{args.frozen_qe}"
    if "frozen" in args.arms:
        t0 = time.time()
        tf = build_frozen_transform(args.dataset, Xu, args.frozen_qe)
        if args.frozen_ncm == "prototype_softmax":
            T_frozen = resolve_softmax_T(tf, X, y, allc, args)
            print(f"frozen transform: {tf}  [fit {time.time()-t0:.0f}s] "
                  f"prototype_softmax T={T_frozen:.4f}")
        else:
            print(f"frozen transform: {tf}  [fit {time.time()-t0:.0f}s] "
                  f"ncm={args.frozen_ncm}")

    for shots in args.shots:
        if shots > max_shots_avail:
            print(f"[skip] shots={shots} > available {max_shots_avail}")
            continue
        cal = shots * K
        base = {"split": args.split, "shots": shots, "cal": cal}
        t_shot = time.time()
        for t in range(args.n_trials):
            rng = np.random.default_rng(args.seed + 1000 * t)
            if args.split == "balanced_both":
                ci, ti = balanced_split(y, allc, shots,
                                        args.test_per_class, rng)
            else:
                ci, ti = random_split(y, cal, args.test_per_class * K, rng)
            yt_idx = np.array([col[int(c)] for c in y[ti]])
            yci_idx = np.array([col[int(c)] for c in y[ci]])

            # ---- frozen (full CP, champion NCM, p-values -> both alphas)
            if "frozen" in args.arms:
                cell = get_cell(ck, f"{fro_arm}|{args.split}|{shots}",
                                {**base, "arm": fro_arm, "T": T_frozen,
                                 "ncm": args.frozen_ncm,
                                 "frozen_qe": args.frozen_qe},
                                K, alpha_keys)
                if len(cell["trials"]) <= t:
                    ncm = make_ncm(args.frozen_ncm, T_frozen)
                    cp = FullConformalPredictor(ncm, alpha=min(args.alphas))
                    cp.calibrate(tf.transform(X[ci]), y[ci], all_classes=allc)
                    try:
                        res = cp.predict(tf.transform(X[ti]), verbose=False,
                                         device=args.device,
                                         return_p_values=True)
                    except (RuntimeError, ValueError):
                        res = cp.predict(tf.transform(X[ti]), verbose=False,
                                         device="cpu", return_p_values=True)
                        cell["n_cpu_fallback"] = \
                            cell.get("n_cpu_fallback", 0) + 1
                    sets_by_alpha = {
                        ak: [{col[int(c)] for c, p in pv.items() if p > a}
                             for pv in res["p_values"]]
                        for ak, a in zip(alpha_keys, args.alphas)}
                    record_trial(cell, yt_idx, sets_by_alpha, K)
                    save_ckpt(ck, args)

            # ---- cvplus (all labels, THR, p-values -> both alphas)
            if "cvplus" in args.arms:
                cell = get_cell(ck, f"cvplus|{args.split}|{shots}",
                                {**base, "arm": "cvplus", "score": "THR"},
                                K, alpha_keys)
                if len(cell["trials"]) <= t:
                    pv = cvplus_pvalues(Zl[ci], y[ci], Zl[ti], allc,
                                        args.lam, args.cv_folds)
                    sets_by_alpha = {
                        ak: [set(np.flatnonzero(pv[i] > a).tolist())
                             for i in range(len(ti))]
                        for ak, a in zip(alpha_keys, args.alphas)}
                    record_trial(cell, yt_idx, sets_by_alpha, K)
                    save_ckpt(ck, args)

            # ---- probe train/cal arms (splitcp + semicp share the probe)
            probe_arms = [a for a in ("splitcp", "semicp") if a in args.arms]
            for rt in args.train_fracs:
                cells = {(a, sf): get_cell(
                    ck, f"{a}|{sf}|{rt:g}|{args.split}|{shots}",
                    {**base, "arm": a, "score": sf, "train_frac": rt},
                    K, alpha_keys)
                    for a in probe_arms for sf in args.scores}
                todo = [key for key, c in cells.items()
                        if len(c["trials"]) <= t]
                if not todo:
                    continue
                rng2 = np.random.default_rng(args.seed + 1000 * t
                                             + int(rt * 100) * 17)
                perm = rng2.permutation(ci)
                n_tr = int(round(rt * len(ci)))
                tr, ca = perm[:n_tr], perm[n_tr:]
                yca_idx = np.array([col[int(c)] for c in y[ca]])
                clf = fit_probe(Zl[tr], y[tr], lam=args.lam)
                P_ca = full_probs(clf, Zl[ca], allc)
                P_ti = full_probs(clf, Zl[ti], allc)
                P_un = full_probs(clf, Zu, allc) if "semicp" in probe_arms \
                    else None
                for sf in args.scores:
                    if ("splitcp", sf) in todo:
                        sets_by_alpha = {}
                        for ak, a in zip(alpha_keys, args.alphas):
                            q = sp.splitcp_qhat(P_ca, yca_idx, a, sf,
                                                args.raps_penalty,
                                                args.raps_kreg)
                            sets_by_alpha[ak] = port_sets(P_ti, q, sf, args)
                        record_trial(cells[("splitcp", sf)], yt_idx,
                                     sets_by_alpha, K)
                    if ("semicp", sf) in todo:
                        s_lab = sp.official_scores_true(
                            P_ca, yca_idx, sf, args.raps_penalty,
                            args.raps_kreg)
                        s_unl = sp.uns_adjusted_scores(
                            P_un, P_ca, yca_idx, sf, args.raps_penalty,
                            args.raps_kreg)
                        merged = np.concatenate([s_lab, s_unl])
                        sets_by_alpha = {}
                        for ak, a in zip(alpha_keys, args.alphas):
                            q = sp.official_qhat(merged, a)
                            sets_by_alpha[ak] = port_sets(P_ti, q, sf, args)
                        record_trial(cells[("semicp", sf)], yt_idx,
                                     sets_by_alpha, K)
                save_ckpt(ck, args)
            if t == 0 or (t + 1) % 10 == 0:
                print(f"  shots={shots:2d} trial {t+1}/{args.n_trials} "
                      f"({time.time()-t_shot:.0f}s elapsed)")
        # per-shots progress summary at headline alpha
        ak0 = alpha_keys[0]
        for key in sorted(ck["cells"]):
            c = ck["cells"][key]
            if c["shots"] != shots or not c["trials"]:
                continue
            covs = [tr[ak0]["cov"] for tr in c["trials"]]
            szs = [tr[ak0]["sz"] for tr in c["trials"]]
            print(f"  [{key:42s}] a={ak0} cov={np.mean(covs):.4f} "
                  f"sz={np.mean(szs):7.2f}+-{np.std(szs)/np.sqrt(len(szs)):.2f}")

    # ---- summary rows (checkpoint stays the source of truth)
    rows = []
    for key, c in ck["cells"].items():
        tot = np.array(c["pooled_tot"], dtype=float)
        for ak, a in zip(alpha_keys, args.alphas):
            trs = [tr[ak] for tr in c["trials"] if ak in tr]
            if not trs:
                continue
            covs = [tr["cov"] for tr in trs]
            szs = [tr["sz"] for tr in trs]
            pcov = np.array(c["pooled_cov"][ak])[tot > 0] / tot[tot > 0]
            rows.append({
                "arm": c["arm"], "score": c.get("score"),
                "train_frac": c.get("train_frac"), "split": c["split"],
                "shots": c["shots"], "cal": c["cal"], "alpha": a,
                "cov": float(np.mean(covs)), "cov_sd": float(np.std(covs)),
                "sz": float(np.mean(szs)), "sz_sd": float(np.std(szs)),
                "sz_se": float(np.std(szs) / np.sqrt(len(szs))),
                "covgap": float(100 * np.mean(np.abs(pcov - (1 - a)))),
                "n_trials": len(trs),
                "n_cpu_fallback": c.get("n_cpu_fallback", 0),
            })
    ck["done"] = bool(rows) and all(r["n_trials"] >= args.n_trials
                                    for r in rows)
    save_ckpt(ck, args)
    out = {"config": config_now, "dataset": args.dataset, "K": K,
           "regime": REGIME[args.dataset]["kind"],
           "pool_participation_ratio": pool_pr, "rows": rows}
    out_json = os.path.join(args.output_dir,
                            f"results_{args.dataset}.json")
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, out_json)
    print(f"\nSaved -> {out_json}  (done={ck['done']})")


if __name__ == "__main__":
    main()
