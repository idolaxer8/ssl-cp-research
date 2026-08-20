"""
R1 -- paper-grade headline experiment (ICLR 2027 Table 2 / Figs 1-2).

Shots-indexed grid (the paper's budget-indexed framing): cal = shots * K,
shots in 2..14, on the 4 main datasets. Arms are the per-regime champion
decomposition:

    raw     identity (no transform)                       -- baseline
    wt      per-regime Whiten/Truncate:
              separable   (cifar100, miniimagenet): pca128_cw, fine k=100
              fine-grained (aircraft, stanford_cars): lw_cluster768, coarse k=10
    qe_wt   alpha-QE pool denoising + the same W/T (qe ON everywhere --
            the fine-grained rows measure the gate's cost directly)

The champion row of Table 2 is assembled in reporting: qe_wt on separable,
wt on fine-grained (qe gated OFF at low PR). Every arm is pool-fit ->
exactly exchangeable; efficiency comparison only.

NCMs: prototype_softmax (champion; pilot-fixed T per arm), prototype_cosine
(theory-faithful), unwhitened_topk_asym (honest on fine-grained).

CHECKPOINT/RESUME (cluster runs get cut off): after EVERY trial the full
state is written atomically to <output_dir>/r1_checkpoint_<dataset>.json
(tmp + os.replace). Re-running the same command resumes: completed cells
are skipped, partially-done cells continue from the next trial. Trial t is
seeded seed + 1000*t (same convention as transform_control_experiment), so
a resumed run produces bit-identical splits to an uninterrupted one.

Usage (cluster, from repo root; one dataset per invocation):
    python src/r1_headline_experiment.py --dataset cifar100 --device cuda \
        --data_dir output/from_cluster/embeddings --n_trials 50
    # after a cutoff: run the SAME command again -> resumes.
See cluster/run_r1_headline.sh for the 4-dataset sequence.
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import (FullConformalPredictor, create_ncm,
                                  PrototypeSoftmaxNCM)
from exchangeable_features import UnlabeledTransform, IdentityTransform

SOFTMAX_NCMS = {"prototype_softmax"}

# per-dataset regime config (pca-rationalization-audit + k-sweep findings):
# separable / high PR -> truncate + FULL-MATRIX LW cluster whiten, fine k;
# fine-grained / low PR -> full-rank LW-cluster whitening, coarse k, no
# truncation. FROZEN 2026-08-20 (user call): the separable champion is
# T->W->D = pca128 -> lw_cluster -> qe-post (arm qe_pca128_lwcw in
# transform_control_experiment), superseding the diagonal pca128_cw form.
PIPELINE_REV = "freeze-2026-08-20"
REGIME = {
    "cifar100":      dict(wt=dict(pca_dim=128, whiten="lw_cluster",
                                  projection="pca", n_clusters=100),
                          qe_in_champion=True),
    "miniimagenet":  dict(wt=dict(pca_dim=128, whiten="lw_cluster",
                                  projection="pca", n_clusters=100),
                          qe_in_champion=True),
    "aircraft":      dict(wt=dict(pca_dim=None, whiten="lw_cluster",
                                  projection=None, n_clusters=10),
                          qe_in_champion=False),
    "stanford_cars": dict(wt=dict(pca_dim=None, whiten="lw_cluster",
                                  projection=None, n_clusters=10),
                          qe_in_champion=False),
}
ARM_NAMES = ["raw", "wt", "qe_wt"]


def load_dataset(data_dir, ds):
    """Labeled X, y and unlabeled pool Xu. stanford_cars has no plain finals
    file -- use the intermediate-layer files' 'final' key (L2-normed)."""
    if ds == "stanford_cars":
        lab = os.path.join(data_dir, "embeddings_stanford_cars_layers.pt")
        unl = os.path.join(data_dir,
                           "embeddings_stanford_cars_unlabeled_layers.pt")
        dl = torch.load(lab, map_location="cpu", weights_only=False)
        X, y = dl["final"].numpy(), dl["labels"].numpy()
        Xu = torch.load(unl, map_location="cpu",
                        weights_only=False)["final"].numpy()
    else:
        lab = os.path.join(data_dir, f"embeddings_{ds}.pt")
        unl = os.path.join(data_dir, f"embeddings_{ds}_unlabeled.pt")
        dl = torch.load(lab, map_location="cpu", weights_only=False)
        X, y = dl["embeddings"].numpy(), dl["labels"].numpy()
        Xu = torch.load(unl, map_location="cpu",
                        weights_only=False)["embeddings"].numpy()
    return X, y, Xu


def pool_participation_ratio(Xu, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xu), min(n, len(Xu)), replace=False)
    Z = Xu[idx] - Xu[idx].mean(axis=0)
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def balanced_split(y, allc, m_cal, m_test, rng):
    ci, ti = [], []
    for c in allc:
        perm = rng.permutation(np.where(y == c)[0])
        ci.append(perm[:m_cal])
        ti.append(perm[m_cal:m_cal + m_test])
    return np.concatenate(ci), np.concatenate(ti)


def random_split(y, cal, test_total, rng):
    idx = rng.permutation(len(y))
    return idx[:cal], idx[cal:cal + test_total]


def build_transform(arm, ds, Xu, args):
    if arm == "raw":
        return IdentityTransform().fit()
    kw = dict(REGIME[ds]["wt"])
    if arm == "qe_wt":
        kw["pre"] = "qe"
        kw["qe_stage"] = "post"     # frozen 08-20: smooth AFTER T->W
        if args.qe_k is not None:
            kw["qe_k"] = args.qe_k
        if args.qe_alpha is not None:
            kw["qe_alpha"] = args.qe_alpha
        if args.qe_hub_gamma is not None:
            kw["qe_hub_gamma"] = args.qe_hub_gamma
    return UnlabeledTransform(**kw).fit(Xu)


def resolve_softmax_T(transform, X, y, allc, args):
    """Pilot-fixed prototype-softmax temperature per (arm, dataset); fixed
    across trials + deterministic in args.seed -> exchangeable + resumable."""
    if str(args.proto_temperature) != "auto":
        return float(args.proto_temperature)
    m = max(4, min(8, max(args.shots)))
    rng = np.random.default_rng(args.seed)
    ci, _ = balanced_split(y, allc, m, 1, rng)
    Zc = transform.transform(X[ci])
    pncm = PrototypeSoftmaxNCM(temperature=None, logit="cosine",
                               allow_nonexchangeable=True).fit(Zc, y[ci])
    return float(pncm._T)


def make_ncm(nm, T):
    if nm in SOFTMAX_NCMS:
        return create_ncm(nm, temperature=T, logit="cosine")
    if nm == "prototype_cosine":
        return create_ncm(nm, logit="cosine")
    return create_ncm(nm, k=5)


# ---------------- checkpoint ----------------

def ckpt_path(args):
    return os.path.join(args.output_dir, f"r1_checkpoint_{args.dataset}.json")


def load_ckpt(args, config_now):
    p = ckpt_path(args)
    if not os.path.exists(p):
        return {"config": config_now, "dataset": args.dataset,
                "done": False, "cells": {}}
    with open(p) as f:
        ck = json.load(f)
    # guard: flags that change cell NUMERICS without changing cell keys must
    # match (else a resume silently mixes configs). Grid lists (shots/arms/
    # ncms/splits) and n_trials may change freely -- they only add, remove,
    # or extend cells.
    keys = ["seed", "alpha", "test_per_class", "proto_temperature",
            "qe_k", "qe_alpha", "qe_hub_gamma", "pipeline_rev"]
    old, new = ck.get("config", {}), config_now
    diff = [k for k in keys if old.get(k) != new.get(k)]
    if diff:
        raise SystemExit(f"[ckpt] config mismatch on {diff} vs {p}; "
                         f"move/delete the checkpoint or match the flags.")
    n_done = sum(len(c["trials"]) for c in ck["cells"].values())
    print(f"[ckpt] resuming from {p}: {len(ck['cells'])} cells, "
          f"{n_done} trials already done")
    return ck


def save_ckpt(ck, args):
    p = ckpt_path(args)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ck, f)
    os.replace(tmp, p)          # atomic: a cutoff never corrupts the file


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="R1 headline: shots-indexed champion arms w/ resume")
    ap.add_argument("--data_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--dataset", default="cifar100", choices=list(REGIME))
    ap.add_argument("--arms", nargs="+", default=ARM_NAMES, choices=ARM_NAMES)
    ap.add_argument("--ncms", nargs="+",
                    default=["prototype_softmax", "prototype_cosine",
                             "unwhitened_topk_asym"])
    ap.add_argument("--splits", nargs="+", default=["balanced_both", "random"],
                    choices=["balanced_both", "random"])
    ap.add_argument("--shots", type=int, nargs="+",
                    default=[2, 4, 6, 8, 10, 12, 14],
                    help="labels per class; cal = shots * K.")
    ap.add_argument("--test_per_class", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--random_min_shots_softmax", type=int, default=8,
                    help="skip softmax NCMs on the random split below this "
                         "many shots (known degenerate corner, CPU-slow).")
    ap.add_argument("--qe_k", type=int, default=None)
    ap.add_argument("--qe_alpha", type=float, default=None)
    ap.add_argument("--qe_hub_gamma", type=float, default=None)
    ap.add_argument("--proto_temperature", default="auto")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="output/r1_headline")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu")
        args.device = "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    X, y, Xu = load_dataset(args.data_dir, args.dataset)
    allc = np.unique(y)
    K = len(allc)
    cls_to_j = {int(c): j for j, c in enumerate(allc)}
    pool_pr = pool_participation_ratio(Xu, seed=args.seed)
    max_shots_avail = min(np.bincount(np.searchsorted(allc, y))) \
        - args.test_per_class
    print(f"{args.dataset}: X={X.shape} K={K} pool={Xu.shape} "
          f"PR={pool_pr:.1f} | shots avail <= {max_shots_avail} | "
          f"device={args.device}")

    config_now = {k: (sorted(v) if isinstance(v, list) else v)
                  for k, v in vars(args).items()}
    config_now["pipeline_rev"] = PIPELINE_REV
    ck = load_ckpt(args, config_now)
    ck["pool_participation_ratio"] = pool_pr
    ck["K"] = K
    ck["qe_in_champion"] = REGIME[args.dataset]["qe_in_champion"]

    for arm in args.arms:
        t0 = time.time()
        tf = build_transform(arm, args.dataset, Xu, args)
        print(f"\n=== arm {arm}: {tf}  [fit {time.time()-t0:.0f}s] ===")
        T_by_ncm = {}
        for nm in args.ncms:
            if nm in SOFTMAX_NCMS:
                T_by_ncm[nm] = resolve_softmax_T(tf, X, y, allc, args)
                print(f"  {nm} fixed T = {T_by_ncm[nm]:.4f}")

        for split in args.splits:
            for shots in args.shots:
                if shots > max_shots_avail:
                    print(f"  [skip] shots={shots} > available "
                          f"{max_shots_avail} (test_per_class reserved)")
                    continue
                cal = shots * K
                for nm in args.ncms:
                    if (split == "random" and nm in SOFTMAX_NCMS
                            and shots < args.random_min_shots_softmax):
                        continue
                    key = f"{arm}|{split}|{nm}|{shots}"
                    cell = ck["cells"].get(key)
                    if cell is None:
                        cell = {"arm": arm, "split": split, "ncm": nm,
                                "shots": shots, "cal": cal, "trials": [],
                                "pooled_cov": [0.0] * K,
                                "pooled_tot": [0] * K,
                                "n_cpu_fallback": 0}
                        if nm in SOFTMAX_NCMS:
                            cell["T"] = T_by_ncm[nm]
                        ck["cells"][key] = cell
                    t_start = len(cell["trials"])
                    if t_start >= args.n_trials:
                        continue
                    if t_start:
                        print(f"  [resume] {key} from trial {t_start}")
                    t0 = time.time()
                    for t in range(t_start, args.n_trials):
                        rng = np.random.default_rng(args.seed + 1000 * t)
                        if split == "balanced_both":
                            ci, ti = balanced_split(y, allc, shots,
                                                    args.test_per_class, rng)
                        else:
                            ci, ti = random_split(y, cal,
                                                  args.test_per_class * K, rng)
                        Xc, yc = tf.transform(X[ci]), y[ci]
                        Xt, yt = tf.transform(X[ti]), y[ti]
                        ncm = make_ncm(nm, T_by_ncm.get(nm))
                        cp = FullConformalPredictor(ncm, alpha=args.alpha)
                        cp.calibrate(Xc, yc, all_classes=allc)
                        try:
                            res = cp.predict(Xt, verbose=False,
                                             device=args.device)
                        except (RuntimeError, ValueError):
                            res = cp.predict(Xt, verbose=False, device="cpu")
                            cell["n_cpu_fallback"] += 1
                        psets = res["prediction_sets"]
                        covered = np.array([yt[i] in psets[i]
                                            for i in range(len(yt))])
                        sizes = np.array([len(s) for s in psets])
                        for c in allc:
                            msk = yt == c
                            if msk.any():
                                j = cls_to_j[int(c)]
                                cell["pooled_cov"][j] += float(
                                    covered[msk].sum())
                                cell["pooled_tot"][j] += int(msk.sum())
                        cell["trials"].append({
                            "cov": float(covered.mean()),
                            "sz": float(sizes.mean()),
                            "csr": float((covered & (sizes == 1)).mean()),
                        })
                        save_ckpt(ck, args)   # survive a cutoff at any trial
                    covs = [tr["cov"] for tr in cell["trials"]]
                    szs = [tr["sz"] for tr in cell["trials"]]
                    tot = np.array(cell["pooled_tot"], dtype=float)
                    pcov = (np.array(cell["pooled_cov"])[tot > 0]
                            / tot[tot > 0])
                    covgap = float(100 * np.mean(np.abs(pcov
                                                        - (1 - args.alpha))))
                    print(f"  [{split:13s}] shots={shots:2d} cal={cal:4d} "
                          f"{nm:20s} cov={np.mean(covs):.4f} "
                          f"sz={np.mean(szs):6.2f}"
                          f"+-{np.std(szs)/np.sqrt(len(szs)):.2f} "
                          f"covgap={covgap:5.2f}pp "
                          f"({time.time()-t0:.0f}s)")
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # final summary rows (checkpoint stays the source of truth)
    rows = []
    for key, cell in ck["cells"].items():
        covs = [tr["cov"] for tr in cell["trials"]]
        szs = [tr["sz"] for tr in cell["trials"]]
        csrs = [tr["csr"] for tr in cell["trials"]]
        tot = np.array(cell["pooled_tot"], dtype=float)
        pcov = np.array(cell["pooled_cov"])[tot > 0] / tot[tot > 0]
        rows.append({
            "arm": cell["arm"], "split": cell["split"], "ncm": cell["ncm"],
            "shots": cell["shots"], "cal": cell["cal"],
            "cov": float(np.mean(covs)), "cov_sd": float(np.std(covs)),
            "sz": float(np.mean(szs)), "sz_sd": float(np.std(szs)),
            "sz_se": float(np.std(szs) / np.sqrt(len(szs))),
            "csr": float(np.mean(csrs)),
            "covgap": float(100 * np.mean(np.abs(pcov - (1 - args.alpha)))),
            "n_trials": len(cell["trials"]),
            "n_cpu_fallback": cell["n_cpu_fallback"],
            "champion": (cell["arm"] == ("qe_wt"
                         if REGIME[args.dataset]["qe_in_champion"]
                         else "wt")),
        })
    ck["done"] = all(r["n_trials"] >= args.n_trials for r in rows) or not rows
    save_ckpt(ck, args)
    out = {"config": config_now, "dataset": args.dataset, "K": K,
           "pool_participation_ratio": pool_pr,
           "qe_in_champion": REGIME[args.dataset]["qe_in_champion"],
           "rows": rows}
    out_json = os.path.join(args.output_dir,
                            f"results_{args.dataset}.json")
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, out_json)
    print(f"\nSaved -> {out_json}  (done={ck['done']})")


if __name__ == "__main__":
    main()
