"""Pilot: label-free, pool-only SELECTION of the feature transform for CP.

Question: can we rank candidate transforms (PCA-d' / whitening variants) by
their eventual CP set size using ONLY the unlabeled pool -- no labels, no cal
data -- so that selection is exchangeability-free-of-charge?

Selectors tested (all pool-only):
  S1  regime scalars      : PR, top-16 variance share (context, dataset-level)
  S2  margin-tail         : near-tie mass of the nearest/second-nearest
                            pseudo-class-centroid distance ratio in the
                            candidate space (CP-native contrast statistic)
  S3  rehearsal           : actual split CP at deployment (alpha, cal budget)
                            on the pseudo-labeled pool, centroid-ratio score;
                            selector value = pseudo mean set size

Fixed pseudo-task: MiniBatchKMeans with K = |label space| (known a priori) on
the raw L2-normed pool; the SAME pseudo-labels are used for every candidate so
we compare transforms on one external task. Transform + centroids are fit on
pool half A; every statistic is evaluated on half B (no self-fit optimism).

Ground truth: measured FCP set sizes from output/pca_pilots/transform_controls
(cal=800 balanced) and the cub200 heldout run (cal=1600). We report, per
dataset x NCM: Spearman(selector, measured sz), the selector's champion pick,
and its regret = sz(pick)/sz(champion) - 1.

Run (from this worktree):
  python src/transform_selection_pilot.py --datasets cifar100 aircraft
Results -> <repo>/output/transform_selection_pilot/ in the MAIN checkout.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from exchangeable_features import UnlabeledTransform, IdentityTransform  # noqa: E402

MAIN_REPO = r"C:\Users\IDO\Desktop\Ido_student\Msc\ssl-cp-research"

ARMS = {
    "raw768":        None,
    "rp128":         dict(pca_dim=128, whiten=None,          projection="random"),
    "rp128_cw":      dict(pca_dim=128, whiten="cluster",     projection="random"),
    "pca128":        dict(pca_dim=128, whiten=None,          projection="pca"),
    "pca128_cw":     dict(pca_dim=128, whiten="cluster",     projection="pca"),
    "pca512_cw":     dict(pca_dim=512, whiten="cluster",     projection="pca"),
    "lw_global768":  dict(pca_dim=None, whiten="lw_global",  projection=None),
    "lw_cluster768": dict(pca_dim=None, whiten="lw_cluster", projection=None),
    "cw768":         dict(pca_dim=None, whiten="cluster",    projection="center"),
}

DATASETS = {
    # name: (pool_path, K, gt_file, gt_cal)
    "cifar100": ("output/from_cluster/embeddings/embeddings_cifar100_unlabeled.pt",
                 100, "output/pca_pilots/transform_controls/results_cifar100.json", 800),
    "miniimagenet": ("output/from_cluster/embeddings/embeddings_miniimagenet_unlabeled.pt",
                    100, "output/pca_pilots/transform_controls/results_miniimagenet.json", 800),
    "aircraft": ("output/from_cluster/embeddings/embeddings_aircraft_unlabeled.pt",
                 100, "output/pca_pilots/transform_controls/results_aircraft.json", 800),
    "cifar10": ("output/from_cluster/embeddings/embeddings_cifar10_unlabeled.pt",
                10, "output/pca_pilots/transform_controls/results_cifar10.json", 800),
    "cub200": ("output/pca_pilots/heldout_data/embeddings_cub200_unlabeled.pt",
               200, "output/pca_pilots/heldout/results_cub200_heldout.json", 1600),
}


def load_pool(path):
    import torch
    d = torch.load(os.path.join(MAIN_REPO, path), map_location="cpu",
                   weights_only=False)
    X = (d["embeddings"] if isinstance(d, dict) else d).float().numpy()
    return np.ascontiguousarray(X, dtype=np.float64)


def load_gt(gt_file, gt_cal):
    d = json.load(open(os.path.join(MAIN_REPO, gt_file)))
    out = {}
    for r in d["rows"]:
        if r["split"] == "balanced_both" and r["cal"] == gt_cal:
            out.setdefault(r["ncm"], {})[r["arm"]] = r["sz"]
    return out


def l2n(X, eps=1e-12):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def pseudo_labels(X_raw, K, seed):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=5,
                         batch_size=1024).fit(l2n(X_raw))
    return km.labels_


def centroid_dists(Zb, mus):
    """cosine distance (1 - cos) of each row of Zb to each centroid."""
    Zn, Mn = l2n(Zb), l2n(mus)
    return 1.0 - Zn @ Mn.T


def fit_arm(arm, Xa, n_clusters_whiten, seed):
    spec = ARMS[arm]
    if spec is None:
        return IdentityTransform().fit()
    return UnlabeledTransform(n_clusters=n_clusters_whiten, random_state=seed,
                              **spec).fit(Xa)


def margin_stats(D):
    """D = point-to-centroid distance matrix. margin = d1/d2 in [0,1]; near 1
    = near-tie (the mass CP pays for)."""
    part = np.partition(D, 1, axis=1)
    m = part[:, 0] / (part[:, 1] + 1e-12)
    return {
        "margin_mean": float(m.mean()),
        "margin_q90": float(np.quantile(m, 0.90)),
        "neartie_frac_09": float((m > 0.9).mean()),
    }


def rehearsal_setsize(D, yb, K, cal_budget, alpha, n_rep, seed):
    """Split-CP dress rehearsal with the centroid-ratio score
    s(x,y) = d(x, mu_y) / min_{c != y} d(x, mu_c)   (mirrors the asym ratio NCM).
    Balanced pseudo-cal of ~cal_budget points; rest = pseudo-test."""
    rng = np.random.default_rng(seed)
    n = len(yb)
    m_cal = max(1, cal_budget // K)
    dmin = D.min(axis=1)
    # score matrix for all points x all candidate labels
    S = np.empty_like(D)
    for c in range(K):
        other = np.delete(D, c, axis=1).min(axis=1)
        S[:, c] = D[:, c] / (other + 1e-12)
    sizes = []
    for _ in range(n_rep):
        cal_idx = []
        for c in range(K):
            pc = np.where(yb == c)[0]
            if len(pc) == 0:
                continue
            cal_idx.append(rng.choice(pc, min(m_cal, len(pc)), replace=False))
        cal_idx = np.concatenate(cal_idx)
        test_mask = np.ones(n, bool)
        test_mask[cal_idx] = False
        cal_scores = S[cal_idx, yb[cal_idx]]
        k = int(np.ceil((len(cal_scores) + 1) * (1 - alpha)))
        if k > len(cal_scores):
            qhat = np.inf
        else:
            qhat = np.sort(cal_scores)[k - 1]
        sizes.append(float((S[test_mask] <= qhat).sum(axis=1).mean()))
    return float(np.mean(sizes))


def run_dataset(ds, args):
    path, K, gt_file, gt_cal = DATASETS[ds]
    gt = load_gt(gt_file, gt_cal)
    arms = [a for a in ARMS if all(a in m for m in gt.values())]
    X = load_pool(path)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    A, B = perm[: len(X) // 2], perm[len(X) // 2:]
    Xa, Xb = X[A], X[B]

    # rung 1 context: pool PR + top-16 variance share (full pool)
    Z = X - X.mean(0, keepdims=True)
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    top16 = float(ev[:16].sum() / ev.sum())

    # fixed pseudo-task from raw half A, assigned to half B
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=K, random_state=args.seed, n_init=5,
                         batch_size=1024).fit(l2n(Xa))
    ya, yb = km.labels_, km.predict(l2n(Xb))

    rows = {}
    for arm in arms:
        t = fit_arm(arm, Xa, args.n_clusters_whiten, args.seed)
        Za, Zb = t.transform(Xa), t.transform(Xb)
        mus = np.stack([
            Za[ya == c].mean(0) if (ya == c).any() else np.zeros(Za.shape[1])
            for c in range(K)])
        D = centroid_dists(Zb, mus)
        row = margin_stats(D)
        row["rehearsal_sz"] = rehearsal_setsize(
            D, yb, K, gt_cal, args.alpha, args.n_rep, args.seed)
        for ncm, m in gt.items():
            row[f"gt_sz_{'geo' if 'topk' in ncm else 'proto'}"] = m[arm]
        rows[arm] = row
        print(f"  {arm:<14} margin_q90={row['margin_q90']:.3f} "
              f"neartie={row['neartie_frac_09']:.3f} "
              f"rehearsal={row['rehearsal_sz']:6.2f}  "
              f"gt_geo={row.get('gt_sz_geo', float('nan')):6.2f} "
              f"gt_proto={row.get('gt_sz_proto', float('nan')):6.2f}",
              flush=True)
    return {"K": K, "pr": pr, "top16_var": top16, "gt_cal": gt_cal,
            "n_pool": len(X), "arms": rows}


def evaluate(ds, res):
    from scipy.stats import spearmanr
    arms = list(res["arms"])
    out = {}
    for gt_key in ("gt_sz_geo", "gt_sz_proto"):
        if gt_key not in res["arms"][arms[0]]:
            continue
        sz = np.array([res["arms"][a][gt_key] for a in arms])
        champ = arms[int(np.argmin(sz))]
        rep = {}
        for sel in ("margin_q90", "neartie_frac_09", "margin_mean",
                    "rehearsal_sz"):
            v = np.array([res["arms"][a][sel] for a in arms])
            pick = arms[int(np.argmin(v))]
            rep[sel] = {
                "spearman": float(spearmanr(v, sz).correlation),
                "pick": pick,
                "regret": float(res["arms"][pick][gt_key] / sz.min() - 1),
            }
        out[gt_key] = {"champion": champ, "selectors": rep}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_rep", type=int, default=5)
    ap.add_argument("--n_clusters_whiten", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(
        MAIN_REPO, "output", "transform_selection_pilot"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_res, all_eval = {}, {}
    for ds in args.datasets:
        print(f"== {ds} ==", flush=True)
        res = run_dataset(ds, args)
        print(f"  [pool] PR={res['pr']:.1f} top16_var={res['top16_var']:.2f}")
        all_res[ds] = res
        all_eval[ds] = evaluate(ds, res)
        for gt_key, ev in all_eval[ds].items():
            print(f"  [{gt_key}] champion={ev['champion']}")
            for sel, r in ev["selectors"].items():
                print(f"    {sel:<16} rho={r['spearman']:+.2f} "
                      f"pick={r['pick']:<14} regret={r['regret']:+.1%}")
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"config": vars(args), "results": all_res,
                   "eval": all_eval}, f, indent=1)
    print("saved ->", os.path.join(args.out_dir, "results.json"))


if __name__ == "__main__":
    main()
