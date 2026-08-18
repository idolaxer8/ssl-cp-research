"""Where should the PCA+whitening transform be FIT FROM? Pool vs the test set itself.

Context
-------
The unlabeled-pool ablation (output/unlabeled_pool_ablation, findings sec 4) showed
the pool's payoff is overwhelmingly the FEATURE TRANSFORM (PCA-128 + cluster-whiten),
not the MS-CS penalty (cal-centroid M ~= unlabeled-cluster M). So the open question
is narrower: does that transform actually need a separate unlabeled pool, or can the
cal+test points we already have -- used WITHOUT their labels -- fit just as good a
transform?

This is the Silva-Rodriguez / SCA-T (Fan & Sesia 2025) transductive idea: fit a
data-dependent, LABEL-FREE map symmetrically on cal union test. Because the map
depends only on the unordered multiset of feature vectors (never the labels), it is
a fixed map w.r.t. any cal<->test swap, so the pipeline stays EXACTLY exchangeable --
the same guarantee a disjoint pool gives. The test set replaces the pool for free;
the only thing lost is fit-set SIZE (cal+test, typically ~1-2k, vs a 10k pool).

Arms (all: NCM=unwhitened_topk_mean, PCA-128 + cluster-whiten(k=100), plain FCP,
uniform-random split => exactly exchangeable, GPU fast path):

  no-pool          identity feats (raw 768)                 -- reference floor
  transductive     transform fit on cal+test feats (no labels), size = cal+test
  pool-matched     transform fit on a random pool subset,    size = cal+test  (FAIR)
  pool-10k         transform fit on the full unlabeled pool  (optimal size)

transductive vs pool-matched is the apples-to-apples test (identical fit-set size,
only the SOURCE differs: in-distribution cal+test vs disjoint pool). pool-10k shows
the headroom from simply having more unlabeled samples.

Reads embeddings / writes results by ABSOLUTE path into the main checkout's output/
(the .pt files are gitignored, so they live only there, not in this worktree).
"""
# ARCHIVE-CANDIDATE (review 2026-08-24): pool-source question settled (findings §4d, 2026-06-10: cal+test transduction ~= separate pool); no substantive commit since. Re-flagged: the 2026-07-10 sweep's note never merged to main.
import sys, os, time, json, math, argparse

REPO = r"C:\Users\IDO\Desktop\Ido_student\Msc\ssl-cp-research"
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import make_transform

ALPHA = 0.1
NCM = "unwhitened_topk_mean"
PCA_DIM = 128
WHITEN = "cluster"
N_CLUSTERS = 100


def cov_sz_plain(cp, Xt, yt, device):
    res = cp.predict(Xt, device=device, verbose=False)
    sets = res["prediction_sets"]
    cov = float(np.mean([yt[i] in set(sets[i]) for i in range(len(yt))]))
    return cov, float(np.mean(res["set_sizes"]))


def run_fcp(tf, X, y, ci, ti, allc, device):
    Xc, yc = tf.transform(X[ci]), y[ci]
    Xt, yt = tf.transform(X[ti]), y[ti]
    ncm = create_ncm(NCM, k=5)
    cp = FullConformalPredictor(ncm, alpha=ALPHA)
    cp.calibrate(Xc, yc, all_classes=allc)
    return cov_sz_plain(cp, Xt, yt, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--n_init", type=int, default=10,
                    help="k-means n_init for the per-trial transforms (lower => faster).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "output/pool_source_comparison"))
    args = ap.parse_args()

    emb = os.path.join(REPO, "output/from_cluster/embeddings_cifar100.pt")
    unl = os.path.join(REPO, "output/from_cluster/embeddings_cifar100_unlabeled.pt")
    dlab = torch.load(emb, map_location="cpu", weights_only=False)
    X = dlab["embeddings"].numpy().astype(np.float64)
    y = dlab["labels"].numpy()
    Xu = torch.load(unl, map_location="cpu", weights_only=False)["embeddings"].numpy().astype(np.float64)
    allc = np.unique(y)

    ident = make_transform(None)
    # pool-10k: fixed pool => fit once, reuse across all trials/cal sizes.
    pool10k = make_transform(Xu, pca_dim=PCA_DIM, whiten=WHITEN,
                             n_clusters=N_CLUSTERS, n_init=args.n_init)
    print(f"labeled {X.shape} ({len(allc)} classes); pool {Xu.shape}")
    print(f"n_trials={args.n_trials} test={args.test_size} dev={args.device} "
          f"n_init={args.n_init} | transform=PCA-{PCA_DIM}+{WHITEN}-whiten(k={N_CLUSTERS})")

    def split(cal, rng):
        idx = rng.permutation(len(X))
        return idx[:cal], idx[cal:cal + args.test_size]

    # arm order matters only for readability; all share the per-trial split (paired).
    arm_names = ["no-pool", "transductive", "pool-matched", "pool-10k"]
    results = {"config": {"alpha": ALPHA, "ncm": NCM, "pca_dim": PCA_DIM,
                          "whiten": WHITEN, "n_clusters": N_CLUSTERS,
                          "test": args.test_size, "n_trials": args.n_trials,
                          "n_init": args.n_init},
               "arms": {a: {"rows": []} for a in arm_names}}

    for cal in args.cal_sizes:
        fit_size = cal + args.test_size
        acc = {a: {"cov": [], "sz": []} for a in arm_names}
        t0 = time.time()
        for tr in range(args.n_trials):
            rng = np.random.default_rng(2025 + tr)          # split (shared across arms)
            rng_pool = np.random.default_rng(99000 + tr)    # pool subset (independent)
            ci, ti = split(cal, rng)

            # transductive: fit on cal+test features, labels never touched
            feats_ct = np.vstack([X[ci], X[ti]])
            tf_tr = make_transform(feats_ct, pca_dim=PCA_DIM, whiten=WHITEN,
                                   n_clusters=N_CLUSTERS, n_init=args.n_init)
            # pool-matched: random pool subset of the SAME size as cal+test
            sub = rng_pool.choice(len(Xu), fit_size, replace=False)
            tf_pm = make_transform(Xu[sub], pca_dim=PCA_DIM, whiten=WHITEN,
                                   n_clusters=N_CLUSTERS, n_init=args.n_init)

            for name, tf in [("no-pool", ident), ("transductive", tf_tr),
                             ("pool-matched", tf_pm), ("pool-10k", pool10k)]:
                cov, sz = run_fcp(tf, X, y, ci, ti, allc, args.device)
                acc[name]["cov"].append(cov)
                acc[name]["sz"].append(sz)

        tgt = 1 - math.floor(ALPHA * (cal + 1)) / (cal + 1)
        print(f"\ncal={cal} (fit-set size for transductive/pool-matched = {fit_size})  "
              f"[{time.time()-t0:.0f}s, target cov {tgt:.3f}]")
        for name in arm_names:
            cov = np.array(acc[name]["cov"]); sz = np.array(acc[name]["sz"])
            cse = cov.std(ddof=1) / math.sqrt(len(cov))
            sse = sz.std(ddof=1) / math.sqrt(len(sz))
            results["arms"][name]["rows"].append(
                {"cal": cal, "fit_size": fit_size, "n_trials": args.n_trials,
                 "cov": float(cov.mean()), "cov_se": float(cse),
                 "sz": float(sz.mean()), "sz_se": float(sse), "exact_target": tgt})
            print(f"  [{name:13s}] cov={cov.mean():.4f}+/-{cse:.4f}  "
                  f"sz={sz.mean():7.2f}+/-{sse:5.2f}")

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    plot(results, args.out)
    print("\nsaved", os.path.join(args.out, "results.json"))


def plot(results, out):
    styles = {"no-pool": ("#888888", "--", "o"),
              "transductive": ("#ff7f0e", "-", "s"),
              "pool-matched": ("#1f77b4", "-", "^"),
              "pool-10k": ("#2ca02c", "-", "o")}
    fig, (axc, axs) = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, d in results["arms"].items():
        c, ls, mk = styles.get(name, ("k", "-", "o"))
        cals = [r["cal"] for r in d["rows"]]
        cov = [r["cov"] for r in d["rows"]]; cse = [r["cov_se"] for r in d["rows"]]
        sz = [r["sz"] for r in d["rows"]]; sse = [r["sz_se"] for r in d["rows"]]
        axc.errorbar(cals, cov, yerr=cse, marker=mk, color=c, ls=ls, lw=2, capsize=3, label=name)
        axs.errorbar(cals, sz, yerr=sse, marker=mk, color=c, ls=ls, lw=2, capsize=3, label=name)
    axc.axhline(1 - results["config"]["alpha"], ls=":", c="k", lw=1.2, label="target 0.90")
    axc.set(xlabel="calibration size", ylabel="marginal coverage",
            title="Coverage -- all arms exactly exchangeable (incl. transductive)")
    axc.legend(fontsize=8, loc="lower right"); axc.grid(alpha=.3)
    axs.set(xlabel="calibration size", ylabel="avg set size (log)", yscale="log",
            title="Set size -- pool vs cal+test as the transform's fit set")
    axs.legend(fontsize=8); axs.grid(alpha=.3, which="both")
    fig.suptitle("Transform fit source: separate pool vs the test set itself "
                 "(CIFAR-100, DINOv2)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out, "pool_source_comparison.png")
    fig.savefig(p, dpi=140)
    print("saved", p)


if __name__ == "__main__":
    main()
