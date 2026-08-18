"""Limits of the transductive (test-set-as-pool) trick: how small can the test
batch get before fitting the transform on cal+test stops paying off? Plus an MS-CS
complement, with the penalty's cluster matrix M sourced from pool vs cal+test.

Follow-up to pool_source_comparison.py. There, at test=1000, fitting the PCA+whiten
transform on cal+test (no labels) recovered most of a 10k pool's benefit and stayed
exactly exchangeable. The open question: the transductive fit-set is cal+test, so as
the TEST batch shrinks below 1k the transform is fit on fewer samples -- where does it
break down, and does the disjoint pool (whose fit-set does NOT shrink) pull ahead?

Fix cal=400, sweep test in {50,100,200,300,500,1000}. Arms (random split => exactly
exchangeable; NCM unwhitened_topk_mean; transform = PCA-128 + cluster-whiten(k=100)):

  PLAIN FCP
    no-pool        identity feats (raw 768)                       -- floor
    transductive   transform fit on cal+test (no labels)          -- shrinks with test
    pool-matched   transform fit on a random pool subset, |cal+test| -- shrinks with test
    pool-10k       transform fit on the full 10k pool             -- flat ceiling

  + MS-CS penalty (lam=0.05, cluster M, exchangeable, GPU)
    transductive+MS-CS  transform AND cluster-M both from cal+test -- fully pool-free
    pool+MS-CS          transform AND cluster-M both from 10k pool -- full-pool ceiling

no-pool and pool-10k are ~flat in test (their fit-set does not depend on test); they
bracket the transductive/pool-matched curves, which rise (degrade) as test shrinks.
transductive vs pool-matched isolates SOURCE at equal size; pool-10k vs pool-matched
isolates SIZE.

Reads embeddings / writes results by ABSOLUTE path into the main checkout's output/.
"""
# ARCHIVE-CANDIDATE (review 2026-08-24): limits arm of the settled pool-source question (findings §4d, 2026-06-10); archive together with pool_source_comparison.py. Re-flagged: the 2026-07-10 sweep's note never merged to main.
import sys, os, time, json, math, argparse

REPO = r"C:\Users\IDO\Desktop\Ido_student\Msc\ssl-cp-research"
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import make_transform
from mscs_unlabeled_experiment import build_cluster_similarity_matrix, run_fcp_with_mscs

ALPHA = 0.1
NCM = "unwhitened_topk_mean"
PCA_DIM = 128
WHITEN = "cluster"
N_CLUSTERS = 100        # for the whitening transform
N_CLUSTERS_MSCS = 20    # for the MS-CS penalty matrix
LAM = 0.05
TAU = -0.5              # negative => effective_tau = 0.5 * median_d^2


def cov_sz_plain(cp, Xt, yt, device):
    res = cp.predict(Xt, device=device, verbose=False)
    sets = res["prediction_sets"]
    cov = float(np.mean([yt[i] in set(sets[i]) for i in range(len(yt))]))
    return cov, float(np.mean(res["set_sizes"]))


def run_plain(tf, Xc_raw, yc, Xt_raw, yt, allc, device):
    Xc, Xt = tf.transform(Xc_raw), tf.transform(Xt_raw)
    ncm = create_ncm(NCM, k=5)
    cp = FullConformalPredictor(ncm, alpha=ALPHA)
    cp.calibrate(Xc, yc, all_classes=allc)
    return cov_sz_plain(cp, Xt, yt, device)


def run_mscs(tf, Xc_raw, yc, Xt_raw, yt, allc, device):
    """MS-CS with cluster-M built in the SAME transformed space the transform defines
    (tf.Xu_transformed_ = transformed fit-set: cal+test for transductive, pool for pool)."""
    Xc, Xt = tf.transform(Xc_raw), tf.transform(Xt_raw)
    (M, c2c, et, _, cc, cn, clc, cld) = build_cluster_similarity_matrix(
        tf.Xu_transformed_, Xc, yc, allc, N_CLUSTERS_MSCS, tau=TAU)
    return run_fcp_with_mscs(
        Xc, yc, Xt, yt, allc, NCM, ALPHA, LAM, M, exchangeable=True, yhat_mode="ncm",
        class_centroids=cc, class_counts=cn, class_to_cluster=c2c,
        cluster_centroids=clc, cluster_dists=cld, effective_tau=et, device=device)


ARMS = ["no-pool", "transductive", "pool-matched", "pool-10k",
        "transductive+MS-CS", "pool+MS-CS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=15)
    ap.add_argument("--cal", type=int, default=400)
    ap.add_argument("--test_sizes", type=int, nargs="+", default=[50, 100, 200, 300, 500, 1000])
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "output/pool_source_limits"))
    args = ap.parse_args()

    emb = os.path.join(REPO, "output/from_cluster/embeddings_cifar100.pt")
    unl = os.path.join(REPO, "output/from_cluster/embeddings_cifar100_unlabeled.pt")
    dlab = torch.load(emb, map_location="cpu", weights_only=False)
    X = dlab["embeddings"].numpy().astype(np.float64); y = dlab["labels"].numpy()
    Xu = torch.load(unl, map_location="cpu", weights_only=False)["embeddings"].numpy().astype(np.float64)
    allc = np.unique(y)

    ident = make_transform(None)
    tf_pool = make_transform(Xu, pca_dim=PCA_DIM, whiten=WHITEN,
                             n_clusters=N_CLUSTERS, n_init=args.n_init)  # fixed => fit once
    print(f"labeled {X.shape} ({len(allc)} classes); pool {Xu.shape}")
    print(f"cal={args.cal} (fixed), test sweep={args.test_sizes}, n_trials={args.n_trials}, "
          f"dev={args.device}, lam={LAM}")

    results = {"config": {"alpha": ALPHA, "ncm": NCM, "cal": args.cal, "lam": LAM,
                          "pca_dim": PCA_DIM, "n_clusters": N_CLUSTERS,
                          "n_clusters_mscs": N_CLUSTERS_MSCS, "n_trials": args.n_trials,
                          "n_init": args.n_init},
               "arms": {a: {"rows": []} for a in ARMS}}

    for test in args.test_sizes:
        fit_size = args.cal + test
        acc = {a: {"cov": [], "sz": []} for a in ARMS}
        t0 = time.time()
        for tr in range(args.n_trials):
            rng = np.random.default_rng(4040 + tr)
            rng_pool = np.random.default_rng(88000 + tr)
            idx = rng.permutation(len(X))
            ci, ti = idx[:args.cal], idx[args.cal:args.cal + test]
            Xci, Xti, yc, yt = X[ci], X[ti], y[ci], y[ti]

            tf_tr = make_transform(np.vstack([Xci, Xti]), pca_dim=PCA_DIM, whiten=WHITEN,
                                   n_clusters=N_CLUSTERS, n_init=args.n_init)
            sub = rng_pool.choice(len(Xu), fit_size, replace=False)
            tf_pm = make_transform(Xu[sub], pca_dim=PCA_DIM, whiten=WHITEN,
                                   n_clusters=N_CLUSTERS, n_init=args.n_init)

            jobs = [("no-pool", run_plain, ident), ("transductive", run_plain, tf_tr),
                    ("pool-matched", run_plain, tf_pm), ("pool-10k", run_plain, tf_pool),
                    ("transductive+MS-CS", run_mscs, tf_tr), ("pool+MS-CS", run_mscs, tf_pool)]
            for name, fn, tf in jobs:
                cov, sz = fn(tf, Xci, yc, Xti, yt, allc, args.device)
                acc[name]["cov"].append(cov); acc[name]["sz"].append(sz)

        tgt = 1 - math.floor(ALPHA * (args.cal + 1)) / (args.cal + 1)
        print(f"\ntest={test:5d} (transductive/pool-matched fit-set={fit_size})  "
              f"[{time.time()-t0:.0f}s, target cov {tgt:.3f}]")
        for name in ARMS:
            cov = np.array(acc[name]["cov"]); sz = np.array(acc[name]["sz"])
            cse = cov.std(ddof=1) / math.sqrt(len(cov)); sse = sz.std(ddof=1) / math.sqrt(len(sz))
            results["arms"][name]["rows"].append(
                {"test": test, "fit_size": fit_size, "cov": float(cov.mean()),
                 "cov_se": float(cse), "sz": float(sz.mean()), "sz_se": float(sse)})
            print(f"  [{name:20s}] cov={cov.mean():.4f}+/-{cse:.4f}  sz={sz.mean():7.2f}+/-{sse:5.2f}")

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    plot(results, args.out)
    print("\nsaved", os.path.join(args.out, "results.json"))


def plot(results, out):
    # color encodes SOURCE, linestyle encodes FCP (solid) vs +MS-CS (dashed)
    styles = {"no-pool":            ("#888888", "--", "o"),
              "transductive":       ("#ff7f0e", "-",  "s"),
              "pool-matched":       ("#1f77b4", "-",  "^"),
              "pool-10k":           ("#2ca02c", "-",  "o"),
              "transductive+MS-CS": ("#ff7f0e", "--", "s"),
              "pool+MS-CS":         ("#2ca02c", "--", "o")}
    cal = results["config"]["cal"]
    fig, (axc, axs) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for name, d in results["arms"].items():
        c, ls, mk = styles[name]
        ts = [r["test"] for r in d["rows"]]
        cov = [r["cov"] for r in d["rows"]]; cse = [r["cov_se"] for r in d["rows"]]
        sz = [r["sz"] for r in d["rows"]]; sse = [r["sz_se"] for r in d["rows"]]
        axc.errorbar(ts, cov, yerr=cse, marker=mk, color=c, ls=ls, lw=2, capsize=3, label=name)
        axs.errorbar(ts, sz, yerr=sse, marker=mk, color=c, ls=ls, lw=2, capsize=3, label=name)
    axc.axhline(1 - results["config"]["alpha"], ls=":", c="k", lw=1.2, label="target 0.90")
    axc.set(xlabel="test-batch size (transform fit-set = cal+test)", ylabel="marginal coverage",
            title=f"Coverage -- valid at every test size (cal={cal}, exact)")
    axc.legend(fontsize=7.5, loc="lower right"); axc.grid(alpha=.3)
    # LINEAR y (no log): set size is in a narrow band at fixed cal, so absolute size reads cleanly
    axs.set(xlabel="test-batch size (transform fit-set = cal+test)", ylabel="avg prediction-set size",
            title=f"Set size (LINEAR) -- transductive degrades as test shrinks (cal={cal})")
    axs.legend(fontsize=7.5); axs.grid(alpha=.3)
    axs.set_ylim(bottom=0)
    fig.suptitle("Limits of the test-set-as-pool trick + MS-CS complement (CIFAR-100, DINOv2)",
                 fontsize=13)
    fig.tight_layout()
    p = os.path.join(out, "pool_source_limits.png")
    fig.savefig(p, dpi=140)
    print("saved", p)


if __name__ == "__main__":
    main()
