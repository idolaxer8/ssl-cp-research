"""High-trial, error-bar rerun of the unlabeled-pool ablation.

The 10-trial ablation showed an apparent monotonic coverage decline to ~0.88 at
cal=800. A 100-trial diagnostic proved that was Monte-Carlo noise: plain FCP sits
exactly at the exchangeable target 1-floor(a(n+1))/(n+1) ~ 0.900 at every cal.
This re-runs ALL FOUR exchangeable arms with many trials and reports coverage and
set size with standard errors, to (a) confirm the two MS-CS penalty arms are also
valid at every cal (incl. 800) and (b) replace the noisy 10-trial set sizes with
tight estimates + error bars.

Arms (uniform-random split => exactly exchangeable; NCM unwhitened_topk_mean):
  no-pool FCP            identity feats, plain FCP             (GPU)
  pool FCP               PCA-128+cluster-whiten feats          (GPU)
  no-pool centroid-MSCS  identity feats, cal-centroid M        (CPU)
  pool cluster-MSCS      PCA+whiten feats, unlabeled-cluster M (CPU)
"""
# ARCHIVE-CANDIDATE (review 2026-08-24): high-trial rerun recorded in findings §4e (all four arms valid at every cal; pool cuts sets 48/54/29%); question settled, no §10 item pending, no substantive commit since 2026-06-10.
import sys, os, time, json, math, argparse
sys.path.insert(0, "src")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import make_transform
from mscs_unlabeled_experiment import (
    build_cluster_similarity_matrix, build_centroid_similarity_matrix,
    update_centroid_M_for_candidate, run_fcp_with_mscs)

ALPHA = 0.1; NCM = "unwhitened_topk_mean"; TAU = -0.5; LAM = 0.05


def cov_sz_plain(cp, Xt, yt, device):
    res = cp.predict(Xt, device=device, verbose=False)
    sets = res["prediction_sets"]
    cov = float(np.mean([yt[i] in set(sets[i]) for i in range(len(yt))]))
    return cov, float(np.mean(res["set_sizes"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials_plain", type=int, default=100)
    ap.add_argument("--n_trials_mscs", type=int, default=50)
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="output/pool_ablation_hightrial")
    ap.add_argument("--embeddings_path", type=str,
                    default="output/from_cluster/embeddings_cifar100.pt")
    ap.add_argument("--unlabeled_path", type=str,
                    default="output/from_cluster/embeddings_cifar100_unlabeled.pt")
    args = ap.parse_args()

    emb = args.embeddings_path
    unl = args.unlabeled_path
    dlab = torch.load(emb, map_location="cpu", weights_only=False)
    X = dlab["embeddings"].numpy().astype(np.float64); y = dlab["labels"].numpy()
    Xu = torch.load(unl, map_location="cpu", weights_only=False)["embeddings"].numpy().astype(np.float64)
    allc = np.unique(y)
    ident = make_transform(None)
    pool = make_transform(Xu, pca_dim=128, whiten="cluster", n_clusters=100)
    print(f"labeled {X.shape}, {len(allc)} classes; pool {Xu.shape}")
    print(f"plain={args.n_trials_plain} (dev={args.device}), mscs={args.n_trials_mscs} (CPU), test={args.test_size}")

    def split(cal, rng):
        idx = rng.permutation(len(X)); return idx[:cal], idx[cal:cal + args.test_size]

    arms = [("no-pool FCP", ident, "plain"),
            ("pool FCP", pool, "plain"),
            ("no-pool centroid-MSCS", ident, "centroid"),
            ("pool cluster-MSCS", pool, "cluster")]

    results = {"config": {"alpha": ALPHA, "ncm": NCM, "lam": LAM, "tau": TAU,
                          "test": args.test_size, "n_trials_plain": args.n_trials_plain,
                          "n_trials_mscs": args.n_trials_mscs}, "arms": {}}

    for name, tf, kind in arms:
        ntr = args.n_trials_plain if kind == "plain" else args.n_trials_mscs
        results["arms"][name] = {"kind": kind, "rows": []}
        for cal in args.cal_sizes:
            covs, szs = [], []; t0 = time.time()
            for tr in range(ntr):
                rng = np.random.default_rng(777 + tr)
                ci, ti = split(cal, rng)
                Xc, yc = tf.transform(X[ci]), y[ci]
                Xt, yt = tf.transform(X[ti]), y[ti]
                if kind == "plain":
                    ncm = create_ncm(NCM, k=5)
                    cp = FullConformalPredictor(ncm, alpha=ALPHA)
                    cp.calibrate(Xc, yc, all_classes=allc)
                    cov, sz = cov_sz_plain(cp, Xt, yt, args.device)
                elif kind == "centroid":
                    (M, et, _, cc, cn) = build_centroid_similarity_matrix(Xc, yc, allc, tau=TAU)
                    upd = (lambda yi, xt, Mb, cc=cc, cn=cn, et=et:
                           update_centroid_M_for_candidate(cc, cn, et, yi, xt, Mb))
                    cov, sz = run_fcp_with_mscs(Xc, yc, Xt, yt, allc, NCM, ALPHA, LAM, M,
                                                exchangeable=True, yhat_mode="ncm",
                                                class_centroids=cc, class_counts=cn,
                                                effective_tau=et, update_M_fn=upd)
                else:
                    (M, c2c, et, _, cc, cn, clc, cld) = build_cluster_similarity_matrix(
                        tf.Xu_transformed_, Xc, yc, allc, 20, tau=TAU)
                    cov, sz = run_fcp_with_mscs(Xc, yc, Xt, yt, allc, NCM, ALPHA, LAM, M,
                                                exchangeable=True, yhat_mode="ncm",
                                                class_centroids=cc, class_counts=cn,
                                                class_to_cluster=c2c, cluster_centroids=clc,
                                                cluster_dists=cld, effective_tau=et)
                covs.append(cov); szs.append(sz)
            covs, szs = np.array(covs), np.array(szs)
            cse = covs.std(ddof=1) / math.sqrt(ntr); sse = szs.std(ddof=1) / math.sqrt(ntr)
            tgt = 1 - math.floor(ALPHA * (cal + 1)) / (cal + 1)
            results["arms"][name]["rows"].append(
                {"cal": cal, "n_trials": ntr, "cov": float(covs.mean()), "cov_se": float(cse),
                 "sz": float(szs.mean()), "sz_se": float(sse), "exact_target": tgt})
            print(f"  [{name:22s}] cal={cal:4d} cov={covs.mean():.4f}+/-{cse:.4f} (tgt {tgt:.4f}) sz={szs.mean():6.2f}+/-{sse:4.2f}  ({time.time()-t0:.0f}s)")

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    plot(results, args.out)
    print("saved", os.path.join(args.out, "results.json"))


def plot(results, out):
    styles = {"no-pool FCP": ("#888888", "--"), "pool FCP": ("#1f77b4", "-"),
              "no-pool centroid-MSCS": ("#d62728", ":"), "pool cluster-MSCS": ("#2ca02c", "-")}
    fig, (axc, axs) = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, d in results["arms"].items():
        c, ls = styles.get(name, ("k", "-"))
        cals = [r["cal"] for r in d["rows"]]
        cov = [r["cov"] for r in d["rows"]]; cse = [r["cov_se"] for r in d["rows"]]
        sz = [r["sz"] for r in d["rows"]]; sse = [r["sz_se"] for r in d["rows"]]
        ntr = d["rows"][0]["n_trials"]
        axc.errorbar(cals, cov, yerr=cse, marker="o", color=c, ls=ls, lw=2, capsize=3, label=f"{name} (n={ntr})")
        axs.errorbar(cals, sz, yerr=sse, marker="o", color=c, ls=ls, lw=2, capsize=3, label=f"{name} (n={ntr})")
    axc.axhline(1 - results["config"]["alpha"], ls=":", c="k", lw=1.2, label="target 0.90")
    axc.set(xlabel="calibration size", ylabel="marginal coverage",
            title="Coverage +/- 1 SE -- valid at every cal (incl. 800)")
    axc.legend(fontsize=8, loc="lower right"); axc.grid(alpha=.3)
    axs.set(xlabel="calibration size", ylabel="avg set size (log)", yscale="log",
            title="Set size +/- 1 SE -- high-trial estimates")
    axs.legend(fontsize=8); axs.grid(alpha=.3, which="both")
    fig.suptitle("Unlabeled-pool ablation -- high-trial validity + efficiency (CIFAR-100, DINOv2)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pool_ablation_hightrial.png"), dpi=140)
    print("saved", os.path.join(out, "pool_ablation_hightrial.png"))


if __name__ == "__main__":
    main()
