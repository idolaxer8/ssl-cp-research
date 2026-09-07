"""Configuration-sensitivity sweep for Appendix D (2026-09-07).

Two knob-pair grids around the deployed configuration, everything else
held at the deploy (whiten='lw_cluster' with shrinkage forced to zero,
qe post, pool-resolved T, prototype-softmax full CP, non-empty
convention):

    kg  smoothing neighborhood kappa (qe_k) x weight exponent gamma
        (qe_alpha), at d'=128, C=K
    dc  truncation dimension d' (pca_dim) x whitening group count C
        (n_clusters, in multiples of K), at kappa=10, gamma=3

Per cell: fit the transform on the pool once, resolve the pool-margin
T (whitening k-means groups as pseudo-classes), then run n_trials
headline-protocol trials (balanced split, 5 test/class, seed+1000t)
of FRCP at --shots labels per class. Reported metric: mean filled set
size sz1 (empty sets receive the max-p-value label) at alpha=0.1.

Usage (one dataset per invocation):
    python src/config_sweep_heatmaps.py --dataset cifar100 \
        --data_dir output/from_cluster/embeddings
Output: output/config_heatmaps_0907/results_<ds>.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import FullConformalPredictor, \
    PrototypeSoftmaxNCM
from exchangeable_features import UnlabeledTransform
from headline_experiment import REGIME, load_dataset
from r1_headline_experiment import balanced_split, make_ncm

ALPHA = 0.1
KG_GRID = dict(kappa=[3, 5, 10, 20, 40], gamma=[1.0, 2.0, 3.0, 5.0])
DC_GRID = dict(dprime=[32, 64, 128, 256, 512], cmult=[0.5, 1.0, 2.0, 4.0])
DEPLOY = dict(kappa=10, gamma=3.0, dprime=128, cmult=1.0)


def make_tf(Xu, dprime, C, kappa, gamma):
    return UnlabeledTransform(
        pca_dim=dprime, projection="pca", whiten="lw_cluster",
        n_clusters=C, pre="qe", qe_stage="post", qe_k=kappa,
        qe_alpha=gamma, lw_shrinkage_force=0.0).fit(Xu)


def pool_T(tf):
    pncm = PrototypeSoftmaxNCM(temperature=None, logit="cosine",
                               allow_nonexchangeable=True
                               ).fit(tf.Xu_transformed_, tf.kmeans_.labels_)
    return float(pncm._T)


def eval_cell(tf, T, X, y, allc, args):
    col = {int(c): j for j, c in enumerate(allc)}
    szs, sz1s, cov1s = [], [], []
    for t in range(args.n_trials):
        rng = np.random.default_rng(args.seed + 1000 * t)
        ci, ti = balanced_split(y, allc, args.shots, args.test_per_class,
                                rng)
        ncm = make_ncm("prototype_softmax", T)
        cp = FullConformalPredictor(ncm, alpha=ALPHA)
        cp.calibrate(tf.transform(X[ci]), y[ci], all_classes=allc)
        res = cp.predict(tf.transform(X[ti]), verbose=False,
                         device="cuda", return_p_values=True)
        yt = np.array([col[int(c)] for c in y[ti]])
        n_t = len(ti)
        sizes = np.zeros(n_t)
        cov1 = np.zeros(n_t, dtype=bool)
        for i, pv in enumerate(res["p_values"]):
            labs = [c for c, p in pv.items() if p > ALPHA]
            sizes[i] = len(labs)
            if labs:
                cov1[i] = int(y[ti][i]) in labs
            else:
                top1 = max(pv, key=pv.get)
                cov1[i] = int(top1) == int(y[ti][i])
        szs.append(sizes.mean())
        sz1s.append(np.where(sizes == 0, 1, sizes).mean())
        cov1s.append(cov1.mean())
    return dict(sz=float(np.mean(szs)),
                sz1=float(np.mean(sz1s)),
                sz1_se=float(np.std(sz1s) / np.sqrt(len(sz1s))),
                cov1=float(np.mean(cov1s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--cub_dir", default="output/pca_pilots/heldout_data")
    ap.add_argument("--dataset", default="cifar100", choices=list(REGIME))
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--test_per_class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proto_temperature", default="auto")
    ap.add_argument("--out_dir", default="output/config_heatmaps_0907")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("cuda required")

    X, y, Xu = load_dataset(args)
    allc = np.unique(y)
    K = len(allc)
    rows = []

    def run_cell(grid, p1n, p1, p2n, p2, dprime, C, kappa, gamma):
        t0 = time.perf_counter()
        tf = make_tf(Xu, dprime, C, kappa, gamma)
        T = pool_T(tf)
        r = eval_cell(tf, T, X, y, allc, args)
        r.update(grid=grid, dprime=dprime, C=C, kappa=kappa, gamma=gamma,
                 T=T)
        if grid == "dc":
            r["cmult"] = p2
        rows.append(r)
        print(f"[{args.dataset}|{grid}] {p1n}={p1} {p2n}={p2} "
              f"sz1={r['sz1']:.2f} cov1={r['cov1']:.3f} T={T:.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    for kappa in KG_GRID["kappa"]:
        for gamma in KG_GRID["gamma"]:
            run_cell("kg", "kappa", kappa, "gamma", gamma,
                     DEPLOY["dprime"], K, kappa, gamma)
    for dprime in DC_GRID["dprime"]:
        for cmult in DC_GRID["cmult"]:
            run_cell("dc", "dprime", dprime, "cmult", cmult,
                     dprime, max(2, int(round(cmult * K))),
                     DEPLOY["kappa"], DEPLOY["gamma"])

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"results_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "K": K, "alpha": ALPHA,
                   "config": vars(args), "deploy": DEPLOY,
                   "rows": rows}, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
