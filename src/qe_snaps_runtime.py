"""
Runtime accounting: qe (feature-level denoising) vs SNAPS (score-level
correction), both on the prototype+poolT champion base.

Cost structure being measured (cifar100 defaults):
  one-time  transform fit (pool):        none vs pre='qe' (qe adds the pool
                                         self-smoothing pass)
  per-point transform apply:             none vs pre='qe' (qe adds one
                                         768 x N_pool matvec + top-k per point)
  per-trial (i.e. per calibration draw)  base = prototype fit + LOO cal scores
                                         + test score matrix + quantile;
            SNAPS adds: pool score matrix (N_pool x K under the trial's
            cal-fit prototypes) + pool kNN of cal+test + the LOO cal-
            correction loop + score mixing. qe adds NOTHING per-trial (its
            cost is the fixed per-point preprocessing above).

Usage:
python src/qe_snaps_runtime.py --embeddings_path ... --unlabeled_path ... \
    --output_dir output/pool_repr_menu/snaps_stack
"""
import os, sys, time, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from conformal_prediction import PrototypeSoftmaxNCM
from exchangeable_features import make_transform
from exchangeable_fcp_experiment import SPLITS
from snaps_pool_experiment import (prototype_score_matrix, cal_correction_loo,
                                   pool_knn, _l2, conformal_quantile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_path",
                    default="output/from_cluster/embeddings/embeddings_cifar100.pt")
    ap.add_argument("--unlabeled_path",
                    default="output/from_cluster/embeddings/embeddings_cifar100_unlabeled.pt")
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--n_rep", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="output/pool_repr_menu/snaps_stack")
    args = ap.parse_args()

    d = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)
    X_raw, y = d["embeddings"].numpy(), d["labels"].numpy()
    Xu_raw = torch.load(args.unlabeled_path, map_location="cpu",
                        weights_only=False)["embeddings"].numpy()
    allc = np.unique(y)
    print(f"labeled {X_raw.shape}, pool {Xu_raw.shape}, K={len(allc)}")

    res = {"config": vars(args), "one_time": {}, "per_trial": []}

    # ---------------- one-time: transform fit + per-point apply ------------
    t0 = time.time(); tf_none = make_transform(Xu_raw, pca_dim=128,
                                               whiten="cluster", n_clusters=100)
    fit_none = time.time() - t0
    t0 = time.time(); tf_qe = make_transform(Xu_raw, pca_dim=128,
                                             whiten="cluster", n_clusters=100,
                                             pre="qe")
    fit_qe = time.time() - t0
    t0 = time.time(); X_none = tf_none.transform(X_raw)
    apply_none = time.time() - t0
    t0 = time.time(); X_qe = tf_qe.transform(X_raw)
    apply_qe = time.time() - t0
    n_pts = len(X_raw)
    res["one_time"] = {"fit_none_s": fit_none, "fit_qe_s": fit_qe,
                       "apply_none_us_per_point": 1e6 * apply_none / n_pts,
                       "apply_qe_us_per_point": 1e6 * apply_qe / n_pts}
    print(f"one-time fit: none {fit_none:.1f}s | qe {fit_qe:.1f}s "
          f"(qe overhead {fit_qe - fit_none:+.1f}s)")
    print(f"per-point apply: none {1e6*apply_none/n_pts:.0f}us | "
          f"qe {1e6*apply_qe/n_pts:.0f}us")

    Xu_none = tf_none.Xu_transformed_
    Xu_qe = tf_qe.Xu_transformed_

    # ---------------- per-trial costs -------------------------------------
    for cal in args.cal_sizes:
        stages = {s: [] for s in ("base_score", "snaps_pool_scores",
                                  "snaps_knn", "snaps_loo", "snaps_mix")}
        for rep in range(args.n_rep):
            rng = np.random.default_rng(args.seed + rep * 1000)
            ci, ti = SPLITS["balanced_both"](X_none, y, cal, 1000, allc, rng)

            for X, Xu, is_snaps in ((X_qe, Xu_qe, False),
                                    (X_none, Xu_none, True)):
                t0 = time.time()
                ncm = PrototypeSoftmaxNCM(temperature=0.09,
                                          allow_nonexchangeable=True
                                          ).fit(X[ci], y[ci])
                S_test = prototype_score_matrix(ncm, X[ti], allc)
                s_cal = ncm.alpha0
                qhat = conformal_quantile(s_cal, args.alpha)
                _ = S_test <= qhat
                t_base = time.time() - t0
                if not is_snaps:
                    stages["base_score"].append(t_base)  # identical shape both arms
                    continue
                t0 = time.time()
                Zu_prep = ncm._prep(Xu)
                F_pool = Zu_prep @ ncm.P
                if not ncm._P_ok.all():
                    F_pool[:, ~ncm._P_ok] = -np.inf
                S_pool = prototype_score_matrix(ncm, Xu, allc)
                stages["snaps_pool_scores"].append(time.time() - t0)
                t0 = time.time()
                Xu_n = _l2(Xu)
                nbr = pool_knn(np.vstack([X[ci], X[ti]]), Xu_n, args.k)
                nbr_cal, nbr_test = nbr[:len(ci)], nbr[len(ci):]
                stages["snaps_knn"].append(time.time() - t0)
                t0 = time.time()
                c_cal = cal_correction_loo(ncm, F_pool, Zu_prep, nbr_cal)
                stages["snaps_loo"].append(time.time() - t0)
                t0 = time.time()
                c_test = S_pool[nbr_test].mean(axis=1)
                s_hat = (1 - args.eta) * s_cal + args.eta * c_cal
                S_hat = (1 - args.eta) * S_test + args.eta * c_test
                qhat = conformal_quantile(s_hat, args.alpha)
                _ = S_hat <= qhat
                stages["snaps_mix"].append(time.time() - t0)

        row = {"cal": cal, **{s: float(np.mean(v)) for s, v in stages.items()}}
        row["snaps_total_overhead"] = sum(row[s] for s in
                                          ("snaps_pool_scores", "snaps_knn",
                                           "snaps_loo", "snaps_mix"))
        res["per_trial"].append(row)
        print(f"cal={cal}: base {row['base_score']*1e3:6.0f}ms | SNAPS overhead "
              f"{row['snaps_total_overhead']*1e3:6.0f}ms "
              f"(pool_scores {row['snaps_pool_scores']*1e3:.0f} + "
              f"knn {row['snaps_knn']*1e3:.0f} + "
              f"LOO {row['snaps_loo']*1e3:.0f} + mix {row['snaps_mix']*1e3:.0f})")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "runtime_qe_vs_snaps.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved -> {out}")

    # ---------------- figure ----------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cals = [r["cal"] for r in res["per_trial"]]
    xs = np.arange(len(cals))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    base = [r["base_score"] * 1e3 for r in res["per_trial"]]
    ax1.bar(xs - 0.2, base, 0.4, label="qe pipeline (base scoring only)",
            color="#d1862c")
    bottom = np.array(base, dtype=float)
    ax1.bar(xs + 0.2, base, 0.4, color="#9e9e9e", label="SNAPS: base scoring")
    for stage, col, lbl in (("snaps_pool_scores", "#5b8db8", "SNAPS: pool score matrix"),
                            ("snaps_knn", "#a8c4dd", "SNAPS: pool kNN"),
                            ("snaps_loo", "#2f5d82", "SNAPS: LOO cal correction"),
                            ("snaps_mix", "#c4d8e8", "SNAPS: mixing")):
        vals = [r[stage] * 1e3 for r in res["per_trial"]]
        ax1.bar(xs + 0.2, vals, 0.4, bottom=bottom, color=col, label=lbl)
        bottom = bottom + np.array(vals)
    ax1.set_xticks(xs); ax1.set_xticklabels([str(c) for c in cals])
    ax1.set_xlabel("cal size"); ax1.set_ylabel("per-calibration-draw cost (ms)")
    ax1.set_title("per-trial cost: qe adds nothing,\nSNAPS re-runs correction machinery")
    ax1.legend(fontsize=7)

    ot = res["one_time"]
    ax2.bar([0, 1], [ot["fit_none_s"], ot["fit_qe_s"]], 0.5,
            color=["#9e9e9e", "#d1862c"])
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["transform fit\n(none)",
                                                 "transform fit\n(qe)"])
    ax2.set_ylabel("one-time cost (s)")
    ax2.set_title(f"one-time costs; per-point apply: "
                  f"{ot['apply_none_us_per_point']:.0f}us -> "
                  f"{ot['apply_qe_us_per_point']:.0f}us with qe")
    fig.suptitle("qe vs SNAPS runtime (cifar100, prototype+poolT, k=10, eta=0.5)",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(args.output_dir, "runtime_qe_vs_snaps.png")
    fig.savefig(p, dpi=150)
    print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
