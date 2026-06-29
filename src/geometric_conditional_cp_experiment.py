"""
Stage 2 runner: geometry-conditional Full CP vs global FCP.

Stage 0 (default): the cheapest decisive win -- does Mondrian-over-geometric-strata
lift the sparse/under-covered stratum toward 0.90 (and shrink the empty-set rate
there) WITHOUT breaking marginal validity? miniImageNet, density strata, cal=800.

Arms (all from ONE predict() per trial, sharing test_scores + static cal_scores):
  global_fcp     -- standard transductive Full CP (the measured-gap reference)
  global_static  -- static-threshold FCP (G=1 Mondrian; ~ global_fcp at O(1/n))
  mondrian_geo   -- per-stratum-threshold FCP (the fix)

Routing: balanced split -> --ncm (prototype_softmax, GPU, all classes present);
random split -> geodesic (GPU, missing-class-safe) for the exact-validity gate.
"""
import sys, os, json, argparse, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import make_transform
from geometric_coverage_experiment import (load_emb, carve_test_and_pool, sample_cal,
                                            per_stratum, knn_radii, density_cov, lid_mle)
from geometric_conditional_cp import (GeometryStratifier, ConfidenceStratifier,
                                        global_sets, mondrian_sets, rlcp_sets)

MAIN_OUT = r"C:\Users\IDO\Desktop\Ido_student\Msc\ssl-cp-research\output\from_cluster"


@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def covered_sizes(psets, yt):
    covered = np.array([yt[i] in set(ps) for i, ps in enumerate(psets)])
    sizes = np.array([len(ps) for ps in psets], dtype=float)
    return covered, sizes


def run(args):
    allc = np.arange(args.n_classes)
    dev = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    Xb, yb = load_emb(os.path.join(args.embeddings_dir, f"embeddings_{args.dataset}.pt"))
    Xu, _ = load_emb(os.path.join(args.embeddings_dir, f"embeddings_{args.dataset}_unlabeled.pt"))
    transform = make_transform(Xu, pca_dim=args.pca_dim, whiten="cluster", n_clusters=args.n_clusters)
    strat = GeometryStratifier(transform, mode=args.covariate, n_strata=args.n_strata, knn=args.knn)
    print(f"[data] {args.dataset} base {Xb.shape} pool {Xu.shape} device={dev} "
          f"cov={args.covariate} G={args.n_strata} cal={args.cal} alpha={args.alpha}")

    # fixed pilot T for prototype (constant across trials => exact)
    T_fixed = None
    prng = np.random.default_rng(args.seed + 9999)
    ti, pi = carve_test_and_pool(yb, args.m_test, allc, prng)
    pci = sample_cal(pi, yb, min(args.cal, 800), True, allc, prng)
    with quiet():
        pncm = create_ncm("prototype_softmax", temperature=None, logit="cosine",
                          allow_nonexchangeable=True)
        pncm.fit(transform.transform(Xb[pci]), yb[pci]); T_fixed = float(pncm._resolve_T())
    print(f"[pilot] prototype fixed T={T_fixed:.4f}")

    out = {}
    for split in args.splits:
        balanced = (split == "balanced")
        ncm_name = "prototype_softmax" if balanced else "unwhitened_topk_asym"
        # confidence control needs a posterior -> prototype (balanced) only
        arms = ["global_fcp", "global_static", "mondrian_geo"]
        if balanced and not args.no_confidence:
            arms.append("mondrian_conf")
        if args.rlcp:
            arms.append("rlcp_geo")
        acc = {a: {"cov": [], "sz": [], "strata": [], "empty": []} for a in arms}
        for t in range(args.n_trials):
            trng = np.random.default_rng(args.seed + 1000 * args.cal + t)
            test_idx, pool_idx = carve_test_and_pool(yb, args.m_test, allc, trng)
            ci = sample_cal(pool_idx, yb, args.cal, balanced, allc, trng)
            Xc_t = transform.transform(Xb[ci]); yc = yb[ci]
            Xt_t = transform.transform(Xb[test_idx]); yt = yb[test_idx]
            with quiet():
                kw = (dict(temperature=T_fixed, logit="cosine") if balanced else dict(k=5))
                ncm = create_ncm(ncm_name, **kw)
                cp = FullConformalPredictor(ncm, alpha=args.alpha)
                cp.calibrate(Xc_t, yc, all_classes=allc)
                res = cp.predict(Xt_t, device=dev, return_test_scores=True, verbose=False)
            ts, cs, cls = res["test_scores"], res["cal_scores"], res["classes"]
            s_cal = strat.stratum_of(Xc_t)
            s_test = strat.stratum_of(Xt_t)
            sets = {
                "global_fcp": res["prediction_sets"],
                "global_static": global_sets(ts, cs, cls, args.alpha),
                "mondrian_geo": mondrian_sets(ts, cs, s_test, s_cal, cls, args.alpha),
            }
            if "mondrian_conf" in arms:
                conf = ConfidenceStratifier(cp.ncm, transform, n_strata=args.n_strata)
                sets["mondrian_conf"] = mondrian_sets(
                    ts, cs, conf.stratum_of(Xt_t), conf.stratum_of(Xc_t), cls, args.alpha)
            if "rlcp_geo" in arms:
                h = args.rlcp_c0 * strat.pool_iqr
                sets["rlcp_geo"] = rlcp_sets(ts, cs, strat.covariate_of(Xt_t),
                                             strat.covariate_of(Xc_t), cls, args.alpha, h, trng)
            for a in arms:
                cov, sz = covered_sizes(sets[a], yt)
                acc[a]["cov"].append(cov); acc[a]["sz"].append(sz)
                acc[a]["strata"].append(s_test)
                acc[a]["empty"].append(sz == 0)
        # aggregate
        out[split] = {}
        print(f"\n=== split={split} (ncm={ncm_name}) ===")
        for a in arms:
            cov = np.concatenate(acc[a]["cov"]); sz = np.concatenate(acc[a]["sz"])
            st = np.concatenate(acc[a]["strata"]); empty = np.concatenate(acc[a]["empty"])
            m = per_stratum(cov, sz, st, args.n_strata, args.alpha)
            # per-stratum empty rate
            erow = [float(empty[st == g].mean()) if (st == g).any() else None
                    for g in range(args.n_strata)]
            m["empty_by_stratum"] = erow
            m["empty_rate"] = float(empty.mean())
            out[split][a] = m
            sparse_cov = m["rows"][-1]["cov"]; sparse_sz = m["rows"][-1]["size"]
            print(f"  {a:14s} marg={m['marg_cov']:.3f} sz={m['marg_size']:.2f} "
                  f"CovGap_geo={m['covgap_geo']:.2f}pp worst={m['worst']:.3f} "
                  f"empty={m['empty_rate']:.3f} | sparsest cov={sparse_cov:.3f} sz={sparse_sz:.2f}")

    os.makedirs(os.path.join(args.out_dir, args.dataset), exist_ok=True)
    p = os.path.join(args.out_dir, args.dataset,
                     f"stage0_{args.covariate}_G{args.n_strata}_cal{args.cal}_a{args.alpha}.json")
    with open(p, "w") as f:
        json.dump({"config": vars(args), "results": out}, f, indent=2)
    print(f"[done] {p}")
    if args.plot:
        make_plot(out, args, p.replace(".json", ".png"))


def make_plot(out, args, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    splits = list(out.keys())
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.4), squeeze=False)
    colors = {"global_fcp": "#c2334d", "global_static": "#e08214",
              "mondrian_geo": "#1b7837", "mondrian_conf": "#3b6fb6",
              "rlcp_geo": "#9970ab"}
    for j, split in enumerate(splits):
        ax = axes[0][j]
        for a, m in out[split].items():
            xs = [r["stratum"] for r in m["rows"] if r["cov"] is not None]
            ys = [r["cov"] for r in m["rows"] if r["cov"] is not None]
            ax.plot(xs, ys, "o-", c=colors.get(a), lw=2.2, ms=7,
                    label=f"{a} (gap {m['covgap_geo']:.1f}, sz {m['marg_size']:.2f})")
        ax.axhline(1 - args.alpha, ls="--", c="k", lw=1.1)
        ax.axhspan(0.5, 1 - args.alpha, color="red", alpha=.05)
        ax.set_xlabel(f"{args.covariate} stratum (dense->sparse)"); ax.set_ylabel("coverage")
        ax.set_title(f"{args.dataset} {split}"); ax.legend(fontsize=7.5); ax.grid(alpha=.25)
    fig.suptitle(f"Mondrian-geometry vs global FCP: per-stratum coverage "
                 f"(cal={args.cal}, alpha={args.alpha}, G={args.n_strata})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140); plt.close(fig); print(f"[plot] {path}")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="miniimagenet")
    ap.add_argument("--embeddings_dir", default=MAIN_OUT)
    ap.add_argument("--out_dir", default="output/geometric_conditional")
    ap.add_argument("--n_classes", type=int, default=100)
    ap.add_argument("--cal", type=int, default=800)
    ap.add_argument("--m_test", type=int, default=20)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--splits", nargs="+", default=["balanced", "random"])
    ap.add_argument("--covariate", default="density", choices=["density", "lid"])
    ap.add_argument("--n_strata", type=int, default=5)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--n_clusters", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no_confidence", action="store_true",
                    help="skip the confidence-conditioning control arm")
    ap.add_argument("--rlcp", action="store_true", help="add the RLCP localized arm")
    ap.add_argument("--rlcp_c0", type=float, default=0.5,
                    help="RLCP bandwidth scale: h = c0 * IQR(pool covariate)")
    ap.add_argument("--plot", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse())
