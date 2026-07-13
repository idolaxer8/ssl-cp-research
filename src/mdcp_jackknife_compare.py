"""
Authors' Jackknife+ Multi-Score CP (Alg B.1, verbatim code) vs OUR MDCP
pilots, on OUR dimension scores.

The CP wrappers being compared -- the score DIMENSIONS are identical in every
in-driver arm (prototype cosine LAC on two pool-fit feature views,
proto:pca128_cw x proto:pca32_cw = the pilot-B multiview / pilot-D pair,
pilot-fixed T per view, cal scored leave-self-out, test scored vs full cal):

  raw1_proto128   1-D split-CP ceil-quantile on the fine dim (floor / control)
  dratio2_proto   OUR pilot-B deployable: split-style scalar reduction, purity
                  = kNN density-ratio over the pseudo-labeled POOL score cloud
                  (ECDF ranks, proto yhat, k_d=10). Uses the pool; all cal
                  goes to the quantile.
  jk2             THEIRS: jackknife+ multi-score (yams_jacknife_vendored
                  .allocate_points, verbatim blob 7ed75b3), H=2 heads = our
                  two dims stacked [H,m,C], raw LAC space (both dims already
                  commensurate in [0,1]; same space as pilot-D fcp arms).
                  Uses ALL m cal points as leave-one-out cell centers; no
                  pool, no cal split. Guarantee 1-2alpha.
  jk1             same machinery, H=1 (fine dim only) -- isolates the JK+
                  wrapper from dimensionality (their README: H=1 degenerates
                  toward ordinary split-style CP).

Overlaid from archived JSONs (same dims, cluster ladder, 20x150):
  fcp_bag         OUR pilot-D full-CP (exact 1-alpha, bag-symmetric purity)

Protocol mirrors mdcp_pool_pilot: CIFAR-100 matched-518 finals, cal in
{200,400,800} x {balanced_both, random}, per-trial rng(seed + 1000*t + cal),
alpha=0.1. Test points are randomly subsampled to --test_cap (default 500,
pilot-D style) to bound the JK+ attribution tensor (8 * k' * n_test * K bytes
-- the feasibility bottleneck) on a 16GB box.

Feasibility metering per JK+ call: wall seconds, k' (unique centers),
analytic attribution bytes, process peak working set (Windows peak_wset).

Conventions note: their result_mat force-includes one label in would-be-empty
sets (part of their method as shipped); our arms instead report empty%.

Run (from the worktree root; embeddings + overlays live in the MAIN checkout):
python src/mdcp_jackknife_compare.py \
  --embeddings_path <main>/output/from_cluster/embeddings/embeddings_cifar100.pt \
  --unlabeled_path  <main>/output/from_cluster/embeddings/embeddings_cifar100_unlabeled.pt \
  --fcp_glob "<main>/output/from_cluster/mdcp_pool_pilot/full_cp_*/mdcp_full_cp_results.json" \
  --output_dir <main>/output/mdcp_pool_pilot/jackknife_compare
"""
import sys, os, json, time, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from mdcp_pool_pilot import (load_embedding_sources, build_view_feats,
                             prototype_lac_scores, balanced_both_split,
                             random_split, scp_eval, ecdf_fit, ecdf_apply,
                             PurityD, build_cloud)
from mdcp_full_cp import pilot_T
from yams_jacknife_vendored import allocate_points

JK_CONFIG = {"fine_grid_test_score": [False, 1, "smooth"]}

try:
    import psutil
    _PROC = psutil.Process()
except ImportError:
    _PROC = None


def _peak_wset():
    if _PROC is None:
        return 0
    mi = _PROC.memory_info()
    return getattr(mi, "peak_wset", mi.rss)


def run_trial(feats, views, y, classes, cal_size, split, alpha, k_d,
              T_by_view, rng, test_cap):
    if split == "balanced_both":
        ci, ti = balanced_both_split(y, classes, cal_size, rng)
    else:
        ci, ti = random_split(y, classes, cal_size, rng)
    ti = ti[rng.permutation(len(ti))[:test_cap]]
    yc = y[ci]
    K = len(classes)
    cal_col = np.searchsorted(classes, yc)
    y_test_col = np.searchsorted(classes, y[ti])

    dims = {}
    for view in views:
        Zv, Zuv = feats[view]
        Zc, Zt = Zv[ci], Zv[ti]
        T = T_by_view[view]
        dims[view] = dict(
            cal=prototype_lac_scores(Zc, Zc, yc, classes, T, loo=True),
            test=prototype_lac_scores(Zt, Zc, yc, classes, T),
            pool=prototype_lac_scores(Zuv, Zc, yc, classes, T))
    fine = views[0]
    cal_rows = np.arange(len(yc))
    res = {}

    # --- floor: 1-D split CP on the fine dim ---
    res["raw1_proto128"] = scp_eval(dims[fine]["cal"][cal_rows, cal_col],
                                    dims[fine]["test"], y_test_col, alpha)

    # --- ours: pilot-B deployable 2-D pool D-ratio (ECDF ranks, proto yhat) ---
    rank = {}
    for v in views:
        ref = ecdf_fit(dims[v]["pool"])
        rank[v] = {p: ecdf_apply(ref, dims[v][p]) for p in ("pool", "cal", "test")}
    yhat_col = np.argmin(dims[fine]["pool"], axis=1)
    S_pool = np.stack([rank[v]["pool"] for v in views], axis=-1)
    cloud, is_true = build_cloud(S_pool, yhat_col)
    D = PurityD(cloud, is_true, k_d, mode="ratio")
    q_cal = np.stack([rank[v]["cal"][cal_rows, cal_col] for v in views], axis=-1)
    n_t = len(ti)
    q_test = np.stack([rank[v]["test"].ravel() for v in views], axis=-1)
    res["dratio2_proto"] = scp_eval(D(q_cal), D(q_test).reshape(n_t, K),
                                    y_test_col, alpha)

    # --- theirs: jackknife+ on the same dims (raw LAC space) ---
    for name, vs in (("jk2", views), ("jk1", (fine,))):
        cal_S = np.stack([dims[v]["cal"] for v in vs])     # (H, m, K)
        test_S = np.stack([dims[v]["test"] for v in vs])   # (H, n, K)
        k_unique = np.unique(
            cal_S[:, cal_rows, cal_col], axis=1).shape[1]
        pw0, t0 = _peak_wset(), time.time()
        cov, mean_set, result_mat, _, _ = allocate_points(
            cal_S, cal_col.astype(float), None, None,
            test_S, y_test_col.astype(float), alpha, JK_CONFIG)
        dt, pw1 = time.time() - t0, _peak_wset()
        res[name] = (float(cov), float(mean_set), 0.0,  # forced non-empty
                     float(np.median(result_mat.sum(axis=1))))
        res[f"_{name}_meta"] = dict(
            sec=dt, k_unique=int(k_unique),
            attr_bytes=int(8 * k_unique * len(ti) * K),
            peak_wset_gb=round(pw1 / 1e9, 2),
            peak_wset_delta_gb=round((pw1 - pw0) / 1e9, 2))
    return res


def load_fcp_overlays(fcp_glob):
    """{split: {cal: {arm: dict}}} from pilot-D cluster JSONs."""
    out = {}
    for path in glob.glob(fcp_glob):
        with open(path) as f:
            data = json.load(f)
        for key, arms in data["results"].items():
            split, cal = key.rsplit("_cal", 1)
            out.setdefault(split, {})[int(cal)] = arms
    return out


def main():
    ap = argparse.ArgumentParser(description="Authors' JK+ vs our MDCP arms")
    ap.add_argument("--embeddings_path", required=True)
    ap.add_argument("--unlabeled_path", required=True)
    ap.add_argument("--dims_views", nargs="+", default=["pca128_cw", "pca32_cw"])
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--splits", nargs="+", default=["balanced_both", "random"])
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--k_d", type=int, default=10)
    ap.add_argument("--test_cap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fcp_glob", default=None)
    ap.add_argument("--output_dir", default="output/mdcp_pool_pilot/jackknife_compare")
    args = ap.parse_args()

    X_src, y = load_embedding_sources(args.embeddings_path)
    Xu_src, _ = load_embedding_sources(args.unlabeled_path)
    classes = np.unique(y)
    views = list(args.dims_views)
    feats = build_view_feats(X_src, Xu_src, views)
    T_by_view = {v: pilot_T(feats[v][0], y, classes, args.seed) for v in views}
    print(f"labeled {len(y)}, pool {len(next(iter(Xu_src.values())))}, "
          f"K={len(classes)}, views={views}, "
          f"T={ {v: round(t, 4) for v, t in T_by_view.items()} }", flush=True)

    results = {}
    for split in args.splits:
        for cal in args.cal_sizes:
            arms = {}
            t0 = time.time()
            for t in range(args.n_trials):
                rng = np.random.default_rng(args.seed + 1000 * t + cal)
                r = run_trial(feats, views, y, classes, cal, split, args.alpha,
                              args.k_d, T_by_view, rng, args.test_cap)
                for k, v in r.items():
                    arms.setdefault(k, []).append(v)
                if t == 0:
                    print(f"  [{split} cal={cal}] trial 1: "
                          f"jk2 {r['_jk2_meta']['sec']:.1f}s "
                          f"(k'={r['_jk2_meta']['k_unique']}, "
                          f"attr {r['_jk2_meta']['attr_bytes']/1e6:.0f}MB, "
                          f"peak_wset {r['_jk2_meta']['peak_wset_gb']}GB)",
                          flush=True)
            key = f"{split}_cal{cal}"
            results[key] = {}
            print(f"\n== {key}  ({args.n_trials} trials, {time.time()-t0:.0f}s) ==",
                  flush=True)
            print(f"{'arm':<16}{'cov':>8}{'+-':>7}{'size':>9}{'+-':>7}")
            for arm, vals in arms.items():
                if arm.startswith("_"):
                    results[key][arm] = dict(
                        sec_mean=float(np.mean([v["sec"] for v in vals])),
                        sec_max=float(np.max([v["sec"] for v in vals])),
                        k_unique_mean=float(np.mean([v["k_unique"] for v in vals])),
                        attr_mb_mean=float(np.mean([v["attr_bytes"] for v in vals]) / 1e6),
                        peak_wset_gb_max=float(np.max([v["peak_wset_gb"] for v in vals])))
                    continue
                a = np.array(vals)
                cov, sz = a[:, 0], a[:, 1]
                results[key][arm] = dict(
                    cov=float(cov.mean()),
                    cov_se=float(cov.std(ddof=1) / np.sqrt(len(cov))),
                    size=float(sz.mean()),
                    size_se=float(sz.std(ddof=1) / np.sqrt(len(sz))))
                print(f"{arm:<16}{cov.mean():>8.3f}"
                      f"{cov.std(ddof=1)/np.sqrt(len(cov)):>7.3f}"
                      f"{sz.mean():>9.2f}{sz.std(ddof=1)/np.sqrt(len(sz)):>7.2f}",
                      flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out = dict(config={k: v for k, v in vars(args).items()}, results=results)
    if args.fcp_glob:
        out["fcp_overlays"] = load_fcp_overlays(args.fcp_glob)
    out_path = os.path.join(args.output_dir, "mdcp_jackknife_compare_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
