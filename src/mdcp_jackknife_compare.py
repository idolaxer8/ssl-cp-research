"""
Authors' Jackknife+ Multi-Score CP (Alg B.1, verbatim code) vs OUR MDCP
pilots, on OUR dimension scores.

The CP wrappers being compared -- the score DIMENSIONS are identical in every
arm (--dims ncm:view specs, pilot-B conventions: prototype cosine LAC with
pilot-fixed T / geodesic asym ratio with k_geo=min(5, shots), cal scored
leave-self-out, test scored vs full cal):

  raw1_<dim>      1-D split-CP ceil-quantile per dim (floor / control)
  dratio2         OUR pilot-B deployable: split-style scalar reduction, purity
                  = kNN density-ratio over the pseudo-labeled POOL score cloud
                  (ECDF ranks, proto-anchor yhat, k_d=10). Uses the pool; all
                  cal goes to the quantile.
  jk2             THEIRS: jackknife+ multi-score (yams_jacknife_vendored
                  .allocate_points, verbatim blob 7ed75b3), H = all dims
                  stacked [H,m,C], RAW score space (their method has no rank
                  transform). Uses ALL m cal points as leave-one-out cell
                  centers; no pool, no cal split. Guarantee 1-2alpha.
  jk1             same machinery, H=1 (first dim only) -- isolates the JK+
                  wrapper from dimensionality.
  jk2_rank        (--jk_rank) jk2 fed the pool-ECDF rank space instead of raw
                  scores -- scale-sensitivity control for mixed-family dims
                  (geo ratios x proto LAC). NOT their method as shipped.

Overlaid from archived JSONs (--fcp_glob): pilot-D full-CP arms (fcp_bag =
exact 1-alpha).

WALL-TIME metering per arm per trial (user question "why only JK+ time"):
  _t.scores   shared dim-score computation for cal+test (every arm needs it)
  _t.dratio2  everything dratio2 adds: POOL dim scores + ECDF + cloud build
              + cKDTrees + cal/test queries
  _t.jk2/jk1  the verbatim allocate_points call
  _t.raw1     the quantile (reported for completeness; ~0)
Full-CP cost is measured separately (src/mdcp_full_cp.py probe) -- it scales
per test point and does not fit this per-trial harness.

Protocol mirrors mdcp_pool_pilot: cal sizes x splits, per-trial
rng(seed + 1000*t + cal), alpha=0.1, test subsampled to --test_cap (bounds
the JK+ attribution tensor 8*k'*n_test*K bytes).

Conventions note: their result_mat force-includes one label in would-be-empty
sets (part of their method as shipped); our arms instead report empty%.

Run (from the worktree root; embeddings + overlays live in the MAIN checkout):
python src/mdcp_jackknife_compare.py \
  --embeddings_path <...>/embeddings_cifar100.pt \
  --unlabeled_path  <...>/embeddings_cifar100_unlabeled.pt \
  --dims proto:pca128_cw proto:pca32_cw \
  --fcp_glob "<main>/output/from_cluster/mdcp_pool_pilot/full_cp_*/mdcp_full_cp_results.json" \
  --output_dir <main>/output/mdcp_pool_pilot/jackknife_compare
"""
import sys, os, json, time, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from mdcp_pool_pilot import (load_embedding_sources, build_view_feats,
                             prototype_lac_scores, geodesic_asym_scores,
                             balanced_both_split, random_split, scp_eval,
                             ecdf_fit, ecdf_apply, PurityD, build_cloud)
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


def _jk_call(cal_S, cal_col, test_S, y_test_col, alpha, k_unique, res, name):
    pw0, t0 = _peak_wset(), time.time()
    cov, mean_set, result_mat, _, _ = allocate_points(
        cal_S, cal_col.astype(float), None, None,
        test_S, y_test_col.astype(float), alpha, JK_CONFIG)
    dt, pw1 = time.time() - t0, _peak_wset()
    res[name] = (float(cov), float(mean_set), 0.0,   # forced non-empty
                 float(np.median(result_mat.sum(axis=1))))
    res.setdefault("_t", {})[name] = dt
    res[f"_{name}_meta"] = dict(
        sec=dt, k_unique=int(k_unique),
        attr_bytes=int(8 * k_unique * test_S.shape[1] * test_S.shape[2]),
        peak_wset_gb=round(pw1 / 1e9, 2),
        peak_wset_delta_gb=round((pw1 - pw0) / 1e9, 2))


def dim_scores(ncm, Zq, Zc, yc, classes, T, k_geo, loo=False):
    if ncm == "geo":
        return geodesic_asym_scores(Zq, Zc, yc, classes, k_geo, self_is_cal=loo)
    return prototype_lac_scores(Zq, Zc, yc, classes, T, loo=loo)


def run_trial(feats, dim_specs, y, classes, cal_size, split, alpha, k_d,
              T_by_view, rng, test_cap, jk_rank=False):
    if split == "balanced_both":
        ci, ti = balanced_both_split(y, classes, cal_size, rng)
    else:
        ci, ti = random_split(y, classes, cal_size, rng)
    ti = ti[rng.permutation(len(ti))[:test_cap]]
    yc = y[ci]
    K = len(classes)
    cal_col = np.searchsorted(classes, yc)
    y_test_col = np.searchsorted(classes, y[ti])
    k_geo = max(1, min(5, len(yc) // K))
    cal_rows = np.arange(len(yc))
    res = {"_t": {}}

    # --- shared: cal + test dim scores (every arm consumes these) ---
    t0 = time.time()
    dims = {}
    for label, ncm, view in dim_specs:
        Zv, _ = feats[view]
        Zc, Zt = Zv[ci], Zv[ti]
        T = T_by_view.get(view)
        dims[label] = dict(
            ncm=ncm, view=view,
            cal=dim_scores(ncm, Zc, Zc, yc, classes, T, k_geo, loo=True),
            test=dim_scores(ncm, Zt, Zc, yc, classes, T, k_geo))
    res["_t"]["scores"] = time.time() - t0
    labels = [s[0] for s in dim_specs]

    # --- floor: 1-D split CP per dim ---
    t0 = time.time()
    for label in labels:
        res[f"raw1_{label}"] = scp_eval(dims[label]["cal"][cal_rows, cal_col],
                                        dims[label]["test"], y_test_col, alpha)
    res["_t"]["raw1"] = time.time() - t0

    # --- ours: pilot-B deployable pool D-ratio (everything it adds is timed:
    # pool dim scores + ECDF + cloud + trees + queries) ---
    t0 = time.time()
    rank = {}
    for label, ncm, view in dim_specs:
        _, Zuv = feats[view]
        Zc = feats[view][0][ci]
        pool = dim_scores(ncm, Zuv, Zc, yc, classes, T_by_view.get(view), k_geo)
        dims[label]["pool"] = pool
        ref = ecdf_fit(pool)
        rank[label] = {"pool": ecdf_apply(ref, pool),
                       "cal": ecdf_apply(ref, dims[label]["cal"]),
                       "test": ecdf_apply(ref, dims[label]["test"])}
    anchor = next((l for l, n, _ in dim_specs if n == "proto"), labels[0])
    yhat_col = np.argmin(dims[anchor]["pool"], axis=1)
    S_pool = np.stack([rank[l]["pool"] for l in labels], axis=-1)
    cloud, is_true = build_cloud(S_pool, yhat_col)
    D = PurityD(cloud, is_true, k_d, mode="ratio")
    q_cal = np.stack([rank[l]["cal"][cal_rows, cal_col] for l in labels], axis=-1)
    n_t = len(ti)
    q_test = np.stack([rank[l]["test"].ravel() for l in labels], axis=-1)
    res["dratio2"] = scp_eval(D(q_cal), D(q_test).reshape(n_t, K),
                              y_test_col, alpha)
    res["_t"]["dratio2"] = time.time() - t0

    # --- theirs: jackknife+ on the same dims (raw score space) ---
    cal_S = np.stack([dims[l]["cal"] for l in labels])       # (H, m, K)
    test_S = np.stack([dims[l]["test"] for l in labels])     # (H, n, K)
    k_unique = np.unique(cal_S[:, cal_rows, cal_col], axis=1).shape[1]
    _jk_call(cal_S, cal_col, test_S, y_test_col, alpha, k_unique, res, "jk2")
    _jk_call(cal_S[:1], cal_col, test_S[:1], y_test_col, alpha,
             np.unique(cal_S[:1, cal_rows, cal_col], axis=1).shape[1],
             res, "jk1")
    if jk_rank:
        cal_R = np.stack([rank[l]["cal"] for l in labels])
        test_R = np.stack([rank[l]["test"] for l in labels])
        ku = np.unique(cal_R[:, cal_rows, cal_col], axis=1).shape[1]
        _jk_call(cal_R, cal_col, test_R, y_test_col, alpha, ku, res, "jk2_rank")
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
    ap.add_argument("--dims", nargs="+",
                    default=["proto:pca128_cw", "proto:pca32_cw"],
                    help="ncm:view specs (ncm in {proto, geo})")
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--splits", nargs="+", default=["balanced_both", "random"])
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--k_d", type=int, default=10)
    ap.add_argument("--test_cap", type=int, default=500)
    ap.add_argument("--jk_rank", action="store_true",
                    help="add jk2_rank arm (JK+ in pool-ECDF rank space)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fcp_glob", default=None)
    ap.add_argument("--output_dir", default="output/mdcp_pool_pilot/jackknife_compare")
    args = ap.parse_args()

    X_src, y = load_embedding_sources(args.embeddings_path)
    Xu_src, _ = load_embedding_sources(args.unlabeled_path)
    classes = np.unique(y)
    dim_specs = []
    for i, spec in enumerate(args.dims):
        ncm, view = spec.split(":")
        dim_specs.append((f"{ncm}_{view}" if args.dims.count(spec) == 1
                          else f"{ncm}_{view}_{i}", ncm, view))
    views = [v for _, _, v in dim_specs]
    feats = build_view_feats(X_src, Xu_src, views)
    T_by_view = {v: pilot_T(feats[v][0], y, classes, args.seed)
                 for _, n, v in dim_specs if n == "proto"}
    print(f"labeled {len(y)}, pool {len(next(iter(Xu_src.values())))}, "
          f"K={len(classes)}, dims={[(l, n, v) for l, n, v in dim_specs]}, "
          f"T={ {v: round(t, 4) for v, t in T_by_view.items()} }", flush=True)

    results = {}
    for split in args.splits:
        for cal in args.cal_sizes:
            arms, tmeta = {}, {}
            t0 = time.time()
            for t in range(args.n_trials):
                rng = np.random.default_rng(args.seed + 1000 * t + cal)
                r = run_trial(feats, dim_specs, y, classes, cal, split,
                              args.alpha, args.k_d, T_by_view, rng,
                              args.test_cap, jk_rank=args.jk_rank)
                for k, v in r.pop("_t").items():
                    tmeta.setdefault(k, []).append(v)
                for k, v in r.items():
                    arms.setdefault(k, []).append(v)
                if t == 0:
                    print(f"  [{split} cal={cal}] trial 1 sec/arm: "
                          + "  ".join(f"{k} {v[0]:.2f}" for k, v in tmeta.items()),
                          flush=True)
            key = f"{split}_cal{cal}"
            results[key] = {"_arm_sec": {k: float(np.mean(v))
                                         for k, v in tmeta.items()}}
            print(f"\n== {key}  ({args.n_trials} trials, {time.time()-t0:.0f}s) ==",
                  flush=True)
            print(f"{'arm':<22}{'cov':>8}{'+-':>7}{'size':>9}{'+-':>7}")
            for arm, vals in arms.items():
                if arm.startswith("_"):
                    results[key][arm] = dict(
                        sec_mean=float(np.mean([v["sec"] for v in vals])),
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
                print(f"{arm:<22}{cov.mean():>8.3f}"
                      f"{cov.std(ddof=1)/np.sqrt(len(cov)):>7.3f}"
                      f"{sz.mean():>9.2f}{sz.std(ddof=1)/np.sqrt(len(sz)):>7.2f}",
                      flush=True)
            print("  sec/arm: " + "  ".join(
                f"{k} {v:.2f}" for k, v in results[key]["_arm_sec"].items()),
                flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out = dict(config={k: v for k, v in vars(args).items()},
               dim_labels=[l for l, _, _ in dim_specs], results=results)
    if args.fcp_glob:
        out["fcp_overlays"] = load_fcp_overlays(args.fcp_glob)
    out_path = os.path.join(args.output_dir, "mdcp_jackknife_compare_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
