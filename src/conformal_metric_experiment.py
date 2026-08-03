"""Conformal-metric-learning (G1) phase runner.

Phases (each writes JSON under <MAIN>/output/conformal_metric/):
  landscape   Phase-1 go/no-go: does the pool-only rehearsal objective track
              TRUE FCP set size across the rung-1 family? Bakes ~n_probe grid
              points and measures real FCP (geodesic, cal=gt_cal). Reports
              Spearman, pool-argmin regret, dynamic range, gate decision.
  rung1       fit_conformal_metric(rung=1) per dataset -> fit report (s, mode).
  rung2       fit_conformal_metric(rung=2) per dataset (includes rung-1 +
              B2 showdown).
  benchmark   Full FCP comparison: menu arms vs pool-only selectors vs the
              learned metric. Row schema mirrors transform_control_experiment.
  validity    50-trial random-split coverage + permutation oracle +
              contamination positive control.

All learned/selected transforms are functions of the unlabeled pool alone
(Prop 2 -> exact coverage). The labeled file is loaded ONLY for FCP
evaluation, never passed to any fitting routine. prototype_softmax runs at
FIXED temperature (auto-T is cal-fit and is refused here).

Usage (from this worktree):
  python src/conformal_metric_experiment.py --phase landscape \
      --datasets cifar100 cub200 aircraft --n_probe 24 --n_trials 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_prediction import FullConformalPredictor, create_ncm  # noqa: E402
from exchangeable_features import UnlabeledTransform, IdentityTransform  # noqa: E402
from transform_control_experiment import (ARMS, balanced_split, random_split,  # noqa: E402
                                          pool_participation_ratio)
import conformal_metric as cml  # noqa: E402
from conformal_metric import (PoolContext, gate_scales, bake,  # noqa: E402
                              fit_conformal_metric, CFG)
from pool_objective import objective_on_half  # noqa: E402

# Repo root holding output/ + embeddings. Overridable for the cluster:
#   export SSL_CP_MAIN=/storage/ido/ssl-cp/ssl-cp-research
MAIN_REPO = os.environ.get(
    "SSL_CP_MAIN", r"C:\Users\IDO\Desktop\Ido_student\Msc\ssl-cp-research")
OUT_ROOT = os.path.join(MAIN_REPO, "output", "conformal_metric")

DATASETS = {
    # name: (data_dir, K, cal_sizes, gt_cal, gt_file)
    "cifar100": ("output/from_cluster/embeddings", 100, [200, 400, 800], 800,
                 "output/pca_pilots/transform_controls/results_cifar100.json"),
    "miniimagenet": ("output/from_cluster/embeddings", 100, [200, 400, 800],
                     800,
                     "output/pca_pilots/transform_controls/results_miniimagenet.json"),
    "aircraft": ("output/from_cluster/embeddings", 100, [200, 400, 800], 800,
                 "output/pca_pilots/transform_controls/results_aircraft.json"),
    "cifar10": ("output/from_cluster/embeddings", 10, [200, 400, 800], 800,
                "output/pca_pilots/transform_controls/results_cifar10.json"),
    "cub200": ("output/pca_pilots/heldout_data", 200, [400, 800, 1600], 1600,
               "output/pca_pilots/heldout/results_cub200_heldout.json"),
}

MENU_ARMS = ["raw768", "pca128_cw", "pca512_cw", "lw_cluster768"]
SOFTMAX_NCMS = {"prototype_softmax"}


def load_dataset(ds):
    data_dir, K, cal_sizes, gt_cal, gt_file = DATASETS[ds]
    base = os.path.join(MAIN_REPO, data_dir)
    d = torch.load(os.path.join(base, f"embeddings_{ds}.pt"),
                   map_location="cpu", weights_only=False)
    X = np.ascontiguousarray(d["embeddings"].numpy(), dtype=np.float64)
    y = d["labels"].numpy()
    Xu = np.ascontiguousarray(
        torch.load(os.path.join(base, f"embeddings_{ds}_unlabeled.pt"),
                   map_location="cpu", weights_only=False)["embeddings"].numpy(),
        dtype=np.float64)
    return X, y, Xu, K, cal_sizes, gt_cal, gt_file


def load_gt_sizes(gt_file, gt_cal, ncm="unwhitened_topk_mean"):
    """Measured champion set sizes (balanced_both @ gt_cal) per menu arm."""
    path = os.path.join(MAIN_REPO, gt_file)
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    return {r["arm"]: r["sz"] for r in d["rows"]
            if r["split"] == "balanced_both" and r["cal"] == gt_cal
            and r["ncm"] == ncm}


def measure_fcp(transform, X, y, allc, cal, split, ncm_name, n_trials, alpha,
                device, seed, proto_T=0.06, test_per_class=5):
    """One (transform, split, cal, ncm) FCP measurement -- the
    transform_control_experiment trial loop, importable."""
    K = len(allc)
    cls_to_j = {int(c): j for j, c in enumerate(allc)}
    covs, szs = [], []
    pooled_cov, pooled_tot = np.zeros(K), np.zeros(K)
    n_fallback = 0
    for t in range(n_trials):
        rng = np.random.default_rng(seed + 1000 * t)
        if split == "balanced_both":
            ci, ti = balanced_split(y, allc, cal // K, test_per_class, rng)
        else:
            ci, ti = random_split(y, cal, test_per_class * K, rng)
        Xc, yc = transform.transform(X[ci]), y[ci]
        Xt, yt = transform.transform(X[ti]), y[ti]
        if ncm_name in SOFTMAX_NCMS:
            ncm = create_ncm(ncm_name, temperature=proto_T, logit="cosine")
        else:
            ncm = create_ncm(ncm_name, k=5)
        cp = FullConformalPredictor(ncm, alpha=alpha)
        cp.calibrate(Xc, yc, all_classes=allc)
        try:
            res = cp.predict(Xt, verbose=False, device=device)
        except (RuntimeError, ValueError):
            res = cp.predict(Xt, verbose=False, device="cpu")
            n_fallback += 1
        psets = res["prediction_sets"]
        covered = np.array([yt[i] in psets[i] for i in range(len(yt))])
        covs.append(float(covered.mean()))
        szs.append(float(np.mean([len(s) for s in psets])))
        for c in allc:
            msk = yt == c
            if msk.any():
                j = cls_to_j[int(c)]
                pooled_cov[j] += float(covered[msk].sum())
                pooled_tot[j] += int(msk.sum())
    valid = pooled_tot > 0
    pcov = pooled_cov[valid] / pooled_tot[valid]
    return {"split": split, "ncm": ncm_name, "cal": cal,
            "cov": float(np.mean(covs)), "cov_sd": float(np.std(covs)),
            "sz": float(np.mean(szs)), "sz_sd": float(np.std(szs)),
            "sz_se": float(np.std(szs) / np.sqrt(len(szs))),
            "covgap": float(100 * np.mean(np.abs(pcov - (1 - alpha)))),
            "n_trials": n_trials, "n_cpu_fallback": n_fallback}


# ---------------------------------------------------------------------------
# Phase: landscape
# ---------------------------------------------------------------------------
CORNERS = [  # champion-equivalent corners of the family, always probed
    dict(j0=128.0, w=0.5, gamma=0.0, whiten="cluster",    tag="~pca128_cw"),
    dict(j0=512.0, w=0.5, gamma=0.0, whiten="cluster",    tag="~pca512_cw"),
    dict(j0=768.0, w=128.0, gamma=0.0, whiten="lw_cluster", tag="~lw_cluster768"),
    dict(j0=768.0, w=128.0, gamma=0.0, whiten="cluster",  tag="~cw768"),
]


def phase_landscape(ds, args):
    for k_mult in (args.pseudo_k_mult or [1]):
        _landscape_once(ds, args, k_mult)


def _landscape_once(ds, args, k_mult):
    X, y, Xu, K, cal_sizes, gt_cal, gt_file = load_dataset(ds)
    allc = np.unique(y)
    cfg = dict(CFG, seed=args.seed, cal_budget=gt_cal, pseudo_k_mult=k_mult)
    ctx = PoolContext(Xu, K, args.alpha, cfg)
    tag = f"_kmult{k_mult}" if k_mult > 1 else ""
    out_dir = os.path.join(OUT_ROOT, "landscape")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    if args.reuse_true:
        # Mitigation fast path: the probes' TRUE FCP sizes are pseudo-task-
        # independent -- reuse them from the baseline (k_mult=1) landscape and
        # recompute ONLY the surrogate objective under the harder pseudo-task.
        base_path = os.path.join(out_dir, f"landscape_{ds}.json")
        base = json.load(open(base_path))
        probes = []
        for p in base["probes"]:
            q = {k: p[k] for k in ("j0", "w", "gamma", "whiten", "tag",
                                   "true_sz", "true_cov", "true_sz_se")}
            s = gate_scales(q["j0"], q["w"], q["gamma"], ctx.lam)
            q.update(ctx.eval_candidate(s, q["whiten"], half="B1"))
            probes.append(q)
        print(f"[{ds}] k_mult={k_mult}: surrogates recomputed for "
              f"{len(probes)} probes (true sizes reused)", flush=True)
    else:
        # surrogate objective over the full rung-1 grid
        from itertools import product as iproduct
        t0 = time.time()
        for whiten in cfg["whiten_modes"]:
            for j0, w, gamma in iproduct(cfg["grid_j0"], cfg["grid_w"],
                                         cfg["grid_gamma"]):
                s = gate_scales(j0, w, gamma, ctx.lam)
                obj = ctx.eval_candidate(s, whiten, half="B1")
                records.append(dict(j0=float(j0), w=float(w),
                                    gamma=float(gamma), whiten=whiten,
                                    tag="", **obj))
        print(f"[{ds}] grid {len(records)} candidates in "
              f"{time.time()-t0:.0f}s", flush=True)

        # probe selection: forced corners + objective-quantile-stratified
        probes = [dict(c) for c in CORNERS]
        for c in probes:
            s = gate_scales(c["j0"], c["w"], c["gamma"], ctx.lam)
            c.update(ctx.eval_candidate(s, c["whiten"], half="B1"))
        ordered = sorted(records, key=lambda r: r["rehearsal_sz"])
        want = max(args.n_probe - len(probes), 0)
        qidx = np.unique(np.linspace(0, len(ordered) - 1, want).astype(int))
        seen = {(p["j0"], p["w"], p["gamma"], p["whiten"]) for p in probes}
        for i in qidx:
            r = ordered[i]
            key = (r["j0"], r["w"], r["gamma"], r["whiten"])
            if key not in seen:
                probes.append(dict(r))
                seen.add(key)

        # bake each probe on the FULL pool + true FCP at gt_cal (geodesic)
        for i, p in enumerate(probes):
            s = gate_scales(p["j0"], p["w"], p["gamma"], ctx.lam)
            t0 = time.time()
            tf, _ = bake(Xu, s, p["whiten"], cfg)
            row = measure_fcp(tf, X, y, allc, gt_cal, "balanced_both",
                              "unwhitened_topk_mean", args.n_trials,
                              args.alpha, args.device, args.seed)
            p["true_sz"], p["true_cov"] = row["sz"], row["cov"]
            p["true_sz_se"] = row["sz_se"]
            print(f"  probe {i+1}/{len(probes)} j0={p['j0']:6.1f} "
                  f"w={p['w']:6.2f} gam={p['gamma']:+.2f} {p['whiten']:<10}"
                  f"{p['tag']:<15} surr={p['rehearsal_sz']:6.3f} "
                  f"true={p['true_sz']:6.2f} cov={p['true_cov']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if args.device == "cuda":
                torch.cuda.empty_cache()

    # gate metrics (+ the pre-registered rehearsal-band/margin tie-break)
    from scipy.stats import spearmanr
    surr = np.array([p["rehearsal_sz"] for p in probes])
    true = np.array([p["true_sz"] for p in probes])
    rho = float(spearmanr(surr, true).correlation)
    argmin_true = float(true[np.argmin(surr)])
    regret = float(argmin_true / true.min() - 1)
    dyn = float(surr.max() / max(surr.min(), 1e-9))
    se = np.array([p.get("rehearsal_se", 0.01) for p in probes])
    band = surr <= surr.min() + 2 * float(np.median(se))
    m90 = np.array([p["margin_q90"] for p in probes])
    tb_idx = int(np.where(band)[0][np.argmin(m90[band])])
    tb_regret = float(true[tb_idx] / true.min() - 1)
    gate = ("GO" if (rho >= 0.7 and min(regret, tb_regret) <= 0.05)
            else ("NO-GO" if rho < 0.4 else "MARGINAL"))
    print(f"[{ds}{tag}] Spearman={rho:+.3f} argmin-regret={regret:+.1%} "
          f"tiebreak-regret={tb_regret:+.1%} dyn-range={dyn:.2f} -> {gate}",
          flush=True)

    out = dict(dataset=ds, alpha=args.alpha, gt_cal=gt_cal,
               n_trials=args.n_trials, cfg_hash=cml.cfg_hash(cfg),
               pseudo_k_mult=k_mult,
               pool_pr=pool_participation_ratio(Xu, seed=args.seed),
               spearman=rho, argmin_regret=regret,
               tiebreak_regret=tb_regret, tiebreak_band_n=int(band.sum()),
               dynamic_range=dyn, gate=gate, probes=probes, grid=records)
    path = os.path.join(out_dir, f"landscape_{ds}{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"saved -> {path}", flush=True)
    try:
        from plot_conformal_metric import plot_landscape
        plot_landscape(out, os.path.join(out_dir, f"landscape_{ds}{tag}.png"))
    except Exception as e:  # plotting must never kill the run
        print(f"[warn] landscape plot failed: {e}")
    return out


# ---------------------------------------------------------------------------
# Phase: rung1 / rung2 fits
# ---------------------------------------------------------------------------
def phase_fit(ds, args, rung):
    X, y, Xu, K, cal_sizes, gt_cal, gt_file = load_dataset(ds)
    cfg = dict(seed=args.seed, cal_budget=gt_cal)
    t0 = time.time()
    tf, report = fit_conformal_metric(Xu, K, alpha=args.alpha, cfg=cfg,
                                      rung=rung, device=args.device)
    report["fit_seconds"] = time.time() - t0
    out_dir = os.path.join(OUT_ROOT, f"rung{rung}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fit_{ds}.json")
    json.dump(report, open(path, "w"), indent=1)
    print(f"[{ds}] rung{rung} fit done in {report['fit_seconds']:.0f}s "
          f"-> {path}", flush=True)
    try:
        from plot_conformal_metric import plot_spectrum
        plot_spectrum(report, ds, os.path.join(out_dir, f"spectrum_{ds}.png"))
    except Exception as e:
        print(f"[warn] spectrum plot failed: {e}")
    return report


def load_g1_transform(ds, Xu, rung):
    """Rebuild the baked transform from a saved fit report (s by eigen-index
    on the full-pool basis -- deterministic given the pool)."""
    path = os.path.join(OUT_ROOT, f"rung{rung}", f"fit_{ds}.json")
    if not os.path.exists(path):
        return None
    rep = json.load(open(path))
    s = np.asarray(rep["s_final"], dtype=np.float64)
    tf, _ = bake(Xu, s, rep["whiten_final"], dict(CFG))
    return tf


# ---------------------------------------------------------------------------
# Phase: benchmark
# ---------------------------------------------------------------------------
def selector_margin_pick(Xu, K, seed, n_clusters_whiten=100):
    """The incumbent label-free baseline: margin_q90 selection over the menu
    arms, pilot protocol (arms fit on pool half A, statistic on half B)."""
    from pool_objective import pseudo_task, class_means, centroid_dists, \
        margin_stats
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xu))
    A, B = perm[:len(Xu) // 2], perm[len(Xu) // 2:]
    Xa, Xb = Xu[A], Xu[B]
    ya, yb, _ = pseudo_task(Xa, Xb, K, seed)
    best, best_v = None, np.inf
    for arm in MENU_ARMS:
        spec = ARMS[arm]
        t = (IdentityTransform().fit() if spec is None else
             UnlabeledTransform(n_clusters=n_clusters_whiten,
                                random_state=seed, **spec).fit(Xa))
        Za, Zb = t.transform(Xa), t.transform(Xb)
        D = centroid_dists(Zb, class_means(Za, ya, K))
        v = margin_stats(D)["margin_q90"]
        if v < best_v:
            best, best_v = arm, v
    return best


def build_menu_transform(arm, Xu, n_clusters_whiten=100):
    spec = ARMS[arm]
    if spec is None:
        return IdentityTransform().fit()
    return UnlabeledTransform(n_clusters=n_clusters_whiten, **spec).fit(Xu)


def build_ablation_arms(ds, Xu, cfg=None):
    """Stage/factor ablation of the learned composite (all pool-fit):
      g1_s_only     learned s, NO stage-2 whitening
      g1_gate_only  best rung-1 grid member with gamma = 0 (gate factor alone)
      g1_power_only best grid member with j0=768, w=128 (power factor alone)
    (cw768 / lw_cluster768 give the s=1 'stage 2 alone' rows.)
    Returns (arms dict, info dict recording which grid members were picked)."""
    cfg = cfg or CFG
    rep_path = os.path.join(OUT_ROOT, "rung1", f"fit_{ds}.json")
    if not os.path.exists(rep_path):
        return {}, {}
    rep = json.load(open(rep_path))
    s_final = np.asarray(rep["s_final"], dtype=np.float64)
    Xu64 = np.asarray(Xu, dtype=np.float64)
    mu_f, V_f, lam_f = cml.pool_eigenbasis(Xu64)

    def spectral_tf(s, whiten):
        return UnlabeledTransform(
            projection="spectral",
            spectral_filter={"mu": mu_f, "V": V_f, "s": s},
            pca_dim=None, whiten=whiten,
            n_clusters=cfg["n_clusters_whiten"], random_state=42).fit(Xu64)

    arms = {"g1_s_only": spectral_tf(s_final, None)}
    info = {"g1_s_only": {"whiten": None}}
    grid = rep.get("rung1", {}).get("grid", [])
    slices = {
        "g1_gate_only": [r for r in grid if r["gamma"] == 0.0],
        "g1_power_only": [r for r in grid
                          if r["j0"] == 768.0 and r["w"] == 128.0],
    }
    for name, rows in slices.items():
        if not rows:
            continue
        best = min(rows, key=lambda r: r["rehearsal_sz"])
        s = gate_scales(best["j0"], best["w"], best["gamma"], lam_f)
        arms[name] = spectral_tf(s, best["whiten"])
        info[name] = {k: best[k] for k in ("j0", "w", "gamma", "whiten",
                                           "rehearsal_sz")}
    return arms, info


def phase_benchmark(ds, args):
    X, y, Xu, K, cal_sizes, gt_cal, gt_file = load_dataset(ds)
    allc = np.unique(y)
    pool_pr = pool_participation_ratio(Xu, seed=args.seed)

    arm_tfs = {}
    for arm in MENU_ARMS:
        arm_tfs[arm] = build_menu_transform(arm, Xu)
    pick = selector_margin_pick(Xu, K, args.seed)
    arm_tfs[f"selector_margin->{pick}"] = arm_tfs[pick]
    regime = "pca128_cw" if pool_pr >= 64.0 else "lw_cluster768"
    arm_tfs[f"selector_regime->{regime}"] = arm_tfs[regime]
    for rung in (1, 2):
        tf = load_g1_transform(ds, Xu, rung)
        if tf is not None:
            arm_tfs[f"g1_r{rung}"] = tf
    ablation_info = {}
    if args.ablation:
        arm_tfs["cw768"] = build_menu_transform("cw768", Xu)
        abl, ablation_info = build_ablation_arms(ds, Xu)
        arm_tfs.update(abl)
    if args.contaminated:
        # positive control: transform refit on pool + ALL labeled data (every
        # future test point leaks into the fit). Routed through the g1 spec.
        rep_path = os.path.join(OUT_ROOT, "rung1", f"fit_{ds}.json")
        if os.path.exists(rep_path):
            rep = json.load(open(rep_path))
            s = np.asarray(rep["s_final"], dtype=np.float64)
            tf_c, _ = bake(np.vstack([Xu, X]), s, rep["whiten_final"],
                           dict(CFG))
            arm_tfs["g1_contaminated"] = tf_c

    rows = []
    for arm, tf in arm_tfs.items():
        print(f"\n=== [{ds}] arm {arm}: {tf} ===", flush=True)
        for split in ["balanced_both", "random"]:
            for cal in cal_sizes:
                if split == "balanced_both" and cal // K < 2:
                    continue
                for nm in ["unwhitened_topk_mean", "prototype_softmax"]:
                    if (split == "random" and nm in SOFTMAX_NCMS
                            and cal < max(800, 8 * K)):
                        # Known degenerate corner: random split with expected
                        # <8 shots/class leaves classes missing -> prototype
                        # GPU path unusable -> ~20min/trial CPU fallback
                        # (burned 3.5h/row on CUB K=200 at cal=800 twice).
                        continue
                    t0 = time.time()
                    row = measure_fcp(tf, X, y, allc, cal, split, nm,
                                      args.n_trials, args.alpha, args.device,
                                      args.seed, proto_T=args.proto_T)
                    row["arm"] = arm
                    rows.append(row)
                    print(f"  [{split:13s}] cal={cal:4d} {nm:22s} "
                          f"cov={row['cov']:.4f} sz={row['sz']:6.2f}"
                          f"+-{row['sz_se']:.2f} covgap={row['covgap']:5.2f}pp"
                          f" ({time.time()-t0:.0f}s)", flush=True)
        if args.device == "cuda":
            torch.cuda.empty_cache()

    out_dir = os.path.join(OUT_ROOT, "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    gt = load_gt_sizes(gt_file, gt_cal)
    out = dict(dataset=ds, alpha=args.alpha, n_trials=args.n_trials,
               pool_pr=pool_pr, proto_T=args.proto_T,
               selector_margin_pick=pick, selector_regime_pick=regime,
               ablation_info=ablation_info, gt_geodesic=gt, rows=rows)
    path = os.path.join(out_dir, f"results_{ds}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsaved -> {path}", flush=True)
    try:
        from plot_conformal_metric import plot_benchmark
        plot_benchmark(out, os.path.join(out_dir, f"benchmark_{ds}.png"))
    except Exception as e:
        print(f"[warn] benchmark plot failed: {e}")
    return out


# ---------------------------------------------------------------------------
# Phase: validity
# ---------------------------------------------------------------------------
def phase_validity(ds, args):
    X, y, Xu, K, cal_sizes, gt_cal, gt_file = load_dataset(ds)
    allc = np.unique(y)
    out = dict(dataset=ds, alpha=args.alpha, checks={})

    tf_g1 = load_g1_transform(ds, Xu, rung=1)
    if tf_g1 is None:
        print(f"[{ds}] no rung1 fit found -- run --phase rung1 first")
        return None
    arms = {"raw768": IdentityTransform().fit(), "g1_r1": tf_g1}

    # (1) random-split coverage, high-trial, vs the known-exact reference
    for arm, tf in arms.items():
        row = measure_fcp(tf, X, y, allc, max(cal_sizes), "random",
                          "unwhitened_topk_mean", args.n_trials, args.alpha,
                          args.device, args.seed)
        se = np.sqrt(args.alpha * (1 - args.alpha)
                     / (row["n_trials"] * 5 * K))
        row["binomial_2se_band"] = [1 - args.alpha - 2 * se,
                                    1 - args.alpha + 2 * se]
        out["checks"][f"random_cov_{arm}"] = row
        print(f"[{ds}] random-split {arm}: cov={row['cov']:.4f} "
              f"(band {row['binomial_2se_band'][0]:.4f}"
              f"-{row['binomial_2se_band'][1]:.4f})", flush=True)

    # (2) permutation oracle: one fixed bag, R re-partitions, fixed transform
    cal = max(cal_sizes)
    rng = np.random.default_rng(args.seed)
    m_cal, m_test = cal // K, 5
    ci, ti = balanced_split(y, allc, m_cal, m_test, rng)
    bag = np.concatenate([ci, ti])
    Zbag, ybag = tf_g1.transform(X[bag]), y[bag]
    n_cal = len(ci)
    covs = []
    for r in range(args.n_perm):
        prm = np.random.default_rng(args.seed + 31 * r).permutation(len(bag))
        bc, bt = prm[:n_cal], prm[n_cal:]
        ncm = create_ncm("unwhitened_topk_mean", k=5)
        cp = FullConformalPredictor(ncm, alpha=args.alpha)
        cp.calibrate(Zbag[bc], ybag[bc], all_classes=allc)
        try:
            res = cp.predict(Zbag[bt], verbose=False, device=args.device)
        except (RuntimeError, ValueError):
            res = cp.predict(Zbag[bt], verbose=False, device="cpu")
        psets = res["prediction_sets"]
        covs.append(float(np.mean([ybag[bt][i] in psets[i]
                                   for i in range(len(bt))])))
    lo, hi = 1 - args.alpha, 1 - args.alpha + 1.0 / (n_cal + 1)
    out["checks"]["perm_oracle"] = dict(
        mean_cov=float(np.mean(covs)), n_perm=args.n_perm,
        theory_band=[lo, hi],
        in_band=bool(lo - 2 * np.std(covs) / np.sqrt(len(covs))
                     <= np.mean(covs)
                     <= hi + 2 * np.std(covs) / np.sqrt(len(covs))))
    print(f"[{ds}] perm oracle: mean cov={np.mean(covs):.4f} "
          f"band=[{lo:.4f},{hi:.4f}] in_band="
          f"{out['checks']['perm_oracle']['in_band']}", flush=True)

    out_dir = os.path.join(OUT_ROOT, "validity")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"validity_{ds}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"saved -> {path}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["landscape", "rung1", "rung2", "benchmark",
                             "validity"])
    ap.add_argument("--datasets", nargs="+", default=["cifar100"],
                    choices=list(DATASETS))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--n_probe", type=int, default=24)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--pseudo_k_mult", type=int, nargs="+", default=None,
                    help="landscape mitigation: harder pseudo-tasks K'=m*K "
                         "(one landscape JSON per multiplier)")
    ap.add_argument("--reuse_true", action="store_true",
                    help="landscape: reuse baseline probes' true FCP sizes "
                         "(pseudo-task-independent), recompute surrogates only")
    ap.add_argument("--proto_T", type=float, default=0.06)
    ap.add_argument("--ablation", action="store_true",
                    help="benchmark: add stage/factor ablation arms "
                         "(g1_s_only, g1_gate_only, g1_power_only, cw768)")
    ap.add_argument("--contaminated", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu")
        args.device = "cpu"

    for ds in args.datasets:
        print(f"\n================ {args.phase} : {ds} ================",
              flush=True)
        if args.phase == "landscape":
            phase_landscape(ds, args)
        elif args.phase == "rung1":
            phase_fit(ds, args, rung=1)
        elif args.phase == "rung2":
            phase_fit(ds, args, rung=2)
        elif args.phase == "benchmark":
            phase_benchmark(ds, args)
        elif args.phase == "validity":
            phase_validity(ds, args)


if __name__ == "__main__":
    main()
