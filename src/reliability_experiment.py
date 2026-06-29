"""
A-pilot: CALIBRATION-CONDITIONAL (PAC) RELIABILITY of few-shot Full CP.

Reframing direction A (see memory novelty-directions-pac-and-geometric-coverage):
the project reports coverage IN EXPECTATION over the calibration draw, but at
2-8 labels/class the draw is the dominant variance source. Here we characterise the
DISTRIBUTION of realized coverage and set size over many calibration draws, at a
FIXED budget B, for:
  - full + prototype_softmax (balanced cal, GPU)        [the project's tightest NCM]
  - full + geodesic           (balanced cal, GPU)
  - SCP-THR, matched budget B (B/2 train head + B/2 cal) [genuine inductive baseline]
  - full + geodesic           (random cal, GPU)          [exact-validity reference]

Headline axis = RELIABILITY VARIANCE: full CP feeds all B into the rank (no
train/cal split) so its coverage distribution should be tighter and -- the
most reviewer-proof part -- its SET SIZE far less volatile (SCP-THR collapses to
sz ~= K when the few-shot head fails on a draw). We also compute the PAC size-cost:
the extra set size to upgrade from "0.90 on average" to "0.90 w.p. >= 1-delta"
via a small-sample Beta correction (SSBC).

Theory anchors: split-CP coverage ~ Beta(k, n+1-k), SD ~= 0.3/sqrt(n+2) at alpha=0.1
(clean only for SCP); full CP has NO closed-form law -> measured. SSBC = Zwart 2025
(arXiv 2509.15349). Plain-text math (repo convention).
"""
import sys, os, json, argparse, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import make_transform
from split_cp_baselines import SoftmaxSplitCP

MAIN_OUT = r"C:\Users\IDO\Desktop\Ido_student\Msc\ssl-cp-research\output\from_cluster"
ALPHA_GRID = np.round(np.arange(0.05, 0.151, 0.01), 3)  # for the PAC size-cost curve


@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def load_emb(path):
    d = torch.load(path, map_location="cpu")
    return np.asarray(d["embeddings"], dtype=np.float64), np.asarray(d["labels"]).astype(int).ravel()


def carve_test_and_pool(y, m_test, allc, rng):
    test_idx, pool_idx = [], []
    for c in allc:
        perm = rng.permutation(np.where(y == c)[0])
        test_idx.append(perm[:m_test]); pool_idx.append(perm[m_test:])
    return np.concatenate(test_idx), np.concatenate(pool_idx)


def sample_cal(pool_idx, y, cal, balanced, allc, rng):
    if balanced:
        m = cal // len(allc)
        return np.concatenate([rng.choice(pool_idx[y[pool_idx] == c], m, replace=False)
                               for c in allc])
    return rng.choice(pool_idx, cal, replace=False)


def size_cov_from_pvals(pvals, y_test, K):
    """From the full-CP per-point {label: p} dicts, the set size and coverage at
    every alpha' in ALPHA_GRID (free re-threshold, no refit)."""
    n = len(y_test)
    P = np.zeros((n, K), dtype=np.float32)          # preallocate (frugal, sparse-safe)
    for i, d in enumerate(pvals):
        for c, v in d.items():
            ci = int(c)
            if 0 <= ci < K:
                P[i, ci] = v
    ptrue = P[np.arange(n), y_test]
    sizes = np.array([float((P > a).sum(1).mean()) for a in ALPHA_GRID])
    covs = np.array([float((ptrue > a).mean()) for a in ALPHA_GRID])
    return sizes, covs


def scp_size_cov_grid(scp, X_test, y_test, K):
    """SCP-THR size+coverage at every alpha' by re-quantiling cal_scores (cheap)."""
    Xs = scp.scaler.transform(X_test)
    probs = scp.classifier.predict_proba(Xs)            # (n, n_cls_seen)
    c2i = {c: i for i, c in enumerate(scp.classifier.classes_)}
    full = np.zeros((len(X_test), K))
    for c in range(K):
        if c in c2i:
            full[:, c] = probs[:, c2i[c]]
    ptrue = full[np.arange(len(y_test)), y_test]
    n_cal = len(scp.cal_scores)
    sizes = np.zeros(len(ALPHA_GRID)); covs = np.zeros(len(ALPHA_GRID))
    for j, a in enumerate(ALPHA_GRID):
        lvl = min(np.ceil((n_cal + 1) * (1 - a)) / n_cal, 1.0)
        q = np.quantile(scp.cal_scores, lvl)
        sizes[j] = (full >= 1.0 - q).sum(1).mean()
        covs[j] = (ptrue >= 1.0 - q).mean()
    return sizes, covs


def stats(arr):
    a = np.asarray(arr, float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                p5=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)),
                lo=float(a.min()), hi=float(a.max()))


def run(args):
    allc = np.arange(args.n_classes); K = args.n_classes
    dev = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    Xb, yb = load_emb(os.path.join(args.embeddings_dir, f"embeddings_{args.dataset}.pt"))
    Xu, _ = load_emb(os.path.join(args.embeddings_dir, f"embeddings_{args.dataset}_unlabeled.pt"))
    transform = make_transform(Xu, pca_dim=args.pca_dim, whiten="cluster", n_clusters=args.n_clusters)
    print(f"[data] {args.dataset} base {Xb.shape} pool {Xu.shape} device={dev} cal={args.cal} draws={args.n_draws}")

    # FIXED test set (same across all draws -> isolates calibration variance)
    trng = np.random.default_rng(args.seed + 7)
    test_idx, pool_idx = carve_test_and_pool(yb, args.m_test, allc, trng)
    Xt_raw, yt = Xb[test_idx], yb[test_idx]
    Xt_t = transform.transform(Xt_raw)
    print(f"[test] fixed test n={len(yt)} (noise sd ~ {np.sqrt(0.9*0.1/len(yt)):.4f})")

    # fixed pilot T for prototype (constant across draws => exact)
    prng = np.random.default_rng(args.seed + 9999)
    pci = sample_cal(pool_idx, yb, min(args.cal, 800), True, allc, prng)
    with quiet():
        pncm = create_ncm("prototype_softmax", temperature=None, logit="cosine", allow_nonexchangeable=True)
        pncm.fit(transform.transform(Xb[pci]), yb[pci]); T_fixed = float(pncm._resolve_T())
    print(f"[pilot] prototype fixed T={T_fixed:.4f}")

    arms = ["full_proto_bal", "full_geo_bal", "scp_thr_bal", "full_geo_rand"]
    rec = {a: {"cov": [], "mean_sz": [], "med_sz": [], "p90_sz": [], "max_sz": [],
               "frac_K": [], "grid_sz": [], "grid_cov": []} for a in arms}

    for d in range(args.n_draws):
        drng = np.random.default_rng(args.seed + d)
        ci_bal = sample_cal(pool_idx, yb, args.cal, True, allc, drng)
        ci_rand = sample_cal(pool_idx, yb, args.cal, False, allc, drng)

        def full_arm(name, ncm_type, ci, **kw):
            Xc_t = transform.transform(Xb[ci]); yc = yb[ci]
            use_dev = dev
            with quiet():
                ncm = create_ncm(ncm_type, **kw)
                cp = FullConformalPredictor(ncm, alpha=args.alpha)
                cp.calibrate(Xc_t, yc, all_classes=allc)
                try:
                    res = cp.predict(Xt_t, device=use_dev, return_p_values=True, verbose=False)
                except Exception:
                    res = cp.predict(Xt_t, device="cpu", return_p_values=True, verbose=False)
            psets = res["prediction_sets"]; sizes = np.array([len(p) for p in psets], float)
            cov = float(np.mean([yt[i] in set(p) for i, p in enumerate(psets)]))
            gsz, gcov = size_cov_from_pvals(res["p_values"], yt, K)
            _store(rec[name], cov, sizes, K, gsz, gcov)

        full_arm("full_proto_bal", "prototype_softmax", ci_bal, temperature=T_fixed, logit="cosine")
        full_arm("full_geo_bal", "unwhitened_topk_asym", ci_bal, k=5)
        full_arm("full_geo_rand", "unwhitened_topk_asym", ci_rand, k=5)

        # SCP-THR, matched budget B = cal: split balanced cal 50/50 train/cal
        half = len(ci_bal) // 2
        perm = drng.permutation(ci_bal); tr, cl = perm[:half], perm[half:]
        with quiet():
            scp = SoftmaxSplitCP(alpha=args.alpha)
            scp.fit(transform.transform(Xb[tr]), yb[tr])
            scp.calibrate(transform.transform(Xb[cl]), yb[cl], all_classes=allc)
            ev = scp.predict(Xt_t)
        psets = ev["prediction_sets"]; sizes = np.array([len(p) for p in psets], float)
        cov = float(np.mean([yt[i] in set(p) for i, p in enumerate(psets)]))
        gsz, gcov = scp_size_cov_grid(scp, Xt_t, yt, K)
        _store(rec["scp_thr_bal"], cov, sizes, K, gsz, gcov)

        if (d + 1) % 20 == 0:
            print(f"  draw {d+1}/{args.n_draws}: "
                  + " | ".join(f"{a.split('_')[1]}:{np.mean(rec[a]['cov']):.3f}/{np.mean(rec[a]['mean_sz']):.2f}"
                               for a in arms))

    summary = _summarize(rec, args)
    os.makedirs(os.path.join(args.out_dir, args.dataset), exist_ok=True)
    p = os.path.join(args.out_dir, args.dataset, f"reliability_cal{args.cal}.json")
    with open(p, "w") as f:
        json.dump({"config": vars(args), "alpha_grid": ALPHA_GRID.tolist(),
                   "summary": summary, "raw": {a: {k: rec[a][k] for k in
                   ["cov", "mean_sz", "p90_sz", "max_sz", "frac_K"]} for a in arms}}, f, indent=2)
    print(f"[done] wrote {p}")
    if args.plot:
        make_plots(rec, summary, args)


def _store(r, cov, sizes, K, gsz, gcov):
    r["cov"].append(cov); r["mean_sz"].append(float(sizes.mean()))
    r["med_sz"].append(float(np.median(sizes))); r["p90_sz"].append(float(np.percentile(sizes, 90)))
    r["max_sz"].append(float(sizes.max())); r["frac_K"].append(float((sizes >= K).mean()))
    r["grid_sz"].append(gsz.tolist()); r["grid_cov"].append(gcov.tolist())


def _summarize(rec, args):
    out = {}
    n_cal_scp = args.cal // 2
    beta_sd_scp = float(np.sqrt(0.9 * 0.1 / (n_cal_scp + 2)))
    beta_sd_full = float(np.sqrt(0.9 * 0.1 / (args.cal + 2)))
    from scipy.stats import beta as Beta
    for a, r in rec.items():
        cov = np.array(r["cov"]);
        S = {"coverage": stats(cov),
             "p_cov_below_0.88": float((cov < 0.88).mean()),
             "p_cov_below_0.90": float((cov < 0.90).mean()),
             "mean_size": stats(r["mean_sz"]), "p90_size": stats(r["p90_sz"]),
             "max_size": stats(r["max_sz"]), "frac_K": stats(r["frac_K"])}
        # PAC size-cost at delta: smallest alpha' with empirical P(cov(a')>=0.90)>=1-delta
        gcov = np.array(r["grid_cov"]); gsz = np.array(r["grid_sz"])  # (draws, grid)
        pac = {}
        for delta in (0.10, 0.05):
            ok = (gcov >= 0.90).mean(0) >= (1 - delta)   # per-alpha' reliability
            if ok.any():
                j = np.where(ok)[0][0]                    # smallest alpha' (grid ascending coverage)
                ap = float(ALPHA_GRID[j]); base = float(gsz[:, ALPHA_GRID == 0.10].mean())
                pac[f"delta_{delta}"] = {"alpha_prime": ap,
                                          "mean_size": float(gsz[:, j].mean()),
                                          "size_at_0.10": base,
                                          "pac_price": float(gsz[:, j].mean() - base)}
            else:
                pac[f"delta_{delta}"] = {"alpha_prime": None, "note": "grid floor 0.05 insufficient"}
        S["pac"] = pac
        out[a] = S
    out["_reference"] = {"beta_sd_scp_ncal%d" % n_cal_scp: beta_sd_scp,
                         "beta_sd_full_ncal%d" % args.cal: beta_sd_full,
                         "note": "Beta SD is exact for SCP-THR (split); full CP has no Beta law (measured)."}
    # console headline
    print("\n==== RELIABILITY SUMMARY (cal=%d, %d draws) ====" % (args.cal, args.n_draws))
    print("arm                  cov_mean  cov_sd   cov_p5  P(<.88)  sz_mean  sz_p90  sz_max  frac_K")
    for a in rec:
        S = out[a]
        print(f"{a:20s} {S['coverage']['mean']:.4f}  {S['coverage']['sd']:.4f}  "
              f"{S['coverage']['p5']:.3f}  {S['p_cov_below_0.88']:.2f}     "
              f"{S['mean_size']['mean']:.2f}    {S['p90_size']['mean']:.2f}   "
              f"{S['max_size']['mean']:.1f}  {S['frac_K']['mean']:.3f}")
    print(f"[ref] Beta SD: SCP(n={n_cal_scp})={beta_sd_scp:.4f}  full-if-Beta(n={args.cal})={beta_sd_full:.4f}")
    return out


def make_plots(rec, summary, args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import beta as Beta
    arms = list(rec.keys())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # 1 coverage histograms
    for a in arms:
        ax[0].hist(rec[a]["cov"], bins=20, alpha=0.45, label=a)
    n_scp = args.cal // 2; k = int(np.ceil((1 - args.alpha) * (n_scp + 1)))
    xs = np.linspace(0.8, 1.0, 200)
    ax[0].plot(xs, Beta(k, n_scp + 1 - k).pdf(xs) / Beta(k, n_scp + 1 - k).pdf(xs).max()
               * ax[0].get_ylim()[1], "k--", lw=1, label=f"Beta(n={n_scp}) [SCP]")
    ax[0].axvline(0.90, c="r", lw=1); ax[0].set_title("realized coverage"); ax[0].legend(fontsize=7)
    # 2 set-size volatility (box of per-draw mean & p90)
    ax[1].boxplot([rec[a]["mean_sz"] for a in arms], labels=[a.split("_")[1] for a in arms], showfliers=True)
    ax[1].set_title("per-draw mean set size"); ax[1].tick_params(axis="x", labelsize=7)
    # 3 PAC size-cost curve (full arms)
    for a in arms:
        gsz = np.array(rec[a]["grid_sz"]).mean(0)
        ax[2].plot(ALPHA_GRID, gsz, "o-", ms=3, label=a)
    ax[2].axvline(0.10, c="r", lw=1); ax[2].set_xlabel("alpha' (deploy level)")
    ax[2].set_ylabel("mean set size"); ax[2].set_title("PAC size-cost"); ax[2].legend(fontsize=7)
    fig.suptitle(f"{args.dataset} reliability: cal={args.cal}, {args.n_draws} draws")
    fig.tight_layout()
    p = os.path.join(args.out_dir, args.dataset, f"reliability_cal{args.cal}.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"[plot] {p}")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--embeddings_dir", default=MAIN_OUT)
    ap.add_argument("--out_dir", default="output/reliability")
    ap.add_argument("--n_classes", type=int, default=100)
    ap.add_argument("--cal", type=int, default=400)
    ap.add_argument("--m_test", type=int, default=20)
    ap.add_argument("--n_draws", type=int, default=100)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--n_clusters", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--plot", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse())
