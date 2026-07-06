"""
Plots for the MDCP pool pilot yhat-lever experiment (src/mdcp_pool_pilot.py).

Figure 1  fig_size_<ds>_<split>.png   set size vs cal size per arm (log y)
                                      + coverage strip (0.9 line).
Figure 2  fig_yhat_lever_<ds>.png     the starved corner (random cal=200):
                                      set size per arm, annotated with pseudo-
                                      label accuracy + median qhat. Message:
                                      corner SUPPORT, not label accuracy, is
                                      the lever.
Figure 3  fig_mechanism_<ds>.png      one-trial 2-D rank-space picture: false
                                      cloud, pseudo-true cloud, synthetic
                                      corner stratum, cal true points, and the
                                      D<=qhat selected region with vs without
                                      corner augmentation.

Run:
python src/plot_mdcp_pilot.py --results_json output/mdcp_pool_pilot/cifar100_yhat/mdcp_pool_pilot_results.json \
    --embeddings_path <...>/embeddings_cifar100.pt --unlabeled_path <...>/embeddings_cifar100_unlabeled.pt \
    --dataset cifar100 --out_dir output/mdcp_pool_pilot/figs
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdcp_pool_pilot as M
from exchangeable_features import make_transform

ARM_STYLE = {
    "raw1_geo":               dict(color="#888888", ls="--", marker="s", label="raw geodesic (1-D SCP)"),
    "raw1_proto":             dict(color="#bbbbbb", ls="--", marker="v", label="raw prototype (1-D SCP)"),
    "dratio2_proto":          dict(color="#1f77b4", ls="-",  marker="o", label="2-D D-ratio, proto yhat"),
    "dratio2_lp":             dict(color="#17becf", ls="-",  marker="^", label="2-D D-ratio, LP yhat"),
    "dratio2_proto_aug":      dict(color="#d62728", ls="-",  marker="o", label="2-D D-ratio, proto yhat + corner aug (full)"),
    "dratio2_lp_aug":         dict(color="#ff7f0e", ls="-",  marker="^", label="2-D D-ratio, LP yhat + corner aug (full)"),
    "dratio2_proto_gt":       dict(color="#8c564b", ls="-",  marker="o", label="2-D D-ratio, proto yhat + corner aug (GT dose)"),
    "dratio2_proto_gt4":      dict(color="#e377c2", ls="-",  marker="o", label="2-D D-ratio, proto yhat + corner aug (4x GT)"),
    "dratio2_lp_gt":          dict(color="#bcbd22", ls="-",  marker="^", label="2-D D-ratio, LP yhat + corner aug (GT dose)"),
    "dratio2_oracle":         dict(color="#2ca02c", ls=":",  marker="d", label="2-D D-ratio, ORACLE yhat"),
    "dratio2_oracle_aug":     dict(color="#98df8a", ls=":",  marker="d", label="2-D D-ratio, ORACLE + corner aug"),
    "dratio2_oracle_present": dict(color="#9467bd", ls=":",  marker="x", label="2-D, oracle-on-present-classes"),
}


def fig_size_vs_cal(results, split, ds, out_dir):
    keys = sorted([k for k in results if k.startswith(split + "_cal")],
                  key=lambda k: int(k.split("cal")[1]))
    if not keys:
        return
    cals = [int(k.split("cal")[1]) for k in keys]
    fig, (ax, axc) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    for arm, st in ARM_STYLE.items():
        if arm not in results[keys[0]]:
            continue
        sz = [results[k][arm]["size"] for k in keys]
        se = [results[k][arm]["size_se"] for k in keys]
        cv = [results[k][arm]["cov"] for k in keys]
        cve = [results[k][arm]["cov_se"] for k in keys]
        ax.errorbar(cals, sz, yerr=se, capsize=3, **st)
        axc.errorbar(cals, cv, yerr=cve, capsize=3, **{**st, "label": None})
    ax.set_yscale("log")
    ax.set_ylabel("mean set size (log)", fontsize=12)
    ax.set_title(f"{ds}, {split} split: MDCP-pool arms vs 1-D baselines "
                 f"(20 trials, alpha=0.1)", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5, ncol=2)
    axc.axhline(0.9, color="k", lw=1, ls="--")
    axc.set_ylabel("coverage", fontsize=11)
    axc.set_xlabel("calibration size", fontsize=12)
    axc.grid(alpha=0.3)
    axc.set_xticks(cals)
    fig.tight_layout()
    p = os.path.join(out_dir, f"fig_size_{ds}_{split}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def fig_yhat_lever(results, ds, out_dir, key="random_cal200"):
    if key not in results:
        return
    r = results[key]
    arms = [a for a in ARM_STYLE if a in r]
    arms.sort(key=lambda a: r[a]["size"])
    sizes = [r[a]["size"] for a in arms]
    ses = [r[a]["size_se"] for a in arms]
    colors = [("#2ca02c" if "oracle" in a else
               "#d62728" if ("_aug" in a or "_gt" in a) else
               "#1f77b4" if a.startswith("dratio2") else "#999999") for a in arms]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(arms) + 2))
    ypos = np.arange(len(arms))
    ax.barh(ypos, sizes, xerr=ses, color=colors, alpha=0.85, capsize=3)
    for i, a in enumerate(arms):
        acc = r.get(f"_yhat_acc_{a.replace('dratio2_', '')}")
        qh = r[a].get("qhat_med")
        note = []
        if acc is not None:
            note.append(f"yhat acc {acc:.2f}")
        if qh is not None:
            note.append(f"med qhat {qh:.3g}")
        if sizes[i] > 55:   # long bar: annotate inside it to stay in-frame
            ax.text(sizes[i] - 2.0, i, "  ".join(note), va="center", ha="right",
                    fontsize=8.5, color="white", fontweight="bold")
        else:
            ax.text(sizes[i] + ses[i] + 1.0, i, "  ".join(note), va="center", fontsize=8.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([ARM_STYLE[a]["label"] for a in arms], fontsize=9.5)
    ax.set_xlabel("mean set size (K=100)", fontsize=12)
    ax.set_xlim(0, min(105, max(sizes) * 1.45))
    ax.set_title(f"{ds}, random cal=200 -- the yhat lever is corner SUPPORT, not accuracy:\n"
                 f"red = analytic missing-class corner augmentation (no labels needed)",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = os.path.join(out_dir, f"fig_yhat_lever_{ds}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def fig_mechanism(Z, y, Zu, yu, classes, T_proto, out_dir, ds, cal=200, seed=0,
                  k_d=10, alpha=0.1):
    """One-trial 2-D rank-space picture: proto cloud vs proto+corner-aug cloud."""
    rng = np.random.default_rng(seed + cal)  # matches run seeds (t=0)
    ci, ti = M.random_split(y, classes, cal, rng)
    Zc, yc, Zt, yt = Z[ci], y[ci], Z[ti], y[ti]
    K = len(classes)
    k_geo = max(1, min(5, len(yc) // K))
    G = dict(pool=M.geodesic_asym_scores(Zu, Zc, yc, classes, k_geo),
             cal=M.geodesic_asym_scores(Zc, Zc, yc, classes, k_geo, self_is_cal=True))
    P = dict(pool=M.prototype_lac_scores(Zu, Zc, yc, classes, T_proto),
             cal=M.prototype_lac_scores(Zc, Zc, yc, classes, T_proto, loo=True))
    rank = {}
    for name, dd in [("geo", G), ("proto", P)]:
        ref = M.ecdf_fit(dd["pool"])
        rank[name] = {p: M.ecdf_apply(ref, dd[p]) for p in dd}
    rows = np.arange(len(yc)); col = np.searchsorted(classes, yc)
    q_cal = np.stack([rank[n]["cal"][rows, col] for n in ("geo", "proto")], axis=-1)
    proto_col = np.argmin(P["pool"], axis=1)
    S_pool = np.stack([rank[n]["pool"] for n in ("geo", "proto")], axis=-1)
    cloud, is_true = M.build_cloud(S_pool, proto_col)
    s1m, s2m = M.missing_class_scores(Zu, Zc, k_geo)
    aug = np.stack([M.ecdf_apply(np.sort(G["pool"].ravel()), s1m),
                    M.ecdf_apply(np.sort(P["pool"].ravel()), s2m)], axis=-1)

    n_missing = K - len(np.unique(yc))
    cal_counts = np.array([(yc == c).sum() for c in np.unique(yc)])
    n_single = int((cal_counts == 1).sum())
    r_cal = len(q_cal)
    qlv = min(np.ceil((r_cal + 1) * (1 - alpha)) / r_cal, 1.0)

    D_pt = M.PurityD(cloud, is_true, k_d)                       # proto cloud
    cl_aug = np.vstack([cloud, aug])
    it_aug = np.concatenate([is_true, np.ones(len(aug), dtype=bool)])
    D_ag = M.PurityD(cl_aug, it_aug, k_d)                       # + corner stratum
    calD_pt, calD_ag = D_pt(q_cal), D_ag(q_cal)
    qh_pt = np.quantile(calD_pt, qlv, method="higher")
    qh_ag = np.quantile(calD_ag, qlv, method="higher")

    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1], height_ratios=[2.2, 1])

    # --- Panel A: full rank space, selected region under the AUG cloud ---
    axA = fig.add_subplot(gs[:, 0])
    grid_n = 350
    gx, gy = np.meshgrid(np.linspace(0, 1, grid_n), np.linspace(0, 1, grid_n))
    sel = (D_ag(np.c_[gx.ravel(), gy.ravel()]) <= qh_ag).reshape(grid_n, grid_n)
    axA.contourf(gx, gy, sel, levels=[0.5, 1.5], colors=["#9ecae1"], alpha=0.45)
    f_idx = np.where(~is_true)[0][::150]
    axA.scatter(cloud[f_idx, 0], cloud[f_idx, 1], s=2, c="#bbbbbb", label="false cloud (sub)")
    t_idx = np.where(is_true)[0][::8]
    axA.scatter(cloud[t_idx, 0], cloud[t_idx, 1], s=4, c="#1f77b4", label="pseudo-true cloud (sub)")
    axA.scatter(q_cal[:, 0], q_cal[:, 1], s=22, c="k", marker="x", label="cal true scores")
    axA.annotate(f"missing/singleton stratum lives in a\npixel-thin strip at s2 rank ~= 1.0\n"
                 f"(see zoom ->): {n_missing} missing + {n_single} singleton classes",
                 xy=(0.85, 1.0), xytext=(0.22, 0.78), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728")
    axA.set_xlabel("s1 rank (geodesic asym)", fontsize=11)
    axA.set_ylabel("s2 rank (prototype LAC)", fontsize=11)
    axA.set_title(f"rank space, selected region with corner aug (qhat={qh_ag:.3g})", fontsize=10.5)
    axA.legend(fontsize=8, loc="lower right")

    # --- Panel B: zoom into the top strip ---
    axB = fig.add_subplot(gs[:, 1])
    strip = cloud[:, 1] > 0.995
    fs = cloud[strip & ~is_true]
    axB.scatter(fs[::20, 0], fs[::20, 1], s=4, c="#bbbbbb", label="missing-class FALSE scores (sub)")
    axB.scatter(aug[::4, 0], aug[::4, 1], s=8, c="#ff7f0e", label="synthetic corner TRUE stratum (sub)")
    cal_strip = q_cal[:, 1] > 0.995
    axB.scatter(q_cal[cal_strip, 0], q_cal[cal_strip, 1], s=60, c="k", marker="x",
                label=f"cal true scores in strip (n={int(cal_strip.sum())})")
    axB.set_xlim(0.55, 1.005)
    lo = min(q_cal[cal_strip, 1].min() if cal_strip.any() else 0.9995, aug[:, 1].min()) - 2e-5
    axB.set_ylim(lo, 1.00003)
    axB.ticklabel_format(useOffset=False, style="plain", axis="y")
    axB.set_xlabel("s1 rank", fontsize=11)
    axB.set_title(f"ZOOM: the corner stratum (s2 rank ~ 1). Pseudo-labels put NO true mass here;\n"
                  f"the analytic augmentation does -- exactly where starved-class cal scores sit",
                  fontsize=9.5)
    axB.legend(fontsize=8, loc="lower left")

    # --- Panel C: cal-true D histograms + qhat lines ---
    axC = fig.add_subplot(gs[:, 2])
    bins = np.logspace(np.log10(max(min(calD_pt.min(), calD_ag.min()), 1e-3)),
                       np.log10(max(calD_pt.max(), calD_ag.max()) * 1.5), 45)
    axC.hist(calD_pt, bins=bins, alpha=0.5, color="#1f77b4", label="cal D, proto cloud")
    axC.hist(calD_ag, bins=bins, alpha=0.5, color="#ff7f0e", label="cal D, + corner aug")
    axC.axvline(qh_pt, color="#1f77b4", ls="--", lw=2, label=f"qhat proto = {qh_pt:.3g}")
    axC.axvline(qh_ag, color="#ff7f0e", ls="--", lw=2, label=f"qhat aug = {qh_ag:.3g}")
    axC.set_xscale("log")
    axC.set_xlabel("purity D of cal true scores (log)", fontsize=11)
    axC.set_ylabel("count", fontsize=11)
    axC.set_title("the quantile explosion, and its repair", fontsize=10.5)
    axC.legend(fontsize=8)

    fig.suptitle(f"{ds}, random cal={cal} ({n_missing} classes missing, {n_single} singletons): "
                 f"pseudo-label clouds lack the starved-class stratum -> qhat explodes "
                 f"({qh_pt:.3g}); the analytic corner augmentation repairs it ({qh_ag:.3g})",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, f"fig_mechanism_{ds}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", required=True)
    ap.add_argument("--embeddings_path")
    ap.add_argument("--unlabeled_path")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--out_dir", default="output/mdcp_pool_pilot/figs")
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = json.load(open(args.results_json))["results"]
    for split in ("random", "balanced_both"):
        fig_size_vs_cal(results, split, args.dataset, args.out_dir)
    fig_yhat_lever(results, args.dataset, args.out_dir)

    if args.embeddings_path and args.unlabeled_path:
        d = torch.load(args.embeddings_path, map_location="cpu", weights_only=False)
        X, y = d["embeddings"].numpy().astype(np.float64), d["labels"].numpy()
        du = torch.load(args.unlabeled_path, map_location="cpu", weights_only=False)
        Xu, yu = du["embeddings"].numpy().astype(np.float64), du["labels"].numpy()
        classes = np.unique(y)
        tr = make_transform(unlabeled=Xu, pca_dim=args.pca_dim, whiten="cluster")
        Z, Zu = tr.transform(X), tr.transform(Xu)
        rng0 = np.random.default_rng(args.seed)
        ci0, _ = M.balanced_both_split(y, classes, 8 * len(classes), rng0)
        from conformal_prediction import PrototypeSoftmaxNCM
        pn = PrototypeSoftmaxNCM(temperature=None, logit="cosine",
                                 allow_nonexchangeable=True)
        pn.fit(Z[ci0], y[ci0])
        fig_mechanism(Z, y, Zu, yu, classes, float(pn._T), args.out_dir, args.dataset,
                      seed=args.seed)


if __name__ == "__main__":
    main()
