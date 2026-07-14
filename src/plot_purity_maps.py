"""2-D purity-map visualizations for the thread's top experiments: the
rank-space score plane with the selected region (D <= qhat), the reference
clouds, and the cal true scores -- one trial (t=0, run-matched seeds).

Run: python src/plot_purity_maps.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdcp_pool_pilot as M
from conformal_prediction import PrototypeSoftmaxNCM

EMB = r"C:/Users/IDO/Desktop/Ido_student/Msc/ssl-cp-research/output/from_cluster/embeddings"
OUT = r"C:/Users/IDO/Desktop/Ido_student/Msc/ssl-cp-research/output/mdcp_pool_pilot/figs"

CONFIGS = [
    dict(name="cifar100_bal200_multires",
         title="CIFAR-100 balanced cal=200 -- multi-resolution pair "
               "(first balanced win: 5.93 vs 7.05)",
         ds="cifar100", dims=[("proto", "final__pca128_cw"),
                              ("proto", "final__pca32_cw")],
         split="balanced_both", cal=200, gt_aug=False,
         xlab="proto @ pca128 rank (fine)", ylab="proto @ pca32 rank (coarse)"),
    dict(name="cars_bal784_geoproto",
         title="Stanford Cars balanced cal=784 -- family-complementary pair "
               "(biggest combo win: 19.7 vs 42.8)",
         ds="stanford_cars", dims=[("geo", "final__pca512_cw"),
                                   ("proto", "final__pca128_cw")],
         split="balanced_both", cal=800, gt_aug=False, strip_zoom=True,
         xlab="geodesic @ pca512 rank", ylab="proto @ pca128 rank"),
    dict(name="mini_rand200_gtaug",
         title="miniImageNet random cal=200 + GT corner aug "
               "(97.6 -> 14.2 @ 0.911)",
         ds="miniimagenet", dims=[("proto", "final__pca128_cw"),
                                  ("proto", "final__pca32_cw")],
         split="random", cal=200, gt_aug=True,
         xlab="proto @ pca128 rank (fine)", ylab="proto @ pca32 rank (coarse)"),
]


def pick_trial_map(cfg, qhat_max=1e3, max_trials=10, **kw):
    """Some configs are bimodal across trials (a strip of LAC=1 cal points
    puts the quantile at an include/exclude knife edge -- e.g. Cars, where
    ~10% of cal is misclassified into the corner stratum by the near-argmax
    temperature). For ILLUSTRATION pick the first trial whose qhat is
    moderate; the trial index is recorded in the title."""
    for t in range(max_trials):
        m = one_trial_map(cfg, trial=t, **kw)
        if m["qhat"] <= qhat_max:
            m["trial"] = t
            return m
    m["trial"] = max_trials - 1
    return m


def one_trial_map(cfg, seed=0, trial=0, k_d=10, alpha=0.1, grid_n=350):
    X_src, y = M.load_embedding_sources(f"{EMB}/embeddings_{cfg['ds']}_layers.pt")
    Xu_src, _ = M.load_embedding_sources(
        f"{EMB}/embeddings_{cfg['ds']}_unlabeled_layers.pt")
    classes = np.unique(y)
    K = len(classes)
    views = [v for _, v in cfg["dims"]]
    feats = M.build_view_feats(X_src, Xu_src, views)
    T = {}
    for v in set(v for n, v in cfg["dims"] if n == "proto"):
        rng0 = np.random.default_rng(seed)
        ci0, _ = M.balanced_both_split(y, classes, 8 * K, rng0)
        p = PrototypeSoftmaxNCM(temperature=None, logit="cosine",
                                allow_nonexchangeable=True)
        p.fit(feats[v][0][ci0], y[ci0])
        T[v] = float(p._T)

    rng = np.random.default_rng(seed + 1000 * trial + cfg["cal"])  # run-matched seeds
    if cfg["split"] == "balanced_both":
        ci, ti = M.balanced_both_split(y, classes, cfg["cal"], rng)
    else:
        ci, ti = M.random_split(y, classes, cfg["cal"], rng)
    yc = y[ci]
    k_geo = max(1, min(5, len(yc) // K))

    dims, rank = {}, {}
    for name, view in cfg["dims"]:
        Zv, Zuv = feats[view]
        Zc = Zv[ci]
        if name == "geo":
            d = dict(pool=M.geodesic_asym_scores(Zuv, Zc, yc, classes, k_geo),
                     cal=M.geodesic_asym_scores(Zc, Zc, yc, classes, k_geo,
                                                self_is_cal=True))
        else:
            d = dict(pool=M.prototype_lac_scores(Zuv, Zc, yc, classes, T[view]),
                     cal=M.prototype_lac_scores(Zc, Zc, yc, classes, T[view],
                                                loo=True))
        key = f"{name}_{view}"
        dims[key] = d
        ref = M.ecdf_fit(d["pool"])
        rank[key] = {p_: M.ecdf_apply(ref, d[p_]) for p_ in d}
    keys = list(dims)

    rows = np.arange(len(yc))
    col = np.searchsorted(classes, yc)
    q_cal = np.stack([rank[k]["cal"][rows, col] for k in keys], axis=-1)
    proto_key = next(k for k in keys if k.startswith("proto"))
    yhat = np.argmin(dims[proto_key]["pool"], axis=1)
    S_pool = np.stack([rank[k]["pool"] for k in keys], axis=-1)
    cloud, is_true = M.build_cloud(S_pool, yhat)

    extra = None
    if cfg["gt_aug"]:
        cols = []
        for name, view in cfg["dims"]:
            Zv, Zuv = feats[view]
            miss = (np.ones(len(Zuv)) if name == "proto"
                    else M.missing_class_scores(Zuv, Zv[ci], k_geo)[0])
            cols.append(M.ecdf_apply(M.ecdf_fit(dims[f"{name}_{view}"]["pool"]),
                                     miss))
        aug = np.stack(cols, axis=-1)
        n1 = int((np.bincount(col, minlength=K) == 1).sum())
        n_gt = min(len(aug), int(round(len(aug) * n1 / len(yc))))
        sub = np.random.default_rng(seed + 7000 * trial + cfg["cal"]).permutation(len(aug))
        extra = aug[sub[:n_gt]]
        cloud = np.vstack([cloud, extra])
        is_true = np.concatenate([is_true, np.ones(len(extra), dtype=bool)])

    D = M.PurityD(cloud, is_true, k_d)
    calD = D(q_cal)
    r = len(calD)
    qhat = np.quantile(calD, min(np.ceil((r + 1) * (1 - alpha)) / r, 1.0),
                       method="higher")

    gx, gy = np.meshgrid(np.linspace(0, 1, grid_n), np.linspace(0, 1, grid_n))
    sel = (D(np.c_[gx.ravel(), gy.ravel()]) <= qhat).reshape(grid_n, grid_n)
    out = dict(gx=gx, gy=gy, sel=sel, cloud=cloud, is_true=is_true,
               q_cal=q_cal, qhat=qhat, extra=extra)
    if cfg.get("strip_zoom"):
        # the proto axis is near-binary on this data (misclassified points at
        # LAC ~= 1): the discriminative action lives in a pixel-thin strip
        y_top = float(q_cal[:, 1].max())
        zy = np.linspace(y_top - 0.004, min(1.0004, y_top + 0.0008), 120)
        zgx, zgy = np.meshgrid(np.linspace(0, 1, 400), zy)
        out["zoom"] = (zgx, zgy,
                       (D(np.c_[zgx.ravel(), zgy.ravel()]) <= qhat)
                       .reshape(zgx.shape))
    return out


def render(cfg, m):
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.contourf(m["gx"], m["gy"], m["sel"], levels=[0.5, 1.5],
                colors=["#9ecae1"], alpha=0.5)
    f = m["cloud"][~m["is_true"]]
    step_f = max(1, len(f) // 6000)
    ax.scatter(f[::step_f, 0], f[::step_f, 1], s=2, c="#cccccc",
               label="false cloud (sub)")
    t = m["cloud"][m["is_true"]]
    step_t = max(1, len(t) // 1500)
    ax.scatter(t[::step_t, 0], t[::step_t, 1], s=5, c="#1f77b4",
               label="pseudo-true cloud (sub)")
    if m["extra"] is not None and len(m["extra"]):
        ax.scatter(m["extra"][:, 0], m["extra"][:, 1], s=14, c="#ff7f0e",
                   label=f"GT corner stratum (n={len(m['extra'])})")
    ax.scatter(m["q_cal"][:, 0], m["q_cal"][:, 1], s=24, c="k", marker="x",
               label="cal true scores")
    ax.set_xlabel(cfg["xlab"], fontsize=11)
    ax.set_ylabel(cfg["ylab"], fontsize=11)
    ax.set_title(f"{cfg['title']}\nselected region D<=qhat={m['qhat']:.3g} shaded "
                 f"(trial {m.get('trial', 0)}, deployable yhat)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if "zoom" in m:
        zgx, zgy, zsel = m["zoom"]
        axi = ax.inset_axes([0.08, 0.35, 0.84, 0.28])
        axi.contourf(zgx, zgy, zsel, levels=[0.5, 1.5], colors=["#9ecae1"],
                     alpha=0.75)
        strip = m["q_cal"][:, 1] > zgy.min()
        axi.scatter(m["q_cal"][strip, 0], m["q_cal"][strip, 1], s=22, c="k",
                    marker="x")
        f = m["cloud"][~m["is_true"]]
        fs = f[f[:, 1] > zgy.min()]
        axi.scatter(fs[::20, 0], fs[::20, 1], s=2, c="#999999")
        axi.set_ylim(zgy.min(), zgy.max())
        axi.ticklabel_format(useOffset=False, style="plain", axis="y")
        axi.set_title("ZOOM: the misclassified strip (proto-LAC = 1) is ENTIRELY "
                      "EXCLUDED --\nits ~10% of cal is the alpha budget; every "
                      "test point's ~194 near-argmax\nfalse candidates land here "
                      "and are rejected wholesale", fontsize=8)
        axi.tick_params(labelsize=7)
    fig.tight_layout()
    p = f"{OUT}/fig_puritymap_{cfg['name']}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for cfg in CONFIGS:
        print("rendering", cfg["name"], "...")
        m = pick_trial_map(cfg)
        print(f"  trial {m['trial']}, qhat {m['qhat']:.3g}")
        render(cfg, m)
