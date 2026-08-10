"""Empirical score-histogram motivation figure for the DWT theory draft.

For each dataset and each arm (raw representation vs DWT = qe pool-neighbor
denoise -> pool PCA -> pool cluster-whiten), plot the distribution of
true-class vs false-class prototype-cosine nonconformity scores on the test
set, with the split-CP quantile q_hat (from calibration) marked. The theory
claim (docs/dwt_theory.md, Section 0/Prop C) is that DWT tightens sets by
shrinking the overlap between the false-class score mass and the true-class
right tail; this figure is the empirical form of that claim, including the
negative regime (aircraft: low pool homophily -> no separation gain).

Same nonconformity scoring in both arms: s(x,c) = -cos(z, m_c), prototypes
m_c = L2-normalized calibration class means in the arm's own space.

Usage (from repo root):
    python src/dwt_score_histograms.py \
        --emb_dir output/from_cluster/embeddings --out_dir output/dwt_histograms
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conformal_prediction import stratified_cal_test_split  # noqa: E402
from exchangeable_features import UnlabeledTransform        # noqa: E402

ALPHA = 0.1
QE_K = 10
QE_A = 3.0
PCA_DIM = 128
N_CLUSTERS = 20
CAL_SIZE = 400
TEST_SIZE = 2000
SEED = 42


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def qe_smooth(X, pool, k=QE_K, a=QE_A, exclude_self=False):
    """Classic alpha-QE against a frozen L2-normed pool:
    T(x) = L2norm(x + sum_{i in NN_k(x)} max(cos(x,u_i),0)^a * u_i).
    Mirrors exchangeable_features._qe_smooth (pool-repr-menu worktree)."""
    P = l2n(pool.astype(np.float64))
    Xn = l2n(X.astype(np.float64))
    out = np.empty_like(Xn)
    for i0 in range(0, len(Xn), 1024):
        ch = Xn[i0:i0 + 1024]
        S = ch @ P.T
        if exclude_self:
            rows = np.arange(len(ch))
            S[rows, i0 + rows] = -np.inf
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        s = np.take_along_axis(S, idx, axis=1)
        w = np.clip(s, 0.0, None) ** a
        v = ch + (w[:, :, None] * P[idx]).sum(axis=1)
        out[i0:i0 + 1024] = l2n(v)
    return out


def knn_homophily(X, y, k=QE_K):
    """h(k): mean fraction of the k nearest OTHER labeled points sharing the
    ego's label (kNN label homophily diagnostic, raw space)."""
    Xn = l2n(X.astype(np.float64))
    same = 0.0
    n = len(Xn)
    for i0 in range(0, n, 1024):
        ch = Xn[i0:i0 + 1024]
        S = ch @ Xn.T
        rows = np.arange(len(ch))
        S[rows, i0 + rows] = -np.inf
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        same += (y[idx] == y[i0:i0 + 1024, None]).mean(axis=1).sum()
    return same / n


def scores_and_qhat(Z_cal, y_cal, Z_test, y_test):
    """Prototype-cosine nonconformity scores + split-CP quantile."""
    K = int(max(y_cal.max(), y_test.max())) + 1
    protos = np.zeros((K, Z_cal.shape[1]))
    for c in range(K):
        m = y_cal == c
        if m.any():
            protos[c] = Z_cal[m].mean(axis=0)
    protos = l2n(protos)
    s_test = -(l2n(Z_test) @ protos.T)    # (n_test, K)

    # cal true-class scores with exact LOO prototypes (each cal point removed
    # from its own class mean before scoring -- same self-leak repair as the
    # repo's prototype NCM; without it q_hat is biased low and coverage breaks)
    n = len(y_cal)
    Zc = l2n(Z_cal)
    cal_true = np.empty(n)
    for c in np.unique(y_cal):
        m = y_cal == c
        n_c = int(m.sum())
        cls = Zc[m]
        if n_c < 2:
            loo = np.repeat(l2n(cls.sum(0, keepdims=True)), n_c, axis=0)
        else:
            loo = l2n((cls.sum(axis=0, keepdims=True) - cls) / (n_c - 1))
        cal_true[m] = -np.einsum("ij,ij->i", cls, loo)
    m_idx = int(np.ceil((1 - ALPHA) * (n + 1))) - 1     # 0-based order stat
    q_hat = np.sort(cal_true)[min(m_idx, n - 1)]

    nt = len(y_test)
    test_true = s_test[np.arange(nt), y_test]
    mask_false = np.ones_like(s_test, dtype=bool)
    mask_false[np.arange(nt), y_test] = False
    test_false = s_test[mask_false]

    coverage = float((test_true <= q_hat).mean())
    sizes = (s_test <= q_hat).sum(axis=1)
    leakage = float(((s_test <= q_hat) & mask_false).sum(axis=1).mean())
    return dict(q_hat=float(q_hat), coverage=coverage,
                avg_size=float(sizes.mean()), leakage=leakage,
                true=test_true, false=test_false,
                cal_disp=float(np.abs(cal_true - q_hat).mean()))


def run_dataset(name, emb_dir):
    lab = torch.load(os.path.join(emb_dir, f"embeddings_{name}.pt"),
                     map_location="cpu", weights_only=False)
    unl = torch.load(os.path.join(emb_dir, f"embeddings_{name}_unlabeled.pt"),
                     map_location="cpu", weights_only=False)
    X = lab["embeddings"].numpy().astype(np.float64)
    y = lab["labels"].numpy().astype(int)
    U = unl["embeddings"].numpy().astype(np.float64)

    Xc, yc, Xt, yt = stratified_cal_test_split(
        X, y, cal_size=CAL_SIZE, test_size=TEST_SIZE, random_state=SEED)

    h = knn_homophily(X, y)

    arms = {}
    # raw arm: natural representation, same scoring
    arms["raw"] = scores_and_qhat(l2n(Xc), yc, l2n(Xt), yt)

    # DWT arm: qe denoise (cal/test + pool) -> PCA + cluster-whiten (pool-fit)
    U_s = qe_smooth(U, U, exclude_self=True)
    Xc_s = qe_smooth(Xc, U)
    Xt_s = qe_smooth(Xt, U)
    tr = UnlabeledTransform(pca_dim=PCA_DIM, whiten="cluster",
                            n_clusters=N_CLUSTERS, random_state=SEED).fit(U_s)
    arms["dwt"] = scores_and_qhat(tr.transform(Xc_s), yc,
                                  tr.transform(Xt_s), yt)
    return h, arms


def make_overlay_fig(store, out_dir, results):
    """Second figure: empirical forms of Prop C's two hypotheses, zoomed on
    the quantile region R_alpha. Col 1: true-class score CDF F_true (raw vs
    DWT) -- condition (i) is 'DWT curve above raw'. Col 2: mean false-count
    G(r) = E[#{c != y : s(x,c) <= r}] -- condition (ii) is 'DWT below raw'."""
    datasets = list(store.keys())
    fig, axes = plt.subplots(len(datasets), 2,
                             figsize=(12, 4.2 * len(datasets)), squeeze=False)
    colors = {"raw": "#4c72b0", "dwt": "#dd8452"}
    for r, ds in enumerate(datasets):
        arms = store[ds]
        qs = {a: arms[a]["q_hat"] for a in arms}
        x0 = min(qs.values()) - 0.15
        x1 = max(qs.values()) + 0.15
        grid = np.linspace(x0, x1, 400)
        curves = {}
        for arm in ["raw", "dwt"]:
            st = arms[arm]
            n_test = len(st["true"])
            F = np.searchsorted(np.sort(st["true"]), grid, side="right") / n_test
            G = np.searchsorted(np.sort(st["false"]), grid, side="right") / n_test
            curves[arm] = (F, G)
        # dominance checks on the shown region
        dom_i = float((curves["dwt"][0] >= curves["raw"][0]).mean())
        dom_ii = float((curves["dwt"][1] <= curves["raw"][1]).mean())
        results[ds]["dominance_on_region"] = {
            "frac_F_dwt_above_raw": dom_i, "frac_G_dwt_below_raw": dom_ii,
            "region": [float(x0), float(x1)]}

        axF, axG = axes[r][0], axes[r][1]
        for arm in ["raw", "dwt"]:
            lab = "raw" if arm == "raw" else "DWT"
            axF.plot(grid, curves[arm][0], color=colors[arm], lw=2, label=lab)
            axG.plot(grid, curves[arm][1], color=colors[arm], lw=2, label=lab)
            for ax in (axF, axG):
                ax.axvline(qs[arm], color=colors[arm], ls="--", lw=1.2)
        axF.axhline(1 - ALPHA, color="gray", ls=":", lw=1,
                    label=f"{1-ALPHA:.0%} level")
        axF.set_title(f"{ds} — true-class score CDF F_true\n"
                      f"condition (i): DWT above raw on "
                      f"{dom_i:.0%} of the region", fontsize=10)
        axF.set_ylabel("F_true(r) = P(s(x,y) <= r)")
        gq = {a: float(np.interp(qs[a], grid, curves[a][1])) for a in qs}
        axG.set_title(f"{ds} — mean false-count G(r)\n"
                      f"condition (ii): DWT below raw on {dom_ii:.0%};  "
                      f"G(q_hat): {gq['raw']:.2f} -> {gq['dwt']:.2f}",
                      fontsize=10)
        axG.set_ylabel("G(r) = E[# false classes with s <= r]")
        for ax in (axF, axG):
            ax.set_xlabel("score r   (dashed lines: each arm's q_hat)")
            ax.set_xlim(x0, x1)
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Prop C hypotheses on the quantile region: (i) true-score "
                 "CDF dominance sets q_hat earlier; (ii) false-count "
                 "dominance means fewer classes below any threshold",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out_dir, "dwt_cdf_G_overlays.png")
    fig.savefig(path, dpi=150)
    print(f"overlay figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--out_dir", default="output/dwt_histograms")
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "aircraft"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    store = {}
    fig, axes = plt.subplots(len(args.datasets), 2,
                             figsize=(12, 4.2 * len(args.datasets)),
                             squeeze=False)
    for r, ds in enumerate(args.datasets):
        print(f"[{ds}] running ...", flush=True)
        h, arms = run_dataset(ds, args.emb_dir)
        results[ds] = {"homophily_k10": h}
        store[ds] = arms
        lo = min(arms[a]["true"].min() for a in arms) - 0.02
        hi = max(np.percentile(arms[a]["false"], 99.9) for a in arms) + 0.02
        bins = np.linspace(lo, hi, 120)
        for c, arm in enumerate(["raw", "dwt"]):
            st = arms[arm]
            ax = axes[r][c]
            ax.hist(st["false"], bins=bins, density=True, alpha=0.45,
                    color="#c44e52", label="false-class scores s(x,c), c!=y")
            ax.hist(st["true"], bins=bins, density=True, alpha=0.55,
                    color="#55a868", label="true-class scores s(x,y)")
            ax.axvline(st["q_hat"], color="k", ls="--", lw=1.5,
                       label="q_hat (cal 90% quantile)")
            title_arm = ("raw representation" if arm == "raw"
                         else "DWT (qe -> PCA-128 -> cluster-whiten)")
            ax.set_title(f"{ds} — {title_arm}\n"
                         f"cov={st['coverage']:.3f}  avg|C|={st['avg_size']:.2f}  "
                         f"false-leak={st['leakage']:.2f}", fontsize=10)
            ax.set_xlabel("nonconformity score  s = -cos(z, prototype)")
            ax.set_ylabel("density")
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="upper left")
            results[ds][arm] = {k: v for k, v in st.items()
                                if k not in ("true", "false")}
        axes[r][0].annotate(f"pool-kNN homophily h(k=10) ~ {h:.2f}",
                            xy=(0.98, 0.95), xycoords="axes fraction",
                            ha="right", fontsize=9, style="italic")
    fig.suptitle("Same scoring, two representations: DWT shrinks the overlap "
                 "between false-class mass and the true-class quantile "
                 f"(alpha={ALPHA}, cal={CAL_SIZE}, balanced split)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = os.path.join(args.out_dir, "dwt_score_histograms.png")
    fig.savefig(fig_path, dpi=150)
    make_overlay_fig(store, args.out_dir, results)
    with open(os.path.join(args.out_dir, "dwt_score_histograms.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"figure -> {fig_path}")


if __name__ == "__main__":
    main()
