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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--out_dir", default="output/dwt_histograms")
    ap.add_argument("--datasets", nargs="+", default=["cifar100", "aircraft"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    fig, axes = plt.subplots(len(args.datasets), 2,
                             figsize=(12, 4.2 * len(args.datasets)),
                             squeeze=False)
    for r, ds in enumerate(args.datasets):
        print(f"[{ds}] running ...", flush=True)
        h, arms = run_dataset(ds, args.emb_dir)
        results[ds] = {"homophily_k10": h}
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
    with open(os.path.join(args.out_dir, "dwt_score_histograms.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"figure -> {fig_path}")


if __name__ == "__main__":
    main()
