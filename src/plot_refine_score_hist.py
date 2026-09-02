"""Feature refinement and the nonconformity-score distribution.

Paper mechanism figure: for each headline dataset (DINOv2 ViT-B), the
distribution of true-class vs false-class prototype-cosine scores on the
test set, BEFORE refinement (raw L2-normalized embedding) and AFTER the
frozen refinement T$\\to$W$\\to$S (headline_experiment.
build_frozen_transform: the exact frozen arm of Table 2 -- pca128 ->
full-matrix LW cluster whiten -> alpha-QE post smoothing, all fit on the
unlabeled pool). Same training-free score in both spaces (the softmax-free
member of the champion family): s(x,c) = -cos(z, m_c), prototypes = LOO
calibration class means. Refinement pushes the true-class mass left and
peels the false-class mass off the calibration quantile, which is exactly
what shrinks prediction sets at fixed coverage.

Layout: 2 rows (raw / refined) x 4 dataset columns; per-column shared
x-range so the shift is visible. q_hat = split-style calibration quantile
at alpha, marked per panel; annotation = coverage, mean set size and mean
false classes inside the set (leak), all single-split illustrative
numbers (the headline table carries the 50-trial statistics).

Usage (from repo root):
    python src/plot_refine_score_hist.py
"""
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_prediction import stratified_cal_test_split   # noqa: E402
from headline_experiment import build_frozen_transform       # noqa: E402

DS = ["cifar10", "cifar100", "miniimagenet", "eurosat"]
DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "miniimagenet": "miniImageNet", "eurosat": "EuroSAT"}
ALPHA = 0.1
C_FALSE, C_TRUE = "#C44E52", "#2E7D32"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def scores_and_qhat(Z_cal, y_cal, Z_test, y_test, alpha=ALPHA):
    """Prototype-cosine scores + split-style quantile; LOO prototypes on
    cal (the repo's self-leak repair, dwt_score_histograms lineage)."""
    K = int(max(y_cal.max(), y_test.max())) + 1
    protos = np.zeros((K, Z_cal.shape[1]))
    for c in range(K):
        m = y_cal == c
        if m.any():
            protos[c] = Z_cal[m].mean(axis=0)
    protos = l2n(protos)
    s_test = -(l2n(Z_test) @ protos.T)

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
    m_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
    q_hat = np.sort(cal_true)[min(m_idx, n - 1)]

    nt = len(y_test)
    test_true = s_test[np.arange(nt), y_test]
    mask_false = np.ones_like(s_test, dtype=bool)
    mask_false[np.arange(nt), y_test] = False
    return dict(
        q_hat=float(q_hat),
        coverage=float((test_true <= q_hat).mean()),
        avg_size=float((s_test <= q_hat).sum(axis=1).mean()),
        leak=float(((s_test <= q_hat) & mask_false).sum(axis=1).mean()),
        true=test_true, false=s_test[mask_false])


def run_dataset(ds, emb_dir, shots, test_size, seed):
    lab = torch.load(os.path.join(emb_dir, f"embeddings_{ds}.pt"),
                     map_location="cpu", weights_only=False)
    unl = torch.load(os.path.join(emb_dir, f"embeddings_{ds}_unlabeled.pt"),
                     map_location="cpu", weights_only=False)
    X = lab["embeddings"].numpy().astype(np.float64)
    y = lab["labels"].numpy().astype(int)
    Xu = unl["embeddings"].numpy().astype(np.float64)
    K = len(np.unique(y))
    Xc, yc, Xt, yt = stratified_cal_test_split(
        X, y, cal_size=shots * K, test_size=test_size, random_state=seed)
    arms = {"raw": scores_and_qhat(l2n(Xc), yc, l2n(Xt), yt)}
    tf = build_frozen_transform(ds, Xu)     # the exact frozen Table 2 arm
    arms["refined"] = scores_and_qhat(tf.transform(Xc), yc,
                                      tf.transform(Xt), yt)
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--eurosat_dir", default="output/local_embeddings")
    ap.add_argument("--out_dir", default="output/headline/plots")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--test_size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, len(DS), figsize=(5.5, 2.9))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.85, bottom=0.15,
                        hspace=0.14, wspace=0.22)
    stats = {}
    for j, ds in enumerate(DS):
        emb_dir = args.eurosat_dir if ds == "eurosat" else args.emb_dir
        print(f"[{ds}] fitting frozen transform + scoring ...", flush=True)
        arms = run_dataset(ds, emb_dir, args.shots, args.test_size,
                           args.seed)
        lo = min(a["true"].min() for a in arms.values()) - 0.03
        hi = max(np.percentile(a["false"], 99.5)
                 for a in arms.values()) + 0.03
        bins = np.linspace(lo, hi, 80)
        for i, arm in enumerate(("raw", "refined")):
            st = arms[arm]
            ax = axes[i][j]
            ax.hist(st["false"], bins=bins, density=True, alpha=0.45,
                    color=C_FALSE, lw=0,
                    label=r"false class  $s(x,c),\,c\neq y$")
            ax.hist(st["true"], bins=bins, density=True, alpha=0.55,
                    color=C_TRUE, lw=0, label=r"true class  $s(x,y)$")
            ax.axvline(st["q_hat"], color="k", ls="--", lw=1.0,
                       label=r"$\hat{q}_{1-\alpha}$")
            ax.set_xlim(lo, hi)
            ax.set_yticks([])
            ax.text(0.02, 0.97,
                    f"cov {st['coverage']:.2f}\n"
                    f"$\\langle|C|\\rangle$ {st['avg_size']:.2f}\n"
                    f"leak {st['leak']:.2f}",
                    transform=ax.transAxes, fontsize=6.2, va="top",
                    ha="left", linespacing=1.25)
            if i == 0:
                ax.set_title(DS_LABEL[ds], pad=3)
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(r"$s = -\cos(z, m_c)$", fontsize=7)
            stats.setdefault(ds, {})[arm] = {
                k: round(float(v), 4) for k, v in st.items()
                if k not in ("true", "false")}
        axes[0][j].tick_params(axis="x", length=0)
    axes[0][0].set_ylabel("raw", fontsize=8)
    axes[1][0].set_ylabel(r"refined (T$\to$W$\to$S)", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.5,
               handlelength=1.4, handletextpad=0.5)

    stem = os.path.join(args.out_dir, "fig_refine_score_hist")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    caption = (
        "Why refinement shrinks sets: true-class (green) and false-class "
        "(red) prototype-cosine score distributions on the test set, "
        "before (top) and after (bottom) the frozen refinement "
        "T$\\to$W$\\to$S, with the calibration quantile $\\hat q_{1-\\alpha}$ "
        f"(dashed; $\\alpha={ALPHA:g}$, {args.shots} labels/class, single "
        "balanced split, DINOv2 ViT-B). Refinement concentrates the "
        "true-class mass below the quantile and moves false-class mass "
        "above it, so the same coverage is bought with fewer false labels "
        "in the set (leak = mean false classes inside the set). Scores are "
        "the softmax-free member of the champion family; the transform is "
        "fit on the unlabeled pool only.")
    with open(stem + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    with open(os.path.join(args.out_dir, "refine_score_hist_stats.json"),
              "w") as f:
        json.dump({"alpha": ALPHA, "shots": args.shots,
                   "seed": args.seed, "stats": stats}, f, indent=2)
    print(f"saved {stem}.pdf/.png")
    for ds in DS:
        r, p = stats[ds]["raw"], stats[ds]["refined"]
        print(f"  {ds:14s} avg|C| {r['avg_size']:6.2f} -> {p['avg_size']:6.2f}"
              f"   leak {r['leak']:5.2f} -> {p['leak']:5.2f}"
              f"   cov {r['coverage']:.3f} -> {p['coverage']:.3f}")


if __name__ == "__main__":
    main()
