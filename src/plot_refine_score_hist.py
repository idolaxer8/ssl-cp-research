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
from conformal_prediction import (PrototypeSoftmaxNCM,       # noqa: E402
                                  stratified_cal_test_split)
from headline_experiment import build_frozen_transform       # noqa: E402
from r1_headline_experiment import balanced_split            # noqa: E402

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


def resolve_T(rep, X, y, seed):
    """Headline auto-temperature protocol (r1_headline_experiment.
    resolve_softmax_T): balanced pilot split at 8 labels/class, fixed
    seed, auto-T fit on the arm's own representation."""
    allc = np.unique(y)
    rng = np.random.default_rng(seed)
    ci, _ = balanced_split(y, allc, 8, 1, rng)
    pncm = PrototypeSoftmaxNCM(temperature=None, logit="cosine",
                               allow_nonexchangeable=True).fit(
                                   rep(X[ci]), y[ci])
    return float(pncm._T)


def scores_and_qhat_softmax(Z_cal, y_cal, Z_test, y_test, T, alpha=ALPHA):
    """Champion prototype-softmax LAC score s = 1 - p_T(c|z); cal true
    scores via the NCM's exact closed-form LOO, test scored against the
    full-cal prototypes (split-style, as in the cosine variant)."""
    ncm = PrototypeSoftmaxNCM(temperature=T, logit="cosine").fit(
        Z_cal, y_cal)
    cal_true = np.sort(ncm.alpha0)
    n = len(cal_true)
    m_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
    q_hat = float(cal_true[min(m_idx, n - 1)])

    L = (l2n(Z_test) @ ncm.P) / T
    L -= L.max(axis=1, keepdims=True)
    E = np.exp(L)
    s_test = 1.0 - E / E.sum(axis=1, keepdims=True)
    nt = len(y_test)
    test_true = s_test[np.arange(nt), y_test]
    mask_false = np.ones_like(s_test, dtype=bool)
    mask_false[np.arange(nt), y_test] = False
    return dict(
        q_hat=q_hat, T=float(T),
        coverage=float((test_true <= q_hat).mean()),
        avg_size=float((s_test <= q_hat).sum(axis=1).mean()),
        leak=float(((s_test <= q_hat) & mask_false).sum(axis=1).mean()),
        true=test_true, false=s_test[mask_false])


def run_dataset(ds, emb_dir, shots, test_size, seed, score):
    lab = torch.load(os.path.join(emb_dir, f"embeddings_{ds}.pt"),
                     map_location="cpu", weights_only=False)
    unl = torch.load(os.path.join(emb_dir, f"embeddings_{ds}_unlabeled.pt"),
                     map_location="cpu", weights_only=False)
    X = lab["embeddings"].numpy().astype(np.float64)
    y = lab["labels"].numpy().astype(int)
    Xu = unl["embeddings"].numpy().astype(np.float64)
    K = len(np.unique(y))
    # small heldout carves (eurosat: 30/class) cannot fill the default
    # test budget; cap at what a balanced split can actually provide
    per_class = int(np.bincount(y).min())
    test_size = min(test_size, (per_class - shots) * K)
    Xc, yc, Xt, yt = stratified_cal_test_split(
        X, y, cal_size=shots * K, test_size=test_size, random_state=seed)
    tf = build_frozen_transform(ds, Xu)     # the exact frozen Table 2 arm
    arms = {}
    for arm, rep in (("raw", l2n), ("refined", tf.transform)):
        if score == "softmax":
            T = resolve_T(rep, X, y, seed)  # per-arm pilot T, headline rule
            arms[arm] = scores_and_qhat_softmax(rep(Xc), yc, rep(Xt), yt, T)
        else:
            arms[arm] = scores_and_qhat(rep(Xc), yc, rep(Xt), yt)
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--eurosat_dir", default="output/local_embeddings")
    ap.add_argument("--out_dir", default="output/headline/plots")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--test_size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", default="cosine",
                    choices=["cosine", "softmax"],
                    help="cosine = softmax-free family member (bounded, "
                         "geometry-readable); softmax = the champion "
                         "prototype-softmax LAC with per-arm pilot T")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sfx = "" if args.score == "cosine" else "_softmax"

    fig, axes = plt.subplots(2, len(DS), figsize=(5.5, 2.9))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.85, bottom=0.15,
                        hspace=0.14, wspace=0.22)
    stats = {}
    for j, ds in enumerate(DS):
        emb_dir = args.eurosat_dir if ds == "eurosat" else args.emb_dir
        print(f"[{ds}] fitting frozen transform + scoring ...", flush=True)
        arms = run_dataset(ds, emb_dir, args.shots, args.test_size,
                           args.seed, args.score)
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
            note = (f"cov {st['coverage']:.2f}\n"
                    f"$\\langle|C|\\rangle$ {st['avg_size']:.2f}\n"
                    f"leak {st['leak']:.2f}")
            if "T" in st:
                note += f"\nT {st['T']:.3f}"
            ax.text(0.02, 0.97, note, transform=ax.transAxes,
                    fontsize=6.2, va="top", ha="left", linespacing=1.25)
            if i == 0:
                ax.set_title(DS_LABEL[ds], pad=3)
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(r"$s = -\cos(z, m_c)$"
                              if args.score == "cosine"
                              else r"$s = 1 - p_T(c\,|\,z)$", fontsize=7)
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

    stem = os.path.join(args.out_dir, f"fig_refine_score_hist{sfx}")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    score_txt = (
        "Scores are the softmax-free member of the champion family"
        if args.score == "cosine" else
        "Scores are the champion prototype-softmax LAC "
        "$s = 1 - p_T(c\\,|\\,z)$ with the per-representation pilot "
        "temperature of the headline protocol")
    caption = (
        "Why refinement shrinks sets: true-class (green) and false-class "
        "(red) nonconformity-score distributions on the test set, "
        "before (top) and after (bottom) the frozen refinement "
        "T$\\to$W$\\to$S, with the calibration quantile $\\hat q_{1-\\alpha}$ "
        f"(dashed; $\\alpha={ALPHA:g}$, {args.shots} labels/class, single "
        "balanced split, DINOv2 ViT-B). Refinement concentrates the "
        "true-class mass below the quantile and moves false-class mass "
        "above it, so the same coverage is bought with fewer false labels "
        f"in the set (leak = mean false classes inside the set). {score_txt}; "
        "the transform is fit on the unlabeled pool only.")
    with open(stem + "_caption.txt", "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    with open(os.path.join(args.out_dir,
                           f"refine_score_hist_stats{sfx}.json"),
              "w") as f:
        json.dump({"alpha": ALPHA, "shots": args.shots, "seed": args.seed,
                   "score": args.score, "stats": stats}, f, indent=2)
    print(f"saved {stem}.pdf/.png")
    for ds in DS:
        r, p = stats[ds]["raw"], stats[ds]["refined"]
        print(f"  {ds:14s} avg|C| {r['avg_size']:6.2f} -> {p['avg_size']:6.2f}"
              f"   leak {r['leak']:5.2f} -> {p['leak']:5.2f}"
              f"   cov {r['coverage']:.3f} -> {p['coverage']:.3f}")


if __name__ == "__main__":
    main()
