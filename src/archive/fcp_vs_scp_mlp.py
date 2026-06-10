"""
FCP+PCA vs SCP with MLP head -- fair comparison.

Methods (all share the same labeled budget per trial):
- FCP:      Full CP, entire budget as calibration (geodesic_topk_mean)
- FCP+PCA:  FCP on PCA-projected embeddings (PCA fit on unlabeled pool)
- SCP-LR:   Split CP with logistic regression (budget/2 train, budget/2 cal, THR score)
- SCP-MLP:  Split CP with 2-layer MLP         (budget/2 train, budget/2 cal, THR score)

Fairness:
- SCP methods split the labeled budget 50/50 (train / cal), stratified
- FCP methods use the full budget as calibration (no train split needed)
- Same stratified labeled subsample, same test set across methods
- PCA uses only the unlabeled pool (exchangeability-safe)

Usage:
    python src/fcp_vs_scp_mlp.py
    python src/fcp_vs_scp_mlp.py --cal_sizes 400 600 800 --n_trials 3
    python src/fcp_vs_scp_mlp.py --mlp_hidden 512 256 --mlp_alpha 1e-3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conformal_prediction import (
    FullConformalPredictor, create_ncm, compute_cp_scores, compute_cp_sets,
)

# ── defaults ──────────────────────────────────────────────────────────────────
ALPHA = 0.1
SEED = 42
N_TRIALS = 5
CAL_SIZES = [300, 400, 600, 800, 1000]
TEST_SIZE = 500
TRAIN_RATIO = 0.5          # fraction of labeled budget used to train the SCP head
PCA_DIM = 128
NCM = "geodesic_topk_mean"

EMB_PATH = "output/embeddings_cifar100.pt"
TEST_PATH = "output/embeddings_cifar100_test.pt"
UNLABELED_PATH = "output/embeddings_cifar100_unlabeled.pt"
OUT_DIR = Path("output/fcp_vs_scp_mlp")


# ── helpers ───────────────────────────────────────────────────────────────────
def stratified_subsample(X, y, all_classes, n_total, rng):
    """Stratified subsample: >=1 per class, rest uniformly at random."""
    n_classes = len(all_classes)
    first = np.array([rng.choice(np.where(y == c)[0], 1, replace=False)[0]
                       for c in all_classes])
    rest = np.setdiff1d(np.arange(len(X)), first)
    n_extra = n_total - n_classes
    if n_extra > 0:
        extra = rng.choice(rest, min(n_extra, len(rest)), replace=False)
        idx = np.concatenate([first, extra])
    else:
        idx = first[:n_total]
    idx = rng.permutation(idx)
    return X[idx], y[idx]


def stratified_train_cal_split(X, y, all_classes, train_ratio, rng):
    """Split into train / cal, ensuring >=1 per class in train.

    Priority: every class must appear in train (so the classifier sees it).
    Classes with only 1 sample go to train only (no cal example).
    """
    train_idx, cal_idx = [], []
    for c in all_classes:
        c_mask = np.where(y == c)[0]
        c_mask = rng.permutation(c_mask)
        n_c = len(c_mask)
        if n_c <= 1:
            # singleton: must go to train so classifier sees this class
            train_idx.extend(c_mask)
            continue
        n_tr = max(1, round(train_ratio * n_c))
        # ensure at least 1 in cal
        if n_tr >= n_c:
            n_tr = n_c - 1
        train_idx.extend(c_mask[:n_tr])
        cal_idx.extend(c_mask[n_tr:])
    train_idx = np.array(train_idx)
    cal_idx = np.array(cal_idx)
    return X[train_idx], y[train_idx], X[cal_idx], y[cal_idx]


def run_scp(clf, X_train, y_train, X_cal, y_cal, X_test, y_test, alpha):
    """Train classifier, calibrate with THR score, evaluate."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_te_s = scaler.transform(X_test)

    clf.fit(X_tr_s, y_train)
    acc = clf.score(X_te_s, y_test)

    probs_cal = clf.predict_proba(X_cal_s)
    probs_test = clf.predict_proba(X_te_s)

    # map true labels -> column indices (default 0 for unseen classes)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
    y_cal_idx = np.array([class_to_idx.get(y, 0) for y in y_cal])
    y_test_idx = np.array([class_to_idx.get(y, -1) for y in y_test])

    # calibrate (THR)
    cal_scores = compute_cp_scores(probs_cal, y_cal_idx, "THR")
    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(cal_scores, q_level)

    # predict
    pred_sets = compute_cp_sets(probs_test, q_hat, "THR")

    # evaluate
    covered = sum(1 for i, ps in enumerate(pred_sets) if y_test_idx[i] in ps)
    sizes = [len(ps) for ps in pred_sets]
    return {
        "coverage": covered / len(y_test),
        "avg_set_size": float(np.mean(sizes)),
        "classifier_acc": acc,
        "n_train": len(y_train),
        "n_cal": len(y_cal),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_sizes", type=int, nargs="+", default=CAL_SIZES)
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--pca_dim", type=int, default=PCA_DIM)
    parser.add_argument("--test_size", type=int, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mlp_hidden", type=int, nargs="+", default=[256])
    parser.add_argument("--mlp_alpha", type=float, default=0.1,
                        help="L2 regularisation for MLP (sklearn alpha)")
    parser.add_argument("--mlp_max_iter", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────
    data = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
    X_labeled = data["embeddings"].numpy()
    y_labeled = data["labels"].numpy()
    all_classes = np.unique(y_labeled)
    n_classes = len(all_classes)

    tdata = torch.load(TEST_PATH, map_location="cpu", weights_only=False)
    X_test_full = tdata["embeddings"].numpy()
    y_test_full = tdata["labels"].numpy()

    udata = torch.load(UNLABELED_PATH, map_location="cpu", weights_only=False)
    X_unlabeled = udata["embeddings"].numpy()

    # stratified test subsample (fixed across all trials)
    rng_test = np.random.default_rng(args.seed)
    test_per_class = args.test_size // n_classes
    test_idx = []
    for c in all_classes:
        c_idx = np.where(y_test_full == c)[0]
        chosen = rng_test.choice(c_idx, min(test_per_class, len(c_idx)), replace=False)
        test_idx.append(chosen)
    test_idx = np.concatenate(test_idx)
    X_test = X_test_full[test_idx]
    y_test = y_test_full[test_idx]

    print(f"Labeled: {X_labeled.shape}, Unlabeled: {X_unlabeled.shape}, "
          f"Test: {X_test.shape}, Classes: {n_classes}")
    print(f"Train ratio: {args.train_ratio}, PCA dim: {args.pca_dim}, NCM: {NCM}")
    print(f"MLP hidden: {tuple(args.mlp_hidden)}, alpha={args.mlp_alpha}")
    print(f"Cal sizes: {args.cal_sizes}, Trials: {args.n_trials}\n")

    # ── fit PCA on unlabeled (once) ───────────────────────────────────────
    pca = PCA(n_components=args.pca_dim)
    pca.fit(X_unlabeled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {X_unlabeled.shape[1]}d -> {args.pca_dim}d  "
          f"(explained variance: {explained:.3f})")
    X_test_pca = pca.transform(X_test)

    # ── run experiments ───────────────────────────────────────────────────
    methods = ["FCP", "FCP+PCA", "SCP-LR", "SCP-MLP"]
    results = {cs: {m: [] for m in methods} for cs in args.cal_sizes}

    for cs in args.cal_sizes:
        t_cs = time.time()
        for trial in range(args.n_trials):
            rng = np.random.default_rng(args.seed + trial * 1000)

            # same labeled subsample for all methods
            X_cal_all, y_cal_all = stratified_subsample(
                X_labeled, y_labeled, all_classes, cs, rng)

            # ── FCP: full budget ──────────────────────────────────────────
            t0 = time.time()
            ncm = create_ncm(NCM, k=5, allow_nonexchangeable=True)
            cp = FullConformalPredictor(ncm, alpha=ALPHA)
            cp.calibrate(X_cal_all, y_cal_all, all_classes=all_classes)
            m = cp.evaluate(X_test, y_test, verbose=False)
            results[cs]["FCP"].append({
                "coverage": m["coverage"],
                "avg_set_size": m["avg_set_size"],
                "time": time.time() - t0,
            })

            # ── FCP+PCA: full budget, projected ───────────────────────────
            t0 = time.time()
            X_cal_pca = pca.transform(X_cal_all)
            ncm_pca = create_ncm(NCM, k=5, allow_nonexchangeable=True)
            cp_pca = FullConformalPredictor(ncm_pca, alpha=ALPHA)
            cp_pca.calibrate(X_cal_pca, y_cal_all, all_classes=all_classes)
            m = cp_pca.evaluate(X_test_pca, y_test, verbose=False)
            results[cs]["FCP+PCA"].append({
                "coverage": m["coverage"],
                "avg_set_size": m["avg_set_size"],
                "time": time.time() - t0,
            })

            # ── stratified train / cal split for SCP methods ──────────────
            rng_split = np.random.default_rng(args.seed + trial * 1000 + 1)
            X_train, y_train, X_cal_scp, y_cal_scp = stratified_train_cal_split(
                X_cal_all, y_cal_all, all_classes, args.train_ratio, rng_split)

            # ── SCP-LR ────────────────────────────────────────────────────
            t0 = time.time()
            lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
            m = run_scp(lr, X_train, y_train, X_cal_scp, y_cal_scp,
                        X_test, y_test, ALPHA)
            m["time"] = time.time() - t0
            results[cs]["SCP-LR"].append(m)

            # ── SCP-MLP ───────────────────────────────────────────────────
            t0 = time.time()
            mlp = MLPClassifier(
                hidden_layer_sizes=tuple(args.mlp_hidden),
                alpha=args.mlp_alpha,
                max_iter=args.mlp_max_iter,
                solver="adam",
                learning_rate="adaptive",
                random_state=42,
            )
            m = run_scp(mlp, X_train, y_train, X_cal_scp, y_cal_scp,
                        X_test, y_test, ALPHA)
            m["time"] = time.time() - t0
            results[cs]["SCP-MLP"].append(m)

        elapsed = time.time() - t_cs
        print(f"cal={cs} ({elapsed:.1f}s, {args.n_trials} trials)")
        for m_name in methods:
            vals = results[cs][m_name]
            covs = [v["coverage"] for v in vals]
            szs = [v["avg_set_size"] for v in vals]
            flag = "OK" if np.mean(covs) >= 0.89 else "!!"
            extra = ""
            if "SCP" in m_name:
                accs = [v["classifier_acc"] for v in vals]
                n_tr = vals[0]["n_train"]
                n_ca = vals[0]["n_cal"]
                extra = f"  acc={np.mean(accs):.3f}  (train={n_tr}, cal={n_ca})"
            print(f"  {m_name:10s}  cov={np.mean(covs):.3f}+/-{np.std(covs):.3f} {flag}  "
                  f"sz={np.mean(szs):.2f}+/-{np.std(szs):.2f}{extra}")
        print()

    # ── save JSON ─────────────────────────────────────────────────────────
    def jsonify(o):
        if isinstance(o, dict):
            return {str(k): jsonify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonify(i) for i in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(out_dir / "results.json", "w") as f:
        json.dump(jsonify(results), f, indent=2)

    # ── plot ──────────────────────────────────────────────────────────────
    styles = {
        "FCP":      {"color": "#1f77b4", "ls": "-",  "marker": "o", "lw": 2.5},
        "FCP+PCA":  {"color": "#e377c2", "ls": "-",  "marker": "p", "lw": 2.5},
        "SCP-LR":   {"color": "#2ca02c", "ls": ":",  "marker": "^", "lw": 2.0},
        "SCP-MLP":  {"color": "#ff7f0e", "ls": "--", "marker": "D", "lw": 2.0},
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    mlp_desc = "x".join(str(h) for h in args.mlp_hidden)
    fig.suptitle(
        f"CIFAR-100: FCP+PCA-{args.pca_dim} vs SCP with MLP-({mlp_desc}) head\n"
        f"(alpha={ALPHA}, train_ratio={args.train_ratio})",
        fontsize=13, fontweight="bold")

    for m_name in methods:
        s = styles[m_name]
        xs, mu_sz, sd_sz, mu_cov, sd_cov, mu_acc, sd_acc = (
            [], [], [], [], [], [], [])

        for cs in args.cal_sizes:
            vals = results[cs][m_name]
            covs = [v["coverage"] for v in vals]
            szs = [v["avg_set_size"] for v in vals]
            xs.append(cs)
            mu_cov.append(np.mean(covs)); sd_cov.append(np.std(covs))
            mu_sz.append(np.mean(szs)); sd_sz.append(np.std(szs))
            if "SCP" in m_name:
                accs = [v["classifier_acc"] for v in vals]
                mu_acc.append(np.mean(accs)); sd_acc.append(np.std(accs))

        xs = np.array(xs)

        if m_name == "FCP":
            label = f"FCP ({NCM}, n_cal=budget)"
        elif m_name == "FCP+PCA":
            label = f"FCP+PCA-{args.pca_dim} (n_cal=budget)"
        else:
            head = "LogReg" if "LR" in m_name else f"MLP({mlp_desc})"
            label = f"SCP-THR {head} (train=cal=budget/2)"

        axes[0].plot(xs, mu_sz, color=s["color"], ls=s["ls"], marker=s["marker"],
                     lw=s["lw"], ms=6, label=label)
        axes[0].fill_between(xs, np.array(mu_sz) - np.array(sd_sz),
                             np.array(mu_sz) + np.array(sd_sz),
                             alpha=0.12, color=s["color"])

        axes[1].plot(xs, np.array(mu_cov) * 100, color=s["color"], ls=s["ls"],
                     marker=s["marker"], lw=s["lw"], ms=6, label=label)
        axes[1].fill_between(xs, (np.array(mu_cov) - np.array(sd_cov)) * 100,
                             (np.array(mu_cov) + np.array(sd_cov)) * 100,
                             alpha=0.12, color=s["color"])

        if "SCP" in m_name:
            axes[2].plot(xs, np.array(mu_acc) * 100, color=s["color"], ls=s["ls"],
                         marker=s["marker"], lw=s["lw"], ms=6, label=label)
            axes[2].fill_between(xs, (np.array(mu_acc) - np.array(sd_acc)) * 100,
                                 (np.array(mu_acc) + np.array(sd_acc)) * 100,
                                 alpha=0.12, color=s["color"])

    axes[1].axhline(90, color="black", ls="--", lw=1.2, label="Target 90%")

    axes[0].set_ylabel("Avg Prediction Set Size", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Coverage (%)", fontsize=11)
    axes[1].set_ylim(70, 102)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Classifier Accuracy (%)", fontsize=11)
    axes[2].set_xlabel("Labeled Budget", fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_dir / "fcp_vs_scp_mlp.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── text summary ──────────────────────────────────────────────────────
    lines = [
        f"CIFAR-100: FCP+PCA-{args.pca_dim} vs SCP-MLP({mlp_desc})"
        f"  (alpha={ALPHA}, train_ratio={args.train_ratio}, {args.n_trials} trials)",
        f"MLP: hidden={tuple(args.mlp_hidden)}, alpha={args.mlp_alpha}, max_iter={args.mlp_max_iter}",
        f"Unlabeled pool: {X_unlabeled.shape[0]}, PCA explained var: {explained:.3f}",
        "=" * 90,
        f"{'Budget':>6}  {'Method':>10}  {'Coverage':>18}  {'Set Size':>18}  {'Clf Acc':>10}",
        "-" * 90,
    ]
    for cs in args.cal_sizes:
        first = True
        for m_name in methods:
            vals = results[cs][m_name]
            covs = [v["coverage"] for v in vals]
            szs = [v["avg_set_size"] for v in vals]
            prefix = f"{cs:>6}" if first else " " * 6
            acc_str = "   n/a"
            if "SCP" in m_name:
                accs = [v["classifier_acc"] for v in vals]
                acc_str = f"{np.mean(accs)*100:5.1f}%"
            lines.append(
                f"{prefix}  {m_name:>10}  "
                f"{np.mean(covs)*100:5.1f}%+/-{np.std(covs)*100:4.1f}%  "
                f"{np.mean(szs):7.2f}+/-{np.std(szs):5.2f}  "
                f"{acc_str}")
            first = False
        lines.append("-" * 90)

    txt = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(txt)

    print(f"\nSaved: {out_dir / 'results.json'}")
    print(f"Saved: {out_dir / 'fcp_vs_scp_mlp.png'}")
    print(f"Saved: {out_dir / 'summary.txt'}")
    print("\n" + txt)


if __name__ == "__main__":
    main()
