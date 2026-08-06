"""
G3 semi-supervised baseline experiment (docs/g3_semisup_baseline_plan.md).

Answers the reviewer question "with 200-800 labels and a 3-10k unlabeled
pool, why not semi-supervised learning + CP?" with two arms that bracket
the design space, plus in-run raw / champion FCP arms so every comparison
shares the same trials (same seed protocol as transform_control_experiment).

Arm A  --arms selftrain   Label-dependent self-training probe + split CP
    (plan sec 2): linear probe on a train fraction of the labeled budget
    (lam = 1e-2 fixed, mean-CE parameterization), R = 2 curriculum
    self-training rounds with class-balanced top-q pseudo-label selection
    (q = 0.2, 0.4; Cascante-Bonilla AAAI 2021), THR split CP on the held
    out calibration fraction (Sadinle et al. 2019). Swept over train/cal
    ratios {0.25, 0.5, 0.75} and reported per ratio AND per round (r = 0 is
    the plain SCP-THR refresh). Valid split CP by construction.

Arm B  --arms poolmlp     Label-free pool-only learned head (plan sec 3):
    k-means pseudo-labels on the pool (same C as the adopted pipeline),
    2-layer MLP (768 -> 1024 -> d', d' MATCHED to the pipeline's per
    dataset knob) trained with CE + label smoothing on cluster labels
    (DeepCluster lineage, Caron et al. ECCV 2018), penultimate output
    L2-normed as the representation, FCP on top. fit(pool)/transform(X)
    are fixed functions of the pool alone -> exactly exchangeable (Prop 2,
    docs/theory.md sec 2), same status as every UnlabeledTransform arm.
    Variants: poolmlp_raw (raw inputs) and poolmlp_qe (qe-denoised inputs,
    stage-1 composition). --mlp_seeds heads per variant; trial t uses head
    t % n_seeds, per-seed means recorded separately.

Usage (local, from repo root; data_dir points at the MAIN repo's
embeddings -- read-only):
python src/g3_semisup_experiment.py --dataset cifar100 \
    --data_dir <main-repo>/output/from_cluster/embeddings --device cuda
"""
import sys, os, time, json, argparse, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from conformal_prediction import FullConformalPredictor, create_ncm
from exchangeable_features import UnlabeledTransform, IdentityTransform
from split_cp_baselines import compute_cp_scores, compute_cp_sets
from transform_control_experiment import (balanced_split, random_split,
                                          resolve_softmax_T)

SOFTMAX_NCMS = {"prototype_softmax"}

# Adopted-pipeline champion knobs per dataset (plan doc sec 5): d', cluster
# count C, and qe mode (classic = defaults k=10/a=3/beta=None; safe =
# k=5/a=3/beta=0.3 -- the sec-6 gate rule, frozen here, not tuned in-run).
CHAMPION = {
    "cifar100":     dict(pca_dim=192, n_clusters=100, qe=dict()),
    "miniimagenet": dict(pca_dim=128, n_clusters=100, qe=dict()),
    "cub200":       dict(pca_dim=512, n_clusters=300,
                         qe=dict(qe_k=5, qe_beta=0.3)),
    "aircraft":     dict(pca_dim=512, n_clusters=100,
                         qe=dict(qe_k=5, qe_beta=0.3)),
}
# d' for the pool-MLP head = the champion's d' (controlled comparison:
# representation dimension is a known regime knob, so it must match).
MLP_DPRIME = {k: v["pca_dim"] for k, v in CHAMPION.items()}

RATIOS = (0.25, 0.5, 0.75)      # train fraction of the labeled budget
Q_SCHEDULE = (0.2, 0.4)         # curriculum top-q per class, rounds 1..R
LAM = 1e-2                      # probe ridge (mean-CE parameterization)


# ---------------------------------------------------------------------------
# Arm B: pool-only DeepCluster-style MLP head
# ---------------------------------------------------------------------------
class PoolMLPTransform:
    """fit(pool) / transform(X) MLP representation trained on pool k-means
    pseudo-labels. Optional input denoiser (an UnlabeledTransform with
    pre='qe', projection='center') is itself pool-fit, so the composition
    stays a fixed function of the pool (Prop-2 exact)."""

    def __init__(self, out_dim, n_clusters, hidden=1024, epochs=100,
                 batch_size=512, lr=1e-3, weight_decay=1e-4,
                 label_smoothing=0.1, seed=0, device="cuda", denoiser=None):
        self.out_dim = out_dim
        self.n_clusters = n_clusters
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.device = device
        self.denoiser = denoiser
        self.net_ = None
        self.train_acc_ = None

    def _inputs(self, X):
        if self.denoiser is not None:
            return np.asarray(self.denoiser.transform(X), dtype=np.float64)
        Xn = np.asarray(X, dtype=np.float64)
        return Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)

    def fit(self, Xu):
        if self.denoiser is not None:
            self.denoiser.fit(Xu)
        Z = self._inputs(Xu)
        km = KMeans(n_clusters=self.n_clusters, n_init=3,
                    random_state=self.seed).fit(Z)
        labels = torch.as_tensor(km.labels_, dtype=torch.long)

        torch.manual_seed(self.seed)
        dev = self.device if torch.cuda.is_available() else "cpu"
        net = nn.Sequential(
            nn.Linear(Z.shape[1], self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.out_dim),
        )
        head = nn.Linear(self.out_dim, self.n_clusters)
        model = nn.Sequential(net, head).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        Zt = torch.as_tensor(Z, dtype=torch.float32)
        n = len(Zt)
        steps = self.epochs * math.ceil(n / self.batch_size)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        lossf = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        g = torch.Generator().manual_seed(self.seed)
        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i0 in range(0, n, self.batch_size):
                idx = perm[i0:i0 + self.batch_size]
                xb = Zt[idx].to(dev)
                yb = labels[idx].to(dev)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i0 in range(0, n, 4096):
                preds.append(model(Zt[i0:i0 + 4096].to(dev)).argmax(1).cpu())
            self.train_acc_ = float((torch.cat(preds) == labels)
                                    .float().mean())
        self.net_ = net.eval()
        self._dev = dev
        return self

    def transform(self, X):
        Z = self._inputs(X)
        Zt = torch.as_tensor(Z, dtype=torch.float32)
        outs = []
        with torch.no_grad():
            for i0 in range(0, len(Zt), 4096):
                outs.append(self.net_(Zt[i0:i0 + 4096].to(self._dev)).cpu())
        H = torch.cat(outs).numpy().astype(np.float64)
        return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)

    def __repr__(self):
        d = "qe->" if self.denoiser is not None else ""
        return (f"PoolMLP({d}768->{self.hidden}->{self.out_dim}, "
                f"C={self.n_clusters}, seed={self.seed}, "
                f"acc={self.train_acc_ if self.train_acc_ is None else round(self.train_acc_, 3)})")


# ---------------------------------------------------------------------------
# Arm A: self-training probe + THR split CP
# ---------------------------------------------------------------------------
def fit_probe(Z, y, lam=LAM):
    """Multinomial logistic probe; (1/n) sum CE + lam ||W||_F^2 mapped onto
    sklearn's  0.5 ||w||^2 + C_skl * sum CE  via  C_skl = 1 / (2 lam n)."""
    if len(np.unique(y)) < 2:
        return None
    C_skl = 1.0 / (2.0 * lam * len(y))
    clf = LogisticRegression(C=C_skl, solver="lbfgs", max_iter=1000)
    clf.fit(Z, y)
    return clf


def full_probs(clf, Z, allc):
    """Probability matrix over ALL K classes (zero columns for classes the
    probe never saw -- symmetric handling; missing classes then carry THR
    score 1 for their cal points, which is what drives the documented
    collapse rather than silently dropping them)."""
    P = np.zeros((len(Z), len(allc)))
    if clf is None:
        return P
    col = {int(c): j for j, c in enumerate(allc)}
    P[:, [col[int(c)] for c in clf.classes_]] = clf.predict_proba(Z)
    return P


def select_pseudo(P_pool, q, rng):
    """Class-balanced top-q selection: per predicted class, the top-q
    fraction of its candidates by confidence (Curriculum Labeling)."""
    yhat = P_pool.argmax(1)
    conf = P_pool.max(1)
    sel = []
    for j in range(P_pool.shape[1]):
        cand = np.where(yhat == j)[0]
        if len(cand) == 0:
            continue
        n_take = max(1, int(math.ceil(q * len(cand))))
        sel.append(cand[np.argsort(-conf[cand])[:n_take]])
    if not sel:
        return np.array([], dtype=int), np.array([], dtype=int)
    sel = np.concatenate(sel)
    return sel, yhat[sel]


def thr_qhat(cal_scores, alpha):
    n = len(cal_scores)
    lvl = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(cal_scores, lvl, method="higher"))


def run_selftrain(X, y, Xu, allc, args):
    """Arm A over cal sizes x ratios x rounds; returns result rows."""
    K = len(allc)
    Zl = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Zu = Xu / (np.linalg.norm(Xu, axis=1, keepdims=True) + 1e-12)
    col = {int(c): j for j, c in enumerate(allc)}
    rows = []
    for cal in args.cal_sizes:
        m_cal = cal // K
        if m_cal < 2:
            continue
        # metric accumulators: (ratio, round) -> per-trial lists
        acc = {(rt, r): {"cov": [], "sz": [], "n_missing": []}
               for rt in RATIOS for r in range(len(Q_SCHEDULE) + 1)}
        t0 = time.time()
        for t in range(args.n_trials):
            rng = np.random.default_rng(args.seed + 1000 * t)
            ci, ti = balanced_split(y, allc, m_cal, args.test_per_class, rng)
            yt = y[ti]
            yt_idx = np.array([col[int(c)] for c in yt])
            for rt in RATIOS:
                # fresh rng per ratio so ratios don't entangle draw order
                rng2 = np.random.default_rng(args.seed + 1000 * t
                                             + int(rt * 100) * 17)
                perm = rng2.permutation(ci)
                n_tr = int(round(rt * len(ci)))
                tr, ca = perm[:n_tr], perm[n_tr:]
                yca_idx = np.array([col[int(c)] for c in y[ca]])
                # round 0 = plain probe (the SCP-THR refresh)
                clf = fit_probe(Zl[tr], y[tr])
                for r in range(len(Q_SCHEDULE) + 1):
                    if r > 0:
                        P_pool = full_probs(clf, Zu, allc)
                        sel, ysel_idx = select_pseudo(P_pool, Q_SCHEDULE[r - 1],
                                                      rng2)
                        if len(sel):
                            Zaug = np.vstack([Zl[tr], Zu[sel]])
                            yaug = np.concatenate(
                                [y[tr], allc[ysel_idx]])
                            clf = fit_probe(Zaug, yaug)
                    n_missing = K - (0 if clf is None
                                     else len(clf.classes_))
                    P_ca = full_probs(clf, Zl[ca], allc)
                    P_ti = full_probs(clf, Zl[ti], allc)
                    q_hat = thr_qhat(
                        compute_cp_scores(P_ca, yca_idx, "THR"), args.alpha)
                    sets = compute_cp_sets(P_ti, q_hat, "THR")
                    covered = np.array([yt_idx[i] in sets[i]
                                        for i in range(len(ti))])
                    a = acc[(rt, r)]
                    a["cov"].append(float(covered.mean()))
                    a["sz"].append(float(np.mean([len(s) for s in sets])))
                    a["n_missing"].append(int(n_missing))
        for (rt, r), a in acc.items():
            szs = np.array(a["sz"])
            rows.append({
                "arm": "selftrain", "cal": cal, "ratio": rt, "round": r,
                "cov": float(np.mean(a["cov"])),
                "cov_sd": float(np.std(a["cov"])),
                "sz": float(szs.mean()), "sz_sd": float(szs.std()),
                "sz_se": float(szs.std() / np.sqrt(len(szs))),
                "n_missing_train": float(np.mean(a["n_missing"])),
                "n_trials": args.n_trials})
            print(f"  [selftrain] cal={cal:4d} ratio={rt:.2f} r={r} "
                  f"cov={rows[-1]['cov']:.4f} sz={rows[-1]['sz']:6.2f}"
                  f"+-{rows[-1]['sz_se']:.2f} "
                  f"missing_cls={rows[-1]['n_missing_train']:.1f}")
        print(f"  cal={cal} done in {time.time()-t0:.0f}s")
    return rows


# ---------------------------------------------------------------------------
# FCP arms (raw / champion / poolmlp_raw / poolmlp_qe) -- reference-driver
# protocol (same seeds -> same splits as the menu runs)
# ---------------------------------------------------------------------------
def build_fcp_arm(arm, ds, Xu, args):
    if arm == "raw":
        return [IdentityTransform().fit()]
    if arm == "champion":
        c = CHAMPION[ds]
        tf = UnlabeledTransform(pre="qe", pca_dim=c["pca_dim"], whiten=None,
                                projection="ldapool",
                                n_clusters=c["n_clusters"], **c["qe"])
        return [tf.fit(Xu)]
    if arm in ("poolmlp_raw", "poolmlp_qe"):
        tfs = []
        for s in range(args.mlp_seeds):
            den = None
            if arm == "poolmlp_qe":
                c = CHAMPION[ds]
                den = UnlabeledTransform(pre="qe", pca_dim=None, whiten=None,
                                         projection="center", **c["qe"])
            tfs.append(PoolMLPTransform(
                out_dim=MLP_DPRIME[ds],
                n_clusters=CHAMPION[ds]["n_clusters"],
                epochs=args.mlp_epochs, seed=args.seed + 101 * s,
                device=args.device, denoiser=den).fit(Xu))
        return tfs
    raise ValueError(arm)


def run_fcp_arms(X, y, Xu, allc, ds, args):
    K = len(allc)
    cls_to_j = {int(c): j for j, c in enumerate(allc)}
    rows = []
    for arm in args.fcp_arms:
        t0 = time.time()
        tfs = build_fcp_arm(arm, ds, Xu, args)
        print(f"\n=== fcp arm {arm}: {tfs[0]}"
              + (f" (x{len(tfs)} seeds)" if len(tfs) > 1 else "")
              + f" [fit {time.time()-t0:.0f}s] ===")
        T_by_ncm = {}
        for nm in args.ncms:
            if nm in SOFTMAX_NCMS:
                T_by_ncm[nm] = resolve_softmax_T(tfs[0], X, y, allc, args)
                print(f"  {nm} fixed T = {T_by_ncm[nm]:.4f}")
        for split in args.splits:
            for cal in args.cal_sizes:
                m_cal = cal // K
                if split == "balanced_both" and m_cal < 2:
                    continue
                for nm in args.ncms:
                    if (split == "random" and nm in SOFTMAX_NCMS
                            and cal < args.random_min_cal_softmax):
                        continue
                    covs, szs, seed_of_trial = [], [], []
                    pooled_cov, pooled_tot = np.zeros(K), np.zeros(K)
                    n_fallback = 0
                    t0 = time.time()
                    for t in range(args.n_trials):
                        tf = tfs[t % len(tfs)]
                        rng = np.random.default_rng(args.seed + 1000 * t)
                        if split == "balanced_both":
                            ci, ti = balanced_split(y, allc, m_cal,
                                                    args.test_per_class, rng)
                        else:
                            ci, ti = random_split(y, cal,
                                                  args.test_per_class * K, rng)
                        Xc, yc = tf.transform(X[ci]), y[ci]
                        Xt, yt = tf.transform(X[ti]), y[ti]
                        if nm in SOFTMAX_NCMS:
                            ncm = create_ncm(nm, temperature=T_by_ncm[nm],
                                             logit="cosine")
                        else:
                            ncm = create_ncm(nm, k=5)
                        cp = FullConformalPredictor(ncm, alpha=args.alpha)
                        cp.calibrate(Xc, yc, all_classes=allc)
                        try:
                            res = cp.predict(Xt, verbose=False,
                                             device=args.device)
                        except (RuntimeError, ValueError):
                            res = cp.predict(Xt, verbose=False, device="cpu")
                            n_fallback += 1
                        psets = res["prediction_sets"]
                        covered = np.array([yt[i] in psets[i]
                                            for i in range(len(yt))])
                        covs.append(float(covered.mean()))
                        szs.append(float(np.mean([len(s) for s in psets])))
                        seed_of_trial.append(t % len(tfs))
                        for c in allc:
                            msk = yt == c
                            if msk.any():
                                j = cls_to_j[int(c)]
                                pooled_cov[j] += float(covered[msk].sum())
                                pooled_tot[j] += int(msk.sum())
                    valid = pooled_tot > 0
                    pcov = pooled_cov[valid] / pooled_tot[valid]
                    szs_a = np.array(szs)
                    row = {"arm": arm, "split": split, "ncm": nm, "cal": cal,
                           "cov": float(np.mean(covs)),
                           "cov_sd": float(np.std(covs)),
                           "sz": float(szs_a.mean()),
                           "sz_sd": float(szs_a.std()),
                           "sz_se": float(szs_a.std() / np.sqrt(len(szs_a))),
                           "covgap": float(100 * np.mean(
                               np.abs(pcov - (1 - args.alpha)))),
                           "n_trials": args.n_trials,
                           "n_cpu_fallback": n_fallback}
                    if len(tfs) > 1:
                        so = np.array(seed_of_trial)
                        row["sz_by_seed"] = [
                            float(szs_a[so == s].mean())
                            for s in range(len(tfs)) if (so == s).any()]
                    if arm.startswith("poolmlp"):
                        row["mlp_train_acc"] = [tf.train_acc_ for tf in tfs]
                    rows.append(row)
                    print(f"  [{split:13s}] cal={cal:4d} {nm:22s} "
                          f"cov={row['cov']:.4f} sz={row['sz']:6.2f}"
                          f"+-{row['sz_se']:.2f} "
                          f"covgap={row['covgap']:5.2f}pp "
                          f"({time.time()-t0:.0f}s)")
        if args.device == "cuda":
            torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description="G3 semi-supervised baselines")
    ap.add_argument("--data_dir", default="output/from_cluster/embeddings")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--lab_path", default=None,
                    help="override labeled .pt path (cub200 layout)")
    ap.add_argument("--unl_path", default=None)
    ap.add_argument("--arms", nargs="+", default=["selftrain", "fcp"],
                    choices=["selftrain", "fcp"])
    ap.add_argument("--fcp_arms", nargs="+",
                    default=["raw", "champion", "poolmlp_raw", "poolmlp_qe"],
                    choices=["raw", "champion", "poolmlp_raw", "poolmlp_qe"])
    ap.add_argument("--ncms", nargs="+",
                    default=["unwhitened_topk_mean", "prototype_softmax"])
    ap.add_argument("--splits", nargs="+", default=["balanced_both", "random"],
                    choices=["balanced_both", "random"])
    ap.add_argument("--cal_sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--test_per_class", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--mlp_seeds", type=int, default=3)
    ap.add_argument("--mlp_epochs", type=int, default=100)
    ap.add_argument("--random_min_cal_softmax", type=int, default=800)
    ap.add_argument("--proto_temperature", default="auto")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--output_dir", default="output/pool_repr_menu/g3_semisup")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu")
        args.device = "cpu"

    lab = args.lab_path or os.path.join(
        args.data_dir, f"embeddings_{args.dataset}.pt")
    unl = args.unl_path or os.path.join(
        args.data_dir, f"embeddings_{args.dataset}_unlabeled.pt")
    d = torch.load(lab, map_location="cpu", weights_only=False)
    X, y = d["embeddings"].numpy(), d["labels"].numpy()
    Xu = torch.load(unl, map_location="cpu",
                    weights_only=False)["embeddings"].numpy()
    allc = np.unique(y)
    print(f"{args.dataset}: X={X.shape} K={len(allc)} pool={Xu.shape} "
          f"device={args.device}")

    out = {"config": vars(args), "dataset": args.dataset,
           "selftrain_rows": [], "fcp_rows": []}
    if "selftrain" in args.arms:
        print("\n### Arm A: self-training probe + THR split CP ###")
        out["selftrain_rows"] = run_selftrain(X, y, Xu, allc, args)
    if "fcp" in args.arms:
        print("\n### FCP arms (raw / champion / pool-MLP) ###")
        out["fcp_rows"] = run_fcp_arms(X, y, Xu, allc, args.dataset, args)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_json = os.path.join(args.output_dir,
                            f"results_{args.dataset}{tag}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
