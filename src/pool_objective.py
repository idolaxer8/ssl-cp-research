"""Pool-only CP-native objective library (conformal metric learning, G1).

Shared statistics for FITTING and SELECTING feature transforms using ONLY the
unlabeled pool -- no labels, no cal/test data -- so that by Proposition 2
(theory.md sec 2) any transform chosen or optimized through these functions
keeps Full CP's coverage guarantee exact.

Extracted from src/transform_selection_pilot.py (2026-07-27 selector pilot),
with two upgrades for fitting use:
  - rehearsal_setsize n_rep default raised 5 -> 20 (the pilot's 5 was flagged
    too noisy for selection; fitting needs a stabler objective),
  - vectorized centroid-ratio score matrix (the pilot looped over classes).

Objective menu:
  rehearsal_setsize  split-CP dress rehearsal at deployment alpha on k-means
                     pseudo-labels, centroid-ratio score (mirrors the asym
                     geodesic NCM). THE primary fitting objective: it is the
                     deployment CP functional (expected set size) label-free.
  margin_stats       near-tie mass of the d1/d2 nearest/second-nearest
                     pseudo-centroid distance ratio. CP set size is a
                     tail-of-margins functional (accuracy-based selection had
                     +89% regret on aircraft); margin_q90 ranked transforms
                     with 0% regret at high PR -> diagnostic / tie-break.
"""
import numpy as np


def l2n(X, eps=1e-12):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def pseudo_task(Xa_raw, Xb_raw, K, seed, n_init=5):
    """Fixed pseudo-supervision: MiniBatchKMeans with K = |label space| (known
    a priori, NOT read from cal/test labels) on the raw L2-normalized pool
    half A; half B gets predicted labels. The SAME pseudo-labels are used for
    every candidate transform so candidates compete on one external task."""
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=n_init,
                         batch_size=1024).fit(l2n(Xa_raw))
    return km.labels_, km.predict(l2n(Xb_raw)), km


def centroid_dists(Zb, mus):
    """Cosine distance (1 - cos) of each row of Zb to each centroid."""
    Zn, Mn = l2n(Zb), l2n(mus)
    return 1.0 - Zn @ Mn.T


def class_means(Za, ya, K):
    """Per-pseudo-class means of transformed half-A points (zero vector for an
    empty pseudo-class, matching the selector pilot)."""
    return np.stack([
        Za[ya == c].mean(0) if (ya == c).any() else np.zeros(Za.shape[1])
        for c in range(K)])


def score_matrix(D):
    """Centroid-ratio score S[x, c] = D[x, c] / min_{c' != c} D[x, c'] --
    mirrors the asymmetric ratio NCM. Vectorized via the two smallest
    distances per row."""
    part = np.partition(D, 1, axis=1)
    min1, min2 = part[:, 0], part[:, 1]
    amin = D.argmin(axis=1)
    other = np.where(np.arange(D.shape[1])[None, :] == amin[:, None],
                     min2[:, None], min1[:, None])
    return D / (other + 1e-12)


def margin_stats(D):
    """margin = d1/d2 in [0, 1]; near 1 = near-tie (the mass CP pays for)."""
    part = np.partition(D, 1, axis=1)
    m = part[:, 0] / (part[:, 1] + 1e-12)
    return {
        "margin_mean": float(m.mean()),
        "margin_q90": float(np.quantile(m, 0.90)),
        "neartie_frac_09": float((m > 0.9).mean()),
    }


def rehearsal_setsize(D, yb, K, cal_budget, alpha, n_rep=20, seed=0,
                      S=None):
    """Split-CP dress rehearsal at deployment (alpha, cal_budget) on the
    pseudo-labeled pool half B.

        balanced pseudo-cal: cal_budget/K points per pseudo-class, rest test
        qhat = empirical quantile of true-pseudo-class cal scores at level
               ceil((n_cal + 1)(1 - alpha)) / n_cal
        value = mean pseudo-test set size, averaged over n_rep cal draws

    Returns (mean, se) over replicates."""
    rng = np.random.default_rng(seed)
    n = len(yb)
    m_cal = max(1, cal_budget // K)
    if S is None:
        S = score_matrix(D)
    sizes = []
    for _ in range(n_rep):
        cal_idx = []
        for c in range(K):
            pc = np.where(yb == c)[0]
            if len(pc) == 0:
                continue
            cal_idx.append(rng.choice(pc, min(m_cal, len(pc)), replace=False))
        cal_idx = np.concatenate(cal_idx)
        test_mask = np.ones(n, bool)
        test_mask[cal_idx] = False
        cal_scores = S[cal_idx, yb[cal_idx]]
        k = int(np.ceil((len(cal_scores) + 1) * (1 - alpha)))
        qhat = np.inf if k > len(cal_scores) else np.sort(cal_scores)[k - 1]
        sizes.append(float((S[test_mask] <= qhat).sum(axis=1).mean()))
    sizes = np.asarray(sizes)
    return float(sizes.mean()), float(sizes.std() / np.sqrt(len(sizes)))


def objective_on_half(Za, Zb, ya, yb, K, cal_budget, alpha, n_rep=20, seed=0):
    """One-call objective for a candidate space: pseudo-centroids from half A,
    rehearsal + margin statistics on half B. Returns a dict."""
    mus = class_means(Za, ya, K)
    D = centroid_dists(Zb, mus)
    out = margin_stats(D)
    out["rehearsal_sz"], out["rehearsal_se"] = rehearsal_setsize(
        D, yb, K, cal_budget, alpha, n_rep=n_rep, seed=seed)
    return out


def multires_rehearsal(Xa_raw, Xb_raw, transform_fn, K, cal_budget, alpha,
                       k_mults=(1, 2, 4), n_rep=20, seed=0):
    """Multi-resolution mitigation arm (low-PR): average the rehearsal size
    over pseudo-tasks with K' = m*K subclususters (harder tasks probe finer
    structure than K-way k-means can see). transform_fn maps raw features to
    the candidate space."""
    Za, Zb = transform_fn(Xa_raw), transform_fn(Xb_raw)
    vals = {}
    for m in k_mults:
        Kp = int(m * K)
        ya, yb, _ = pseudo_task(Xa_raw, Xb_raw, Kp, seed + 17 * m)
        mus = class_means(Za, ya, Kp)
        D = centroid_dists(Zb, mus)
        # cal budget scales with K' so shots/pseudo-class stay comparable
        vals[Kp], _ = rehearsal_setsize(D, yb, Kp, cal_budget * m, alpha,
                                        n_rep=n_rep, seed=seed)
    return float(np.mean(list(vals.values()))), vals
