"""Faithful numpy port of the official SemiCP pipeline (Zhou et al.,
"Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score",
arXiv 2505.21147, CVPR 2026) and of the TorchCP split-CP primitives it is
built on.

Provenance (the official repo github.com/Shinning-Zhou/SemiCP carries NO
license file, so the code is PORTED line-for-line rather than vendored;
every function cites its source):

    official_scores_all / official_scores_true
        SemiCP src/score/aps.py  (class APS, non-randomized branch: U = 0)
        SemiCP src/score/raps.py (class RAPS: reg = max(penalty*(rank_1b -
        kreg), 0), added to the cumulative sum) -- paper configs use
        RAPS(penalty=0.01, kreg=2) (CONFIG.py, all experiment blocks)
        THR = TorchCP v1.0.1 thr.py: score = 1 - softmax prob
    official_qhat
        TorchCP v1.0.1 torchcp/utils/common.py::calculate_conformal_value:
        level = ceil((N+1)*(1-alpha))/N, torch.quantile(...,
        interpolation='lower') (= exact order statistic; numpy
        method='lower' is identical), level > 1 -> q_hat = +inf
    uns_adjusted_scores
        SemiCP src/score/uns.py::UNS.__call__ (headline non-randomized
        path) + UNS.cal_closest_indices: 1-NN matching in SCALAR
        predicted-label-score space over ALL labeled points, bias
        correction  s_u = S(x_u, yhat_u) + [S(x_j*, y_j*) - S(x_j*,
        yhat_j*)]
    semicp_qhat
        SemiCP src/predictor/semisplit.py::SemiPredictor.
        calculate_threshold: concat(true labeled scores, adjusted
        unlabeled scores) -> official_qhat on the merged vector (no
        weighting)
    predict_sets
        TorchCP v1.0.1 SplitPredictor._generate_prediction_set:
        C(x) = { y : S(x, y) <= q_hat }  (boundary class EXCLUDED when its
        score exceeds q_hat -- note our legacy split_cp_baselines.py
        included it; this port is the paper-faithful rule)

Deliberate adaptations (everything else is 1:1):
  * input is a PROBABILITY matrix (n, K), not logits -- the official code
    applies softmax(logits) first; callers here pass predict_proba output
    (softmax of the probe's logits), which is the same quantity.
  * numpy instead of torch (CPU; the matrices involved are small).
  * only the non-randomized scores are ported -- the paper's main figures
    are non-randomized (randomized APS/RAPS live in their appendix).
  * argmax tie-breaking: official kthvalue/torch.sort vs np.argmax can
    differ on EXACT float ties (probability-zero events).
"""
import math
import numpy as np

RAPS_PENALTY = 0.01     # official CONFIG.py: RAPS(penalty=0.01, kreg=2)
RAPS_KREG = 2

SCORE_FNS = ("THR", "APS", "RAPS")


def official_scores_all(P, score_fn, penalty=RAPS_PENALTY, kreg=RAPS_KREG):
    """(n, K) nonconformity scores for every class, non-randomized.

    THR:  1 - p_y                       (TorchCP thr.py)
    APS:  inclusive descending-sorted cumulative probability through y's
          rank                          (SemiCP aps.py, U = 0)
    RAPS: APS + penalty * max(0, rank_1based - kreg)   (SemiCP raps.py)
    """
    P = np.asarray(P, dtype=np.float64)
    if score_fn == "THR":
        return 1.0 - P
    order = np.argsort(-P, axis=1)                       # descending
    csum = np.cumsum(np.take_along_axis(P, order, axis=1), axis=1)
    if score_fn == "RAPS":
        ranks_1b = np.arange(1, P.shape[1] + 1, dtype=np.float64)
        csum = csum + np.maximum(penalty * (ranks_1b - kreg), 0.0)
    elif score_fn != "APS":
        raise ValueError(f"unknown score_fn {score_fn!r}")
    S = np.empty_like(csum)
    np.put_along_axis(S, order, csum, axis=1)
    return S


def official_scores_true(P, y_idx, score_fn, penalty=RAPS_PENALTY,
                         kreg=RAPS_KREG):
    """(n,) score at the true label (official _calculate_single_label)."""
    S = official_scores_all(P, score_fn, penalty, kreg)
    return S[np.arange(len(S)), np.asarray(y_idx, dtype=int)]


def official_qhat(scores, alpha):
    """TorchCP calculate_conformal_value: exact order-statistic quantile
    ('lower' interpolation) at level ceil((N+1)(1-alpha))/N; +inf (-> full
    prediction sets) when the level exceeds 1."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return np.inf
    level = math.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return np.inf
    return float(np.quantile(scores, level, method="lower"))


def uns_adjusted_scores(P_unl, P_cal, ycal_idx, score_fn,
                        penalty=RAPS_PENALTY, kreg=RAPS_KREG,
                        chunk=4096):
    """(N,) NNM bias-corrected pseudo-scores (uns.py::UNS, non-randomized).

    For each unlabeled x_u: pseudo-label yhat_u = argmax prob; match the
    labeled point j* minimizing |S(x_j, yhat_j) - S(x_u, yhat_u)| (scalar
    score space, k=1, no label restriction); return
    S(x_u, yhat_u) + [S(x_j*, y_j*) - S(x_j*, yhat_j*)].
    """
    S_cal = official_scores_all(P_cal, score_fn, penalty, kreg)
    n_cal = len(S_cal)
    ar = np.arange(n_cal)
    yhat_cal = np.asarray(P_cal).argmax(axis=1)
    s_pred_cal = S_cal[ar, yhat_cal]                       # S(x_j, yhat_j)
    s_true_cal = S_cal[ar, np.asarray(ycal_idx, dtype=int)]
    diff = s_true_cal - s_pred_cal                         # per-j correction

    S_unl = official_scores_all(P_unl, score_fn, penalty, kreg)
    yhat_unl = np.asarray(P_unl).argmax(axis=1)
    s_pred_unl = S_unl[np.arange(len(S_unl)), yhat_unl]    # S(x_u, yhat_u)

    out = np.empty_like(s_pred_unl)
    for lo in range(0, len(s_pred_unl), chunk):
        blk = s_pred_unl[lo:lo + chunk]
        # official argmin(axis=0) over the labeled axis; np.argmin matches
        # torch (first index) on ties
        j = np.abs(s_pred_cal[:, None] - blk[None, :]).argmin(axis=0)
        out[lo:lo + chunk] = blk + diff[j]
    return out


def semicp_qhat(P_cal, ycal_idx, P_unl, alpha, score_fn,
                penalty=RAPS_PENALTY, kreg=RAPS_KREG):
    """SemiPredictor.calculate_threshold: merged labeled + NNM quantile."""
    s_lab = official_scores_true(P_cal, ycal_idx, score_fn, penalty, kreg)
    s_unl = uns_adjusted_scores(P_unl, P_cal, ycal_idx, score_fn,
                                penalty, kreg)
    return official_qhat(np.concatenate([s_lab, s_unl]), alpha)


def splitcp_qhat(P_cal, ycal_idx, alpha, score_fn,
                 penalty=RAPS_PENALTY, kreg=RAPS_KREG):
    """Plain split CP threshold on the labeled scores only (the paper's
    'standard CP' baseline; SemiPredictor with unlabeled part removed)."""
    return official_qhat(
        official_scores_true(P_cal, ycal_idx, score_fn, penalty, kreg),
        alpha)


def predict_sets(P_test, q_hat, score_fn, penalty=RAPS_PENALTY,
                 kreg=RAPS_KREG):
    """List of label-index arrays; C(x) = {y : S(x, y) <= q_hat}."""
    S = official_scores_all(P_test, score_fn, penalty, kreg)
    return [np.flatnonzero(S[i] <= q_hat) for i in range(len(S))]
