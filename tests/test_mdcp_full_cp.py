"""Pilot D (full-CP MDCP) correctness tests.

The load-bearing test is the SYMMETRY ORACLE: reference_bag_D recomputes every
member's purity straight from the bag definition with no cal/test asymmetry;
the engine's cached-incremental path (candidate realizes the bag from
cal-only caches) must reproduce it to float64 precision for every arm. That
identity IS the exchangeability proof of the implementation.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mdcp_full_cp import (PrototypeLACDim, FullCPMDCP, reference_bag_D,
                          bag_prob_rows)
from mdcp_pool_pilot import prototype_lac_scores

RNG = np.random.default_rng(3)
K, M, NPOOL, D1, D2 = 6, 41, 300, 24, 12
KD = 3


def _make_data(missing_class=None):
    mus = RNG.normal(size=(K, D1)) * 3
    y = RNG.integers(0, K, size=M)
    if missing_class is not None:
        y[y == missing_class] = (missing_class + 1) % K
    Zc1 = mus[y] + RNG.normal(size=(M, D1))
    Zc2 = Zc1[:, :D2] + RNG.normal(size=(M, D2)) * 0.5
    yp = RNG.integers(0, K, size=NPOOL)
    Zp1 = mus[yp] + RNG.normal(size=(NPOOL, D1))
    Zp2 = Zp1[:, :D2] + RNG.normal(size=(NPOOL, D2)) * 0.5
    x1 = mus[0] + RNG.normal(size=D1)
    return y, (Zc1, Zc2), (Zp1, Zp2), (x1, x1[:D2])


def _engine(y, Zc, Zp, pool_subsample=NPOOL * K):
    dims = [PrototypeLACDim("v1", T=0.5), PrototypeLACDim("v2", T=0.8)]
    eng = FullCPMDCP(dims, k_d=KD, arms=("pool", "bag", "count"),
                     pool_subsample=pool_subsample, device="cpu", seed=0)
    eng.fit_trial(list(Zc), y, list(Zp), np.arange(K))
    return eng


def test_cal_rows_match_pilotB_convention():
    """PrototypeLACDim's cached cal prob rows == pilot B's
    prototype_lac_scores(loo=True) (the PrototypeSoftmaxNCM convention)."""
    y, Zc, Zp, _ = _make_data()
    dim = PrototypeLACDim("v1", T=0.5).fit_trial(Zc[0], y, Zp[0], np.arange(K))
    probs = dim.E / dim.Zrow[:, None]
    lac_ref = prototype_lac_scores(Zc[0], Zc[0], y, np.arange(K), 0.5, loo=True)
    assert np.allclose(1.0 - probs, lac_ref, atol=1e-10)


def test_symmetry_oracle_engine_equals_reference():
    """Engine candidate path == symmetric reference, all arms, incl. a
    candidate class that is MISSING from cal (corner stratum)."""
    for missing in (None, 4):
        y, Zc, Zp, x = _make_data(missing_class=missing)
        eng = _engine(y, Zc, Zp)
        st = eng.point_state(list(x))
        for y_cand in ([0, 2, 4] if missing else [0, 2]):
            D_eng = eng.candidate_D(st, y_cand)
            y_bag = np.concatenate([y, [y_cand]])
            Zb = [np.vstack([Zc[i], x[i][None, :]]) for i in range(2)]
            for arm in ("pool", "bag", "count"):
                D_ref = reference_bag_D(Zb, y_bag, list(Zp), [0.5, 0.8], KD,
                                        arm, pool_pairs=eng.pool_pairs)
                assert np.allclose(D_eng[arm].numpy(), D_ref, rtol=1e-8,
                                   atol=1e-10), \
                    f"arm={arm} y_cand={y_cand} missing={missing}"


def test_bag_permutation_invariance():
    """reference_bag_D depends on the bag as a SET: permuting members
    permutes D identically (the exchangeability property itself)."""
    y, Zc, Zp, x = _make_data()
    y_bag = np.concatenate([y, [1]])
    Zb = [np.vstack([Zc[i], x[i][None, :]]) for i in range(2)]
    D0 = reference_bag_D(Zb, y_bag, list(Zp), [0.5, 0.8], KD, "bag")
    perm = RNG.permutation(len(y_bag))
    Dp = reference_bag_D([z[perm] for z in Zb], y_bag[perm], list(Zp),
                         [0.5, 0.8], KD, "bag")
    assert np.allclose(D0[perm], Dp, atol=1e-10)


def test_missing_class_corner_convention():
    """Candidate class absent from cal: the test point's true score must be
    exactly 1 (empty-class LAC) in every dim -> the corner stratum exists."""
    y, Zc, Zp, x = _make_data(missing_class=4)
    eng = _engine(y, Zc, Zp)
    st = eng.point_state(list(x))
    for di in range(2):
        t_probs = st["tpr"][di].numpy()
        assert t_probs[4] == 0.0                       # empty class -> p = 0
    # and cal singletons: LOO empty -> own prob 0 -> true coord 1
    ysing = y.copy()
    ysing[0] = 4                                       # make class 4 a singleton
    eng2 = _engine(ysing, Zc, Zp)
    probs0 = eng2.dims[0].E[0] / eng2.dims[0].Zrow[0]
    assert probs0[4] == 0.0


def test_pvalues_valid_shape_and_range():
    y, Zc, Zp, x = _make_data()
    eng = _engine(y, Zc, Zp, pool_subsample=500)
    p = eng.predict_point(list(x))
    for arm in ("pool", "bag", "count"):
        assert p[arm].shape == (K,)
        assert np.all(p[arm] >= 1.0 / (M + 1)) and np.all(p[arm] <= 1.0)


def test_batched_path_matches_loop_path():
    """Stage-2 fast path == Stage-1 oracle-verified loop path (all arms,
    exact p-value identity in float64)."""
    y, Zc, Zp, x = _make_data()
    eng = _engine(y, Zc, Zp, pool_subsample=800)
    p_loop = eng.predict_point(list(x))
    p_fast = eng.predict_point_batched(list(x), y_chunk=4)
    for arm in ("pool", "bag", "count"):
        assert np.array_equal(p_loop[arm], p_fast[arm]), arm


if __name__ == "__main__":
    test_cal_rows_match_pilotB_convention()
    test_symmetry_oracle_engine_equals_reference()
    test_bag_permutation_invariance()
    test_missing_class_corner_convention()
    test_pvalues_valid_shape_and_range()
    print("ALL PASS")
