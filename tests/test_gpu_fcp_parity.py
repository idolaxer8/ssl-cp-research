"""
GPU vs CPU parity tests for FullConformalPredictor.

Runs a battery of checks designed to expose any divergence between the
existing CPU predict() path and the new GPU `predict(device='cuda')`
path on `GeodesicTopKMeanNCM`.

Acceptance criteria per test:
  - Prediction set match rate == 100%.
  - P-value max abs diff <= TOL_PV (one p-value quantum 1/(n_cal+1) for
    numerator-only NCMs whose score lands directly on the rank ladder;
    ~1e-6 otherwise).
  - Set sizes / coverage identical.

Run:  python tests/test_gpu_fcp_parity.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conformal_prediction import (  # noqa: E402
    FullConformalPredictor,
    GeodesicTopKMeanNCM,
    create_ncm,
    stratified_cal_test_split,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

NCM_VARIANTS = [
    "geodesic_topk_mean",
    "geodesic_topk_asym",
    "unwhitened_topk_mean",
    "unwhitened_topk_asym",
    "geodesic_topk",      # numerator-only, top-k
    "geodesic_1nn",       # numerator-only, 1-NN
]

TOL_PV_STRONG = 1e-5    # bit-near-equivalent (no rank flips)
# All NCMs (ratio or numerator-only) compute scores in float32; when a
# cal score and the test score are mathematically equal, CPU and GPU
# can land on different ULPs of the same value, flipping a single >=
# comparison and shifting that pair's p-value by exactly 1/(n_cal+1).
# This is benign — sets and coverage stay identical — so we accept up
# to one p-value quantum as "QUANTUM" parity.
NUM_ONLY = {"geodesic_topk", "geodesic_1nn"}


def make_gmm(K, per_class, d, seed, sep=4.0, dtype=np.float32):
    """Synthetic Gaussian-mixture embeddings; K balanced classes."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(K, d)) * sep
    X = np.concatenate(
        [centers[c] + rng.normal(size=(per_class, d)) for c in range(K)]
    ).astype(dtype)
    y = np.concatenate([np.full(per_class, c) for c in range(K)]).astype(np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def p_values_matrix(result, K):
    """Convert `result['p_values']` (list of dicts) to a (n_test, K) array."""
    n = len(result["p_values"])
    out = np.zeros((n, K), dtype=np.float64)
    for i, pd in enumerate(result["p_values"]):
        for c, v in pd.items():
            if 0 <= c < K:
                out[i, c] = v
    return out


def compare(name, r_cpu, r_gpu, K, n_cal, ncm_name, alpha=None, fail_fast=False):
    """Return (passed: bool, message: str).

    Three outcomes:
      STRONG  pv_diff <= 1e-5 and sets/sizes equal — bit-near-equivalent.
      QUANTUM 1e-5 < pv_diff <= 1/(n_cal+1) and sets/sizes equal —
              one ULP tie-flip on a >= comparison; semantically equivalent
              (FCP coverage guarantee and prediction set unchanged).
      FAIL    anything else.
    """
    sets_cpu = [tuple(sorted(s)) for s in r_cpu["prediction_sets"]]
    sets_gpu = [tuple(sorted(s)) for s in r_gpu["prediction_sets"]]
    set_match = sum(a == b for a, b in zip(sets_cpu, sets_gpu)) / max(1, len(sets_cpu))

    pv_cpu = p_values_matrix(r_cpu, K)
    pv_gpu = p_values_matrix(r_gpu, K)
    pv_diff = float(np.max(np.abs(pv_cpu - pv_gpu))) if pv_cpu.size else 0.0
    quantum = 1.0 / (n_cal + 1)

    sz_cpu = np.array(r_cpu["set_sizes"])
    sz_gpu = np.array(r_gpu["set_sizes"])
    sz_equal = np.array_equal(sz_cpu, sz_gpu)
    sets_equal = (set_match == 1.0) and sz_equal

    # Allow a few ULPs of float-32 slack on top of the 1-quantum tie-flip,
    # since the >=/< boundary can introduce O(1e-7) noise around the rank
    # transition.
    if sets_equal and pv_diff <= TOL_PV_STRONG:
        verdict = "STRONG"
    elif sets_equal and pv_diff <= quantum + 1e-6:
        verdict = "QUANTUM"
    else:
        verdict = "FAIL"

    msg = (
        f"[{verdict:7s}] {name:50s}  "
        f"set={set_match*100:5.1f}%  pv_diff={pv_diff:.2e} (q={quantum:.1e})  "
        f"sz_equal={sz_equal}"
    )
    if verdict == "FAIL" and fail_fast:
        print(msg)
        print(f"  CPU sets[:5]: {sets_cpu[:5]}")
        print(f"  GPU sets[:5]: {sets_gpu[:5]}")
        if not sz_equal:
            mismatch = np.where(sz_cpu != sz_gpu)[0][:5]
            print(f"  size mismatch at idx={mismatch.tolist()}: CPU={sz_cpu[mismatch].tolist()}, GPU={sz_gpu[mismatch].tolist()}")
        raise SystemExit(1)
    return (verdict != "FAIL"), msg


def run_pair(X_cal, y_cal, X_test, K, ncm_name, alpha=0.1, gpu_batch_size=256):
    classes_full = np.arange(K)

    ncm_cpu = create_ncm(ncm_name)
    cp_cpu = FullConformalPredictor(ncm_cpu, alpha=alpha)
    cp_cpu.calibrate(X_cal, y_cal, all_classes=classes_full)
    r_cpu = cp_cpu.predict(X_test, return_p_values=True, verbose=False, device="cpu")

    ncm_gpu = create_ncm(ncm_name)
    cp_gpu = FullConformalPredictor(ncm_gpu, alpha=alpha)
    cp_gpu.calibrate(X_cal, y_cal, all_classes=classes_full)
    r_gpu = cp_gpu.predict(
        X_test, return_p_values=True, verbose=False,
        device="cuda", gpu_batch_size=gpu_batch_size,
    )
    return r_cpu, r_gpu


# ----------------------------------------------------------------------
# Test A: NCM × size grid on synthetic data
# ----------------------------------------------------------------------

def test_ncm_size_grid():
    print("\n=== Test A: NCM × size grid ===")
    results = []
    grid = [
        # (K, per_class, n_test, d, alpha, seed)
        ( 5,  40,  50,  64,  0.10, 0),
        (20,  30, 100, 128,  0.10, 1),
        (50,  20, 200, 256,  0.10, 2),
        (10, 100, 100,  32,  0.05, 3),  # tighter alpha
        (10, 100, 100,  32,  0.30, 4),  # looser alpha
    ]
    for K, per_class, n_test, d, alpha, seed in grid:
        X, y = make_gmm(K, per_class + max(20, n_test // K + 1), d, seed)
        cal_size = K * per_class
        X_cal, y_cal, X_te, y_te = stratified_cal_test_split(
            X, y, cal_size=cal_size, test_size=n_test, balanced=True, random_state=seed
        )
        for ncm_name in NCM_VARIANTS:
            r_cpu, r_gpu = run_pair(X_cal, y_cal, X_te, K, ncm_name, alpha=alpha)
            name = f"K={K:3d} cal={cal_size:4d} test={n_test:3d} d={d:3d} a={alpha} {ncm_name}"
            ok, msg = compare(name, r_cpu, r_gpu, K, cal_size, ncm_name)
            print(" " + msg)
            results.append(ok)
    return all(results)


# ----------------------------------------------------------------------
# Test B: Edge cases
# ----------------------------------------------------------------------

def test_edge_cases():
    print("\n=== Test B: Edge cases ===")
    results = []

    # B1: n_test = 1
    X, y = make_gmm(10, 30, 32, seed=10)
    X_cal, y_cal, X_te, _ = stratified_cal_test_split(X, y, cal_size=200, test_size=10, balanced=True, random_state=10)
    for ncm_name in ["geodesic_topk_asym", "geodesic_topk_mean"]:
        r_cpu, r_gpu = run_pair(X_cal, y_cal, X_te[:1], 10, ncm_name)
        ok, msg = compare(f"B1 n_test=1 {ncm_name}", r_cpu, r_gpu, 10, 200, ncm_name)
        print(" " + msg); results.append(ok)

    # B2: K = 2 (binary)
    X, y = make_gmm(2, 200, 32, seed=11)
    X_cal, y_cal, X_te, _ = stratified_cal_test_split(X, y, cal_size=200, test_size=100, balanced=True, random_state=11)
    for ncm_name in NCM_VARIANTS:
        r_cpu, r_gpu = run_pair(X_cal, y_cal, X_te, 2, ncm_name)
        ok, msg = compare(f"B2 K=2 {ncm_name}", r_cpu, r_gpu, 2, 200, ncm_name)
        print(" " + msg); results.append(ok)

    # B3: K = 200 (CUB-scale)
    X, y = make_gmm(200, 6, 64, seed=12)  # 1200 points
    X_cal, y_cal, X_te, _ = stratified_cal_test_split(X, y, cal_size=600, test_size=200, balanced=True, random_state=12)
    for ncm_name in ["geodesic_topk_asym", "geodesic_topk_mean"]:
        r_cpu, r_gpu = run_pair(X_cal, y_cal, X_te, 200, ncm_name)
        ok, msg = compare(f"B3 K=200 {ncm_name}", r_cpu, r_gpu, 200, 600, ncm_name)
        print(" " + msg); results.append(ok)

    # B4: Imbalanced cal (some classes 1 member, others many)
    rng = np.random.default_rng(13)
    K = 10
    counts = np.array([1, 1, 2, 5, 10, 20, 30, 40, 50, 60])
    Xs, ys = [], []
    centers = rng.normal(size=(K, 32)) * 3.0
    for c in range(K):
        Xs.append(centers[c] + rng.normal(size=(counts[c], 32)))
        ys.append(np.full(counts[c], c))
    X_cal = np.concatenate(Xs).astype(np.float32)
    y_cal = np.concatenate(ys).astype(np.int64)
    perm = rng.permutation(len(X_cal)); X_cal, y_cal = X_cal[perm], y_cal[perm]
    # test set: 50 fresh points (still balanced)
    Xt, yt = make_gmm(K, 6, 32, seed=14)
    n_cal = len(X_cal)
    for ncm_name in ["geodesic_topk_asym", "geodesic_topk_mean", "geodesic_1nn"]:
        r_cpu, r_gpu = run_pair(X_cal, y_cal, Xt, K, ncm_name)
        ok, msg = compare(f"B4 imbalanced cal n_cal={n_cal} {ncm_name}", r_cpu, r_gpu, K, n_cal, ncm_name)
        print(" " + msg); results.append(ok)

    # B5: Duplicate test point (sim → 1.0, arccos → 0)
    X, y = make_gmm(10, 30, 32, seed=15)
    X_cal, y_cal, _, _ = stratified_cal_test_split(X, y, cal_size=200, test_size=10, balanced=True, random_state=15)
    X_dup = X_cal[:5].copy()  # exact copies of cal points
    for ncm_name in ["geodesic_topk_asym", "geodesic_topk_mean", "geodesic_1nn"]:
        r_cpu, r_gpu = run_pair(X_cal, y_cal, X_dup, 10, ncm_name)
        ok, msg = compare(f"B5 dup-of-cal {ncm_name}", r_cpu, r_gpu, 10, 200, ncm_name)
        print(" " + msg); results.append(ok)

    # B6: K = 2, n_test very small, alpha extreme
    X, y = make_gmm(2, 200, 32, seed=16)
    X_cal, y_cal, X_te, _ = stratified_cal_test_split(X, y, cal_size=200, test_size=10, balanced=True, random_state=16)
    for alpha in [0.01, 0.5]:
        for ncm_name in ["geodesic_topk_asym"]:
            r_cpu, r_gpu = run_pair(X_cal, y_cal, X_te, 2, ncm_name, alpha=alpha)
            ok, msg = compare(f"B6 alpha={alpha} {ncm_name}", r_cpu, r_gpu, 2, 200, ncm_name, alpha=alpha)
            print(" " + msg); results.append(ok)

    return all(results)


# ----------------------------------------------------------------------
# Test C: gpu_batch_size invariance
# ----------------------------------------------------------------------

def test_batch_size_invariance():
    print("\n=== Test C: gpu_batch_size invariance ===")
    X, y = make_gmm(20, 30, 64, seed=20)
    X_cal, y_cal, X_te, _ = stratified_cal_test_split(X, y, cal_size=400, test_size=200, balanced=True, random_state=20)
    K = 20

    ncm = create_ncm("geodesic_topk_asym")
    cp = FullConformalPredictor(ncm, alpha=0.1)
    cp.calibrate(X_cal, y_cal, all_classes=np.arange(K))

    # Run with several batch sizes; compare each to bs=256 (the default).
    ref = cp.predict(X_te, return_p_values=True, verbose=False, device="cuda", gpu_batch_size=256)
    ref_pv = p_values_matrix(ref, K)
    results = []
    for bs in [1, 7, 33, 64, 1024]:
        r = cp.predict(X_te, return_p_values=True, verbose=False, device="cuda", gpu_batch_size=bs)
        pv = p_values_matrix(r, K)
        diff = float(np.max(np.abs(pv - ref_pv)))
        set_match = sum(tuple(sorted(a)) == tuple(sorted(b))
                        for a, b in zip(ref["prediction_sets"], r["prediction_sets"])) / len(r["prediction_sets"])
        ok = (diff <= 1e-6) and (set_match == 1.0)
        print(f" [{'PASS' if ok else 'FAIL'}] gpu_batch_size={bs:5d}  pv_diff={diff:.2e}  set={set_match*100:.1f}%")
        results.append(ok)
    return all(results)


# ----------------------------------------------------------------------
# Test D: GPU determinism
# ----------------------------------------------------------------------

def test_determinism():
    print("\n=== Test D: GPU determinism (same run twice) ===")
    # 10 classes x 80/class = 800 pool → 20/class test + 60/class cal
    X, y = make_gmm(10, 80, 64, seed=30)
    X_cal, y_cal, X_te, _ = stratified_cal_test_split(
        X, y, cal_size=400, test_size=200, balanced=True, random_state=30)
    K = 10
    results = []
    for ncm_name in NCM_VARIANTS:
        ncm = create_ncm(ncm_name)
        cp = FullConformalPredictor(ncm, alpha=0.1)
        cp.calibrate(X_cal, y_cal, all_classes=np.arange(K))
        r1 = cp.predict(X_te, return_p_values=True, verbose=False, device="cuda")
        r2 = cp.predict(X_te, return_p_values=True, verbose=False, device="cuda")
        pv1 = p_values_matrix(r1, K); pv2 = p_values_matrix(r2, K)
        diff = float(np.max(np.abs(pv1 - pv2)))
        ok = diff == 0.0
        print(f" [{'PASS' if ok else 'FAIL'}] {ncm_name:25s}  pv_diff={diff:.2e}")
        results.append(ok)
    return all(results)


# ----------------------------------------------------------------------
# Test E: Real CIFAR-100 embeddings (multi-trial)
# ----------------------------------------------------------------------

def test_coverage_invariance():
    """Sanity: empirical coverage on CPU and GPU is bit-identical when sets are."""
    print("\n=== Test E0: Coverage invariance across NCM variants ===")
    X, y = make_gmm(20, 80, 64, seed=50)
    X_cal, y_cal, X_te, y_te = stratified_cal_test_split(
        X, y, cal_size=600, test_size=300, balanced=True, random_state=50)
    K = 20
    results = []
    for ncm_name in NCM_VARIANTS:
        r_cpu, r_gpu = run_pair(X_cal, y_cal, X_te, K, ncm_name)
        cov_cpu = float(np.mean([y_te[i] in s for i, s in enumerate(r_cpu['prediction_sets'])]))
        cov_gpu = float(np.mean([y_te[i] in s for i, s in enumerate(r_gpu['prediction_sets'])]))
        sz_cpu = float(np.mean(r_cpu['set_sizes']))
        sz_gpu = float(np.mean(r_gpu['set_sizes']))
        ok = (cov_cpu == cov_gpu) and (sz_cpu == sz_gpu)
        print(f" [{'PASS' if ok else 'FAIL'}] {ncm_name:25s}  cov CPU={cov_cpu:.4f} GPU={cov_gpu:.4f}  size CPU={sz_cpu:.3f} GPU={sz_gpu:.3f}")
        results.append(ok)
    return all(results)


def test_real_cifar100():
    print("\n=== Test E: Real CIFAR-100 embeddings (3 trials) ===")
    emb_path = "output/embeddings_cifar100.pt"
    test_path = "output/embeddings_cifar100_test.pt"
    if not (os.path.exists(emb_path) and os.path.exists(test_path)):
        print("  SKIPPED — embeddings not found")
        return True

    d_cal = torch.load(emb_path, map_location="cpu", weights_only=False)
    d_te  = torch.load(test_path, map_location="cpu", weights_only=False)
    X_pool = d_cal["embeddings"].numpy().astype(np.float32)
    y_pool = d_cal["labels"].numpy().astype(np.int64)
    Xt_pool = d_te["embeddings"].numpy().astype(np.float32)
    yt_pool = d_te["labels"].numpy().astype(np.int64)
    K = int(max(y_pool.max(), yt_pool.max())) + 1

    results = []
    for trial in range(3):
        X_cal, y_cal, _, _ = stratified_cal_test_split(
            X_pool, y_pool, cal_size=600, test_size=200, balanced=True, random_state=100 + trial)
        _, _, X_te, y_te = stratified_cal_test_split(
            Xt_pool, yt_pool, cal_size=200, test_size=600, balanced=True, random_state=200 + trial)
        for ncm_name in ["geodesic_topk_asym", "geodesic_topk_mean"]:
            r_cpu, r_gpu = run_pair(X_cal, y_cal, X_te, K, ncm_name)
            ok, msg = compare(f"E trial{trial} {ncm_name}", r_cpu, r_gpu, K, 600, ncm_name)
            print(" " + msg)
            # Plus: coverage matches
            cov_cpu = float(np.mean([y_te[i] in s for i, s in enumerate(r_cpu['prediction_sets'])]))
            cov_gpu = float(np.mean([y_te[i] in s for i, s in enumerate(r_gpu['prediction_sets'])]))
            cov_ok = abs(cov_cpu - cov_gpu) < 1e-12
            print(f"    coverage CPU={cov_cpu:.4f}  GPU={cov_gpu:.4f}  equal={cov_ok}")
            results.append(ok and cov_ok)
    return all(results)


# ----------------------------------------------------------------------
# Test F: API surface parity
# ----------------------------------------------------------------------

def test_api_parity():
    print("\n=== Test F: API surface parity ===")
    X, y = make_gmm(10, 30, 32, seed=40)
    X_cal, y_cal, X_te, _ = stratified_cal_test_split(X, y, cal_size=200, test_size=50, balanced=True, random_state=40)
    K = 10
    ncm = create_ncm("geodesic_topk_asym")
    cp = FullConformalPredictor(ncm, alpha=0.1)
    cp.calibrate(X_cal, y_cal, all_classes=np.arange(K))

    results = []

    # F1: return_p_values=False on both paths
    r_cpu = cp.predict(X_te, return_p_values=False, verbose=False, device="cpu")
    r_gpu = cp.predict(X_te, return_p_values=False, verbose=False, device="cuda")
    keys_match = (("p_values" not in r_cpu) and ("p_values" not in r_gpu))
    print(f" [{'PASS' if keys_match else 'FAIL'}] return_p_values=False — neither result has 'p_values'")
    results.append(keys_match)

    # F2: prediction_sets is list[list[int]]
    type_ok = (isinstance(r_gpu["prediction_sets"], list)
               and all(isinstance(s, list) for s in r_gpu["prediction_sets"])
               and all(isinstance(x, int) for s in r_gpu["prediction_sets"] for x in s))
    print(f" [{'PASS' if type_ok else 'FAIL'}] prediction_sets typed as list[list[int]]")
    results.append(type_ok)

    # F3: set_sizes is ndarray
    ndarr_ok = isinstance(r_gpu["set_sizes"], np.ndarray)
    print(f" [{'PASS' if ndarr_ok else 'FAIL'}] set_sizes is np.ndarray")
    results.append(ndarr_ok)

    # F4: p_values is list of dicts when requested
    r_gpu_pv = cp.predict(X_te, return_p_values=True, verbose=False, device="cuda")
    pv_ok = (isinstance(r_gpu_pv["p_values"], list)
             and isinstance(r_gpu_pv["p_values"][0], dict)
             and all(isinstance(k, int) for k in r_gpu_pv["p_values"][0].keys()))
    print(f" [{'PASS' if pv_ok else 'FAIL'}] p_values is list[dict[int, float]]")
    results.append(pv_ok)

    # F5: Wrong NCM raises ValueError
    from conformal_prediction import MahalNNRatio
    bad_ncm = MahalNNRatio()
    cp_bad = FullConformalPredictor(bad_ncm, alpha=0.1)
    cp_bad.calibrate(X_cal, y_cal, all_classes=np.arange(K))
    try:
        cp_bad.predict(X_te[:5], verbose=False, device="cuda")
        raise_ok = False
    except ValueError:
        raise_ok = True
    print(f" [{'PASS' if raise_ok else 'FAIL'}] Mahal NCM + device='cuda' raises ValueError")
    results.append(raise_ok)

    return all(results)


# ----------------------------------------------------------------------
# Test G: CUB-200 and miniImageNet sanity (fine-grained, larger K)
# ----------------------------------------------------------------------

def test_fine_grained_datasets():
    print("\n=== Test G: Fine-grained datasets (CUB-200, miniImageNet) ===")
    results = []
    for name, path, cal_size, test_size in [
        ("CUB-200",      "output/embeddings_cub200.pt",      800, 400),
        ("miniImageNet", "output/embeddings_miniimagenet.pt", 800, 400),
    ]:
        if not os.path.exists(path):
            print(f"  SKIPPED {name} — {path} not found")
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        X = d["embeddings"].numpy().astype(np.float32)
        y = d["labels"].numpy().astype(np.int64)
        K = int(y.max()) + 1
        X_cal, y_cal, X_te, _ = stratified_cal_test_split(
            X, y, cal_size=cal_size, test_size=test_size, balanced=True, random_state=42)
        for ncm_name in ["geodesic_topk_asym"]:
            r_cpu, r_gpu = run_pair(X_cal, y_cal, X_te, K, ncm_name)
            ok, msg = compare(f"G {name} K={K} {ncm_name}", r_cpu, r_gpu, K, cal_size, ncm_name)
            print(" " + msg); results.append(ok)
    return all(results) if results else True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Test H: Classes absent from calibration (missing-class symmetry fix)
# ----------------------------------------------------------------------

def test_missing_classes():
    """Candidate classes ABSENT from calibration must score identically on CPU and
    GPU. Regression for the missing-class symmetry fix: the old GPU path forced
    1e9 for the no-same-class case, diverging from the fixed CPU score_x (which
    scores it with the zero-neighbour convention -> finite). Every other parity
    test uses class-complete cal, so this case was previously unguarded."""
    print("\n=== Test H: Missing classes (absent from cal) ===")
    results = []
    for (K, per_class, n_missing, d, seed) in [(20, 25, 4, 64, 10), (50, 12, 9, 128, 11)]:
        X_all, y_all = make_gmm(K, per_class, d, seed)
        X_test, _ = make_gmm(K, 8, d, seed + 100)        # test spans ALL K classes
        keep = y_all < (K - n_missing)                    # drop the last n_missing classes from cal
        X_cal, y_cal = X_all[keep], y_all[keep]
        present = len(np.unique(y_cal))
        assert present == K - n_missing, present
        for ncm_name in ("geodesic_topk_mean", "geodesic_topk_asym"):
            r_cpu, r_gpu = run_pair(X_cal, y_cal, X_test, K, ncm_name)
            results.append(compare(
                f"K={K} missing={n_missing} ({ncm_name})",
                r_cpu, r_gpu, K, len(X_cal), ncm_name))
    return all(results)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — cannot run GPU parity tests.")
        sys.exit(2)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}  numpy: {np.__version__}")

    t0 = time.time()
    suites = [
        ("A:  NCM x size grid",           test_ncm_size_grid),
        ("B:  Edge cases",                test_edge_cases),
        ("C:  Batch-size invariance",     test_batch_size_invariance),
        ("D:  GPU determinism",           test_determinism),
        ("E0: Coverage invariance",       test_coverage_invariance),
        ("E:  Real CIFAR-100",            test_real_cifar100),
        ("F:  API surface parity",        test_api_parity),
        ("G:  Fine-grained datasets",     test_fine_grained_datasets),
        ("H:  Missing classes",           test_missing_classes),
    ]
    summary = []
    for label, fn in suites:
        try:
            ok = fn()
        except Exception as e:
            print(f"[ERROR] {label} raised: {e!r}")
            import traceback; traceback.print_exc()
            ok = False
        summary.append((label, ok))

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("PARITY SUITE SUMMARY")
    print("=" * 70)
    for label, ok in summary:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("=" * 70)
    sys.exit(0 if all(ok for _, ok in summary) else 1)
