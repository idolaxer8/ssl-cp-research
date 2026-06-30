"""
Set-parity tests for the MS-CS GPU/torch fast path (src/mscs_gpu.py) vs the CPU
loop in run_fcp_with_mscs. Synthetic GMM data with classes deliberately missing
from cal (to exercise the missing-class path). Run: python tests/test_mscs_gpu_parity.py
"""
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
from mscs_unlabeled_experiment import (  # noqa: E402
    build_cluster_similarity_matrix, build_centroid_similarity_matrix,
    update_centroid_M_for_candidate, build_prototype_similarity_matrix,
    update_prototype_M_for_candidate, run_fcp_with_mscs)


def gmm(K, per, d, seed, sep=4.0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(K, d)) * sep
    X = np.concatenate([C[c] + rng.normal(size=(per, d)) for c in range(K)]).astype(np.float32)
    y = np.concatenate([np.full(per, c) for c in range(K)]).astype(np.int64)
    p = rng.permutation(len(X))
    return X[p], y[p]


def _sets(ret):
    return ret[2]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping MS-CS GPU parity tests.")
        return 0
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    K, d = 25, 48
    Xu, _ = gmm(K, 60, d, seed=7)                       # unlabeled pool for cluster M
    allc = np.arange(K)
    ncm = "unwhitened_topk_mean"
    ok = True

    for cal, per_test in [(75, 40), (200, 60)]:          # cal=75 -> classes missing from cal
        X, y = gmm(K, max(8, cal // K + 6), d, seed=3)
        rng = np.random.default_rng(0); idx = rng.permutation(len(X))
        Xc, yc = X[idx[:cal]], y[idx[:cal]]
        Xt, yt = X[idx[cal:cal + per_test]], y[idx[cal:cal + per_test]]
        nmiss = K - len(np.unique(yc))
        Mc, c2c, et, md, cc, cn, clc, cld = build_cluster_similarity_matrix(Xu, Xc, yc, allc, 6, tau=-0.5)
        clu = dict(class_centroids=cc, class_counts=cn, class_to_cluster=c2c,
                   cluster_centroids=clc, cluster_dists=cld, effective_tau=et)

        for exch in (False, True):
            common = dict(exchangeable=exch, yhat_mode="ncm", return_sets=True,
                          allow_nonexchangeable=True, **clu)
            r_cpu = run_fcp_with_mscs(Xc, yc, Xt, yt, allc, ncm, 0.1, 0.05, Mc, device="cpu", **common)
            r_gpu = run_fcp_with_mscs(Xc, yc, Xt, yt, allc, ncm, 0.1, 0.05, Mc, device="cuda", **common)
            same = all(set(a) == set(b) for a, b in zip(_sets(r_cpu), _sets(r_gpu)))
            print(f"  cal={cal} missing={nmiss} exch={exch!s:5}: cluster-M parity={same} "
                  f"cov {r_cpu[0]:.4f}/{r_gpu[0]:.4f} sz {r_cpu[1]:.3f}/{r_gpu[1]:.3f}")
            ok = ok and same

        # Fallback: centroid-M (update_M_fn) is unsupported on GPU -> must fall back to CPU
        Mcen, etc, mdc, ccc, cnc = build_centroid_similarity_matrix(Xc, yc, allc, tau=-0.5)
        upd = (lambda yi, xt, Mb, ccc=ccc, cnc=cnc, etc=etc:
               update_centroid_M_for_candidate(ccc, cnc, etc, yi, xt, Mb))
        fb = dict(exchangeable=True, yhat_mode="ncm", return_sets=True, allow_nonexchangeable=True,
                  update_M_fn=upd, class_centroids=ccc, class_counts=cnc, effective_tau=etc)
        rc = run_fcp_with_mscs(Xc, yc, Xt, yt, allc, ncm, 0.1, 0.05, Mcen, device="cpu", **fb)
        rg = run_fcp_with_mscs(Xc, yc, Xt, yt, allc, ncm, 0.1, 0.05, Mcen, device="cuda", **fb)  # falls back
        same = all(set(a) == set(b) for a, b in zip(_sets(rc), _sets(rg)))
        print(f"  cal={cal} centroid-M (GPU->CPU fallback): parity={same}")
        ok = ok and same

    # --- prototype_softmax + softmax-native prototype-cosine M (GPU path) ---
    # GPU prototype path requires every candidate class present in cal -> balanced
    # cal. Checks the internal prototype-M_aug equals the CPU update_prototype_M.
    print("prototype_softmax + prototype-cosine M (similarity='prototype'):")
    for cal_per, test_per, lam in [(6, 8, 0.1), (10, 6, 0.3)]:
        Xp, yp = gmm(K, cal_per + test_per + 4, d, seed=11)
        ci, ti, rngp = [], [], np.random.default_rng(2)
        for c in range(K):
            ix = rngp.permutation(np.where(yp == c)[0])
            ci.append(ix[:cal_per]); ti.append(ix[cal_per:cal_per + test_per])
        ci, ti = np.concatenate(ci), np.concatenate(ti)
        Xc, yc, Xt, yt = Xp[ci], yp[ci], Xp[ti], yp[ti]
        Mp, csum, cnt, Pn = build_prototype_similarity_matrix(Xc, yc, allc, cosine=True)
        upd = (lambda yi, xt, Mb, csum=csum, cnt=cnt, Pn=Pn:
               update_prototype_M_for_candidate(csum, cnt, Pn, True, yi, xt, Mb))
        cm = dict(exchangeable=True, yhat_mode="ncm", return_sets=True,
                  temperature=0.1, logit="cosine", similarity="prototype", update_M_fn=upd)
        rc = run_fcp_with_mscs(Xc, yc, Xt, yt, allc, "prototype_softmax", 0.1, lam, Mp, device="cpu", **cm)
        rg = run_fcp_with_mscs(Xc, yc, Xt, yt, allc, "prototype_softmax", 0.1, lam, Mp, device="cuda", **cm)
        same = all(set(a) == set(b) for a, b in zip(_sets(rc), _sets(rg)))
        print(f"  cal={cal_per*K} lam={lam}: prototype-M parity={same} "
              f"cov {rc[0]:.4f}/{rg[0]:.4f} sz {rc[1]:.3f}/{rg[1]:.3f}")
        ok = ok and same

    print("\n" + ("PASS" if ok else "FAIL") + "  MS-CS GPU parity")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
