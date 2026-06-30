"""
GPU / device-agnostic fast path for the MS-CS FCP penalty (`run_fcp_with_mscs`).

The CPU `run_fcp_with_mscs` loops over n_test x K candidates in Python; this
module vectorises the whole thing in torch (set-parity with the CPU path, ~10-30x
faster on GPU). Supported config (the canonical recommended pipeline):

  * NCM: a GeodesicTopKMeanNCM with topk_same=True, topk_other=True,
    numerator_only=False  (geodesic_topk_mean / unwhitened_topk_mean).
  * yhat_mode = "ncm"  (variant-A argmax top-k similarity).
  * similarity: cluster M with the built-in centroid-shift update
    (update_M_fn=None), exchangeable=True  -- OR the frozen penalty
    (exchangeable=False).

Anything else (centroid-M via update_M_fn, yhat_mode="1nn", asym NCM, other NCMs)
raises MSCSGpuUnsupported so the caller can fall back to the CPU path.

Math mirrors, line-for-line, GeodesicTopKMeanNCM.score_x /
updated_calibration_scores_for / predict_class[_augmented_cal] and
update_M_for_candidate. Two feature spaces are kept distinct (as on CPU):
the NCM scores/yhat live in the WHITENED+L2 space (X_cal_wn); the cluster-M
centroid shift lives in the RAW feature space (X_cal as passed in).
"""
import numpy as np
import torch


class MSCSGpuUnsupported(Exception):
    """Raised when the requested MS-CS config isn't covered by the torch path."""


def _cache(ncm, dev, all_classes):
    """Move the calibrated NCM's cal-side lookups to `dev` once, laid out over the
    FULL candidate set `all_classes` (K rows). Classes absent from cal get the
    missing-class layout: all-False same_mask row (-> n_same 0) and ct_sum/keff 0
    (-> augmented yhat mean = sim, matching predict_class_augmented_cal's 'absent'
    branch). Indices are M positions (class_to_idx), not raw labels."""
    classes = np.asarray(all_classes)
    if getattr(ncm, "_mscs_cache", None) is not None \
       and ncm._mscs_cache["dev"] == dev and ncm._mscs_cache["K"] == len(classes):
        return ncm._mscs_cache
    ncm._ensure_cal_yhat()
    def f(a, dt=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=dt)
    K = len(classes); n_cal = len(ncm.y_cal)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    y_cal_t = f(ncm.y_cal, torch.long)
    classes_t = f(classes, torch.long)
    same_mask = (y_cal_t.unsqueeze(0) == classes_t.unsqueeze(1))    # (K, n_cal)
    n_same = same_mask.sum(1)

    # per-class cal top-k for yhat, mapped present-cols -> all_classes rows
    ct_sum = np.zeros((K, n_cal)); ct_kth = np.full((K, n_cal), -1.0); ct_keff = np.zeros((K, n_cal))
    for present_col, c in enumerate(ncm._yh_classes):
        pos = class_to_idx[int(c)]
        ct_sum[pos] = ncm.cal_topk_sum[:, present_col]
        ct_kth[pos] = ncm.cal_topk_kth[:, present_col]
        ct_keff[pos] = ncm.cal_topk_keff[:, present_col]

    cache = {
        "dev": dev, "K": K, "classes": classes,
        "Xcw": f(ncm.X_cal_wn), "inv_std": f(ncm.inv_std),
        "same_mask": same_mask, "n_same": n_same, "n_other": (n_cal - n_same),
        "sum_same": f(ncm.sum_same_sims).view(1, 1, -1),
        "kth_same": f(ncm.kth_same_sim).view(1, 1, -1),
        "k_same":   f(ncm._k_same_eff).view(1, 1, -1),
        "sum_other": f(ncm.sum_other_sims).view(1, 1, -1),
        "kth_other": f(ncm.kth_other_sim).view(1, 1, -1),
        "k_other":   f(ncm._k_other_eff).view(1, 1, -1),
        "ct_sum": f(ct_sum), "ct_kth": f(ct_kth), "ct_keff": f(ct_keff),
        "ct_best": f(ncm.cal_topk_best),                            # (n_cal,)
        "cal_yhat_col": f([class_to_idx[int(c)] for c in ncm.cal_y_hat], torch.long),
        "cal_row": f([class_to_idx[int(c)] for c in ncm.y_cal], torch.long),
    }
    ncm._mscs_cache = cache
    return cache


def _base_scores(C, Sw, k, eps=1e-8):
    """test_base (B,K), upd_base (B,K,n_cal), mean_s (B,K) -- mirrors score_x /
    updated_calibration_scores_for (with the missing-class fix: an absent class
    has an all -inf same row -> mean 0 -> arccos(0)=pi/2)."""
    neg_inf = float("-inf")
    Sb = Sw.unsqueeze(1)                              # (B,1,n_cal)
    mc = C["same_mask"].unsqueeze(0)                 # (1,K,n_cal)

    def topk_mean(vals, eff_k):     # eff_k = min(n_in_class, k) per candidate (K,)
        top, _ = vals.topk(k=k, dim=-1)
        top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
        return top.sum(-1) / eff_k.clamp_min(1).float().unsqueeze(0)

    mean_s = topk_mean(torch.where(mc, Sb, torch.full_like(Sb, neg_inf)),
                       C["n_same"].clamp(max=k))
    d_same = torch.arccos(mean_s.clamp(-1, 1))
    mean_o = topk_mean(torch.where(~mc, Sb, torch.full_like(Sb, neg_inf)),
                       C["n_other"].clamp(max=k))
    d_other = torch.arccos(mean_o.clamp(-1, 1))
    test_base = d_same / (d_other + eps)

    def upd(sum_v, kth_v, k_v, applies):
        full = (k_v >= k)
        enter = full & (Sb > kth_v)
        grow = ~full
        new_sum = sum_v + torch.where(applies & enter, Sb - kth_v, torch.zeros_like(Sb)) \
                        + torch.where(applies & grow, Sb, torch.zeros_like(Sb))
        new_k = k_v + (applies & grow).float()
        return torch.arccos((new_sum / new_k.clamp_min(1.0)).clamp(-1, 1))

    d_same_u = upd(C["sum_same"], C["kth_same"], C["k_same"], mc)
    d_other_u = upd(C["sum_other"], C["kth_other"], C["k_other"], ~mc)
    upd_base = d_same_u / (d_other_u + eps)
    return test_base, upd_base, mean_s


def run_mscs_torch(cp, X_cal, y_cal, X_test, y_test, all_classes, alpha, lam, M,
                   *, exchangeable, yhat_mode, update_M_fn, similarity="cluster",
                   class_centroids, class_counts, class_to_cluster,
                   cluster_centroids, cluster_dists, effective_tau,
                   device="cuda", batch_size=128, return_sets=False):
    """torch implementation of run_fcp_with_mscs for the supported config.
    Raises MSCSGpuUnsupported otherwise."""
    from conformal_prediction import GeodesicTopKMeanNCM
    ncm = cp.ncm
    if not isinstance(ncm, GeodesicTopKMeanNCM) or not (ncm.topk_same and ncm.topk_other) \
       or ncm.numerator_only:
        raise MSCSGpuUnsupported("needs GeodesicTopKMeanNCM topk_same+topk_other")
    if yhat_mode != "ncm":
        raise MSCSGpuUnsupported("only yhat_mode='ncm'")
    if similarity != "cluster":
        # only the cluster-M centroid-shift update is vectorised for the geodesic
        # NCM; prototype/centroid M -> CPU fallback (update_M_fn handles them there)
        raise MSCSGpuUnsupported(f"geodesic GPU MS-CS supports cluster-M only "
                                 f"(got similarity='{similarity}')")
    if exchangeable and update_M_fn is not None:
        raise MSCSGpuUnsupported("only built-in cluster-M update (update_M_fn=None)")
    if device == "cuda" and not torch.cuda.is_available():
        raise MSCSGpuUnsupported("cuda requested but unavailable")
    dev = torch.device(device)
    C = _cache(ncm, dev, all_classes)
    k = int(ncm.k)
    classes = C["classes"]
    K = len(classes); n_cal = len(X_cal); n_test = len(X_test)

    def f(a, dt=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=dt)

    Mt = f(M)                                            # (K,K)
    Xcw = C["Xcw"]
    # whiten + L2 the test points (same transform the NCM applies internally)
    Xtw = f(X_test) * C["inv_std"]
    Xtw = Xtw / Xtw.norm(dim=1, keepdim=True).clamp_min(1e-10)

    if exchangeable:
        cc = f(class_centroids); cn = f(class_counts)
        clcen = f(cluster_centroids); cld = f(cluster_dists)
        c2c = f(class_to_cluster, torch.long)           # (K,) cluster of each class
        Xtr = f(X_test)                                  # raw test (for centroid shift)
        tau = float(effective_tau)
    arangeK = torch.arange(K, device=dev)

    cal_row = C["cal_row"]                                # (n_cal,)
    cal_yhat_col = C["cal_yhat_col"]                      # (n_cal,)
    ct_best = C["ct_best"]                                # (n_cal,)

    set_sizes = np.empty(n_test, dtype=np.int64)
    covered = np.empty(n_test, dtype=bool)
    pred_sets = [] if return_sets else None
    yt = np.asarray(y_test)

    for s in range(0, n_test, batch_size):
        e = min(s + batch_size, n_test)
        Xb = Xtw[s:e]; B = Xb.shape[0]
        Sw = Xb @ Xcw.t()                                # (B, n_cal)
        test_base, upd_base, mean_s = _base_scores(C, Sw, k)
        yhat_test = mean_s.argmax(dim=1)                 # (B,) == M-col of test yhat

        if not exchangeable:
            # frozen penalty: cal side fixed, test side uses base yhat/M
            cal_pen = lam * (1.0 - Mt[cal_row, cal_yhat_col]).view(1, 1, -1)   # (1,1,n_cal)
            pen_test = lam * (1.0 - Mt[arangeK.unsqueeze(0).expand(B, K),
                                       yhat_test.unsqueeze(1).expand(B, K)])   # (B,K)
        else:
            # ---- augmented cal yhat (B,K,n_cal) ----
            ctk = C["ct_kth"].unsqueeze(0)               # (1,K,n_cal)
            cts = C["ct_sum"].unsqueeze(0)
            cte = C["ct_keff"].unsqueeze(0)
            Sb = Sw.unsqueeze(1)                          # (B,1,n_cal)
            full = (cte >= k)
            enter = full & (Sb > ctk)
            grow = ~full
            new_sum = cts + torch.where(enter, Sb - ctk, torch.zeros_like(Sb)) \
                          + torch.where(grow, Sb, torch.zeros_like(Sb))
            new_keff = cte + grow.float()
            mean_y_aug = new_sum / new_keff.clamp_min(1.0)               # (B,K,n_cal)
            flip = mean_y_aug >= ct_best.view(1, 1, -1)                  # (B,K,n_cal)
            yhat_aug = torch.where(flip, arangeK.view(1, K, 1).expand(B, K, n_cal),
                                   cal_yhat_col.view(1, 1, -1).expand(B, K, n_cal))  # (B,K,n_cal) M-col

            # ---- M_aug row yc (B,K,K): update_M_for_candidate ----
            new_cen = (cn.view(1, K, 1) * cc.view(1, K, -1) + Xtr[s:e].view(B, 1, -1)) \
                      / (cn.view(1, K, 1) + 1.0)                          # (B,K,d)
            d2 = ((new_cen.unsqueeze(2) - clcen.view(1, 1, -1, clcen.shape[1])) ** 2).sum(-1)  # (B,K,nclust)
            new_clu = d2.argmin(dim=-1)                                   # (B,K)
            changed = new_clu != c2c.view(1, K)                          # (B,K)
            same_clu = c2c.view(1, 1, K) == new_clu.unsqueeze(-1)        # (B,K,K)
            dd = cld[new_clu.unsqueeze(-1).expand(B, K, K),
                     c2c.view(1, 1, K).expand(B, K, K)]                  # (B,K,K)
            changed_val = torch.where(same_clu, torch.ones_like(dd),
                                      torch.exp(-dd ** 2 / tau))
            base_row = Mt.unsqueeze(0).expand(B, K, K)                   # M[yc, c]
            M_aug_row = torch.where(changed.unsqueeze(-1), changed_val, base_row)
            diagK = torch.eye(K, device=dev, dtype=torch.bool).unsqueeze(0)
            M_aug_row = torch.where(diagK, torch.ones_like(M_aug_row), M_aug_row)  # diag=1

            # ---- cal penalty (B,K,n_cal): M_aug[cal_row, yhat_aug] ----
            base_pen = Mt[cal_row.view(1, 1, -1).expand(B, K, n_cal), yhat_aug]   # (B,K,n_cal)
            # row==yc : use M_aug_row[b, yc, yhat_aug]  (gather along last dim)
            val_row = torch.gather(M_aug_row, 2, yhat_aug)                        # (B,K,n_cal)
            # col==yc : use M_aug_row[b, yc, cal_row]
            val_col = torch.gather(M_aug_row, 2,
                                   cal_row.view(1, 1, -1).expand(B, K, n_cal))
            mask_row = cal_row.view(1, 1, -1) == arangeK.view(1, K, 1)            # (1,K,n_cal)
            mask_col = yhat_aug == arangeK.view(1, K, 1)                          # (B,K,n_cal)
            M_eff = torch.where(mask_row, val_row, torch.where(mask_col, val_col, base_pen))
            cal_pen = lam * (1.0 - M_eff)                                         # (B,K,n_cal)

            # ---- test penalty (B,K): M_aug_row[b, yc, yhat_test] ----
            pen_test = lam * (1.0 - torch.gather(
                M_aug_row, 2, yhat_test.view(B, 1, 1).expand(B, K, 1)).squeeze(-1))

        final_test = test_base + pen_test                                        # (B,K)
        final_upd = upd_base + cal_pen                                            # (B,K,n_cal)
        n_greater = (final_upd >= final_test.unsqueeze(-1)).sum(dim=-1)           # (B,K)
        pvals = (n_greater.float() + 1.0) / (n_cal + 1.0)
        include = (pvals > alpha).cpu().numpy()                                   # (B,K)
        for bi in range(B):
            idx = np.nonzero(include[bi])[0]
            set_sizes[s + bi] = len(idx)
            yi = int(yt[s + bi])
            covered[s + bi] = yi in set(int(classes[c]) for c in idx)
            if return_sets:
                pred_sets.append([int(classes[c]) for c in idx])

    coverage = float(covered.mean())
    avg_size = float(set_sizes.mean())
    if return_sets:
        return coverage, avg_size, pred_sets
    return coverage, avg_size


def run_prototype_mscs_torch(cp, X_cal, y_cal, X_test, y_test, all_classes, alpha, lam, M,
                             *, exchangeable, yhat_mode, update_M_fn, similarity="cluster",
                             class_centroids, class_counts, class_to_cluster,
                             cluster_centroids, cluster_dists, effective_tau,
                             device="cuda", batch_size=128, return_sets=False):
    """GPU MS-CS for PrototypeSoftmaxNCM (cosine).

    Two exchangeable M-update modes (``similarity``):
      * "cluster"   -- the Fargion k-means cluster-M centroid-shift update (needs
                       the cluster_* artifacts; update_M_fn must be None).
      * "prototype" -- the SOFTMAX-NATIVE cosine-prototype M. M is the Gram of the
                       NCM's own class-mean prototypes, so the augmented row for a
                       candidate (z, k) is just the cosine of the shifted prototype
                       normalize(class_sum_k + z) against every prototype -- computed
                       internally here from P / class_sum (reusing f_test and
                       norm_aug2 already formed for the scores). The CPU fallback's
                       update_M_fn is ignored here; the math is identical.

    Reuses the NCM-independent penalty + p-value blocks; only the conformal scores
    (denominator-swap prototype path), the suspected class y_hat (argmax prototype
    similarity), and the M_aug row are prototype-specific. float64 -> bit-parity
    with the CPU run_fcp_with_mscs(prototype, yhat_mode='ncm'). Raises
    MSCSGpuUnsupported for anything else (dot logits, frozen, centroid-M, missing
    classes) so the caller falls back to the CPU loop."""
    from conformal_prediction import PrototypeSoftmaxNCM
    ncm = cp.ncm
    if not isinstance(ncm, PrototypeSoftmaxNCM):
        raise MSCSGpuUnsupported("needs PrototypeSoftmaxNCM")
    if not ncm.cosine:
        raise MSCSGpuUnsupported("prototype GPU MS-CS is cosine-only (dot exp overflows)")
    if yhat_mode != "ncm":
        raise MSCSGpuUnsupported("prototype GPU MS-CS uses yhat_mode='ncm' (argmax sim)")
    if not exchangeable:
        raise MSCSGpuUnsupported("prototype GPU MS-CS is exchangeable-only")
    if similarity == "cluster":
        if update_M_fn is not None:
            raise MSCSGpuUnsupported("cluster-M GPU path needs update_M_fn=None")
    elif similarity == "prototype":
        pass   # update_M_fn (CPU updater) ignored; M_aug built from prototypes here
    else:
        raise MSCSGpuUnsupported(f"prototype GPU MS-CS: unsupported similarity "
                                 f"'{similarity}' (use 'cluster' or 'prototype')")
    if device == "cuda" and not torch.cuda.is_available():
        raise MSCSGpuUnsupported("cuda requested but unavailable")
    classes = np.asarray(all_classes)
    if not (len(ncm.classes_) == len(classes) and np.array_equal(np.asarray(ncm.classes_), classes)):
        raise MSCSGpuUnsupported("GPU path needs every candidate class present in cal")

    dev = torch.device(device)
    f64 = torch.float64

    def f(a, dt=f64):
        return torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=dt)

    K = len(classes); n_cal = len(X_cal); n_test = len(X_test)
    T = float(ncm._T); eps = float(ncm.eps)

    Z = f(ncm.Z)                                         # (n_cal, d) prepped (L2) cal
    P = f(ncm.P)                                         # (d, K) prototype directions
    class_sum = f(ncm.class_sum)                         # (d, K)
    F_base = f(ncm.F_base)                               # (n_cal, K) cal LOO logits
    ycol = f(ncm.y_col, torch.long)                      # (n_cal,) true-class col (== M col)
    Mt = f(M)                                            # (K, K)
    arangeK = torch.arange(K, device=dev)

    # prototype precomputes (cal-only)
    znorm2 = (Z * Z).sum(1)                              # (n_cal,)
    css_norm2 = (class_sum * class_sum).sum(0)           # (K,)
    E_base = torch.exp(F_base / T)                       # (n_cal, K) bounded (cosine)
    D_base = E_base.sum(1)                               # (n_cal,)
    E_true = E_base.gather(1, ycol.view(n_cal, 1)).squeeze(1)   # (n_cal,)
    ZS = (Z @ class_sum)                                 # (n_cal, K) <z_j, sum_k>
    ZSt = ZS.t().contiguous()                            # (K, n_cal)
    Ebt = E_base.t().contiguous()                        # (K, n_cal)
    # base-consistent LOO y_hat = argmax of the NCM's OWN logits (F_base). Read the
    # numpy-precomputed top-2 of F_base so GPU and CPU use identical values/ties.
    ncm._ensure_cal_yhat()
    t1c = f(ncm._yh_top1_col, torch.long)                # (n_cal,) F_base argmax col
    t1v = f(ncm._yh_top1_val)                            # (n_cal,)
    t2c = f(ncm._yh_top2_col, torch.long)                # (n_cal,) F_base 2nd-best col
    t2v = f(ncm._yh_top2_val)                            # (n_cal,)
    cal_row = ycol                                       # (n_cal,) M row (true class)
    # only column k changes per candidate -> augmented argmax is k when its augmented
    # logit beats the best of the OTHER columns (top-1, or top-2 if k WAS the top-1).
    is_top1_yh = (t1c.view(1, n_cal) == arangeK.view(K, 1))           # (K, n_cal)
    yh_thresh = torch.where(is_top1_yh, t2v.view(1, n_cal), t1v.view(1, n_cal))     # (K, n_cal)
    yh_fallback = torch.where(is_top1_yh, t2c.view(1, n_cal), t1c.view(1, n_cal))   # (K, n_cal)

    if similarity == "cluster":
        # exchangeable cluster-M artifacts (transformed space, NOT L2-normalised)
        cc = f(class_centroids); cn = f(class_counts)
        clcen = f(cluster_centroids); cld = f(cluster_dists)
        c2c = f(class_to_cluster, torch.long); tau = float(effective_tau)
        Xtr_all = f(X_test)                              # transformed test (M-update space)
    else:
        # prototype-M: M_aug[k, j] = <normalize(class_sum_k + z), P_j>. Reuses
        # f_test (= z @ P) and norm_aug2 (= |class_sum_k + z|^2) from the score
        # block; only CSP[k, j] = <class_sum_k, P_j> is new (K x K, computed once).
        CSP = (class_sum.t() @ P).contiguous()           # (K, K)
        diagK_bool = torch.eye(K, device=dev, dtype=torch.bool)
    Xt_norm = f(X_test)                                  # L2 test (prototype space)
    Xt_norm = Xt_norm / Xt_norm.norm(dim=1, keepdim=True).clamp_min(eps)

    set_sizes = np.empty(n_test, dtype=np.int64)
    covered = np.empty(n_test, dtype=bool)
    pred_sets = [] if return_sets else None
    yt = np.asarray(y_test)

    for s in range(0, n_test, batch_size):
        e = min(s + batch_size, n_test)
        Xb = Xt_norm[s:e]; B = Xb.shape[0]
        # ---- prototype base scores ----
        f_test = Xb @ P                                  # (B, K) test logits
        test_base = 1.0 - torch.softmax(f_test / T, dim=1)          # (B, K)
        yhat_test = f_test.argmax(dim=1)                 # (B,) test y_hat M-col
        Zzx = Xb @ Z.t()                                 # (B, n_cal) <z_b, z_j>
        num_ic = ZSt.unsqueeze(0) + Zzx.unsqueeze(1)     # (B, K, n_cal) <z_j, sum_k+z_b>
        csTzx = Xb @ class_sum                           # (B, K) <z_b, sum_k>
        zxn2 = (Xb * Xb).sum(1)                          # (B,)
        norm_aug2 = css_norm2.view(1, K) + 2.0 * csTzx + zxn2.view(B, 1)   # (B, K)
        simY = num_ic / torch.sqrt(norm_aug2).clamp_min(eps).unsqueeze(2)  # (B,K,n_cal) non-LOO
        # member (y_j==k) LOO logit for the conformal score
        memnum = num_ic - znorm2.view(1, 1, n_cal)
        memden2 = norm_aug2.unsqueeze(2) - 2.0 * num_ic + znorm2.view(1, 1, n_cal)
        simLOO = memnum / torch.sqrt(memden2.clamp_min(eps * eps))
        is_mem = (ycol.view(1, 1, n_cal) == arangeK.view(1, K, 1))         # (1,K,n_cal)
        col_logit = torch.where(is_mem, simLOO, simY)    # augmented class-k logit for j
        e_new = torch.exp(col_logit / T)                 # (B,K,n_cal)
        D = D_base.view(1, 1, n_cal) - Ebt.unsqueeze(0) + e_new           # (B,K,n_cal)
        num = torch.where(is_mem, e_new, E_true.view(1, 1, n_cal))
        upd_base = 1.0 - num / D                          # (B,K,n_cal) prototype cal scores
        # ---- augmented cal y_hat = argmax of base logits with column k -> col_logit
        # (consistent with the prototype score's own softmax, leave-one-out) ----
        flip = col_logit >= yh_thresh.view(1, K, n_cal)
        yhat_aug = torch.where(flip, arangeK.view(1, K, 1).expand(B, K, n_cal),
                               yh_fallback.view(1, K, n_cal).expand(B, K, n_cal))  # M-col

        # ---- M_aug row yc (B,K,K): row k = updated similarities of candidate k ----
        if similarity == "cluster":
            # cluster-M centroid-shift update (update_M_for_candidate; NCM-indep.)
            new_cen = (cn.view(1, K, 1) * cc.view(1, K, -1) + Xtr_all[s:e].view(B, 1, -1)) \
                      / (cn.view(1, K, 1) + 1.0)
            d2 = ((new_cen.unsqueeze(2) - clcen.view(1, 1, -1, clcen.shape[1])) ** 2).sum(-1)
            new_clu = d2.argmin(dim=-1)                   # (B,K)
            changed = new_clu != c2c.view(1, K)
            same_clu = c2c.view(1, 1, K) == new_clu.unsqueeze(-1)
            dd = cld[new_clu.unsqueeze(-1).expand(B, K, K), c2c.view(1, 1, K).expand(B, K, K)]
            changed_val = torch.where(same_clu, torch.ones_like(dd), torch.exp(-dd ** 2 / tau))
            base_row = Mt.unsqueeze(0).expand(B, K, K)
            M_aug_row = torch.where(changed.unsqueeze(-1), changed_val, base_row)
            diagK = torch.eye(K, device=dev, dtype=torch.bool).unsqueeze(0)
            M_aug_row = torch.where(diagK, torch.ones_like(M_aug_row), M_aug_row)
        else:
            # prototype-M: candidate (z_b, k) shifts ONLY class k's prototype to
            # normalize(class_sum_k + z_b); row k = cos of that against every proto.
            # <class_sum_k + z_b, P_j> = CSP[k,j] + f_test[b,j]; |class_sum_k+z_b| =
            # sqrt(norm_aug2[b,k]) -- both already in hand. (matches the CPU
            # update_prototype_M_for_candidate; diag forced to 1.)
            M_aug_row = (CSP.unsqueeze(0) + f_test.unsqueeze(1)) \
                / torch.sqrt(norm_aug2).clamp_min(eps).unsqueeze(2)        # (B,K,K)
            M_aug_row = torch.where(diagK_bool.unsqueeze(0),
                                    torch.ones_like(M_aug_row), M_aug_row)

        # ---- cal penalty (B,K,n_cal): M_aug[cal_row, yhat_aug] ----
        base_pen = Mt[cal_row.view(1, 1, -1).expand(B, K, n_cal), yhat_aug]
        val_row = torch.gather(M_aug_row, 2, yhat_aug)
        val_col = torch.gather(M_aug_row, 2, cal_row.view(1, 1, -1).expand(B, K, n_cal))
        mask_row = cal_row.view(1, 1, -1) == arangeK.view(1, K, 1)
        mask_col = yhat_aug == arangeK.view(1, K, 1)
        M_eff = torch.where(mask_row, val_row, torch.where(mask_col, val_col, base_pen))
        cal_pen = lam * (1.0 - M_eff)                     # (B,K,n_cal)
        pen_test = lam * (1.0 - torch.gather(
            M_aug_row, 2, yhat_test.view(B, 1, 1).expand(B, K, 1)).squeeze(-1))   # (B,K)

        final_test = test_base + pen_test                 # (B,K)
        final_upd = upd_base + cal_pen                     # (B,K,n_cal)
        n_greater = (final_upd >= final_test.unsqueeze(-1)).sum(dim=-1)            # (B,K)
        pvals = (n_greater.double() + 1.0) / (n_cal + 1.0)
        include = (pvals > alpha).cpu().numpy()
        for bi in range(B):
            idx = np.nonzero(include[bi])[0]
            set_sizes[s + bi] = len(idx)
            yi = int(yt[s + bi])
            covered[s + bi] = yi in set(int(classes[c]) for c in idx)
            if return_sets:
                pred_sets.append([int(classes[c]) for c in idx])

    coverage = float(covered.mean())
    avg_size = float(set_sizes.mean())
    if return_sets:
        return coverage, avg_size, pred_sets
    return coverage, avg_size
