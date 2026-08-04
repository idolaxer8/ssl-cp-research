"""
Exchangeability-preserving feature pipeline for Full Conformal Prediction.

Core principle
--------------
FCP's coverage guarantee needs every nonconformity score to be a permutation-
symmetric function of the calibration+test bag. A data-dependent transform that
is fit on the *calibration* set (e.g. the pooled-within-class whitening baked
into GeodesicTopKMeanNCM) breaks that symmetry and under-/over-covers at small
cal. A transform fit on an INDEPENDENT unlabeled pool (disjoint from cal/test)
is a *fixed* map w.r.t. the bag, so it preserves exchangeability exactly.

This module puts every unlabeled-pool-fit step in one place:

    UnlabeledTransform.fit(X_unlabeled)
        - optional PCA            (fit on unlabeled)
        - within-cluster whitening (pooled var around k-means centroids; the
          unlabeled pool has no labels, so cluster pseudo-labels stand in for
          classes -- clusters ~ classes give the discriminative whitening that
          the cal-fit version provided, now exchangeable)
        - k-means clustering       (reused to build the MS-CS similarity matrix)
    UnlabeledTransform.transform(X) -> PCA + whitening applied to cal/test

Use with a `whiten=False` NCM (e.g. create_ncm("unwhitened_topk_mean")): the NCM
adds the per-point L2 normalisation + geodesic top-k, so
    unwhitened_NCM( UnlabeledTransform.transform(x) )
reproduces the whitened-geodesic score, but with whitening sourced from the
unlabeled pool -> fully exchangeable.

IdentityTransform is the no-unlabeled fallback: pass-through features, so the
pipeline degrades to plain (unwhitened) FCP -- still fully exchangeable, just
without the whitening/PCA/MS-CS efficiency gains.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class IdentityTransform:
    """No-op transform for the no-unlabeled-pool fallback. Fully exchangeable
    (it touches neither cal nor test in a data-dependent way)."""

    has_clusters = False

    def fit(self, X_unlabeled=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)

    def __repr__(self):
        return "IdentityTransform()"


class UnlabeledTransform:
    """Projection + whitening + k-means, all fit on the unlabeled pool.

    Args:
        pca_dim:     target projection dimension (None or >= D disables it).
        whiten:      'cluster' (pooled within-k-means-cluster per-dim variance,
                     default), 'global' (total per-dim variance), 'lw_global' /
                     'lw_cluster' (FULL-matrix ZCA whitening from a Ledoit-Wolf
                     shrunk covariance -- the continuous, truncation-free
                     alternative to PCA; use with pca_dim=None),
                     'lw_cluster_soft' (PER-CLUSTER LW ZCA whiteners blended by
                     soft k-means responsibilities -- a local-metric field; the
                     k-means/Ledoit-Wolf instantiation of MPPCA (Tipping &
                     Bishop 1999), cluster-conditional-Mahalanobis precedent on
                     SSL features: SSD, Sehwag et al. ICLR 2021), or None.
        n_clusters:  k-means clusters (whitening pseudo-labels + MS-CS). Coarse
                     (20) -> looser whitening; fine (~n_classes) -> tighter.
        reg:         whitening regularisation floor (matches GeodesicTopKMeanNCM).
        random_state: seed for PCA / k-means (the pool is fixed, so this is fixed).
        projection:  'pca' (default), 'random' (Gaussian Johnson-Lindenstrauss
                     matrix -- data-INDEPENDENT control arm; only the pool mean
                     is used, for centering parity with PCA), 'pca_tail' (DROP
                     the top pca_dim principal directions, keep the D - pca_dim
                     tail -- the mirror image of PCA truncation, for the
                     signal-in-the-tail probe on collapsed-spectrum data),
                     'lpp' (Locality Preserving Projection, He & Niyogi NIPS
                     2003 -- linear approximation of Laplacian eigenmaps: the
                     pca_dim generalized eigenvectors of (Xc^T L Xc) f =
                     lam (Xc^T D Xc) f with smallest lam, L/D from a binary
                     symmetrized pool kNN graph. Variance-BLIND reduction:
                     keeps low-variance directions in which graph neighbors
                     stay close, drops high-variance directions that scatter
                     them -- built for the collapsed-spectrum regime),
                     'center' (subtract the pool mean only, no reduction), or
                     None.
        rp_seed:     seed for the random projection matrix (default random_state).
        pre:         optional pre-stage applied BEFORE the projection; the rest
                     of the pipeline is refit on the pre-transformed pool:
                     'yj'  per-dim Yeo-Johnson marginal Gaussianization (MLE
                           lambda per dim on the pool) + per-dim standardize +
                           row re-L2-normalization (Distribution Calibration
                           lineage, Yang et al. ICLR 2021, arXiv 2101.06395);
                     'qe'  alpha-query-expansion pool-neighbor smoothing
                           (Radenovic, Tolias & Chum, TPAMI 2019): x ->
                           L2norm(x + sum_i s_i^alpha u_i) over the qe_k
                           nearest pool points u_i, s_i = max(cos(x,u_i),0)
                           -- label-free local manifold denoising;
                     None (default).
        qe_k, qe_alpha: neighbor count / similarity power for pre='qe'
                     (retrieval-canonical 10 / 3.0).
        qe_mode:     which half of the qe coupling is active (ablation dial,
                     docs/pool_repr_menu_plan.md sec 7):
                     'both' (default)  smooth the pool before downstream fits
                                       AND smooth cal/test at transform time;
                     'fit_only'        downstream stages fit on the smoothed
                                       pool, cal/test enter RAW ("denoised
                                       covariance metric" arm);
                     'apply_only'      cal/test smoothed, downstream stages
                                       fit on the RAW pool ("input denoising
                                       only" arm).
        lpp_graph_k: pool kNN-graph degree for projection='lpp' (binary,
                     symmetrized; default 15).

    Every variant remains a function of the unlabeled pool alone (the JL matrix
    is not even that -- a fixed random matrix), so Proposition 2 (theory.md sec 2)
    applies verbatim: the transform is a fixed map w.r.t. the cal/test bag and
    exchangeability is preserved exactly. The 'qe' pre-stage evaluates each
    point against the FROZEN pool only -- cal and test points never enter each
    other's smoothing -- so it too is a fixed per-point map.
    """

    has_clusters = True

    def __init__(self, pca_dim=None, whiten="cluster", n_clusters=20,
                 reg=1e-4, random_state=42, n_init=10,
                 projection="pca", rp_seed=None,
                 pre=None, qe_k=10, qe_alpha=3.0, lpp_graph_k=15,
                 qe_mode="both"):
        assert whiten in ("cluster", "global", "lw_global", "lw_cluster",
                          "lw_cluster_soft", None)
        assert projection in ("pca", "pca_tail", "random", "lpp", "center", None)
        assert pre in (None, "yj", "qe")
        assert qe_mode in ("both", "fit_only", "apply_only")
        self.pca_dim = pca_dim
        self.whiten = whiten
        self.n_clusters = n_clusters
        self.reg = reg
        self.random_state = random_state
        self.n_init = n_init
        self.projection = projection
        self.rp_seed = random_state if rp_seed is None else rp_seed
        self.pre = pre
        self.qe_k = qe_k
        self.qe_alpha = qe_alpha
        self.qe_mode = qe_mode
        self.lpp_graph_k = lpp_graph_k
        # fitted state
        self.pca_ = None
        self.rp_ = None                  # JL matrix (D x d), 'random' projection
        self.tail_basis_ = None          # (D-r, D) tail eigenbasis, 'pca_tail'
        self.lpp_basis_ = None           # (D, d) LPP eigenbasis, 'lpp'
        self.center_ = None              # pool mean (random/tail/center/lw paths)
        self.kmeans_ = None
        self.inv_std_ = None             # diagonal whitening ('cluster'/'global')
        self.W_ = None                   # full-matrix ZCA whitening ('lw_*')
        self.lw_shrinkage_ = None        # fitted Ledoit-Wolf shrinkage intensity
        self.cluster_centroids_ = None   # in transformed (post-whiten) space
        self.cluster_dists_ = None
        self.yj_lambdas_ = None          # per-dim YJ lambda ('yj' pre)
        self.yj_mean_ = None
        self.yj_std_ = None
        self.qe_pool_ = None             # L2-normed raw pool ('qe' pre)
        self.soft_mus_ = None            # (C, d) cluster means ('lw_cluster_soft')
        self.soft_As_ = None             # (C, d, d) per-cluster ZCA whiteners
        self.soft_tau2_ = None           # responsibility scale (pool statistic)

    # -- fit on the unlabeled pool only -------------------------------------
    def fit(self, X_unlabeled):
        X = np.asarray(X_unlabeled, dtype=np.float64)

        # 0) optional pre-stage; everything downstream is refit on pre(pool)
        if self.pre == "yj":
            X = self._fit_apply_yj(X)
        elif self.pre == "qe":
            self.qe_pool_ = X / (np.linalg.norm(X, axis=1,
                                                keepdims=True) + 1e-12)
            if self.qe_mode in ("both", "fit_only"):
                X = self._qe_smooth(X, fitting_pool=True)

        # 1) projection (fit on unlabeled)
        reduce = self.pca_dim is not None and self.pca_dim < X.shape[1]
        if self.projection == "pca" and reduce:
            self.pca_ = PCA(n_components=self.pca_dim,
                            random_state=self.random_state).fit(X)
            Xp = self.pca_.transform(X)
        elif self.projection == "random" and reduce:
            # Johnson-Lindenstrauss Gaussian projection: data-independent by
            # construction (control for PCA's data-adaptivity). Centering by the
            # pool mean keeps parity with PCA (which centers internally) --
            # this matters downstream because the NCMs L2-normalise.
            self.center_ = X.mean(axis=0)
            rng = np.random.default_rng(self.rp_seed)
            self.rp_ = rng.standard_normal(
                (X.shape[1], self.pca_dim)) / np.sqrt(self.pca_dim)
            Xp = (X - self.center_) @ self.rp_
        elif self.projection == "pca_tail" and reduce:
            # keep the ORTHOGONAL COMPLEMENT of the top pca_dim principal
            # directions -- the mirror of PCA truncation. Signal-in-the-tail
            # probe: on collapsed-spectrum data the class signal survives this;
            # on separable data it should crater.
            full = PCA(n_components=min(X.shape) if min(X.shape) < X.shape[1]
                       else X.shape[1],
                       random_state=self.random_state).fit(X)
            self.center_ = full.mean_
            self.tail_basis_ = full.components_[self.pca_dim:]
            Xp = (X - self.center_) @ self.tail_basis_.T
        elif self.projection == "lpp" and reduce:
            # Locality Preserving Projection (He & Niyogi 2003): generalized
            # eigenvectors of the pool kNN-graph Laplacian quadratic forms.
            # Graph built on COSINE neighbors of the (pre-transformed) pool;
            # eigenproblem on centered features.
            self.center_ = X.mean(axis=0)
            self.lpp_basis_ = self._fit_lpp(X, self.center_)
            Xp = (X - self.center_) @ self.lpp_basis_
        elif self.projection == "center" or self.whiten in ("lw_global",
                                                            "lw_cluster",
                                                            "lw_cluster_soft"):
            # full-rank arms: center by the pool mean (PCA parity)
            self.center_ = X.mean(axis=0)
            Xp = X - self.center_
        else:
            Xp = X

        # 2) k-means (whitening pseudo-labels + MS-CS clusters), in PCA space
        self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                              random_state=self.random_state,
                              n_init=self.n_init).fit(Xp)
        labels = self.kmeans_.labels_

        # 3) whitening (fit on unlabeled)
        if self.whiten == "cluster":
            resid = Xp.copy()
            for c in range(self.n_clusters):
                m = labels == c
                if m.any():
                    resid[m] -= Xp[m].mean(axis=0)
            var = (resid ** 2).mean(axis=0)
        elif self.whiten == "global":
            var = Xp.var(axis=0)
        else:
            var = None

        if var is not None:
            adaptive_reg = max(self.reg, 0.01 * float(np.median(var)))
            self.inv_std_ = 1.0 / np.sqrt(var + adaptive_reg)
        else:
            self.inv_std_ = None

        if self.whiten in ("lw_global", "lw_cluster"):
            # Full-matrix ZCA whitening from a Ledoit-Wolf SHRUNK covariance:
            # Sigma_hat = (1-rho)*S + rho*(tr(S)/d)*I, rho chosen analytically
            # (Ledoit & Wolf 2004). Well-conditioned even when per-cluster
            # n < d, so no rank truncation is needed -- the continuous
            # regularisation alternative to hard PCA truncation.
            from sklearn.covariance import LedoitWolf
            if self.whiten == "lw_cluster":
                resid = Xp.copy()
                for c in range(self.n_clusters):
                    m = labels == c
                    if m.any():
                        resid[m] -= Xp[m].mean(axis=0)
            else:
                resid = Xp - Xp.mean(axis=0)
            lw = LedoitWolf(assume_centered=True).fit(resid)
            self.lw_shrinkage_ = float(lw.shrinkage_)
            evals, evecs = np.linalg.eigh(lw.covariance_)
            evals = np.maximum(evals, 1e-12)
            # symmetric (ZCA) inverse square root: rotation-neutral whitening
            self.W_ = evecs @ np.diag(evals ** -0.5) @ evecs.T

        if self.whiten == "lw_cluster_soft":
            # PER-CLUSTER LW ZCA whiteners + soft responsibilities: a local
            # metric field. T(x) = sum_c r_c(x) * A_c (x - mu_c),
            # r_c(x) = softmax_c(-||x - mu_c||^2 / (2 tau^2)),
            # tau^2 = mean squared distance of pool points to their NEAREST
            # cluster mean (pool statistic -- no free knob). LW shrinkage
            # handles n_c < d per cluster; degenerate clusters (<2 members)
            # fall back to the identity metric.
            from sklearn.covariance import LedoitWolf
            d = Xp.shape[1]
            mus, As, shrinks = [], [], []
            for c in range(self.n_clusters):
                m = labels == c
                if m.sum() < 2:
                    mus.append(Xp[m].mean(axis=0) if m.any()
                               else np.zeros(d))
                    As.append(np.eye(d))
                    continue
                mu_c = Xp[m].mean(axis=0)
                lw = LedoitWolf(assume_centered=True).fit(Xp[m] - mu_c)
                shrinks.append(float(lw.shrinkage_))
                evals, evecs = np.linalg.eigh(lw.covariance_)
                evals = np.maximum(evals, 1e-12)
                mus.append(mu_c)
                As.append(evecs @ np.diag(evals ** -0.5) @ evecs.T)
            self.soft_mus_ = np.stack(mus)
            self.soft_As_ = np.stack(As)
            self.lw_shrinkage_ = float(np.mean(shrinks)) if shrinks else None
            d2 = ((Xp ** 2).sum(1, keepdims=True)
                  - 2 * Xp @ self.soft_mus_.T
                  + (self.soft_mus_ ** 2).sum(1)[None, :])   # (N, C)
            self.soft_tau2_ = float(np.maximum(d2, 0).min(axis=1).mean())

        # cluster centroids/dists in the FINAL (post-whiten) transformed space,
        # so MS-CS distances match the space the NCM scores in.
        cen = self.kmeans_.cluster_centers_
        if self.inv_std_ is not None:
            cen = cen * self.inv_std_
        elif self.W_ is not None:
            cen = cen @ self.W_
        elif self.soft_As_ is not None:
            cen = self._soft_whiten(cen)
        self.cluster_centroids_ = cen
        from sklearn.metrics import euclidean_distances
        self.cluster_dists_ = euclidean_distances(cen, cen)

        # Cache the transformed unlabeled pool so downstream MS-CS can build its
        # similarity matrix in the SAME (post-transform) space, reusing the one
        # unlabeled pool rather than re-deriving features.
        if self.inv_std_ is not None:
            self.Xu_transformed_ = Xp * self.inv_std_
        elif self.W_ is not None:
            self.Xu_transformed_ = Xp @ self.W_
        elif self.soft_As_ is not None:
            self.Xu_transformed_ = self._soft_whiten(Xp)
        else:
            self.Xu_transformed_ = Xp
        return self

    # -- apply to cal/test --------------------------------------------------
    def transform(self, X):
        Xp = np.asarray(X, dtype=np.float64)
        if self.pre == "yj":
            Xp = self._apply_yj(Xp)
        elif self.pre == "qe" and self.qe_mode in ("both", "apply_only"):
            Xp = self._qe_smooth(Xp, fitting_pool=False)
        if self.pca_ is not None:
            Xp = self.pca_.transform(Xp)
        elif self.rp_ is not None:
            Xp = (Xp - self.center_) @ self.rp_
        elif self.tail_basis_ is not None:
            Xp = (Xp - self.center_) @ self.tail_basis_.T
        elif self.lpp_basis_ is not None:
            Xp = (Xp - self.center_) @ self.lpp_basis_
        elif self.center_ is not None:
            Xp = Xp - self.center_
        if self.inv_std_ is not None:
            Xp = Xp * self.inv_std_
        elif self.W_ is not None:
            Xp = Xp @ self.W_
        elif self.soft_As_ is not None:
            Xp = self._soft_whiten(Xp)
        return Xp

    # -- pre-stage / projection / whitening helpers (all pool-fit) ----------
    def _fit_apply_yj(self, X):
        """Per-dim Yeo-Johnson MLE on the pool, then per-dim standardize and
        row re-L2-normalize (the NCMs are angular). Returns transformed pool."""
        from scipy import stats
        n_dim = X.shape[1]
        lams = np.ones(n_dim)
        Xt = np.empty_like(X)
        for j in range(n_dim):
            xj = X[:, j]
            if xj.std() < 1e-12:
                Xt[:, j] = xj
            else:
                Xt[:, j], lams[j] = stats.yeojohnson(xj)
        self.yj_lambdas_ = lams
        self.yj_mean_ = Xt.mean(axis=0)
        self.yj_std_ = Xt.std(axis=0) + 1e-12
        Xt = (Xt - self.yj_mean_) / self.yj_std_
        return Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12)

    def _apply_yj(self, X):
        from scipy import stats
        Xt = np.empty_like(X)
        for j in range(X.shape[1]):
            Xt[:, j] = stats.yeojohnson(X[:, j], lmbda=self.yj_lambdas_[j])
        Xt = (Xt - self.yj_mean_) / self.yj_std_
        return Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12)

    def _qe_smooth(self, X, fitting_pool=False):
        """alpha-QE smoothing against the frozen pool:
        T(x) = L2norm( x + sum_{i in NN_k(x)} s_i^alpha * u_i ),
        s_i = max(cos(x, u_i), 0). When smoothing the pool itself
        (fitting_pool=True) each point's own entry is excluded."""
        P = self.qe_pool_                      # (N, D) float64, L2-normed
        k, a = self.qe_k, self.qe_alpha
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        out = np.empty_like(X)
        for i0 in range(0, len(Xn), 1024):
            ch = Xn[i0:i0 + 1024]
            S = ch @ P.T                       # (m, N)
            if fitting_pool:
                rows = np.arange(len(ch))
                S[rows, i0 + rows] = -np.inf   # exclude self
            idx = np.argpartition(-S, k, axis=1)[:, :k]
            s = np.take_along_axis(S, idx, axis=1)
            w = np.clip(s, 0.0, None) ** a     # (m, k)
            nb = P[idx]                        # (m, k, D)
            v = ch + (w[:, :, None] * nb).sum(axis=1)
            out[i0:i0 + 1024] = v / (np.linalg.norm(v, axis=1,
                                                    keepdims=True) + 1e-12)
        return out

    def _fit_lpp(self, X, mu):
        """LPP eigenbasis from the pool: binary symmetrized cosine kNN graph
        (degree g = lpp_graph_k), then the pca_dim generalized eigenvectors of
        (Xc^T L Xc) f = lam (Xc^T D Xc + eps I) f with SMALLEST lam."""
        import scipy.sparse as sp
        from scipy.linalg import eigh
        N, D = X.shape
        g = self.lpp_graph_k
        Xn = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
              ).astype(np.float32)
        nb_idx = np.empty((N, g), dtype=np.int64)
        for i0 in range(0, N, 1024):
            S = Xn[i0:i0 + 1024] @ Xn.T
            rows = np.arange(S.shape[0])
            S[rows, i0 + rows] = -np.inf       # exclude self
            nb_idx[i0:i0 + 1024] = np.argpartition(-S, g, axis=1)[:, :g]
        rows = np.repeat(np.arange(N), g)
        W = sp.coo_matrix((np.ones(N * g), (rows, nb_idx.ravel())),
                          shape=(N, N)).tocsr()
        W = W.maximum(W.T)                     # symmetrize (binary)
        deg = np.asarray(W.sum(axis=1)).ravel()
        Xc = X - mu
        S1 = Xc.T @ (Xc * deg[:, None])        # Xc^T D Xc
        S2 = Xc.T @ (W @ Xc)                   # Xc^T W Xc
        A_mat = S1 - S2                        # Xc^T L Xc
        A_mat = (A_mat + A_mat.T) / 2
        B_mat = (S1 + S1.T) / 2
        B_mat += (1e-4 * np.trace(B_mat) / D) * np.eye(D)
        evals, evecs = eigh(A_mat, B_mat)
        return evecs[:, :self.pca_dim]         # (D, d), ascending eigenvalues

    def _soft_whiten(self, Xp):
        """T(x) = sum_c r_c(x) A_c (x - mu_c), r = softmax(-d2 / (2 tau2))."""
        Xp = np.asarray(Xp, dtype=np.float64)
        out = np.zeros_like(Xp)
        d2 = ((Xp ** 2).sum(1, keepdims=True)
              - 2 * Xp @ self.soft_mus_.T
              + (self.soft_mus_ ** 2).sum(1)[None, :])
        logits = -d2 / (2 * self.soft_tau2_)
        logits -= logits.max(axis=1, keepdims=True)
        r = np.exp(logits)
        r /= r.sum(axis=1, keepdims=True)
        for c in range(len(self.soft_mus_)):
            out += r[:, c:c + 1] * ((Xp - self.soft_mus_[c]) @ self.soft_As_[c])
        return out

    def cluster_of(self, X_transformed):
        """Nearest-cluster index for already-transformed points (post-whiten
        space). Used by MS-CS to map class centroids -> clusters."""
        Xt = np.asarray(X_transformed, dtype=np.float64)
        # distances to whitened centroids
        d = ((Xt[:, None, :] - self.cluster_centroids_[None, :, :]) ** 2).sum(-1)
        return np.argmin(d, axis=1)

    def __repr__(self):
        proj = "" if self.projection == "pca" else f", projection={self.projection!r}"
        lw = (f", lw_shrinkage={self.lw_shrinkage_:.4f}"
              if self.lw_shrinkage_ is not None else "")
        pre = "" if self.pre is None else f", pre={self.pre!r}"
        if self.pre == "qe" and self.qe_mode != "both":
            pre += f", qe_mode={self.qe_mode!r}"
        return (f"UnlabeledTransform(pca_dim={self.pca_dim}, whiten={self.whiten!r}, "
                f"n_clusters={self.n_clusters}{proj}{pre}{lw})")


def make_transform(unlabeled=None, pca_dim=None, whiten="cluster",
                   n_clusters=20, **kw):
    """Factory: UnlabeledTransform fit on `unlabeled` if provided, else the
    fully-exchangeable IdentityTransform fallback."""
    if unlabeled is None:
        return IdentityTransform().fit()
    return UnlabeledTransform(pca_dim=pca_dim, whiten=whiten,
                              n_clusters=n_clusters, **kw).fit(unlabeled)
