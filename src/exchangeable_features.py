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
                     alternative to PCA; use with pca_dim=None), or None.
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
                     'center' (subtract the pool mean only, no reduction), or
                     None.
        rp_seed:     seed for the random projection matrix (default random_state).

    Every variant remains a function of the unlabeled pool alone (the JL matrix
    is not even that -- a fixed random matrix), so Proposition 2 (theory.md sec 2)
    applies verbatim: the transform is a fixed map w.r.t. the cal/test bag and
    exchangeability is preserved exactly.
    """

    has_clusters = True

    def __init__(self, pca_dim=None, whiten="cluster", n_clusters=20,
                 reg=1e-4, random_state=42, n_init=10,
                 projection="pca", rp_seed=None, spectral_filter=None):
        assert whiten in ("cluster", "global", "lw_global", "lw_cluster", None)
        assert projection in ("pca", "pca_tail", "random", "center", "spectral",
                              None)
        if projection == "spectral":
            assert spectral_filter is not None, \
                "projection='spectral' needs spectral_filter={'mu','V','s'}"
        self.spectral_filter = spectral_filter
        self.pca_dim = pca_dim
        self.whiten = whiten
        self.n_clusters = n_clusters
        self.reg = reg
        self.random_state = random_state
        self.n_init = n_init
        self.projection = projection
        self.rp_seed = random_state if rp_seed is None else rp_seed
        # fitted state
        self.pca_ = None
        self.rp_ = None                  # JL matrix (D x d), 'random' projection
        self.tail_basis_ = None          # (D-r, D) tail eigenbasis, 'pca_tail'
        self.V_ = None                   # (D, D) eigenbasis columns, 'spectral'
        self.s_ = None                   # (D,) learned per-dim scales, 'spectral'
        self.center_ = None              # pool mean (random/tail/center/lw paths)
        self.kmeans_ = None
        self.inv_std_ = None             # diagonal whitening ('cluster'/'global')
        self.W_ = None                   # full-matrix ZCA whitening ('lw_*')
        self.lw_shrinkage_ = None        # fitted Ledoit-Wolf shrinkage intensity
        self.cluster_centroids_ = None   # in transformed (post-whiten) space
        self.cluster_dists_ = None

    # -- fit on the unlabeled pool only -------------------------------------
    def fit(self, X_unlabeled):
        X = np.asarray(X_unlabeled, dtype=np.float64)

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
        elif self.projection == "spectral":
            # Learned spectral filter (conformal metric learning): rotate to a
            # pool eigenbasis and rescale eigendirection j by a learned
            # s_j >= 0 (soft truncation + spectral reweighting in one map).
            # mu/V/s are all functions of the pool alone, so Prop 2 applies to
            # the composite exactly like the other pool-fit arms. Must precede
            # the center/lw catch-all so spectral + lw_cluster projects first.
            f = self.spectral_filter
            self.center_ = np.asarray(f["mu"], dtype=np.float64)
            self.V_ = np.asarray(f["V"], dtype=np.float64)
            self.s_ = np.asarray(f["s"], dtype=np.float64)
            Xp = ((X - self.center_) @ self.V_) * self.s_
        elif self.projection == "center" or self.whiten in ("lw_global",
                                                            "lw_cluster"):
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

        # cluster centroids/dists in the FINAL (post-whiten) transformed space,
        # so MS-CS distances match the space the NCM scores in.
        cen = self.kmeans_.cluster_centers_
        if self.inv_std_ is not None:
            cen = cen * self.inv_std_
        elif self.W_ is not None:
            cen = cen @ self.W_
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
        else:
            self.Xu_transformed_ = Xp
        return self

    # -- apply to cal/test --------------------------------------------------
    def transform(self, X):
        Xp = np.asarray(X, dtype=np.float64)
        if self.pca_ is not None:
            Xp = self.pca_.transform(Xp)
        elif self.rp_ is not None:
            Xp = (Xp - self.center_) @ self.rp_
        elif self.tail_basis_ is not None:
            Xp = (Xp - self.center_) @ self.tail_basis_.T
        elif self.V_ is not None:
            Xp = ((Xp - self.center_) @ self.V_) * self.s_
        elif self.center_ is not None:
            Xp = Xp - self.center_
        if self.inv_std_ is not None:
            Xp = Xp * self.inv_std_
        elif self.W_ is not None:
            Xp = Xp @ self.W_
        return Xp

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
        spec = ""
        if self.s_ is not None:
            eff = float(self.s_.sum() ** 2 / (self.s_ ** 2).sum())
            spec = f", s_eff_dim={eff:.1f}"
        return (f"UnlabeledTransform(pca_dim={self.pca_dim}, whiten={self.whiten!r}, "
                f"n_clusters={self.n_clusters}{proj}{spec}{lw})")


def make_transform(unlabeled=None, pca_dim=None, whiten="cluster",
                   n_clusters=20, **kw):
    """Factory: UnlabeledTransform fit on `unlabeled` if provided, else the
    fully-exchangeable IdentityTransform fallback."""
    if unlabeled is None:
        return IdentityTransform().fit()
    return UnlabeledTransform(pca_dim=pca_dim, whiten=whiten,
                              n_clusters=n_clusters, **kw).fit(unlabeled)
