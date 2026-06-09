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
    """PCA + within-cluster whitening + k-means, all fit on the unlabeled pool.

    Args:
        pca_dim:     target PCA dimension (None or >= D disables PCA).
        whiten:      'cluster' (pooled within-k-means-cluster variance, default),
                     'global' (total per-dim variance), or None (no whitening).
        n_clusters:  k-means clusters (whitening pseudo-labels + MS-CS). Coarse
                     (20) -> looser whitening; fine (~n_classes) -> tighter.
        reg:         whitening regularisation floor (matches GeodesicTopKMeanNCM).
        random_state: seed for PCA / k-means (the pool is fixed, so this is fixed).
    """

    has_clusters = True

    def __init__(self, pca_dim=None, whiten="cluster", n_clusters=20,
                 reg=1e-4, random_state=42, n_init=10):
        assert whiten in ("cluster", "global", None)
        self.pca_dim = pca_dim
        self.whiten = whiten
        self.n_clusters = n_clusters
        self.reg = reg
        self.random_state = random_state
        self.n_init = n_init
        # fitted state
        self.pca_ = None
        self.kmeans_ = None
        self.inv_std_ = None
        self.cluster_centroids_ = None   # in transformed (post-whiten) space
        self.cluster_dists_ = None

    # -- fit on the unlabeled pool only -------------------------------------
    def fit(self, X_unlabeled):
        X = np.asarray(X_unlabeled, dtype=np.float64)

        # 1) PCA (fit on unlabeled)
        if self.pca_dim is not None and self.pca_dim < X.shape[1]:
            self.pca_ = PCA(n_components=self.pca_dim,
                            random_state=self.random_state).fit(X)
            Xp = self.pca_.transform(X)
        else:
            self.pca_ = None
            Xp = X

        # 2) k-means (whitening pseudo-labels + MS-CS clusters), in PCA space
        self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                              random_state=self.random_state,
                              n_init=self.n_init).fit(Xp)
        labels = self.kmeans_.labels_

        # 3) whitening inv_std (fit on unlabeled)
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

        # cluster centroids/dists in the FINAL (post-whiten) transformed space,
        # so MS-CS distances match the space the NCM scores in.
        cen = self.kmeans_.cluster_centers_
        if self.inv_std_ is not None:
            cen = cen * self.inv_std_
        self.cluster_centroids_ = cen
        from sklearn.metrics import euclidean_distances
        self.cluster_dists_ = euclidean_distances(cen, cen)

        # Cache the transformed unlabeled pool so downstream MS-CS can build its
        # similarity matrix in the SAME (post-transform) space, reusing the one
        # unlabeled pool rather than re-deriving features.
        self.Xu_transformed_ = (Xp * self.inv_std_) if self.inv_std_ is not None else Xp
        return self

    # -- apply to cal/test --------------------------------------------------
    def transform(self, X):
        Xp = np.asarray(X, dtype=np.float64)
        if self.pca_ is not None:
            Xp = self.pca_.transform(Xp)
        if self.inv_std_ is not None:
            Xp = Xp * self.inv_std_
        return Xp

    def cluster_of(self, X_transformed):
        """Nearest-cluster index for already-transformed points (post-whiten
        space). Used by MS-CS to map class centroids -> clusters."""
        Xt = np.asarray(X_transformed, dtype=np.float64)
        # distances to whitened centroids
        d = ((Xt[:, None, :] - self.cluster_centroids_[None, :, :]) ** 2).sum(-1)
        return np.argmin(d, axis=1)

    def __repr__(self):
        return (f"UnlabeledTransform(pca_dim={self.pca_dim}, whiten={self.whiten!r}, "
                f"n_clusters={self.n_clusters})")


def make_transform(unlabeled=None, pca_dim=None, whiten="cluster",
                   n_clusters=20, **kw):
    """Factory: UnlabeledTransform fit on `unlabeled` if provided, else the
    fully-exchangeable IdentityTransform fallback."""
    if unlabeled is None:
        return IdentityTransform().fit()
    return UnlabeledTransform(pca_dim=pca_dim, whiten=whiten,
                              n_clusters=n_clusters, **kw).fit(unlabeled)
