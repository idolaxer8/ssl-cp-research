"""Unit tests for the transform-control arms (JL random projection +
Ledoit-Wolf full-matrix whitening) added to exchangeable_features.py.

Run: python tests/test_transform_controls.py   (or pytest)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from exchangeable_features import UnlabeledTransform


def _synth(n=2000, d=60, seed=0):
    """Anisotropic Gaussian pool (spiked covariance)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    scales = np.concatenate([np.full(5, 8.0), np.ones(d - 5)])
    X = rng.standard_normal((n, d)) * scales @ A.T / np.sqrt(d)
    return X + rng.standard_normal(d)  # nonzero mean


def test_backcompat_default_is_pca():
    X = _synth()
    tf = UnlabeledTransform(pca_dim=16, whiten="cluster", n_clusters=8).fit(X)
    assert tf.pca_ is not None and tf.rp_ is None and tf.W_ is None
    Z = tf.transform(X)
    assert Z.shape == (len(X), 16)
    # pool cache is in the same space as transform()
    assert np.allclose(Z, tf.Xu_transformed_, atol=1e-10)


def test_random_projection_jl():
    X = _synth(d=300)
    tf = UnlabeledTransform(pca_dim=64, whiten=None, projection="random",
                            n_clusters=8, rp_seed=7).fit(X)
    assert tf.rp_ is not None and tf.pca_ is None
    Z = tf.transform(X[:200])
    assert Z.shape == (200, 64)
    # JL: pairwise distances approximately preserved (loose bound at d=64)
    from sklearn.metrics import euclidean_distances
    D0 = euclidean_distances(X[:200] - X.mean(0))
    D1 = euclidean_distances(Z)
    iu = np.triu_indices(200, 1)
    ratio = D1[iu] / np.maximum(D0[iu], 1e-12)
    assert 0.8 < np.median(ratio) < 1.2, f"median JL ratio {np.median(ratio)}"
    # determinism: same seed -> same matrix
    tf2 = UnlabeledTransform(pca_dim=64, whiten=None, projection="random",
                             n_clusters=8, rp_seed=7).fit(X)
    assert np.allclose(tf.transform(X[:10]), tf2.transform(X[:10]))
    # different seed -> different matrix
    tf3 = UnlabeledTransform(pca_dim=64, whiten=None, projection="random",
                             n_clusters=8, rp_seed=8).fit(X)
    assert not np.allclose(tf.transform(X[:10]), tf3.transform(X[:10]))


def test_rp_plus_cluster_whiten():
    X = _synth(d=300)
    tf = UnlabeledTransform(pca_dim=64, whiten="cluster", projection="random",
                            n_clusters=8).fit(X)
    assert tf.inv_std_ is not None and tf.inv_std_.shape == (64,)
    Z = tf.transform(X)
    assert np.allclose(Z, tf.Xu_transformed_, atol=1e-10)
    assert tf.cluster_centroids_.shape == (8, 64)
    assert tf.cluster_of(Z[:5]).shape == (5,)


def test_lw_global_whitens():
    X = _synth()
    tf = UnlabeledTransform(pca_dim=None, whiten="lw_global", projection=None,
                            n_clusters=8).fit(X)
    assert tf.W_ is not None and tf.inv_std_ is None
    assert 0.0 <= tf.lw_shrinkage_ <= 1.0
    # W is symmetric (ZCA)
    assert np.allclose(tf.W_, tf.W_.T, atol=1e-8)
    Z = tf.transform(X)
    # exact algebraic invariant: W @ Sigma_shrunk @ W == I
    from sklearn.covariance import LedoitWolf
    resid = (X - X.mean(0))
    Sig = LedoitWolf(assume_centered=True).fit(resid - resid.mean(0)).covariance_
    I = tf.W_ @ Sig @ tf.W_
    assert np.allclose(I, np.eye(len(I)), atol=1e-6), \
        f"W Sigma_shrunk W != I (max dev {np.abs(I - np.eye(len(I))).max()})"
    # and the empirical cov of the whitened pool moves much closer to identity
    d = X.shape[1]
    err_raw = np.linalg.norm(np.corrcoef(X, rowvar=False) - np.eye(d)) / np.sqrt(d)
    err_wht = np.linalg.norm(np.cov(Z, rowvar=False) - np.eye(d)) / np.sqrt(d)
    assert err_wht < 0.5 * err_raw, f"whitening didn't decorrelate: {err_wht} vs {err_raw}"
    assert np.allclose(Z, tf.Xu_transformed_, atol=1e-10)


def test_lw_cluster_whitens_within():
    X = _synth()
    tf = UnlabeledTransform(pca_dim=None, whiten="lw_cluster", projection=None,
                            n_clusters=8).fit(X)
    Z = tf.transform(X)
    labels = tf.kmeans_.labels_
    resid = Z.copy()
    for c in range(8):
        m = labels == c
        if m.any():
            resid[m] -= Z[m].mean(axis=0)
    C = np.cov(resid, rowvar=False)
    # looser than the global test: the LW shrinkage bias is larger here (the
    # whitener uses the SHRUNK covariance, deviation ~ rho * eigen-spread)
    err = np.linalg.norm(C - np.eye(C.shape[0])) / np.sqrt(C.shape[0])
    assert err < 0.4, f"within-cluster whitened cov far from identity: {err}"
    assert tf.cluster_centroids_.shape[1] == X.shape[1]


def test_fixed_map_determinism():
    """Same pool -> identical transform (fixed map w.r.t. any future bag)."""
    X = _synth()
    q = _synth(n=50, d=60, seed=99)
    for kw in (dict(pca_dim=16, whiten="cluster"),
               dict(pca_dim=16, whiten="cluster", projection="random"),
               dict(pca_dim=None, whiten="lw_cluster", projection=None)):
        a = UnlabeledTransform(n_clusters=8, **kw).fit(X)
        b = UnlabeledTransform(n_clusters=8, **kw).fit(X)
        assert np.allclose(a.transform(q), b.transform(q), atol=1e-10), kw


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
