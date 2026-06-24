"""Linearity diagnostics for DINOv2 CIFAR-100 embeddings.

Tests three hypotheses raised in the AE-vs-PCA discussion:

1. SVD spectrum / intrinsic dim -- where does the variance live?
2. Nonlinear-AE recon MSE vs PCA recon MSE -- can a nonlinear bottleneck
   actually beat PCA on its own objective?
3. Linear-AE recon MSE vs PCA -- sanity that AE training converges.

Run from repo root:
    python src/linearity_diagnostics.py
"""


import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

sys.path.insert(0, "src")
from autoencoder_utils import EmbeddingAutoencoder


class _LinearAE(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.decoder = nn.Linear(bottleneck_dim, input_dim, bias=False)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_linear_ae(X, bottleneck_dim, epochs=200, lr=1e-2, seed=42):
    """Train a strictly-linear AE (no nonlinearities, no batchnorm)."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    mu = X_t.mean(0, keepdim=True)
    X_c = X_t - mu

    model = _LinearAE(X.shape[1], bottleneck_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        recon = model(X_c)
        loss = ((recon - X_c) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse = ((model(X_c) - X_c) ** 2).mean().item()
    return mse


def pca_reconstruction_mse(X, k):
    Xc = X - X.mean(0, keepdims=True)
    pca = PCA(n_components=k, random_state=42).fit(Xc)
    recon = pca.inverse_transform(pca.transform(Xc))
    return float(((recon - Xc) ** 2).mean())


def nonlinear_ae_reconstruction_mse(X, bottleneck_dim, seed=42):
    """Use the production AE (1-hidden-layer MLP with ReLU) and compute
    reconstruction MSE on centered data, matching PCA's setup."""
    Xc = (X - X.mean(0, keepdims=True)).astype(np.float32)
    ae = EmbeddingAutoencoder(bottleneck_dim=bottleneck_dim, hidden_dim=512,
                               epochs=200, patience=20, seed=seed)
    ae.fit(Xc)
    return ae.reconstruction_error(Xc)


def twonn_intrinsic_dim(X, sample=2000, seed=42):
    """Facco et al. (2017) TwoNN estimator. Robust, parameter-free."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X[idx]
    # Pairwise distances
    from scipy.spatial.distance import cdist
    D = cdist(Xs, Xs)
    np.fill_diagonal(D, np.inf)
    r1 = np.partition(D, 0, axis=1)[:, 0]
    # second nearest
    D2 = D.copy()
    for i in range(len(Xs)):
        D2[i, np.argmin(D[i])] = np.inf
    r2 = np.partition(D2, 0, axis=1)[:, 0]
    mu = r2 / r1
    mu = mu[np.isfinite(mu) & (mu > 1)]
    # CDF fit: -log(1 - F) = d * log(mu)
    mu_sorted = np.sort(mu)
    F = np.arange(1, len(mu_sorted) + 1) / (len(mu_sorted) + 1)
    y = -np.log(1 - F)
    x = np.log(mu_sorted)
    # Linear regression through origin
    d_hat = float(np.sum(x * y) / np.sum(x * x))
    return d_hat


def participation_ratio(eigvals):
    """Effective dimensionality from spectrum: (sum lam)^2 / sum lam^2."""
    s = np.sum(eigvals)
    return float(s * s / np.sum(eigvals ** 2))


def main():
    print("=" * 70)
    print("Linearity diagnostics on DINOv2 CIFAR-100 embeddings (10K unlabeled)")
    print("=" * 70)

    data = torch.load("output/embeddings_cifar100_unlabeled.pt", weights_only=True)
    X = data["embeddings"].numpy().astype(np.float64)
    print(f"X shape: {X.shape}")

    Xc = X - X.mean(0, keepdims=True)
    total_var = float((Xc ** 2).mean())  # MSE if we reconstruct as the mean

    # -------------------- 1. SVD spectrum --------------------
    print("\n--- SVD spectrum ---")
    t0 = time.time()
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    eigvals = S ** 2 / (len(Xc) - 1)
    var_explained = np.cumsum(eigvals) / np.sum(eigvals)
    print(f"  SVD: {time.time()-t0:.1f}s, top-5 eigvals: {eigvals[:5].round(2)}")
    for k in [8, 16, 32, 64, 128, 256, 512]:
        print(f"  d={k:4d}: cum variance explained = {var_explained[k-1]*100:.2f}%")
    pr = participation_ratio(eigvals)
    print(f"  Participation ratio (effective rank): {pr:.1f}")
    # find knee at 90, 95, 99
    for thresh in [0.90, 0.95, 0.99]:
        d = int(np.searchsorted(var_explained, thresh)) + 1
        print(f"  d for {thresh*100:.0f}% variance: {d}")

    # -------------------- 2. Intrinsic dim (TwoNN) --------------------
    print("\n--- TwoNN intrinsic dimension ---")
    t0 = time.time()
    d_int = twonn_intrinsic_dim(X, sample=2000)
    print(f"  TwoNN d_intrinsic = {d_int:.2f}  (time {time.time()-t0:.1f}s)")

    # -------------------- 3. Reconstruction MSE comparison --------------------
    print("\n--- Reconstruction MSE @ bottleneck d ---")
    print(f"  Baseline (constant mean): total variance = {total_var:.6f}")
    print(f"  {'d':>5} | {'PCA':>12} | {'LinearAE':>12} | {'NonlinearAE':>14} | {'NL/PCA':>8}")
    for d in [32, 128]:
        mse_pca = pca_reconstruction_mse(X, d)
        mse_lin = train_linear_ae(X, d, epochs=300)
        mse_nl = nonlinear_ae_reconstruction_mse(X, d)
        ratio = mse_nl / mse_pca
        print(f"  {d:>5d} | {mse_pca:>12.6f} | {mse_lin:>12.6f} | {mse_nl:>14.6f} | {ratio:>7.3f}x")

    print("\nInterpretation:")
    print("  - If LinearAE MSE ~= PCA MSE: AE optimization is healthy (expected).")
    print("  - If NonlinearAE MSE < PCA MSE: nonlinear structure EXISTS at the")
    print("    reconstruction level, but downstream FCP didn't exploit it (NCM-alignment).")
    print("  - If NonlinearAE MSE ~= PCA MSE: manifold is genuinely near-linear.")
    print("  - Participation ratio + 99% knee tell you the effective intrinsic dim.")


if __name__ == "__main__":
    main()
