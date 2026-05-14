"""
Autoencoder bottleneck for FCP dimensionality reduction.

Unsupervised (reconstruction loss only) -- preserves exchangeability like PCA,
but can capture nonlinear manifold structure in SSL embeddings.

Usage:
    ae = EmbeddingAutoencoder(bottleneck_dim=128)
    ae.fit(X_unlabeled)              # numpy (N, 768)
    X_reduced = ae.transform(X)      # numpy in, numpy out
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _AEModule(nn.Module):
    """Symmetric 1-hidden-layer MLP autoencoder."""

    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


class EmbeddingAutoencoder:
    """sklearn-compatible wrapper around a PyTorch MLP autoencoder.

    Parameters
    ----------
    bottleneck_dim : int
        Dimensionality of the bottleneck (output of transform).
    hidden_dim : int
        Width of the single hidden layer in encoder/decoder.
    lr : float
        Learning rate for Adam.
    epochs : int
        Maximum training epochs (with early stopping).
    batch_size : int
        Mini-batch size.
    patience : int
        Early stopping patience (epochs without val improvement).
    val_fraction : float
        Fraction of training data held out for validation.
    weight_decay : float
        L2 regularization.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, bottleneck_dim=128, hidden_dim=512, lr=1e-3,
                 epochs=100, batch_size=256, patience=10,
                 val_fraction=0.1, weight_decay=1e-5, seed=42):
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.weight_decay = weight_decay
        self.seed = seed
        self.model_ = None
        self.device_ = None

    def fit(self, X):
        """Train autoencoder on X (numpy array, shape (N, D))."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X.shape[1]

        # Train/val split
        n = len(X)
        n_val = max(1, int(n * self.val_fraction))
        perm = np.random.permutation(n)
        X_train = torch.from_numpy(X[perm[n_val:]].astype(np.float32))
        X_val = torch.from_numpy(X[perm[:n_val]].astype(np.float32))

        train_loader = DataLoader(
            TensorDataset(X_train), batch_size=self.batch_size, shuffle=True,
            drop_last=False)

        self.model_ = _AEModule(input_dim, self.hidden_dim, self.bottleneck_dim).to(self.device_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        t0 = time.time()

        for epoch in range(self.epochs):
            # Train
            self.model_.train()
            train_loss = 0.0
            n_batches = 0
            for (batch,) in train_loader:
                batch = batch.to(self.device_)
                recon = self.model_(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= n_batches

            # Val
            self.model_.eval()
            with torch.no_grad():
                X_val_dev = X_val.to(self.device_)
                val_loss = criterion(self.model_(X_val_dev), X_val_dev).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        # Restore best
        self.model_.load_state_dict(best_state)
        self.model_.eval()

        elapsed = time.time() - t0
        final_epoch = epoch + 1
        print(f"AE fit: {input_dim}d -> {self.bottleneck_dim}d, "
              f"{final_epoch} epochs, train_loss={train_loss:.6f}, "
              f"val_loss={best_val_loss:.6f}, time={elapsed:.1f}s")

        return self

    def transform(self, X):
        """Encode X through the bottleneck. Returns numpy array."""
        self.model_.eval()
        X_t = torch.from_numpy(X.astype(np.float32))
        results = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                batch = X_t[i:i + self.batch_size].to(self.device_)
                results.append(self.model_.encode(batch).cpu())
        return torch.cat(results, dim=0).numpy()

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def reconstruction_error(self, X):
        """Mean squared reconstruction error on X."""
        self.model_.eval()
        X_t = torch.from_numpy(X.astype(np.float32))
        total = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                batch = X_t[i:i + self.batch_size].to(self.device_)
                recon = self.model_(batch)
                total += ((recon - batch) ** 2).sum().item()
                n += batch.shape[0] * batch.shape[1]
        return total / n
