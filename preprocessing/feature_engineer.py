"""
Feature engineering: dimensionality reduction via PCA and autoencoder.

Applied after normalization, before windowing.
Both methods are optional and controlled via PreprocessingConfig.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

from config.settings import Config
from preprocessing.data_loader import SubjectData

logger = logging.getLogger(__name__)


class _Autoencoder(nn.Module):
    """Symmetric autoencoder for feature compression."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        mid = (input_dim + latent_dim) // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


class FeatureEngineer:
    """
    Dimensionality reduction applied to fused feature vectors.

    Supports PCA and/or autoencoder embeddings.
    Fit on training data, then transform train + test identically.

    Usage:
        eng = FeatureEngineer(config)
        eng.fit(train_features)                # (N, D) concatenated training features
        reduced = eng.transform(features)      # (N, reduced_dim)
    """

    def __init__(self, config: Config):
        self.cfg = config.preprocessing
        self.pca: PCA | None = None
        self.autoencoder: _Autoencoder | None = None
        self._device = "cpu"
        self._fitted = False

    def fit(self, features: np.ndarray, device: str = "cpu") -> "FeatureEngineer":
        """
        Fit reduction model(s) on training features.

        Args:
            features: (N, D) array of fused feature vectors
            device: torch device for autoencoder training
        """
        self._device = device

        if self.cfg.use_pca:
            n_components = min(self.cfg.pca_components, features.shape[1], features.shape[0])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(features)
            explained = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA fitted: {n_components} components, {explained:.1%} variance explained")

        if self.cfg.use_autoencoder:
            input_dim = features.shape[1]
            if self.cfg.use_pca:
                input_dim = self.pca.n_components_
                features = self.pca.transform(features)

            self.autoencoder = _Autoencoder(input_dim, self.cfg.autoencoder_latent_dim)
            self.autoencoder.to(device)
            self._train_autoencoder(features)

        self._fitted = True
        return self

    def _train_autoencoder(self, features: np.ndarray):
        """Train autoencoder on feature array."""
        dataset = TensorDataset(torch.FloatTensor(features))
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.cfg.autoencoder_lr)
        criterion = nn.MSELoss()

        self.autoencoder.train()
        for epoch in range(self.cfg.autoencoder_epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self._device)
                recon = self.autoencoder(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            if (epoch + 1) % 10 == 0:
                avg = total_loss / len(features)
                logger.info(f"Autoencoder epoch {epoch + 1}/{self.cfg.autoencoder_epochs}, loss={avg:.6f}")
        self.autoencoder.eval()

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features through fitted reduction pipeline.

        Args:
            features: (N, D) raw fused features

        Returns:
            (N, reduced_dim) reduced features
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        result = features.copy()

        if self.cfg.use_pca and self.pca is not None:
            result = self.pca.transform(result)

        if self.cfg.use_autoencoder and self.autoencoder is not None:
            with torch.no_grad():
                tensor = torch.FloatTensor(result).to(self._device)
                result = self.autoencoder.encode(tensor).cpu().numpy()

        return result.astype(np.float32)

    def get_output_dim(self, input_dim: int) -> int:
        """Compute output dimension given input dimension."""
        if self.cfg.use_autoencoder:
            return self.cfg.autoencoder_latent_dim
        if self.cfg.use_pca:
            return min(self.cfg.pca_components, input_dim)
        return input_dim
