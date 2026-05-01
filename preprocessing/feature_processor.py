"""
Feature processing pipeline for SEED-VIG data.

Implements:
    - Sliding window segmentation with configurable size/overlap
    - Normalization (z-score, min-max, robust)
    - Smoothing (moving average, LDS/Kalman)
    - Windowed dataset construction for RL environment
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from config.settings import Config
from preprocessing.data_loader import SubjectData

logger = logging.getLogger(__name__)


@dataclass
class ProcessedSubject:
    """Processed subject data ready for environment consumption."""
    subject_id: int
    features: np.ndarray      # (num_windows, window_size, feature_dim)
    perclos: np.ndarray       # (num_windows, window_size)
    perclos_mean: np.ndarray  # (num_windows,) — mean PERCLOS per window
    feature_dim: int
    num_windows: int


class FeatureProcessor:
    """
    Full feature processing pipeline.

    Pipeline order:
        raw features → smoothing → normalization → windowing → ProcessedSubject

    Usage:
        processor = FeatureProcessor(config)
        processor.fit(train_subjects)           # fit scaler on training data
        processed = processor.transform(subject) # transform single subject
    """

    def __init__(self, config: Config):
        self.cfg = config.preprocessing
        self.scaler = None
        self._fitted = False

    def _create_scaler(self):
        """Create sklearn scaler based on config."""
        if self.cfg.norm_method == "zscore":
            return StandardScaler()
        elif self.cfg.norm_method == "minmax":
            return MinMaxScaler(feature_range=(-1, 1))
        elif self.cfg.norm_method == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown norm method: {self.cfg.norm_method}")

    @staticmethod
    def moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """
        Apply causal moving average along time axis (axis=0).

        Uses cumsum trick for O(n) computation.
        """
        if window <= 1:
            return data
        kernel = np.ones(window) / window
        if data.ndim == 1:
            # Pad beginning with first value to maintain length
            padded = np.concatenate([np.full(window - 1, data[0]), data])
            return np.convolve(padded, kernel, mode="valid")
        else:
            result = np.empty_like(data)
            for col in range(data.shape[1]):
                padded = np.concatenate([np.full(window - 1, data[0, col]), data[:, col]])
                result[:, col] = np.convolve(padded, kernel, mode="valid")
            return result

    @staticmethod
    def lds_smooth(data: np.ndarray, transition_cov: float, observation_cov: float) -> np.ndarray:
        """
        Linear Dynamical System (Kalman) smoothing.

        Forward-backward Kalman smoother assuming identity transition/observation matrices.

        Model:
            x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
            y_t = x_t + v_t,       v_t ~ N(0, R)

        Args:
            data: (T, D) or (T,) observed sequence
            transition_cov: Q — process noise variance
            observation_cov: R — observation noise variance

        Returns:
            Smoothed sequence, same shape as input
        """
        if data.ndim == 1:
            data = data[:, np.newaxis]
            squeeze = True
        else:
            squeeze = False

        T, D = data.shape
        Q = transition_cov
        R = observation_cov

        # Forward pass (Kalman filter)
        x_filt = np.zeros((T, D))
        P_filt = np.zeros(T)
        x_pred = np.zeros((T, D))
        P_pred = np.zeros(T)

        # Initialize
        x_filt[0] = data[0]
        P_filt[0] = R

        for t in range(1, T):
            # Predict
            x_pred[t] = x_filt[t - 1]
            P_pred[t] = P_filt[t - 1] + Q
            # Update
            K = P_pred[t] / (P_pred[t] + R)  # Kalman gain
            x_filt[t] = x_pred[t] + K * (data[t] - x_pred[t])
            P_filt[t] = (1 - K) * P_pred[t]

        # Backward pass (RTS smoother)
        x_smooth = np.zeros((T, D))
        x_smooth[T - 1] = x_filt[T - 1]

        for t in range(T - 2, -1, -1):
            if P_pred[t + 1] > 0:
                L = P_filt[t] / P_pred[t + 1]
            else:
                L = 0.0
            x_smooth[t] = x_filt[t] + L * (x_smooth[t + 1] - x_pred[t + 1])

        return x_smooth.squeeze() if squeeze else x_smooth

    def smooth(self, data: np.ndarray) -> np.ndarray:
        """Apply configured smoothing method."""
        if self.cfg.smoothing_method == "none":
            return data
        elif self.cfg.smoothing_method == "moving_avg":
            return self.moving_average(data, self.cfg.smoothing_window)
        elif self.cfg.smoothing_method == "lds":
            return self.lds_smooth(
                data,
                self.cfg.lds_transition_cov,
                self.cfg.lds_observation_cov,
            )
        else:
            raise ValueError(f"Unknown smoothing method: {self.cfg.smoothing_method}")

    def fit(self, subjects: list[SubjectData]) -> "FeatureProcessor":
        """
        Fit normalization scaler on training subjects.

        Concatenates all training subjects' features to compute global statistics.
        Must be called before transform().
        """
        self.scaler = self._create_scaler()
        all_features = []
        for subj in subjects:
            feat = subj.get_fused_features()
            feat = self.smooth(feat)
            all_features.append(feat)
        combined = np.concatenate(all_features, axis=0)
        self.scaler.fit(combined)
        self._fitted = True
        logger.info(f"Scaler fitted on {combined.shape[0]} samples from {len(subjects)} subjects")
        return self

    def transform(self, subject: SubjectData) -> ProcessedSubject:
        """
        Transform a single subject through the full pipeline.

        Pipeline: smooth → normalize → window
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        features = subject.get_fused_features()  # (T, D)
        perclos = subject.perclos.copy()          # (T,)

        # Smooth features
        features = self.smooth(features)

        # Normalize
        features = self.scaler.transform(features)

        # Sliding window segmentation
        ws = self.cfg.window_size
        stride = self.cfg.window_stride
        T, D = features.shape

        if T < ws:
            raise ValueError(
                f"Subject {subject.subject_id}: {T} samples < window_size {ws}"
            )

        num_windows = (T - ws) // stride + 1
        feat_windows = np.zeros((num_windows, ws, D), dtype=np.float32)
        perc_windows = np.zeros((num_windows, ws), dtype=np.float32)

        for i in range(num_windows):
            start = i * stride
            end = start + ws
            feat_windows[i] = features[start:end]
            perc_windows[i] = perclos[start:end]

        perclos_mean = perc_windows.mean(axis=1)

        return ProcessedSubject(
            subject_id=subject.subject_id,
            features=feat_windows,
            perclos=perc_windows,
            perclos_mean=perclos_mean,
            feature_dim=D,
            num_windows=num_windows,
        )

    def fit_transform(self, subjects: list[SubjectData]) -> list[ProcessedSubject]:
        """Fit on subjects and transform all."""
        self.fit(subjects)
        return [self.transform(s) for s in subjects]
