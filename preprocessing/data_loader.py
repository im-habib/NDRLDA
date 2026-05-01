"""
SEED-VIG dataset loader.

Handles loading .mat files from the SEED-VIG dataset, extracting EEG (DE/PSD),
forehead EEG, EOG features, and PERCLOS ground truth labels.

SEED-VIG structure (per subject .mat file):
    - de_movingAve:  (samples, channels, bands) — differential entropy, moving average smoothed
    - de_LDS:        (samples, channels, bands) — differential entropy, LDS smoothed
    - psd_movingAve: (samples, channels, bands) — power spectral density, moving average smoothed
    - psd_LDS:       (samples, channels, bands) — power spectral density, LDS smoothed
    - eog:           (samples, eog_dim) — EOG features (36-dim)
    - perclos:       (samples,) or (samples, 1) — continuous PERCLOS values [0, 1]

Channels 0-16 = standard EEG, channels 17-20 = forehead EEG.
Bands: delta(0), theta(1), alpha(2), beta(3), gamma(4).
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

from config.settings import Config, DataConfig

logger = logging.getLogger(__name__)


@dataclass
class SubjectData:
    """Container for a single subject's loaded data."""
    subject_id: int
    eeg_features: np.ndarray    # (T, channels, bands)
    eog_features: np.ndarray    # (T, 36)
    perclos: np.ndarray         # (T,)
    num_samples: int

    def get_fused_features(self) -> np.ndarray:
        """
        Fuse EEG and EOG into single feature vector per timestep.

        Returns:
            (T, eeg_channels * bands + eog_dim) array
        """
        T = self.num_samples
        eeg_flat = self.eeg_features.reshape(T, -1)  # (T, channels * bands)
        return np.concatenate([eeg_flat, self.eog_features], axis=1)


class SEEDVIGLoader:
    """
    Loads and manages SEED-VIG dataset.

    Supports subject-wise loading, multimodal fusion, and batch streaming.

    Usage:
        loader = SEEDVIGLoader(config)
        subject_data = loader.load_subject(1)
        all_data = loader.load_all_subjects()
        train, test = loader.split_subjects(test_ids=[1, 5, 10])
    """

    # Common .mat key patterns in SEED-VIG
    _EEG_KEYS = {
        "de_movingAve": "de_movingAve",
        "de_LDS": "de_LDS",
        "psd_movingAve": "psd_movingAve",
        "psd_LDS": "psd_LDS",
    }
    _EOG_KEYS = ("eog", "EOG_feature", "eog_feature", "EOG")
    _PERCLOS_KEYS = ("perclos", "PERCLOS", "label", "labels")

    def __init__(self, config: Config):
        self.cfg = config.data
        self.preproc_cfg = config.preprocessing
        self.data_dir = Path(self.cfg.data_dir)
        self._cache: dict[int, SubjectData] = {}

    def _find_mat_files(self) -> list[Path]:
        """Discover .mat files in data directory."""
        patterns = ["*.mat", "**/*.mat"]
        files = []
        for pat in patterns:
            files.extend(self.data_dir.glob(pat))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(
                f"No .mat files found in {self.data_dir}. "
                "Download SEED-VIG from https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html"
            )
        return files

    def _extract_key(self, mat_data: dict, candidates: tuple | list) -> np.ndarray | None:
        """Try multiple key names to find data in .mat file."""
        for key in candidates:
            if key in mat_data:
                arr = np.asarray(mat_data[key])
                if arr.ndim > 0:
                    return arr.squeeze()
        return None

    def _load_mat(self, filepath: Path, feature_type: str) -> dict:
        """
        Load a single .mat file and extract relevant arrays.

        Args:
            filepath: Path to .mat file
            feature_type: Which EEG feature to use (e.g., "de_LDS")

        Returns:
            Dict with keys: "eeg", "eog", "perclos"
        """
        try:
            mat = sio.loadmat(str(filepath), squeeze_me=False)
        except NotImplementedError:
            # HDF5-based .mat (v7.3) — use h5py
            import h5py
            mat = {}
            with h5py.File(str(filepath), "r") as f:
                for key in f.keys():
                    mat[key] = np.array(f[key])

        # Extract EEG features
        eeg_key = self._EEG_KEYS.get(feature_type, feature_type)
        eeg = self._extract_key(mat, [eeg_key])
        if eeg is None:
            # Try loading any available EEG key
            for k in self._EEG_KEYS.values():
                eeg = self._extract_key(mat, [k])
                if eeg is not None:
                    logger.warning(f"Requested '{feature_type}' not found, using '{k}' instead")
                    break
        if eeg is None:
            raise KeyError(f"No EEG feature keys found in {filepath.name}")

        # Ensure shape (T, channels, bands)
        if eeg.ndim == 2:
            # Might be (T, channels*bands) — reshape
            total = self.cfg.total_eeg_channels * self.cfg.frequency_bands
            if eeg.shape[1] == total:
                eeg = eeg.reshape(-1, self.cfg.total_eeg_channels, self.cfg.frequency_bands)
            else:
                raise ValueError(f"Unexpected EEG shape {eeg.shape} in {filepath.name}")

        # Extract EOG
        eog = self._extract_key(mat, self._EOG_KEYS)
        if eog is None:
            logger.warning(f"No EOG features in {filepath.name}, using zeros")
            eog = np.zeros((eeg.shape[0], self.cfg.eog_dim))
        if eog.ndim == 1:
            eog = eog.reshape(-1, 1)

        # Extract PERCLOS
        perclos = self._extract_key(mat, self._PERCLOS_KEYS)
        if perclos is None:
            raise KeyError(f"No PERCLOS/label found in {filepath.name}")
        perclos = perclos.flatten().astype(np.float64)

        # Align lengths
        min_len = min(eeg.shape[0], eog.shape[0], len(perclos))
        eeg = eeg[:min_len]
        eog = eog[:min_len]
        perclos = perclos[:min_len]

        return {"eeg": eeg, "eog": eog, "perclos": perclos}

    def load_subject(self, subject_id: int, force_reload: bool = False) -> SubjectData:
        """
        Load data for a single subject.

        Args:
            subject_id: 1-indexed subject ID
            force_reload: Bypass cache

        Returns:
            SubjectData instance
        """
        if not force_reload and subject_id in self._cache:
            return self._cache[subject_id]

        mat_files = self._find_mat_files()

        # Match subject to file — SEED-VIG files typically named with subject index
        target_file = None
        for f in mat_files:
            # Match patterns like "1.mat", "sub01.mat", "subject_1.mat", "S01.mat"
            stem = f.stem.lower()
            # Extract numeric part
            nums = [int(s) for s in stem.split("_") if s.isdigit()]
            if not nums:
                nums = [int("".join(c for c in stem if c.isdigit()) or "-1")]
            if subject_id in nums:
                target_file = f
                break

        if target_file is None:
            # Fallback: use positional index
            if subject_id - 1 < len(mat_files):
                target_file = mat_files[subject_id - 1]
            else:
                raise FileNotFoundError(f"Cannot find .mat file for subject {subject_id}")

        logger.info(f"Loading subject {subject_id} from {target_file.name}")
        raw = self._load_mat(target_file, self.cfg.primary_eeg_feature)

        data = SubjectData(
            subject_id=subject_id,
            eeg_features=raw["eeg"].astype(np.float32),
            eog_features=raw["eog"].astype(np.float32),
            perclos=raw["perclos"].astype(np.float32),
            num_samples=len(raw["perclos"]),
        )
        self._cache[subject_id] = data
        return data

    def load_all_subjects(self) -> list[SubjectData]:
        """Load all available subjects."""
        subjects = []
        mat_files = self._find_mat_files()
        for i in range(1, min(self.cfg.num_subjects + 1, len(mat_files) + 1)):
            try:
                subjects.append(self.load_subject(i))
            except (FileNotFoundError, KeyError) as e:
                logger.warning(f"Skipping subject {i}: {e}")
        logger.info(f"Loaded {len(subjects)} subjects")
        return subjects

    def split_subjects(
        self,
        test_ids: list[int] | tuple[int, ...],
    ) -> tuple[list[SubjectData], list[SubjectData]]:
        """
        Subject-independent train/test split.

        Args:
            test_ids: Subject IDs for test set

        Returns:
            (train_subjects, test_subjects)
        """
        all_subjects = self.load_all_subjects()
        test_set = [s for s in all_subjects if s.subject_id in test_ids]
        train_set = [s for s in all_subjects if s.subject_id not in test_ids]
        logger.info(
            f"Split: {len(train_set)} train subjects, {len(test_set)} test subjects"
        )
        return train_set, test_set

    def stream_windows(
        self,
        subject_data: SubjectData,
        window_size: int | None = None,
        stride: int | None = None,
    ):
        """
        Generator yielding sliding windows for batch streaming simulation.

        Yields:
            (features_window, perclos_window) — shapes (W, D) and (W,)
        """
        ws = window_size or self.preproc_cfg.window_size
        st = stride or self.preproc_cfg.window_stride
        features = subject_data.get_fused_features()
        perclos = subject_data.perclos

        for start in range(0, len(features) - ws + 1, st):
            end = start + ws
            yield features[start:end], perclos[start:end]
