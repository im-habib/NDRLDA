"""
SEED-VIG dataset loader.

Handles loading .mat files from the SEED-VIG dataset, which stores data in
separate folders per modality:
    EEG_Feature_5Bands/  — EEG features: (channels=17, samples=885, bands=5)
    EOG_Feature/         — EOG features: (samples=885, features=36)
    perclos_labels/      — PERCLOS labels: (samples=885, 1)

Each folder contains 23 .mat files (one per experiment), matched by filename.

EEG keys in .mat files:
    de_movingAve, de_LDS, psd_movingAve, psd_LDS

EOG keys in .mat files:
    features_table_ica, features_table_minus, features_table_icav_minh

PERCLOS keys:
    perclos
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
    """Container for a single experiment's loaded data."""
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
    Loads and manages SEED-VIG dataset from its multi-folder structure.

    SEED-VIG layout:
        data_dir/
            EEG_Feature_5Bands/   *.mat  (17, 885, 5) per key
            EOG_Feature/          *.mat  (885, 36) per key
            perclos_labels/       *.mat  (885, 1)

    Usage:
        loader = SEEDVIGLoader(config)
        subject_data = loader.load_subject(1)
        all_data = loader.load_all_subjects()
        train, test = loader.split_subjects(test_ids=[1, 5, 10])
    """

    _EEG_KEYS = ("de_LDS", "de_movingAve", "psd_LDS", "psd_movingAve")
    _EOG_KEYS = ("features_table_ica", "features_table_minus", "features_table_icav_minh")
    _PERCLOS_KEYS = ("perclos", "PERCLOS", "label", "labels")

    def __init__(self, config: Config):
        self.cfg = config.data
        self.preproc_cfg = config.preprocessing
        self.data_dir = Path(self.cfg.data_dir)
        self._cache: dict[int, SubjectData] = {}
        self._experiment_files: list[Path] | None = None

    def _discover_experiments(self) -> list[Path]:
        """Discover experiment .mat files by scanning EEG_Feature_5Bands folder."""
        if self._experiment_files is not None:
            return self._experiment_files

        eeg_dir = self.data_dir / "EEG_Feature_5Bands"
        if not eeg_dir.exists():
            raise FileNotFoundError(
                f"EEG_Feature_5Bands not found in {self.data_dir}. "
                "Download SEED-VIG from https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html"
            )

        files = sorted(eeg_dir.glob("*.mat"))
        if not files:
            raise FileNotFoundError(f"No .mat files in {eeg_dir}")

        self._experiment_files = files
        logger.info(f"Discovered {len(files)} experiments in {eeg_dir}")
        return files

    def _load_mat_safe(self, filepath: Path) -> dict:
        """Load .mat file, handling both v5 and v7.3 formats."""
        try:
            return sio.loadmat(str(filepath), squeeze_me=False)
        except NotImplementedError:
            import h5py
            mat = {}
            with h5py.File(str(filepath), "r") as f:
                for key in f.keys():
                    mat[key] = np.array(f[key])
            return mat

    def _extract_key(self, mat_data: dict, candidates: tuple | list) -> np.ndarray | None:
        """Try multiple key names to find data in .mat file."""
        for key in candidates:
            if key in mat_data:
                arr = np.asarray(mat_data[key])
                if arr.ndim > 0:
                    return arr
        return None

    def _load_eeg(self, filename: str) -> np.ndarray:
        """
        Load EEG features for one experiment.

        Raw shape in .mat: (channels=17, samples=885, bands=5)
        Returns: (samples, channels, bands)
        """
        filepath = self.data_dir / "EEG_Feature_5Bands" / filename
        mat = self._load_mat_safe(filepath)

        # Try configured primary feature first, then fallbacks
        preferred = self.cfg.primary_eeg_feature
        candidates = [preferred] + [k for k in self._EEG_KEYS if k != preferred]
        eeg = self._extract_key(mat, candidates)

        if eeg is None:
            raise KeyError(f"No EEG feature keys found in {filepath.name}. "
                           f"Available: {[k for k in mat if not k.startswith('__')]}")

        eeg = eeg.squeeze()

        # Transpose from (channels, samples, bands) → (samples, channels, bands)
        if eeg.ndim == 3 and eeg.shape[0] == self.cfg.total_eeg_channels:
            eeg = np.transpose(eeg, (1, 0, 2))
        elif eeg.ndim == 3 and eeg.shape[1] == self.cfg.total_eeg_channels:
            # Already (samples, channels, bands) — unlikely but handle it
            pass
        else:
            raise ValueError(f"Unexpected EEG shape {eeg.shape} in {filepath.name}")

        logger.debug(f"EEG loaded: {filepath.name} → {eeg.shape}")
        return eeg

    def _load_eog(self, filename: str) -> np.ndarray:
        """
        Load EOG features for one experiment.

        Raw shape in .mat: (samples=885, features=36)
        Returns: (samples, 36)
        """
        filepath = self.data_dir / "EOG_Feature" / filename
        if not filepath.exists():
            logger.warning(f"EOG file not found: {filepath}, using zeros")
            return None

        mat = self._load_mat_safe(filepath)

        # Try configured method first
        preferred = self.cfg.eog_method
        candidates = [preferred] + [k for k in self._EOG_KEYS if k != preferred]
        eog = self._extract_key(mat, candidates)

        if eog is None:
            logger.warning(f"No EOG keys found in {filepath.name}, using zeros")
            return None

        eog = eog.squeeze()
        if eog.ndim == 1:
            eog = eog.reshape(-1, 1)

        logger.debug(f"EOG loaded: {filepath.name} → {eog.shape}")
        return eog

    def _load_perclos(self, filename: str) -> np.ndarray:
        """
        Load PERCLOS labels for one experiment.

        Raw shape in .mat: (samples=885, 1)
        Returns: (samples,)
        """
        filepath = self.data_dir / "perclos_labels" / filename
        if not filepath.exists():
            raise FileNotFoundError(f"PERCLOS labels not found: {filepath}")

        mat = self._load_mat_safe(filepath)
        perclos = self._extract_key(mat, self._PERCLOS_KEYS)

        if perclos is None:
            raise KeyError(f"No PERCLOS/label found in {filepath.name}")

        perclos = perclos.flatten().astype(np.float64)
        logger.debug(f"PERCLOS loaded: {filepath.name} → {perclos.shape}")
        return perclos

    def load_subject(self, subject_id: int, force_reload: bool = False) -> SubjectData:
        """
        Load data for experiment by 1-indexed ID.

        Args:
            subject_id: 1-indexed experiment ID (1 to 23)
            force_reload: Bypass cache

        Returns:
            SubjectData instance
        """
        if not force_reload and subject_id in self._cache:
            return self._cache[subject_id]

        experiments = self._discover_experiments()

        if subject_id < 1 or subject_id > len(experiments):
            raise FileNotFoundError(
                f"Experiment {subject_id} out of range [1, {len(experiments)}]"
            )

        # Use positional index (sorted alphabetically)
        target_file = experiments[subject_id - 1]
        filename = target_file.name

        logger.info(f"Loading experiment {subject_id}: {filename}")

        # Load each modality from its folder
        eeg = self._load_eeg(filename)
        eog = self._load_eog(filename)
        perclos = self._load_perclos(filename)

        # Fill missing EOG with zeros
        if eog is None:
            eog = np.zeros((eeg.shape[0], self.cfg.eog_dim), dtype=np.float32)

        # Align lengths across modalities
        min_len = min(eeg.shape[0], eog.shape[0], len(perclos))
        eeg = eeg[:min_len]
        eog = eog[:min_len]
        perclos = perclos[:min_len]

        data = SubjectData(
            subject_id=subject_id,
            eeg_features=eeg.astype(np.float32),
            eog_features=eog.astype(np.float32),
            perclos=perclos.astype(np.float32),
            num_samples=min_len,
        )
        self._cache[subject_id] = data
        return data

    def load_all_subjects(self) -> list[SubjectData]:
        """Load all available experiments."""
        experiments = self._discover_experiments()
        subjects = []
        for i in range(1, len(experiments) + 1):
            try:
                subjects.append(self.load_subject(i))
            except (FileNotFoundError, KeyError) as e:
                logger.warning(f"Skipping experiment {i}: {e}")
        logger.info(f"Loaded {len(subjects)} experiments")
        return subjects

    def split_subjects(
        self,
        test_ids: list[int] | tuple[int, ...],
    ) -> tuple[list[SubjectData], list[SubjectData]]:
        """
        Subject-independent train/test split.

        Args:
            test_ids: Experiment IDs for test set

        Returns:
            (train_subjects, test_subjects)
        """
        all_subjects = self.load_all_subjects()
        test_set = [s for s in all_subjects if s.subject_id in test_ids]
        train_set = [s for s in all_subjects if s.subject_id not in test_ids]
        logger.info(
            f"Split: {len(train_set)} train, {len(test_set)} test"
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
