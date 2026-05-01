"""
Training pipeline for the vigilance DRL agent.

Handles:
    - Data loading and preprocessing
    - Subject-independent train/test splitting
    - Environment creation
    - PPO agent training with callbacks
    - Validation during training
    - GPU acceleration and mixed precision
"""

import logging
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from config.settings import Config
from preprocessing.data_loader import SEEDVIGLoader
from preprocessing.feature_processor import FeatureProcessor, ProcessedSubject
from preprocessing.feature_engineer import FeatureEngineer
from environment.vigilance_env import VigilanceEnv
from models.ppo_agent import create_ppo_agent
from training.callbacks import VigilanceCallback, EarlyStoppingCallback

logger = logging.getLogger(__name__)


class Trainer:
    """
    End-to-end training pipeline.

    Usage:
        trainer = Trainer(config)
        trainer.prepare_data()
        trainer.train()
        trainer.save_results()
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.loader = SEEDVIGLoader(config)
        self.processor = FeatureProcessor(config)
        self.engineer = FeatureEngineer(config)

        self.train_subjects: list[ProcessedSubject] = []
        self.val_subjects: list[ProcessedSubject] = []
        self.test_subjects: list[ProcessedSubject] = []
        self.agent = None

    def prepare_data(self):
        """
        Load, preprocess, and split data.

        Pipeline:
            .mat files → SEEDVIGLoader → raw SubjectData
            → FeatureProcessor (smooth, normalize, window) → ProcessedSubject
            → FeatureEngineer (optional PCA/autoencoder) → final ProcessedSubject
        """
        logger.info("Loading SEED-VIG data...")
        test_ids = list(self.cfg.training.test_subjects)
        train_raw, test_raw = self.loader.split_subjects(test_ids)

        # Validation split from training subjects
        np.random.seed(self.cfg.training.seed)
        n_val = max(1, int(len(train_raw) * self.cfg.training.val_ratio))
        val_indices = np.random.choice(len(train_raw), n_val, replace=False)
        val_raw = [train_raw[i] for i in val_indices]
        train_raw = [s for i, s in enumerate(train_raw) if i not in val_indices]

        logger.info(f"Subjects — train: {len(train_raw)}, val: {len(val_raw)}, test: {len(test_raw)}")

        # Optional feature engineering (fit on train only)
        if self.cfg.preprocessing.use_pca or self.cfg.preprocessing.use_autoencoder:
            device = self._resolve_device()
            train_features = np.concatenate(
                [s.get_fused_features() for s in train_raw], axis=0
            )
            self.engineer.fit(train_features, device=device)
            # Apply to all splits before windowing
            for subj in train_raw + val_raw + test_raw:
                fused = subj.get_fused_features()
                reduced = self.engineer.transform(fused)
                # Replace EEG/EOG with reduced features
                subj.eeg_features = reduced[:, np.newaxis, :]  # (T, 1, reduced_dim)
                subj.eog_features = np.zeros((len(reduced), 0), dtype=np.float32)

        # Fit processor on training data, transform all
        self.processor.fit(train_raw)
        self.train_subjects = [self.processor.transform(s) for s in train_raw]
        self.val_subjects = [self.processor.transform(s) for s in val_raw]
        self.test_subjects = [self.processor.transform(s) for s in test_raw]

        logger.info(
            f"Processing complete. "
            f"Feature dim: {self.train_subjects[0].feature_dim}, "
            f"Windows per subject: ~{np.mean([s.num_windows for s in self.train_subjects]):.0f}"
        )

    def _resolve_device(self) -> str:
        device = self.cfg.training.device
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _make_env(self, subjects: list[ProcessedSubject], monitor: bool = True):
        """Create a Gymnasium environment wrapped for SB3."""
        def _create():
            env = VigilanceEnv(self.cfg, subjects)
            if monitor:
                env = Monitor(env)
            return env
        return _create

    def train(self) -> None:
        """
        Run PPO training.

        Creates vectorized environment, configures callbacks,
        and trains for cfg.ppo.total_timesteps.
        """
        if not self.train_subjects:
            raise RuntimeError("Call prepare_data() first")

        logger.info("Creating training environment...")
        # Vectorized env for parallel rollouts
        train_env = DummyVecEnv([self._make_env(self.train_subjects)])

        # Evaluation env (validation subjects)
        eval_env = None
        if self.val_subjects:
            eval_env = DummyVecEnv([self._make_env(self.val_subjects)])

        # Create agent
        tb_log = str(self.cfg.training.log_dir) if self.cfg.training.use_tensorboard else None
        self.agent = create_ppo_agent(train_env, self.cfg, tensorboard_log=tb_log)

        # Callbacks
        callbacks = []

        # Vigilance-specific logging + checkpointing
        vig_callback = VigilanceCallback(
            checkpoint_dir=self.cfg.training.checkpoint_dir,
            save_freq=self.cfg.training.save_every_n_steps,
            keep_last_n=self.cfg.training.keep_last_n,
        )
        callbacks.append(vig_callback)

        # Eval callback (validation)
        if eval_env:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.cfg.training.checkpoint_dir / "best"),
                eval_freq=self.cfg.training.eval_freq,
                n_eval_episodes=self.cfg.training.n_eval_episodes,
                deterministic=True,
            )
            callbacks.append(eval_callback)

        # Early stopping
        early_stop = EarlyStoppingCallback(
            check_freq=self.cfg.training.eval_freq,
            patience=15,
        )
        callbacks.append(early_stop)

        # Train
        logger.info(
            f"Starting PPO training: {self.cfg.ppo.total_timesteps:,} timesteps, "
            f"device={self._resolve_device()}"
        )
        self.agent.learn(
            total_timesteps=self.cfg.ppo.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )

        # Summary
        summary = vig_callback.get_stats_summary()
        if summary:
            logger.info("Training summary:")
            for key, vals in summary.items():
                logger.info(f"  {key}: mean={vals['mean']:.4f}, last_10={vals['last_10_mean']:.4f}")

        train_env.close()
        if eval_env:
            eval_env.close()

    def save_results(self, output_dir: Path | None = None):
        """Save final model and training configuration."""
        out = output_dir or self.cfg.training.checkpoint_dir
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)

        if self.agent:
            self.agent.save(str(out / "ppo_vigilance_final"))

        # Save config as JSON for reproducibility
        import dataclasses
        config_dict = {}
        for field in dataclasses.fields(self.cfg):
            val = getattr(self.cfg, field.name)
            if dataclasses.is_dataclass(val):
                sub = {}
                for sf in dataclasses.fields(val):
                    sv = getattr(val, sf.name)
                    if isinstance(sv, Path):
                        sv = str(sv)
                    elif isinstance(sv, tuple):
                        sv = list(sv)
                    sub[sf.name] = sv
                config_dict[field.name] = sub
        with open(out / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Results saved to {out}")
