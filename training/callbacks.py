"""
Custom Stable-Baselines3 callbacks for vigilance training.

Provides:
    - VigilanceCallback: logs vigilance-specific metrics, handles checkpointing
    - EarlyStoppingCallback: stops training when reward plateaus
"""

import logging
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class VigilanceCallback(BaseCallback):
    """
    Logs vigilance-specific metrics from episode info dicts.

    Tracks: mean PERCLOS, danger ratio, intervention rate, strong intervention rate,
    cumulative reward. Saves checkpoints at configurable intervals.

    Metrics logged to tensorboard under 'vigilance/' prefix.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_freq: int = 50_000,
        keep_last_n: int = 3,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.keep_last_n = keep_last_n
        self._saved_checkpoints: list[Path] = []

        # Running episode stats
        self._episode_stats: list[dict] = []

    def _on_step(self) -> bool:
        # Collect episode stats from info dicts
        for info in self.locals.get("infos", []):
            if "episode_stats" in info:
                stats = info["episode_stats"]
                self._episode_stats.append(stats)

                # Log to tensorboard
                if self.logger:
                    for key in ["cumulative_reward", "mean_perclos", "danger_ratio",
                                "intervention_rate", "strong_intervention_rate"]:
                        if key in stats:
                            self.logger.record(f"vigilance/{key}", stats[key])

        # Checkpoint
        if self.n_calls % self.save_freq == 0:
            self._save_checkpoint()

        return True

    def _save_checkpoint(self):
        """Save model checkpoint, manage rotation."""
        path = self.checkpoint_dir / f"ppo_vigilance_{self.num_timesteps}.zip"
        self.model.save(str(path))
        self._saved_checkpoints.append(path)
        logger.info(f"Checkpoint saved: {path.name}")

        # Rotate old checkpoints
        while len(self._saved_checkpoints) > self.keep_last_n:
            old = self._saved_checkpoints.pop(0)
            if old.exists():
                old.unlink()

    def _on_training_end(self):
        """Save final checkpoint and stats summary."""
        # Final checkpoint
        final_path = self.checkpoint_dir / "ppo_vigilance_final.zip"
        self.model.save(str(final_path))
        logger.info(f"Final model saved: {final_path}")

        # Save training stats
        if self._episode_stats:
            stats_path = self.checkpoint_dir / "training_stats.json"
            with open(stats_path, "w") as f:
                json.dump(self._episode_stats, f, indent=2)
            logger.info(f"Training stats saved: {stats_path}")

    def get_stats_summary(self) -> dict:
        """Summarize collected episode stats."""
        if not self._episode_stats:
            return {}
        keys = ["cumulative_reward", "mean_perclos", "danger_ratio",
                "intervention_rate", "strong_intervention_rate"]
        summary = {}
        for key in keys:
            values = [s[key] for s in self._episode_stats if key in s]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "last_10_mean": float(np.mean(values[-10:])),
                }
        return summary


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training when mean reward stops improving.

    Checks every `check_freq` steps whether mean reward over last
    `window` episodes exceeds the previous best by at least `min_delta`.
    Stops after `patience` checks without improvement.
    """

    def __init__(
        self,
        check_freq: int = 10_000,
        patience: int = 10,
        min_delta: float = 0.5,
        window: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_delta = min_delta
        self.window = window
        self._best_reward = -np.inf
        self._no_improve_count = 0
        self._rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_stats" in info:
                self._rewards.append(info["episode_stats"].get("cumulative_reward", 0))

        if self.n_calls % self.check_freq == 0 and len(self._rewards) >= self.window:
            recent_mean = np.mean(self._rewards[-self.window:])
            if recent_mean > self._best_reward + self.min_delta:
                self._best_reward = recent_mean
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
                if self._no_improve_count >= self.patience:
                    logger.info(
                        f"Early stopping: no improvement for {self.patience} checks. "
                        f"Best mean reward: {self._best_reward:.2f}"
                    )
                    return False
        return True
