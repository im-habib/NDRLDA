"""
Custom Gymnasium environment for driver vigilance control.

The environment replays a subject's physiological trajectory and allows the DRL agent
to issue intervention actions at each timestep. PERCLOS dynamics are modulated by
agent actions through a simulated alert-effect model.

State Space:
    Flattened window of processed multimodal features: ℝ^(window_size × feature_dim)
    Augmented with: current PERCLOS, mean PERCLOS over window, step fraction.

Action Space:
    Discrete(5):
        0 = no intervention
        1 = soft visual alert
        2 = audio alert
        3 = strong haptic + audio alert
        4 = emergency rest recommendation

Transition Model:
    PERCLOS evolves from recorded data, modulated by intervention effects:
        PERCLOS'_t = PERCLOS_t - effect(a_t) · decay^(consecutive_uses)

    The agent cannot "cheat" — interventions have bounded, decaying effects.

Episode:
    One episode = one subject's full trajectory (or max_episode_steps).
    Episode ends when trajectory is exhausted or step limit reached.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.settings import Config
from environment.reward import RewardFunction
from preprocessing.feature_processor import ProcessedSubject


class VigilanceEnv(gym.Env):
    """
    Gymnasium environment for vigilance-based DRL training.

    The env steps through pre-processed windowed features of a subject,
    simulates PERCLOS modulation from agent actions, and computes shaped rewards.

    Usage:
        env = VigilanceEnv(config, processed_subjects)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Config,
        subjects: list[ProcessedSubject],
        render_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = config
        self.env_cfg = config.env
        self.subjects = subjects
        self.render_mode = render_mode

        # Validate subjects
        if not subjects:
            raise ValueError("At least one ProcessedSubject required")

        # Dimensions
        sample = subjects[0]
        self.feature_dim = sample.feature_dim
        self.window_size = sample.features.shape[1]

        # Observation: flattened window features + 3 auxiliary features
        # [features_flat, current_perclos, window_mean_perclos, step_fraction]
        obs_dim = self.feature_dim * self.window_size + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Discrete(self.env_cfg.num_actions)

        # Reward function
        self.reward_fn = RewardFunction(config.reward, config.env)

        # Episode state
        self._current_subject: ProcessedSubject | None = None
        self._step_idx: int = 0
        self._perclos_modified: np.ndarray | None = None
        self._action_counts: np.ndarray = np.zeros(self.env_cfg.num_actions, dtype=int)
        self._consecutive_action: dict = {"action": -1, "count": 0}
        self._episode_rewards: list[float] = []
        self._episode_actions: list[int] = []
        self._episode_perclos: list[float] = []

    def _select_subject(self, subject_id: int | None = None) -> ProcessedSubject:
        """Select subject for episode (random if not specified)."""
        if subject_id is not None:
            matches = [s for s in self.subjects if s.subject_id == subject_id]
            if matches:
                return matches[0]
        return self.subjects[self.np_random.integers(len(self.subjects))]

    def _build_observation(self) -> np.ndarray:
        """Construct observation vector from current state."""
        subj = self._current_subject
        idx = self._step_idx

        # Windowed features (flattened)
        features_flat = subj.features[idx].flatten()

        # Current (possibly modified) PERCLOS
        current_perclos = self._perclos_modified[idx]

        # Window mean PERCLOS
        window_perclos = self._perclos_modified[max(0, idx - self.window_size + 1):idx + 1]
        mean_perclos = float(np.mean(window_perclos))

        # Progress fraction
        step_fraction = idx / max(subj.num_windows - 1, 1)

        aux = np.array([current_perclos, mean_perclos, step_fraction], dtype=np.float32)
        return np.concatenate([features_flat, aux])

    def reset(self, *, seed=None, options=None):
        """
        Reset environment for new episode.

        Options:
            subject_id (int): Force specific subject
        """
        super().reset(seed=seed)

        subject_id = options.get("subject_id") if options else None
        self._current_subject = self._select_subject(subject_id)
        self._step_idx = 0

        # Copy PERCLOS for modification by interventions
        self._perclos_modified = self._current_subject.perclos_mean.copy()

        # Reset tracking
        self._action_counts = np.zeros(self.env_cfg.num_actions, dtype=int)
        self._consecutive_action = {"action": -1, "count": 0}
        self._episode_rewards = []
        self._episode_actions = []
        self._episode_perclos = []

        # Reset reward function
        self.reward_fn.reset()

        obs = self._build_observation()
        info = {
            "subject_id": self._current_subject.subject_id,
            "num_windows": self._current_subject.num_windows,
            "initial_perclos": float(self._perclos_modified[0]),
        }
        return obs, info

    def _apply_intervention_effect(self, action: int):
        """
        Modulate future PERCLOS based on intervention action.

        Effect model:
            PERCLOS'_t = PERCLOS_t - effect(a) * decay^(consecutive_uses)

        Effect propagates forward with exponential decay.
        """
        if action == 0:
            return

        base_effect = self.env_cfg.alert_effect[action]

        # Decay for repeated same action
        if action == self._consecutive_action["action"]:
            self._consecutive_action["count"] += 1
        else:
            self._consecutive_action = {"action": action, "count": 0}

        decay = self.env_cfg.alert_effect_decay ** self._consecutive_action["count"]
        effect = base_effect * decay

        # Apply effect to current and next few steps (with temporal decay)
        idx = self._step_idx
        num_windows = self._current_subject.num_windows
        effect_horizon = min(10, num_windows - idx)

        for offset in range(effect_horizon):
            future_idx = idx + offset
            if future_idx >= num_windows:
                break
            temporal_decay = 0.8 ** offset
            self._perclos_modified[future_idx] = max(
                0.0,
                self._perclos_modified[future_idx] - effect * temporal_decay,
            )

    def step(self, action: int):
        """
        Execute one timestep.

        Args:
            action: Intervention level {0, 1, 2, 3, 4}

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = int(action)
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Previous PERCLOS
        prev_perclos = float(self._perclos_modified[self._step_idx])

        # Apply intervention effect
        self._apply_intervention_effect(action)

        # Advance timestep
        self._step_idx += 1
        self._action_counts[action] += 1

        # Check termination
        terminated = False
        truncated = False
        if self._step_idx >= self._current_subject.num_windows:
            terminated = True
        elif self._step_idx >= self.env_cfg.max_episode_steps:
            truncated = True

        # Current PERCLOS (after intervention effect + natural progression)
        if not terminated:
            current_perclos = float(self._perclos_modified[self._step_idx])
        else:
            current_perclos = prev_perclos

        # Compute reward
        reward = self.reward_fn.compute(current_perclos, prev_perclos, action)

        # Track
        self._episode_rewards.append(reward)
        self._episode_actions.append(action)
        self._episode_perclos.append(current_perclos)

        # Build observation
        obs = self._build_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )

        info = {
            "perclos": current_perclos,
            "prev_perclos": prev_perclos,
            "action": action,
            "step": self._step_idx,
            "subject_id": self._current_subject.subject_id,
        }

        if terminated or truncated:
            info["episode_stats"] = self._compute_episode_stats()

        return obs, reward, terminated, truncated, info

    def _compute_episode_stats(self) -> dict:
        """Compute summary statistics for completed episode."""
        perclos_arr = np.array(self._episode_perclos)
        actions_arr = np.array(self._episode_actions)
        return {
            "cumulative_reward": sum(self._episode_rewards),
            "mean_perclos": float(np.mean(perclos_arr)),
            "max_perclos": float(np.max(perclos_arr)) if len(perclos_arr) > 0 else 0.0,
            "danger_ratio": float(np.mean(perclos_arr > self.env_cfg.perclos_danger)),
            "intervention_rate": float(np.mean(actions_arr > 0)),
            "strong_intervention_rate": float(np.mean(actions_arr >= 3)),
            "action_distribution": {
                self.env_cfg.action_names[i]: int(self._action_counts[i])
                for i in range(self.env_cfg.num_actions)
            },
            "num_steps": len(self._episode_actions),
        }

    def get_episode_history(self) -> dict:
        """Return full episode history for visualization."""
        return {
            "perclos": list(self._episode_perclos),
            "actions": list(self._episode_actions),
            "rewards": list(self._episode_rewards),
            "subject_id": self._current_subject.subject_id if self._current_subject else None,
        }
