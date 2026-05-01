"""
Reward engineering for the vigilance control environment.

Mathematical Formulation
========================

The composite reward at timestep t given state s_t and action a_t:

    R(s_t, a_t) = R_safety + R_comfort + R_efficiency + R_trend + R_alert_fatigue + R_bonus

Components:

1. Safety Penalty (penalize high drowsiness):
    R_safety = -α · max(0, PERCLOS_t - τ_safe)²

    Quadratic penalty grows rapidly as PERCLOS exceeds safe threshold.
    Forces agent to prevent dangerous drowsiness states.

2. Comfort Cost (penalize unnecessary/strong interventions):
    R_comfort = -β · cost(a_t)

    cost(a_t) ∈ {0.0, 0.1, 0.3, 0.6, 1.0} for actions {0, 1, 2, 3, 4}
    Encourages minimal effective intervention.

3. Efficiency Reward (reward effective interventions):
    R_efficiency = γ · max(0, PERCLOS_{t-1} - PERCLOS_t) · 𝟙(a_t > 0)

    Agent gets rewarded only when an intervention actually reduces PERCLOS.
    Prevents rewarding natural PERCLOS fluctuations.

4. Trend Reward (reward improving vigilance trajectory):
    R_trend = δ · (EMA_{t-1} - EMA_t)

    EMA_t = α_ema · PERCLOS_t + (1 - α_ema) · EMA_{t-1}
    Rewards sustained vigilance improvement, not just instantaneous changes.

5. Alert Fatigue Penalty (penalize repeated same-level alerts):
    R_alert_fatigue = -ζ · 𝟙(consecutive_same_alert > patience)

    Discourages agent from spamming the same alert level.

6. Bonus/Penalty:
    R_bonus = danger_penalty   if PERCLOS_t > τ_danger
            = recovery_bonus   if PERCLOS_t < τ_safe AND PERCLOS_{t-1} >= τ_safe
            = 0                otherwise

Temporal Smoothing:
    R_smoothed_t = (1 - λ) · R_t + λ · R_smoothed_{t-1}

    Reduces reward noise, stabilizes policy gradient estimates.
"""

import numpy as np

from config.settings import RewardConfig, EnvironmentConfig


class RewardFunction:
    """
    Stateful reward calculator.

    Maintains internal state (EMA, previous PERCLOS, alert history) across
    timesteps within an episode. Must call reset() at episode start.

    Usage:
        reward_fn = RewardFunction(reward_config, env_config)
        reward_fn.reset()
        r = reward_fn.compute(perclos_t, perclos_prev, action)
    """

    def __init__(self, reward_cfg: RewardConfig, env_cfg: EnvironmentConfig):
        self.rcfg = reward_cfg
        self.ecfg = env_cfg
        self.reset()

    def reset(self):
        """Reset internal state for new episode."""
        self._ema_prev: float = 0.5  # initial EMA estimate (mid-range)
        self._smoothed_reward: float = 0.0
        self._last_action: int = -1
        self._consecutive_same: int = 0
        self._prev_perclos: float = 0.5
        self._step: int = 0

    def _safety_penalty(self, perclos: float) -> float:
        """R_safety = -α · max(0, PERCLOS - τ_safe)²"""
        excess = max(0.0, perclos - self.rcfg.tau_safe)
        return -self.rcfg.alpha_safety * excess ** 2

    def _comfort_cost(self, action: int) -> float:
        """R_comfort = -β · cost(a)"""
        cost = self.rcfg.action_costs[action]
        return -self.rcfg.beta_comfort * cost

    def _efficiency_reward(self, perclos: float, perclos_prev: float, action: int) -> float:
        """R_efficiency = γ · max(0, PERCLOS_{t-1} - PERCLOS_t) · 𝟙(a > 0)"""
        if action == 0:
            return 0.0
        improvement = max(0.0, perclos_prev - perclos)
        return self.rcfg.gamma_efficiency * improvement

    def _trend_reward(self, perclos: float) -> float:
        """R_trend = δ · (EMA_{t-1} - EMA_t)"""
        alpha = self.rcfg.trend_ema_alpha
        ema_new = alpha * perclos + (1 - alpha) * self._ema_prev
        reward = self.rcfg.delta_trend * (self._ema_prev - ema_new)
        self._ema_prev = ema_new
        return reward

    def _alert_fatigue_penalty(self, action: int) -> float:
        """R_alert_fatigue = -ζ · 𝟙(consecutive > patience)"""
        if action == self._last_action and action > 0:
            self._consecutive_same += 1
        else:
            self._consecutive_same = 0
        self._last_action = action

        if self._consecutive_same > self.rcfg.alert_fatigue_patience:
            return -self.rcfg.zeta_alert_fatigue
        return 0.0

    def _bonus_penalty(self, perclos: float, perclos_prev: float) -> float:
        """Categorical bonus/penalty for critical state transitions."""
        reward = 0.0
        if perclos > self.ecfg.perclos_danger:
            reward += self.rcfg.danger_zone_penalty
        if perclos < self.rcfg.tau_safe and perclos_prev >= self.rcfg.tau_safe:
            reward += self.rcfg.recovery_bonus
        return reward

    def compute(self, perclos: float, perclos_prev: float | None, action: int) -> float:
        """
        Compute composite reward for current timestep.

        Args:
            perclos: Current PERCLOS value [0, 1]
            perclos_prev: Previous PERCLOS (None on first step → uses internal state)
            action: Discrete action taken {0, 1, 2, 3, 4}

        Returns:
            Temporally smoothed scalar reward
        """
        if perclos_prev is None:
            perclos_prev = self._prev_perclos

        # Component rewards
        r_safety = self._safety_penalty(perclos)
        r_comfort = self._comfort_cost(action)
        r_efficiency = self._efficiency_reward(perclos, perclos_prev, action)
        r_trend = self._trend_reward(perclos)
        r_fatigue = self._alert_fatigue_penalty(action)
        r_bonus = self._bonus_penalty(perclos, perclos_prev)

        raw_reward = r_safety + r_comfort + r_efficiency + r_trend + r_fatigue + r_bonus

        # Temporal smoothing
        lam = self.rcfg.reward_smoothing_lambda
        self._smoothed_reward = (1 - lam) * raw_reward + lam * self._smoothed_reward

        self._prev_perclos = perclos
        self._step += 1

        return self._smoothed_reward

    def get_reward_breakdown(self, perclos: float, perclos_prev: float, action: int) -> dict:
        """
        Compute reward with full component breakdown (for analysis only).

        Does NOT update internal state — use compute() for actual training.
        """
        excess = max(0.0, perclos - self.rcfg.tau_safe)
        return {
            "safety": -self.rcfg.alpha_safety * excess ** 2,
            "comfort": -self.rcfg.beta_comfort * self.rcfg.action_costs[action],
            "efficiency": (
                self.rcfg.gamma_efficiency * max(0.0, perclos_prev - perclos)
                if action > 0 else 0.0
            ),
            "trend_perclos": perclos,
            "bonus": (
                self.rcfg.danger_zone_penalty if perclos > self.ecfg.perclos_danger
                else (self.rcfg.recovery_bonus if perclos < self.rcfg.tau_safe and perclos_prev >= self.rcfg.tau_safe else 0.0)
            ),
        }
