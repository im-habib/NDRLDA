"""
Evaluation metrics for the vigilance control system.

Metrics computed per-subject and aggregated across subjects:

1. Vigilance Improvement:
    VI = (mean_PERCLOS_baseline - mean_PERCLOS_controlled) / mean_PERCLOS_baseline

2. Cumulative Reward:
    CR = Σ_t R(s_t, a_t)

3. Alert Efficiency:
    AE = (# effective interventions) / (# total interventions)
    An intervention is "effective" if PERCLOS decreases within k steps after it.

4. False Alert Rate:
    FAR = (# interventions when PERCLOS < safe_threshold) / (# total interventions)

5. Intervention Frequency:
    IF = (# steps with a > 0) / (# total steps)

6. Strong Intervention Frequency:
    SIF = (# steps with a >= 3) / (# total steps)

7. Dangerous Time Ratio:
    DTR = (# steps with PERCLOS > danger_threshold) / (# total steps)

8. Fatigue Recovery Effectiveness:
    FRE = (# times PERCLOS crosses from danger to safe) / (# times PERCLOS enters danger)

9. Mean Time in Danger:
    MTID = average consecutive steps spent above danger threshold
"""

import numpy as np


class VigilanceMetrics:
    """
    Computes all evaluation metrics from episode history.

    Usage:
        metrics = VigilanceMetrics(danger_threshold=0.70, safe_threshold=0.30)
        result = metrics.compute(perclos_list, actions_list, rewards_list)
    """

    def __init__(
        self,
        danger_threshold: float = 0.70,
        safe_threshold: float = 0.30,
        effectiveness_horizon: int = 5,
    ):
        self.danger_threshold = danger_threshold
        self.safe_threshold = safe_threshold
        self.effectiveness_horizon = effectiveness_horizon

    def compute(
        self,
        perclos: list[float] | np.ndarray,
        actions: list[int] | np.ndarray,
        rewards: list[float] | np.ndarray,
        baseline_perclos: np.ndarray | None = None,
    ) -> dict:
        """
        Compute all metrics from a single episode.

        Args:
            perclos: PERCLOS values at each step (after intervention effects)
            actions: Agent actions at each step
            rewards: Rewards received at each step
            baseline_perclos: Original PERCLOS without intervention (for VI metric)

        Returns:
            Dict of metric_name → value
        """
        perclos = np.asarray(perclos, dtype=float)
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        T = len(perclos)

        if T == 0:
            return {}

        results = {}

        # 1. Vigilance Improvement
        if baseline_perclos is not None:
            baseline = np.asarray(baseline_perclos[:T], dtype=float)
            baseline_mean = np.mean(baseline)
            if baseline_mean > 0:
                results["vigilance_improvement"] = float(
                    (baseline_mean - np.mean(perclos)) / baseline_mean
                )
            else:
                results["vigilance_improvement"] = 0.0
        results["mean_perclos"] = float(np.mean(perclos))

        # 2. Cumulative Reward
        results["cumulative_reward"] = float(np.sum(rewards))
        results["mean_reward"] = float(np.mean(rewards))

        # 3. Alert Efficiency
        intervention_steps = np.where(actions > 0)[0]
        n_interventions = len(intervention_steps)
        if n_interventions > 0:
            effective = 0
            for step in intervention_steps:
                horizon_end = min(step + self.effectiveness_horizon, T)
                if horizon_end > step + 1:
                    future_perclos = perclos[step + 1:horizon_end]
                    if len(future_perclos) > 0 and np.min(future_perclos) < perclos[step]:
                        effective += 1
            results["alert_efficiency"] = effective / n_interventions
        else:
            results["alert_efficiency"] = 1.0  # no false alerts if no alerts

        # 4. False Alert Rate
        if n_interventions > 0:
            false_alerts = np.sum(perclos[intervention_steps] < self.safe_threshold)
            results["false_alert_rate"] = float(false_alerts / n_interventions)
        else:
            results["false_alert_rate"] = 0.0

        # 5. Intervention Frequency
        results["intervention_rate"] = float(np.mean(actions > 0))

        # 6. Strong Intervention Frequency
        results["strong_intervention_rate"] = float(np.mean(actions >= 3))

        # 7. Dangerous Time Ratio
        results["danger_ratio"] = float(np.mean(perclos > self.danger_threshold))

        # 8. Fatigue Recovery Effectiveness
        danger_entries = 0
        recoveries = 0
        in_danger = False
        for t in range(T):
            if perclos[t] > self.danger_threshold:
                if not in_danger:
                    danger_entries += 1
                    in_danger = True
            elif perclos[t] < self.safe_threshold:
                if in_danger:
                    recoveries += 1
                    in_danger = False
            else:
                in_danger = perclos[t] > self.danger_threshold

        if danger_entries > 0:
            results["fatigue_recovery_effectiveness"] = recoveries / danger_entries
        else:
            results["fatigue_recovery_effectiveness"] = 1.0

        # 9. Mean Time in Danger
        danger_durations = []
        current_duration = 0
        for t in range(T):
            if perclos[t] > self.danger_threshold:
                current_duration += 1
            else:
                if current_duration > 0:
                    danger_durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            danger_durations.append(current_duration)
        results["mean_time_in_danger"] = float(np.mean(danger_durations)) if danger_durations else 0.0

        # Action distribution
        results["action_distribution"] = {
            int(a): int(np.sum(actions == a)) for a in range(5)
        }
        results["num_steps"] = T

        return results

    @staticmethod
    def aggregate(per_subject_metrics: list[dict]) -> dict:
        """
        Aggregate metrics across multiple subjects.

        Returns mean ± std for each numeric metric.
        """
        if not per_subject_metrics:
            return {}

        numeric_keys = [
            k for k in per_subject_metrics[0]
            if isinstance(per_subject_metrics[0][k], (int, float))
        ]

        aggregated = {}
        for key in numeric_keys:
            values = [m[key] for m in per_subject_metrics if key in m]
            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }
        return aggregated
