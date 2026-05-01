"""
Subject-independent evaluation pipeline.

Evaluates trained PPO agent on held-out test subjects.
Produces per-subject metrics and cross-subject aggregated results.
Supports domain generalization analysis.
"""

import json
import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config.settings import Config
from preprocessing.feature_processor import ProcessedSubject
from environment.vigilance_env import VigilanceEnv
from evaluation.metrics import VigilanceMetrics

logger = logging.getLogger(__name__)


class SubjectIndependentEvaluator:
    """
    Evaluates agent on held-out test subjects.

    Runs deterministic policy on each test subject's trajectory,
    collects episode history, computes metrics.

    Usage:
        evaluator = SubjectIndependentEvaluator(config)
        results = evaluator.evaluate(agent, test_subjects)
        evaluator.save_results(results, output_dir)
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.metrics = VigilanceMetrics(
            danger_threshold=config.evaluation.danger_threshold,
            safe_threshold=config.reward.tau_safe,
        )

    def evaluate_subject(
        self,
        agent: PPO,
        subject: ProcessedSubject,
        deterministic: bool = True,
    ) -> dict:
        """
        Evaluate agent on a single subject.

        Args:
            agent: Trained PPO agent
            subject: Processed subject data
            deterministic: Use deterministic (greedy) policy

        Returns:
            Dict with metrics and episode history
        """
        env = VigilanceEnv(self.cfg, [subject])
        obs, info = env.reset(options={"subject_id": subject.subject_id})

        all_perclos = []
        all_actions = []
        all_rewards = []
        baseline_perclos = subject.perclos_mean.tolist()

        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            all_perclos.append(info.get("perclos", 0.0))
            all_actions.append(int(action))
            all_rewards.append(float(reward))

        # Compute metrics
        metrics = self.metrics.compute(
            all_perclos, all_actions, all_rewards,
            baseline_perclos=np.array(baseline_perclos),
        )

        return {
            "subject_id": subject.subject_id,
            "metrics": metrics,
            "history": {
                "perclos": all_perclos,
                "actions": all_actions,
                "rewards": all_rewards,
                "baseline_perclos": baseline_perclos[:len(all_perclos)],
            },
        }

    def evaluate(
        self,
        agent: PPO,
        test_subjects: list[ProcessedSubject],
        deterministic: bool = True,
    ) -> dict:
        """
        Evaluate agent across all test subjects.

        Returns:
            {
                "per_subject": [subject_result, ...],
                "aggregated": aggregated_metrics,
                "summary": human-readable summary,
            }
        """
        per_subject = []
        for subj in test_subjects:
            logger.info(f"Evaluating subject {subj.subject_id}...")
            result = self.evaluate_subject(agent, subj, deterministic)
            per_subject.append(result)

            m = result["metrics"]
            logger.info(
                f"  Subject {subj.subject_id}: "
                f"reward={m.get('cumulative_reward', 0):.2f}, "
                f"mean_perclos={m.get('mean_perclos', 0):.3f}, "
                f"danger_ratio={m.get('danger_ratio', 0):.3f}, "
                f"intervention_rate={m.get('intervention_rate', 0):.3f}"
            )

        # Aggregate
        all_metrics = [r["metrics"] for r in per_subject]
        aggregated = VigilanceMetrics.aggregate(all_metrics)

        # Summary
        summary_lines = ["=" * 60, "CROSS-SUBJECT EVALUATION RESULTS", "=" * 60]
        for key, val in aggregated.items():
            if isinstance(val, dict) and "mean" in val:
                summary_lines.append(f"  {key}: {val['mean']:.4f} ± {val['std']:.4f}")
        summary = "\n".join(summary_lines)
        logger.info("\n" + summary)

        return {
            "per_subject": per_subject,
            "aggregated": aggregated,
            "summary": summary,
        }

    def save_results(self, results: dict, output_dir: Path | None = None):
        """Save evaluation results to JSON and per-subject files."""
        out = output_dir or self.cfg.evaluation.results_dir
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)

        # Main results (without full history to keep file size small)
        summary_data = {
            "aggregated": results["aggregated"],
            "per_subject_metrics": [
                {"subject_id": r["subject_id"], "metrics": r["metrics"]}
                for r in results["per_subject"]
            ],
        }
        with open(out / "evaluation_results.json", "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        # Per-subject histories (for visualization)
        for r in results["per_subject"]:
            sid = r["subject_id"]
            with open(out / f"subject_{sid}_history.json", "w") as f:
                json.dump(r["history"], f)

        logger.info(f"Results saved to {out}")
