#!/usr/bin/env python3
"""
NDRLDA Evaluation Entry Point.

Evaluates trained PPO agent on held-out test subjects with
subject-independent cross-validation.

Usage:
    source penv/bin/activate
    python evaluate.py                                    # use default checkpoint
    python evaluate.py --model checkpoints/ppo_final.zip  # specific model
    python evaluate.py --benchmark-latency                # include latency benchmark
"""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config.settings import Config
from preprocessing.data_loader import SEEDVIGLoader
from preprocessing.feature_processor import FeatureProcessor
from evaluation.evaluator import SubjectIndependentEvaluator
from evaluation.latency import LatencyBenchmark
from visualization.plots import VigilancePlotter

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NDRLDA vigilance agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model .zip (default: checkpoints/ppo_vigilance_final)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--test-subjects", nargs="+", type=int, default=None)
    parser.add_argument("--benchmark-latency", action="store_true")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Generate visualization plots")
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg = Config()

    if args.test_subjects:
        cfg.training.test_subjects = tuple(args.test_subjects)

    # Load model
    model_path = args.model or str(cfg.training.checkpoint_dir / "ppo_vigilance_final")
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    model_path = Path(model_path)

    if not model_path.exists():
        # Try without .zip
        alt = model_path.with_suffix("")
        if alt.exists():
            model_path = alt
        else:
            logger.error(f"Model not found: {model_path}")
            return

    logger.info(f"Loading model from {model_path}")
    agent = PPO.load(str(model_path))

    # Prepare test data
    loader = SEEDVIGLoader(cfg)
    processor = FeatureProcessor(cfg)

    # Load training subjects to fit processor, then transform test subjects
    train_raw, test_raw = loader.split_subjects(list(cfg.training.test_subjects))
    processor.fit(train_raw)
    test_processed = [processor.transform(s) for s in test_raw]

    logger.info(f"Evaluating on {len(test_processed)} test subjects")

    # Evaluate
    evaluator = SubjectIndependentEvaluator(cfg)
    results = evaluator.evaluate(agent, test_processed)

    # Save
    output_dir = Path(args.output_dir) if args.output_dir else cfg.evaluation.results_dir
    evaluator.save_results(results, output_dir)

    # Latency benchmark
    if args.benchmark_latency and test_processed:
        from environment.vigilance_env import VigilanceEnv
        env = VigilanceEnv(cfg, [test_processed[0]])
        obs, _ = env.reset()
        bench = LatencyBenchmark(cfg)
        latency_results = bench.run(agent, obs)
        with open(output_dir / "latency_benchmark.json", "w") as f:
            json.dump(latency_results, f, indent=2)

    # Plots
    if args.plot:
        plotter = VigilancePlotter(cfg)
        histories = [r["history"] for r in results["per_subject"]]
        metrics = [r["metrics"] for r in results["per_subject"]]
        subject_ids = [r["subject_id"] for r in results["per_subject"]]
        plot_dir = output_dir / "plots"

        for r in results["per_subject"]:
            plotter.plot_subject_trajectory(
                r["history"], r["subject_id"],
                save_dir=plot_dir,
            )
            plotter.plot_subject_dashboard(
                r["history"], r["subject_id"],
                metrics=r["metrics"],
                save_dir=plot_dir,
            )
        plotter.plot_cross_subject_summary(
            results["aggregated"],
            save_dir=plot_dir,
            subject_ids=subject_ids,
        )
        plotter.plot_intervention_heatmap(
            histories,
            save_dir=plot_dir,
            subject_ids=subject_ids,
        )
        plotter.plot_perclos_heatmap(
            histories,
            save_dir=plot_dir,
            subject_ids=subject_ids,
        )
        plotter.plot_action_distribution(
            histories,
            save_dir=plot_dir,
            subject_ids=subject_ids,
        )
        plotter.plot_perclos_distribution(
            histories,
            save_dir=plot_dir,
            subject_ids=subject_ids,
        )
        plotter.plot_metric_radar(
            results["aggregated"],
            save_dir=plot_dir,
        )
        plotter.plot_policy_tradeoff(
            metrics,
            save_dir=plot_dir,
            subject_ids=subject_ids,
        )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
