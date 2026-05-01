#!/usr/bin/env python3
"""
NDRLDA Real-Time Simulation Entry Point.

Simulates continuous driver monitoring with trained agent.

Usage:
    source penv/bin/activate
    python simulate.py                           # simulate on first test subject
    python simulate.py --subject 5               # specific subject
    python simulate.py --realtime --speed 2.0    # real-time pacing at 2x speed
    python simulate.py --export-onnx             # also export ONNX model
"""

import argparse
import json
import logging
from pathlib import Path

from stable_baselines3 import PPO

from config.settings import Config
from preprocessing.data_loader import SEEDVIGLoader
from preprocessing.feature_processor import FeatureProcessor
from deployment.realtime import RealtimeSimulator
from deployment.optimize import ModelOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description="NDRLDA real-time simulation")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--subject", type=int, default=None,
                        help="Subject ID to simulate (default: first test subject)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Simulation speed multiplier")
    parser.add_argument("--realtime", action="store_true",
                        help="Simulate at real-time pace (with sleep)")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export model to ONNX")
    parser.add_argument("--analyze", action="store_true",
                        help="Run deployment analysis")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    cfg = Config()

    # Load model
    model_path = args.model or str(cfg.training.checkpoint_dir / "ppo_vigilance_final")
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    model_path = Path(model_path)

    if not model_path.exists():
        alt = model_path.with_suffix("")
        if alt.exists():
            model_path = alt
        else:
            logger.error(f"Model not found: {model_path}")
            return

    logger.info(f"Loading model: {model_path}")
    agent = PPO.load(str(model_path))

    # Load data
    loader = SEEDVIGLoader(cfg)
    processor = FeatureProcessor(cfg)

    train_raw, test_raw = loader.split_subjects(list(cfg.training.test_subjects))
    processor.fit(train_raw)

    # Select subject
    if args.subject:
        subject_data = loader.load_subject(args.subject)
    elif test_raw:
        subject_data = test_raw[0]
    else:
        logger.error("No subjects available")
        return

    logger.info(f"Simulating subject {subject_data.subject_id} ({subject_data.num_samples} samples)")

    # Run simulation
    simulator = RealtimeSimulator(cfg, agent, processor)
    results = simulator.run(
        subject_data,
        speed=args.speed,
        realtime=args.realtime,
    )

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else cfg.deployment.export_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "simulation_results.json", "w") as f:
        json.dump(results["summary"], f, indent=2)

    # ONNX export
    if args.export_onnx:
        optimizer = ModelOptimizer(cfg)
        processed = processor.transform(subject_data)
        sample_obs = processed.features[0].flatten()
        aux = [0.5, 0.5, 0.0]
        import numpy as np
        sample_obs = np.concatenate([sample_obs, aux]).astype(np.float32)
        optimizer.export_onnx(agent, sample_obs)

    # Deployment analysis
    if args.analyze:
        optimizer = ModelOptimizer(cfg)
        processed = processor.transform(subject_data)
        sample_obs = processed.features[0].flatten()
        import numpy as np
        aux = [0.5, 0.5, 0.0]
        sample_obs = np.concatenate([sample_obs, aux]).astype(np.float32)
        report = optimizer.analyze(agent, sample_obs)
        with open(output_dir / "deployment_analysis.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

    logger.info("Simulation complete.")


if __name__ == "__main__":
    main()
