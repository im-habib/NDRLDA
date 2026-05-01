#!/usr/bin/env python3
"""
NDRLDA Training Entry Point.

Trains a PPO agent for adaptive driver vigilance control
on the SEED-VIG dataset.

Usage:
    source penv/bin/activate
    python train.py                          # default config
    python train.py --timesteps 1000000      # override timesteps
    python train.py --encoder attention --temporal transformer  # architecture
    python train.py --test-subjects 1 5 10   # custom test split
"""

import argparse
import logging
import sys

from config.settings import Config
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train NDRLDA vigilance agent")

    # Training
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Architecture
    parser.add_argument("--encoder", choices=["mlp", "cnn", "attention"], default=None)
    parser.add_argument("--temporal", choices=["lstm", "transformer"], default=None)
    parser.add_argument("--window-size", type=int, default=None)

    # Data
    parser.add_argument("--test-subjects", nargs="+", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)

    # Preprocessing
    parser.add_argument("--norm", choices=["zscore", "minmax", "robust"], default=None)
    parser.add_argument("--smoothing", choices=["moving_avg", "lds", "none"], default=None)
    parser.add_argument("--use-pca", action="store_true", default=False)
    parser.add_argument("--use-autoencoder", action="store_true", default=False)

    # System
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=None)

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg = Config()

    # Apply CLI overrides
    if args.timesteps is not None:
        cfg.ppo.total_timesteps = args.timesteps
    if args.lr is not None:
        cfg.ppo.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.ppo.batch_size = args.batch_size
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.encoder is not None:
        cfg.encoder.encoder_type = args.encoder
    if args.temporal is not None:
        cfg.temporal.temporal_type = args.temporal
    if args.window_size is not None:
        cfg.preprocessing.window_size = args.window_size
    if args.test_subjects is not None:
        cfg.training.test_subjects = tuple(args.test_subjects)
    if args.data_dir is not None:
        from pathlib import Path
        cfg.data.data_dir = Path(args.data_dir)
    if args.norm is not None:
        cfg.preprocessing.norm_method = args.norm
    if args.smoothing is not None:
        cfg.preprocessing.smoothing_method = args.smoothing
    if args.use_pca:
        cfg.preprocessing.use_pca = True
    if args.use_autoencoder:
        cfg.preprocessing.use_autoencoder = True
    if args.device is not None:
        cfg.training.device = args.device
    if args.verbose is not None:
        cfg.training.verbose = args.verbose

    trainer = Trainer(cfg)
    trainer.prepare_data()
    trainer.train()
    trainer.save_results()

    logging.getLogger(__name__).info("Training complete.")


if __name__ == "__main__":
    main()
