"""
Latency benchmarking for edge deployment feasibility.

Measures inference latency of the full pipeline:
    feature processing → model forward pass → action selection

Reports: mean, median, p95, p99 latency and throughput.
"""

import time
import logging

import numpy as np
import torch

from config.settings import Config

logger = logging.getLogger(__name__)


class LatencyBenchmark:
    """
    Benchmark inference latency of the trained agent.

    Usage:
        bench = LatencyBenchmark(config)
        results = bench.run(agent, sample_obs)
    """

    def __init__(self, config: Config):
        self.cfg = config.deployment
        self.eval_cfg = config.evaluation
        self.n_runs = self.eval_cfg.latency_n_runs
        self.warmup = self.eval_cfg.latency_warmup

    def run(self, agent, sample_observation: np.ndarray) -> dict:
        """
        Benchmark agent inference latency.

        Args:
            agent: Trained SB3 PPO agent
            sample_observation: Single observation vector (obs_dim,)

        Returns:
            Dict with latency statistics in milliseconds
        """
        obs = sample_observation.copy()

        # Warmup (JIT compilation, cache warming)
        for _ in range(self.warmup):
            agent.predict(obs, deterministic=True)

        # Benchmark
        latencies = []
        for _ in range(self.n_runs):
            start = time.perf_counter_ns()
            agent.predict(obs, deterministic=True)
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1e6)  # ns → ms

        latencies = np.array(latencies)

        results = {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "std_ms": float(np.std(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "throughput_hz": float(1000.0 / np.mean(latencies)),
            "n_runs": self.n_runs,
            "meets_target": bool(np.percentile(latencies, 95) <= self.cfg.target_latency_ms),
            "target_ms": self.cfg.target_latency_ms,
        }

        logger.info(
            f"Latency benchmark: mean={results['mean_ms']:.2f}ms, "
            f"p95={results['p95_ms']:.2f}ms, "
            f"throughput={results['throughput_hz']:.0f}Hz, "
            f"meets_target={'YES' if results['meets_target'] else 'NO'}"
        )
        return results

    def run_component_breakdown(self, agent, sample_observation: np.ndarray) -> dict:
        """
        Benchmark individual pipeline components separately.

        Breaks down: feature extraction vs policy head inference.
        """
        obs = sample_observation.copy()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        policy = agent.policy
        policy.eval()

        # Feature extraction latency
        fe_latencies = []
        for _ in range(self.n_runs):
            start = time.perf_counter_ns()
            with torch.no_grad():
                features = policy.extract_features(obs_tensor, policy.features_extractor)
            end = time.perf_counter_ns()
            fe_latencies.append((end - start) / 1e6)

        # Policy head latency
        ph_latencies = []
        with torch.no_grad():
            features = policy.extract_features(obs_tensor, policy.features_extractor)
        for _ in range(self.n_runs):
            start = time.perf_counter_ns()
            with torch.no_grad():
                policy.action_net(policy.mlp_extractor.forward_actor(features))
            end = time.perf_counter_ns()
            ph_latencies.append((end - start) / 1e6)

        return {
            "feature_extraction": {
                "mean_ms": float(np.mean(fe_latencies)),
                "p95_ms": float(np.percentile(fe_latencies, 95)),
            },
            "policy_head": {
                "mean_ms": float(np.mean(ph_latencies)),
                "p95_ms": float(np.percentile(ph_latencies, 95)),
            },
        }
