"""
Real-time simulation pipeline.

Simulates continuous driver monitoring as it would work in a vehicle:
    incoming EEG window → state generation → PPO action selection
    → alert generation → reward computation → next state

Supports both PyTorch and ONNX inference backends.
"""

import logging
import time
from pathlib import Path
from collections import deque

import numpy as np

from config.settings import Config
from preprocessing.feature_processor import FeatureProcessor
from preprocessing.data_loader import SubjectData

logger = logging.getLogger(__name__)


class RealtimeSimulator:
    """
    Simulates real-time driver monitoring pipeline.

    Streams windowed features as if arriving from a live EEG device,
    processes them through the agent, and generates alerts.

    Usage:
        simulator = RealtimeSimulator(config, agent, processor)
        results = simulator.run(subject_data, speed=1.0)
    """

    def __init__(self, config: Config, agent, processor: FeatureProcessor):
        self.cfg = config
        self.agent = agent
        self.processor = processor
        self.window_size = config.preprocessing.window_size
        self.sample_rate = config.data.sample_rate_hz

    def run(
        self,
        subject_data: SubjectData,
        speed: float = 1.0,
        realtime: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        Run real-time simulation on subject data.

        Args:
            subject_data: Raw subject data (pre-processing applied on-the-fly)
            speed: Simulation speed multiplier (1.0 = real-time)
            realtime: If True, actually sleep between steps to simulate real-time
            verbose: Print alerts as they occur

        Returns:
            Simulation results with timeline, alerts, and latencies
        """
        features = subject_data.get_fused_features()
        perclos = subject_data.perclos
        T = len(features)

        if T < self.window_size:
            raise ValueError(f"Subject has {T} samples, need at least {self.window_size}")

        # Build windowed observations on-the-fly (simulating streaming)
        processed = self.processor.transform(subject_data)

        timeline = []
        alerts = []
        latencies = []
        feature_buffer = deque(maxlen=self.window_size)

        # Observation buffer for auxiliary features
        prev_perclos = float(perclos[0])

        for step in range(processed.num_windows):
            sim_time = step / self.sample_rate

            # Simulate incoming data
            window_features = processed.features[step]  # (W, D)
            current_perclos = float(processed.perclos_mean[step])

            # Build observation (same as env)
            features_flat = window_features.flatten()
            step_fraction = step / max(processed.num_windows - 1, 1)

            # Simple windowed mean for PERCLOS
            start_idx = max(0, step - self.window_size + 1)
            window_perclos_mean = float(np.mean(processed.perclos_mean[start_idx:step + 1]))

            aux = np.array([current_perclos, window_perclos_mean, step_fraction], dtype=np.float32)
            observation = np.concatenate([features_flat, aux])

            # Inference
            t_start = time.perf_counter_ns()
            action, _ = self.agent.predict(observation, deterministic=True)
            t_end = time.perf_counter_ns()
            latency_ms = (t_end - t_start) / 1e6
            latencies.append(latency_ms)

            action = int(action)
            action_name = self.cfg.env.action_names[action]

            # Record
            record = {
                "time_s": sim_time,
                "step": step,
                "perclos": current_perclos,
                "action": action,
                "action_name": action_name,
                "latency_ms": latency_ms,
            }
            timeline.append(record)

            if action > 0:
                alert = {
                    "time_s": sim_time,
                    "level": action,
                    "name": action_name,
                    "perclos": current_perclos,
                }
                alerts.append(alert)
                if verbose:
                    severity = ["", "INFO", "WARN", "ALERT", "EMERGENCY"][action]
                    logger.info(
                        f"[{sim_time:6.1f}s] [{severity}] {action_name} "
                        f"(PERCLOS={current_perclos:.3f}, latency={latency_ms:.1f}ms)"
                    )

            prev_perclos = current_perclos

            # Real-time pacing
            if realtime:
                time.sleep(1.0 / (self.sample_rate * speed))

        # Summary
        latencies_arr = np.array(latencies)
        results = {
            "timeline": timeline,
            "alerts": alerts,
            "summary": {
                "total_steps": processed.num_windows,
                "total_alerts": len(alerts),
                "alert_rate": len(alerts) / processed.num_windows,
                "mean_latency_ms": float(np.mean(latencies_arr)),
                "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
                "max_latency_ms": float(np.max(latencies_arr)),
                "throughput_hz": float(1000.0 / np.mean(latencies_arr)),
                "mean_perclos": float(np.mean([r["perclos"] for r in timeline])),
                "simulation_duration_s": processed.num_windows / self.sample_rate,
            },
        }

        if verbose:
            s = results["summary"]
            logger.info(
                f"\nSimulation complete: {s['total_steps']} steps, "
                f"{s['total_alerts']} alerts ({s['alert_rate']:.1%}), "
                f"latency={s['mean_latency_ms']:.2f}ms (p95={s['p95_latency_ms']:.2f}ms)"
            )

        return results
