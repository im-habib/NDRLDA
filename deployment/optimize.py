"""
Model optimization for edge deployment.

Supports:
    - ONNX export for cross-platform inference
    - Model quantization (dynamic quantization)
    - Model size analysis
    - Inference optimization recommendations
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config.settings import Config

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Optimizes trained model for edge deployment in vehicles.

    Pipeline:
        PyTorch model → ONNX export → optional quantization → size/latency analysis

    Usage:
        optimizer = ModelOptimizer(config)
        optimizer.export_onnx(agent, sample_obs)
        optimizer.quantize_model(agent)
        report = optimizer.analyze(agent, sample_obs)
    """

    def __init__(self, config: Config):
        self.cfg = config.deployment
        self.cfg.export_dir.mkdir(parents=True, exist_ok=True)

    def export_onnx(
        self,
        agent,
        sample_observation: np.ndarray,
        output_path: Path | str | None = None,
    ) -> Path:
        """
        Export policy network to ONNX format.

        Args:
            agent: Trained SB3 PPO agent
            sample_observation: Example observation for tracing
            output_path: Custom output path (default: export_dir/policy.onnx)

        Returns:
            Path to exported ONNX file
        """
        output_path = Path(output_path) if output_path else self.cfg.export_dir / "policy.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        policy = agent.policy
        policy.eval()

        # Create wrapper that takes observation and returns action
        class PolicyWrapper(nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            def forward(self, obs):
                features = self.policy.extract_features(obs, self.policy.features_extractor)
                latent_pi = self.policy.mlp_extractor.forward_actor(features)
                action_logits = self.policy.action_net(latent_pi)
                return action_logits

        wrapper = PolicyWrapper(policy)
        wrapper.eval()

        sample_tensor = torch.FloatTensor(sample_observation).unsqueeze(0)

        torch.onnx.export(
            wrapper,
            sample_tensor,
            str(output_path),
            export_params=True,
            opset_version=self.cfg.onnx_opset,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action_logits": {0: "batch_size"},
            },
        )

        # Validate
        import onnx
        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model exported: {output_path} ({size_mb:.2f} MB)")
        return output_path

    def quantize_model(self, agent) -> nn.Module:
        """
        Apply dynamic quantization to policy network.

        Quantizes Linear layers to int8, reducing model size ~4x
        with minimal accuracy loss.

        Returns:
            Quantized PyTorch model
        """
        policy = agent.policy.cpu()
        policy.eval()

        quantized = torch.ao.quantization.quantize_dynamic(
            policy,
            {nn.Linear},
            dtype=torch.qint8,
        )

        # Compare sizes
        original_size = sum(p.numel() * p.element_size() for p in policy.parameters())
        # Quantized size estimation
        quantized_size = sum(
            p.numel() * (1 if p.dtype == torch.qint8 else p.element_size())
            for p in quantized.parameters()
        )

        logger.info(
            f"Quantization: {original_size / 1e6:.2f}MB → ~{quantized_size / 1e6:.2f}MB "
            f"({quantized_size / original_size:.1%})"
        )
        return quantized

    def analyze(self, agent, sample_observation: np.ndarray) -> dict:
        """
        Comprehensive deployment analysis.

        Returns report with model size, parameter count, layer breakdown,
        and deployment recommendations.
        """
        policy = agent.policy

        # Parameter analysis
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in policy.parameters()) / 1e6

        # Layer-wise breakdown
        layer_info = []
        for name, module in policy.named_modules():
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                layer_info.append({
                    "name": name,
                    "type": type(module).__name__,
                    "params": params,
                })

        # FLOPs estimation (rough)
        flops = 0
        for name, module in policy.named_modules():
            if isinstance(module, nn.Linear):
                flops += 2 * module.in_features * module.out_features

        report = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": model_size_mb,
            "estimated_flops": flops,
            "layer_breakdown": layer_info,
            "input_dim": sample_observation.shape[0],
            "recommendations": self._get_recommendations(model_size_mb, flops),
        }

        logger.info(
            f"Model analysis: {total_params:,} params, {model_size_mb:.2f}MB, "
            f"~{flops:,} FLOPs"
        )
        return report

    def _get_recommendations(self, size_mb: float, flops: int) -> list[str]:
        """Generate deployment recommendations based on model characteristics."""
        recs = []
        if size_mb > 50:
            recs.append("Model >50MB — consider pruning or knowledge distillation")
        if size_mb > 10:
            recs.append("Apply INT8 quantization via ONNX Runtime or TensorRT")
        if flops > 1e8:
            recs.append("High FLOPs — consider replacing attention with efficient alternatives")
        recs.append("Use ONNX Runtime for optimized CPU inference on edge devices")
        recs.append("Batch size 1 inference with pre-allocated tensors for lowest latency")
        if size_mb < 5:
            recs.append("Model small enough for microcontroller deployment (e.g., STM32)")
        return recs
