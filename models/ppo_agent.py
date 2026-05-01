"""
PPO agent creation and configuration for Stable-Baselines3.

PPO Algorithm Summary
=====================

Proximal Policy Optimization (Schulman et al., 2017) maximizes a clipped
surrogate objective that constrains policy updates for stability.

Clipped Surrogate Objective:
    L^CLIP(θ) = E_t[ min( r_t(θ) · Â_t, clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]

    where:
        r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   [probability ratio]
        Â_t = GAE advantage estimate
        ε = clip_range (e.g., 0.2)

Value Function Loss:
    L^VF = (V_θ(s_t) - V_t^target)²

Entropy Bonus (exploration):
    H[π_θ](s_t) = -Σ_a π_θ(a|s_t) log π_θ(a|s_t)

Total Loss:
    L = -L^CLIP + c1 · L^VF - c2 · H[π_θ]

Generalized Advantage Estimation (GAE):
    Â_t^GAE(γ,λ) = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}
    δ_t = r_t + γ · V(s_{t+1}) - V(s_t)     [TD residual]

Training Loop:
    1. Collect rollout of n_steps using current policy
    2. Compute GAE advantages and returns
    3. For n_epochs:
        a. Shuffle rollout into mini-batches of batch_size
        b. Compute clipped loss, value loss, entropy bonus
        c. Update θ via gradient descent
    4. Repeat from 1 until total_timesteps reached
"""

import logging

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from config.settings import Config
from models.actor_critic import VigilanceFeatureExtractor

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> str:
    """Resolve 'auto' device to best available."""
    if device_str != "auto":
        return device_str
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_ppo_agent(
    env,
    config: Config,
    tensorboard_log: str | None = None,
) -> PPO:
    """
    Create a PPO agent with custom vigilance-aware architecture.

    Args:
        env: Gymnasium environment (VigilanceEnv)
        config: Master configuration
        tensorboard_log: Path for tensorboard logs (None to disable)

    Returns:
        Configured SB3 PPO agent ready for training
    """
    device = _resolve_device(config.training.device)
    logger.info(f"Creating PPO agent on device: {device}")

    # Custom policy kwargs — inject our feature extractor
    policy_kwargs = dict(
        features_extractor_class=VigilanceFeatureExtractor,
        features_extractor_kwargs=dict(config=config),
        net_arch=dict(
            pi=list(config.ppo.policy_hidden_dims),
            vf=list(config.ppo.value_hidden_dims),
        ),
        activation_fn=_get_activation(config.ppo.activation),
        ortho_init=config.ppo.ortho_init,
    )

    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=config.training.verbose,
        seed=config.training.seed,
        device=device,
    )

    # Log model summary
    total_params = sum(p.numel() for p in agent.policy.parameters())
    trainable = sum(p.numel() for p in agent.policy.parameters() if p.requires_grad)
    logger.info(f"PPO agent created: {total_params:,} params ({trainable:,} trainable)")

    return agent


def _get_activation(name: str):
    """Map activation name to PyTorch class."""
    activations = {
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
        "gelu": torch.nn.GELU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations)}")
    return activations[name]
