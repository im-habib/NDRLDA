"""
Actor-Critic network for PPO-based vigilance control.

Architecture overview:
    observation → SharedEncoder → TemporalModule → state s_t
    s_t → ActorHead → π(a|s)     [policy: action probabilities]
    s_t → CriticHead → V(s)      [value: expected cumulative reward]

The shared encoder + temporal module are trained jointly by both heads,
enabling efficient feature learning from the combined policy gradient
and value function loss.

Integration with Stable-Baselines3:
    Uses SB3's ActorCriticPolicy base with custom feature extractor
    that wraps our encoder + temporal pipeline.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from config.settings import Config
from models.encoders import create_encoder
from models.temporal import create_temporal


class VigilanceFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom SB3 feature extractor wrapping encoder + temporal module.

    SB3 expects a feature extractor that maps observation → feature vector.
    We reshape the flat observation back into (window_size, feature_dim),
    run it through the encoder per-timestep, then through the temporal module.

    Output: state representation s_t of dimension = temporal.output_dim
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        config: Config,
    ):
        # Compute feature dimensions
        self.window_size = config.preprocessing.window_size
        # obs_dim = feature_dim * window_size + 3 (aux features)
        obs_dim = observation_space.shape[0]
        self.raw_feature_dim = (obs_dim - 3) // self.window_size
        self.aux_dim = 3

        # Create encoder
        encoder = create_encoder(
            config.encoder.encoder_type,
            self.raw_feature_dim,
            config.encoder,
        )

        # Create temporal module
        temporal = create_temporal(
            config.temporal.temporal_type,
            encoder.output_dim,
            config.temporal,
        )

        # Final output dim = temporal output + aux features
        features_dim = temporal.output_dim + self.aux_dim

        super().__init__(observation_space, features_dim=features_dim)

        self.encoder = encoder
        self.temporal = temporal

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (B, obs_dim) — flat observation from env

        Returns: (B, features_dim) — state representation s_t
        """
        B = observations.shape[0]

        # Split observation into windowed features and auxiliary
        flat_features = observations[:, :-self.aux_dim]  # (B, W*D)
        aux = observations[:, -self.aux_dim:]             # (B, 3)

        # Reshape to (B, W, D)
        windowed = flat_features.reshape(B, self.window_size, self.raw_feature_dim)

        # Encode each timestep: (B, W, D) → (B, W, encoder_dim)
        B, W, D = windowed.shape
        flat_input = windowed.reshape(B * W, D)
        encoded_flat = self.encoder(flat_input)  # (B*W, encoder_dim)
        encoded = encoded_flat.reshape(B, W, -1)  # (B, W, encoder_dim)

        # Temporal modeling: (B, W, encoder_dim) → (B, temporal_dim)
        temporal_state = self.temporal(encoded)

        # Concatenate with auxiliary features
        state = torch.cat([temporal_state, aux], dim=-1)  # (B, features_dim)
        return state


class VigilanceActorCritic(nn.Module):
    """
    Standalone actor-critic network (non-SB3 usage).

    For direct PyTorch training loops or custom PPO implementations.

    Architecture:
        obs → feature_extractor → s_t
        s_t → actor_head → logits → Categorical → π(a|s)
        s_t → critic_head → V(s)
    """

    def __init__(self, observation_space: gym.spaces.Box, config: Config):
        super().__init__()
        self.config = config

        # Feature extractor (shared encoder + temporal)
        self.feature_extractor = VigilanceFeatureExtractor(observation_space, config)
        feat_dim = self.feature_extractor.features_dim

        # Actor head: policy π(a|s)
        actor_layers = []
        prev = feat_dim
        for dim in config.ppo.policy_hidden_dims:
            actor_layers.extend([nn.Linear(prev, dim), nn.Tanh()])
            prev = dim
        actor_layers.append(nn.Linear(prev, config.env.num_actions))
        self.actor_head = nn.Sequential(*actor_layers)

        # Critic head: value V(s)
        critic_layers = []
        prev = feat_dim
        for dim in config.ppo.value_hidden_dims:
            critic_layers.extend([nn.Linear(prev, dim), nn.Tanh()])
            prev = dim
        critic_layers.append(nn.Linear(prev, 1))
        self.critic_head = nn.Sequential(*critic_layers)

    def forward(self, obs: torch.Tensor):
        """
        Returns:
            action_logits: (B, num_actions)
            value: (B, 1)
        """
        features = self.feature_extractor(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        """
        Sample action from policy and return log_prob, entropy, value.

        Used in PPO rollout collection and update steps.
        """
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)
