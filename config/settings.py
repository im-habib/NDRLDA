"""
Centralized configuration for the NDRLDA pipeline.

All hyperparameters live here as dataclasses. Modules import Config
and access nested fields (e.g., config.ppo.learning_rate).
CLI entry points override individual fields after construction.
"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """SEED-VIG dataset paths and structure."""
    data_dir: str = "data"
    primary_eeg_feature: str = "de_LDS"
    eog_method: str = "features_table_ica"
    eeg_channels: int = 17
    forehead_channels: int = 0
    frequency_bands: int = 5
    eog_dim: int = 36
    sample_rate_hz: float = 1.0
    band_names: tuple[str, ...] = ("delta", "theta", "alpha", "beta", "gamma")

    @property
    def total_eeg_channels(self) -> int:
        return self.eeg_channels + self.forehead_channels


@dataclass
class PreprocessingConfig:
    """Feature processing: normalization, smoothing, windowing, reduction."""
    norm_method: str = "robust"
    smoothing_method: str = "lds"
    smoothing_window: int = 5
    lds_transition_cov: float = 0.01
    lds_observation_cov: float = 0.1
    window_size: int = 10
    window_stride: int = 1
    use_pca: bool = False
    pca_components: int = 32
    use_autoencoder: bool = False
    autoencoder_latent_dim: int = 32
    autoencoder_lr: float = 1e-3
    autoencoder_epochs: int = 50


@dataclass
class EnvironmentConfig:
    """Gymnasium environment parameters."""
    num_actions: int = 5
    action_names: tuple[str, ...] = (
        "No Alert",
        "Soft Visual",
        "Audio Warning",
        "Strong Haptic+Audio",
        "Emergency Rest",
    )
    alert_effect: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.25)
    alert_effect_decay: float = 0.9
    max_episode_steps: int = 500
    perclos_danger: float = 0.50
    perclos_warning: float = 0.35
    perclos_safe: float = 0.20


@dataclass
class RewardConfig:
    """Composite reward function weights and thresholds."""
    tau_safe: float = 0.20
    alpha_safety: float = 2.0
    action_costs: tuple[float, ...] = (0.0, 0.1, 0.3, 0.6, 1.0)
    beta_comfort: float = 0.5
    gamma_efficiency: float = 0.3
    trend_ema_alpha: float = 0.3
    delta_trend: float = 0.5
    alert_fatigue_patience: int = 10
    zeta_alert_fatigue: float = 0.2
    danger_zone_penalty: float = 1.0
    recovery_bonus: float = 0.5
    reward_smoothing_lambda: float = 0.8
    perclos_danger: float = 0.50


@dataclass
class EncoderConfig:
    """Swappable encoder architecture settings."""
    encoder_type: str = "mlp"
    mlp_hidden_dims: tuple[int, ...] = (256, 128)
    mlp_dropout: float = 0.1
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    cnn_kernel_sizes: tuple[int, ...] = (3, 3, 3)
    cnn_dropout: float = 0.1
    attention_heads: int = 4
    attention_dim: int = 128
    attention_layers: int = 2
    attention_dropout: float = 0.1


@dataclass
class TemporalConfig:
    """Swappable temporal module settings."""
    temporal_type: str = "lstm"
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    lstm_bidirectional: bool = False
    transformer_d_model: int = 128
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters (Stable-Baselines3)."""
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 20
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_hidden_dims: tuple[int, ...] = (256, 128)
    value_hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "tanh"
    ortho_init: bool = True


@dataclass
class TrainingConfig:
    """Training pipeline settings."""
    seed: int = 42
    device: str = "auto"
    verbose: int = 1
    test_subjects: tuple[int, ...] = (1, 5, 10)
    val_ratio: float = 0.15
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    use_tensorboard: bool = True
    save_every_n_steps: int = 50_000
    keep_last_n: int = 3
    eval_freq: int = 10_000
    n_eval_episodes: int = 5


@dataclass
class EvaluationConfig:
    """Evaluation and latency benchmarking settings."""
    results_dir: Path = field(default_factory=lambda: Path("results"))
    danger_threshold: float = 0.50
    latency_n_runs: int = 1000
    latency_warmup: int = 50


@dataclass
class DeploymentConfig:
    """ONNX export and deployment settings."""
    export_dir: Path = field(default_factory=lambda: Path("deployment/exports"))
    onnx_opset: int = 17
    target_latency_ms: float = 10.0


@dataclass
class Config:
    """Master configuration — single source of truth for all hyperparameters."""
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
