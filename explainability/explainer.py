"""
Explainability module for vigilance control decisions.

Implements:
    - Feature importance analysis (permutation-based)
    - Attention visualization (for attention-based encoder/temporal)
    - SHAP value computation for action decisions
    - Decision explanation generation
"""

import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from config.settings import Config

logger = logging.getLogger(__name__)


class VigilanceExplainer:
    """
    Explains why the agent issues specific alerts.

    Provides feature importance, attention maps, and SHAP analysis
    to make the agent's decisions interpretable for research reporting.

    Usage:
        explainer = VigilanceExplainer(config)
        importance = explainer.feature_importance(agent, observations)
        explainer.plot_importance(importance, save_dir)
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> list[str]:
        """Build human-readable feature names for the observation vector."""
        names = []
        bands = self.cfg.data.band_names
        # EEG channels (17 standard + 4 forehead)
        for ch in range(self.cfg.data.total_eeg_channels):
            prefix = f"EEG_{ch}" if ch < 17 else f"FH_{ch - 17}"
            for band in bands:
                names.append(f"{prefix}_{band}")
        # EOG features
        for i in range(self.cfg.data.eog_dim):
            names.append(f"EOG_{i}")
        return names

    def feature_importance(
        self,
        agent,
        observations: np.ndarray,
        n_repeats: int = 10,
    ) -> dict:
        """
        Permutation-based feature importance.

        For each feature, shuffle its values across the observation batch
        and measure the change in predicted action distribution.

        Args:
            agent: Trained PPO agent
            observations: (N, obs_dim) batch of observations
            n_repeats: Number of permutation repeats

        Returns:
            Dict mapping feature_group → importance_score
        """
        obs = observations.copy()
        N, D = obs.shape

        # Baseline action distribution
        baseline_actions = np.array([
            agent.predict(obs[i], deterministic=True)[0] for i in range(N)
        ])

        # Feature dimension (minus aux features at end)
        feat_dim = len(self.feature_names)
        window_size = self.cfg.preprocessing.window_size

        # Group importance by feature (average across window positions)
        importances = np.zeros(feat_dim)

        for f_idx in range(feat_dim):
            disruptions = []
            for _ in range(n_repeats):
                obs_perm = obs.copy()
                # Shuffle this feature across all window positions
                for w in range(window_size):
                    col_idx = w * feat_dim + f_idx
                    if col_idx < D - 3:  # don't touch aux features
                        np.random.shuffle(obs_perm[:, col_idx])

                perm_actions = np.array([
                    agent.predict(obs_perm[i], deterministic=True)[0] for i in range(N)
                ])
                disruption = np.mean(perm_actions != baseline_actions)
                disruptions.append(disruption)
            importances[f_idx] = np.mean(disruptions)

        # Normalize
        total = importances.sum()
        if total > 0:
            importances = importances / total

        # Group by modality
        result = {
            "per_feature": dict(zip(self.feature_names, importances.tolist())),
            "by_modality": self._group_by_modality(importances),
            "by_band": self._group_by_band(importances),
            "raw_importances": importances,
        }
        return result

    def _group_by_modality(self, importances: np.ndarray) -> dict:
        """Aggregate importance by signal modality."""
        eeg_dim = self.cfg.data.eeg_channels * self.cfg.data.frequency_bands
        fh_dim = self.cfg.data.forehead_channels * self.cfg.data.frequency_bands
        eog_start = eeg_dim + fh_dim

        return {
            "EEG (17ch)": float(importances[:eeg_dim].sum()),
            "Forehead EEG (4ch)": float(importances[eeg_dim:eog_start].sum()),
            "EOG (36d)": float(importances[eog_start:].sum()),
        }

    def _group_by_band(self, importances: np.ndarray) -> dict:
        """Aggregate EEG importance by frequency band."""
        bands = self.cfg.data.band_names
        n_channels = self.cfg.data.total_eeg_channels
        result = {}
        for b_idx, band in enumerate(bands):
            # Features at indices: channel * 5 + b_idx for each channel
            indices = [ch * len(bands) + b_idx for ch in range(n_channels)]
            valid = [i for i in indices if i < len(importances)]
            result[band] = float(importances[valid].sum()) if valid else 0.0
        return result

    def extract_attention_weights(self, agent, observation: np.ndarray) -> np.ndarray | None:
        """
        Extract attention weights from attention-based encoder/temporal module.

        Returns attention weight matrix if available, None otherwise.
        """
        policy = agent.policy
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

        # Try to access attention weights via hooks
        attention_weights = []

        def hook_fn(module, input, output):
            if hasattr(output, "shape") and output.dim() >= 3:
                attention_weights.append(output.detach().cpu().numpy())

        # Register hooks on transformer layers
        hooks = []
        for name, module in policy.named_modules():
            if "self_attn" in name or "multihead" in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            policy.predict(obs_tensor, deterministic=True)

        for h in hooks:
            h.remove()

        if attention_weights:
            return attention_weights[0].squeeze()
        return None

    def plot_importance(
        self,
        importance_result: dict,
        save_dir: Path | str | None = None,
        top_k: int = 20,
        show: bool = False,
    ):
        """
        Plot feature importance analysis.

        Two subplots:
            Left: Top-K individual feature importances
            Right: Modality-level and band-level importance
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Top-K features
        per_feat = importance_result["per_feature"]
        sorted_feats = sorted(per_feat.items(), key=lambda x: x[1], reverse=True)[:top_k]
        names, values = zip(*sorted_feats) if sorted_feats else ([], [])
        y_pos = np.arange(len(names))
        ax1.barh(y_pos, values, color="#3498db", alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("Relative Importance")
        ax1.set_title(f"Top-{top_k} Feature Importances")

        # Right: Modality + band breakdown
        modality = importance_result.get("by_modality", {})
        band = importance_result.get("by_band", {})

        # Stacked: modality on top, band on bottom
        ax2_top = ax2
        labels = list(modality.keys())
        values = list(modality.values())
        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        bars = ax2_top.bar(labels, values, color=colors[:len(labels)], alpha=0.8)
        ax2_top.set_title("Importance by Modality")
        ax2_top.set_ylabel("Relative Importance")

        # Add band breakdown as text annotation
        band_text = "\n".join(f"  {k}: {v:.3f}" for k, v in band.items())
        ax2_top.text(0.95, 0.95, f"By Band:\n{band_text}",
                     transform=ax2_top.transAxes, fontsize=8,
                     verticalalignment="top", horizontalalignment="right",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fig.suptitle("NDRLDA — Feature Importance Analysis", fontsize=14)
        plt.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(save_dir) / "feature_importance.png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def explain_decision(
        self,
        agent,
        observation: np.ndarray,
        perclos: float,
    ) -> str:
        """
        Generate human-readable explanation for a single decision.

        Useful for debugging and paper examples.
        """
        action, _ = agent.predict(observation, deterministic=True)
        action = int(action)

        action_name = self.cfg.env.action_names[action]

        if perclos > self.cfg.env.perclos_danger:
            state_desc = "DANGEROUS drowsiness level"
        elif perclos > self.cfg.env.perclos_warning:
            state_desc = "moderate drowsiness"
        elif perclos > self.cfg.env.perclos_safe:
            state_desc = "mild drowsiness"
        else:
            state_desc = "alert state"

        explanation = (
            f"Decision: {action_name} (action={action})\n"
            f"PERCLOS: {perclos:.3f} — {state_desc}\n"
            f"Rationale: "
        )

        if action == 0:
            explanation += "Driver vigilance within safe range; no intervention needed."
        elif action <= 2:
            explanation += (
                f"Moderate drowsiness detected (PERCLOS={perclos:.3f}). "
                f"Issuing {'visual' if action == 1 else 'audio'} alert to gently restore attention."
            )
        elif action <= 4:
            explanation += (
                f"{'High' if action == 3 else 'Critical'} drowsiness level. "
                f"Strong intervention required to prevent safety hazard."
            )

        return explanation
