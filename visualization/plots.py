"""
Visualization module for vigilance control results.

Generates:
    - Per-subject vigilance trajectory plots (PERCLOS + actions + rewards over time)
    - Cross-subject performance summary bar charts
    - Reward learning curves
    - Intervention heatmaps
    - Action distribution pie/bar charts
    - Policy learning progress visualization
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from config.settings import Config

logger = logging.getLogger(__name__)

# Consistent color scheme
ACTION_COLORS = {
    0: "#2ecc71",  # no intervention — green
    1: "#f1c40f",  # soft visual — yellow
    2: "#e67e22",  # audio — orange
    3: "#e74c3c",  # strong haptic — red
    4: "#8e44ad",  # emergency — purple
}
ACTION_LABELS = {
    0: "No Intervention",
    1: "Soft Visual",
    2: "Audio Alert",
    3: "Strong Haptic+Audio",
    4: "Emergency Rest",
}


class VigilancePlotter:
    """
    Visualization generator for vigilance control experiments.

    Usage:
        plotter = VigilancePlotter(config)
        plotter.plot_subject_trajectory(history, subject_id, save_dir)
        plotter.plot_cross_subject_summary(aggregated_metrics, save_dir)
    """

    def __init__(self, config: Config):
        self.cfg = config
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        })

    def plot_subject_trajectory(
        self,
        history: dict,
        subject_id: int,
        save_dir: Path | str | None = None,
        show: bool = False,
    ):
        """
        Plot vigilance trajectory for a single subject.

        Three-panel plot:
            Top: PERCLOS trajectory (controlled vs baseline) with danger zones
            Middle: Action timeline (color-coded interventions)
            Bottom: Cumulative reward curve
        """
        perclos = np.array(history["perclos"])
        actions = np.array(history["actions"])
        rewards = np.array(history["rewards"])
        baseline = np.array(history.get("baseline_perclos", []))
        T = len(perclos)
        timesteps = np.arange(T)

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2], hspace=0.3)

        # --- Panel 1: PERCLOS Trajectory ---
        ax1 = fig.add_subplot(gs[0])
        if len(baseline) == T:
            ax1.plot(timesteps, baseline, color="#bdc3c7", alpha=0.7,
                     linewidth=1, label="Baseline (no intervention)")
        ax1.plot(timesteps, perclos, color="#2980b9", linewidth=1.5,
                 label="Controlled (with agent)")

        # Danger zone shading
        ax1.axhline(y=self.cfg.env.perclos_danger, color="#e74c3c",
                     linestyle="--", alpha=0.5, label=f"Danger ({self.cfg.env.perclos_danger})")
        ax1.axhline(y=self.cfg.env.perclos_safe, color="#2ecc71",
                     linestyle="--", alpha=0.5, label=f"Safe ({self.cfg.env.perclos_safe})")
        ax1.fill_between(timesteps, self.cfg.env.perclos_danger, 1.0,
                         alpha=0.1, color="#e74c3c")

        ax1.set_ylabel("PERCLOS")
        ax1.set_title(f"Subject {subject_id} — Vigilance Trajectory")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_ylim(-0.05, 1.05)

        # --- Panel 2: Action Timeline ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        for t in range(T):
            ax2.bar(t, 1, width=1.0, color=ACTION_COLORS[actions[t]], alpha=0.8)

        legend_elements = [Patch(facecolor=c, label=ACTION_LABELS[a])
                           for a, c in ACTION_COLORS.items()]
        ax2.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=3)
        ax2.set_ylabel("Action")
        ax2.set_yticks([])

        # --- Panel 3: Cumulative Reward ---
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        cum_reward = np.cumsum(rewards)
        ax3.plot(timesteps, cum_reward, color="#8e44ad", linewidth=1.5)
        ax3.fill_between(timesteps, 0, cum_reward, alpha=0.15, color="#8e44ad")
        ax3.set_xlabel("Timestep")
        ax3.set_ylabel("Cumulative Reward")
        ax3.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

        fig.suptitle(f"NDRLDA — Subject {subject_id} Evaluation", fontsize=14, y=1.01)
        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / f"subject_{subject_id}_trajectory.png",
                        bbox_inches="tight")
            logger.info(f"Saved trajectory plot for subject {subject_id}")
        if show:
            plt.show()
        plt.close(fig)

    def plot_cross_subject_summary(
        self,
        aggregated: dict,
        save_dir: Path | str | None = None,
        show: bool = False,
    ):
        """
        Bar chart comparing metrics across subjects.

        Shows mean ± std for key metrics.
        """
        display_metrics = [
            ("mean_perclos", "Mean PERCLOS", True),
            ("danger_ratio", "Danger Ratio", True),
            ("intervention_rate", "Intervention Rate", False),
            ("alert_efficiency", "Alert Efficiency", False),
            ("false_alert_rate", "False Alert Rate", True),
            ("cumulative_reward", "Cumulative Reward", False),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for idx, (key, label, lower_better) in enumerate(display_metrics):
            ax = axes[idx]
            if key in aggregated and isinstance(aggregated[key], dict):
                data = aggregated[key]
                values = data.get("values", [])
                if values:
                    subjects = list(range(1, len(values) + 1))
                    colors = ["#e74c3c" if lower_better else "#2ecc71"] * len(values)
                    ax.bar(subjects, values, color=colors, alpha=0.7)
                    ax.axhline(y=data["mean"], color="#2c3e50", linestyle="--",
                               linewidth=1.5, label=f"Mean: {data['mean']:.3f}")
                    ax.set_xlabel("Test Subject")
                ax.set_title(label)
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label)

        fig.suptitle("Cross-Subject Evaluation Summary", fontsize=14)
        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / "cross_subject_summary.png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_reward_curve(
        self,
        training_stats: list[dict],
        save_dir: Path | str | None = None,
        show: bool = False,
    ):
        """Plot training reward learning curve with moving average."""
        rewards = [s.get("cumulative_reward", 0) for s in training_stats]
        if not rewards:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        episodes = np.arange(len(rewards))
        ax.plot(episodes, rewards, alpha=0.3, color="#3498db", label="Episode Reward")

        # Moving average
        window = min(50, len(rewards) // 5 + 1)
        if window > 1:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window - 1, len(rewards)), ma,
                    color="#e74c3c", linewidth=2, label=f"MA({window})")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Training Reward Curve")
        ax.legend()
        plt.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(save_dir) / "reward_curve.png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_intervention_heatmap(
        self,
        per_subject_histories: list[dict],
        save_dir: Path | str | None = None,
        show: bool = False,
    ):
        """
        Heatmap of intervention intensity across subjects and time.

        Rows = subjects, columns = time (binned), color = mean action level.
        """
        n_bins = 50  # time bins
        n_subjects = len(per_subject_histories)
        heatmap = np.zeros((n_subjects, n_bins))

        for i, hist in enumerate(per_subject_histories):
            actions = np.array(hist["actions"])
            T = len(actions)
            bin_edges = np.linspace(0, T, n_bins + 1, dtype=int)
            for b in range(n_bins):
                start, end = bin_edges[b], bin_edges[b + 1]
                if end > start:
                    heatmap[i, b] = np.mean(actions[start:end])

        fig, ax = plt.subplots(figsize=(14, max(4, n_subjects * 0.5)))
        im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", vmin=0, vmax=4)
        ax.set_xlabel("Time (normalized)")
        ax.set_ylabel("Subject")
        ax.set_title("Intervention Intensity Heatmap")
        plt.colorbar(im, ax=ax, label="Mean Action Level")
        plt.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(save_dir) / "intervention_heatmap.png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
