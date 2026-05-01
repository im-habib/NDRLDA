"""
Visualization module for vigilance control results.

Generates:
    - Per-subject vigilance trajectory plots (PERCLOS + actions + rewards over time)
    - Cross-subject performance summary bar charts
    - Reward learning curves
    - Per-subject dashboard storyboards
    - PERCLOS risk heatmaps
    - Intervention heatmaps
    - Action distribution pie/bar charts
    - Metric radar charts
    - Policy trade-off scatter plots
    - Policy learning progress visualization
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
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
ZONE_COLORS = {
    "safe": "#c8f7dc",
    "watch": "#fff1b8",
    "danger": "#ffd6d2",
}
METRIC_COLORS = {
    "blue": "#3498db",
    "teal": "#1abc9c",
    "rose": "#e74c3c",
    "violet": "#8e44ad",
    "ink": "#2c3e50",
    "muted": "#95a5a6",
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
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.edgecolor": "#d7dde8",
            "grid.color": "#e8ecf3",
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

    def _save_or_show(
        self,
        fig: plt.Figure,
        save_dir: Path | str | None,
        filename: str,
        show: bool,
    ):
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / filename, bbox_inches="tight")
            logger.info(f"Saved plot: {save_dir / filename}")
        if show:
            plt.show()
        plt.close(fig)

    def _shade_perclos_zones(self, ax: plt.Axes, xmax: int | float):
        safe = self.cfg.env.perclos_safe
        warning = self.cfg.env.perclos_warning
        danger = self.cfg.env.perclos_danger
        ax.axhspan(0, safe, color=ZONE_COLORS["safe"], alpha=0.35, zorder=0)
        ax.axhspan(safe, danger, color=ZONE_COLORS["watch"], alpha=0.25, zorder=0)
        ax.axhspan(danger, 1.0, color=ZONE_COLORS["danger"], alpha=0.30, zorder=0)
        ax.axhline(safe, color="#27ae60", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(warning, color="#f39c12", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(danger, color="#c0392b", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlim(0, max(1, xmax))
        ax.set_ylim(-0.03, 1.03)

    def _rolling_mean(self, values: np.ndarray, window: int = 15) -> np.ndarray:
        if len(values) == 0:
            return values
        window = max(1, min(window, len(values)))
        if window == 1:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="same")

    def _subject_labels(
        self,
        count: int,
        subject_ids: list[int] | tuple[int, ...] | None = None,
    ) -> list[str]:
        if subject_ids:
            return [f"S{sid}" for sid in subject_ids]
        return [f"S{i + 1}" for i in range(count)]

    def _bin_series(self, values: np.ndarray, n_bins: int, reducer=np.mean) -> np.ndarray:
        if len(values) == 0:
            return np.zeros(n_bins)
        edges = np.linspace(0, len(values), n_bins + 1, dtype=int)
        binned = np.zeros(n_bins)
        for idx in range(n_bins):
            start, end = edges[idx], edges[idx + 1]
            if end > start:
                binned[idx] = reducer(values[start:end])
            else:
                binned[idx] = values[min(start, len(values) - 1)]
        return binned

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
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 2], hspace=0.3)

        # --- Panel 1: PERCLOS Trajectory ---
        ax1 = fig.add_subplot(gs[0])
        self._shade_perclos_zones(ax1, T - 1)
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

        # --- Panel 2: Action Timeline ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        for t in range(T):
            ax2.bar(t, 1, width=1.0, color=ACTION_COLORS.get(actions[t], "#7f8c8d"), alpha=0.8)

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

        fig.suptitle(f"NDRLDA — Subject {subject_id} Evaluation", fontsize=14, y=0.98)
        fig.subplots_adjust(top=0.91, bottom=0.08, hspace=0.35)

        self._save_or_show(fig, save_dir, f"subject_{subject_id}_trajectory.png", show)

    def plot_subject_dashboard(
        self,
        history: dict,
        subject_id: int,
        metrics: dict | None = None,
        save_dir: Path | str | None = None,
        show: bool = False,
    ):
        """
        Compact storyboard for one subject.

        The dashboard combines:
            - controlled vs baseline PERCLOS,
            - intervention markers,
            - reward trend,
            - action mix,
            - safe/watch/danger occupancy.
        """
        perclos = np.array(history["perclos"], dtype=float)
        actions = np.array(history["actions"], dtype=int)
        rewards = np.array(history["rewards"], dtype=float)
        baseline = np.array(history.get("baseline_perclos", []), dtype=float)
        timesteps = np.arange(len(perclos))
        metrics = metrics or {}

        fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(3, 4, height_ratios=[0.8, 2.4, 1.7], hspace=0.45, wspace=0.35)

        score_items = [
            (
                "Mean PERCLOS",
                metrics.get("mean_perclos", np.mean(perclos) if len(perclos) else 0.0),
                METRIC_COLORS["blue"],
                "{:.3f}",
            ),
            (
                "Danger Time",
                metrics.get(
                    "danger_ratio",
                    np.mean(perclos > self.cfg.env.perclos_danger) if len(perclos) else 0.0,
                ),
                METRIC_COLORS["rose"],
                "{:.1%}",
            ),
            (
                "Alert Rate",
                metrics.get("intervention_rate", np.mean(actions > 0) if len(actions) else 0.0),
                METRIC_COLORS["teal"],
                "{:.1%}",
            ),
            (
                "Reward",
                metrics.get("cumulative_reward", np.sum(rewards) if len(rewards) else 0.0),
                METRIC_COLORS["violet"],
                "{:.1f}",
            ),
        ]
        for idx, (label, value, color, value_format) in enumerate(score_items):
            ax = fig.add_subplot(gs[0, idx])
            ax.set_facecolor("#ffffff")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#dce3ee")
            ax.set_xticks([])
            ax.set_yticks([])
            value_text = value_format.format(float(value))
            ax.text(0.04, 0.64, label, transform=ax.transAxes, fontsize=8, color="#667085")
            ax.text(0.04, 0.18, value_text, transform=ax.transAxes, fontsize=16,
                    color=color, fontweight="bold")

        ax_trend = fig.add_subplot(gs[1, :3])
        self._shade_perclos_zones(ax_trend, len(perclos) - 1)
        if len(baseline) == len(perclos):
            ax_trend.plot(timesteps, baseline, color="#b8c2cc", linewidth=1.1,
                          alpha=0.75, label="Baseline")
        ax_trend.plot(timesteps, perclos, color=METRIC_COLORS["blue"], linewidth=1.6,
                      label="Controlled")
        intervention_steps = np.where(actions > 0)[0]
        if len(intervention_steps):
            ax_trend.scatter(
                intervention_steps,
                perclos[intervention_steps],
                s=28 + 12 * actions[intervention_steps],
                c=[ACTION_COLORS.get(a, "#7f8c8d") for a in actions[intervention_steps]],
                edgecolors="white",
                linewidths=0.5,
                alpha=0.95,
                label="Intervention",
                zorder=3,
            )
        ax_trend.set_title(f"Subject {subject_id}: vigilance path")
        ax_trend.set_ylabel("PERCLOS")
        ax_trend.legend(loc="upper right", fontsize=8)

        ax_reward = fig.add_subplot(gs[2, :3], sharex=ax_trend)
        reward_smooth = self._rolling_mean(rewards, window=21)
        ax_reward.bar(timesteps, rewards, color="#d8e6f3", width=1.0, label="Step reward")
        ax_reward.plot(timesteps, reward_smooth, color=METRIC_COLORS["violet"],
                       linewidth=1.6, label="Smoothed reward")
        ax_reward.axhline(0, color="#2c3e50", linewidth=0.8, alpha=0.4)
        ax_reward.set_xlabel("Timestep")
        ax_reward.set_ylabel("Reward")
        ax_reward.set_title("Reward signal")
        ax_reward.legend(loc="upper right", fontsize=8)

        ax_actions = fig.add_subplot(gs[1, 3])
        counts = np.array([np.sum(actions == a) for a in ACTION_COLORS], dtype=float)
        colors = [ACTION_COLORS[a] for a in ACTION_COLORS]
        labels = [ACTION_LABELS[a] for a in ACTION_COLORS]
        if counts.sum() > 0:
            wedges, _ = ax_actions.pie(
                counts,
                colors=colors,
                startangle=90,
                wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.5},
            )
            ax_actions.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.92, 0.5),
                              fontsize=7, frameon=False)
        ax_actions.text(0, 0, f"{int(counts.sum())}\nsteps", ha="center", va="center",
                        fontsize=11, color=METRIC_COLORS["ink"], fontweight="bold")
        ax_actions.set_title("Action mix")

        ax_zones = fig.add_subplot(gs[2, 3])
        safe = np.mean(perclos <= self.cfg.env.perclos_safe) if len(perclos) else 0
        danger = np.mean(perclos >= self.cfg.env.perclos_danger) if len(perclos) else 0
        watch = max(0.0, 1.0 - safe - danger)
        zone_values = [safe, watch, danger]
        zone_labels = ["Safe", "Watch", "Danger"]
        zone_colors = [ZONE_COLORS["safe"], ZONE_COLORS["watch"], ZONE_COLORS["danger"]]
        left = 0
        for value, label, color in zip(zone_values, zone_labels, zone_colors):
            ax_zones.barh([0], [value], left=left, color=color, edgecolor="white", height=0.42)
            if value >= 0.08:
                ax_zones.text(left + value / 2, 0, f"{value:.0%}", ha="center",
                              va="center", fontsize=9, color=METRIC_COLORS["ink"])
            left += value
        ax_zones.set_xlim(0, 1)
        ax_zones.set_yticks([])
        ax_zones.set_xlabel("Share of episode")
        ax_zones.set_title("Vigilance zones")
        ax_zones.legend(
            [Patch(facecolor=c, edgecolor="white") for c in zone_colors],
            zone_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            fontsize=7,
            frameon=False,
        )

        fig.suptitle(f"NDRLDA Subject {subject_id} Dashboard", fontsize=15, y=0.98)
        self._save_or_show(fig, save_dir, f"subject_{subject_id}_dashboard.png", show)

    def plot_cross_subject_summary(
        self,
        aggregated: dict,
        save_dir: Path | str | None = None,
        show: bool = False,
        subject_ids: list[int] | tuple[int, ...] | None = None,
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
                    subjects = self._subject_labels(len(values), subject_ids)
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

        self._save_or_show(fig, save_dir, "cross_subject_summary.png", show)

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

        self._save_or_show(fig, save_dir, "reward_curve.png", show)

    def plot_intervention_heatmap(
        self,
        per_subject_histories: list[dict],
        save_dir: Path | str | None = None,
        show: bool = False,
        subject_ids: list[int] | tuple[int, ...] | None = None,
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
        ax.set_yticks(np.arange(n_subjects))
        ax.set_yticklabels(self._subject_labels(n_subjects, subject_ids))
        ax.set_title("Intervention Intensity Heatmap")
        plt.colorbar(im, ax=ax, label="Mean Action Level")
        plt.tight_layout()

        self._save_or_show(fig, save_dir, "intervention_heatmap.png", show)

    def plot_perclos_heatmap(
        self,
        per_subject_histories: list[dict],
        save_dir: Path | str | None = None,
        show: bool = False,
        subject_ids: list[int] | tuple[int, ...] | None = None,
    ):
        """
        Heatmap of binned PERCLOS risk across subjects.

        Color bands match the safe/watch/danger thresholds, making it easy to
        spot sustained risky segments across the held-out subjects.
        """
        n_bins = 80
        n_subjects = len(per_subject_histories)
        if n_subjects == 0:
            return

        heatmap = np.zeros((n_subjects, n_bins))
        for idx, hist in enumerate(per_subject_histories):
            perclos = np.array(hist["perclos"], dtype=float)
            heatmap[idx] = self._bin_series(perclos, n_bins)

        cmap = ListedColormap([ZONE_COLORS["safe"], ZONE_COLORS["watch"], ZONE_COLORS["danger"]])
        bounds = [0.0, self.cfg.env.perclos_safe, self.cfg.env.perclos_danger, 1.0]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(14, max(4, n_subjects * 0.55)))
        im = ax.imshow(heatmap, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title("PERCLOS Risk Heatmap")
        ax.set_xlabel("Time (normalized)")
        ax.set_ylabel("Subject")
        ax.set_yticks(np.arange(n_subjects))
        ax.set_yticklabels(self._subject_labels(n_subjects, subject_ids))
        cbar = plt.colorbar(im, ax=ax, boundaries=bounds, ticks=bounds)
        cbar.set_label("PERCLOS")
        plt.tight_layout()

        self._save_or_show(fig, save_dir, "perclos_risk_heatmap.png", show)

    def plot_action_distribution(
        self,
        per_subject_histories: list[dict],
        save_dir: Path | str | None = None,
        show: bool = False,
        subject_ids: list[int] | tuple[int, ...] | None = None,
    ):
        """
        Action distribution by subject plus overall action mix.

        The stacked bars show whether the policy is gentle, aggressive, or
        mostly passive for each held-out subject.
        """
        n_subjects = len(per_subject_histories)
        if n_subjects == 0:
            return

        counts = np.zeros((n_subjects, len(ACTION_COLORS)))
        for idx, hist in enumerate(per_subject_histories):
            actions = np.array(hist["actions"], dtype=int)
            for action in ACTION_COLORS:
                counts[idx, action] = np.sum(actions == action)
        totals = counts.sum(axis=1, keepdims=True)
        shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

        fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.2, 1]})
        left = np.zeros(n_subjects)
        y = np.arange(n_subjects)
        for action in ACTION_COLORS:
            ax_bar.barh(
                y,
                shares[:, action],
                left=left,
                height=0.6,
                color=ACTION_COLORS[action],
                edgecolor="white",
                label=ACTION_LABELS[action],
            )
            left += shares[:, action]
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(self._subject_labels(n_subjects, subject_ids))
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Share of timesteps")
        ax_bar.set_title("Action Mix by Subject")
        ax_bar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3,
                      fontsize=8, frameon=False)
        ax_bar.invert_yaxis()

        total_counts = counts.sum(axis=0)
        wedges, _ = ax_pie.pie(
            total_counts,
            colors=[ACTION_COLORS[a] for a in ACTION_COLORS],
            startangle=90,
            wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.4},
        )
        ax_pie.text(0, 0, f"{int(total_counts.sum())}\nsteps", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=METRIC_COLORS["ink"])
        ax_pie.set_title("Overall")
        ax_pie.legend(wedges, [ACTION_LABELS[a] for a in ACTION_COLORS],
                      loc="center left", bbox_to_anchor=(0.96, 0.5),
                      fontsize=7, frameon=False)

        fig.suptitle("Intervention Action Distribution", fontsize=14)
        plt.tight_layout()
        self._save_or_show(fig, save_dir, "action_distribution.png", show)

    def plot_metric_radar(
        self,
        aggregated: dict,
        save_dir: Path | str | None = None,
        show: bool = False,
    ):
        """
        Radar chart of normalized overall evaluation quality.

        Each spoke is scaled so larger is better. This is intentionally a quick
        visual scorecard, not a replacement for the exact numeric metrics.
        """
        metrics = [
            ("vigilance_improvement", "Improvement", False),
            ("danger_ratio", "Safe Time", True),
            ("alert_efficiency", "Efficiency", False),
            ("false_alert_rate", "Precision", True),
            ("intervention_rate", "Comfort", True),
            ("strong_intervention_rate", "Gentleness", True),
        ]

        labels = []
        values = []
        for key, label, invert in metrics:
            data = aggregated.get(key, {})
            if not isinstance(data, dict) or "mean" not in data:
                continue
            value = float(data["mean"])
            if key == "vigilance_improvement":
                value = np.clip(value, 0.0, 1.0)
            else:
                value = np.clip(value, 0.0, 1.0)
            values.append(1.0 - value if invert else value)
            labels.append(label)

        if len(values) < 3:
            return

        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
        values = np.array(values)
        angles_closed = np.concatenate([angles, angles[:1]])
        values_closed = np.concatenate([values, values[:1]])

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        ax.plot(angles_closed, values_closed, color=METRIC_COLORS["blue"], linewidth=2)
        ax.fill(angles_closed, values_closed, color=METRIC_COLORS["blue"], alpha=0.20)
        ax.scatter(angles, values, s=70, color=METRIC_COLORS["teal"], edgecolor="white", zorder=3)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
        ax.set_title("Evaluation Quality Radar", y=1.08)
        self._save_or_show(fig, save_dir, "metric_radar.png", show)

    def plot_policy_tradeoff(
        self,
        per_subject_metrics: list[dict],
        save_dir: Path | str | None = None,
        show: bool = False,
        subject_ids: list[int] | tuple[int, ...] | None = None,
    ):
        """
        Scatter view of reward, risk, and intervention cost trade-offs.

        X = intervention rate, Y = danger ratio, bubble size = cumulative reward,
        color = alert efficiency.
        """
        if not per_subject_metrics:
            return

        intervention = np.array([m.get("intervention_rate", 0.0) for m in per_subject_metrics], dtype=float)
        danger = np.array([m.get("danger_ratio", 0.0) for m in per_subject_metrics], dtype=float)
        reward = np.array([m.get("cumulative_reward", 0.0) for m in per_subject_metrics], dtype=float)
        efficiency = np.array([m.get("alert_efficiency", 0.0) for m in per_subject_metrics], dtype=float)
        labels = self._subject_labels(len(per_subject_metrics), subject_ids)

        reward_span = np.ptp(reward)
        if reward_span == 0:
            sizes = np.full_like(reward, 350, dtype=float)
        else:
            sizes = 180 + 520 * (reward - reward.min()) / reward_span

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(
            intervention,
            danger,
            s=sizes,
            c=efficiency,
            cmap="viridis",
            vmin=0,
            vmax=1,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.2,
        )
        for x, y_pos, label in zip(intervention, danger, labels):
            ax.annotate(label, (x, y_pos), xytext=(6, 6), textcoords="offset points",
                        fontsize=9, color=METRIC_COLORS["ink"])
        ax.axhline(self.cfg.evaluation.danger_threshold, color=METRIC_COLORS["rose"],
                   linestyle="--", alpha=0.35, label="Danger threshold")
        ax.set_xlim(-0.04, min(1.04, max(1.0, intervention.max() + 0.12)))
        ax.set_ylim(-0.04, min(1.04, max(1.0, danger.max() + 0.12)))
        ax.set_xlabel("Intervention rate")
        ax.set_ylabel("Danger time ratio")
        ax.set_title("Policy Trade-off Map")
        ax.legend(loc="upper right", fontsize=8)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Alert efficiency")
        plt.tight_layout()
        self._save_or_show(fig, save_dir, "policy_tradeoff_map.png", show)

    def plot_perclos_distribution(
        self,
        per_subject_histories: list[dict],
        save_dir: Path | str | None = None,
        show: bool = False,
        subject_ids: list[int] | tuple[int, ...] | None = None,
    ):
        """
        Violin-style distribution of controlled PERCLOS per subject.

        This complements the time-series plots by showing spread and skew in
        vigilance state over the full episode.
        """
        if not per_subject_histories:
            return

        perclos_values = [np.array(hist["perclos"], dtype=float) for hist in per_subject_histories]
        labels = self._subject_labels(len(per_subject_histories), subject_ids)
        positions = np.arange(1, len(perclos_values) + 1)

        fig, ax = plt.subplots(figsize=(max(8, len(perclos_values) * 1.2), 6))
        self._shade_perclos_zones(ax, len(perclos_values) + 1)
        parts = ax.violinplot(perclos_values, positions=positions, widths=0.72,
                              showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(METRIC_COLORS["blue"])
            body.set_edgecolor("white")
            body.set_alpha(0.5)
        if "cmeans" in parts:
            parts["cmeans"].set_color(METRIC_COLORS["ink"])
            parts["cmeans"].set_linewidth(1.4)
        ax.boxplot(
            perclos_values,
            positions=positions,
            widths=0.16,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": METRIC_COLORS["ink"], "linewidth": 0.8},
            medianprops={"color": METRIC_COLORS["rose"], "linewidth": 1.4},
            whiskerprops={"color": METRIC_COLORS["ink"], "linewidth": 0.8},
            capprops={"color": METRIC_COLORS["ink"], "linewidth": 0.8},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.25},
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlim(0.4, len(perclos_values) + 0.6)
        ax.set_ylabel("PERCLOS")
        ax.set_title("PERCLOS Distribution by Subject")
        plt.tight_layout()
        self._save_or_show(fig, save_dir, "perclos_distribution.png", show)
