# -*- coding: utf-8 -*-
"""
Plot packet success rate figures for the three supplementary experiments.
All chart titles and labels are in English (scienceplots / ieee style).
"""

import os
import argparse
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_configs import EXPERIMENT_CONFIGS  # noqa: E402

# Same palette as plot_packet_success_rate.py (project root)
COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22", "#34495e"]
MARKERS = ["o", "s", "^", "D", "v", "p", "*", "X"]

ALGORITHMS = ["DeepAAQM", "SAC", "CODEL"]
ALGO_COLORS = COLORS[:3]  # DQN, SAC, CODEL
PRIORITY_COLORS = ["#3498db", "#e67e22", "#2ecc71"]  # P1, P2, P3 (compare_dqn_sac_codel)
PRIORITY_LABELS = ["P1", "P2", "P3"]

# Support legacy Chinese column names in existing CSV files
COL_ALGO = ("algorithm", "算法")
COL_P1 = ("mean_P1_success_rate", "平均P1成功率")
COL_P2 = ("mean_P2_success_rate", "平均P2成功率")
COL_P3 = ("mean_P3_success_rate", "平均P3成功率")
COL_PKT_MEAN = ("mean_packet_success_rate", "平均packet成功率")
COL_PKT_STD = ("std_packet_success_rate", "packet成功率标准差")


def _col(df: pd.DataFrame, candidates: tuple) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing column, tried: {candidates}")


def _load_summary(results_root: str, exp_name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_root, exp_name, "packet_success_rate_summary.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def plot_priority_bars_for_experiment(
    summary_df: pd.DataFrame,
    plot_title: str,
    save_path: str,
) -> None:
    algo_col = _col(summary_df, COL_ALGO)
    algorithms = [a for a in ALGORITHMS if a in summary_df[algo_col].values]
    if not algorithms:
        print(f"  Skip (no data): {save_path}")
        return

    p_cols = [
        _col(summary_df, COL_P1),
        _col(summary_df, COL_P2),
        _col(summary_df, COL_P3),
    ]

    x = np.arange(len(algorithms))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (plabel, col) in enumerate(zip(PRIORITY_LABELS, p_cols)):
        values = []
        for algo in algorithms:
            row = summary_df[summary_df[algo_col] == algo]
            values.append(float(row[col].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x + (idx - 1) * width, values, width, label=plabel,
            color=PRIORITY_COLORS[idx], alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=13, fontweight="bold")
    ax.set_ylabel("Packet Success Rate", fontsize=14, fontweight="bold")
    ax.set_title(f"Priority Success Rate by Algorithm\n{plot_title}", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Priority", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_packet_success_bars_for_experiment(
    summary_df: pd.DataFrame,
    plot_title: str,
    save_path: str,
) -> None:
    algo_col = _col(summary_df, COL_ALGO)
    mean_col = _col(summary_df, COL_PKT_MEAN)
    std_col = _col(summary_df, COL_PKT_STD)

    algorithms = [a for a in ALGORITHMS if a in summary_df[algo_col].values]
    if not algorithms:
        return

    rates, stds = [], []
    for algo in algorithms:
        row = summary_df[summary_df[algo_col] == algo].iloc[0]
        rates.append(float(row[mean_col]))
        stds.append(float(row[std_col]) if pd.notna(row[std_col]) else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        algorithms, rates, yerr=stds, capsize=8,
        color=ALGO_COLORS[: len(algorithms)], alpha=0.8,
        edgecolor="black", linewidth=1.5, width=0.6,
    )
    ax.set_ylabel("Packet Success Rate", fontsize=14, fontweight="bold")
    ax.set_title(f"Overall Packet Success Rate\n{plot_title}", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for bar, r in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{r:.3f}", ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cross_experiment_summary(all_summaries: Dict[str, pd.DataFrame], save_path: str) -> None:
    exp_names = [c["name"] for c in EXPERIMENT_CONFIGS]
    exp_labels = [c["plot_title"] for c in EXPERIMENT_CONFIGS]

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo_idx, algo in enumerate(ALGORITHMS):
        ys = []
        for name in exp_names:
            df = all_summaries.get(name)
            if df is None or df.empty:
                ys.append(np.nan)
                continue
            algo_col = _col(df, COL_ALGO)
            p3_col = _col(df, COL_P3)
            row = df[df[algo_col] == algo]
            ys.append(float(row[p3_col].iloc[0]) if not row.empty else np.nan)
        ax.plot(
            range(len(exp_names)), ys,
            label=algo,
            color=ALGO_COLORS[algo_idx],
            linewidth=2,
            marker=MARKERS[algo_idx],
            markersize=4,
            alpha=0.85,
        )

    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Packet Success Rate", fontsize=12, fontweight="bold")
    ax.set_title("High-Priority (P3) Success Rate Across Three Experiments", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_all(results_root: str, figures_dir: Optional[str] = None) -> None:
    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(__file__), "figures", "packet_success_rate")
    os.makedirs(figures_dir, exist_ok=True)

    all_summaries = {}
    for cfg in EXPERIMENT_CONFIGS:
        name = cfg["name"]
        summary = _load_summary(results_root, name)
        if summary is None:
            print(f"Missing: {results_root}/{name}/packet_success_rate_summary.csv")
            continue
        all_summaries[name] = summary
        plot_title = cfg.get("plot_title", name)

        print(f"\nPlotting: {plot_title}")
        plot_priority_bars_for_experiment(
            summary, plot_title,
            os.path.join(figures_dir, f"{name}_priority_success_rate.png"),
        )
        plot_packet_success_bars_for_experiment(
            summary, plot_title,
            os.path.join(figures_dir, f"{name}_overall_packet_success_rate.png"),
        )

    if len(all_summaries) >= 2:
        plot_cross_experiment_summary(
            all_summaries,
            os.path.join(figures_dir, "cross_experiment_P3_success_rate.png"),
        )

    print(f"\nAll figures saved to: {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot supplementary packet success rate figures")
    parser.add_argument(
        "--results-root",
        default=os.path.join(os.path.dirname(__file__), "results"),
    )
    parser.add_argument("--figures-dir", default=None)
    args = parser.parse_args()
    plot_all(args.results_root, args.figures_dir)


if __name__ == "__main__":
    main()
