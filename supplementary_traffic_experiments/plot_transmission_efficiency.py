# -*- coding: utf-8 -*-
"""
三次补实验：传输效率（每包能耗）三图并排折线图。

纵轴：efficiency × 0.24 mJ（与 plot_transmisson_efficiency.py 一致）
CSV：results/<exp>/transmission_efficiency_comparison_chart_data.csv
列：Episode, DeepAAQM, SACAQM, CODEL

用法：
    python plot_transmission_efficiency.py
    python plot_transmission_efficiency.py --results-root results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee", "bright"])
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 6

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures" / "transmission_efficiency"

ENERGY_PER_FORWARD_MJ = 0.24
Y_PAD_FRAC = 0.12

EXP_NAMES = {
    "exp1_sinusoidal": "Async sinusoidal",
    "exp2_state_switching": "Async regime-switching",
    "exp3_near_saturated": "Async near-saturation",
}

EXP_ORDER = ["exp1_sinusoidal", "exp2_state_switching", "exp3_near_saturated"]


def _load_te_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"Episode", "DeepAAQM", "SACAQM", "CODEL"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    return df


def _discover_te_csvs(results_root: Path):
    pairs = []
    for name in EXP_ORDER:
        csv_path = results_root / name / "transmission_efficiency_comparison_chart_data.csv"
        if csv_path.is_file():
            pairs.append((name, csv_path))
    return pairs


def plot_all_experiments(exp_data, save_path: Path):
    """exp_data: list of (exp_name, DataFrame)"""
    fig, axes = plt.subplots(1, len(exp_data), figsize=(10, 3.5), sharey=True)
    if len(exp_data) == 1:
        axes = [axes]

    all_y = []

    for ax, (exp_name, df) in zip(axes, exp_data):
        x = df["Episode"].to_numpy()
        y_sac = df["SACAQM"].to_numpy(dtype=float) * ENERGY_PER_FORWARD_MJ
        y_dqn = df["DeepAAQM"].to_numpy(dtype=float) * ENERGY_PER_FORWARD_MJ
        y_codel = df["CODEL"].to_numpy(dtype=float) * ENERGY_PER_FORWARD_MJ

        ax.plot(x, y_sac, label="SACAQM")
        ax.plot(x, y_dqn, label="DeepAAQM")
        ax.plot(x, y_codel, label="CoDel")

        ax.set_xlabel("Episode")
        ax.set_title(EXP_NAMES.get(exp_name, exp_name), y=-0.28)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        all_y.extend([y_sac, y_dqn, y_codel])

    axes[0].set_ylabel(r"Energy Consumption per Packet (mJ)")

    y_stack = np.concatenate([np.asarray(y) for y in all_y])
    y_stack = y_stack[np.isfinite(y_stack)]
    if y_stack.size:
        lo, hi = float(np.min(y_stack)), float(np.max(y_stack))
        span = hi - lo
        pad = Y_PAD_FRAC * span if span > 0 else 0.02
        axes[0].set_ylim(lo - pad, hi + pad)

    axes[-1].legend(loc="upper right", frameon=True, fontsize=10)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_from_results(results_root: Path = None, figures_dir: Path = None):
    results_root = Path(results_root) if results_root else RESULTS_DIR
    figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR

    pairs = _discover_te_csvs(results_root)
    if not pairs:
        print(f"No transmission efficiency CSV found under {results_root}")
        return

    exp_data = [(name, _load_te_csv(path)) for name, path in pairs]
    plot_all_experiments(
        exp_data,
        figures_dir / "all_experiments_transmission_efficiency.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Plot transmission efficiency (3 panels)")
    parser.add_argument("--results-root", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR))
    args = parser.parse_args()
    plot_from_results(Path(args.results_root), Path(args.figures_dir))


if __name__ == "__main__":
    main()
