# -*- coding: utf-8 -*-
"""
绘制三组实验（P1/P2/P3）分组柱状图，并横向拼接成一个图（共享 y 轴）

用法：
    python plot.py
    python plot.py --csv results/exp1_sinusoidal/packet_success_rate_summary.csv
    python plot.py --exp 1
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
FIGURES_DIR = HERE / "figures" / "packet_success_rate"

BAR_WIDTH = 0.22

ALGO_ORDER = ["DeepAAQM", "SACAQM", "CODEL"]

ALGO_DISPLAY = {
    "DQN": "DeepAAQM",
    "SAC": "SACAQM",
    "CODEL": "CoDel",
    "codel": "CoDel",
    "CoDel": "CoDel",
}

EXP_NAMES = {
    "exp1_sinusoidal": "Async sinusoidal",
    "exp2_state_switching": "Async regime-switching",
    "exp3_near_saturated": "Async near-saturation",
}


def _load_summary_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "mean_P1_success_rate" in df.columns:
        algo_col = "algorithm" if "algorithm" in df.columns else "算法"
        return pd.DataFrame({
            "algorithm": df[algo_col],
            "P1": df["mean_P1_success_rate"],
            "P2": df["mean_P2_success_rate"],
            "P3": df["mean_P3_success_rate"],
        })

    if "P1" in df.columns and "P2" in df.columns and "P3" in df.columns:
        algo_col = "algorithm" if "algorithm" in df.columns else "算法"
        return pd.DataFrame({
            "algorithm": df[algo_col],
            "P1": df["P1"],
            "P2": df["P2"],
            "P3": df["P3"],
        })

    raise ValueError(f"Unrecognized CSV columns: {list(df.columns)}")


def _normalize_algo(name: str) -> str:
    name = str(name).strip()
    upper = name.upper()
    if upper == "DQN":
        return "DeepAAQM"
    if upper == "SAC":
        return "SACAQM"
    if upper in ["CODEL", "CODEL"]:
        return "CODEL"
    return upper


def _display_algo(name: str) -> str:
    return ALGO_DISPLAY.get(str(name).strip(), ALGO_DISPLAY.get(str(name).strip().upper(), str(name)))


def _sort_algorithms(df: pd.DataFrame) -> pd.DataFrame:
    order = {name: i for i, name in enumerate(ALGO_ORDER)}
    df = df.copy()
    df["_ord"] = df["algorithm"].map(lambda x: order.get(_normalize_algo(x), 99))
    return df.sort_values("_ord").drop(columns="_ord")


def plot_all_experiments(exp_data, save_path: Path):
    fig, axes = plt.subplots(1, len(exp_data), figsize=(10, 3.5), sharey=True)

    if len(exp_data) == 1:
        axes = [axes]

    for i, (ax, (exp_name, df)) in enumerate(zip(axes, exp_data)):
        df = _sort_algorithms(df)

        labels = [_display_algo(a) for a in df["algorithm"].to_numpy()]
        y1 = df["P1"].to_numpy(dtype=float)
        y2 = df["P2"].to_numpy(dtype=float)
        y3 = df["P3"].to_numpy(dtype=float)

        x = np.arange(len(labels))
        w = BAR_WIDTH

        ax.bar(x - w, y1, w, label="P1-low")
        ax.bar(x, y2, w, label="P2-medium")
        ax.bar(x + w, y3, w, label="P3-high")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.set_title(EXP_NAMES.get(exp_name, exp_name), y=-0.18)

        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    # 👉 共享 y 轴
    axes[0].set_ylabel("Packet Success Rate")
    axes[0].set_ylim(0, 1.05)

    # 👉 legend 放到图内部（右上角）
    axes[-1].legend(loc="upper right", frameon=True)

    



    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined figure: {save_path}")


def discover_summary_csvs():
    pairs = []
    for exp_dir in sorted(RESULTS_DIR.glob("exp*")):
        csv_path = exp_dir / "packet_success_rate_summary.csv"
        if csv_path.is_file():
            pairs.append((exp_dir.name, csv_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Plot packet success rate")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--out-dir", type=str, default=str(FIGURES_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    exp_map = {
        1: "exp1_sinusoidal",
        2: "exp2_state_switching",
        3: "exp3_near_saturated",
    }

    # 单个 CSV
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_file():
            csv_path = HERE / args.csv

        df = _load_summary_csv(csv_path)
        exp_name = csv_path.parent.name

        plot_all_experiments(
            [(exp_name, df)],
            out_dir / f"{exp_name}_combined.png"
        )
        return

    # 单个实验
    if args.exp is not None:
        exp_name = exp_map[args.exp]
        csv_path = RESULTS_DIR / exp_name / "packet_success_rate_summary.csv"

        df = _load_summary_csv(csv_path)

        plot_all_experiments(
            [(exp_name, df)],
            out_dir / f"{exp_name}_combined.png"
        )
        return

    # 默认：三个实验拼一起
    targets = discover_summary_csvs()

    if not targets:
        print("No CSV found.")
        return

    exp_data = []
    for exp_name, csv_path in targets:
        if not csv_path.is_file():
            continue
        df = _load_summary_csv(csv_path)
        exp_data.append((exp_name, df))

    plot_all_experiments(
        exp_data,
        out_dir / "all_experiments_priority_success_rate.png"
    )

    print(f"\nDone. Figures saved in: {out_dir}")


if __name__ == "__main__":
    main()