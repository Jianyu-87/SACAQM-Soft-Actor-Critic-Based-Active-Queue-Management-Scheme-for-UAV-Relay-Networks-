# -*- coding: utf-8 -*-
"""
绘制高优先级数据包成功率图。
- 折线图：横轴低/中/高负载，纵轴成功率，不同算法不同线（需三份负载数据）。
- 柱状图：横轴算法名，纵轴平均成功率，带标准差误差条（单份 priority3_comparison.csv）。

数据格式说明：
- priority3_comparison 格式（算法, 平均成功率, 标准差, 最小值, 最大值）：
  --csv 单文件 → 柱状图；--low/--medium/--high 三个文件 → 折线图（带误差条）。
- 带负载列格式（algorithm, load, success_rate）：--csv 单文件 → 折线图。
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd = None

# 负载顺序（横轴）
LOAD_LABELS = ["低", "中", "高"]

# 线条/柱子样式
STYLES = [
    {"color": "#2E86AB", "marker": "o", "linestyle": "-", "linewidth": 2},
    {"color": "#E94F37", "marker": "s", "linestyle": "-", "linewidth": 2},
    {"color": "#44AF69", "marker": "^", "linestyle": "-", "linewidth": 2},
    {"color": "#F18F01", "marker": "D", "linestyle": "-", "linewidth": 2},
    {"color": "#6C5CE7", "marker": "v", "linestyle": "-", "linewidth": 2},
    {"color": "#00B894", "marker": "p", "linestyle": "-", "linewidth": 2},
]


def _normalize_rate(value):
    """将 0~100 转为 0~1。"""
    v = float(value)
    if v > 1.0:
        return v / 100.0
    return v


def load_priority3_comparison_csv(csv_path):
    """
    从 priority3_comparison.csv 格式加载：算法, 平均成功率, 标准差, 最小值, 最大值。
    返回 (algorithms, rates, stds)，均为列表，顺序一致。
    """
    if pd is None:
        raise ImportError("从 CSV 读取需要 pandas，请安装: pip install pandas")
    df = pd.read_csv(csv_path)
    algo_col = "算法"
    rate_col = "平均成功率"
    std_col = "标准差"
    if algo_col not in df.columns or rate_col not in df.columns:
        raise ValueError(f"CSV 需包含列 算法、平均成功率，当前列: {list(df.columns)}")
    algorithms = df[algo_col].astype(str).str.strip().tolist()
    rates = df[rate_col].apply(_normalize_rate).tolist()
    stds = df[std_col].fillna(0).tolist() if std_col in df.columns else [0.0] * len(algorithms)
    return algorithms, rates, stds


def load_three_priority3_csv(low_path, medium_path, high_path):
    """
    从三个 priority3_comparison 格式的 CSV（分别对应低/中/高负载）合并为折线图数据。
    返回 data = { "算法名": [低, 中, 高] }, data_stds = { "算法名": [std低, std中, std高] }。
    按第一个文件的算法顺序，后续文件按算法名（不区分大小写）匹配。
    """
    def load_one(path):
        if not path or not os.path.isfile(path):
            return None, None, None
        algs, rates, stds = load_priority3_comparison_csv(path)
        return algs, rates, stds

    low_algs, low_rates, low_stds = load_one(low_path)
    med_algs, med_rates, med_stds = load_one(medium_path)
    high_algs, high_rates, high_stds = load_one(high_path)

    # 以第一个非空文件的算法列表为基准
    if low_algs:
        base_algs = low_algs
        low_d = {a.upper(): (r, s) for a, r, s in zip(low_algs, low_rates, low_stds)}
    else:
        base_algs = med_algs if med_algs else high_algs
        low_d = {}
    if med_algs:
        med_d = {a.upper(): (r, s) for a, r, s in zip(med_algs, med_rates, med_stds)}
    else:
        med_d = {}
    if high_algs:
        high_d = {a.upper(): (r, s) for a, r, s in zip(high_algs, high_rates, high_stds)}
    else:
        high_d = {}

    data = {}
    data_stds = {}
    for algo in base_algs:
        key = algo.upper()
        r_low, s_low = low_d.get(key, (None, 0.0))
        r_med, s_med = med_d.get(key, (None, 0.0))
        r_high, s_high = high_d.get(key, (None, 0.0))
        if r_low is None and r_med is None and r_high is None:
            continue
        data[algo] = [
            r_low if r_low is not None else np.nan,
            r_med if r_med is not None else np.nan,
            r_high if r_high is not None else np.nan,
        ]
        data_stds[algo] = [s_low, s_med, s_high]
    return data, data_stds


def load_from_csv_with_load_column(csv_path):
    """
    从带负载列的 CSV 加载。期望列：algorithm/算法, load/负载, success_rate/成功率。
    返回 { "算法名": [低, 中, 高] }。
    """
    if pd is None:
        raise ImportError("从 CSV 读取需要 pandas，请安装: pip install pandas")
    df = pd.read_csv(csv_path)
    algo_col = "algorithm" if "algorithm" in df.columns else "算法"
    load_col = "load" if "load" in df.columns else "负载"
    rate_col = "success_rate" if "success_rate" in df.columns else "成功率"
    for c in (algo_col, load_col, rate_col):
        if c not in df.columns:
            raise ValueError(f"CSV 需包含列: {c}，当前列: {list(df.columns)}")
    load_map = {"低": 0, "中": 1, "高": 2, "low": 0, "medium": 1, "high": 2}
    result = {}
    for _, row in df.iterrows():
        algo = str(row[algo_col]).strip()
        load_str = str(row[load_col]).strip()
        load_idx = load_map.get(load_str) or load_map.get(load_str.lower())
        if load_idx is None:
            continue
        rate = _normalize_rate(row[rate_col])
        if algo not in result:
            result[algo] = [None, None, None]
        result[algo][load_idx] = rate
    return {k: v for k, v in result.items() if all(x is not None for x in v)}


def plot_success_rate_by_load(data, data_stds=None, title=None, ylabel="高优先级数据包成功率", save_path=None):
    """
    绘制成功率-负载折线图。data: { "算法名": [低, 中, 高] }；可选 data_stds 同结构画误差条。
    """
    if not data:
        raise ValueError("data 为空，无法绘图")

    x = np.arange(len(LOAD_LABELS))
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (algo, rates) in enumerate(data.items()):
        style = STYLES[i % len(STYLES)]
        stds = (data_stds or {}).get(algo, [0, 0, 0])
        # 过滤 nan 以便绘图
        mask = ~np.isnan(rates)
        x_plot = x[mask]
        y_plot = np.array(rates)[mask]
        err_plot = np.array(stds)[mask] if len(stds) == 3 else np.zeros_like(y_plot)
        if len(err_plot) > 0 and np.any(err_plot > 0):
            ax.errorbar(
                x_plot, y_plot, yerr=err_plot,
                label=algo, color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=style["linewidth"],
                markersize=9, capsize=4,
            )
        else:
            ax.plot(
                x_plot, y_plot, label=algo,
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=style["linewidth"], markersize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(LOAD_LABELS, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("负载", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {save_path}")
    plt.close(fig)


def plot_success_rate_bar(algorithms, rates, stds, title=None, ylabel="高优先级数据包成功率", save_path=None):
    """柱状图：横轴算法，纵轴平均成功率，带标准差误差条。"""
    if not algorithms:
        raise ValueError("algorithms 为空，无法绘图")

    x = np.arange(len(algorithms))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.6
    colors = [STYLES[i % len(STYLES)]["color"] for i in range(len(algorithms))]
    bars = ax.bar(x, rates, width, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("算法", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_axisbelow(True)
    for bar, r, s in zip(bars, rates, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02, f"{r:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="高优先级成功率图（折线图/柱状图）")
    parser.add_argument("--csv", type=str, default=None, help="单 CSV：priority3 格式→柱状图；带 load 列→折线图")
    parser.add_argument("--low", type=str, default=None, help="低负载 priority3_comparison.csv（与 --medium/--high 一起用→折线图）")
    parser.add_argument("--medium", type=str, default=None, help="中负载 priority3_comparison.csv")
    parser.add_argument("--high", type=str, default=None, help="高负载 priority3_comparison.csv")
    parser.add_argument("--title", type=str, default="高优先级数据包成功率", help="图标题")
    parser.add_argument("--out", type=str, default="high_priority_success_by_load.png", help="输出图片路径")
    args = parser.parse_args()

    # 三文件折线图（priority3 格式）
    if args.low or args.medium or args.high:
        data, data_stds = load_three_priority3_csv(args.low, args.medium, args.high)
        if data:
            plot_success_rate_by_load(data, data_stds=data_stds, title=args.title, save_path=args.out)
            return
        if args.low or args.medium or args.high:
            print("未得到有效数据，请检查 --low/--medium/--high 路径及 CSV 格式")
            return

    # 单 CSV
    if args.csv and os.path.isfile(args.csv):
        if pd is None:
            print("读取 CSV 需要 pandas，请安装: pip install pandas")
            return
        df = pd.read_csv(args.csv)
        # 带负载列 → 折线图
        if "负载" in df.columns or "load" in df.columns:
            data = load_from_csv_with_load_column(args.csv)
            if data:
                plot_success_rate_by_load(data, title=args.title, save_path=args.out)
                return
        # priority3 格式（算法, 平均成功率, 标准差）→ 柱状图
        if "算法" in df.columns and "平均成功率" in df.columns:
            try:
                algorithms, rates, stds = load_priority3_comparison_csv(args.csv)
                if algorithms:
                    plot_success_rate_bar(algorithms, rates, stds, title=args.title, save_path=args.out)
                    return
            except Exception as e:
                print(f"解析 priority3 格式失败: {e}")
        print("CSV 需包含「算法、平均成功率」或「algorithm/load/success_rate」列")
        return

    # 无输入时用示例数据画柱状图
    algorithms = ["DQN", "SAC", "CODEL"]
    rates = [0.688, 0.9996, 0.0318]
    stds = [0.0465, 0.00074, 0.0399]
    plot_success_rate_bar(algorithms, rates, stds, title=args.title, save_path=args.out)


if __name__ == "__main__":
    main()
