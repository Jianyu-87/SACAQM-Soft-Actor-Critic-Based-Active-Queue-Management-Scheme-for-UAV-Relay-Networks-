#!/usr/bin/env python3
"""
从 Stable Baselines3 的 monitor.csv 绘制数据包传输成功率 (Packet Success Rate) 随 Episode 的收敛曲线。
支持多组日志对比（如 SAC / DQN），滑动平均，以及从 monitor 或侧边文件读取 transmitted/dropped 统计。
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# 成功率定义: SuccessRate = Packets_Transmitted / (Packets_Transmitted + Packets_Dropped)
# 若仅有 transmission_rate 字段也可直接用作成功率


def _read_monitor_csv(monitor_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    读取 monitor.csv，处理 SB3 的注释首行与表头。
    返回: (数据 DataFrame, 原始表头列名列表)
    """
    with open(monitor_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 跳过以 # 开头的注释行
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            continue
        data_start = i
        break
    header_line = lines[data_start].strip()
    columns = [c.strip() for c in header_line.split(",")]
    # 从下一行开始解析数据
    data_lines = [line.strip() for line in lines[data_start + 1 :] if line.strip()]
    if not data_lines:
        return pd.DataFrame(columns=columns), columns
    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= len(columns):
            row = parts[: len(columns)]
        else:
            row = parts + [np.nan] * (len(columns) - len(parts))
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    # 数值列转换
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    return df, columns


def _compute_success_rate_from_columns(
    df: pd.DataFrame, tx_col: str, drop_col: str
) -> np.ndarray:
    """根据 transmitted 与 dropped 列计算每 episode 成功率。"""
    tx = df[tx_col].fillna(0).values.astype(float)
    drop = df[drop_col].fillna(0).values.astype(float)
    total = tx + drop
    # 避免除零
    total = np.where(total <= 0, 1.0, total)
    return (tx / total).astype(np.float64)


def _find_episode_stats_file(log_dir: str) -> Optional[str]:
    """在日志目录下查找可能包含每 episode 统计的文件。"""
    candidates = [
        "episode_success_rate.csv",  # dqn_train 等写入的成功率文件优先
        "episode_stats.csv",
        "episode_success_rate.json",
        "episode_stats.json",
    ]
    for name in candidates:
        path = os.path.join(log_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _load_episode_stats_for_success_rate(
    path: str, n_episodes: int
) -> Optional[np.ndarray]:
    """
    从侧边文件加载与 episode 一一对应的成功率。
    支持列: transmission_rate, 或 packets_transmitted + packets_dropped。
    """
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".json":
            df = pd.read_json(path)
        else:
            return None
    except Exception:
        return None
    if len(df) == 0:
        return None
    # 优先使用 transmission_rate（与成功率定义一致）
    for rate_col in ["transmission_rate", "success_rate", "packet_success_rate"]:
        if rate_col in df.columns:
            vals = pd.to_numeric(df[rate_col], errors="coerce").fillna(0).values
            if len(vals) >= n_episodes:
                return vals[:n_episodes].astype(np.float64)
            if len(vals) <= n_episodes:
                pad = np.full(n_episodes - len(vals), np.nan)
                return np.concatenate([vals.astype(np.float64), pad])
    # 否则用 transmitted / (transmitted + dropped)
    tx_col = None
    drop_col = None
    for c in ["packets_transmitted", "transmitted", "packets_transmitted_this_episode"]:
        if c in df.columns:
            tx_col = c
            break
    for c in ["packets_dropped", "dropped", "packets_dropped_this_episode"]:
        if c in df.columns:
            drop_col = c
            break
    if tx_col is not None and drop_col is not None:
        tx = pd.to_numeric(df[tx_col], errors="coerce").fillna(0).values
        drop = pd.to_numeric(df[drop_col], errors="coerce").fillna(0).values
        total = tx + drop
        total = np.where(total <= 0, 1.0, total)
        rates = (tx / total).astype(np.float64)
        if len(rates) >= n_episodes:
            return rates[:n_episodes]
        if len(rates) < n_episodes:
            pad = np.full(n_episodes - len(rates), np.nan)
            return np.concatenate([rates, pad])
    return None


def get_success_rates_from_log_dir(
    log_dir: str,
    monitor_subpath: str = "monitor.csv",
    episode_stats_path: Optional[str] = None,
    use_reward_as_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从单个日志目录得到 Episode 索引与对应的 Packet Success Rate。

    优先级:
    1) monitor.csv 表头中的自定义列（packets_transmitted / packets_dropped 或 transmission_rate）;
    2) 同目录下的 episode_stats 文件或传入的 episode_stats_path;
    3) 若 use_reward_as_fallback=True，用 reward 的 min-max 归一化模拟趋势（仅作示意）。

    返回:
        episodes: 1-based episode 索引，长度与 success_rates 一致
        success_rates: 每 episode 的成功率，范围 [0, 1]
    """
    monitor_path = os.path.join(log_dir, monitor_subpath)
    if not os.path.isfile(monitor_path):
        raise FileNotFoundError(f"未找到 monitor 文件: {monitor_path}")

    df, columns = _read_monitor_csv(monitor_path)
    n_episodes = len(df)
    if n_episodes == 0:
        return np.array([]), np.array([])

    # 检查 monitor 表头是否已有传输/丢弃或成功率列（兼容多种命名）
    tx_candidates = [
        c for c in columns
        if re.search(r"transmit|packets?_transmitted", c, re.I)
    ]
    drop_candidates = [
        c for c in columns
        if re.search(r"dropped|packets?_dropped", c, re.I)
    ]
    rate_candidates = [
        c for c in columns
        if re.search(r"transmission_rate|success_rate|packet_success", c, re.I)
    ]

    if rate_candidates:
        col = rate_candidates[0]
        success_rates = pd.to_numeric(df[col], errors="coerce").fillna(0).values.astype(np.float64)
        success_rates = np.clip(success_rates, 0.0, 1.0)
    elif tx_candidates and drop_candidates:
        success_rates = _compute_success_rate_from_columns(
            df, tx_candidates[0], drop_candidates[0]
        )
    else:
        # 尝试从侧边文件读取
        sidecar = episode_stats_path or _find_episode_stats_file(log_dir)
        loaded = _load_episode_stats_for_success_rate(sidecar, n_episodes) if sidecar else None
        if loaded is not None:
            success_rates = np.clip(np.nan_to_num(loaded, nan=0.0), 0.0, 1.0)
        elif use_reward_as_fallback and "r" in df.columns:
            # 用 reward 做 min-max 归一化作为趋势示意（不保证与真实成功率一致）
            r = df["r"].astype(float).values
            r_min, r_max = r.min(), r.max()
            if r_max > r_min:
                success_rates = (r - r_min) / (r_max - r_min)
            else:
                success_rates = np.ones_like(r) * 0.5
        else:
            raise ValueError(
                f"未在 monitor 或侧边文件中找到 transmitted/dropped 或 success rate 数据。"
                f" 表头: {columns}。可提供 episode_stats_path 或启用 use_reward_as_fallback。"
            )

    episodes = np.arange(1, n_episodes + 1, dtype=np.int64)
    return episodes, success_rates


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """滑动平均，窗口内不足时用已有数据平均。"""
    if window <= 1 or len(x) == 0:
        return x.copy()
    window = min(window, len(x))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_packet_success_rate(
    log_configs: List[Tuple[str, str]],
    window: int = 50,
    output_path: Optional[str] = None,
    y_percent: bool = False,
    figsize: Tuple[float, float] = (10, 6),
    marker_style: bool = True,
    use_reward_as_fallback: bool = False,
    episode_stats_paths: Optional[List[Optional[str]]] = None,
) -> None:
    """
    绘制多组日志的 Packet Success Rate 收敛曲线（带滑动平均与可选 Marker）。

    log_configs: [(log_dir, label), ...]，例如 [("SAC_dynamic_uav_logs/run1", "SAC"), ("dynamic_uav_logs/run1", "DQN")]
    window: 滑动平均窗口大小
    output_path: 保存图片路径
    y_percent: 纵轴是否显示为 0–100%
    marker_style: 是否使用带 Marker 的曲线（便于多线对比）
    use_reward_as_fallback: 当无 transmitted/dropped 数据时，是否用 reward 的 min-max 归一化作趋势示意
    episode_stats_paths: 与 log_configs 同长度的列表，每项为对应日志的 episode 统计文件路径（可选）
    """
    # 预定义样式（带不同 marker 便于区分）
    markers = ["o", "s", "^", "D", "v", "p", "*", "X"]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22", "#34495e"]

    fig, ax = plt.subplots(figsize=figsize)
    max_ep = 0

    paths_opt = episode_stats_paths or [None] * len(log_configs)
    for idx, (log_dir_or_monitor_dir, label) in enumerate(log_configs):
        try:
            # 若传入的是 monitor.csv 所在目录
            if os.path.isfile(log_dir_or_monitor_dir):
                log_dir = os.path.dirname(log_dir_or_monitor_dir)
                monitor_subpath = os.path.basename(log_dir_or_monitor_dir)
            else:
                log_dir = log_dir_or_monitor_dir
                monitor_subpath = "monitor.csv"
            ep_path = paths_opt[idx] if idx < len(paths_opt) else None
            episodes, success_rates = get_success_rates_from_log_dir(
                log_dir,
                monitor_subpath=monitor_subpath,
                episode_stats_path=ep_path,
                use_reward_as_fallback=use_reward_as_fallback,
            )
        except Exception as e:
            print(f"跳过 {label} ({log_dir_or_monitor_dir}): {e}")
            continue
        if len(episodes) == 0:
            continue
        smoothed = moving_average(success_rates, window)
        n_valid = len(smoothed)
        ep_smooth = episodes[window - 1 : window - 1 + n_valid] if n_valid > 0 else np.array([])
        if n_valid == 0:
            ep_smooth = episodes
            smoothed = success_rates
        y_plot = smoothed * 100.0 if y_percent else smoothed
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        if marker_style:
            ax.plot(
                ep_smooth,
                y_plot,
                label=label,
                color=color,
                linewidth=2,
                marker=marker,
                markersize=4,
                markevery=max(1, n_valid // 50),
                alpha=0.85,
            )
        else:
            ax.plot(ep_smooth, y_plot, label=label, color=color, linewidth=2, alpha=0.85)
        max_ep = max(max_ep, int(episodes[-1]) if len(episodes) else 0)

    ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ylabel = "Packet Success Rate (%)" if y_percent else "Packet Success Rate"
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105 if y_percent else 1.05)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    if max_ep > 0:
        ax.set_xlim(left=0)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"图片已保存: {output_path}")
    plt.show()


def _resolve_latest_run(parent_dir: str) -> Optional[str]:
    """若 parent_dir 下为多个带时间戳的子目录，返回最新一个的路径。"""
    if not os.path.isdir(parent_dir):
        return None
    subdirs = [
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    if not subdirs:
        return parent_dir
    def mtime(d):
        return os.path.getmtime(os.path.join(parent_dir, d))
    latest = max(subdirs, key=mtime)
    return os.path.join(parent_dir, latest)


def main():
    parser = argparse.ArgumentParser(
        description="绘制 monitor.csv 对应的 Packet Success Rate 收敛曲线，支持多组对比。"
    )
    parser.add_argument(
        "log_dirs",
        nargs="*",
        default=[],
        help="日志目录或 monitor 所在目录，可多个。若未提供则使用示例配置。",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="与 log_dirs 一一对应的曲线标签。若缺省则用目录名。",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="滑动平均窗口大小（默认 50）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出图片路径",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="纵轴显示为 0–100%%",
    )
    parser.add_argument(
        "--no-marker",
        action="store_true",
        help="不使用 Marker 的曲线",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="对每个 log_dir 取其下最新一次运行子目录（按修改时间）",
    )
    parser.add_argument(
        "--fallback-reward",
        action="store_true",
        help="当无 transmitted/dropped 数据时，用 reward 的 min-max 归一化作趋势示意",
    )
    parser.add_argument(
        "--episode-stats",
        nargs="*",
        default=[],
        help="与 log_dirs 一一对应的 episode 统计文件路径（CSV/JSON），用于计算成功率",
    )
    args = parser.parse_args()

    if args.log_dirs:
        dirs = list(args.log_dirs)
        if args.latest:
            resolved = []
            for d in dirs:
                p = _resolve_latest_run(d)
                resolved.append(p if p else d)
            dirs = resolved
        labels = list(args.labels) if args.labels else [os.path.basename(os.path.normpath(d)) for d in dirs]
        # 若只给了一个 log_dir，labels 长度要对齐
        while len(labels) < len(dirs):
            labels.append(os.path.basename(os.path.normpath(dirs[len(labels)])))
        log_configs = list(zip(dirs, labels[: len(dirs)]))
        episode_stats_paths = None
        if args.episode_stats:
            episode_stats_paths = list(args.episode_stats)
            while len(episode_stats_paths) < len(log_configs):
                episode_stats_paths.append(None)
            episode_stats_paths = episode_stats_paths[: len(log_configs)]
    else:
        episode_stats_paths = None
        # 示例：可改为你的 SAC / DQN 日志路径
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            (os.path.join(base, "dynamic_uav_logs"), "DQN"),
            (os.path.join(base, "SAC_dynamic_uav_logs"), "SAC"),
        ]
        log_configs = []
        for log_dir, label in candidates:
            if args.latest:
                p = _resolve_latest_run(log_dir)
                log_configs.append((p or log_dir, label))
            elif os.path.isdir(log_dir):
                p = _resolve_latest_run(log_dir)
                log_configs.append((p or log_dir, label))
        if not log_configs:
            print("未找到示例日志目录（dynamic_uav_logs / SAC_dynamic_uav_logs），请通过参数传入 log_dirs。")
            return
        episode_stats_paths = None

    output_path = args.output
    if not output_path and log_configs:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "packet_success_rate_convergence.png",
        )

    plot_packet_success_rate(
        log_configs,
        window=args.window,
        output_path=output_path,
        y_percent=args.percent,
        marker_style=not args.no_marker,
        use_reward_as_fallback=args.fallback_reward,
        episode_stats_paths=episode_stats_paths if args.log_dirs and args.episode_stats else None,
    )


if __name__ == "__main__":
    main()
