# -*- coding: utf-8 -*-
"""
收敛性验证实验：在不同到达（流量）条件下训练并绘制学习曲线（收敛图），用于论文中证明算法收敛。

输出：收敛图（评估回报 vs 训练步数）+ 终端打印「收敛后性能」汇总表；默认每种流量模式单种子一条曲线。

用法：
-----
1) 先训练再绘图（推荐）：
   python convergence_experiment.py --run --total_timesteps 500000
   - 会在 convergence_study/exp_YYYYMMDD_HHMMSS/ 下为每种流量、每种子建子目录并训练，
     训练结束后自动从各 run 的 log 里读 evaluations.npz，画收敛图并保存为 convergence_curves.png，
     同时在终端打印各流量模式「最后 10% 评估点」的均值±标准差。

2) 只绘图（不训练，用已有结果）：
   python convergence_experiment.py --plot_only --results_dir convergence_study/exp_20260305_145216
   - 若省略 --results_dir，会尝试用 results_root 下最新的 exp_* 目录。

3) 常用参数：
   --patterns constant dynamic          # 只跑指定流量模式
   --seeds 0 1 2                        # 多种子（图上会画 mean±std）
   --total_timesteps 300000 --eval_freq 10000
   --results_root my_study              # 实验根目录
   --save path/to/out.png               # 指定收敛图保存路径
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 可选：仅绘图时不依赖 sb3 和 env
def _import_train():
    from train_dynamic_uav import train_sac
    return train_sac


# 论文中常用的流量模式标签（与 dynamic_traffic_uav 中一致）
DEFAULT_PATTERNS = [
    "constant",
    "dynamic",
    "burst",
    "extreme_fluctuation_1",
    "extreme_fluctuation_2",
]

# 种子（seed）：控制“随机性”的初始值。影响：环境初始状态、网络权重初始化、训练中的采样等。
# 同一种子 → 可复现同一次训练；不同种子 → 不同随机轨迹。多种子时画均值±标准差看稳定性；只想要收敛图时用单种子即可。


def run_convergence_training(
    results_root="convergence_study",
    patterns=None,
    seeds=(0,),
    total_timesteps=500000,
    eval_freq=5000,
):
    """
    在不同流量模式下、多种子训练，并把每次训练的 log_dir 放到 results_root 下固定结构中，
    便于后续统一读取 evaluations.npz 绘图。
    """
    train_sac = _import_train()
    patterns = patterns or DEFAULT_PATTERNS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(results_root, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    run_log_dirs = []  # [(pattern, seed, log_dir), ...]
    for pattern in patterns:
        for seed in seeds:
            run_name = f"{pattern}_seed{seed}"
            log_dir = os.path.join(exp_dir, run_name, "logs")
            model_dir = os.path.join(exp_dir, run_name, "models")
            tb_dir = os.path.join(exp_dir, run_name, "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            print(f"\n>>> 收敛性实验: {pattern} seed={seed}")
            try:
                train_sac(
                    traffic_pattern=pattern,
                    total_timesteps=total_timesteps,
                    eval_freq=eval_freq,
                    use_mixed_env=False,
                    seed=seed,
                    log_dir=log_dir,
                    model_dir=model_dir,
                    tb_log_dir=tb_dir,
                )
                run_log_dirs.append((pattern, seed, log_dir))
            except Exception as e:
                print(f"训练失败 {run_name}: {e}")
                continue

    return exp_dir, run_log_dirs


def _load_monitor_csv(log_dir):
    """当无 evaluations.npz 时，用 monitor.csv（训练时 Monitor 写入）构造收敛图数据。"""
    path = os.path.join(log_dir, "monitor.csv")
    if not os.path.isfile(path):
        return None
    # monitor.csv 格式：第1行 # 元数据，第2行 r,l,t，第3行起为数据
    try:
        data = np.genfromtxt(path, delimiter=",", skip_header=2, names=["r", "l", "t"], dtype=float)
    except Exception:
        return None
    if data.size == 0:
        return None
    rewards = np.atleast_1d(np.asarray(data["r"], dtype=float))
    lengths = np.atleast_1d(np.asarray(data["l"], dtype=float))
    timesteps = np.cumsum(lengths).astype(np.float64)
    return {"timesteps": timesteps, "mean_rewards": rewards}


def load_eval_log(log_dir):
    """从单次训练的 log_dir 中读取 EvalCallback 的 evaluations.npz；若无则用 monitor.csv 代替。"""
    for path in [
        os.path.join(log_dir, "evaluations", "evaluations.npz"),  # EvalCallback 默认写在此
        os.path.join(log_dir, "evaluations.npz"),
    ]:
        if os.path.isfile(path):
            break
    else:
        path = None
    if path is not None:
        data = np.load(path, allow_pickle=True)
        timesteps = np.asarray(data["timesteps"], dtype=np.float64)
        results = data["results"]
        if results.ndim == 2:
            mean_rewards = np.mean(results, axis=1)
        else:
            mean_rewards = np.array([np.mean(r) for r in results])
        return {"timesteps": timesteps, "mean_rewards": mean_rewards}
    return _load_monitor_csv(log_dir)


def collect_eval_logs_from_dir(results_dir, pattern_seed_dirs=None):
    """
    从 results_dir 下收集所有 evaluations.npz。
    results_dir 下可以是：
      - 直接包含 evaluations.npz，或
      - 子目录 run_name（如 constant_seed0）且 run_name/logs/evaluations.npz 或 run_name/evaluations.npz
    pattern_seed_dirs: 若为 None 则自动扫描子目录；否则为 [(pattern, seed, log_dir), ...]
    """
    collected = {}  # (pattern, seed) -> {"timesteps", "mean_rewards"}
    if pattern_seed_dirs is not None:
        for pattern, seed, log_dir in pattern_seed_dirs:
            d = load_eval_log(log_dir)
            if d is not None:
                collected[(pattern, seed)] = d
        return collected
    for name in os.listdir(results_dir):
        sub = os.path.join(results_dir, name)
        if not os.path.isdir(sub):
            continue
        # 尝试 "pattern_seedN" 解析
        m = re.match(r"^(.+)_seed(\d+)$", name)
        if m:
            pattern, seed_str = m.group(1), m.group(2)
            seed = int(seed_str)
        else:
            pattern, seed = name, 0
        for candidate in [sub, os.path.join(sub, "logs")]:
            d = load_eval_log(candidate)
            if d is not None:
                collected[(pattern, seed)] = d
                break
    return collected


def get_final_performance_table(collected, last_ratio=0.1):
    """
    取每条曲线最后 last_ratio 比例评估点的回报均值，按 (pattern, seed) 聚合，
    返回按 pattern 的 mean±std，用于论文中的“收敛后性能”表格。
    """
    table = {}
    for (pattern, seed), d in collected.items():
        r = d["mean_rewards"]
        n = max(1, int(len(r) * last_ratio))
        last_mean = np.mean(r[-n:])
        if pattern not in table:
            table[pattern] = []
        table[pattern].append(last_mean)
    summary = {}
    for pattern, vals in table.items():
        summary[pattern] = (np.mean(vals), np.std(vals) if len(vals) > 1 else 0.0)
    return summary


def plot_convergence(
    collected,
    save_path=None,
    title="算法收敛性：不同到达条件下的学习曲线",
    xlabel="训练步数 (Environment Steps)",
    ylabel="评估回报均值 (Mean Evaluation Return)",
    patterns_order=None,
    show_std=True,
):
    """
    绘制收敛曲线：按 pattern 聚合多种子为 mean±std，展示在各到达条件下均趋于稳定。
    collected: dict (pattern, seed) -> {"timesteps", "mean_rewards"}
    """
    patterns_order = patterns_order or DEFAULT_PATTERNS
    fig, ax = plt.subplots(figsize=(10, 6))
    for pattern in patterns_order:
        seeds_data = [collected[(pattern, s)] for s in sorted(set(s for (p, s) in collected if p == pattern))]
        if not seeds_data:
            continue
        ts = seeds_data[0]["timesteps"]
        rewards = np.array([d["mean_rewards"] for d in seeds_data])
        if rewards.shape[1] != len(ts):
            min_len = min(len(d["mean_rewards"]) for d in seeds_data)
            ts = ts[:min_len]
            rewards = np.array([d["mean_rewards"][:min_len] for d in seeds_data])
        mean_r = np.mean(rewards, axis=0)
        std_r = np.std(rewards, axis=0) if rewards.shape[0] > 1 else np.zeros_like(mean_r)
        label = pattern.replace("_", " ").title()
        ax.plot(ts, mean_r, label=label, linewidth=2)
        if show_std and np.any(std_r > 0):
            ax.fill_between(ts, mean_r - std_r, mean_r + std_r, alpha=0.25)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"收敛曲线已保存: {save_path}")
    plt.close(fig)


def print_convergence_summary(collected, last_ratio=0.1):
    """打印收敛后性能汇总表（可复制到论文）。"""
    summary = get_final_performance_table(collected, last_ratio=last_ratio)
    print("\n--- 收敛后评估回报 (最后 {:.0%} 评估点) ---".format(last_ratio))
    for pattern in sorted(summary.keys()):
        mu, std = summary[pattern]
        print("  {}: {:.4f} ± {:.4f}".format(pattern, mu, std))
    print("")


def main():
    parser = argparse.ArgumentParser(description="收敛性实验：多流量条件训练并绘制学习曲线")
    parser.add_argument("--run", action="store_true", help="执行训练（否则仅绘图需配合 --plot_only）")
    parser.add_argument("--plot_only", action="store_true", help="仅从已有结果目录绘图")
    parser.add_argument("--results_dir", type=str, default=None, help="已有结果根目录（仅绘图时使用）")
    parser.add_argument("--results_root", type=str, default="convergence_study", help="新实验保存根目录")
    parser.add_argument("--patterns", type=str, nargs="+", default=DEFAULT_PATTERNS, help="流量模式列表")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="随机种子列表（只画收敛图时用 [0] 即可）")
    parser.add_argument("--total_timesteps", type=int, default=500000, help="每次训练总步数")
    parser.add_argument("--eval_freq", type=int, default=5000, help="评估频率")
    parser.add_argument("--save", type=str, default=None, help="收敛曲线图保存路径（默认在结果目录下）")
    args = parser.parse_args()

    if args.run:
        exp_dir, run_log_dirs = run_convergence_training(
            results_root=args.results_root,
            patterns=args.patterns,
            seeds=args.seeds,
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
        )
        collected = collect_eval_logs_from_dir(exp_dir, pattern_seed_dirs=run_log_dirs)
        results_dir = exp_dir
    elif args.plot_only:
        results_dir = args.results_dir
        if not results_dir and os.path.isdir(args.results_root):
            # 取最新一次实验目录
            subs = [d for d in os.listdir(args.results_root) if os.path.isdir(os.path.join(args.results_root, d)) and d.startswith("exp_")]
            if subs:
                results_dir = os.path.join(args.results_root, max(subs, key=lambda x: x))
        if not results_dir or not os.path.isdir(results_dir):
            print("请指定有效的 --results_dir 或先运行 --run 生成结果目录")
            return
        collected = collect_eval_logs_from_dir(results_dir)
    else:
        print("请使用 --run 执行训练并绘图，或 --plot_only 并指定 --results_dir 仅绘图")
        return

    if not collected:
        print("未找到任何 evaluations.npz，无法绘图")
        return

    print_convergence_summary(collected, last_ratio=0.1)
    save_path = args.save or os.path.join(results_dir, "convergence_curves.png")
    plot_convergence(
        collected,
        save_path=save_path,
        patterns_order=args.patterns,
    )


if __name__ == "__main__":
    main()
