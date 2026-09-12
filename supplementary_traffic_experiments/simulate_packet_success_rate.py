# -*- coding: utf-8 -*-
"""
三次补实验：DQN / SAC / CODEL 数据包成功率仿真。

成功率定义（与 plot_packet_success_rate.py 一致）：
  packet_success_rate = transmitted / (transmitted + expired)
各优先级成功率：
  priority_success_rate[p] = delay_satisfied_by_priority[p] / priority_generated[p]
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiment_configs import EXPERIMENT_CONFIGS, make_env, make_codel_env, make_dqn_env  # noqa: E402

DEFAULT_DQN_MODEL = os.path.join(
    _ROOT,
    "dynamic_uav_models/dynamic_20260304_112107/best_model/best_model.zip",
)
DEFAULT_SAC_MODEL = os.path.join(
    _ROOT,
    "SAC_dynamic_uav_models/mixed_20260313_090408/best_model/best_model.zip",
)

ALGO_LABELS = {"dqn": "DQN", "sac": "SAC", "codel": "CODEL"}


def _finalize_episode_stats(
    env,
    episode_reward,
    step_count,
    total_buffer_usage,
    transmission_count: Optional[int] = None,
) -> dict:
    """从环境状态汇总单个 episode 统计。"""
    packets_tx = int(sum(env.priority_transmitted))
    if transmission_count is None:
        transmission_count = int(getattr(env, "transmission_count", 0))

    if packets_tx > 0 and transmission_count > 0:
        transmission_efficiency = transmission_count / packets_tx
    else:
        transmission_efficiency = 0.0

    stats = {
        "priority_transmitted": env.priority_transmitted.copy(),
        "priority_generated": env.priority_generated.copy(),
        "transmit_num": packets_tx,
        "transmission_count": transmission_count,
        "transmission_efficiency": transmission_efficiency,
        "total_reward": episode_reward,
        "packets_expired": env.packets_expired,
        "delay_satisfied_count": env.delay_satisfied_count,
        "priority_success_rate": [0.0, 0.0, 0.0],
        "packet_success_rate": 0.0,
    }

    tx = env.packets_transmitted
    expired = env.packets_expired
    total = tx + expired
    stats["packet_success_rate"] = tx / total if total > 0 else 0.0

    for priority in [1, 2, 3]:
        gen = stats["priority_generated"][priority]
        if gen > 0:
            stats["priority_success_rate"][priority - 1] = (
                env.delay_satisfied_by_priority[priority] / gen
            )

    stats["avg_buffer_usage"] = total_buffer_usage / max(step_count, 1)
    return stats


def simulate_rl_algorithm(
    algo: str,
    model_path: Optional[str],
    exp_index: int,
    num_episodes: int = 20,
    max_steps: int = 2000,
) -> List[dict]:
    """运行 DQN 或 SAC。"""
    if algo == "dqn":
        from stable_baselines3 import DQN
        loader = DQN.load
    elif algo == "sac":
        from stable_baselines3 import SAC
        loader = SAC.load
    else:
        raise ValueError(f"RL 算法应为 dqn/sac，收到: {algo}")

    if not model_path or not os.path.isfile(model_path):
        print(f"  跳过 {algo.upper()}：模型不存在 {model_path}")
        return []

    model = loader(model_path)
    env_factory = make_dqn_env if algo == "dqn" else make_env
    env = env_factory(exp_index, max_steps=max_steps, max_queue=150, max_transmissions=None)

    if model.observation_space.shape != env.observation_space.shape:
        print(
            f"  跳过 {algo.upper()}：模型 obs {model.observation_space.shape} "
            f"与环境 obs {env.observation_space.shape} 不匹配"
        )
        return []

    all_stats = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        reward_sum = 0.0
        step_count = 0
        buffer_sum = 0
        done = False

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            step_count += 1
            buffer_sum += info.get("buffer_size", 0)
            done = terminated or truncated

        stats = _finalize_episode_stats(env, reward_sum, step_count, buffer_sum)
        stats["episode"] = ep + 1
        stats["algorithm"] = algo
        all_stats.append(stats)
        print(
            f"    {algo.upper()} ep{ep + 1}: "
            f"packet_sr={stats['packet_success_rate']:.3f} "
            f"P3={stats['priority_success_rate'][2]:.3f} "
            f"te={stats['transmission_efficiency']:.3f}"
        )

    return all_stats


def simulate_codel(
    exp_index: int,
    num_episodes: int = 20,
    max_steps: int = 2000,
) -> List[dict]:
    """运行 CODEL 基线。"""
    env = make_codel_env(
        exp_index,
        max_steps=max_steps,
        max_transmissions=None,
    )
    all_stats = []

    for ep in range(num_episodes):
        env.reset()
        reward_sum = 0.0
        step_count = 0
        buffer_sum = 0
        transmission_count = 0
        done = False

        while not done and step_count < max_steps:
            reward, done, info = env.step_codel()
            reward_sum += reward
            step_count += 1
            buffer_sum += info.get("buffer_size", 0)
            if info.get("packets_transmitted_this_step", 0) > 0:
                transmission_count += 1

        stats = _finalize_episode_stats(
            env, reward_sum, step_count, buffer_sum, transmission_count=transmission_count
        )
        stats["episode"] = ep + 1
        stats["algorithm"] = "codel"
        all_stats.append(stats)
        print(
            f"    CODEL ep{ep + 1}: "
            f"packet_sr={stats['packet_success_rate']:.3f} "
            f"P3={stats['priority_success_rate'][2]:.3f} "
            f"te={stats['transmission_efficiency']:.3f}"
        )

    return all_stats


def run_one_experiment(
    exp_index: int,
    save_dir: str,
    dqn_model: str = DEFAULT_DQN_MODEL,
    sac_model: str = DEFAULT_SAC_MODEL,
    num_episodes: int = 20,
    max_steps: int = 2000,
) -> Dict[str, List[dict]]:
    """运行单次补实验（三种算法）并保存 CSV。"""
    cfg = EXPERIMENT_CONFIGS[exp_index]
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(cfg["title"])
    print(f"结果目录: {save_dir}")
    print(f"{'=' * 60}")

    results = {}
    for algo, model in [("dqn", dqn_model), ("sac", sac_model)]:
        print(f"\n--- {algo.upper()} ---")
        results[algo] = simulate_rl_algorithm(
            algo, model, exp_index, num_episodes, max_steps
        )

    print("\n--- CODEL ---")
    results["codel"] = simulate_codel(exp_index, num_episodes, max_steps)

    rows = []
    for algo, stats_list in results.items():
        for s in stats_list:
            rows.append({
                "experiment": cfg["name"],
                "algorithm": ALGO_LABELS.get(algo, algo),
                "episode": s["episode"],
                "packet_success_rate": s["packet_success_rate"],
                "P1_success_rate": s["priority_success_rate"][0],
                "P2_success_rate": s["priority_success_rate"][1],
                "P3_success_rate": s["priority_success_rate"][2],
                "packets_transmitted": sum(s["priority_transmitted"]),
                "packets_expired": s["packets_expired"],
                "transmission_count": s["transmission_count"],
                "transmission_efficiency": s["transmission_efficiency"],
                "total_reward": s["total_reward"],
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "episode_packet_success_rate.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    for algo in ["dqn", "sac", "codel"]:
        sub = df[df["algorithm"] == ALGO_LABELS[algo]]
        if sub.empty:
            continue
        summary_rows.append({
            "experiment": cfg["name"],
            "algorithm": ALGO_LABELS[algo],
            "mean_packet_success_rate": sub["packet_success_rate"].mean(),
            "std_packet_success_rate": sub["packet_success_rate"].std(),
            "mean_P1_success_rate": sub["P1_success_rate"].mean(),
            "mean_P2_success_rate": sub["P2_success_rate"].mean(),
            "mean_P3_success_rate": sub["P3_success_rate"].mean(),
            "std_P3_success_rate": sub["P3_success_rate"].std(),
            "mean_transmission_efficiency": sub["transmission_efficiency"].mean(),
            "std_transmission_efficiency": sub["transmission_efficiency"].std(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(save_dir, "packet_success_rate_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 与 compare_dqn_sac_codel / plot_transmisson_efficiency 一致的宽表 CSV
    te_path = _save_transmission_efficiency_csv(results, save_dir)
    print(f"\n已保存: {csv_path}")
    print(f"已保存: {summary_path}")
    print(f"已保存: {te_path}")

    return results


def _save_transmission_efficiency_csv(results: Dict[str, List[dict]], save_dir: str) -> str:
    """写出 Episode, DeepAAQM, SACAQM, CODEL 宽表传输效率 CSV。"""
    col_map = {"dqn": "DeepAAQM", "sac": "SACAQM", "codel": "CODEL"}
    max_ep = max((len(v) for v in results.values()), default=0)
    rows = []
    for i in range(max_ep):
        row = {"Episode": i + 1}
        for algo, col in col_map.items():
            stats_list = results.get(algo) or []
            if i < len(stats_list):
                row[col] = stats_list[i]["transmission_efficiency"]
        rows.append(row)

    te_path = os.path.join(save_dir, "transmission_efficiency_comparison_chart_data.csv")
    pd.DataFrame(rows).to_csv(te_path, index=False, encoding="utf-8-sig")
    return te_path


def run_all_experiments(
    results_root: Optional[str] = None,
    num_episodes: int = 20,
    max_steps: int = 2000,
    dqn_model: str = DEFAULT_DQN_MODEL,
    sac_model: str = DEFAULT_SAC_MODEL,
) -> Dict[int, Dict[str, List[dict]]]:
    """运行三次补实验。"""
    if results_root is None:
        results_root = os.path.join(_HERE, "results")

    all_results = {}
    for i in range(len(EXPERIMENT_CONFIGS)):
        save_dir = os.path.join(results_root, EXPERIMENT_CONFIGS[i]["name"])
        all_results[i] = run_one_experiment(
            i, save_dir, dqn_model, sac_model, num_episodes, max_steps
        )
    return all_results
