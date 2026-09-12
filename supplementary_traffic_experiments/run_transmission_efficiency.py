# -*- coding: utf-8 -*-
"""
补实验入口：三次独立流量场景的传输效率评测（默认 100 episodes）并绘图。

运行（项目根目录）：
    /home/qwh/miniconda3/envs/dqn/bin/python \\
        supplementary_traffic_experiments/run_transmission_efficiency.py

流程：
  1. 对实验1/2/3 分别运行 DQN、SAC、CODEL（默认 100 episodes）
  2. 写出 transmission_efficiency_comparison_chart_data.csv
  3. 绘制三图并排折线（纵轴：每包能耗 mJ）
"""

import argparse
import os
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root in sys.path:
        sys.path.remove(root)
    if here not in sys.path:
        sys.path.insert(0, here)
    if root not in sys.path:
        sys.path.insert(1, root)

    parser = argparse.ArgumentParser(description="补实验：传输效率（100 episodes）")
    parser.add_argument("--episodes", type=int, default=100, help="每个算法 episode 数")
    parser.add_argument("--max-steps", type=int, default=2000, help="每 episode 最大步数")
    parser.add_argument("--dqn-model", type=str, default=None)
    parser.add_argument("--sac-model", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=os.path.join(here, "results"))
    parser.add_argument("--skip-sim", action="store_true", help="跳过仿真，仅绘图")
    parser.add_argument("--skip-plot", action="store_true", help="跳过绘图，仅仿真")
    args = parser.parse_args()

    from simulate_packet_success_rate import (
        DEFAULT_DQN_MODEL,
        DEFAULT_SAC_MODEL,
        run_all_experiments,
    )
    from plot_transmission_efficiency import plot_from_results

    dqn_model = args.dqn_model or DEFAULT_DQN_MODEL
    sac_model = args.sac_model or DEFAULT_SAC_MODEL

    print("=" * 60)
    print("补实验：传输效率（DQN / SAC / CODEL × 3 次实验）")
    print("=" * 60)
    print(f"DQN 模型: {dqn_model}")
    print(f"SAC 模型: {sac_model}")
    print(f"Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print(f"结果目录: {args.results_root}")

    if not args.skip_sim:
        run_all_experiments(
            results_root=args.results_root,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            dqn_model=dqn_model,
            sac_model=sac_model,
        )

    if not args.skip_plot:
        print("\n" + "=" * 60)
        print("开始绘制传输效率图")
        print("=" * 60)
        plot_from_results(args.results_root)

    print("\n完成。")


if __name__ == "__main__":
    main()
