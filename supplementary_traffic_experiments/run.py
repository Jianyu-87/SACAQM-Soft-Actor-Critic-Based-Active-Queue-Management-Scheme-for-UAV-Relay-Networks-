# -*- coding: utf-8 -*-
"""
补实验统一入口：绘制三次独立到达模式图。

运行方式（任选其一）：

  1. 项目根目录下，使用 dqn 环境（推荐）：
     /home/qwh/miniconda3/envs/dqn/bin/python supplementary_traffic_experiments/run.py

  2. 执行 shell 脚本（已内置 dqn 环境路径）：
     bash supplementary_traffic_experiments/run.sh

  3. 进入本目录后运行：
     python run.py

输出目录：
  supplementary_traffic_experiments/figures/
    traffic_arrival_exp1.png  高=burst，中/低=正弦(dynamic)
    traffic_arrival_exp2.png  高=burst，中/低=状态切换(Pattern Shift)
    traffic_arrival_exp3.png  高=burst，中/低=近饱和(Extreme Congestion)
"""

import os
import sys


def main():
    # 保证无论从哪个目录执行，都能 import 到本包和项目根目录的原 env
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for path in (here, root):
        if path not in sys.path:
            sys.path.insert(0, path)

    from plot_traffic_arrival_patterns import main as plot_main

    print("=" * 60)
    print("补实验：各优先级独立到达模式可视化")
    print("=" * 60)
    plot_main()


if __name__ == "__main__":
    main()
