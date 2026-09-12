#!/usr/bin/env bash
# 补实验：绘制三次独立到达模式图
# 使用项目常用的 dqn conda 环境

set -e
cd "$(dirname "$0")/.."
/home/qwh/miniconda3/envs/dqn/bin/python supplementary_traffic_experiments/run.py
