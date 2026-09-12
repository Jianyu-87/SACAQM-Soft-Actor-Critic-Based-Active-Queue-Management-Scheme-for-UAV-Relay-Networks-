#!/usr/bin/env bash
# 补实验：数据包成功率测试 + 绘图
# 可选参数示例：
#   bash supplementary_traffic_experiments/run_packet_success_rate.sh --episodes 20
#   bash supplementary_traffic_experiments/run_packet_success_rate.sh --skip-sim   # 仅重绘已有结果

set -e
cd "$(dirname "$0")/.."
/home/qwh/miniconda3/envs/dqn/bin/python supplementary_traffic_experiments/run_packet_success_rate.py "$@"
