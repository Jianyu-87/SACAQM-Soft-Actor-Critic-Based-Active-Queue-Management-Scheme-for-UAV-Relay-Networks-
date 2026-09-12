# -*- coding: utf-8 -*-
"""
与 dynamic_traffic_uav.py 中 _get_current_traffic_load() 完全一致的负载计算。
供补实验各优先级独立模式复用，保证参数/公式与原环境相同。
"""

import math


def get_traffic_load(
    traffic_pattern: str,
    current_step: int,
    traffic_cycle_length: int = 100,
    burst_duration: int = 20,
) -> float:
    """复制 dynamic_traffic_uav.DynamicTrafficUAVEnv._get_current_traffic_load 逻辑。"""
    if traffic_pattern == "constant":
        return 0.8
    if traffic_pattern == "dynamic":
        cycle_position = current_step % traffic_cycle_length
        return 0.8 + 0.4 * math.sin(2 * math.pi * cycle_position / traffic_cycle_length)
    if traffic_pattern == "burst":
        if (current_step % traffic_cycle_length) < burst_duration:
            return 1.5
        return 0.5
    if traffic_pattern == "fluctuation":
        cycle_position = current_step % traffic_cycle_length
        if (cycle_position % 50) < 25:
            return 1.0
        return 0.5
    if traffic_pattern == "extreme_fluctuation1":
        return 1.0
    if traffic_pattern == "extreme_fluctuation2":
        return 0.5
    if traffic_pattern == "Pattern Shift":
        if (current_step < 750) or (current_step > 1500):
            return 0.5
        return 1.0
    if traffic_pattern == "Extreme Congestion":
        return 1.0
    if traffic_pattern == "Periodic Brust Traffic":
        if (current_step % traffic_cycle_length) < burst_duration:
            return 1.0
        return 0.5
    if traffic_pattern == "low_load":
        return 0.3
    if traffic_pattern == "medium_load":
        return 0.7
    if traffic_pattern == "high_load":
        return 1.0
    return 1.0
