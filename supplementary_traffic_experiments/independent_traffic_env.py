# -*- coding: utf-8 -*-
"""
各优先级独立到达过程的环境扩展（不修改原 dynamic_traffic_uav.py）。

继承 DynamicTrafficUAVEnv，通过 priority_traffic_patterns 为每个优先级
指定独立的 env 内置流量模式，实现异步/独立到达时序。
"""

import os
import sys

import numpy as np

# 允许从项目根目录导入原环境
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dynamic_traffic_uav import DynamicTrafficUAVEnv, Packet  # noqa: E402

from traffic_independence_mixin import TrafficIndependenceMixin  # noqa: E402


class IndependentTrafficUAVEnv(TrafficIndependenceMixin, DynamicTrafficUAVEnv):
    """
    各优先级独立流量模式的环境。

    priority_traffic_patterns 示例::
        {3: "burst", 2: "dynamic", 1: "dynamic"}
    """

    def __init__(self, priority_traffic_patterns=None, **kwargs):
        super().__init__(**kwargs)
        self.priority_traffic_patterns = priority_traffic_patterns

    def simulate_arrival_counts(self, num_steps, seed=None):
        """
        仅模拟到达过程（不传输），返回 shape=(num_steps, 3) 的 [高, 中, 低] 到达计数。
        """
        self.reset(seed=seed)
        counts = np.zeros((num_steps, 3), dtype=int)
        for t in range(num_steps):
            self.current_step += 1
            counts[t, 0] = self._generate_high_priority_packets()
            counts[t, 1] = self._generate_medium_priority_packets()
            counts[t, 2] = self._generate_low_priority_packets()
        return counts
