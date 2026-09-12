# -*- coding: utf-8 -*-
"""
DQN 环境（153 维 obs）+ 各优先级独立到达模式。
到达过程参数与 dynamic_traffic_uav.py / dqn_dynamic_uav.py 保持一致。
"""

import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dqn_dynamic_uav import DynamicTrafficUAVEnv, Packet  # noqa: E402

from dynamic_traffic_load import get_traffic_load  # noqa: E402


class IndependentDQNTrafficUAVEnv(DynamicTrafficUAVEnv):
    """DQN 专用独立流量环境，兼容 153 维 observation。"""

    def __init__(self, priority_traffic_patterns=None, **kwargs):
        super().__init__(**kwargs)
        self.priority_traffic_patterns = priority_traffic_patterns

    def _get_pattern_for_priority(self, priority):
        if self.priority_traffic_patterns:
            return self.priority_traffic_patterns.get(priority, self.traffic_pattern)
        return self.traffic_pattern

    def _traffic_load_for_priority(self, priority):
        if not self.priority_traffic_patterns:
            return self._get_current_traffic_load()
        pattern = self._get_pattern_for_priority(priority)
        return get_traffic_load(
            pattern,
            self.current_step,
            self.traffic_cycle_length,
            self.burst_duration,
        )

    def _generate_high_priority_packets(self):
        if not self.priority_traffic_patterns:
            return super()._generate_high_priority_packets()

        traffic_load = self._traffic_load_for_priority(3)
        base_lambda = 0.5 * traffic_load
        cycle_length = 300
        cycle_position = self.current_step % cycle_length
        cycle_factor = 1.0 + 0.3 * math.sin(2 * math.pi * cycle_position / cycle_length)
        final_lambda = base_lambda * cycle_factor

        packets_generated = 0
        for _ in range(np.random.poisson(final_lambda)):
            if len(self.buffer) < self.max_queue:
                self.buffer.append(Packet(
                    sensor_id=0, max_delay=5,
                    arrival_time=self.current_step, priority=3,
                ))
                self.priority_generated[3] += 1
                packets_generated += 1
        return packets_generated

    def _generate_medium_priority_packets(self):
        if not self.priority_traffic_patterns:
            return super()._generate_medium_priority_packets()

        traffic_load = self._traffic_load_for_priority(2)
        base_lambda = 1.0 * traffic_load
        burst_cycle = 150
        burst_duration = 35
        cycle_position = self.current_step % burst_cycle
        burst_factor = 2.0 if cycle_position < burst_duration else 1.0
        final_lambda = base_lambda * burst_factor

        packets_generated = 0
        for _ in range(np.random.poisson(final_lambda)):
            if len(self.buffer) < self.max_queue:
                self.buffer.append(Packet(
                    sensor_id=1, max_delay=10,
                    arrival_time=self.current_step, priority=2,
                ))
                self.priority_generated[2] += 1
                packets_generated += 1
        return packets_generated

    def _generate_low_priority_packets(self):
        if not self.priority_traffic_patterns:
            return super()._generate_low_priority_packets()

        traffic_load = self._traffic_load_for_priority(1)
        base_lambda = 2.0 * traffic_load
        stability_factor = 1.0 + 0.1 * (np.random.random() - 0.5)
        final_lambda = base_lambda * stability_factor

        packets_generated = 0
        for _ in range(np.random.poisson(final_lambda)):
            if len(self.buffer) < self.max_queue:
                self.buffer.append(Packet(
                    sensor_id=2, max_delay=15,
                    arrival_time=self.current_step, priority=1,
                ))
                self.priority_generated[1] += 1
                packets_generated += 1
        return packets_generated
