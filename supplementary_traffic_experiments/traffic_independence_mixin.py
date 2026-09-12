# -*- coding: utf-8 -*-
"""
各优先级独立流量模式的 Mixin，供 RL / CoDel 环境复用。

与 dynamic_traffic_uav.py 保持相同的到达过程参数：
  - base_lambda: 高 0.5 / 中 1.0 / 低 2.0 × traffic_load
  - 高优先级: cycle_length=300, cycle_factor=1+0.3*sin(...)
  - 中优先级: burst_cycle=150, burst_duration=35, burst_factor=2.0/1.0
  - 低优先级: stability_factor=1+0.1*(U(0,1)-0.5)
  - max_delay: 5 / 10 / 15
  - traffic_cycle_length=100, burst_duration=20（负载模式参数）

唯一区别：各优先级使用 priority_traffic_patterns 指定的独立 traffic_pattern。
"""

import math

import numpy as np

from dynamic_traffic_load import get_traffic_load
from dynamic_traffic_uav import Packet


class TrafficIndependenceMixin:
    """为环境注入 priority_traffic_patterns 独立到达逻辑。"""

    priority_traffic_patterns = None

    def _get_pattern_for_priority(self, priority):
        if self.priority_traffic_patterns:
            return self.priority_traffic_patterns.get(priority, self.traffic_pattern)
        return self.traffic_pattern

    def _traffic_load_for_priority(self, priority):
        """按优先级取独立 pattern 的负载（公式同 dynamic_traffic_uav）。"""
        if not self.priority_traffic_patterns:
            return self._get_current_traffic_load()
        pattern = self._get_pattern_for_priority(priority)
        return get_traffic_load(
            pattern,
            self.current_step,
            self.traffic_cycle_length,
            self.burst_duration,
        )

    def _get_current_traffic_load(self, pattern=None):
        if pattern is None:
            return super()._get_current_traffic_load()
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
                self.buffer.append(Packet(priority=3, arrival_time=self.current_step, max_delay=5))
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
                self.buffer.append(Packet(priority=2, arrival_time=self.current_step, max_delay=10))
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
                self.buffer.append(Packet(priority=1, arrival_time=self.current_step, max_delay=15))
                self.priority_generated[1] += 1
                packets_generated += 1
        return packets_generated
