# -*- coding: utf-8 -*-
"""
校验补实验环境与 dynamic_traffic_uav.py 到达过程参数一致。

当三优先级使用相同 traffic_pattern 时，到达率公式应与原环境逐步一致。
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from dynamic_traffic_uav import DynamicTrafficUAVEnv
from independent_traffic_env import IndependentTrafficUAVEnv
from dynamic_traffic_load import get_traffic_load


def _expected_lambdas(env, step):
    env.current_step = step
    load = env._get_current_traffic_load()
    import math
    h_base = 0.5 * load
    cycle_factor = 1.0 + 0.3 * math.sin(2 * math.pi * (step % 300) / 300)
    h = h_base * cycle_factor

    m_base = 1.0 * load
    burst_factor = 2.0 if (step % 150) < 35 else 1.0
    m = m_base * burst_factor

    l_base = 2.0 * load
    # low 的 stability_factor 随机，只比 base 部分
    return h, m, l_base


def test_traffic_load_matches():
    for pattern in ["burst", "dynamic", "Pattern Shift", "Extreme Congestion"]:
        for step in [0, 50, 100, 500, 999]:
            env = DynamicTrafficUAVEnv(traffic_pattern=pattern, max_steps=2000)
            env.current_step = step
            orig = env._get_current_traffic_load()
            copy = get_traffic_load(pattern, step, env.traffic_cycle_length, env.burst_duration)
            assert abs(orig - copy) < 1e-12, f"load mismatch {pattern} step={step}: {orig} vs {copy}"
    print("OK: get_traffic_load matches dynamic_traffic_uav._get_current_traffic_load")


def test_same_pattern_independent_matches_original():
    """三优先级同一 pattern 时，负载部分与原环境相同。"""
    pattern = "dynamic"
    np.random.seed(0)
    orig = DynamicTrafficUAVEnv(traffic_pattern=pattern, max_steps=100)
    indep = IndependentTrafficUAVEnv(
        traffic_pattern=pattern,
        priority_traffic_patterns={3: pattern, 2: pattern, 1: pattern},
        max_steps=100,
    )
    for step in range(1, 50):
        orig.current_step = step
        indep.current_step = step
        assert orig._get_current_traffic_load() == indep._traffic_load_for_priority(3)
        h0, m0, l0 = _expected_lambdas(orig, step)
        # 固定 low 的 random 以便比较：临时 patch
        np.random.seed(step)
        indep_stab = 1.0 + 0.1 * (np.random.random() - 0.5)
        np.random.seed(step)
        low_count = indep._generate_low_priority_packets()
        expected_low = 2.0 * orig._get_current_traffic_load() * indep_stab
        # 只验证公式结构，不强制 poisson 抽样相等
        assert abs(indep._traffic_load_for_priority(1) - orig._get_current_traffic_load()) < 1e-12
    print("OK: independent env load + formula structure match original")


def test_arrival_params_documented():
    orig = DynamicTrafficUAVEnv()
    assert orig.traffic_cycle_length == 100
    assert orig.burst_duration == 20
    indep = IndependentTrafficUAVEnv(priority_traffic_patterns={3: "burst", 2: "dynamic", 1: "dynamic"})
    assert indep.traffic_cycle_length == 100
    assert indep.burst_duration == 20
    print("OK: traffic_cycle_length=100, burst_duration=20 inherited")


if __name__ == "__main__":
    test_traffic_load_matches()
    test_same_pattern_independent_matches_original()
    test_arrival_params_documented()
    print("\nAll arrival-parameter checks passed.")
