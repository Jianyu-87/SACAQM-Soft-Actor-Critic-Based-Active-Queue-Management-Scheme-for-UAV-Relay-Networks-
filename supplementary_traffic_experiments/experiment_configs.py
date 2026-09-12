# -*- coding: utf-8 -*-
"""三次补实验的环境配置。"""

EXPERIMENT_CONFIGS = [
    {
        "name": "exp1_sinusoidal",
        "title": "实验1：高优先级突发 + 中/低正弦到达",
        "plot_title": "Exp.1: High=Burst, Med/Low=Sinusoidal (dynamic)",
        "priority_traffic_patterns": {
            3: "burst",
            2: "dynamic",
            1: "dynamic",
        },
    },
    {
        "name": "exp2_state_switching",
        "title": "实验2：高优先级突发 + 中/低状态切换到达",
        "plot_title": "Exp.2: High=Burst, Med/Low=State Switching (Pattern Shift)",
        "priority_traffic_patterns": {
            3: "burst",
            2: "Pattern Shift",
            1: "Pattern Shift",
        },
    },
    {
        "name": "exp3_near_saturated",
        "title": "实验3：高优先级突发 + 中/低近饱和到达",
        "plot_title": "Exp.3: High=Burst, Med/Low=Near-Saturated (Extreme Congestion)",
        "priority_traffic_patterns": {
            3: "burst",
            2: "Extreme Congestion",
            1: "Extreme Congestion",
        },
    },
]


def make_dqn_env(config_index, **env_kwargs):
    """按实验编号创建 IndependentDQNTrafficUAVEnv（153 维 obs）。"""
    from independent_dqn_env import IndependentDQNTrafficUAVEnv

    cfg = EXPERIMENT_CONFIGS[config_index]
    patterns = cfg["priority_traffic_patterns"]
    return IndependentDQNTrafficUAVEnv(
        priority_traffic_patterns=patterns,
        traffic_pattern=patterns[2],
        **env_kwargs,
    )


def make_env(config_index, **env_kwargs):
    """按实验编号创建 IndependentTrafficUAVEnv。"""
    from independent_traffic_env import IndependentTrafficUAVEnv

    cfg = EXPERIMENT_CONFIGS[config_index]
    patterns = cfg["priority_traffic_patterns"]
    return IndependentTrafficUAVEnv(
        priority_traffic_patterns=patterns,
        traffic_pattern=patterns[2],
        **env_kwargs,
    )


def make_codel_env(config_index, **env_kwargs):
    """按实验编号创建 IndependentCoDelTrafficUAVEnv。"""
    from independent_codel_env import IndependentCoDelTrafficUAVEnv

    cfg = EXPERIMENT_CONFIGS[config_index]
    patterns = cfg["priority_traffic_patterns"]
    defaults = dict(
        target_delay=4,
        interval=12,
        max_drops_per_interval=3,
        adaptive_processing=True,
        max_queue=100,
    )
    defaults.update(env_kwargs)
    return IndependentCoDelTrafficUAVEnv(
        priority_traffic_patterns=patterns,
        traffic_pattern=patterns[2],
        **defaults,
    )
