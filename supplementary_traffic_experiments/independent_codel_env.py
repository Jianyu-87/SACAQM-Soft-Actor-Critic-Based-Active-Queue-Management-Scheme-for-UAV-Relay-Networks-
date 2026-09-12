# -*- coding: utf-8 -*-
"""CoDel 环境 + 各优先级独立到达模式。"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from codel_dynamic_uav import CoDelDynamicUAVEnv  # noqa: E402

from traffic_independence_mixin import TrafficIndependenceMixin  # noqa: E402


class IndependentCoDelTrafficUAVEnv(TrafficIndependenceMixin, CoDelDynamicUAVEnv):
    """CoDel 算法在独立流量模式下的环境。"""

    def __init__(self, priority_traffic_patterns=None, **kwargs):
        super().__init__(**kwargs)
        self.priority_traffic_patterns = priority_traffic_patterns
