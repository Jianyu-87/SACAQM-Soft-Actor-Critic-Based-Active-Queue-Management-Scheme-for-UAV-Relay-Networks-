# 补实验说明：各优先级独立流量到达模式下的数据包成功率

本文档只说明**实验如何做**，不包含结果分析。

## 1. 实验目的

在不修改原环境 `dynamic_traffic_uav.py` 的前提下，构造一组补实验：让高 / 中 / 低优先级数据包**各自独立**采用不同的流量到达模式，再对比 DQN、SAC、CoDel 三种算法的数据包成功率。

原环境中三个优先级共用同一个 `traffic_pattern`，到达时序同步；本补实验通过 `priority_traffic_patterns` 为每个优先级指定独立模式，实现异步 / 独立到达过程。

## 2. 目录结构

```
supplementary_traffic_experiments/
├── experiment_configs.py          # 三次实验的流量配置
├── dynamic_traffic_load.py        # 与原环境一致的负载计算公式
├── traffic_independence_mixin.py  # 各优先级独立到达逻辑（Mixin）
├── independent_traffic_env.py     # SAC / 通用独立流量环境（7 维 obs）
├── independent_dqn_env.py         # DQN 独立流量环境（153 维 obs）
├── independent_codel_env.py       # CoDel 独立流量环境
├── simulate_packet_success_rate.py# 仿真：跑算法并写 CSV
├── plot_packet_success_rate.py    # 成功率绘图
├── plot_traffic_arrival_patterns.py # 到达模式可视化
├── plot.py                        # 从 CSV 画 P1/P2/P3 分组柱状图
├── verify_arrival_params.py       # 到达参数与原环境一致性校验
├── run_packet_success_rate.py     # 统一入口（仿真 + 绘图）
├── run_packet_success_rate.sh
├── results/                       # 仿真输出 CSV
└── figures/                       # 图片输出
```

## 3. 流量设置

### 3.1 与原环境保持一致的部分

到达过程参数与 `dynamic_traffic_uav.py` 对齐：

| 项目 | 取值 |
|------|------|
| 高优先级 base_lambda | `0.5 × traffic_load` |
| 中优先级 base_lambda | `1.0 × traffic_load` |
| 低优先级 base_lambda | `2.0 × traffic_load` |
| 高优先级调制 | `cycle_length=300`，`cycle_factor = 1 + 0.3·sin(...)` |
| 中优先级调制 | `burst_cycle=150`，`burst_duration=35`，factor 2.0 / 1.0 |
| 低优先级调制 | `stability_factor = 1 + 0.1·(U(0,1)−0.5)` |
| max_delay | 高 5 / 中 10 / 低 15 |
| 负载周期参数 | `traffic_cycle_length=100`，`burst_duration=20` |

负载模式公式（`burst` / `dynamic` / `Pattern Shift` / `Extreme Congestion` 等）从原环境复制到 `dynamic_traffic_load.py`，并用 `verify_arrival_params.py` 做一致性校验。

### 3.2 与原环境的唯一设计差异

原环境：三个优先级共用同一个 `traffic_pattern` 的 `traffic_load`。

补实验：每个优先级读取各自的 `priority_traffic_patterns`，公式相同，但 pattern 不同，从而形成独立到达过程。

### 3.3 三次实验配置

| 实验 | 名称 | 高优先级 (P3) | 中 / 低优先级 (P2 / P1) |
|------|------|---------------|-------------------------|
| Exp.1 | `exp1_sinusoidal` | `burst`（突发） | `dynamic`（正弦） |
| Exp.2 | `exp2_state_switching` | `burst` | `Pattern Shift`（状态切换） |
| Exp.3 | `exp3_near_saturated` | `burst` | `Extreme Congestion`（近饱和） |

配置定义见 `experiment_configs.py`。

## 4. 环境与算法适配

- **SAC / CoDel**：基于 `dynamic_traffic_uav.DynamicTrafficUAVEnv`（7 维观测），通过 Mixin / 子类注入独立到达逻辑。
- **DQN**：训练模型为 153 维观测，因此单独基于 `dqn_dynamic_uav.DynamicTrafficUAVEnv` 封装 `IndependentDQNTrafficUAVEnv`，到达公式与上述一致。

默认模型路径：

- DQN：`dynamic_uav_models/dynamic_20260304_112107/best_model/best_model.zip`
- SAC：`SAC_dynamic_uav_models/mixed_20260313_090408/best_model/best_model.zip`
- CoDel：无需预训练模型，环境内直接运行

## 5. 评测流程

对每个实验场景，依次运行 DQN、SAC、CoDel：

1. 按该实验的 `priority_traffic_patterns` 创建对应环境。
2. DQN / SAC：加载已训练模型，`predict(..., deterministic=True)` 做确定性推理。
3. CoDel：同一流量配置下调用 `step_codel()` 步进。
4. 默认每个算法 **20 个 episode**，每 episode 最多 **2000 步**。
5. 每个 episode 结束后统计成功率等指标，写入 CSV。

统一入口：

```bash
bash supplementary_traffic_experiments/run_packet_success_rate.sh --episodes 20
```

或：

```bash
/home/qwh/miniconda3/envs/dqn/bin/python supplementary_traffic_experiments/run_packet_success_rate.py
```

常用参数：

- `--episodes`：每个算法的 episode 数
- `--max-steps`：每 episode 最大步数
- `--dqn-model` / `--sac-model`：指定模型路径
- `--skip-sim`：跳过仿真，仅根据已有 CSV 重绘
- `--skip-plot`：只仿真不绘图

## 6. 评价指标

- **总体数据包成功率**  
  `packet_success_rate = transmitted / (transmitted + expired)`

- **各优先级成功率**  
  `priority_success_rate[p] = delay_satisfied_by_priority[p] / priority_generated[p]`  
  （按时延约束是否满足统计）

汇总 CSV 中同时给出多个 episode 的 **mean** 与 **std**（标准差）。

## 7. 输出文件

### 7.1 仿真结果

```
results/
├── exp1_sinusoidal/
│   ├── episode_packet_success_rate.csv   # 逐 episode 明细
│   └── packet_success_rate_summary.csv   # 算法级 mean / std
├── exp2_state_switching/
└── exp3_near_saturated/
```

### 7.2 图表

- 到达模式图：`figures/traffic_arrival_exp*.png`（只模拟到达、不传输）
- 成功率图：`figures/packet_success_rate/`  
  - 各实验 P1/P2/P3 分组柱状图  
  - 总体成功率柱状图  
  - 三次实验 P3 对比折线图  

从 CSV 单独出图：

```bash
cd supplementary_traffic_experiments
python plot.py              # 三次实验各一张
python plot.py --exp 1      # 只画实验 1
```

## 8. 小结

本补实验在**保持原环境到达公式与参数不变**的条件下，仅将高 / 中 / 低优先级的流量模式解耦为独立过程；在三种解耦场景下，用同一套评测协议对比 DQN、SAC、CoDel 的数据包成功率。
