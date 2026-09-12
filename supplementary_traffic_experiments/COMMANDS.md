# 补实验命令合集

工作目录默认在项目根目录：

```bash
cd /home/qwh/nndqn
```

Python 环境（推荐）：

```bash
PY=/home/qwh/miniconda3/envs/dqn/bin/python
```

下文中的 `$PY` 均指上述解释器。

---

## 1. 一键跑完整流程（仿真 + 绘图）

```bash
# 推荐：正式实验（20 episodes × 2000 steps）
bash supplementary_traffic_experiments/run_packet_success_rate.sh --episodes 20 --max-steps 2000

# 或直接调用 Python
$PY supplementary_traffic_experiments/run_packet_success_rate.py --episodes 20 --max-steps 2000
```

输出：
- CSV → `supplementary_traffic_experiments/results/exp*/`
- 图 → `supplementary_traffic_experiments/figures/packet_success_rate/`

---

## 2. 只仿真 / 只绘图

```bash
# 只仿真写 CSV，不画图（DATA_SUMMARY 用的就是这条）
$PY supplementary_traffic_experiments/run_packet_success_rate.py --episodes 20 --max-steps 2000 --skip-plot

# 已有 CSV，只重绘成功率图
$PY supplementary_traffic_experiments/run_packet_success_rate.py --skip-sim

# shell 等价写法
bash supplementary_traffic_experiments/run_packet_success_rate.sh --episodes 20 --max-steps 2000 --skip-plot
bash supplementary_traffic_experiments/run_packet_success_rate.sh --skip-sim
```

---

## 3. 指定模型路径

```bash
$PY supplementary_traffic_experiments/run_packet_success_rate.py \
  --episodes 20 \
  --max-steps 2000 \
  --dqn-model /home/qwh/nndqn/dynamic_uav_models/dynamic_20260304_112107/best_model/best_model.zip \
  --sac-model /home/qwh/nndqn/SAC_dynamic_uav_models/mixed_20260313_090408/best_model/best_model.zip
```

默认模型（不传参数时）：
- DQN：`dynamic_uav_models/dynamic_20260304_112107/best_model/best_model.zip`
- SAC：`SAC_dynamic_uav_models/mixed_20260313_090408/best_model/best_model.zip`

---

## 4. 快速冒烟（少 episode / 少步数）

```bash
$PY supplementary_traffic_experiments/run_packet_success_rate.py --episodes 2 --max-steps 500
```

---

## 5. 到达模式可视化（三张到达曲线图）

```bash
# 统一入口
$PY supplementary_traffic_experiments/run.py

# 或 shell
bash supplementary_traffic_experiments/run.sh

# 或直接跑绘图脚本
cd supplementary_traffic_experiments
$PY plot_traffic_arrival_patterns.py
```

输出：`supplementary_traffic_experiments/figures/traffic_arrival_exp{1,2,3}.png`

---

## 6. 从 CSV 画 P1/P2/P3 分组柱状图（plot.py）

需在 `supplementary_traffic_experiments/` 目录下，或保证能 import 到同目录模块：

```bash
cd /home/qwh/nndqn/supplementary_traffic_experiments

# 三次实验各出一张图
$PY plot.py

# 只画实验 1 / 2 / 3
$PY plot.py --exp 1
$PY plot.py --exp 2
$PY plot.py --exp 3

# 指定单个 CSV
$PY plot.py --csv results/exp1_sinusoidal/packet_success_rate_summary.csv
```

输出：`figures/packet_success_rate/exp*_priority_success_rate.png`

---

## 7. 成功率全套绘图（plot_packet_success_rate.py）

```bash
cd /home/qwh/nndqn/supplementary_traffic_experiments
$PY plot_packet_success_rate.py

# 指定结果目录 / 输出目录
$PY plot_packet_success_rate.py \
  --results-root results \
  --figures-dir figures/packet_success_rate
```

---

## 8. 到达参数一致性校验

```bash
$PY supplementary_traffic_experiments/verify_arrival_params.py
```

用于确认补实验环境与 `dynamic_traffic_uav.py` 的到达公式 / 参数一致。

---

## 9. 常用参数一览（run_packet_success_rate.py）

| 参数 | 含义 | 默认 |
|------|------|------|
| `--episodes` | 每个算法 episode 数 | 20 |
| `--max-steps` | 每 episode 最大步数 | 2000 |
| `--dqn-model` | DQN 模型路径 | 见上文默认 |
| `--sac-model` | SAC 模型路径 | 见上文默认 |
| `--results-root` | CSV 输出根目录 | `.../results` |
| `--skip-sim` | 跳过仿真，仅绘图 | 关 |
| `--skip-plot` | 跳过绘图，仅仿真 | 关 |

---

## 10. 结果文件位置

```text
supplementary_traffic_experiments/
├── results/
│   ├── exp1_sinusoidal/
│   │   ├── episode_packet_success_rate.csv
│   │   └── packet_success_rate_summary.csv
│   ├── exp2_state_switching/
│   └── exp3_near_saturated/
├── figures/
│   ├── traffic_arrival_exp*.png
│   └── packet_success_rate/
├── README.md          # 实验方法说明
├── DATA_SUMMARY.md    # A/B/C 数据汇总
└── COMMANDS.md        # 本命令合集
```

---

## 11. 与 DATA_SUMMARY.md 对应的复现命令

生成 A/B 表（含能耗）所用命令：

```bash
cd /home/qwh/nndqn
/home/qwh/miniconda3/envs/dqn/bin/python supplementary_traffic_experiments/run_packet_success_rate.py \
  --episodes 20 \
  --max-steps 2000 \
  --skip-plot
```

C 表（原同步对照）不是本目录新跑的，而是读取已有结果：
- `algorithm_comparison_20260316_151550/`（bursty）
- `algorithm_comparison_20260316_151028/`（near-saturation）
