import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import scienceplots

# 应用SciencePlots样式
plt.style.use(['science', 'ieee','bright'])

# 自定义样式
plt.rcParams['lines.linewidth'] = 2
# 将整体字号调小（含坐标轴标签、刻度、图例等）
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 6

# --- CSV 纵轴为「次数」时：单次转发能耗 (mJ)，仅用于绘图缩放，不改变 read_csv ---
ENERGY_PER_FORWARD_MJ = 0.24
# "as_csv" — 纵轴与 CSV 相同（次数）
# "energy_mj" — 纵轴 = 次数 × ENERGY_PER_FORWARD_MJ（mJ，例如每 episode 总能耗）
Y_DISPLAY_MODE = "energy_mj"  # 或 "as_csv"

# --- 纵轴范围 ---
# None：由三条曲线自动取 min/max，并在上下各加一段冗余（见 Y_PAD_FRAC），不贴边
# 能耗模式请用 None 或按数据量级设元组；(0.23, 0.55) 仅适合 0~1 的无量纲曲线
Y_LIM = None
# 自动模式：在数据跨度上下各留该比例的空档（例如 0.12 表示上下各约 12% 跨度）
Y_PAD_FRAC = 0.12


def transform_y_for_plot(y, mode: str, e_mj: float):
    y = np.asarray(y, dtype=float)
    if mode == "as_csv":
        return y
    if mode == "energy_mj":
        return y * e_mj
    raise ValueError(f"未知 Y_DISPLAY_MODE: {mode}")

# --- CSV：把路径改成你的文件；列名与表头一致 ---
# pd.read_csv 常用参数：encoding='utf-8-sig'（Excel 导出的 UTF-8 BOM）
# sep=','（默认逗号）、delimiter=None、usecols=['A','B'] 只读部分列
# CSV_PATH = Path("/home/qwh/nndqn/algorithm_comparison_20260316_144737/transmission_efficiency_comparison_chart_data.csv")
# CSV_PATH = Path("/home/qwh/nndqn/algorithm_comparison_20260316_150509/transmission_efficiency_comparison_chart_data.csv")
# CSV_PATH = Path("/home/qwh/nndqn/algorithm_comparison_20260316_151028/transmission_efficiency_comparison_chart_data.csv")
CSV_PATH = Path("/home/qwh/nndqn/algorithm_comparison_20260316_151550/transmission_efficiency_comparison_chart_data.csv")

if CSV_PATH.is_file():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    x = df["Episode"].to_numpy()
    y1 = df["SACAQM"].to_numpy()
    y2 = df["DeepAAQM"].to_numpy()
    y3 = df["CODEL"].to_numpy()
else:
    # 无文件：示例为「次数」量级
    x = np.linspace(0, 10, 100)
    y1 = 120 + 15 * np.sin(x)
    y2 = 110 + 12 * np.sin(x + 0.5)
    y3 = 130 + 10 * np.sin(x + 1.0)

y1 = transform_y_for_plot(y1, Y_DISPLAY_MODE, ENERGY_PER_FORWARD_MJ)
y2 = transform_y_for_plot(y2, Y_DISPLAY_MODE, ENERGY_PER_FORWARD_MJ)
y3 = transform_y_for_plot(y3, Y_DISPLAY_MODE, ENERGY_PER_FORWARD_MJ)

# 计算 SAC 相比 DQN（此处用 DeepAAQM 代表）的平均能量减少量
# 注意：此处统计的是 transform_y_for_plot 之后的 y（即与当前 Y_DISPLAY_MODE 一致）
sac_mean = float(np.mean(y1))
dqn_mean = float(np.mean(y2))
less_mean = (dqn_mean - sac_mean) / dqn_mean  # SAC 更省能量 => less_mean > 0

if Y_DISPLAY_MODE == "energy_mj":
    unit = "mJ"
    label = "Energy Consumption per Packet"
else:
    unit = "value"
    label = "Value"

print(
    f"[{label}] 平均 SAC 比 DQN 少了 {less_mean:.6f} {unit} "
    f"(SAC={sac_mean:.6f}, DQN={dqn_mean:.6f}, n={len(x)})"
)

# 绘制折线图
plt.plot(x, y1, label="SACAQM")
plt.plot(x, y2, label="DeepAAQM")
plt.plot(x, y3, label="CoDel")
plt.xlabel('Episode')
if Y_DISPLAY_MODE == "as_csv":
    plt.ylabel('Count')
elif Y_DISPLAY_MODE == "energy_mj":
    plt.ylabel(r'Energy Consumption per Packet (mJ)')
else:
    plt.ylabel('Value')
if Y_LIM is not None:
    plt.ylim(Y_LIM)
else:
    y_stack = np.concatenate([np.asarray(y1), np.asarray(y2), np.asarray(y3)])
    y_stack = y_stack[np.isfinite(y_stack)]
    if y_stack.size:
        lo, hi = float(np.min(y_stack)), float(np.max(y_stack))
        span = hi - lo
        pad = Y_PAD_FRAC * span if span > 0 else 0.02
        plt.ylim(lo - pad, hi + pad)
plt.legend()


# 显示图表

plt.savefig('transmission_efficiency_comparison_4.png')