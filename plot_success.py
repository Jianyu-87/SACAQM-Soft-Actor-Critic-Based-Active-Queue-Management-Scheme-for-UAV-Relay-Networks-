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
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 4

# CSV_PATH = Path("/home/qwh/nndqn/algorithm_comparison_20260316_144737/priority_success_rate_comparison_chart_data.csv")
# CSV_PATH = Path("/home/qwh/nndqn/algorithm_comparison_20260316_150509/priority_success_rate_comparison_chart_data.csv")
# CSV_PATH = Path("/home/qwh/nndqn/algorithm_comparison_20260316_151028/priority_success_rate_comparison_chart_data.csv")
CSV_PATH = Path("/home/qwh/nndqn/algorithm_comparison_20260316_151550/priority_success_rate_comparison_chart_data.csv")
# 分组柱宽与间距：每组内 P1/P2/P3 三根并排
BAR_WIDTH = 0.22

if CSV_PATH.is_file():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    labels = df["算法"].to_numpy()
    y1 = df["P1"].to_numpy(dtype=float)
    y2 = df["P2"].to_numpy(dtype=float)
    y3 = df["P3"].to_numpy(dtype=float)
else:
    labels = np.array(["A", "B", "C"])
    y1 = np.array([0.9, 0.85, 0.7])
    y2 = np.array([0.8, 0.9, 0.75])
    y3 = np.array([0.7, 0.95, 0.6])

n = len(labels)
x = np.arange(n)
w = BAR_WIDTH

fig, ax = plt.subplots()
ax.bar(x - w, y1, w, label="P1-low")
ax.bar(x, y2, w, label="P2-medium")
ax.bar(x + w, y3, w, label="P3-high")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Packet Success Rate")
ax.legend()
fig.tight_layout()
plt.savefig("priority_success_rate_bar4.png", dpi=300)
