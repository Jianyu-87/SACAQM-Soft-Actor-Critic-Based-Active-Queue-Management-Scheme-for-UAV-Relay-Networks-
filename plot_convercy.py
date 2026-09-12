from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import os

# SciencePlots 的 science/ieee 需 pip install SciencePlots 且先 import 才会注册到 matplotlib
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee", "bright"])
except (ImportError, OSError):
    plt.style.use("ggplot")

# 自定义样式（多曲线时用较细线宽 + 标记区分，避免粗线叠在一起更难读）
plt.rcParams['lines.linewidth'] = 1.8
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_uav_tensorboard")

# (run 标识, rho 值, events 文件路径)
RUNS = [
    ("mixed_load", None, os.path.join(_BASE, "mixed_20260313_090408", "SAC_1", "events.out.tfevents.1773363852.qf-NF5280M6.613060.0")),
    ("low_load", 0.4, os.path.join(_BASE, "low_load_20260322_103123", "SAC_1", "events.out.tfevents.1774146687.qf-NF5280M6.4057795.0")),
    ("medium_load", 0.7, os.path.join(_BASE, "medium_load_20260322_103146", "SAC_1", "events.out.tfevents.1774146710.qf-NF5280M6.4058779.0")),
    ("high_load", 1.0, os.path.join(_BASE, "high_load_20260322_103206", "SAC_1", "events.out.tfevents.1774146730.qf-NF5280M6.4059996.0")),
]

# 与 RUNS 一一对应：标记形状 + 线型（印刷黑白时也能区分）
LINE_SPECS = [
    {"marker": "o", "linestyle": "-"},    # 圆 + 实线
    {"marker": "s", "linestyle": "-"},   # 方 + 虚线
    {"marker": "^", "linestyle": "-"},    # 三角 + 实线
    {"marker": "D", "linestyle": "-"},   # 菱形 + 点划
]
if len(LINE_SPECS) != len(RUNS):
    raise ValueError("RUNS 与 LINE_SPECS 条数须一致")

tag = "rollout/ep_rew_mean"

rows = []
for run_name, _, log_path in RUNS:
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"找不到日志文件: {log_path}")
    acc = event_accumulator.EventAccumulator(log_path)
    acc.Reload()
    scalars = acc.Tags().get("scalars", [])
    if tag not in scalars:
        raise KeyError(f"{run_name}: 无标量标签 {tag!r}，当前有: {scalars}")
    for e in acc.Scalars(tag):
        rows.append({"run": run_name, "step": e.step, tag: e.value})

df = pd.DataFrame(rows)
print(df.head())
print("各 run 样本数:\n", df.groupby("run").size())

# 合并保存（长表：run + step + 标量列）
out_csv = f"sac_four_loads_{tag.replace('/', '_')}.csv"
df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"已保存 DataFrame: {out_csv}")

plt.figure(figsize=(10, 6))
for (run_name, rho, _), spec in zip(RUNS, LINE_SPECS):
    sub = df[df["run"] == run_name].sort_values("step")
    n = len(sub)
    # 每隔若干点打一个标记，既看得出形状又不会太密
    markevery = max(1, n // 10) if n > 10 else 1
    legend_label = "mixed_load" if rho is None else rf"$\rho$ = {rho}"
    plt.plot(
        sub["step"],
        sub[tag],
        label=legend_label,
        linewidth=1.8,
        marker=spec["marker"],
        linestyle=spec["linestyle"],
        markersize=6,
        markeredgewidth=0.8,
        markevery=markevery,
    )

plt.xlabel("Step")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
out_png = "sac_four_loads_total_reward_curve.png"
plt.savefig(out_png)
print(f"已保存图片: {out_png}")
