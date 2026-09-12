import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from dqn_robustness_test import simulate_dqn_robustness
from sac_robustness_test import simulate_sac_robustness
from codel_robustness_test import simulate_codel_robustness
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee', 'no-latex'])

def _format_traffic_pattern_label(traffic_pattern: str) -> str:
    if traffic_pattern is None:
        return "unknown"
    s = str(traffic_pattern).strip()
    if not s:
        return "unknown"
    # 常见模式做一下更友好的展示（不影响你传入的原值）
    pretty = {
        "dynamic": "dynamic",
        "constant": "constant",
        "burst": "burst",
        "mixed": "mixed",
        "extreme_fluctuation_1": "extreme fluctuation 1",
        "extreme_fluctuation_2": "extreme fluctuation 2",
    }
    return pretty.get(s, s)

def compare_algorithms_priority3(
    dqn_model_path=None,
    sac_model_path=None,
    num_episodes=20,
    max_steps_per_episode=2000,
    traffic_pattern="dynamic",
    results_dir=None
):
    """
    对比DQN、SAC和CODEL三种算法的优先级3成功率
    
    参数:
        dqn_model_path: DQN模型路径
        sac_model_path: SAC模型路径
        num_episodes: 每个算法运行的episode数量
        max_steps_per_episode: 每个episode的最大步数
        traffic_pattern: 流量模式
        results_dir: 结果保存目录
    """
    print("="*60)
    print("算法对比：DQN vs SAC vs CODEL")
    print(f"对比指标：优先级3成功率")
    print(f"流量模式：{traffic_pattern}")
    print(f"Episode数量：{num_episodes}")
    print("="*60)
    
    # 创建结果目录
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"algorithm_comparison_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n结果将保存到: {results_dir}\n")
    
    results = {
        "dqn": {"priority3_success_rates": [], "episode_stats": []},
        "sac": {"priority3_success_rates": [], "episode_stats": []},
        "codel": {"priority3_success_rates": [], "episode_stats": []}
    }
    
    # 测试DQN算法
    if dqn_model_path and os.path.exists(dqn_model_path):
        print("\n" + "-"*60)
        print("测试DQN算法...")
        print("-"*60)
        try:
            episode_df, step_df, success_rates = simulate_dqn_robustness(
                model_path=dqn_model_path,
                num_episodes=num_episodes,
                max_steps_per_episode=max_steps_per_episode,
                traffic_pattern=traffic_pattern
            )
            if episode_df is not None:
                # DQN的priority_success_rate是长度为3的列表，索引2对应优先级3
                priority3_rates = episode_df['priority_success_rate'].apply(lambda x: x[2] if len(x) > 2 else 0.0)
                results["dqn"]["priority3_success_rates"] = priority3_rates.tolist()
                results["dqn"]["episode_stats"] = episode_df.to_dict('records')
                avg_priority3 = priority3_rates.mean()
                print(f"\nDQN算法 - 优先级3平均成功率: {avg_priority3:.3f}")
        except Exception as e:
            print(f"DQN测试失败: {e}")
    else:
        print(f"\n跳过DQN测试（模型路径不存在: {dqn_model_path}）")
    
    # 测试SAC算法
    if sac_model_path and os.path.exists(sac_model_path):
        print("\n" + "-"*60)
        print("测试SAC算法...")
        print("-"*60)
        try:
            episode_df, step_df, success_rates = simulate_sac_robustness(
                model_path=sac_model_path,
                num_episodes=num_episodes,
                max_steps_per_episode=max_steps_per_episode,
                traffic_pattern=traffic_pattern
            )
            if episode_df is not None:
                # SAC的priority_success_rate是长度为3的列表，索引2对应优先级3
                priority3_rates = episode_df['priority_success_rate'].apply(lambda x: x[2] if len(x) > 2 else 0.0)
                results["sac"]["priority3_success_rates"] = priority3_rates.tolist()
                results["sac"]["episode_stats"] = episode_df.to_dict('records')
                avg_priority3 = priority3_rates.mean()
                print(f"\nSAC算法 - 优先级3平均成功率: {avg_priority3:.3f}")
        except Exception as e:
            print(f"SAC测试失败: {e}")
    else:
        print(f"\n跳过SAC测试（模型路径不存在: {sac_model_path}）")
    
    # 测试CODEL算法
    print("\n" + "-"*60)
    print("测试CODEL算法...")
    print("-"*60)
    try:
        episode_df, step_df, success_rates = simulate_codel_robustness(
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            traffic_pattern=traffic_pattern
        )
        if episode_df is not None and not episode_df.empty:
            # CODEL的priority_success_rate是长度为3的列表，索引2对应优先级3
            if 'priority_success_rate' in episode_df.columns:
                priority3_rates = episode_df['priority_success_rate'].apply(lambda x: x[2] if isinstance(x, list) and len(x) > 2 else 0.0)
                results["codel"]["priority3_success_rates"] = priority3_rates.tolist()
                results["codel"]["episode_stats"] = episode_df.to_dict('records')
                avg_priority3 = priority3_rates.mean()
                print(f"\nCODEL算法 - 优先级3平均成功率: {avg_priority3:.3f}")
                print(f"CODEL数据收集成功，共 {len(priority3_rates)} 个episode")
            else:
                print("警告: episode_df中没有'priority_success_rate'列")
        else:
            print("警告: CODEL测试返回的episode_df为空或None")
    except Exception as e:
        print(f"CODEL测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成对比结果
    print("\n" + "="*60)
    print("对比结果汇总")
    print("="*60)
    
    comparison_data = []
    
    if results["dqn"]["priority3_success_rates"]:
        dqn_avg = np.mean(results["dqn"]["priority3_success_rates"])
        dqn_std = np.std(results["dqn"]["priority3_success_rates"])
        comparison_data.append({
            "算法": "DQN",
            "平均成功率": dqn_avg,
            "标准差": dqn_std,
            "最小值": np.min(results["dqn"]["priority3_success_rates"]),
            "最大值": np.max(results["dqn"]["priority3_success_rates"])
        })
        print(f"\nDQN:")
        print(f"  平均成功率: {dqn_avg:.3f} ± {dqn_std:.3f}")
        print(f"  范围: [{np.min(results['dqn']['priority3_success_rates']):.3f}, {np.max(results['dqn']['priority3_success_rates']):.3f}]")
    
    if results["sac"]["priority3_success_rates"]:
        sac_avg = np.mean(results["sac"]["priority3_success_rates"])
        sac_std = np.std(results["sac"]["priority3_success_rates"])
        comparison_data.append({
            "算法": "SAC",
            "平均成功率": sac_avg,
            "标准差": sac_std,
            "最小值": np.min(results["sac"]["priority3_success_rates"]),
            "最大值": np.max(results["sac"]["priority3_success_rates"])
        })
        print(f"\nSAC:")
        print(f"  平均成功率: {sac_avg:.3f} ± {sac_std:.3f}")
        print(f"  范围: [{np.min(results['sac']['priority3_success_rates']):.3f}, {np.max(results['sac']['priority3_success_rates']):.3f}]")
    
    if results["codel"]["priority3_success_rates"]:
        codel_avg = np.mean(results["codel"]["priority3_success_rates"])
        codel_std = np.std(results["codel"]["priority3_success_rates"])
        comparison_data.append({
            "算法": "CODEL",
            "平均成功率": codel_avg,
            "标准差": codel_std,
            "最小值": np.min(results["codel"]["priority3_success_rates"]),
            "最大值": np.max(results["codel"]["priority3_success_rates"])
        })
        print(f"\nCODEL:")
        print(f"  平均成功率: {codel_avg:.3f} ± {codel_std:.3f}")
        print(f"  范围: [{np.min(results['codel']['priority3_success_rates']):.3f}, {np.max(results['codel']['priority3_success_rates']):.3f}]")
    
    # 保存对比结果
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(os.path.join(results_dir, "priority3_comparison.csv"), index=False, encoding='utf-8-sig')
        print(f"\n对比结果已保存到: {os.path.join(results_dir, 'priority3_comparison.csv')}")
    
    # 绘制对比图表
    plot_comparison(results, results_dir, traffic_pattern)
    
    # 绘制转发次数对比图
    plot_transmission_comparison(results, results_dir, traffic_pattern)
    
    # 绘制优先级成功率对比图
    plot_priority_success_rate(results, results_dir, traffic_pattern)
    
    plot_trammission_num_comparison(results, results_dir, traffic_pattern)
    # 保存详细结果
    save_detailed_results(results, results_dir)
    
    return results

def plot_comparison(results, save_dir, traffic_pattern):
    """绘制对比图表"""
    # 准备数据
    algorithms = []
    avg_rates = []
    std_rates = []
    
    # 调试信息
    print("\n绘图数据检查:")
    print(f"  DQN数据: {len(results['dqn']['priority3_success_rates'])} 个episode")
    print(f"  SAC数据: {len(results['sac']['priority3_success_rates'])} 个episode")
    print(f"  CODEL数据: {len(results['codel']['priority3_success_rates'])} 个episode")
    
    if results["dqn"]["priority3_success_rates"]:
        algorithms.append("DQN")
        avg_rates.append(np.mean(results["dqn"]["priority3_success_rates"]))
        std_rates.append(np.std(results["dqn"]["priority3_success_rates"]))
    
    if results["sac"]["priority3_success_rates"]:
        algorithms.append("SAC")
        avg_rates.append(np.mean(results["sac"]["priority3_success_rates"]))
        std_rates.append(np.std(results["sac"]["priority3_success_rates"]))
    
    if results["codel"]["priority3_success_rates"]:
        algorithms.append("CODEL")
        avg_rates.append(np.mean(results["codel"]["priority3_success_rates"]))
        std_rates.append(np.std(results["codel"]["priority3_success_rates"]))
    else:
        print("  警告: CODEL数据为空，无法添加到图表")
    
    if not algorithms:
        print("没有可用的数据进行绘图")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 柱状图对比
    bars = ax.bar(algorithms, avg_rates, yerr=std_rates, capsize=8, 
                   color=['#3498db', '#e74c3c', '#2ecc71'][:len(algorithms)],
                   alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.set_ylabel('priority3 success rate', fontsize=14, fontweight='bold')
    tp = _format_traffic_pattern_label(traffic_pattern)
    ax.set_title(f'Algorithm Comparison: Priority 3 Success Rate\nTraffic pattern: {tp}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # 在柱状图上添加数值标签
    for bar, avg, std in zip(bars, avg_rates, std_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.03,
                f'{avg:.3f}\n±{std:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, "priority3_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图表已保存到: {save_path}")
    plt.close()
    # 保存绘图数据 CSV（与图一致）
    chart_df = pd.DataFrame({
        "算法": algorithms,
        "平均成功率": avg_rates,
        "标准差": std_rates
    })
    csv_path = os.path.join(save_dir, "priority3_comparison_chart_data.csv")
    chart_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"优先级3对比图数据已保存到: {csv_path}")

def plot_transmission_comparison(results, save_dir, traffic_pattern):
    """绘制转发效率对比图（转发次数/转发包数）"""
    # 准备数据
    dqn_efficiency = []
    sac_efficiency = []
    codel_efficiency = []
    
    # 提取DQN的转发效率
    if results["dqn"]["episode_stats"]:
        for stat in results["dqn"]["episode_stats"]:
            # transmit_num = env.transmission_count（执行传输动作的次数）
            transmission_count = stat.get('transmit_num', 0)  # 转发次数（执行传输动作的次数）
            # 计算总传输包数（从priority_transmitted中求和）
            priority_transmitted = stat.get('priority_transmitted', [0, 0, 0, 0])
            packets_transmitted = sum(priority_transmitted)  # 总传输包数
            if transmission_count > 0:
                efficiency = transmission_count / packets_transmitted
            else:
                efficiency = 0.0
            dqn_efficiency.append(efficiency)
    
    # 提取SAC的转发效率
    if results["sac"]["episode_stats"]:
        for stat in results["sac"]["episode_stats"]:
            # transmit_num = env.transmission_count（执行传输动作的次数）
            transmission_count = stat.get('transmit_num', 0)  # 转发次数（执行传输动作的次数）
            # 计算总传输包数（从priority_transmitted中求和）
            priority_transmitted = stat.get('priority_transmitted', [0, 0, 0, 0])
            packets_transmitted = sum(priority_transmitted)  # 总传输包数
            if transmission_count > 0:
                efficiency = transmission_count / packets_transmitted
            else:
                efficiency = 0.0
            sac_efficiency.append(efficiency)
    
    # 提取CODEL的转发效率
    if results["codel"]["episode_stats"]:
        for stat in results["codel"]["episode_stats"]:
            packets_transmitted = stat.get('transmit_num', 0)  # 传输的包数
            # transmission_count = 执行传输动作的次数（有传输的步数，每次step_codel算一次动作）
            transmission_count = stat.get('transmission_count', 0)  # 转发次数（执行传输动作的次数）
            if transmission_count > 0:

                efficiency = transmission_count / packets_transmitted
            else:
                efficiency = 0.0
            codel_efficiency.append(efficiency)
    
    if not dqn_efficiency and not sac_efficiency and not codel_efficiency:
        print("没有可用的转发效率数据进行绘图")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 确定最大轮数
    max_episodes = max(
        len(dqn_efficiency) if dqn_efficiency else 0,
        len(sac_efficiency) if sac_efficiency else 0,
        len(codel_efficiency) if codel_efficiency else 0
    )
    
    if max_episodes == 0:
        print("没有可用的数据进行绘图")
        return
    
    episodes = list(range(1, max_episodes + 1))
    
    sum_dqn_efficiency = 0
    sum_sac_efficiency = 0
    sum_codel_efficiency = 0

    for i in range(len(dqn_efficiency)):
        sum_dqn_efficiency += dqn_efficiency[i]
    for i in range(len(sac_efficiency)):
        sum_sac_efficiency += sac_efficiency[i]
    for i in range(len(codel_efficiency)):
        sum_codel_efficiency += codel_efficiency[i]
    print(f"average_dqn_efficiency: {sum_dqn_efficiency / len(dqn_efficiency):.3f}")
    print(f"average_sac_efficiency: {sum_sac_efficiency / len(sac_efficiency):.3f}")
    print(f"average_codel_efficiency: {sum_codel_efficiency / len(codel_efficiency):.3f}")
    
    # 绘制折线图
    if dqn_efficiency:
        ax.plot(episodes[:len(dqn_efficiency)], dqn_efficiency, 
                label='DeepAAQM', color='#3498db', linewidth=2, marker='o', markersize=3, alpha=0.7)
    
    if sac_efficiency:
        ax.plot(episodes[:len(sac_efficiency)], sac_efficiency, 
                label='SACAQM', color='#e74c3c', linewidth=2, marker='s', markersize=3, alpha=0.7)
    
    if codel_efficiency:
        ax.plot(episodes[:len(codel_efficiency)], codel_efficiency, 
                label='CODEL', color='#2ecc71', linewidth=2, marker='^', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Transmission Efficiency', fontsize=14, fontweight='bold')
    tp = _format_traffic_pattern_label(traffic_pattern)
    ax.set_title(f'Transmission Efficiency per Episode\nTraffic pattern: {tp}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, "transmission_efficiency_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"转发效率对比图已保存到: {save_path}")
    plt.close()
    # 保存绘图数据 CSV（与图一致：每 episode 各算法的效率）
    csv_rows = []
    for i in range(max_episodes):
        row = {"Episode": i + 1}
        if i < len(dqn_efficiency):
            row["DeepAAQM"] = dqn_efficiency[i]
        if i < len(sac_efficiency):
            row["SACAQM"] = sac_efficiency[i]
        if i < len(codel_efficiency):
            row["CODEL"] = codel_efficiency[i]
        csv_rows.append(row)
    eff_df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(save_dir, "transmission_efficiency_comparison_chart_data.csv")
    eff_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"转发效率对比图数据已保存到: {csv_path}")

def plot_priority_success_rate(results, save_dir, traffic_pattern):
    """绘制每个算法在不同优先级下的成功率柱状图"""
    algorithms = ["dqn", "sac", "codel"]
    algorithm_labels = ["DeepAAQM", "SACAQM", "CODEL"]
    priority_labels = ["P1", "P2", "P3"]
    
    # 收集平均成功率
    avg_rates = {algo: [0.0, 0.0, 0.0] for algo in algorithms}
    available = False
    
    for algo in algorithms:
        stats = results[algo]["episode_stats"]
        if not stats:
            continue
        priority_sums = np.zeros(3)
        count = 0
        for stat in stats:
            priority_success = stat.get('priority_success_rate', [0.0, 0.0, 0.0])
            if isinstance(priority_success, (list, tuple)) and len(priority_success) >= 3:
                priority_sums += np.array(priority_success[:3])
                count += 1
        if count > 0:
            avg_rates[algo] = (priority_sums / count).tolist()
            available = True
    
    if not available:
        print("没有可用的优先级成功率数据进行绘图")
        return
    
    # 绘制分组柱状图
    x = np.arange(len(algorithms))
    width = 0.2
    colors = ['#3498db', '#e67e22', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, priority_label in enumerate(priority_labels):
        values = [avg_rates[algo][idx] for algo in algorithms]
        print(f"values: {values}")
        ax.bar(x + (idx - 1) * width, values, width, label=f"{priority_label}", color=colors[idx], alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(algorithm_labels, fontsize=13, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=14, fontweight='bold')
    tp = _format_traffic_pattern_label(traffic_pattern)
    ax.set_title(f'Priority Success Rate by Algorithm\nTraffic pattern: {tp}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Priority', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "priority_success_rate_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"优先级成功率对比图已保存到: {save_path}")
    plt.close()
    # 保存绘图数据 CSV（与图一致：算法 × P1/P2/P3）
    priority_csv_rows = []
    for algo, label in zip(algorithms, algorithm_labels):
        priority_csv_rows.append({
            "算法": label,
            "P1": avg_rates[algo][0],
            "P2": avg_rates[algo][1],
            "P3": avg_rates[algo][2]
        })
    priority_df = pd.DataFrame(priority_csv_rows)
    csv_path = os.path.join(save_dir, "priority_success_rate_comparison_chart_data.csv")
    priority_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"优先级成功率对比图数据已保存到: {csv_path}")


def plot_trammission_num_comparison(results, save_dir, traffic_pattern):
    # 准备数据
    dqn_transmission_num = []
    sac_transmission_num = []
    codel_transmission_num = []

    if results["dqn"]["episode_stats"]:
        for stat in results["dqn"]["episode_stats"]:
            # transmit_num = env.transmission_count（执行传输动作的次数）
            transmission_count = stat.get('transmit_num', 0)  # 转发次数（执行传输动作的次数）
            dqn_transmission_num.append(transmission_count)
    
    # 提取SAC的转发效率
    if results["sac"]["episode_stats"]:
        for stat in results["sac"]["episode_stats"]:
            # transmit_num = env.transmission_count（执行传输动作的次数）
            transmission_count = stat.get('transmit_num', 0)  # 转发次数（执行传输动作的次数）
            sac_transmission_num.append(transmission_count)
    
    # 提取CODEL的转发效率
    if results["codel"]["episode_stats"]:
        for stat in results["codel"]["episode_stats"]:
            transmission_count = stat.get('transmission_count', 0)  # 转发次数（执行传输动作的次数）
            codel_transmission_num.append(transmission_count)
    
    if not dqn_transmission_num and not sac_transmission_num and not codel_transmission_num:
        print("没有可用的转发效率数据进行绘图")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 确定最大轮数
    max_episodes = max(
        len(dqn_transmission_num) if dqn_transmission_num else 0,
        len(sac_transmission_num) if sac_transmission_num else 0,
        len(codel_transmission_num) if codel_transmission_num else 0
    )
    
    if max_episodes == 0:
        print("没有可用的数据进行绘图")
        return
    
    episodes = list(range(1, max_episodes + 1))
    
    # 绘制折线图
    if dqn_transmission_num:
        ax.plot(episodes[:len(dqn_transmission_num)], dqn_transmission_num, 
                label='DQN', color='#3498db', linewidth=2, marker='o', markersize=3, alpha=0.7)
    
    if sac_transmission_num:
        ax.plot(episodes[:len(sac_transmission_num)], sac_transmission_num, 
                label='SAC', color='#e74c3c', linewidth=2, marker='s', markersize=3, alpha=0.7)
    
    if codel_transmission_num:
        ax.plot(episodes[:len(codel_transmission_num)], codel_transmission_num, 
                label='CODEL', color='#2ecc71', linewidth=2, marker='^', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Transmission Number', fontsize=14, fontweight='bold')
    tp = _format_traffic_pattern_label(traffic_pattern)
    ax.set_title(f'Algorithm Comparison: Transmission Number per Episode\nTraffic pattern: {tp}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, "transmission_number_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"转发次数对比图已保存到: {save_path}")
    plt.close()
    # 保存绘图数据 CSV（与图一致：每 episode 各算法的转发次数）
    csv_rows = []
    for i in range(max_episodes):
        row = {"Episode": i + 1}
        if i < len(dqn_transmission_num):
            row["DQN"] = dqn_transmission_num[i]
        if i < len(sac_transmission_num):
            row["SAC"] = sac_transmission_num[i]
        if i < len(codel_transmission_num):
            row["CODEL"] = codel_transmission_num[i]
        csv_rows.append(row)
    trans_df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(save_dir, "transmission_number_comparison_chart_data.csv")
    trans_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"转发次数对比图数据已保存到: {csv_path}")

def save_detailed_results(results, save_dir):
    """保存详细结果"""
    # 保存DQN结果
    if results["dqn"]["episode_stats"]:
        dqn_df = pd.DataFrame(results["dqn"]["episode_stats"])
        dqn_df.to_csv(os.path.join(save_dir, "dqn_episode_stats.csv"), index=False)
    
    # 保存SAC结果
    if results["sac"]["episode_stats"]:
        sac_df = pd.DataFrame(results["sac"]["episode_stats"])
        sac_df.to_csv(os.path.join(save_dir, "sac_episode_stats.csv"), index=False)
    
    # 保存CODEL结果
    if results["codel"]["episode_stats"]:
        codel_df = pd.DataFrame(results["codel"]["episode_stats"])
        codel_df.to_csv(os.path.join(save_dir, "codel_episode_stats.csv"), index=False)

if __name__ == "__main__":
    # 设置参数
    num_episodes = 100  # 修改为100轮
    max_steps_per_episode = 2000
    traffic_pattern = "Periodic Periodic Burst Traffic"#"dynamic" "Pattern Shift" "Extreme Congestion" "Periodic Brust Traffic"

    # 模型路径（请根据实际情况修改）
    #dqn_model_path = "/home/qwh/nndqn/dynamic_uav_models/dynamic_20260122_103458/dqn_dynamic_uav_final.zip"
    dqn_model_path = "/home/qwh/nndqn/dynamic_uav_models/dynamic_20260304_112107/best_model/best_model.zip" #原模型
    
    #sac_model_path = "/home/qwh/nndqn/SAC_dynamic_uav_models/mixed_20260310_162949/best_model/best_model.zip" #新模型
    #sac_model_path = "/home/qwh/nndqn/SAC_dynamic_uav_models/mixed_20260311_084558/best_model/best_model.zip" #效果最好模型
    #sac_model_path = "/home/qwh/nndqn/SAC_dynamic_uav_models/mixed_20260312_162440/best_model/best_model.zip" #最新模型
    sac_model_path = "/home/qwh/nndqn/SAC_dynamic_uav_models/mixed_20260313_090408/best_model/best_model.zip" #最新模型
    
    # 检查模型文件是否存在
    if not os.path.exists(dqn_model_path):
        print(f"警告: DQN模型文件不存在: {dqn_model_path}")
        dqn_model_path = None
    
    if not os.path.exists(sac_model_path):
        print(f"警告: SAC模型文件不存在: {sac_model_path}")
        sac_model_path = None
    
    # 运行对比
    results = compare_algorithms_priority3(
        dqn_model_path=dqn_model_path,
        sac_model_path=sac_model_path,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        traffic_pattern=traffic_pattern
    )
    
    print("\n对比完成！")


