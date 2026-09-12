import os
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from dynamic_traffic_uav import DynamicTrafficUAVEnv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def simulate_sac_robustness(model_path, num_episodes=20, max_steps_per_episode=2000, traffic_pattern="extreme_fluctuation_1"):
    """
    使用训练好的SAC模型进行鲁棒性测试
    测试在极端波动流量模式下的性能
    
    参数:
        model_path: 训练好的模型路径
        num_episodes: 模拟的episode数量
        max_steps_per_episode: 每个episode的最大步数
        traffic_pattern: 流量模式
    """
    print(f"开始SAC鲁棒性测试 - 流量模式: {traffic_pattern}")
    print(f"模拟episodes: {num_episodes}, 每episode最大步数: {max_steps_per_episode}")
    print("="*60)
    
    # 加载训练好的模型
    try:
        model = SAC.load(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 创建环境 - 使用极端波动流量模式
    # 注意：max_queue必须与训练时一致（默认150），否则观察空间不匹配
    env = DynamicTrafficUAVEnv(
        traffic_pattern=traffic_pattern,
        max_steps=max_steps_per_episode,
        max_queue=150,  # 修改为150，与训练时一致
        max_transmissions=None
    )
    
    # 检查观察空间维度
    print(f"观察空间维度: {env.observation_space.shape}")
    print(f"动作空间维度: {env.action_space.shape}")
    
    # 存储所有episode的统计数据
    all_episode_stats = []
    
    # 存储每个step的详细数据用于分析
    step_by_step_data = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # 重置环境
        obs, _ = env.reset()
        episode_reward = 0
        episode_stats = {
            'episode': episode + 1,
            'priority_transmitted': [0, 0, 0, 0],
            'priority_generated': [0, 0, 0, 0],
            'priority_success_rate': [0, 0, 0],
            'transmit_num': 0,
            'energy_consumed': 0,
            'total_reward': 0,
            'avg_delay': 0.0,
            'priority_avg_delay': [0.0, 0.0, 0.0],
            'packets_generated': 0,
            'packets_expired': 0,
            'buffer_overflow_count': 0,
            'max_buffer_usage': 0,
            'min_buffer_usage': float('inf'),
            'avg_buffer_usage': 0.0,
            'buffer_usage_history': [],
            'traffic_load_history': [],
            'reward_history': [],
            'delay_satisfied_count': 0,
            'high_traffic_performance': {'transmitted': 0, 'expired': 0},
            'low_traffic_performance': {'transmitted': 0, 'expired': 0},
        }
        
        step_count = 0
        terminated = False
        truncated = False
        total_buffer_usage = 0
        
        while not terminated and not truncated and step_count < max_steps_per_episode:
            # 获取当前流量负载
            current_traffic_load = env._get_current_traffic_load()
            
            # 使用训练好的模型进行预测
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # 记录每个step的详细数据
            buffer_size = info['buffer_size']
            total_buffer_usage += buffer_size
            episode_stats['max_buffer_usage'] = max(episode_stats['max_buffer_usage'], buffer_size)
            episode_stats['min_buffer_usage'] = min(episode_stats['min_buffer_usage'], buffer_size)
            episode_stats['buffer_usage_history'].append(buffer_size)
            episode_stats['traffic_load_history'].append(current_traffic_load)
            episode_stats['reward_history'].append(reward)
            
            # 记录step数据
            step_data = {
                'episode': episode + 1,
                'step': step_count,
                'traffic_load': current_traffic_load,
                'buffer_size': buffer_size,
                'reward': reward,
                'action': action.tolist() if isinstance(action, np.ndarray) else action,
                'packets_transmitted': info.get('packets_transmitted_this_step', 0),
                'packets_expired': info.get('packets_expired_this_step', 0),
                'energy_consumed': info.get('energy_consumed_this_step', 0)
            }
            step_by_step_data.append(step_data)
            
            # 根据流量负载分类统计性能
            # 注意：packets_generated在step中无法直接获取，需要在episode结束时从环境统计中获取
            if current_traffic_load > 1.5:  # 高流量
                episode_stats['high_traffic_performance']['transmitted'] += info.get('packets_transmitted_this_step', 0)
                episode_stats['high_traffic_performance']['expired'] += info.get('packets_expired_this_step', 0)
            else:  # 低流量
                episode_stats['low_traffic_performance']['transmitted'] += info.get('packets_transmitted_this_step', 0)
                episode_stats['low_traffic_performance']['expired'] += info.get('packets_expired_this_step', 0)
            
            # 每100步打印一次进度
            if step_count % 100 == 0:
                print(f"  步骤 {step_count}: 奖励={reward:.2f}, 缓冲区={buffer_size}, "
                      f"流量负载={current_traffic_load:.2f}, 动作={action}")
        
        # 记录episode结束时的统计信息
        episode_stats.update({
            'priority_transmitted': env.priority_transmitted.copy(),
            'priority_generated': env.priority_generated.copy(),
            'transmit_num': env.transmission_count,
            'energy_consumed': env.energy_consumed,
            'total_reward': episode_reward,
            'avg_delay': env.total_delay / max(env.packets_transmitted, 1),
            'packets_generated': env.packets_generated_sum,
            'packets_expired': env.packets_expired,
            'avg_buffer_usage': total_buffer_usage / max(step_count, 1),
            'delay_satisfied_count': env.delay_satisfied_count,
        })
        
        # 计算每个优先级的平均延迟和成功率
        for priority in [1, 2, 3]:
            if env.priority_transmitted[priority] > 0:
                priority_delay = env.priority_delay[priority]
                episode_stats['priority_avg_delay'][priority-1] = priority_delay / env.priority_transmitted[priority]
            else:
                episode_stats['priority_avg_delay'][priority-1] = 0.0
            
            # 计算成功率
            if episode_stats['priority_generated'][priority] > 0:
                success_count = env.delay_satisfied_by_priority[priority]
                total_generated = episode_stats['priority_generated'][priority]
                episode_stats['priority_success_rate'][priority-1] = success_count / total_generated
            else:
                episode_stats['priority_success_rate'][priority-1] = 0.0
        
        # 计算高/低流量下的传输效率（使用传输数/过期数比例）
        # 由于无法精确区分高/低流量下的生成数，使用传输效率作为替代指标
        high_total = episode_stats['high_traffic_performance']['transmitted'] + episode_stats['high_traffic_performance']['expired']
        low_total = episode_stats['low_traffic_performance']['transmitted'] + episode_stats['low_traffic_performance']['expired']
        
        if high_total > 0:
            high_traffic_success_rate = (
                episode_stats['high_traffic_performance']['transmitted'] / high_total
            )
        else:
            high_traffic_success_rate = 0.0
        
        if low_total > 0:
            low_traffic_success_rate = (
                episode_stats['low_traffic_performance']['transmitted'] / low_total
            )
        else:
            low_traffic_success_rate = 0.0
        
        episode_stats['high_traffic_success_rate'] = high_traffic_success_rate
        episode_stats['low_traffic_success_rate'] = low_traffic_success_rate
        
        all_episode_stats.append(episode_stats)
        
        # 打印episode总结
        print(f"Episode {episode + 1} 完成:")
        print(f"  优先级1成功率: {episode_stats['priority_success_rate'][0]:.3f}")
        print(f"  优先级2成功率: {episode_stats['priority_success_rate'][1]:.3f}")
        print(f"  优先级3成功率: {episode_stats['priority_success_rate'][2]:.3f}")
        print(f"  优先级1传输个数: {episode_stats['priority_transmitted'][1]}")
        print(f"  优先级2传输个数: {episode_stats['priority_transmitted'][2]}")
        print(f"  优先级3传输个数: {episode_stats['priority_transmitted'][3]}")
        print(f"  优先级1生成个数: {episode_stats['priority_generated'][1]}")
        print(f"  优先级2生成个数: {episode_stats['priority_generated'][2]}")
        print(f"  优先级3生成个数: {episode_stats['priority_generated'][3]}")
        print(f"  传输次数: {episode_stats['transmit_num']}")
        print(f"  过期包数: {episode_stats['packets_expired']}")
        print(f"  平均缓冲区使用: {episode_stats['avg_buffer_usage']:.2f}")
        print(f"  最大缓冲区使用: {episode_stats['max_buffer_usage']}")
        print(f"  高流量成功率: {high_traffic_success_rate:.3f}")
        print(f"  低流量成功率: {low_traffic_success_rate:.3f}")
        print(f"  总奖励: {episode_reward:.2f}")
    # 计算总体统计
    print("\n" + "="*60)
    print("SAC鲁棒性测试完成 - 极端波动流量模式统计")
    print("="*60)
    
    # 转换为DataFrame便于分析
    episode_df = pd.DataFrame(all_episode_stats)
    step_df = pd.DataFrame(step_by_step_data)
    
    # 计算平均成功率
    avg_success_rates = []
    for i in range(1, 4):
        avg_success_rate = np.mean([stats['priority_success_rate'][i-1] for stats in all_episode_stats])
        avg_success_rates.append(avg_success_rate)
        print(f"优先级 {i} 平均成功率: {avg_success_rate:.3f}")
    
    # 计算总体平均成功率
    overall_avg = np.mean(avg_success_rates)
    print(f"\n总体平均成功率: {overall_avg:.3f}")
    print(f"优先级1生成个数: {np.mean([stats['priority_generated'][1] for stats in all_episode_stats]):.2f}")
    print(f"优先级2生成个数: {np.mean([stats['priority_generated'][2] for stats in all_episode_stats]):.2f}")
    print(f"优先级3生成个数: {np.mean([stats['priority_generated'][3] for stats in all_episode_stats]):.2f}")
    print(f"总体平均传输次数: {np.mean([stats['transmit_num'] for stats in all_episode_stats]):.2f}")
    print(f"总体平均过期包数: {np.mean([stats['packets_expired'] for stats in all_episode_stats]):.2f}")
    print(f"总体平均延迟: {np.mean([stats['avg_delay'] for stats in all_episode_stats]):.2f}")
    print(f"总体平均缓冲区使用: {np.mean([stats['avg_buffer_usage'] for stats in all_episode_stats]):.2f}")
    print(f"总体平均最大缓冲区使用: {np.mean([stats['max_buffer_usage'] for stats in all_episode_stats]):.2f}")
    # 高/低流量性能对比
    avg_high_traffic_success = np.mean([stats['high_traffic_success_rate'] for stats in all_episode_stats])
    avg_low_traffic_success = np.mean([stats['low_traffic_success_rate'] for stats in all_episode_stats])
    print(f"\n高流量平均成功率: {avg_high_traffic_success:.3f}")
    print(f"低流量平均成功率: {avg_low_traffic_success:.3f}")
    print(f"性能差异: {abs(avg_high_traffic_success - avg_low_traffic_success):.3f}")
    
    # 保存统计数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_dir = f"sac_robustness_test_{timestamp}"
    os.makedirs(stats_dir, exist_ok=True)
    
    # 保存CSV文件
    episode_df.to_csv(os.path.join(stats_dir, "episode_stats.csv"), index=False)
    step_df.to_csv(os.path.join(stats_dir, "step_by_step_data.csv"), index=False)
    
    # 生成可视化图表
    #plot_robustness_results(episode_df, step_df, stats_dir)
    
    print(f"\n统计数据已保存到: {stats_dir}")
    
    return episode_df, step_df, avg_success_rates

def plot_robustness_results(episode_df, step_df, save_dir):
    """绘制鲁棒性测试结果的可视化图表"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 优先级成功率对比
    ax1 = plt.subplot(3, 3, 1)
    priority_labels = ['Priority 1', 'Priority 2', 'Priority 3']
    avg_rates = [
        episode_df['priority_success_rate'].apply(lambda x: x[0]).mean(),
        episode_df['priority_success_rate'].apply(lambda x: x[1]).mean(),
        episode_df['priority_success_rate'].apply(lambda x: x[2]).mean()
    ]
    bars = ax1.bar(priority_labels, avg_rates, color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8)
    ax1.set_title('Priority Success Rate')
    ax1.set_ylabel('Success Rate')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    for bar, rate in zip(bars, avg_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 高/低流量性能对比
    ax2 = plt.subplot(3, 3, 2)
    high_low_data = [
        episode_df['high_traffic_success_rate'].mean(),
        episode_df['low_traffic_success_rate'].mean()
    ]
    bars = ax2.bar(['High Traffic', 'Low Traffic'], high_low_data, 
                   color=['#ff6b6b', '#4ecdc4'], alpha=0.8)
    ax2.set_title('Performance: High vs Low Traffic')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    for bar, rate in zip(bars, high_low_data):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 缓冲区使用情况
    ax3 = plt.subplot(3, 3, 3)
    buffer_stats = [
        episode_df['avg_buffer_usage'].mean(),
        episode_df['max_buffer_usage'].mean(),
        episode_df['min_buffer_usage'].mean()
    ]
    bars = ax3.bar(['Avg', 'Max', 'Min'], buffer_stats, 
                   color=['#95a5a6', '#e74c3c', '#2ecc71'], alpha=0.8)
    ax3.set_title('Buffer Usage Statistics')
    ax3.set_ylabel('Buffer Size')
    ax3.grid(True, alpha=0.3)
    for bar, val in zip(bars, buffer_stats):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Episode成功率趋势
    ax4 = plt.subplot(3, 3, 4)
    episode_numbers = episode_df['episode']
    overall_success = episode_df['priority_success_rate'].apply(
        lambda x: np.mean([x[0], x[1], x[2]])
    )
    ax4.plot(episode_numbers, overall_success, marker='o', linewidth=2, markersize=4)
    ax4.set_title('Overall Success Rate Over Episodes')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Success Rate')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=overall_success.mean(), color='r', linestyle='--', 
                label=f'Mean: {overall_success.mean():.3f}')
    ax4.legend()
    
    # 5. 传输次数 vs 过期包数
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(episode_df['transmit_num'], episode_df['packets_expired'], 
                alpha=0.6, s=50, c=episode_df['episode'], cmap='viridis')
    ax5.set_title('Transmissions vs Expired Packets')
    ax5.set_xlabel('Transmission Count')
    ax5.set_ylabel('Expired Packets')
    ax5.grid(True, alpha=0.3)
    
    # 6. 平均延迟
    ax6 = plt.subplot(3, 3, 6)
    priority_delays = [
        episode_df['priority_avg_delay'].apply(lambda x: x[0]).mean(),
        episode_df['priority_avg_delay'].apply(lambda x: x[1]).mean(),
        episode_df['priority_avg_delay'].apply(lambda x: x[2]).mean()
    ]
    bars = ax6.bar(priority_labels, priority_delays, 
                   color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8)
    ax6.set_title('Average Delay by Priority')
    ax6.set_ylabel('Average Delay (steps)')
    ax6.grid(True, alpha=0.3)
    for bar, delay in zip(bars, priority_delays):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{delay:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. 流量负载随时间变化（第一个episode的示例）
    ax7 = plt.subplot(3, 3, 7)
    first_episode_steps = step_df[step_df['episode'] == 1]
    if len(first_episode_steps) > 0:
        steps = first_episode_steps['step']
        traffic_load = first_episode_steps['traffic_load']
        buffer_size = first_episode_steps['buffer_size']
        ax7_twin = ax7.twinx()
        line1 = ax7.plot(steps, traffic_load, 'b-', label='Traffic Load', linewidth=2)
        line2 = ax7_twin.plot(steps, buffer_size, 'r-', label='Buffer Size', linewidth=2, alpha=0.7)
        ax7.set_title('Traffic Load & Buffer Size (Episode 1)')
        ax7.set_xlabel('Step')
        ax7.set_ylabel('Traffic Load', color='b')
        ax7_twin.set_ylabel('Buffer Size', color='r')
        ax7.tick_params(axis='y', labelcolor='b')
        ax7_twin.tick_params(axis='y', labelcolor='r')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax7.legend(lines, labels, loc='upper left')
        ax7.grid(True, alpha=0.3)
    
    # 8. 奖励分布
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(episode_df['total_reward'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
    ax8.set_title('Total Reward Distribution')
    ax8.set_xlabel('Total Reward')
    ax8.set_ylabel('Frequency')
    ax8.axvline(x=episode_df['total_reward'].mean(), color='r', linestyle='--', 
                label=f'Mean: {episode_df["total_reward"].mean():.2f}')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 能量消耗 vs 成功率
    ax9 = plt.subplot(3, 3, 9)
    overall_success_rates = episode_df['priority_success_rate'].apply(
        lambda x: np.mean([x[0], x[1], x[2]])
    )
    ax9.scatter(episode_df['energy_consumed'], overall_success_rates, 
                alpha=0.6, s=50, c=episode_df['episode'], cmap='plasma')
    ax9.set_title('Energy Consumption vs Success Rate')
    ax9.set_xlabel('Energy Consumed')
    ax9.set_ylabel('Overall Success Rate')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, "sac_robustness_test_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: {save_path}")
    
    plt.show()
    
    return save_path

if __name__ == "__main__":
    # 设置参数
    num_episodes = 20
    max_steps_per_episode = 2000
    
    # 查找训练好的模型路径
    # 请根据实际情况修改模型路径
    #model_path = "SAC_dynamic_uav_models/dynamic_20251203_143318/sac_dynamic_uav_final.zip"
    model_path = "/home/qwh/nndqn/SAC_dynamic_uav_models/dynamic_20260104_163756/sac_dynamic_uav_final.zip"
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在: {model_path}")
        print("请先训练模型，或修改模型路径")
        print("\n可用模型路径示例:")
        print("  - SAC_dynamic_uav_models/dynamic_*/sac_dynamic_uav_*.zip")
        print("  - SAC_dynamic_uav_models/constant_*/sac_dynamic_uav_*.zip")
        exit(1)
    
    print("SAC算法鲁棒性测试")
    print("="*60)
    print("测试场景: 极端波动流量模式（小速率0.2 <-> 大速率2.5）")
    print("="*60)
    
    # 运行鲁棒性测试
    episode_df, step_df, success_rates = simulate_sac_robustness(
        model_path=model_path,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        traffic_pattern="extreme_fluctuation_1"
    )
    
    if episode_df is not None:
        print("\n鲁棒性测试完成！")
        print("\n关键性能指标:")
        print(f"  优先级1平均成功率: {success_rates[0]:.3f}")
        print(f"  优先级2平均成功率: {success_rates[1]:.3f}")
        print(f"  优先级3平均成功率: {success_rates[2]:.3f}")
        print(f"  总体平均成功率: {np.mean(success_rates):.3f}")
        print(f"  平均过期包数: {episode_df['packets_expired'].mean():.2f}")
        print(f"  平均缓冲区使用率: {episode_df['avg_buffer_usage'].mean():.2f}")
        print(f"  高流量成功率: {episode_df['high_traffic_success_rate'].mean():.3f}")
        print(f"  低流量成功率: {episode_df['low_traffic_success_rate'].mean():.3f}")
