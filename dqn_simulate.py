import os
import time
import numpy as np
import torch
from stable_baselines3 import DQN
from dqn_dynamic_uav import DynamicTrafficUAVEnv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def simulate_dqn(model_path, traffic_pattern="dynamic", num_episodes=10, max_steps_per_episode=2000):
    """
    使用训练好的DQN模型进行模拟运行，只统计优先级成功率
    
    参数:
        model_path: 训练好的模型路径
        traffic_pattern: 流量模式
        num_episodes: 模拟的episode数量
        max_steps_per_episode: 每个episode的最大步数
    """
    print(f"开始DQN模拟运行 - 流量模式: {traffic_pattern}")
    print(f"模拟episodes: {num_episodes}, 每episode最大步数: {max_steps_per_episode}")
    print("="*60)
    
    # 创建环境
    env = DynamicTrafficUAVEnv(traffic_pattern=traffic_pattern, max_transmissions = None)
    
    # 加载训练好的模型
    try:
        model = DQN.load(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 存储所有episode的统计数据
    all_episode_stats = []
    
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
            'transmission_count': 0,
            'delay_satisfied_count': 0,
            'transmission_efficiency': 0.0,
            'total_transmission_attempts': 0,  # 新增
            'successful_transmissions': 0,      # 新增
            'failed_transmissions': 0,          # 新增
            'success_rate': 0.0,                # 新增
            'failure_rate': 0.0,                # 新增
            'high_priority_transmitted': 0,     # 新增：高优先级包转发统计
            'transmissions_to_complete_target': 0,  # 新增
            'target_completed': False,           # 新增
            'packets_dropped': 0,
            'advancedrop': 0,
            'overtime_transmitted': 0
        }
        
        step_count = 0
        terminated = False
        truncated = False
        
        while not terminated and not truncated and step_count < max_steps_per_episode:
            # 使用训练好的模型进行预测
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # 每100步打印一次进度
            if step_count % 100 == 0:
                print(f"  步骤 {step_count}: 奖励={reward:.2f}, 缓冲区={info['buffer_size']}, 动作={action}")
        
        # 记录episode结束时的统计信息
        episode_stats.update({
            'priority_transmitted': env.priority_transmitted.copy(),
            'priority_generated': env.priority_generated.copy(),
            'transmit_num': env.transmit_num,
            'transmission_count': env.transmission_count,
            'delay_satisfied_count': env.delay_satisfied_count,
            'transmission_efficiency': env.delay_satisfied_count / max(1, env.transmission_count),
            'total_transmission_attempts': env.total_transmission_attempts,  # 新增
            'successful_transmissions': env.successful_transmissions,          # 新增
            'failed_transmissions': env.failed_transmissions,                  # 新增
            'success_rate': env.successful_transmissions / max(1, env.total_transmission_attempts),  # 新增
            'failure_rate': env.failed_transmissions / max(1, env.total_transmission_attempts),       # 新增
            'high_priority_transmitted': env.high_priority_transmitted,       # 新增
            'transmissions_to_complete_target': env.transmissions_to_complete_target,  # 新增
            'target_completed': env.target_completed,                          # 新增
            'packets_dropped': env.packets_dropped,
            'advancedrop': env.advancedrop,
            'overtime_transmitted': env.overtime_transmitted
        })
        
        # 计算优先级成功率
        for i in range(1, 4):
            if episode_stats['priority_generated'][i] > 0:
                # 使用满足延迟要求的包数量计算成功率
                success_count = env.delay_satisfied_by_priority[i]
                total_generated = episode_stats['priority_generated'][i]
                episode_stats['priority_success_rate'][i-1] = success_count / total_generated
            else:
                episode_stats['priority_success_rate'][i-1] = 0
        
        all_episode_stats.append(episode_stats)
        
        # 打印episode总结
        print(f"Episode {episode + 1} 完成:")
        print(f"  优先级1成功率: {episode_stats['priority_success_rate'][0]:.3f}")
        print(f"  优先级2成功率: {episode_stats['priority_success_rate'][1]:.3f}")
        print(f"  优先级3成功率: {episode_stats['priority_success_rate'][2]:.3f}")
        print(f"  转发次数: {episode_stats['transmit_num']}")
        print(f"  丢弃包数: {episode_stats['packets_dropped']}")
        print(f"  提前丢弃包数: {episode_stats['advancedrop']}")
        print(f"  超时传输包数: {episode_stats['overtime_transmitted']}")
        # 打印转发时延统计
        env.print_transmission_delay_stats()
    
    # 计算总体统计
    print("\n" + "="*60)
    print("DQN模拟运行完成 - 优先级成功率统计")
    print("="*60)
    
    # 转换为DataFrame便于分析
    episode_df = pd.DataFrame(all_episode_stats)
    
    # 计算平均成功率
    avg_success_rates = []
    for i in range(1, 4):
        avg_success_rate = np.mean([stats['priority_success_rate'][i-1] for stats in all_episode_stats])
        avg_success_rates.append(avg_success_rate)
        print(f"优先级 {i} 平均成功率: {avg_success_rate:.3f}")
    
    # 计算总体平均成功率
    overall_avg = np.mean(avg_success_rates)
    print(f"\n总体平均成功率: {overall_avg:.3f}")
    print(f"总体平均转发次数: {np.mean([stats['transmit_num'] for stats in all_episode_stats])}")
    print(f"总体平均提前丢弃包数: {np.mean([stats['advancedrop'] for stats in all_episode_stats])}")
    print(f"总体平均超时传输包数: {np.mean([stats['overtime_transmitted'] for stats in all_episode_stats])}")
    # 保存统计数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_dir = f"dqn_simulation_stats_{timestamp}"
    os.makedirs(stats_dir, exist_ok=True)
    
    # 保存CSV文件
    episode_df.to_csv(os.path.join(stats_dir, "episode_stats.csv"), index=False)
    
    # 生成简单的成功率图表
    #plot_success_rates(avg_success_rates, stats_dir, traffic_pattern)
    
    print(f"\n统计数据已保存到: {stats_dir}")
    
    return episode_df, avg_success_rates

def plot_success_rates(success_rates, save_dir, traffic_pattern):
    """绘制优先级成功率图表"""
    plt.figure(figsize=(10, 6))
    
    priority_labels = ['Priority 1', 'Priority 2', 'Priority 3']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    bars = plt.bar(priority_labels, success_rates, color=colors, alpha=0.8, edgecolor='black')
    
    # 在柱子上添加数值标签
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Priority Level')
    plt.ylabel('Success Rate')
    plt.title(f'Priority Packet Success Rate - {traffic_pattern.capitalize()} Traffic')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加平均线
    avg_rate = np.mean(success_rates)
    plt.axhline(y=avg_rate, color='red', linestyle='--', linewidth=2, 
                label=f'Average: {avg_rate:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"priority_success_rates_{traffic_pattern}.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"成功率图表已保存到: {os.path.join(save_dir, f'priority_success_rates_{traffic_pattern}.png')}")


if __name__ == "__main__":
    # 设置参数
    traffic_pattern = "extreme_fluctuation"  # 可以改为 "constant" 或 "burst"
    num_episodes = 20
    max_steps_per_episode = 2000
    
    # 查找最新的模型
    #model_path = "dynamic_uav_models/dynamic_20251021_185502/dqn_dynamic_uav_final.zip" #target_delay=5
    #model_path = "dynamic_uav_models/dynamic_20251204_172409/dqn_dynamic_uav_final.zip" #target_delay=5
    model_path = "/home/qwh/nndqn/dynamic_uav_models/dynamic_20260113_160534/dqn_dynamic_uav_final.zip" #target_delay=5
    #model_path = "dynamic_uav_models/dynamic_20251022_091452/dqn_dynamic_uav_final.zip" #target_delay=10
    #model_path = "dynamic_uav_models/dynamic_20251022_100643/dqn_dynamic_uav_final.zip" #target_delay=15
    if model_path is None:
        print("请先训练模型或检查模型路径")
        exit(1)
    
    # 运行模拟
    episode_df, success_rates = simulate_dqn(
        model_path=model_path,
        traffic_pattern=traffic_pattern,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode
    )
    
    if episode_df is not None:
        print("\n模拟完成！")
        print("优先级成功率统计:")
        for i, rate in enumerate(success_rates):
            print(f"  优先级 {i+1}: {rate:.3f}")
        print(f"  总体平均: {np.mean(success_rates):.3f}")
        
        # 新增：转发效率统计（修正版）
        avg_transmission_efficiency = episode_df['transmission_efficiency'].mean()
        avg_transmission_count = episode_df['transmission_count'].mean()
        avg_total_attempts = episode_df['total_transmission_attempts'].mean()
        avg_success_rate = episode_df['success_rate'].mean()
        avg_failure_rate = episode_df['failure_rate'].mean()
        
        print(f"\n转发效率统计:")
        print(f"  平均转发次数: {avg_transmission_count:.1f}")
        print(f"  平均总尝试次数: {avg_total_attempts:.1f}")
        print(f"  平均真实转发效率: {avg_transmission_efficiency:.3f} satisfied packets per attempt")
        print(f"  平均成功率: {avg_success_rate:.3f}")
        print(f"  平均失败率: {avg_failure_rate:.3f}")
        
        # 新增：高优先级包转发目标测试统计
        avg_high_priority_transmitted = episode_df['high_priority_transmitted'].mean()
        avg_transmissions_to_complete = episode_df['transmissions_to_complete_target'].mean()
        target_completion_rate = episode_df['target_completed'].mean()
        
        print(f"\n高优先级包转发目标测试统计:")
        print(f"  平均完成高优先级包数: {avg_high_priority_transmitted:.1f}")
        print(f"  平均完成所需转发次数: {avg_transmissions_to_complete:.1f}")
        print(f"  目标完成率: {target_completion_rate:.1%}")
