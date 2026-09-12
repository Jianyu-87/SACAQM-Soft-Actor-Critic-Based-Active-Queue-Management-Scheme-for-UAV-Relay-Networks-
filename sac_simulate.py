import os
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from dynamic_traffic_uav import DynamicTrafficUAVEnv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def simulate_sac_model_all_patterns(model_path, num_episodes=10, max_steps_per_episode=2000):
    """
    使用训练好的SAC模型进行模拟运行，一次性测试所有流量模式
    
    参数:
        model_path: 训练好的模型路径
        num_episodes: 每个流量模式的episode数量
        max_steps_per_episode: 每个episode的最大步数
    """
    print(f"开始SAC模拟运行 - 测试所有流量模式")
    print(f"模拟episodes: {num_episodes}, 每episode最大步数: {max_steps_per_episode}")
    print("="*60)
    
    # 加载训练好的模型
    try:
        model = SAC.load(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 所有流量模式
    #patterns = ["constant", "dynamic", "burst"]
    patterns = ["dynamic"]
    all_results = {}
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_dir = f"sac_simulation_all_patterns_{timestamp}"
    os.makedirs(stats_dir, exist_ok=True)
    
    for pattern in patterns:
        print(f"\n{'='*50}")
        print(f"仿真 {pattern} 流量模式")
        print(f"{'='*50}")
    
        # 创建环境 - 确保与训练时参数一致
        env = DynamicTrafficUAVEnv(
            traffic_pattern=pattern, 
            max_steps=max_steps_per_episode,
            max_queue=150,  # 修改为150，与训练时一致
            max_transmissions=None
        )
        
        # 检查观察空间维度
        print(f"观察空间维度: {env.observation_space.shape}")
        print(f"动作空间维度: {env.action_space.shape}")
        
        # 存储所有episode的统计数据
        all_episode_stats = []
    
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print("-" * 40)
        
            # 重置环境
            obs, _ = env.reset()
            episode_reward = 0
            # 修改第62-71行的episode_stats初始化
            episode_stats = {
                'episode': episode + 1,
                'pattern': pattern,
                'priority_transmitted': [0, 0, 0, 0],
                'priority_generated': [0, 0, 0, 0],
                'priority_success_rate': [0, 0, 0],
                'transmit_num': 0,
                'energy_consumed': 0,
                'total_reward': 0,
                'avg_delay': 0.0,  # 总体平均延迟
                'priority_avg_delay': [0.0, 0.0, 0.0],  # 每个优先级的平均延迟
                'packets_generated': 0,
                'transmission_actions': 0,  # 新增：转发动作次数
                'delay_satisfied_count': 0,  # 新增：满足延迟要求的包数
                'transmission_efficiency': 0.0,  # 新增：转发效率
                'high_priority_transmitted': 0,     # 新增：高优先级包转发统计
                'transmissions_to_complete_target': 0,  # 新增
                'target_completed': False           # 新增
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
            # 修改第90-98行的统计信息更新
            episode_stats.update({
                'priority_transmitted': env.priority_transmitted.copy(),
                'priority_generated': env.priority_generated.copy(),
                'transmit_num': env.transmission_count,
                'energy_consumed': env.energy_consumed,
                'total_reward': episode_reward,
                'avg_delay': env.total_delay / max(env.packets_transmitted, 1),  # 计算总体平均延迟
                'packets_generated': env.packets_generated_sum,
                'delay_satisfied_count': env.delay_satisfied_count,  # 新增
            })
        
            # 计算每个优先级的平均延迟和成功率
            for priority in [1, 2, 3]:
                if env.priority_transmitted[priority] > 0:
                    # 使用正确的属性名
                    priority_delay = env.priority_delay[priority]
                    episode_stats['priority_avg_delay'][priority-1] = priority_delay / env.priority_transmitted[priority]
                else:
                    episode_stats['priority_avg_delay'][priority-1] = 0.0
                
                # 计算成功率：满足延迟要求的包数 / 生成的包数
                if episode_stats['priority_generated'][priority] > 0:
                    success_count = env.delay_satisfied_by_priority[priority]
                    total_generated = episode_stats['priority_generated'][priority]
                    episode_stats['priority_success_rate'][priority-1] = success_count / total_generated
                else:
                    episode_stats['priority_success_rate'][priority-1] = 0.0
            
            all_episode_stats.append(episode_stats)
            
            # 保存每个episode的转发日志
            episode_log_filename = os.path.join(stats_dir, f"transmission_log_episode_{episode + 1}.csv")
            env.save_transmission_log(episode_log_filename)
        
            # 打印episode总结
            # 修改第111-117行的episode总结打印
            print(f"Episode {episode + 1} 完成:")
            print(f"  优先级1成功率: {episode_stats['priority_success_rate'][0]:.3f}")
            print(f"  优先级2成功率: {episode_stats['priority_success_rate'][1]:.3f}")
            print(f"  优先级3成功率: {episode_stats['priority_success_rate'][2]:.3f}")
            print(f"  传输次数: {episode_stats['transmit_num']}")
            print(f"  转发动作次数: {episode_stats['transmission_actions']}")
            print(f"  转发效率: {episode_stats['transmission_efficiency']:.3f} satisfied packets per transmission")
            print(f"  总体平均延迟: {episode_stats['avg_delay']:.2f}")
            print(f"  优先级1平均延迟: {episode_stats['priority_avg_delay'][0]:.2f}")
            print(f"  优先级2平均延迟: {episode_stats['priority_avg_delay'][1]:.2f}")
            print(f"  优先级3平均延迟: {episode_stats['priority_avg_delay'][2]:.2f}")
            print(f"  总奖励: {episode_reward:.2f}")
        
        # 计算该流量模式的总体统计
        print(f"\n{pattern} 流量模式完成 - 优先级成功率统计")
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
        print(f"总体平均传输次数: {np.mean([stats['transmit_num'] for stats in all_episode_stats]):.2f}")
        print(f"总体平均延迟: {np.mean([stats['avg_delay'] for stats in all_episode_stats]):.2f}")
        print(f"优先级1平均延迟: {np.mean([stats['priority_avg_delay'][0] for stats in all_episode_stats]):.2f}")
        print(f"优先级2平均延迟: {np.mean([stats['priority_avg_delay'][1] for stats in all_episode_stats]):.2f}")
        print(f"优先级3平均延迟: {np.mean([stats['priority_avg_delay'][2] for stats in all_episode_stats]):.2f}")
        print(f"总体平均奖励: {np.mean([stats['total_reward'] for stats in all_episode_stats]):.2f}")
        
        # 新增：转发效率统计
        avg_transmission_actions = np.mean([stats['transmission_actions'] for stats in all_episode_stats])
        avg_transmission_efficiency = np.mean([stats['transmission_efficiency'] for stats in all_episode_stats])
        print(f"\n转发效率统计:")
        print(f"  平均转发动作次数: {avg_transmission_actions:.1f}")
        print(f"  平均转发效率: {avg_transmission_efficiency:.3f} satisfied packets per transmission")
        
        # 保存该流量模式的统计数据
        # 修改第140-145行的结果保存
        all_results[pattern] = {
            "episode_df": episode_df,
            "success_rates": avg_success_rates,
            "packets_generated": np.mean([stats['packets_generated'] for stats in all_episode_stats]),
            "avg_transmit": np.mean([stats['transmit_num'] for stats in all_episode_stats]),
            "avg_reward": np.mean([stats['total_reward'] for stats in all_episode_stats]),
            "avg_delay": np.mean([stats['avg_delay'] for stats in all_episode_stats]),  # 总体平均延迟
            "priority_avg_delay": [  # 每个优先级的平均延迟
                np.mean([stats['priority_avg_delay'][i] for stats in all_episode_stats]) 
                for i in range(3)
            ],
            "avg_transmission_actions": np.mean([stats['transmission_actions'] for stats in all_episode_stats]),  # 新增
            "avg_transmission_efficiency": np.mean([stats['transmission_efficiency'] for stats in all_episode_stats])  # 新增
        }
    
    # 保存所有结果
    
    # 保存每个流量模式的CSV文件
    for pattern, data in all_results.items():
        data['episode_df'].to_csv(os.path.join(stats_dir, f"episode_stats_{pattern}.csv"), index=False)
    
    # 生成比较图表
    plot_all_patterns_comparison(all_results, stats_dir)
    
    print(f"\n所有统计数据已保存到: {stats_dir}")
    
    return all_results

def plot_all_patterns_comparison(all_results, save_dir):
    """绘制所有流量模式的比较图表"""
    patterns = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SAC模型所有流量模式性能比较', fontsize=16)
    
    # 1. 优先级成功率比较
    priority_labels = ['Priority 1', 'Priority 2', 'Priority 3']
    x = np.arange(len(priority_labels))
    width = 0.25
    
    for i, pattern in enumerate(patterns):
        success_rates = all_results[pattern]['success_rates']
        axes[0, 0].bar(x + i*width, success_rates, width, label=pattern, alpha=0.8)
    
    axes[0, 0].set_title('Priority Success Rate Comparison')
    axes[0, 0].set_xlabel('Priority Level')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(priority_labels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 平均传输次数比较
    transmit_counts = [all_results[pattern]['avg_transmit'] for pattern in patterns]
    bars = axes[0, 1].bar(patterns, transmit_counts, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    axes[0, 1].set_title('Average Transmission Count')
    axes[0, 1].set_ylabel('Transmission Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 在柱子上显示数值
    for bar, value in zip(bars, transmit_counts):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}', ha='center', va='bottom')
    
    # 3. 平均奖励比较
    rewards = [all_results[pattern]['avg_reward'] for pattern in patterns]
    bars = axes[1, 0].bar(patterns, rewards, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    axes[1, 0].set_title('Average Total Reward')
    axes[1, 0].set_ylabel('Total Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 在柱子上显示数值
    for bar, value in zip(bars, rewards):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}', ha='center', va='bottom')
    
    # 4. 总体成功率比较
    overall_success = [np.mean(all_results[pattern]['success_rates']) for pattern in patterns]
    bars = axes[1, 1].bar(patterns, overall_success, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    axes[1, 1].set_title('Overall Average Success Rate')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 在柱子上显示数值
    for bar, value in zip(bars, overall_success):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, "sac_all_patterns_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"比较图表已保存到: {save_path}")
    
    plt.show()
    
    return save_path

if __name__ == "__main__":
    # 示例使用
    print("SAC模型仿真模拟 - 所有流量模式")
    print("="*60)
    
    # 假设您有训练好的模型
    #model_path = "SAC_dynamic_uav_models/constant_20251027_092559/sac_dynamic_uav_500000_steps.zip"
    #model_path = "/home/qwh/nndqn/SAC_dynamic_uav_models/dynamic_20260104_163756/sac_dynamic_uav_final.zip"
    #model_path = "/home/qwh/nndqn/SAC_dynamic_uav_models/dynamic_20260109_091333/sac_dynamic_uav_final.zip"
    model_path = "/home/qwh/nndqn/SAC_dynamic_uav_models/dynamic_20260117_214427/sac_dynamic_uav_final.zip"
    # 检查模型文件是否存在
    if os.path.exists(model_path):
        print(f"找到模型文件: {model_path}")
        
        # 运行所有流量模式的仿真
        all_results = simulate_sac_model_all_patterns(model_path, num_episodes=10)
        
        # 打印最终比较结果
        print(f"\n{'='*60}")
        print("所有流量模式最终比较结果")
        print(f"{'='*60}")
        
        for pattern, data in all_results.items():
            print(f"\n{pattern} 流量模式:")
            print(f"  优先级1成功率: {data['success_rates'][0]:.3f}")
            print(f"  优先级2成功率: {data['success_rates'][1]:.3f}")
            print(f"  优先级3成功率: {data['success_rates'][2]:.3f}")
            print(f"  平均传输次数: {data['avg_transmit']:.2f}")
            print(f"  平均转发动作次数: {data['avg_transmission_actions']:.1f}")
            print(f"  平均转发效率: {data['avg_transmission_efficiency']:.3f} satisfied packets per transmission")
            
            # 新增：高优先级包转发目标测试统计
            avg_high_priority_transmitted = np.mean([stats['high_priority_transmitted'] for stats in all_results[pattern]['episode_df'].to_dict('records')])
            avg_transmissions_to_complete = np.mean([stats['transmissions_to_complete_target'] for stats in all_results[pattern]['episode_df'].to_dict('records')])
            target_completion_rate = np.mean([stats['target_completed'] for stats in all_results[pattern]['episode_df'].to_dict('records')])
            
            print(f"  平均完成高优先级包数: {avg_high_priority_transmitted:.1f}")
            print(f"  平均完成所需转发次数: {avg_transmissions_to_complete:.1f}")
            print(f"  目标完成率: {target_completion_rate:.1%}")
            print(f"  总体平均延迟: {data['avg_delay']:.2f}")
            print(f"  优先级1平均延迟: {data['priority_avg_delay'][0]:.2f}")
            print(f"  优先级2平均延迟: {data['priority_avg_delay'][1]:.2f}")
            print(f"  优先级3平均延迟: {data['priority_avg_delay'][2]:.2f}")
            print(f"  平均奖励: {data['avg_reward']:.2f}")
            print(f"  生成包数: {data['packets_generated']:.2f}")
        
    else:
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型，或修改模型路径")