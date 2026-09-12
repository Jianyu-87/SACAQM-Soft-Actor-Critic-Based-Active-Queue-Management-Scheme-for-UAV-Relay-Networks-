import numpy as np
import pandas as pd
from codel_dynamic_uav import CoDelDynamicUAVEnv

def simulate_codel_robustness(num_episodes=20, max_steps_per_episode=2000, traffic_pattern="dynamic"):
    """
    使用CODEL算法进行鲁棒性测试
    测试在不同流量模式下的性能
    
    参数:
        num_episodes: 模拟的episode数量
        max_steps_per_episode: 每个episode的最大步数
        traffic_pattern: 流量模式
    
    返回:
        episode_df: episode统计DataFrame
        step_df: step统计DataFrame
        success_rates: 各优先级成功率列表
    """
    print(f"开始CODEL鲁棒性测试 - {traffic_pattern}流量模式")
    print(f"模拟episodes: {num_episodes}, 每episode最大步数: {max_steps_per_episode}")
    print("="*60)
    
    # 创建环境
    # CoDel参数设置说明：
    # - target_delay: 目标延迟阈值，超过此值的包将被考虑丢弃
    #   建议设置为略低于高优先级包的max_delay(5)，以保护高优先级包
    #   当前设置：4（略低于高优先级max_delay=5，给高优先级包留出缓冲）
    # - interval: 控制间隔，用于检测持续拥塞
    #   建议设置为target_delay的2-4倍，足够检测拥塞但响应及时
    #   当前设置：12（target_delay的3倍，平衡检测和响应速度）
    env = CoDelDynamicUAVEnv(
        traffic_pattern=traffic_pattern,
        target_delay=4,      # 略低于高优先级max_delay=5，保护高优先级包
        interval=12,         # target_delay的3倍，平衡检测和响应
        max_drops_per_interval=3,
        adaptive_processing=True,
        max_steps=max_steps_per_episode,
        max_queue=100,
        max_transmissions=None
    )
    
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
            'transmission_count': 0,  # 转发次数（有传输的步数）
        }
        
        step_count = 0
        done = False
        total_buffer_usage = 0
        total_expired = 0  # 累计过期包数
        transmission_count = 0  # 转发次数（有传输的步数）
        
        while not done and step_count < max_steps_per_episode:
            # 获取当前流量负载
            current_traffic_load = env._get_current_traffic_load()
            
            # 使用CODEL算法执行一步
            reward, done, info = env.step_codel()
            
            episode_reward += reward
            step_count += 1
            
            # 统计转发次数（如果这一步有传输，则计数+1）
            if info.get('packets_transmitted_this_step', 0) > 0:
                transmission_count += 1
            
            # 累计过期包数
            expired_this_step = info.get('packets_expired', 0)
            total_expired += expired_this_step
            
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
                'packets_transmitted': info.get('packets_transmitted_this_step', 0),
                'packets_expired': expired_this_step
            }
            step_by_step_data.append(step_data)
            
            # 根据流量负载分类统计性能
            if current_traffic_load > 1.5:  # 高流量
                episode_stats['high_traffic_performance']['transmitted'] += info.get('packets_transmitted_this_step', 0)
                episode_stats['high_traffic_performance']['expired'] += expired_this_step
            else:  # 低流量
                episode_stats['low_traffic_performance']['transmitted'] += info.get('packets_transmitted_this_step', 0)
                episode_stats['low_traffic_performance']['expired'] += expired_this_step
            
            # 每100步打印一次进度
            if step_count % 100 == 0:
                print(f"  步骤 {step_count}: 奖励={reward:.2f}, 缓冲区={buffer_size}, "
                      f"流量负载={current_traffic_load:.2f}")
        
        # 记录episode结束时的统计信息
        total_packets_generated = sum(env.priority_generated)
        
        episode_stats.update({
            'priority_transmitted': env.priority_transmitted.copy(),
            'priority_generated': env.priority_generated.copy(),
            'transmit_num': env.packets_transmitted,  # 传输的包数
            'transmission_count': transmission_count,  # 转发次数（有传输的步数）
            'total_reward': episode_reward,
            'avg_delay': env.total_delay / max(env.packets_transmitted, 1),
            'packets_generated': total_packets_generated,
            'packets_expired': total_expired,  # 使用累计的过期包数
            'avg_buffer_usage': total_buffer_usage / max(step_count, 1),
            'delay_satisfied_count': env.delay_satisfied_count,
        })
        
        # 计算每个优先级的平均延迟和成功率
        for priority in [1, 2, 3]:
            # CODEL环境可能没有priority_transmission_delays，使用平均延迟
            if env.packets_transmitted > 0:
                episode_stats['priority_avg_delay'][priority-1] = env.total_delay / env.packets_transmitted
            else:
                episode_stats['priority_avg_delay'][priority-1] = 0.0
            
            # 计算成功率
            if episode_stats['priority_generated'][priority] > 0:
                success_count = env.delay_satisfied_by_priority[priority]
                total_generated = episode_stats['priority_generated'][priority]
                episode_stats['priority_success_rate'][priority-1] = success_count / total_generated
            else:
                episode_stats['priority_success_rate'][priority-1] = 0.0
        
        # 计算高/低流量下的传输效率
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
        print(f"  优先级3传输个数: {episode_stats['priority_transmitted'][3]}")
        print(f"  优先级3生成个数: {episode_stats['priority_generated'][3]}")
        print(f"  传输次数: {episode_stats['transmit_num']}")
        print(f"  总奖励: {episode_reward:.2f}")
    
    # 计算总体统计
    print("\n" + "="*60)
    print("CODEL鲁棒性测试完成统计")
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
    
    return episode_df, step_df, avg_success_rates

