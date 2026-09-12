import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import math
import random
from collections import deque
import matplotlib.pyplot as plt
from dynamic_traffic_uav import DynamicTrafficUAVEnv, Packet
import datetime
import pandas as pd # Added for DataFrame

class CoDelController:
    """
    CoDel (Controlled Delay) 算法实现
    
    CoDel是一种主动队列管理算法，通过监控数据包的排队延迟来控制缓冲区膨胀。
    当排队延迟持续超过目标延迟时，CoDel会开始丢弃数据包，丢弃率随时间增加。
    """
    def __init__(self, target_delay=8, interval=15, max_drops_per_interval=3):
        """
        初始化CoDel控制器
        
        参数:
            target_delay: 目标延迟阈值(ms)，超过此值的包将被考虑丢弃
            interval: 控制间隔(ms)，在此间隔内检测持续拥塞
            max_drops_per_interval: 每个间隔内最多丢弃的包数量
        """
        self.target_delay = target_delay  # 目标延迟阈值
        self.interval = interval          # 控制间隔
        self.max_drops_per_interval = max_drops_per_interval  # 最大丢弃数
        self.dropping = False             # 是否处于丢弃状态
        self.drop_next = 0                # 下一个丢弃时间
        self.count = 0                    # 连续丢弃计数
        self.last_drop_time = 0           # 上次丢弃时间
        self.drops_in_current_interval = 0  # 当前间隔内的丢弃数
    
    def should_drop(self, packet, current_time):
        """
        判断是否应该丢弃数据包
        
        参数:
            packet: 数据包
            current_time: 当前时间
            
        返回:
            bool: 是否应该丢弃
        """
        # 计算排队延迟
        sojourn_time = packet.delay
        
        # 检查是否低于目标延迟
        below_target = sojourn_time < self.target_delay
        
        # 如果当前不在丢弃状态
        if not self.dropping:
            if below_target:
                # 延迟正常，不丢弃
                return False
            else:
                # 延迟超标，进入丢弃状态
                self.dropping = True
                self.drops_in_current_interval = 0
                # 计算下一个丢弃时间
                self.drop_next = current_time + self.interval / np.sqrt(max(1, self.count))
                # 重置计数
                self.count = 1
                # 记录丢弃时间
                self.last_drop_time = current_time
                # 丢弃当前包
                return True
        
        # 如果当前在丢弃状态
        else:
            if below_target:
                # 延迟恢复正常，退出丢弃状态
                self.dropping = False
                self.drops_in_current_interval = 0
                # 不丢弃
                return False
            else:
                # 延迟仍然超标
                if current_time > self.drop_next:
                    # 检查是否达到最大丢弃数
                    if self.drops_in_current_interval >= self.max_drops_per_interval:
                        return False
                    
                    # 到达下一个丢弃时间
                    self.count += 1
                    self.drops_in_current_interval += 1
                    # 更新下一个丢弃时间，间隔随count增加而减小
                    self.drop_next = current_time + self.interval / np.sqrt(self.count)
                    # 记录丢弃时间
                    self.last_drop_time = current_time
                    # 丢弃当前包
                    return True
                
                # 未到丢弃时间，但考虑基于优先级的丢弃
                if self.drops_in_current_interval < self.max_drops_per_interval:
                    # 增加基于优先级的丢弃概率
                    priority_factor = 1.0 / packet.priority  # 低优先级更容易被丢弃
                    delay_excess = (sojourn_time - self.target_delay) / self.target_delay
                    drop_probability = min(0.6, delay_excess * priority_factor)
                    
                    if random.random() < drop_probability:
                        self.drops_in_current_interval += 1
                        return True
                
                # 不丢弃
                return False


class CoDelDynamicUAVEnv(DynamicTrafficUAVEnv):
    """
    使用CoDel算法的动态流量UAV环境
    """
    def __init__(self, target_delay=8, interval=15, max_drops_per_interval=3, adaptive_processing=True, max_energy=10000, **kwargs):
        """
        初始化CoDel环境
        
        参数:
            target_delay: CoDel目标延迟
            interval: CoDel控制间隔
            max_drops_per_interval: 每个间隔内最多丢弃的包数量
            adaptive_processing: 是否启用自适应处理量
            max_energy: 最大能量
            **kwargs: 传递给DynamicTrafficUAVEnv的参数
        """
        super(CoDelDynamicUAVEnv, self).__init__(**kwargs)
        self.codel = CoDelController(target_delay=target_delay, interval=interval, max_drops_per_interval=max_drops_per_interval)
        self.adaptive_processing = adaptive_processing
        
        # 能量相关属性
        self.max_energy = max_energy
        self.remaining_energy = max_energy
        
        # 初始化历史记录
        self.history = {
            "buffer_length": [],
            "packets_generated": [],
            "packets_transmitted": [],
            "packets_dropped": [],
            "avg_delay": [],
            "delay_satisfied_ratio": []
        }
        
        # 记录CoDel特定统计信息
        self.codel_drops = 0  # CoDel算法主动丢弃的包数量
        self.delay_satisfied_count = 0 # 满足延迟要求的数据包数量
        self.delay_satisfied_by_priority = {1: 0, 2: 0, 3: 0} # 按优先级满足延迟要求的数据包数量
    
    def step_codel(self):
        """
        使用CoDel算法执行一步
        
        返回:
            reward: 奖励
            done: 是否结束
            info: 信息字典
        """
        reward = 0
        done = False
        
        # 使用与 SAC/DQN 环境相同的瑞利 + 香农信道容量模型，计算最大可传输包数
        self.current_step += 1
        if self.current_step % self.coherence_steps == 0:
            # 更新瑞利衰落信道
            self._update_next_snr()
            # 更新信道容量
            self.current_capacity = self.calculate_channel_capacity(self.channel_gain_squared)
            effective_bits = self.current_capacity * self.transmission_efficiency * self.time_slot_duration
            channel_maxnumber = int(effective_bits / self.packet_size_bits)
            # 根据可传输包数映射信道状态
            self.current_state = self._map_packets_to_state(channel_maxnumber)
            self.steps_until_channel_update = 0
        else:
            self.steps_until_channel_update = self.coherence_steps - (self.current_step % self.coherence_steps)
        
        # 更新 SNR / 容量历史
        self.snr_history.append(self.current_snr)
        if hasattr(self, "snr_history_length"):
            if len(self.snr_history) > self.snr_history_length:
                self.snr_history.pop(0)
        elif len(self.snr_history) > 10:
            self.snr_history.pop(0)
        
        if not hasattr(self, "capacity_history"):
            self.capacity_history = []
        self.capacity_history.append(getattr(self, "current_capacity", 0))
        if len(self.capacity_history) > 10:
            self.capacity_history.pop(0)
        
        # 根据信道容量得到本步最大传输包数（与 SAC/DQN 一致的裁剪策略）
        effective_bits = self.current_capacity * self.transmission_efficiency * self.time_slot_duration
        channel_maxnumber = int(effective_bits / self.packet_size_bits)
        channel_maxnumber = max(1, min(channel_maxnumber, 20))
        
        # 先更新延迟并立即丢弃超时包（与 DQN/SAC 保持一致）
        expired_count = 0
        expired_indices = []
        for i in range(len(self.buffer)):
            packet = self.buffer[i]
            packet.update_delay()  # 更新延迟
            packet.check_expired()  # 检查是否超时
            if packet.is_expired:  # 如果超时，立即标记为待删除
                expired_indices.append(i)
        
        # 从后往前删除超时数据包，避免索引偏移
        for idx in sorted(expired_indices, reverse=True):
            del self.buffer[idx]
            self.packets_dropped += 1
            expired_count += 1
            reward -= 1  # 超时丢弃惩罚
        
        # 本步统计
        transmitted_count = 0
        dropped_count = 0
        
        # 计算自适应处理量
        if self.adaptive_processing:
            base_process = 5
            
            # 如果缓冲区接近满，增加处理量（尽快清空缓冲区）
            if len(self.buffer) > self.max_queue * 0.8:
                base_process = min(8, channel_maxnumber)  # 尽量多处理，但不超过信道容量
            
            
            
            # 如果 CoDel 处于丢弃状态，减少处理量（拥塞时降低处理速度）
            if self.codel.dropping:
                base_process = max(1, base_process // 2)
            
            # 缓冲区很小时，保持正常处理量（不需要特殊限制）
            
            max_process = min(base_process, len(self.buffer))
        else:
            max_process = min(5, len(self.buffer))
        
        # 限制处理量不超过信道容量
        max_process = min(max_process, channel_maxnumber)
        
        any_transmitted = False
        
        for _ in range(max_process):
            if not self.buffer:  # 检查缓冲区是否为空
                break
            
            # 检查是否已达到信道容量限制
            if transmitted_count >= channel_maxnumber:
                break
                
            # 获取队首包
            packet = self.buffer[0]
            
            
            
            # 使用CoDel算法决定是否丢弃
            if self.codel.should_drop(packet, self.current_step):
                # CoDel决定丢弃
                self.buffer.popleft()
                self.packets_dropped += 1
                self.codel_drops += 1
                dropped_count += 1
                
                # 丢弃惩罚
                priority = packet.priority  # Packet类直接有priority属性
                delay_ratio = packet.delay / packet.max_delay
                if delay_ratio < 0.5:
                    reward -= 5 * priority  # 过早丢弃惩罚
                else:
                    reward -= 2 * priority  # 接近超时丢弃轻微惩罚
            else:
                # CoDel决定转发
                if self.remaining_energy > 0:
                    # 从缓冲区移除
                    packet = self.buffer.popleft()
                    
                    # 计算奖励
                    priority = packet.priority  # Packet类直接有priority属性
                    if packet.delay <= packet.max_delay:
                        delay_utilization = packet.delay / packet.max_delay
                        reward += 2 * delay_utilization * (1 + 0.5 * priority)
                        
                        # 新增：统计满足延迟要求的数据包
                        self.delay_satisfied_count += 1
                        self.delay_satisfied_by_priority[packet.priority] += 1
                    else:
                        excess_delay = packet.delay - packet.max_delay
                        reward -= excess_delay * 3
                    
                    # 更新统计
                    self.packets_transmitted += 1
                    self.total_delay += packet.delay
                    self.priority_transmitted[packet.priority] += 1
                    transmitted_count += 1
                    any_transmitted = True
                else:
                    # 能量不足，停止处理
                    reward -= 5
                    break
        
        # 能量消耗
        if any_transmitted:
            self.remaining_energy -= 1
        
        # 新数据包到达（延迟已在开始时更新，与DQN/SAC保持一致）
        packets_generated = self._generate_packets()
        
        # 更新历史记录
        self.history["buffer_length"].append(len(self.buffer))
        self.history["packets_generated"].append(packets_generated)
        self.history["packets_transmitted"].append(transmitted_count)
        self.history["packets_dropped"].append(dropped_count + expired_count)
        if self.packets_transmitted > 0:
            self.history["avg_delay"].append(self.total_delay / self.packets_transmitted)
            # 新增：计算并记录延迟满足率
            delay_satisfied_ratio = self.delay_satisfied_count / self.packets_transmitted
            self.history["delay_satisfied_ratio"].append(delay_satisfied_ratio)
        else:
            self.history["avg_delay"].append(0)
            self.history["delay_satisfied_ratio"].append(0)
        
        # episode 终止条件
        if self.remaining_energy <= 0:
            done = True
        
        # 信息字典
        info = {
            "packets_transmitted": self.packets_transmitted,
            "packets_dropped": self.packets_dropped,
            "packets_expired": expired_count,
            "codel_drops": self.codel_drops,
            "remaining_energy": self.remaining_energy,
            "buffer_size": len(self.buffer),
            "avg_delay": self.total_delay / max(1, self.packets_transmitted),
            "packets_transmitted_this_step": transmitted_count,
            "packets_dropped_this_step": dropped_count + expired_count,
            "transmission_rate": self.packets_transmitted / (self.packets_transmitted + self.packets_dropped + 1),
            "priority_transmitted": self.priority_transmitted,
            "priority_generated": self.priority_generated.copy(),  # 新增：优先级生成统计
            "current_traffic_load": self._get_current_traffic_load(),
            "delay_satisfied_count": self.delay_satisfied_count,  # 新增：满足延迟要求的数据包数量
            "delay_satisfied_ratio": self.delay_satisfied_count / max(1, self.packets_transmitted),  # 新增：延迟满足率
            "channel_state": self.current_state,  # 新增：信道状态
            "channel_snr": self.current_snr,  # 新增：信道SNR
            "channel_max_packets": channel_maxnumber  # 新增：信道最大传输包数
        }
        
        return reward, done, info
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        obs, _ = super().reset(seed=seed, options=options)
        self.codel_drops = 0
        self.remaining_energy = self.max_energy  # 重置能量
        self.delay_satisfied_count = 0  # 重置延迟满足计数
        self.delay_satisfied_by_priority = {1: 0, 2: 0, 3: 0}  # 重置按优先级的延迟满足计数
        
        # 重置历史记录
        self.history = {
            "buffer_length": [],
            "packets_generated": [],
            "packets_transmitted": [],
            "packets_dropped": [],
            "avg_delay": [],
            "delay_satisfied_ratio": []
        }
        
        return obs, {}
    
    def render(self, mode='human'):
        """渲染环境状态"""
        print(f"Step: {self.current_step}")
        print(f"Traffic load: {['Low', 'Medium', 'High'][self._get_current_traffic_load()]}")
        print(f"Buffer size: {len(self.buffer)}/{self.max_queue}")
        print(f"Remaining energy: {self.remaining_energy}/{self.max_energy}")
        print(f"Packets transmitted: {self.packets_transmitted}")
        print(f"Packets dropped: {self.packets_dropped} (CoDel drops: {self.codel_drops})")
        if self.packets_transmitted > 0:
            print(f"Average delay: {self.total_delay/self.packets_transmitted:.2f}")
            # 新增：显示延迟满足率
            delay_satisfied_ratio = self.delay_satisfied_count / self.packets_transmitted
            print(f"Delay satisfied ratio: {delay_satisfied_ratio:.1%}")
        
        transmission_rate = self.packets_transmitted / (self.packets_transmitted + self.packets_dropped + 1)
        print(f"Transmission rate: {transmission_rate:.3f}")
        
        # 打印优先级传输统计
        print("Priority transmission stats:")
        for i in range(1, 4):
            print(f"  Priority {i}: {self.priority_transmitted[i]} packets")
            
        # 新增：打印优先级延迟满足率
        print("Priority delay satisfied stats:")
        for i in range(1, 4):
            if self.priority_transmitted[i] > 0:
                delay_satisfied_ratio = self.delay_satisfied_by_priority[i] / self.priority_transmitted[i]
                print(f"  Priority {i}: {self.delay_satisfied_by_priority[i]}/{self.priority_transmitted[i]} = {delay_satisfied_ratio:.1%}")
            else:
                print(f"  Priority {i}: 0/0 = N/A")
        
        print("\n" + "-"*50)
        return None


def test_codel(traffic_pattern="dynamic", target_delay=8, interval=15, max_drops_per_interval=3, 
              adaptive_processing=True, max_energy=10000, render_interval=100, results_dir=None):
    """
    测试CoDel算法在不同流量模式下的性能
    
    参数:
        traffic_pattern: 流量模式 ("dynamic", "constant", "burst")
        target_delay: CoDel目标延迟
        interval: CoDel控制间隔
        max_drops_per_interval: 每个间隔内最多丢弃的包数量
        adaptive_processing: 是否启用自适应处理量
        max_energy: 最大能量
        render_interval: 渲染间隔
        results_dir: 结果保存目录
    
    返回:
        stats: 统计信息字典
        env: 环境对象
        df_delay: 包含所有数据包延迟信息的DataFrame
    """
    # 创建环境
    env = CoDelDynamicUAVEnv(
        traffic_pattern=traffic_pattern,
        target_delay=target_delay,
        interval=interval,
        max_drops_per_interval=max_drops_per_interval,
        adaptive_processing=adaptive_processing,
        max_energy=max_energy
    )
    
    # 重置环境
    env.reset()
    
    # 统计信息
    stats = {
        "rewards": [],
        "transmitted": [],
        "dropped": [],
        "codel_drops": [],
        "avg_delay": [],
        "buffer_length": [],
        "transmission_rate": [],
        "energy_efficiency": [],
        "priority_transmitted": [0, 0, 0, 0],  # 按优先级统计
        "priority_generated": [0, 0, 0, 0],     # 新增：按优先级统计生成的包数
        "priority_success_rate": [0, 0, 0, 0],  # 新增：按优先级统计传输成功率
        "delay_satisfied_count": 0,             # 新增：满足延迟要求的数据包数量
        "delay_satisfied_ratio": 0.0,           # 新增：延迟满足率
        "step_rewards": []
    }
    
    # 新增：用于记录所有数据包延迟信息的列表
    packet_delay_data = []
    
    # 运行环境
    done = False
    total_reward = 0
    step = 0
    
    while not done:
        # 使用CoDel算法执行一步
        reward, done, info = env.step_codel()
        
        # 新增：记录当前步骤中所有数据包的延迟信息
        for packet in env.buffer:
            # 根据priority推断sensor_id（priority 3对应sensor 0, priority 2对应sensor 1, priority 1对应sensor 2）
            sensor_id = 3 - packet.priority
            packet_delay_data.append({
                'step': step,
                'sensor_id': sensor_id,
                'priority': packet.priority,
                'current_delay': packet.delay,
                'max_delay': packet.max_delay,
                'delay_ratio': packet.delay / packet.max_delay,
                'traffic_pattern': traffic_pattern,
                'algorithm': 'CoDel',
                'delay_satisfied': packet.delay <= packet.max_delay  # 新增：是否满足延迟要求
            })
        
        # 更新统计信息
        total_reward += reward
        stats["step_rewards"].append(reward)
        
        # 每render_interval步渲染一次
        if step % render_interval == 0:
            print(f"\nStep {step}")
            env.render()
            print(f"Reward: {reward:.2f}")
        
        step += 1
    
    # 创建包含所有数据包延迟信息的DataFrame
    df_delay = pd.DataFrame(packet_delay_data)
    
    # 记录最终统计信息
    stats["rewards"].append(total_reward)
    stats["transmitted"].append(env.packets_transmitted)
    stats["dropped"].append(env.packets_dropped)
    stats["codel_drops"].append(env.codel_drops)
    stats["avg_delay"].append(env.total_delay / max(1, env.packets_transmitted))
    stats["buffer_length"].append(np.mean(env.history["buffer_length"]))
    stats["transmission_rate"].append(env.packets_transmitted / (env.packets_transmitted + env.packets_dropped + 1))
    energy_used = env.max_energy - env.remaining_energy
    stats["energy_efficiency"].append(env.packets_transmitted / max(1, energy_used))
    
    # 记录优先级传输统计
    for i in range(1, 4):
        stats["priority_transmitted"][i] = env.priority_transmitted[i]
    
    # 新增：记录优先级生成统计和传输成功率
    for i in range(1, 4):
        stats["priority_generated"][i] = env.priority_generated[i]
        
        # 计算传输成功率
        if stats["priority_generated"][i] > 0:
            success_rate = stats["priority_transmitted"][i] / stats["priority_generated"][i]
            # 添加安全检查，确保成功率不超过1
            if success_rate > 1.0:
                print(f"WARNING: Priority {i} success rate > 1.0 ({success_rate:.3f})")
                print(f"  Generated: {stats['priority_generated'][i]}, Transmitted: {stats['priority_transmitted'][i]}")
                # 强制限制为1.0
                success_rate = 1.0
            stats["priority_success_rate"][i] = success_rate
        else:
            stats["priority_success_rate"][i] = 0
    
    # 新增：记录延迟满足率统计
    stats["delay_satisfied_count"] = env.delay_satisfied_count
    stats["delay_satisfied_ratio"] = env.delay_satisfied_count / max(1, env.packets_transmitted)
    
    # 新增：计算每个优先级的延迟统计信息
    if not df_delay.empty:
        print("\n" + "="*50)
        print("PACKET DELAY STATISTICS BY PRIORITY:")
        print("="*50)
        
        for priority in [1, 2, 3]:
            priority_data = df_delay[df_delay['priority'] == priority]
            if not priority_data.empty:
                avg_delay = priority_data['current_delay'].mean()
                p95_delay = priority_data['current_delay'].quantile(0.95)
                p99_delay = priority_data['current_delay'].quantile(0.99)
                
                # 新增：计算该优先级满足延迟要求的包的比例
                delay_satisfied_count = priority_data['delay_satisfied'].sum()
                delay_satisfied_ratio = delay_satisfied_count / len(priority_data)
                
                print(f"Priority {priority}:")
                print(f"  Average Delay: {avg_delay:.2f} ms")
                print(f"  95th Percentile: {p95_delay:.2f} ms")
                print(f"  99th Percentile: {p99_delay:.2f} ms")
                print(f"  Total Packets: {len(priority_data)}")
                print(f"  Delay Satisfied: {delay_satisfied_count} ({delay_satisfied_ratio:.1%})")
                print()
        
        # 保存延迟数据到CSV文件
        if results_dir:
            delay_csv_path = os.path.join(results_dir, f"codel_delay_data_{traffic_pattern}.csv")
            df_delay.to_csv(delay_csv_path, index=False)
            print(f"延迟数据已保存到: {delay_csv_path}")
    
    # 打印最终结果
    print("\n" + "="*50)
    print(f"CoDel Algorithm Results ({traffic_pattern} traffic):")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Packets transmitted: {env.packets_transmitted}")
    print(f"Packets dropped: {env.packets_dropped} (CoDel drops: {env.codel_drops})")
    print(f"Transmission rate: {stats['transmission_rate'][0]:.3f}")
    
    # 新增：打印延迟满足率
    print(f"Delay satisfied ratio: {stats['delay_satisfied_ratio']:.1%}")
    
    # 新增：打印优先级传输成功率
    print(f"\nPriority transmission success rates:")
    for i in range(1, 4):
        generated = stats["priority_generated"][i]
        transmitted = stats["priority_transmitted"][i]
        if generated > 0:
            success_rate = transmitted / generated
            print(f"  Priority {i}: {transmitted}/{generated} = {success_rate:.3f}")
            # 添加调试信息
            if success_rate > 1.0:
                print(f"    WARNING: Success rate > 1.0! This indicates an error in statistics.")
        else:
            print(f"  Priority {i}: 0/0 = N/A")
    
    # 添加调试信息
    print(f"\nDebug info:")
    print(f"  Total packets generated: {sum(stats['priority_generated'][1:])}")
    print(f"  Total packets transmitted: {sum(stats['priority_transmitted'][1:])}")
    print(f"  Overall transmission rate: {sum(stats['priority_transmitted'][1:]) / max(1, sum(stats['priority_generated'][1:])):.3f}")
    
    print(f"Energy efficiency: {stats['energy_efficiency'][0]:.3f} packets/energy")
    print(f"Priority transmission: P1={stats['priority_transmitted'][1]}, P2={stats['priority_transmitted'][2]}, P3={stats['priority_transmitted'][3]}")
    print("="*50)
    
    # 绘制历史数据
    history_fig = env.plot_history(show=False)
    if results_dir and history_fig:
        history_fig.savefig(os.path.join(results_dir, f"codel_history_{traffic_pattern}.png"), dpi=300)
        plt.close(history_fig)
    
    # 绘制CoDel特定统计图表
    stats_fig = plot_codel_stats(env, stats, show=False)
    if results_dir and stats_fig:
        stats_fig.savefig(os.path.join(results_dir, f"codel_stats_{traffic_pattern}.png"), dpi=300)
        plt.close(stats_fig)
    
    return stats, env, df_delay

def plot_codel_stats(env, stats, show=True):
    """绘制CoDel特定统计图表"""
    plt.figure(figsize=(15, 12))  # 增加图表高度以容纳更多子图
    
    # 绘制奖励曲线
    plt.subplot(3, 2, 1)  # 修改为3行2列的布局
    plt.plot(stats["step_rewards"])
    plt.title("Step Rewards")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.grid(True)
    
    # 绘制缓冲区长度
    plt.subplot(3, 2, 2)
    plt.plot(env.history["buffer_length"])
    plt.title("Buffer Length")
    plt.xlabel("Steps")
    plt.ylabel("Length")
    plt.grid(True)
    
    # 绘制丢弃包与CoDel丢弃包的比例
    plt.subplot(3, 2, 3)
    labels = ['Total Drops', 'CoDel Drops', 'Expired']
    values = [
        env.packets_dropped, 
        env.codel_drops, 
        env.packets_dropped - env.codel_drops
    ]
    plt.bar(labels, values)
    plt.title("Packet Drop Analysis")
    plt.ylabel("Count")
    plt.grid(True, axis='y')
    
    # 绘制优先级传输比例
    plt.subplot(3, 2, 4)
    labels = ['Priority 1', 'Priority 2', 'Priority 3']
    values = [
        env.priority_transmitted[1],
        env.priority_transmitted[2],
        env.priority_transmitted[3]
    ]
    plt.bar(labels, values)
    plt.title("Priority Transmission")
    plt.ylabel("Count")
    plt.grid(True, axis='y')
    
    # 新增：绘制延迟满足率历史
    plt.subplot(3, 2, 5)
    if "delay_satisfied_ratio" in env.history:
        plt.plot(env.history["delay_satisfied_ratio"])
    else:
        # 如果历史中没有记录，则使用最终统计值绘制一条水平线
        plt.axhline(y=stats["delay_satisfied_ratio"], color='r', linestyle='-')
    plt.title("Delay Satisfied Ratio")
    plt.xlabel("Steps")
    plt.ylabel("Ratio")
    plt.ylim(0, 1.0)
    plt.grid(True)
    
    # 新增：绘制优先级延迟满足率饼图
    plt.subplot(3, 2, 6)
    if env.packets_transmitted > 0:
        labels = ['Priority 1', 'Priority 2', 'Priority 3']
        sizes = []
        for i in range(1, 4):
            if env.priority_transmitted[i] > 0:
                sizes.append(env.delay_satisfied_by_priority[i] / env.priority_transmitted[i])
            else:
                sizes.append(0)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title("Delay Satisfied Ratio by Priority")
    else:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')
        plt.title("Delay Satisfied Ratio by Priority")
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return plt.gcf()

def compare_with_sac(sac_model_path, traffic_patterns=["dynamic", "constant", "burst"], results_dir=None):
    """
    比较CoDel算法与SAC算法在不同流量模式下的性能
    
    参数:
        sac_model_path: SAC模型路径
        traffic_patterns: 要测试的流量模式列表
        results_dir: 结果保存目录
    """
    from stable_baselines3 import SAC
    
    # 加载SAC模型
    model = SAC.load(sac_model_path)
    
    # 结果存储
    results = {
        "sac": {},
        "codel": {}
    }
    
    # 对每种流量模式进行测试
    for pattern in traffic_patterns:
        print(f"\n\n{'='*50}")
        print(f"Testing on {pattern} traffic pattern")
        print(f"{'='*50}")
        
        # 测试SAC
        print("\nTesting SAC algorithm...")
        env_sac = DynamicTrafficUAVEnv(traffic_pattern=pattern)
        obs, _ = env_sac.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_sac.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            if step % 100 == 0:
                print(f"Step {step}, Buffer: {info['buffer_size']}, Reward: {reward:.2f}")
        
        # 记录SAC结果
        results["sac"][pattern] = {
            "reward": total_reward,
            "transmitted": env_sac.packets_transmitted,
            "dropped": env_sac.packets_dropped,
            "avg_delay": env_sac.total_delay / max(1, env_sac.packets_transmitted),
            "transmission_rate": env_sac.packets_transmitted / (env_sac.packets_transmitted + env_sac.packets_dropped + 1),
            "energy_efficiency": env_sac.packets_transmitted / (env_sac.max_energy - env_sac.remaining_energy),
            "priority_transmitted": env_sac.priority_transmitted.copy()
        }
        
        # 测试CoDel
        print("\nTesting CoDel algorithm...")
        stats_codel, env_codel, df_delay_codel = test_codel(traffic_pattern=pattern, render_interval=1000, results_dir=results_dir)
        
        # 记录CoDel结果
        results["codel"][pattern] = {
            "reward": stats_codel["rewards"][0],
            "transmitted": stats_codel["transmitted"][0],
            "dropped": stats_codel["dropped"][0],
            "codel_drops": stats_codel["codel_drops"][0],
            "avg_delay": stats_codel["avg_delay"][0],
            "transmission_rate": stats_codel["transmission_rate"][0],
            "energy_efficiency": stats_codel["energy_efficiency"][0],
            "priority_transmitted": stats_codel["priority_transmitted"]
        }
    
    # 绘制比较图表
    plot_comparison(results, traffic_patterns, results_dir)
    
    return results

def plot_comparison(results, traffic_patterns, results_dir=None):
    """绘制SAC与CoDel的比较图表"""
    metrics = [
        ("transmission_rate", "Transmission Rate"),
        ("avg_delay", "Average Delay (ms)"),
        ("energy_efficiency", "Energy Efficiency (packets/energy)")
    ]
    
    plt.figure(figsize=(15, 12))
    
    for i, (metric, title) in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        x = np.arange(len(traffic_patterns))
        width = 0.35
        
        # 获取指标值
        sac_values = [results["sac"][p][metric] for p in traffic_patterns]
        codel_values = [results["codel"][p][metric] for p in traffic_patterns]
        
        # 绘制柱状图
        plt.bar(x - width/2, sac_values, width, label='SAC')
        plt.bar(x + width/2, codel_values, width, label='CoDel')
        
        plt.title(title)
        plt.xticks(x, traffic_patterns)
        plt.ylabel(title)
        plt.grid(True, axis='y')
        plt.legend()
    
    # 绘制优先级传输比例
    plt.subplot(2, 2, 4)
    
    # 计算每种算法的优先级传输比例
    sac_priority_ratios = {}
    codel_priority_ratios = {}
    
    for pattern in traffic_patterns:
        # SAC优先级比例
        sac_total = sum(results["sac"][pattern]["priority_transmitted"][1:])
        sac_priority_ratios[pattern] = [
            results["sac"][pattern]["priority_transmitted"][i] / max(1, sac_total)
            for i in range(1, 4)
        ]
        
        # CoDel优先级比例
        codel_total = sum(results["codel"][pattern]["priority_transmitted"][1:])
        codel_priority_ratios[pattern] = [
            results["codel"][pattern]["priority_transmitted"][i] / max(1, codel_total)
            for i in range(1, 4)
        ]
    
    # 选择dynamic模式进行优先级比较
    pattern = "dynamic"
    
    x = np.arange(3)  # 3个优先级
    width = 0.35
    
    plt.bar(x - width/2, sac_priority_ratios[pattern], width, label='SAC')
    plt.bar(x + width/2, codel_priority_ratios[pattern], width, label='CoDel')
    
    plt.title(f"Priority Transmission Ratio ({pattern} traffic)")
    plt.xticks(x, ['Priority 1', 'Priority 2', 'Priority 3'])
    plt.ylabel("Ratio")
    plt.grid(True, axis='y')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    if results_dir:
        plt.savefig(os.path.join(results_dir, "sac_vs_codel_comparison.png"), dpi=300)
    else:
        plt.savefig("sac_vs_codel_comparison.png", dpi=300)
    
    plt.show()
    
    # 打印详细比较结果
    print("\n" + "="*50)
    print("Comparison Results:")
    print("="*50)
    
    for pattern in traffic_patterns:
        print(f"\n{pattern.upper()} TRAFFIC PATTERN:")
        print("-"*30)
        print(f"{'Metric':<20} {'SAC':<15} {'CoDel':<15} {'Difference':<15}")
        print("-"*65)
        
        for metric, title in metrics + [("reward", "Total Reward"), ("transmitted", "Packets Transmitted"), ("dropped", "Packets Dropped")]:
            sac_value = results["sac"][pattern][metric]
            codel_value = results["codel"][pattern][metric]
            diff = sac_value - codel_value
            diff_percent = diff / max(abs(codel_value), 1) * 100
            
            print(f"{title:<20} {sac_value:<15.2f} {codel_value:<15.2f} {diff:+.2f} ({diff_percent:+.1f}%)")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    # 创建结果目录
    results_dir = "codel_results_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(results_dir, exist_ok=True)
    print(f"创建结果目录: {results_dir}")
    
    # 测试CoDel算法
    # 可以调整target_delay和interval参数以获得最佳性能
    stats, env, df_delay = test_codel(
        traffic_pattern="dynamic",
        target_delay=25,  # 目标延迟阈值
        interval=100,     # 控制间隔
        max_energy=10000,
        results_dir=results_dir
    )
    
    # 与SAC进行比较
    # 替换为实际的SAC模型路径
    # compare_with_sac(
    #     sac_model_path="dynamic_uav_models/dynamic_1754297429/sac_dynamic_uav_final",
    #     traffic_patterns=["dynamic", "constant", "burst"],
    #     results_dir=results_dir
    # )
    
    print(f"\n所有结果已保存到目录: {results_dir}") 