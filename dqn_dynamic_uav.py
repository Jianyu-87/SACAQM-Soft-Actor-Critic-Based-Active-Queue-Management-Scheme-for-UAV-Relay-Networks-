import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import math
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from torch.cuda import utilization

class Packet:
    def __init__(self, sensor_id, max_delay, arrival_time, priority):
        self.sensor_id = sensor_id  # 传感器ID
        self.max_delay = max_delay  # 最大可容忍延迟
        self.arrival_time = arrival_time  # 到达时间
        self.delay = 0  # 当前延迟
        self.priority = priority  # 包的优先级
        self.transmission_delay = None  # 转发时的延迟（新增）
        self.is_expired_flag = False  # 超时标记

    def update_delay(self):
        self.delay += 1
        
    def is_expired(self):
        return self.delay > self.max_delay
    
    def get_urgency(self):
        """计算包的紧急度（优先级和延迟的组合）"""
        # 延迟率：当前延迟占最大延迟的比例
        delay_ratio = self.delay / self.max_delay
        # 紧急度 = 优先级 * (1 + 延迟率)
        # 这样高优先级且接近超时的包会有最高的紧急度
        return self.priority * (1 + delay_ratio)

class DynamicTrafficUAVEnv(gym.Env):
    def __init__(self,
                 max_queue=150,  # 减小缓冲区容量
                 max_steps=10000,  # 改为最大步数限制
                 traffic_pattern="dynamic",  # 流量模式：dynamic, constant, burst
                 target_delay=5,  # 目标延迟
                 max_transmissions=None,  # 最大转发次数限制（None表示无限制）
                 bandwidth=1.15e4,  # 带宽(Hz)，用于香农公式计算（与xindao.py一致）
                 packet_size_bits=8000,  # 每个数据包的比特数（与xindao.py一致）
                 time_slot_duration=1.0,  # 时隙持续时间(秒)
                 transmission_efficiency=0.7,  # 传输效率因子(eta)
                 correlation_factor=0.7,  # SNR时间相关性因子
                 coherence_steps=3,  # 块衰落周期（每多少步更新一次信道，与xindao.py一致）
                 packet_thresholds=[1, 3, 5, 7],  # 包数阈值（与xindao.py一致）
                 noise_std=7.0,  # 噪声标准差（已弃用，保留用于兼容性）
                 transmit_power=1.0,  # P_t: 发射功率 (W)
                 noise_psd=1e-4):  # N₀: 噪声功率谱密度 (W/Hz)
        super(DynamicTrafficUAVEnv, self).__init__()
        
        # 缓冲区队列
        self.buffer = deque(maxlen=max_queue)
        self.max_queue = max_queue
        
        # 信道容量计算参数（基于香农公式）
        self.bandwidth = bandwidth  # 带宽(Hz)，用于香农公式计算
        self.packet_size_bits = packet_size_bits  # 每个数据包的比特数
        self.time_slot_duration = time_slot_duration  # 时隙持续时间(秒)，每个step对应的时间长度
        self.transmission_efficiency = transmission_efficiency  # 传输效率因子(eta)，考虑编码效率和头部开销
        
        # 香农公式参数：C = B * log₂(1 + (|h|² * P_t) / (N₀ * B))
        self.transmit_power = transmit_power  # P_t: 发射功率 (W)
        self.noise_psd = noise_psd  # N₀: 噪声功率谱密度 (W/Hz)
        
        # 信道相关性参数（来自xindao.py）
        self.correlation_factor = correlation_factor  # SNR时间相关性因子 (0-1)
        self.coherence_steps = coherence_steps  # 块衰落：每多少步更新一次信道（模拟相干时间）
        self.packet_thresholds = packet_thresholds  # 包数阈值：[deep_fade上限, poor上限, fair上限, good上限]
        self.noise_std = noise_std  # 噪声标准差（已弃用，保留用于兼容性）
        
        # 传感器配置 - 添加优先级
        self.sensors = [
            {"id": 0, "base_rate": 1, "max_delay": 5, "priority": 3},  # 高优先级，低延迟容忍
            {"id": 1, "base_rate": 1, "max_delay": 10, "priority": 2},  # 中优先级，中延迟容忍
            {"id": 2, "base_rate": 1, "max_delay": 15, "priority": 1}   # 低优先级，高延迟容忍
        ]
        
        # 流量模式
        self.traffic_pattern = traffic_pattern
        self.traffic_cycle_length = 100  # 流量周期长度
        self.burst_duration = 20  # 突发流量持续时间
        
        # 步数相关
        self.max_steps = max_steps
        self.current_step = 0
        
        self.target_delay = target_delay
        self.max_transmissions = max_transmissions
        
        # 环境参数
        self.packets_transmitted = 0
        self.packets_dropped = 0
        self.packets_expired = 0  # 因超时自动丢弃的包
        self.total_delay = 0
        self.priority_transmitted = [0, 0, 0, 0]  # 按优先级统计传输的包数量
        self.transmit_num = 0
        self.advancedrop = 0
        self.overtime_transmitted = 0
        
        # 新增：按优先级统计生成的包数量
        self.priority_generated = [0, 0, 0, 0]  # 按优先级统计生成的包数量
        self.priority_expired = [0, 0, 0, 0]    # 按优先级统计超时的包数量
        
        # 新增：延迟满足率统计
        self.delay_satisfied_count = 0  # 满足延迟要求的数据包数量
        self.delay_satisfied_by_priority = [0, 0, 0, 0]  # 按优先级统计满足延迟要求的数据包数量
        
        # 新增：转发尝试统计
        self.total_transmission_attempts = 0  # 总转发尝试次数（包括成功和失败）
        self.successful_transmissions = 0      # 成功转发次数
        self.failed_transmissions = 0          # 失败转发次数（超时等）
        
        # 新增：高优先级包转发目标测试
        self.high_priority_transmitted = 0     # 已转发的高优先级包数
        self.high_priority_target = 200        # 目标：转发200个高优先级包
        self.transmissions_to_complete_target = 0  # 完成目标所需的转发次数
        self.target_completed = False          # 是否已完成目标

        # 新增：信道状态模拟器（使用瑞利衰落模型）
        self.noise_power = 1.0
        self.snr_thresholds = [8, 13, 18, 23]  # 保留用于兼容性，但不再用于状态判断
        self.avg_snr = 14.5  # 平均SNR（与xindao.py保持一致）
        self.current_step = 0
        self.snr_history = []
        self.channel_history = []
        self.capacity_history = []  # 信道容量历史
        
        # 瑞利衰落参数
        # 根据平均SNR计算瑞利分布的尺度参数
        # 平均SNR（线性域）= (|h|² * P_t) / (N₀ * B)
        # 对于瑞利衰落，E[|h|²] = 2*sigma²
        # 因此：avg_snr_linear = (2*sigma² * P_t) / (N₀ * B)
        # 所以：sigma² = (avg_snr_linear * N₀ * B) / (2 * P_t)
        avg_snr_linear = 10**(self.avg_snr / 10.0)
        self.rayleigh_sigma = np.sqrt((avg_snr_linear * self.noise_psd * self.bandwidth) / (2.0 * self.transmit_power))
        
        # 瑞利衰落的复信道增益（实部和虚部）
        self.h_real = np.random.normal(0, self.rayleigh_sigma)
        self.h_imag = np.random.normal(0, self.rayleigh_sigma)
        # 信道增益幅度平方 |h|²
        self.channel_gain_squared = self.h_real**2 + self.h_imag**2
        
        # 计算初始SNR（线性域）= (|h|² * P_t) / (N₀ * B)，然后转换为 dB
        snr_linear = (self.channel_gain_squared * self.transmit_power) / (self.noise_psd * self.bandwidth)
        self.current_snr = 10 * np.log10(snr_linear + 1e-10)
        self.current_snr = np.clip(self.current_snr, -5, 45)
        
        # 使用香农公式计算当前信道容量：C = B * log₂(1 + (|h|² * P_t) / (N₀ * B))
        self.current_capacity = self.calculate_channel_capacity(self.channel_gain_squared)
        # 先计算包数，再根据包数确定状态
        effective_bits = self.current_capacity * self.transmission_efficiency * self.time_slot_duration
        channel_maxnumber = int(effective_bits / self.packet_size_bits)
        self.current_state = self._map_packets_to_state(channel_maxnumber)
        
        # 块衰落相关：记录距离下次信道更新的步数
        self.steps_until_channel_update = 0  # 距离下次信道更新的步数
        
        # 记录历史数据
        self.history = {
            "buffer_length": [],
            "packets_generated": [],
            "packets_transmitted": [],
            "packets_dropped": [],
            "advancedrop": [],
            "avg_delay": [],
            "delay_satisfied_ratio": [], # 新增：延迟满足率历史记录
            "transmit_num":[] # 新增：转发次数历史记录
        }
        
        
        # 0: 全部转发
        # 1: 全部等待
        # 2: 全部丢弃
        self.action_space = spaces.Discrete(3)
        
        # 状态空间:
        # 1. 当前步数比例
        # 2. 缓冲区长度比例
        # 3. 当前流量状态
        # 4. 目标延迟比例
        # 5. 当前信道状态
        # 6. 队列中所有数据包的延迟比例
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3 + self.max_queue,),
            dtype=np.float32
        )
        
        # 新增：转发时延统计
        self.transmission_delays = []  # 记录所有转发时延
        self.priority_transmission_delays = {1: [], 2: [], 3: []}  # 按优先级记录转发时延
        self.step_transmission_delays = []  # 记录每步的转发时延
    
    def reset(self, seed=None, options=None):
        # 重置环境
        self.buffer.clear()
        self.current_step = 0
        self.packets_transmitted = 0
        self.packets_dropped = 0
        self.packets_expired = 0
        self.total_delay = 0
        self.priority_transmitted = [0, 0, 0, 0]
        self.transmit_num = 0
        self.advancedrop = 0
        self.overtime_transmitted = 0

        # 重置优先级生成统计
        self.priority_generated = [0, 0, 0, 0]
        self.priority_expired = [0, 0, 0, 0]
        
        # 重置转发次数统计
        self.transmission_count = 0
        
        # 重置延迟满足率统计
        self.delay_satisfied_count = 0
        self.delay_satisfied_by_priority = [0, 0, 0, 0]
        
        # 重置转发尝试统计
        self.total_transmission_attempts = 0
        self.successful_transmissions = 0
        self.failed_transmissions = 0
        
        # 重置高优先级包转发目标测试
        self.high_priority_transmitted = 0
        self.transmissions_to_complete_target = 0
        self.target_completed = False

        # 重置信道状态模拟器（使用瑞利衰落模型）
        self.snr_history = []
        self.channel_history = []
        self.capacity_history = []
        
        # 重置瑞利衰落信道增益（重新采样）
        self.h_real = np.random.normal(0, self.rayleigh_sigma)
        self.h_imag = np.random.normal(0, self.rayleigh_sigma)
        
        # 计算初始信道增益幅度平方 |h|²
        self.channel_gain_squared = self.h_real**2 + self.h_imag**2
        
        # 计算初始SNR（线性域）= (|h|² * P_t) / (N₀ * B)，然后转换为 dB
        snr_linear = (self.channel_gain_squared * self.transmit_power) / (self.noise_psd * self.bandwidth)
        self.current_snr = 10 * np.log10(snr_linear + 1e-10)
        self.current_snr = np.clip(self.current_snr, -5, 45)
        
        # 使用香农公式计算当前信道容量：C = B * log₂(1 + (|h|² * P_t) / (N₀ * B))
        self.current_capacity = self.calculate_channel_capacity(self.channel_gain_squared)
        # 先计算包数，再根据包数确定状态
        effective_bits = self.current_capacity * self.transmission_efficiency * self.time_slot_duration
        channel_maxnumber = int(effective_bits / self.packet_size_bits)
        self.current_state = self._map_packets_to_state(channel_maxnumber)
        # 重置块衰落计数器
        self.steps_until_channel_update = 0
        
        # 重置历史记录
        self.history = {
            "buffer_length": [],
            "packets_generated": [],
            "packets_transmitted": [],
            "packets_dropped": [],
            "avg_delay": [],
            "delay_satisfied_ratio": [],  # 新增：延迟满足率历史记录
            "transmit_num":[] # 新增：转发次数历史记录
        }
        
        # 重置转发时延统计
        self.transmission_delays = []
        self.priority_transmission_delays = {1: [], 2: [], 3: []}
        self.step_transmission_delays = []
        
        # 初始生成一些数据包
        self._generate_packets()
        
        # 返回观察值和info字典（gymnasium要求）
        return self._get_observation(), {}
    
    def step(self, action):
        # 一次性处理队列中所有包的统一决策
        # action: 0(全部转发), 1(全部等待), 2(全部丢弃)
        
        reward = 0
        done = False
        info = {}

        # 本步统计
        transmitted_indices = []
        dropped_indices = []
        waited_indices = []
        any_packet_transmitted = False

        packets_processed = 0
        packets_transmitted_this_step = 0
        packets_dropped_this_step = 0
        packets_waited_this_step = 0
        channel_maxnumber = 0

        # 先更新步数（用于块衰落判断）
        self.current_step += 1
        
        # 使用xindao.py的块衰落模型：只有当step是coherence_steps的倍数时才更新信道
        if self.current_step % self.coherence_steps == 0:
            # 更新瑞利衰落信道增益
            self._update_next_snr()
            # 更新信道指标（使用香农公式：C = B * log₂(1 + (|h|² * P_t) / (N₀ * B))）
            self.current_capacity = self.calculate_channel_capacity(self.channel_gain_squared)
            # 先计算包数，再根据包数确定状态（而不是SNR）
            effective_bits = self.current_capacity * self.transmission_efficiency * self.time_slot_duration
            channel_maxnumber = int(effective_bits / self.packet_size_bits)
            self.current_state = self._map_packets_to_state(channel_maxnumber)
            self.steps_until_channel_update = 0
        else:
            # 信道保持不变，更新距离下次更新的步数
            self.steps_until_channel_update = self.coherence_steps - (self.current_step % self.coherence_steps)
        
        # 更新SNR历史序列
        self.snr_history.append(self.current_snr)
        if len(self.snr_history) > 10:
            self.snr_history.pop(0)
        # 更新信道容量历史
        self.capacity_history.append(self.current_capacity)
        if len(self.capacity_history) > 10:
            self.capacity_history.pop(0)
        
        # 基于香农公式计算的信道容量，动态确定最大传输包数
        # 使用xindao.py的计算方式：考虑传输效率因子和时隙持续时间
        # 有效传输比特数 = 信道容量(bits/s) × 效率因子 × 时隙持续时间(s)
        effective_bits = self.current_capacity * self.transmission_efficiency * self.time_slot_duration
        # 可传输包数 = 有效比特数 / 包大小
        channel_maxnumber = int(effective_bits / self.packet_size_bits)
        # 设置最小值为1，最大值为合理的上限（例如20）
        channel_maxnumber = max(1, min(channel_maxnumber, 20))

        # 记录动作（用于调试）
        action_taken = action
        
        # 移除智能转发条件，让智能体完全自主决策
        
        # 检查转发次数限制（仅在测试时生效）
        if self.max_transmissions is not None and self.transmission_count >= self.max_transmissions:
            # 达到转发次数限制，强制等待
            action = 1
            action_taken = 1

        # 计算队列中所有包的加权平均延迟用于奖励计算
        if len(self.buffer) > 0:
            # 使用平方加权平均延迟
            weighted_delay = sum(packet.delay * packet.delay for packet in self.buffer) / len(self.buffer)
            max_delay = max(packet.delay for packet in self.buffer)
            min_delay = min(packet.delay for packet in self.buffer)
            
            # 根据动作计算奖励
            if action == 0:  # 全部转发
                # 只传输前channel_maxnumber个包
                selected_packets = list(self.buffer)[:channel_maxnumber]
                # 仅当实际发送至少 1 个包时才计入转发次数（空转发不计入）
                if len(selected_packets) > 0:
                    self.transmit_num += 1
                    self.transmission_count += 1
                    self.total_transmission_attempts += 1
                    if not self.target_completed:
                        self.transmissions_to_complete_target += 1
                # 奖励：按包计算，过期包惩罚、未过期奖励；统计在缓冲区更新处统一做，避免双重统计
                reward = 0.0
                for packet in selected_packets:
                    if packet.is_expired():
                        reward += -10.0
                    else:
                        reward += 10.0
                for i in range(len(selected_packets)):
                    transmitted_indices.append(i)
                packets_transmitted_this_step = len(selected_packets)
                any_packet_transmitted = len(selected_packets) > 0

                energy_cost = 6
                reward -= energy_cost


            elif action == 1:  # 全部等待（不惩罚）
                reward = 1.0
                for i in range(len(self.buffer)):
                    waited_indices.append(i)
                    packets_waited_this_step += 1

            elif action == 2:  # 全部丢弃
                # 丢弃动作：给予惩罚，避免智能体过度使用丢弃
                reward = 0.0
                queue_ratio = len(self.buffer) / self.max_queue
                total_packets = len(self.buffer)
                
                for packet in self.buffer:
                    delay_ratio = packet.delay / self.target_delay
                    
                    
                    if delay_ratio <= 1.0:
                        # 接近最大延迟丢弃 - 轻微惩罚
                        packet_reward = -2.0
                    else:
                        # 超时后丢弃 - 轻微惩罚（不鼓励丢弃，即使是超时包）
                        packet_reward = -1.0
                    
                    reward += packet_reward
                
                # 丢弃动作整体给予惩罚（根据丢弃包数量，避免过度丢弃）
                # 丢弃越多，惩罚越大，防止智能体只学会丢弃
                base_penalty = 1
                reward -= base_penalty 

                for i in range(len(self.buffer)):
                    dropped_indices.append(i)
                    packets_dropped_this_step += 1
                

               

        # === 缓冲区更新 ===
        # 在删除前记录本步转发包的时延（删除后索引会错位）
        step_delays = [self.buffer[i].delay for i in transmitted_indices if i < len(self.buffer)]
        
        # 处理转发和丢弃的包
        for idx in sorted(set(transmitted_indices + dropped_indices), reverse=True):
            if idx >= len(self.buffer):
                continue  # 安全检查
                
            packet = self.buffer[idx]
            if idx in transmitted_indices:
                packet.transmission_delay = packet.delay
                if packet.is_expired():
                    # 已过期仍被转发：只计为过期，不计入有效传输
                    self.packets_expired += 1
                    self.priority_expired[packet.priority] += 1
                else:
                    # 未过期转发：计入有效传输与延迟统计
                    self.packets_transmitted += 1
                    self.total_delay += packet.delay
                    self.transmission_delays.append(packet.delay)
                    self.priority_transmission_delays[packet.priority].append(packet.delay)
                    self.priority_transmitted[packet.priority] += 1
                    if packet.delay <= packet.max_delay:
                        self.delay_satisfied_count += 1
                        self.delay_satisfied_by_priority[packet.priority] += 1
                    if packet.delay > packet.max_delay:
                        self.overtime_transmitted += 1
                    if packet.priority == 3:
                        self.high_priority_transmitted += 1
                        if self.high_priority_transmitted >= self.high_priority_target and not self.target_completed:
                            self.target_completed = True
                
            if idx in dropped_indices:
                self.packets_dropped += 1
                if packet.delay <= packet.max_delay:
                    self.advancedrop += 1
            del self.buffer[idx]

        # 延迟更新；不自动删除过期包，过期包继续留在队列中
        packets_expired_this_step = 0
        
        for packet in self.buffer:
            packet.update_delay()
            if packet.is_expired():
                packet.is_expired_flag = True
                packets_expired_this_step += 1

        # 新数据包到达
        packets_generated = self._generate_packets()

        # 注意：current_step已在step方法开始处更新，用于块衰落判断

        # 更新历史记录
        self.history["buffer_length"].append(len(self.buffer))
        self.history["packets_generated"].append(packets_generated)
        self.history["packets_transmitted"].append(packets_transmitted_this_step)
        self.history["packets_dropped"].append(packets_dropped_this_step)
        
        if self.packets_transmitted > 0:
            self.history["avg_delay"].append(self.total_delay / self.packets_transmitted)
            # 计算并记录延迟满足率
            delay_satisfied_ratio = self.delay_satisfied_count / self.packets_transmitted
            self.history["delay_satisfied_ratio"].append(delay_satisfied_ratio)
        else:
            self.history["avg_delay"].append(0)
            self.history["delay_satisfied_ratio"].append(0)

        # 记录本步的转发时延（step_delays 已在缓冲区更新前记录）
        self.step_transmission_delays.append(step_delays if packets_transmitted_this_step > 0 else [])

        # episode 终止条件：达到最大步数或转发次数限制
        if self.current_step >= self.max_steps:
            done = True
        elif self.max_transmissions is not None and self.transmission_count >= self.max_transmissions:
            # 达到转发次数限制时也结束episode
            done = True
            # 终止时的额外奖励：基于传输效率
            transmission_rate = self.packets_transmitted / (self.packets_transmitted + self.packets_dropped + 1)
            
            # 计算平均延迟利用率
            if self.packets_transmitted > 0:
                avg_delay_ratio = (self.total_delay / self.packets_transmitted) / 25
                delay_efficiency_bonus = avg_delay_ratio * 15
            else:
                delay_efficiency_bonus = 0
            
            reward += transmission_rate * 20 + delay_efficiency_bonus

        # 记录本步统计信息
        total_processed = self.packets_transmitted + self.packets_dropped
        packet_success_rate = (
            self.packets_transmitted / (total_processed + 1e-8)
            if total_processed > 0 else 0.0
        )
        info = {
            "packets_transmitted_this_step": packets_transmitted_this_step,
            "packets_dropped_this_step": packets_dropped_this_step,
            "packets_waited_this_step": packets_waited_this_step,
            "packets_expired_this_step": packets_expired_this_step,  # 本步队列中已过期的包数（不自动删除）
            "action": action_taken,
            "current_traffic_load": self._get_current_traffic_load(),
            "buffer_size": len(self.buffer),
            "priority_generated": self.priority_generated.copy(),
            "priority_transmitted": self.priority_transmitted.copy(),
            "total_packets_processed": self.packets_transmitted + self.packets_dropped + self.packets_expired,
            "channel_state": self.current_state,
            "current_snr": self.current_snr,
            "current_capacity": self.current_capacity,  # 当前信道容量(bits/s)
            "channel_max_packets": channel_maxnumber,
            "actual_transmitted": packets_transmitted_this_step,
            "advancedrop": self.advancedrop,
            "overtime_transmitted": self.overtime_transmitted,
            # 以下为 episode 级统计，供 Monitor 的 info_keywords 写入 monitor.csv（episode 结束时记录）
            "packets_transmitted": self.packets_transmitted,
            "packets_dropped": self.packets_dropped,
            "packet_success_rate": packet_success_rate,
        }

        # 获取观察值
        observation = self._get_observation()

        # 返回五元组
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info
    

    def generate_rayleigh_snr(self, num_samples, avg_snr_db=None):
        """
        生成瑞利衰落信道下的瞬时SNR值
        
        参数:
        - num_samples: 样本数量
        - avg_snr_db: 平均SNR(dB)，如果为None则使用self.avg_snr
        """
        if avg_snr_db is None:
            avg_snr_db = self.avg_snr
            
        # 将平均SNR从dB转换为线性值
        avg_snr_linear = 10**(avg_snr_db/10.0)
        
        # 在瑞利衰落下，瞬时SNR服从指数分布
        # 平均值为avg_snr_linear
        snr_linear = np.random.exponential(1/avg_snr_linear, num_samples)
        
        # 将SNR转换回dB值
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db

    def generate_correlated_snr(self, num_samples, correlation_factor=0.7):
        """
        生成具有时间相关性的SNR序列（保留用于兼容性）
        参数:
        - num_samples: 样本数量
        - correlation_factor: 相关性强度 (0-1)
        """
        # 生成独立的高斯噪声
        noise = np.random.normal(0, 3, num_samples)
        
        # AR(1)模型: SNR[t] = correlation_factor * SNR[t-1] + sqrt(1-correlation_factor^2) * noise[t]
        snr_sequence = np.zeros(num_samples)
        snr_sequence[0] = self.avg_snr + noise[0]
        
        for t in range(1, num_samples):
            snr_sequence[t] = (correlation_factor * snr_sequence[t-1] + 
                             np.sqrt(1 - correlation_factor**2) * noise[t])
        
        return snr_sequence

    def _update_next_snr(self):
        """
        瑞利衰落模型：使用相关的高斯过程生成复信道增益
        信道增益 h = h_real + j*h_imag，其中 h_real 和 h_imag 独立同分布
        |h| 服从瑞利分布
        """
        # 生成独立的高斯噪声
        noise_real = np.random.normal(0, self.rayleigh_sigma)
        noise_imag = np.random.normal(0, self.rayleigh_sigma)
        
        # 使用 AR(1) 过程更新复信道增益的实部和虚部，保持时间相关性
        # h_t = rho * h_{t-1} + sqrt(1-rho²) * noise
        self.h_real = (self.correlation_factor * self.h_real + 
                       np.sqrt(1 - self.correlation_factor**2) * noise_real)
        self.h_imag = (self.correlation_factor * self.h_imag + 
                       np.sqrt(1 - self.correlation_factor**2) * noise_imag)
        
        # 计算信道增益的幅度平方 |h|²
        self.channel_gain_squared = self.h_real**2 + self.h_imag**2
        
        # 计算 SNR（线性域）= (|h|² * P_t) / (N₀ * B)，然后转换为 dB
        snr_linear = (self.channel_gain_squared * self.transmit_power) / (self.noise_psd * self.bandwidth)
        self.current_snr = 10 * np.log10(snr_linear + 1e-10)  # 加小值防止 log10(0)
        
        # 物理限制：SNR不太可能小于-5dB或大于45dB，稍微截断一下防止数值爆炸
        self.current_snr = np.clip(self.current_snr, -5, 45)

    def calculate_channel_capacity(self, channel_gain_squared):
        """
        香农公式：C = B * log₂(1 + (|h|² * P_t) / (N₀ * B))
        参数：
            channel_gain_squared: |h|²，信道增益幅度的平方
        返回：
            capacity: 信道容量 (bps)
        """
        # 计算 SNR（线性域）= (|h|² * P_t) / (N₀ * B)
        snr_linear = (channel_gain_squared * self.transmit_power) / (self.noise_psd * self.bandwidth)
        # 香农公式
        capacity = self.bandwidth * np.log2(1 + snr_linear)
        return capacity

    def _map_snr_to_state(self, snr_db):
        """
        将SNR映射到信道状态（保留用于兼容性）
        """
        if snr_db < self.snr_thresholds[0]:
            return "deep_fade"      # 深度衰落
        elif snr_db < self.snr_thresholds[1]:
            return "poor"           # 差信道
        elif snr_db < self.snr_thresholds[2]:
            return "fair"           # 一般信道
        elif snr_db < self.snr_thresholds[3]:
            return "good"           # 良好信道
        else:
            return "excellent"      # 优质信道
    
    def _map_packets_to_state(self, num_packets):
        """
        根据可传输包数确定信道状态（新方法，来自xindao.py）
        包数阈值定义：
        - deep_fade: 0 到 packet_thresholds[0] 个包（默认0-1个）
        - poor: packet_thresholds[0]+1 到 packet_thresholds[1] 个包（默认2-3个）
        - fair: packet_thresholds[1]+1 到 packet_thresholds[2] 个包（默认4-5个）
        - good: packet_thresholds[2]+1 到 packet_thresholds[3] 个包（默认6-7个）
        - excellent: packet_thresholds[3]+1 个包及以上（默认8+个）
        """
        if num_packets <= self.packet_thresholds[0]:
            return "deep_fade"
        elif num_packets <= self.packet_thresholds[1]:
            return "poor"
        elif num_packets <= self.packet_thresholds[2]:
            return "fair"
        elif num_packets <= self.packet_thresholds[3]:
            return "good"
        else:
            return "excellent"


    def _get_current_traffic_load(self):
        """获取当前流量负载 - 与SAC环境保持一致"""
        if self.traffic_pattern == "constant":
            return 0.8
        elif self.traffic_pattern == "dynamic":
            cycle_position = self.current_step % self.traffic_cycle_length
            # 调整波动范围从0.7-1.0到0.8-1.2，平均约1.0，与constant模式更接近
            # 同时保持动态波动特性，让智能体学习适应流量变化
            return 0.8 + 0.4 * math.sin(2 * math.pi * cycle_position / self.traffic_cycle_length)
        elif self.traffic_pattern == "fluctuation":
            # 极端波动模式：在小速率(0.2)和大速率(2.0)之间剧烈波动
            # 高负载阶段：traffic_load=2.0，平均生成约 7.0 个包/步（超过信道容量5-6个/步，形成高负载）
            cycle_position = self.current_step % self.traffic_cycle_length
            # 使用方波实现急剧切换
            if (cycle_position % 50) < 25:  # 前25步为高流量
                return 1.0  # 高负载：平均生成约 7.0 个包/步
            else:  # 后25步为低流量
                return 0.5  # 低负载：平均生成约 0.7 个包/步
        elif self.traffic_pattern == "high load":
            return 1.0
        elif self.traffic_pattern == "low load":
            return 0.5

        elif self.traffic_pattern == "Pattern Shift":
            if (self.current_step < 750) or (self.current_step > 1500):
                return 0.5
            else:
                return 1.0
        elif self.traffic_pattern == "Extreme Congestion":
            return 1.0
        elif self.traffic_pattern == "Periodic Brust Traffic":
            if (self.current_step % self.traffic_cycle_length) < self.burst_duration:
                return 1.0
            else:
                return 0.5
        return 1.0
    
    def _generate_packets(self):
        """根据当前流量状态生成数据包 - 使用泊松分布（与SAC环境一致）"""
        traffic_load = self._get_current_traffic_load()
        packets_generated = 0
        
        # 生成三种优先级的数据包（与SAC环境保持一致）
        packets_generated += self._generate_high_priority_packets()
        packets_generated += self._generate_medium_priority_packets()
        packets_generated += self._generate_low_priority_packets()
        
        return packets_generated
    
    def _generate_high_priority_packets(self):
        """生成高优先级数据包"""
        traffic_load = self._get_current_traffic_load()
        packets_generated = 0
        
        base_lambda = 0.5 * traffic_load  # 高优先级生成率（从0.3提升到0.5，适应更高信道容量）
        cycle_length = 300
        cycle_position = self.current_step % cycle_length
        cycle_factor = 1.0 + 0.3 * math.sin(2 * math.pi * cycle_position / cycle_length)
        final_lambda = base_lambda * cycle_factor
        
        packets_to_generate = np.random.poisson(final_lambda)
        
        for _ in range(packets_to_generate):
            if len(self.buffer) < self.max_queue:
                packet = Packet(
                    sensor_id=0,  # 高优先级传感器ID
                    max_delay=5,
                    arrival_time=self.current_step,
                    priority=3
                )
                self.buffer.append(packet)
                self.priority_generated[3] += 1
                packets_generated += 1
        
        return packets_generated
    
    def _generate_medium_priority_packets(self):
        """生成中优先级数据包"""
        traffic_load = self._get_current_traffic_load()
        packets_generated = 0
        
        base_lambda = 1.0 * traffic_load  # 中优先级生成率（从0.6提升到1.0，适应更高信道容量）
        burst_cycle = 150
        burst_duration = 35
        
        cycle_position = self.current_step % burst_cycle
        
        if cycle_position < burst_duration:
            burst_factor = 2.0  # 突发因子
        else:
            burst_factor = 1.0
        
        final_lambda = base_lambda * burst_factor
        packets_to_generate = np.random.poisson(final_lambda)
        
        for _ in range(packets_to_generate):
            if len(self.buffer) < self.max_queue:
                packet = Packet(
                    sensor_id=1,  # 中优先级传感器ID
                    max_delay=10,
                    arrival_time=self.current_step,
                    priority=2
                )
                self.buffer.append(packet)
                self.priority_generated[2] += 1
                packets_generated += 1
        
        return packets_generated
    
    def _generate_low_priority_packets(self):
        """生成低优先级数据包"""
        traffic_load = self._get_current_traffic_load()
        packets_generated = 0
        
        base_lambda = 2.0 * traffic_load  # 低优先级生成率（从1.2提升到2.0，适应更高信道容量）
        stability_factor = 1.0 + 0.1 * (np.random.random() - 0.5)
        final_lambda = base_lambda * stability_factor
        
        packets_to_generate = np.random.poisson(final_lambda)
        
        for _ in range(packets_to_generate):
            if len(self.buffer) < self.max_queue:
                packet = Packet(
                    sensor_id=2,  # 低优先级传感器ID
                    max_delay=15,
                    arrival_time=self.current_step,
                    priority=1
                )
                self.buffer.append(packet)
                self.priority_generated[1] += 1
                packets_generated += 1
        
        return packets_generated
    
    def _get_observation(self):
        """获取当前状态的观察"""
        # 初始化观察数组
        obs = np.zeros(3 + self.max_queue, dtype=np.float32)
        
        
        obs[0] = len(self.buffer) / self.max_queue
        obs[1] = self.target_delay / 20.0  # 归一化目标延迟
        # 将所有延迟作为特征向量
        if len(self.buffer) > 0:
            delays = [packet.delay for packet in self.buffer]
            for i in range(min(len(delays), self.max_queue)):
                obs[2 + i] = delays[i]
        # 如果没有包，延迟位置保持0.0
        obs[2 + self.max_queue] = np.clip(self.current_capacity / 10, 0.0, 1.0)
        return obs
    
    def render(self, mode='human'):
        """渲染环境状态"""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Traffic load: {self._get_current_traffic_load():.2f}")
        print(f"Channel state: {self.current_state} (SNR: {self.current_snr:.2f}dB)")
        print(f"Buffer size: {len(self.buffer)}/{self.max_queue}")
        print(f"Packets transmitted: {self.packets_transmitted}")
        print(f"Packets dropped: {self.packets_dropped}")
        print(f"Packets expired: {self.packets_expired}")
        print(f"Total packets processed: {self.packets_transmitted + self.packets_dropped + self.packets_expired}")
        
        if self.packets_transmitted > 0:
            print(f"Average delay: {self.total_delay/self.packets_transmitted:.2f}")
            # 新增：显示延迟满足率
            delay_satisfied_ratio = self.delay_satisfied_count / self.packets_transmitted
            print(f"Delay satisfied ratio: {delay_satisfied_ratio:.1%}")
        
        # 修正传输率计算，包含超时包
        total_processed = self.packets_transmitted + self.packets_dropped + self.packets_expired
        if total_processed > 0:
            transmission_rate = self.packets_transmitted / total_processed
            print(f"Transmission rate: {transmission_rate:.3f}")
            print(f"Drop rate: {self.packets_dropped / total_processed:.3f}")
            print(f"Expired rate: {self.packets_expired / total_processed:.3f}")
        else:
            print("Transmission rate: N/A")
        
        # 新增：转发效率统计（修正版）
        if self.total_transmission_attempts > 0:
            # 真实转发效率：满足延迟要求的包数 / 总转发尝试次数
            real_transmission_efficiency = self.delay_satisfied_count / self.total_transmission_attempts
            # 成功转发率：成功转发的包数 / 总转发尝试次数
            success_rate = self.successful_transmissions / self.total_transmission_attempts
            # 失败转发率：失败转发的包数 / 总转发尝试次数
            failure_rate = self.failed_transmissions / self.total_transmission_attempts
            
            print(f"Real transmission efficiency: {real_transmission_efficiency:.3f} satisfied packets per attempt")
            print(f"Success rate: {success_rate:.3f}")
            print(f"Failure rate: {failure_rate:.3f}")
            if self.max_transmissions is not None:
                print(f"Transmission count: {self.transmission_count}/{self.max_transmissions}")
                print(f"Transmission remaining: {self.max_transmissions - self.transmission_count}")
        else:
            print("Transmission efficiency: N/A")
        
        # 新增：高优先级包转发目标测试结果
        print(f"\n高优先级包转发目标测试:")
        print(f"  目标: 转发{self.high_priority_target}个最高优先级包")
        print(f"  已完成: {self.high_priority_transmitted}个")
        print(f"  完成状态: {'已完成' if self.target_completed else '未完成'}")
        if self.target_completed:
            print(f"  完成所需转发次数: {self.transmissions_to_complete_target}")
        else:
            print(f"  当前已用转发次数: {self.transmissions_to_complete_target}")
        
        # 打印优先级传输统计
        print("Priority transmission stats:")
        for i in range(1, 4):
            print(f"  Priority {i}: {self.priority_transmitted[i]} packets")
        
        # 新增：打印优先级生成统计
        print("Priority generated stats:")
        for i in range(1, 4):
            print(f"  Priority {i}: {self.priority_generated[i]} packets")
        
        # 新增：打印优先级延迟满足率
        print("Priority delay satisfied stats:")
        for i in range(1, 4):
            if self.priority_transmitted[i] > 0:
                delay_satisfied_ratio = self.delay_satisfied_by_priority[i] / self.priority_transmitted[i]
                print(f"  Priority {i}: {self.delay_satisfied_by_priority[i]}/{self.priority_transmitted[i]} = {delay_satisfied_ratio:.1%}")
            else:
                print(f"  Priority {i}: 0/0 = N/A")
        
        # 新增：打印优先级传输成功率（修正计算）
        print("Priority success rates:")
        for i in range(1, 4):
            total_processed = self.priority_transmitted[i] + self.priority_expired[i]
            if total_processed > 0:
                success_rate = self.priority_transmitted[i] / total_processed
                print(f"  Priority {i}: {self.priority_transmitted[i]}/{total_processed} = {success_rate:.3f}")
                print(f"    Generated: {self.priority_generated[i]}, Transmitted: {self.priority_transmitted[i]}, Expired: {self.priority_expired[i]}")
            else:
                print(f"  Priority {i}: 0/0 = N/A")
        
        if len(self.buffer) > 0:
            print("\nQueue front packets:")
            for i in range(min(5, len(self.buffer))):
                p = self.buffer[i]
                delay_ratio = p.delay / p.max_delay
                status = "CRITICAL" if delay_ratio >= 0.9 else "WARNING" if delay_ratio >= 0.7 else "OK"
                print(f"  {i}: Sensor {p.sensor_id}, Priority {p.priority}, Delay {p.delay}/{p.max_delay} ({delay_ratio:.2f}) [{status}]")
        
        print("\n" + "-"*50)
        return None
    
    def plot_history(self, show=True):
        """绘制历史数据图表"""
        plt.figure(figsize=(15, 12))  # 增加图表大小以容纳更多子图
        
        # 绘制缓冲区长度
        plt.subplot(3, 2, 1)  # 修改为3行2列的布局
        plt.plot(self.history["buffer_length"])
        plt.title("Buffer Length")
        plt.xlabel("Steps")
        plt.ylabel("Length")
        plt.grid(True, alpha=0.3)
        
        # 绘制包生成和传输对比
        plt.subplot(3, 2, 2)
        plt.plot(self.history["packets_generated"], label="Generated")
        plt.plot(self.history["packets_transmitted"], label="Transmitted")
        plt.title("Packets Generated vs Transmitted")
        plt.xlabel("Steps")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制平均延迟
        plt.subplot(3, 2, 3)
        plt.plot(self.history["avg_delay"])
        plt.title("Average Delay")
        plt.xlabel("Steps")
        plt.ylabel("Delay")
        plt.grid(True, alpha=0.3)
        
        # 绘制丢弃的包数量
        plt.subplot(3, 2, 4)
        plt.plot(self.history["packets_dropped"])
        plt.title("Packets Dropped")
        plt.xlabel("Steps")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        # 新增：绘制延迟满足率
        plt.subplot(3, 2, 5)
        plt.plot(self.history["delay_satisfied_ratio"])
        plt.title("Delay Satisfied Ratio")
        plt.xlabel("Steps")
        plt.ylabel("Ratio")
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        
        # 新增：绘制优先级延迟满足率饼图
        plt.subplot(3, 2, 6)
        if self.packets_transmitted > 0:
            labels = ['Priority 1', 'Priority 2', 'Priority 3']
            sizes = []
            for i in range(1, 4):
                if self.priority_transmitted[i] > 0:
                    sizes.append(self.delay_satisfied_by_priority[i] / self.priority_transmitted[i])
                else:
                    sizes.append(0)
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title("Delay Satisfied Ratio by Priority")
        else:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.title("Delay Satisfied Ratio by Priority")
        
        plt.tight_layout()
        
        if show:
            plt.savefig(f"dynamic_traffic_history_{self.traffic_pattern}.png")
            plt.show()
        
        return plt.gcf()
    
    def get_transmission_delay_stats(self):
        """获取转发时延统计信息"""
        if not self.transmission_delays:
            return {
                'total_transmissions': 0,
                'avg_delay': 0,
                'min_delay': 0,
                'max_delay': 0,
                'std_delay': 0,
                'priority_stats': {
                    1: {'count': 0, 'avg_delay': 0, 'min_delay': 0, 'max_delay': 0, 'std_delay': 0},
                    2: {'count': 0, 'avg_delay': 0, 'min_delay': 0, 'max_delay': 0, 'std_delay': 0},
                    3: {'count': 0, 'avg_delay': 0, 'min_delay': 0, 'max_delay': 0, 'std_delay': 0}
                }
            }
        
        stats = {
            'total_transmissions': len(self.transmission_delays),
            'avg_delay': np.mean(self.transmission_delays),
            'min_delay': np.min(self.transmission_delays),
            'max_delay': np.max(self.transmission_delays),
            'std_delay': np.std(self.transmission_delays),
            'priority_stats': {}
        }
        
        # 按优先级统计
        for priority in [1, 2, 3]:
            delays = self.priority_transmission_delays[priority]
            if delays:
                stats['priority_stats'][priority] = {
                    'count': len(delays),
                    'avg_delay': np.mean(delays),
                    'min_delay': np.min(delays),
                    'max_delay': np.max(delays),
                    'std_delay': np.std(delays)
                }
            else:
                stats['priority_stats'][priority] = {
                    'count': 0,
                    'avg_delay': 0,
                    'min_delay': 0,
                    'max_delay': 0,
                    'std_delay': 0
                }
        
        return stats

    def print_transmission_delay_stats(self):
        """打印转发时延统计信息"""
        stats = self.get_transmission_delay_stats()
    
        print("\n" + "="*60)
        print("数据包转发时延统计")
        print("="*60)
    
        print(f"总转发数量: {stats['total_transmissions']}")
        print(f"平均转发时延: {stats['avg_delay']:.2f}")
        print(f"最小转发时延: {stats['min_delay']}")
        print(f"最大转发时延: {stats['max_delay']}")
        print(f"转发时延标准差: {stats['std_delay']:.2f}")
    
        print("\n按优先级统计:")
        for priority in [1, 2, 3]:
            p_stats = stats['priority_stats'][priority]
            print(f"  优先级 {priority}:")
            print(f"    转发数量: {p_stats['count']}")
            print(f"    平均时延: {p_stats['avg_delay']:.2f}")
            print(f"    最小时延: {p_stats['min_delay']}")
            print(f"    最大时延: {p_stats['max_delay']}")
            print(f"    时延标准差: {p_stats['std_delay']:.2f}")
    
    print("="*60)

    def plot_transmission_delays(self, show=True):
        """绘制转发时延分布图"""
        if not self.transmission_delays:
            print("没有转发数据包，无法绘制时延分布图")
            return
        
        plt.figure(figsize=(15, 10))
        
        # 1. 总体时延分布直方图
        plt.subplot(2, 2, 1)
        plt.hist(self.transmission_delays, bins=20, alpha=0.7, edgecolor='black')
        plt.title('总体转发时延分布')
        plt.xlabel('转发时延')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        # 2. 按优先级时延分布
        plt.subplot(2, 2, 2)
        for priority in [1, 2, 3]:
            delays = self.priority_transmission_delays[priority]
            if delays:
                plt.hist(delays, bins=10, alpha=0.6, label=f'优先级 {priority}')
        plt.title('按优先级转发时延分布')
        plt.xlabel('转发时延')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 时延随时间变化
        plt.subplot(2, 2, 3)
        step_avg_delays = [np.mean(delays) if delays else 0 for delays in self.step_transmission_delays]
        plt.plot(step_avg_delays)
        plt.title('每步平均转发时延变化')
        plt.xlabel('步数')
        plt.ylabel('平均转发时延')
        plt.grid(True, alpha=0.3)
        
        # 4. 时延箱线图
        plt.subplot(2, 2, 4)
        priority_delays = [self.priority_transmission_delays[p] for p in [1, 2, 3]]
        priority_labels = ['优先级 1', '优先级 2', '优先级 3']
        plt.boxplot(priority_delays, labels=priority_labels)
        plt.title('按优先级转发时延箱线图')
        plt.ylabel('转发时延')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.savefig('transmission_delay_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return plt.gcf()
    

def test_dqn(model_path, traffic_pattern="dynamic", max_steps=10000, render_interval=100, results_dir=None):
    """
    测试DQN算法在指定流量模式下的性能
    
    参数:
        model_path: DQN模型路径
        traffic_pattern: 流量模式 ("dynamic", "constant", "burst")
        max_steps: 最大步数
        render_interval: 渲染间隔
        results_dir: 结果保存目录
    
    返回:
        stats: 统计信息字典
        env: 环境对象
        df_delay: 包含所有数据包延迟信息的DataFrame
    """
    # 加载DQN模型
    model = DQN.load(model_path)
    
    # 创建环境
    env = DynamicTrafficUAVEnv(
        traffic_pattern=traffic_pattern,
        max_steps=max_steps,
        max_transmissions=100  # 测试时限制转发次数为100次
    )
    
    # 重置环境
    obs, _ = env.reset()
    
    # 统计信息
    stats = {
        "rewards": [],
        "transmitted": [],
        "dropped": [],
        "avg_delay": [],
        "buffer_length": [],
        "transmission_rate": [],
        "energy_efficiency": [],
        "priority_transmitted": [0, 0, 0, 0],  # 按优先级统计
        "priority_generated": [0, 0, 0, 0],     # 按优先级统计生成的包数
        "priority_expired": [0, 0, 0, 0],       # 按优先级统计超时的包数
        "priority_success_rate": [0, 0, 0, 0],  # 按优先级统计传输成功率
        "delay_satisfied_count": 0,             # 满足延迟要求的数据包数量
        "delay_satisfied_ratio": 0.0,           # 延迟满足率
        "step_rewards": [],
        "action_distribution": [0, 0, 0]  # 转发、等待、丢弃的次数
    }
    
    # 新增：用于记录所有数据包延迟信息的列表
    packet_delay_data = []
    
    # 运行环境
    done = False
    total_reward = 0
    step = 0
    
    while not done:
        # 使用DQN模型预测动作
        action, _states = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 更新统计信息
        total_reward += reward
        stats["step_rewards"].append(reward)
        
        # 记录动作分布
        stats["action_distribution"][action] += 1
        
        # 新增：记录当前步骤中所有数据包的延迟信息
        for packet in env.buffer:
            packet_delay_data.append({
                'step': step,
                'sensor_id': packet.sensor_id,
                'priority': packet.priority,
                'current_delay': packet.delay,
                'max_delay': packet.max_delay,
                'delay_ratio': packet.delay / packet.max_delay,
                'traffic_pattern': traffic_pattern,
                'algorithm': 'DQN',
                'delay_satisfied': packet.delay <= packet.max_delay  # 新增：是否满足延迟要求
            })
        
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
    stats["expired"].append(env.packets_expired)  # 新增：记录超时包数
    stats["avg_delay"].append(env.total_delay / max(1, env.packets_transmitted))
    stats["buffer_length"].append(np.mean(env.history["buffer_length"]))
    
    # 修正传输率计算，包含超时包
    total_processed = env.packets_transmitted + env.packets_dropped + env.packets_expired
    stats["transmission_rate"].append(env.packets_transmitted / max(1, total_processed))
    
    # 计算步数效率（替代能量效率）
    stats["energy_efficiency"].append(env.packets_transmitted / max(1, env.current_step))
    
    # 记录优先级传输统计
    for i in range(1, 4):
        stats["priority_transmitted"][i] = env.priority_transmitted[i]
        stats["priority_expired"][i] = env.priority_expired[i]
    
    # 新增：记录优先级生成统计和传输成功率
    for i in range(1, 4):
        stats["priority_generated"][i] = env.priority_generated[i]
        
        # 计算传输成功率（修正：基于实际处理的包数）
        total_processed = stats["priority_transmitted"][i] + stats["priority_expired"][i]
        if total_processed > 0:
            success_rate = stats["priority_transmitted"][i] / total_processed
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
            delay_csv_path = os.path.join(results_dir, f"dqn_delay_data_{traffic_pattern}.csv")
            df_delay.to_csv(delay_csv_path, index=False)
            print(f"延迟数据已保存到: {delay_csv_path}")
    
    # 打印最终结果
    print("\n" + "="*50)
    print(f"DQN Algorithm Results ({traffic_pattern} traffic):")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Packets transmitted: {env.packets_transmitted}")
    print(f"Packets dropped: {env.packets_dropped}")
    print(f"Packets expired: {env.packets_expired}")
    print(f"Total processed: {env.packets_transmitted + env.packets_dropped + env.packets_expired}")
    print(f"Transmission rate: {stats['transmission_rate'][0]:.3f}")
    print(f"Drop rate: {env.packets_dropped / max(1, env.packets_transmitted + env.packets_dropped + env.packets_expired):.3f}")
    print(f"Expired rate: {env.packets_expired / max(1, env.packets_transmitted + env.packets_dropped + env.packets_expired):.3f}")
    print(f"Average delay: {stats['avg_delay'][0]:.2f}")
    print(f"Step efficiency: {stats['energy_efficiency'][0]:.3f} packets/step")
    print(f"Priority transmission: P1={stats['priority_transmitted'][1]}, P2={stats['priority_transmitted'][2]}, P3={stats['priority_transmitted'][3]}")
    
    # 新增：转发效率统计（修正版）
    if env.total_transmission_attempts > 0:
        real_transmission_efficiency = env.delay_satisfied_count / env.total_transmission_attempts
        success_rate = env.successful_transmissions / env.total_transmission_attempts
        failure_rate = env.failed_transmissions / env.total_transmission_attempts
        
        print(f"Real transmission efficiency: {real_transmission_efficiency:.3f} satisfied packets per attempt")
        print(f"Success rate: {success_rate:.3f}")
        print(f"Failure rate: {failure_rate:.3f}")
        print(f"Transmission count: {env.transmission_count}/{env.max_transmissions}")
    else:
        print("Transmission efficiency: N/A")
    
    # 新增：高优先级包转发目标测试结果
    print(f"\n高优先级包转发目标测试:")
    print(f"  目标: 转发{env.high_priority_target}个最高优先级包")
    print(f"  已完成: {env.high_priority_transmitted}个")
    print(f"  完成状态: {'已完成' if env.target_completed else '未完成'}")
    if env.target_completed:
        print(f"  完成所需转发次数: {env.transmissions_to_complete_target}")
    else:
        print(f"  当前已用转发次数: {env.transmissions_to_complete_target}")
    
    # 新增：打印延迟满足率
    print(f"Delay satisfied ratio: {stats['delay_satisfied_ratio']:.1%}")
    
    # 新增：打印优先级传输成功率（修正计算）
    print(f"\nPriority transmission success rates:")
    for i in range(1, 4):
        generated = stats["priority_generated"][i]
        transmitted = stats["priority_transmitted"][i]
        expired = stats["priority_expired"][i]
        total_processed = transmitted + expired
        
        if total_processed > 0:
            success_rate = transmitted / total_processed
            print(f"  Priority {i}: {transmitted}/{total_processed} = {success_rate:.3f}")
            print(f"    Generated: {generated}, Transmitted: {transmitted}, Expired: {expired}")
        else:
            print(f"  Priority {i}: 0/0 = N/A")
    
    # 添加调试信息
    print(f"\nDebug info:")
    print(f"  Total packets generated: {sum(stats['priority_generated'][1:])}")
    print(f"  Total packets transmitted: {sum(stats['priority_transmitted'][1:])}")
    print(f"  Overall transmission rate: {sum(stats['priority_transmitted'][1:]) / max(1, sum(stats['priority_generated'][1:])):.3f}")
    
    # 打印动作分布
    total_actions = sum(stats["action_distribution"])
    if total_actions > 0:
        print("Action distribution:")
        print(f"  Forward: {stats['action_distribution'][0]} ({stats['action_distribution'][0]/total_actions:.1%})")
        print(f"  Wait:    {stats['action_distribution'][1]} ({stats['action_distribution'][1]/total_actions:.1%})")
        print(f"  Drop:    {stats['action_distribution'][2]} ({stats['action_distribution'][2]/total_actions:.1%})")
    
    print("="*50)
    
    # 绘制历史数据
    history_fig = env.plot_history(show=False)
    if results_dir and history_fig:
        history_fig.savefig(os.path.join(results_dir, f"dqn_history_{traffic_pattern}.png"), dpi=300)
        plt.close(history_fig)
    
    return stats, env, df_delay


# 简单测试环境
if __name__ == "__main__":
    # 创建环境
    env = DynamicTrafficUAVEnv(traffic_pattern="dynamic", max_steps=10000)
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(200):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        if step % 20 == 0:  # 每20步打印一次状态
            print(f"\nStep {step+1}")
            print(f"Traffic load: {['Low', 'Medium', 'High'][info['current_traffic_load']]}")
            print(f"Buffer size: {info['buffer_size']}")
            print(f"Packets transmitted: {info['packets_transmitted_this_step']}")
            print(f"Packets dropped: {info['packets_dropped_this_step']}")
            print(f"Packets waited: {info['packets_waited_this_step']}")
            print(f"Reward: {reward:.2f}")
        
        if done:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Total packets transmitted: {env.packets_transmitted}")
    print(f"Total packets dropped: {env.packets_dropped}")
    print(f"Total packets expired: {env.packets_expired}")
    total_processed = env.packets_transmitted + env.packets_dropped + env.packets_expired
    print(f"Transmission rate: {env.packets_transmitted / max(1, total_processed):.3f}")
    print(f"Drop rate: {env.packets_dropped / max(1, total_processed):.3f}")
    print(f"Expired rate: {env.packets_expired / max(1, total_processed):.3f}")
    
    # 绘制历史数据
    env.plot_history() 
