import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import math
from collections import deque
import matplotlib.pyplot as plt

class Packet:
    def __init__(self, priority, arrival_time, max_delay):
        self.priority = priority
        self.arrival_time = arrival_time
        self.max_delay = max_delay
        self.delay = 0
        self.is_expired = False  # 添加过期标记

    def update_delay(self):
        self.delay += 1
        
    def check_expired(self):
        """检查是否过期，但不移除"""
        if self.delay > self.max_delay:
            self.is_expired = True
        return self.is_expired
    
    def get_urgency(self):
        """计算包的紧急度（优先级和延迟的组合）"""
        delay_ratio = self.delay / self.max_delay
        return self.priority * (1 + delay_ratio)

class RayleighAdaptiveRelay:
    def __init__(self, noise_power=1.0, snr_thresholds=[6, 11, 16, 21], 
                 bandwidth=1.15e4, time_slot_duration=1.0, correlation_factor=0.7,
                 packet_thresholds=[1, 3, 5, 7], transmit_power=1.0, noise_psd=1e-4):
        self.noise_power = noise_power
        self.snr_thresholds = snr_thresholds  # 保留用于兼容性，但不再用于状态判断
        self.packet_thresholds = packet_thresholds  # 包数阈值：[deep_fade上限, poor上限, fair上限, good上限]
        self.bandwidth = bandwidth
        self.avg_snr = 14.5  # 平均SNR dB（平衡：既有差状态，又保持平均5个包）
        self.packet_size_bits = 8000
        self.time_slot_duration = time_slot_duration
        
        # 香农公式参数：C = B * log₂(1 + (|h|² * P_t) / (N₀ * B))
        self.transmit_power = transmit_power  # P_t: 发射功率 (W)
        self.noise_psd = noise_psd  # N₀: 噪声功率谱密度 (W/Hz)
        
        # 信道相关性参数
        self.correlation_factor = correlation_factor
        self.coherence_steps = 3  # 块衰落：每多少步更新一次信道 (减少周期以增加状态变化)
        
        # 瑞利衰落参数
        # 根据平均SNR计算瑞利分布的尺度参数
        # 平均SNR（线性域）= (|h|² * P_t) / (N₀ * B)
        # 对于瑞利衰落，E[|h|²] = 2*sigma²
        # 因此：avg_snr_linear = (2*sigma² * P_t) / (N₀ * B)
        # 所以：sigma² = (avg_snr_linear * N₀ * B) / (2 * P_t)
        avg_snr_linear = 10**(self.avg_snr / 10.0)
        self.rayleigh_sigma = np.sqrt((avg_snr_linear * self.noise_psd * self.bandwidth) / (2.0 * self.transmit_power))
        
        # 状态初始化
        self.current_step = 0
        self.snr_history = []
        self.capacity_history = []
        self.current_snr = self.avg_snr
        # 瑞利衰落的复信道增益（实部和虚部）
        self.h_real = np.random.normal(0, self.rayleigh_sigma)
        self.h_imag = np.random.normal(0, self.rayleigh_sigma)
        # 信道增益幅度平方 |h|²
        self.channel_gain_squared = self.h_real**2 + self.h_imag**2
        
        # 预先计算初始状态
        self._update_channel_metrics()

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

    def calculate_packet_count(self, channel_gain_squared):
        """计算最大可传输包数"""
        capacity_bits = self.calculate_channel_capacity(channel_gain_squared)
        eta = 0.7 
        effective_bits = capacity_bits * eta * self.time_slot_duration
        num_packets = int(effective_bits / self.packet_size_bits)
        # 至少保证能传0个（防止出现负数，虽然不太可能）
        return max(0, num_packets)

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
        
        # 物理限制：SNR不太可能小于-10dB或大于50dB，稍微截断一下防止数值爆炸
        self.current_snr = np.clip(self.current_snr, -5, 45)

    def _map_snr_to_state(self, snr_db):
        """根据SNR映射状态（保留用于兼容性）"""
        if snr_db < self.snr_thresholds[0]: return "deep_fade"
        elif snr_db < self.snr_thresholds[1]: return "poor"
        elif snr_db < self.snr_thresholds[2]: return "fair"
        elif snr_db < self.snr_thresholds[3]: return "good"
        else: return "excellent"
    
    def _map_packets_to_state(self, num_packets):
        """
        根据可传输包数确定信道状态（新方法）
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

    def _update_channel_metrics(self):
        """统一更新所有信道指标"""
        # 使用香农公式：C = B * log₂(1 + (|h|² * P_t) / (N₀ * B))
        self.current_capacity = self.calculate_channel_capacity(self.channel_gain_squared)
        self.current_max_packets_number = self.calculate_packet_count(self.channel_gain_squared)
        # 根据包数确定状态（而不是SNR）
        self.current_state = self._map_packets_to_state(self.current_max_packets_number)

    def reset(self):
        self.current_step = 0
        self.snr_history = []
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
        
        self._update_channel_metrics()
        
        # 填充初始历史，避免一开始是空的
        for _ in range(10):
            self.snr_history.append(self.current_snr)
            self.capacity_history.append(self.current_capacity)
            
        return self.current_state

    def step(self):
        self.current_step += 1
        
        # 实现块衰落：只有当 step 是 coherence_steps 的倍数时才改变信道
        # 如果你想每一步都变，把 self.coherence_steps 设为 1
        if self.current_step % self.coherence_steps == 0:
            self._update_next_snr()
            self._update_channel_metrics()
        
        # 记录历史
        self.snr_history.append(self.current_snr)
        self.capacity_history.append(self.current_capacity)
        
        # 维护历史长度（滑动窗口）
        if len(self.snr_history) > 10:
            self.snr_history.pop(0)
            self.capacity_history.pop(0)
            
        return self.current_state, self.current_snr, self.current_capacity, self.current_max_packets_number
    

class DynamicTrafficUAVEnv(gym.Env):
    def __init__(self, max_queue=150, max_steps=10000, traffic_pattern="dynamic", max_transmissions=None,
                 bandwidth=1.15e4, packet_size_bits=8000, time_slot_duration=1.0, transmission_efficiency=0.7,
                 correlation_factor=0.7, coherence_steps=3, packet_thresholds=[1, 3, 5, 7], noise_std=7.0,
                 transmit_power=1.0, noise_psd=1e-4, transmit_threshold=0.5, reward_scale=0.01):
        super(DynamicTrafficUAVEnv, self).__init__()
        
        # 环境参数
        self.max_queue = max_queue
        self.transmit_threshold = transmit_threshold  # wait_tendency < 此值时判定为传输
        self.reward_scale = reward_scale  # 奖励缩放，过小会导致梯度信号弱、难收敛
        self.max_steps = max_steps
        self.traffic_pattern = traffic_pattern
        
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
        self.noise_std = noise_std  # 噪声标准差（控制SNR波动幅度，已弃用，保留用于兼容性）
        
        # 传感器配置
        self.sensors = [
            {"id": 0, "base_rate": 0.05, "max_delay": 5, "priority": 3},   # 高优先级：降低生成率
            {"id": 1, "base_rate": 0.1, "max_delay": 10, "priority": 2},  # 中优先级：降低生成率
            {"id": 2, "base_rate": 0.2, "max_delay": 15, "priority": 1}   # 低优先级：降低生成率
        ]
        
        # 流量模式
        self.traffic_cycle_length = 100
        self.burst_duration = 20
        
        # 缓冲区
        self.buffer = deque(maxlen=max_queue)
        
        # 统计
        self.packets_transmitted = 0
        self.packets_dropped = 0
        self.packets_expired = 0
        self.total_delay = 0
        self.priority_transmitted = [0, 0, 0, 0]
        self.priority_generated = [0, 0, 0, 0]
        self.delay_satisfied_count = 0
        self.delay_satisfied_by_priority = [0, 0, 0, 0]
        self.priority_delay = [0, 0, 0, 0]  # 添加每个优先级的延迟统计
        self.packets_generated_sum = 0
        
        # 添加转发日志记录
        self.transmission_log = []  # 记录每次转发的详细信息
        
        # 补充传输机制
        self.supplement_history = deque(maxlen=10)  # 最近10步的补充记录
        self.consecutive_supplements = 0  # 连续补充次数
        
        # 能量相关
        self.energy_consumed = 0
        self.transmission_count = 0
        self.max_transmissions = max_transmissions

        # 新增：信道状态模拟器（使用瑞利衰落模型）
        self.noise_power = 1.0
        self.snr_thresholds = [8, 13, 18, 23]  # 保留用于兼容性，但不再用于状态判断
        self.avg_snr = 14.5  # 平均SNR (dB)（与xindao.py保持一致）
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
        
        # 历史SNR序列长度（用于捕捉时序相关性）
        self.snr_history_length = 5  # 保留过去5个时刻的SNR值
        
        # 块衰落相关：记录距离下次信道更新的步数
        self.steps_until_channel_update = 0  # 距离下次信道更新的步数
        
        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # 观察空间：5
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),  # 5维：缓冲区大小、高优先级延迟、中优先级延迟、低优先级延迟、信道容量
            dtype=np.float32
        )
        

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.buffer.clear()
        self.current_step = 0
        self.packets_transmitted = 0
        self.packets_dropped = 0
        self.packets_expired = 0
        self.total_delay = 0
        self.priority_transmitted = [0, 0, 0, 0]
        self.priority_generated = [0, 0, 0, 0]
        self.delay_satisfied_count = 0
        self.delay_satisfied_by_priority = [0, 0, 0, 0]
        self.priority_delay = [0, 0, 0, 0]  # 重置每个优先级的延迟统计
        self.energy_consumed = 0
        self.transmission_count = 0
        self.packets_generated_sum = 0
        
        # 重置转发日志
        self.transmission_log = []
        
        # 重置补充传输机制
        self.supplement_history.clear()
        self.consecutive_supplements = 0

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
        # 初始化SNR历史，用当前SNR填充
        self.snr_history = [self.current_snr] * self.snr_history_length
        # 重置块衰落计数器
        self.steps_until_channel_update = 0
        
        # 初始生成一些数据包
        self._generate_packets()
        
        return self._get_observation(), {}

    def step(self, action):
        """执行一步（SACAQM：动作解析 + 动态评分填满信道 + 平滑奖励）"""
        # 先更新步数和块衰落信道状态（保留原有逻辑）
        self.current_step += 1
        if self.current_step % self.coherence_steps == 0:
            self._update_next_snr()
            self.current_capacity = self.calculate_channel_capacity(self.channel_gain_squared)
            effective_bits = self.current_capacity * self.transmission_efficiency * self.time_slot_duration
            channel_maxnumber = int(effective_bits / self.packet_size_bits)
            self.current_state = self._map_packets_to_state(channel_maxnumber)
            self.steps_until_channel_update = 0
        else:
            self.steps_until_channel_update = self.coherence_steps - (self.current_step % self.coherence_steps)

        # 更新 SNR / 信道容量历史
        self.snr_history.append(self.current_snr)
        if len(self.snr_history) > self.snr_history_length:
            self.snr_history.pop(0)
        self.capacity_history.append(self.current_capacity)
        if len(self.capacity_history) > 10:
            self.capacity_history.pop(0)

        # 确保信道容量在合理范围内
        effective_bits = self.current_capacity * self.transmission_efficiency * self.time_slot_duration
        channel_maxnumber = int(effective_bits / self.packet_size_bits)
        channel_maxnumber = max(1, min(channel_maxnumber, 20))

        # ==========================================
        # 重构一：连续且平滑的动作映射
        # ==========================================
        # 将 [0, 1] 的网络输出映射到更有区分度的范围
        energy_threshold = float(action[0]) * 10.0  # 能量门限（综合得分达到此值才“值得传”）
        w_priority = float(action[1]) * 10.0        # 优先级权重
        w_urgency = float(action[2]) * 10.0         # 紧迫度权重

        reward = 0.0
        energy_consumed_this_step = 0
        successful_transmissions = 0

        # ==========================================
        # 重构二：门限过滤与动态调度（消除“传/不传”硬开关）
        # ==========================================
        # candidates：超过能量门限的包（决定“是否值得耗能”）
        candidates = []
        # all_valid：所有未过期包（用于一旦耗能就“补齐容量”，避免传不满）
        all_valid = []
        if len(self.buffer) > 0:
            for p in self.buffer:
                if p.is_expired:
                    continue
                delay_ratio = p.delay / p.max_delay
                p.dynamic_score = w_priority * p.priority + w_urgency * delay_ratio
                all_valid.append(p)
                if p.dynamic_score >= energy_threshold:
                    candidates.append(p)

        # 先用“超过门限”的包决定是否触发传输
        candidates.sort(key=lambda p: getattr(p, 'dynamic_score', 0), reverse=True)
        packets_to_transmit = candidates[:channel_maxnumber]

        # 一旦触发传输（packets_to_transmit 非空），就用剩余容量从“未过期包”中按分数补齐，尽量用满信道
        if packets_to_transmit and len(packets_to_transmit) < channel_maxnumber and all_valid:
            chosen = set(id(p) for p in packets_to_transmit)
            all_valid.sort(key=lambda p: getattr(p, 'dynamic_score', 0), reverse=True)
            for p in all_valid:
                if len(packets_to_transmit) >= channel_maxnumber:
                    break
                if id(p) in chosen:
                    continue
                packets_to_transmit.append(p)
                chosen.add(id(p))

        num_packets = len(packets_to_transmit)

        # 强制限制：如果达到最大转发次数，禁止传输
        can_transmit = not (self.max_transmissions is not None and self.transmission_count >= self.max_transmissions)
        if not can_transmit:
            packets_to_transmit = []
            num_packets = 0

        if num_packets > 0:
            energy_consumed_this_step = 1.0
            self.energy_consumed += 1.0
            self.transmission_count += 1
            reward -= 2.0  # 基础传输成本

            # 按「当前时延/目标时延」比例连续给奖，无硬阈值；鼓励适当等待后再发
            base_reward = {3: 6.0, 2: 3.0, 1: 1.0}
            for packet in packets_to_transmit:
                delay_ratio = min(packet.delay / packet.max_delay, 1.0)  # 封顶 1，不奖励超时
                reward += base_reward.get(packet.priority, 1.0) * delay_ratio

                # 记录转发日志（沿用字段形状）
                self.transmission_log.append({
                    'step': self.current_step,
                    'packet_id': id(packet),
                    'priority': packet.priority,
                    'delay': packet.delay,
                    'max_delay': packet.max_delay,
                    'delay_ratio': packet.delay / packet.max_delay,
                    'arrival_time': packet.arrival_time,
                    'is_delay_satisfied': packet.delay <= packet.max_delay,
                    'action_values': {
                        'energy_threshold': energy_threshold,
                        'w_priority': w_priority,
                        'w_urgency': w_urgency,
                    }
                })

                self.packets_transmitted += 1
                self.total_delay += packet.delay
                self.priority_transmitted[packet.priority] += 1
                self.priority_delay[packet.priority] += packet.delay
                if packet.delay <= packet.max_delay:
                    self.delay_satisfied_count += 1
                    self.delay_satisfied_by_priority[packet.priority] += 1
                successful_transmissions += 1
                self.buffer.remove(packet)
        else:
            # 队列非空但没有包超过能量门限：静默；或队列为空：同样走该分支
            queue_ratio = len(self.buffer) / self.max_queue
            reward -= 2.0 * (queue_ratio ** 2)

        # ==========================================
        # 核心三：延迟更新与平滑的超时惩罚
        # ==========================================
        expired_count = 0
        for packet in list(self.buffer):
            packet.update_delay()
            if packet.check_expired():
                if packet.priority == 3:
                    reward -= 4.0
                elif packet.priority == 2:
                    reward -= 2.0
                else:
                    reward -= 1.0

                self.packets_expired += 1
                expired_count += 1
                self.buffer.remove(packet)

        # 生成新包并检查终止条件
        self.packets_generated_sum += self._generate_packets()
        done = (
            self.current_step >= self.max_steps
            or (self.max_transmissions is not None and self.transmission_count >= self.max_transmissions)
        )

        info = {
            "packets_transmitted_this_step": successful_transmissions,
            "packets_expired_this_step": expired_count,
            "packets_waited_this_step": 0,
            "energy_consumed_this_step": energy_consumed_this_step,
            "total_energy_consumed": self.energy_consumed,
            "transmission_action_taken": energy_consumed_this_step > 0,
            "buffer_size": len(self.buffer),
            "total_packets_expired": self.packets_expired,
            "priority_generated": self.priority_generated.copy(),
            "priority_transmitted": self.priority_transmitted.copy(),
            "channel_max_packets": channel_maxnumber,
            "channel_utilization": successful_transmissions / max(channel_maxnumber, 1),
        }

        return self._get_observation(), reward * self.reward_scale, done, False, info


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


    def _generate_packets(self):
        """生成数据包"""
        packets_generated = 0
        
        # 生成三种优先级的数据包
        packets_generated += self._generate_high_priority_packets()
        packets_generated += self._generate_medium_priority_packets()
        packets_generated += self._generate_low_priority_packets()
        
        return packets_generated

    def _generate_high_priority_packets(self):
        """生成高优先级数据包"""
        traffic_load = self._get_current_traffic_load()
        packets_generated = 0
        
        base_lambda = 0.5 * traffic_load  # 提高高优先级生成率（从0.3提升到0.5，适应更高信道容量）
        cycle_length = 300
        cycle_position = self.current_step % cycle_length
        cycle_factor = 1.0 + 0.3 * math.sin(2 * math.pi * cycle_position / cycle_length)
        final_lambda = base_lambda * cycle_factor
        
        packets_to_generate = np.random.poisson(final_lambda)
        
        for _ in range(packets_to_generate):
            if len(self.buffer) < self.max_queue:
                packet = Packet(
                    priority=3,
                    arrival_time=self.current_step,
                    max_delay=5
                )
                self.buffer.append(packet)
                self.priority_generated[3] += 1
                packets_generated += 1
        
        return packets_generated

    def _generate_medium_priority_packets(self):
        """生成中优先级数据包"""
        traffic_load = self._get_current_traffic_load()
        packets_generated = 0
        
        base_lambda = 1.0 * traffic_load  # 提高中优先级生成率（从0.6提升到1.0，适应更高信道容量）
        burst_cycle = 150
        burst_duration = 35
        
        cycle_position = self.current_step % burst_cycle
        
        if cycle_position < burst_duration:
            burst_factor = 2.0  # 降低突发因子
        else:
            burst_factor = 1.0
        
        final_lambda = base_lambda * burst_factor
        packets_to_generate = np.random.poisson(final_lambda)
        
        for _ in range(packets_to_generate):
            if len(self.buffer) < self.max_queue:
                packet = Packet(
                    priority=2,
                    arrival_time=self.current_step,
                    max_delay=10
                )
                self.buffer.append(packet)
                self.priority_generated[2] += 1
                packets_generated += 1
        
        return packets_generated

    def _generate_low_priority_packets(self):
        """生成低优先级数据包"""
        traffic_load = self._get_current_traffic_load()
        packets_generated = 0
        
        base_lambda = 2.0 * traffic_load  # 提高低优先级生成率（从1.2提升到2.0，适应更高信道容量）
        stability_factor = 1.0 + 0.1 * (np.random.random() - 0.5)
        final_lambda = base_lambda * stability_factor
        
        packets_to_generate = np.random.poisson(final_lambda)
        
        for _ in range(packets_to_generate):
            if len(self.buffer) < self.max_queue:
                packet = Packet(
                    priority=1,
                    arrival_time=self.current_step,
                    max_delay=15
                )
                self.buffer.append(packet)
                self.priority_generated[1] += 1
                packets_generated += 1
        
        return packets_generated

    def _get_observation(self):
        """观测：各优先级缓冲包数(0-2)、各优先级最紧急包延迟比(3-5)、信道容量归一化(6)。
        注：第6维是信道容量(bps)归一化，与「本时隙可传包数」成比例但非同一量纲。"""
        obs = np.zeros(7, dtype=np.float32)
        idx = 0
        # 各优先级未过期包数量（归一化到 max_queue）
        for priority in [3, 2, 1]:
            count = len([p for p in self.buffer if p.priority == priority and not p.is_expired])
            obs[idx] = count / self.max_queue
            idx += 1
        
        
        for priority in [3, 2, 1]:
            priority_packets = [p for p in self.buffer if p.priority == priority and not p.is_expired]
            if priority_packets:
                max_delay_packet = max(priority_packets, key=lambda p: p.delay / p.max_delay)
                obs[idx] = max_delay_packet.delay / max_delay_packet.max_delay
            else:
                obs[idx] = 0.0
            idx += 1
        
        capacity_max = 10
        obs[idx] = np.clip(self.current_capacity / capacity_max, 0.0, 1.0)
        idx += 1    

        return obs

    def _get_current_traffic_load(self):
        """获取当前流量负载"""
        if self.traffic_pattern == "constant":
            return 0.8
        elif self.traffic_pattern == "dynamic":
            cycle_position = self.current_step % self.traffic_cycle_length
            return 0.8 + 0.4 * math.sin(2 * math.pi * cycle_position / self.traffic_cycle_length)
        elif self.traffic_pattern == "burst":
            if (self.current_step % self.traffic_cycle_length) < self.burst_duration:
                return 1.5
            else:
                return 0.5 
        elif self.traffic_pattern == "fluctuation":
            # 极端波动模式：在小速率(0.2)和大速率(2.0)之间剧烈波动
            # 高负载阶段：traffic_load=2.0，平均生成约 7.0 个包/步（超过信道容量5-6个/步，形成高负载）
            cycle_position = self.current_step % self.traffic_cycle_length
            # 使用方波实现急剧切换
            if (cycle_position % 50) < 25:  # 前25步为高流量
                return 1.0  # 高负载：平均生成约 7.0 个包/步
            else:  # 后25步为低流量
                return 0.5  # 低负载：平均生成约 0.7 个包/步
        
        
        elif self.traffic_pattern == "extreme_fluctuation1":
            return 1.0  # 高负载模式：平均生成约 2.9 个包/步
        elif self.traffic_pattern == "extreme_fluctuation2":
            return 0.5  # 低负载模式：平均生成约 0.45 个包/步

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

        elif self.traffic_pattern == "low_load":
            return 0.3
        elif self.traffic_pattern == "medium_load":
            return 0.7
        elif self.traffic_pattern == "high_load":
            return 1.0

        return 1.0

    
    def _select_supplement_packets(self, agent_transmitted_count):
        """选择补充传输的包"""
        queue_ratio = len(self.buffer) / self.max_queue
        
        # 检查是否应该触发补充传输
        if not self._should_trigger_supplement(agent_transmitted_count, queue_ratio):
            return []
        
        supplement_packets = []
        
        # 按优先级顺序选择补充包（包括超时包）
        for priority in [3, 2, 1]:  # 高优先级优先
            priority_packets = [p for p in self.buffer 
                              if p.priority == priority 
                              and p.delay/p.max_delay > 0.7]  # 延迟超过一半（包括超时包）
            
            # 按延迟比例排序，优先选择延迟大的
            priority_packets.sort(key=lambda p: p.delay/p.max_delay, reverse=True)
            
            # 每个优先级最多补充5个包
            max_per_priority = 5
            supplement_packets.extend(priority_packets[:max_per_priority])
        
        # 限制总补充数量
        max_total = 15  # 最多补充15个包（3个优先级 × 5个包）
        return supplement_packets[:max_total]
    
    

    def render(self, mode='human'):
        """渲染环境状态"""
        print(f"Step: {self.current_step}")
        print(f"Buffer size: {len(self.buffer)}/{self.max_queue}")
        print(f"Energy consumed: {self.energy_consumed}")
        print(f"Packets transmitted: {self.packets_transmitted}")
        print(f"Packets expired: {self.packets_expired}")
        if self.packets_transmitted > 0:
            print(f"Average delay: {self.total_delay/self.packets_transmitted:.2f}")
            delay_satisfied_ratio = self.delay_satisfied_count / self.packets_transmitted
            print(f"Delay satisfied ratio: {delay_satisfied_ratio:.1%}")
        
        print("Priority transmission stats:")
        for i in range(1, 4):
            print(f"  Priority {i}: {self.priority_transmitted[i]} packets")
        
        print("Priority generated stats:")
        for i in range(1, 4):
            print(f"  Priority {i}: {self.priority_generated[i]} packets")
        
        print("-" * 50)
    
    def save_transmission_log(self, filename):
        """保存转发日志到文件"""
        import json
        import pandas as pd
        from datetime import datetime
        
        if not self.transmission_log:
            print("没有转发日志可保存")
            return
        
        # 保存为JSON格式（详细记录）
        json_filename = filename.replace('.csv', '.json')
        
        # 转换numpy类型为Python原生类型，以便JSON序列化
        json_log = []
        for record in self.transmission_log:
            json_record = {
                'step': int(record['step']),
                'packet_id': int(record['packet_id']),
                'priority': int(record['priority']),
                'delay': float(record['delay']),
                'max_delay': float(record['max_delay']),
                'delay_ratio': float(record['delay_ratio']),
                'arrival_time': int(record['arrival_time']),
                'is_delay_satisfied': bool(record['is_delay_satisfied']),
                'action_values': {
                    'wait_tendency': float(record['action_values']['wait_tendency']),
                    'delay_threshold': float(record['action_values']['delay_threshold']),
                    'priority_weight': float(record['action_values']['priority_weight'])
                }
            }
            json_log.append(json_record)
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_log, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV格式（便于分析）
        df_data = []
        for record in self.transmission_log:
            row = {
                'step': record['step'],
                'packet_id': record['packet_id'],
                'priority': record['priority'],
                'delay': record['delay'],
                'max_delay': record['max_delay'],
                'delay_ratio': record['delay_ratio'],
                'arrival_time': record['arrival_time'],
                'is_delay_satisfied': record['is_delay_satisfied'],
                'wait_tendency': record['action_values']['wait_tendency'],
                'delay_threshold': record['action_values']['delay_threshold'],
                'priority_weight': record['action_values']['priority_weight']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"转发日志已保存到:")
        print(f"  JSON格式: {json_filename}")
        print(f"  CSV格式: {filename}")
        print(f"  总转发记录数: {len(self.transmission_log)}")
        
        # 打印统计信息
        if len(self.transmission_log) > 0:
            print(f"\n转发统计:")
            print(f"  平均延迟比例: {df['delay_ratio'].mean():.3f}")
            print(f"  延迟满足率: {df['is_delay_satisfied'].mean():.3f}")
            print(f"  优先级分布: {df['priority'].value_counts().to_dict()}")
            
            # 按优先级统计
            for priority in [1, 2, 3]:
                priority_data = df[df['priority'] == priority]
                if len(priority_data) > 0:
                    print(f"  优先级{priority}: 平均延迟比例={priority_data['delay_ratio'].mean():.3f}, "
                          f"延迟满足率={priority_data['is_delay_satisfied'].mean():.3f}")

# 为了兼容性，保留原来的类名作为别名
EnergyAwareQueueEnv = DynamicTrafficUAVEnv

# 简单测试环境
if __name__ == "__main__":
    # 创建环境
    env = DynamicTrafficUAVEnv(traffic_pattern="constant")
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
            print(f"Buffer size: {info['buffer_size']}")
            print(f"Packets transmitted: {info['packets_transmitted_this_step']}")
            print(f"Packets expired: {info['packets_expired_this_step']}")
            print(f"Reward: {reward:.2f}")
        
        if done:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Total packets transmitted: {env.packets_transmitted}")
    print(f"Total packets expired: {env.packets_expired}")
    print(f"Transmission rate: {env.packets_transmitted / (env.packets_transmitted + env.packets_expired + 1):.3f}")