import numpy as np
import matplotlib.pyplot as plt

# 从环境参数中获取的值
bandwidth = 1e4  # 10 kHz
packet_size_bits = 8000  # 8000 bits per packet
avg_snr = 18.0  # 平均SNR 18 dB
snr_thresholds = [8, 13, 18, 23]  # SNR阈值

def calculate_channel_capacity(snr_db, bandwidth):
    """使用香农公式计算信道容量"""
    snr_linear = 10**(snr_db / 10.0)
    capacity = bandwidth * np.log2(1 + snr_linear)
    return capacity

def calculate_packets(capacity, packet_size_bits):
    """计算可传输的包数"""
    packets = int(capacity / packet_size_bits)
    # 应用限制：最小1，最大20
    packets = max(1, min(packets, 20))
    return packets

# 计算不同SNR下的信道容量和包数
print("=" * 80)
print("信道容量与数据包数量计算")
print("=" * 80)
print(f"带宽: {bandwidth/1e6:.1f} MHz")
print(f"包大小: {packet_size_bits} bits")
print(f"平均SNR: {avg_snr} dB")
print()

# 计算SNR阈值点
print("SNR阈值点对应的信道容量和包数:")
print("-" * 80)
for i, snr in enumerate(snr_thresholds):
    capacity = calculate_channel_capacity(snr, bandwidth)
    packets = calculate_packets(capacity, packet_size_bits)
    state_names = ["deep_fade", "poor", "fair", "good", "excellent"]
    if i < len(state_names) - 1:
        print(f"SNR = {snr:5.1f} dB ({state_names[i]} → {state_names[i+1]}): "
              f"容量 = {capacity/1e6:7.3f} Mbps, "
              f"包数 = {packets:2d} 包/秒")
    else:
        print(f"SNR = {snr:5.1f} dB (>{state_names[i]}): "
              f"容量 = {capacity/1e6:7.3f} Mbps, "
              f"包数 = {packets:2d} 包/秒")

print()

# 计算平均SNR下的值
avg_capacity = calculate_channel_capacity(avg_snr, bandwidth)
avg_packets = calculate_packets(avg_capacity, packet_size_bits)
print(f"平均SNR ({avg_snr} dB) 对应的值:")
print(f"  容量 = {avg_capacity/1e6:.3f} Mbps")
print(f"  包数 = {avg_packets} 包/秒")
print()

# 计算SNR范围（考虑瑞利衰落和相关性）
# 根据代码，SNR可能在一个范围内波动
print("SNR范围分析（考虑瑞利衰落）:")
print("-" * 80)

# 生成一系列SNR值来计算范围
snr_range = np.arange(0, 35, 0.5)  # 从0到35dB，步长0.5
capacities = [calculate_channel_capacity(snr, bandwidth) for snr in snr_range]
packets_list = [calculate_packets(cap, packet_size_bits) for cap in capacities]

# 找到包数变化的临界点
unique_packets = sorted(set(packets_list))
print(f"可传输包数范围: {min(unique_packets)} - {max(unique_packets)} 包/秒")
print()

# 显示包数变化对应的SNR范围
print("包数对应的SNR范围:")
print("-" * 80)
for pkt_count in unique_packets:
    # 找到该包数对应的SNR范围
    snr_for_pkt = [snr_range[i] for i, p in enumerate(packets_list) if p == pkt_count]
    min_snr = min(snr_for_pkt)
    max_snr = max(snr_for_pkt)
    capacity_at_min = calculate_channel_capacity(min_snr, bandwidth)
    capacity_at_max = calculate_channel_capacity(max_snr, bandwidth)
    print(f"{pkt_count:2d} 包/秒: SNR范围 [{min_snr:5.1f}, {max_snr:5.1f}] dB, "
          f"容量范围 [{capacity_at_min/1e6:6.3f}, {capacity_at_max/1e6:6.3f}] Mbps")

print()
print("=" * 80)
print("详细计算过程:")
print("=" * 80)
print("香农公式: C = B × log₂(1 + SNR)")
print(f"其中 B = {bandwidth/1e6:.1f} MHz")
print()
print("示例计算（SNR = 18 dB）:")
snr_example = 18.0
snr_linear_example = 10**(snr_example / 10.0)
capacity_example = bandwidth * np.log2(1 + snr_linear_example)
packets_example = capacity_example / packet_size_bits
print(f"  SNR (线性) = 10^(18/10) = {snr_linear_example:.2f}")
print(f"  容量 = {bandwidth/1e6:.1f} × 10⁶ × log₂(1 + {snr_linear_example:.2f}) = {capacity_example/1e6:.3f} Mbps")
print(f"  理论包数 = {capacity_example/1e6:.3f} × 10⁶ / 8000 = {packets_example:.2f} 包/秒")
print(f"  实际包数（取整并限制） = {calculate_packets(capacity_example, packet_size_bits)} 包/秒")

