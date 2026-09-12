import os
import csv
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from dqn_dynamic_uav import DynamicTrafficUAVEnv
import matplotlib.pyplot as plt


class EpisodeSuccessRateLogger(BaseCallback):
    """
    在每次 episode 结束时，将 packets_transmitted / packets_dropped / packet_success_rate
    写入单独 CSV 文件（不写 monitor.csv），便于 plot_packet_success_rate 读取。
    单环境与向量化环境均支持，数据来自 step 返回的 infos。
    """
    def __init__(self, log_dir: str, filename: str = "episode_success_rate.csv", verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_dir = os.path.abspath(log_dir)
        self.filepath = os.path.join(self.log_dir, filename)
        self.file_handle = None
        self.writer = None
        self.episode_count = 0

    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self.file_handle = open(self.filepath, "w", newline="\n", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file_handle,
            fieldnames=["episode", "packets_transmitted", "packets_dropped", "packet_success_rate"],
        )
        self.writer.writeheader()
        self.file_handle.flush()
        print(f"[EpisodeSuccessRateLogger] 成功率将写入: {self.filepath}")

    def _on_step(self) -> bool:
        if self.writer is None:
            return True
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])
        if dones is None:
            return True
        if not getattr(dones, "__len__", None):
            dones = [dones]
        if not getattr(infos, "__len__", None):
            infos = [infos] if infos else [{}]
        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            tx = int(info.get("packets_transmitted", 0))
            drop = int(info.get("packets_dropped", 0))
            rate = info.get("packet_success_rate")
            if rate is None:
                total = tx + drop
                rate = (tx / total) if total > 0 else 0.0
            else:
                rate = float(rate)
            self.episode_count += 1
            self.writer.writerow({
                "episode": self.episode_count,
                "packets_transmitted": tx,
                "packets_dropped": drop,
                "packet_success_rate": round(rate, 6),
            })
        if dones.any() if hasattr(dones, "any") else any(dones):
            self.file_handle.flush()
        return True

    def _on_training_end(self) -> None:
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except Exception:
                pass
            self.file_handle = None
            self.writer = None


def make_env(traffic_pattern, rank, seed=0):
    """
    创建带有不同流量模式和随机种子的环境（用于向量化并行环境）
    """
    def _init():
        env = DynamicTrafficUAVEnv(traffic_pattern=traffic_pattern)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_dqn(traffic_pattern="dynamic", total_timesteps=500000, eval_freq=5000, use_mixed_env=False):
    """
    训练DQN智能体以适应动态流量环境

    参数:
        traffic_pattern: 流量模式/标签（单环境训练时生效，例如 "dynamic", "constant", "burst"）
        total_timesteps: 总训练步数
        eval_freq: 评估频率（每多少步评估一次）
        use_mixed_env: 是否使用多流量模式的向量化环境进行训练
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建日志目录
    log_dir = f"dynamic_uav_logs/{traffic_pattern}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # 创建模型保存目录
    model_dir = f"dynamic_uav_models/{traffic_pattern}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # 创建 TensorBoard 日志目录
    tb_log_dir = f"dynamic_uav_tensorboard/{traffic_pattern}_{timestamp}"
    os.makedirs(tb_log_dir, exist_ok=True)

    # 创建环境
    if use_mixed_env:
        # 使用多种流量模式构建向量化环境，与 SAC 训练一致
        env_patterns = ["dynamic", "extreme_fluctuation_1", "extreme_fluctuation_2"]
        env_fns = [make_env(pattern, i) for i, pattern in enumerate(env_patterns)]
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env, filename=log_dir)
    else:
        env = DynamicTrafficUAVEnv(traffic_pattern=traffic_pattern)
        env = Monitor(env, log_dir)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建 DQN 模型
    policy_kwargs = dict(
        net_arch=[256, 256, 128]
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200000,
        learning_starts=10000,
        batch_size=128,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log=tb_log_dir,
        policy_kwargs=policy_kwargs,
        device=device
    )

    # 创建检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="dqn_dynamic_uav"
    )

    # 创建评估回调（固定单一评估环境作为标准）
    eval_env = DynamicTrafficUAVEnv(traffic_pattern="dynamic")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=os.path.join(log_dir, "evaluations"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    callbacks = [checkpoint_callback, eval_callback, EpisodeSuccessRateLogger(log_dir)]
    callback_list = CallbackList(callbacks)

    # 打印训练信息
    print(f"\n{'='*80}")
    print(f"开始训练 {traffic_pattern} 流量模式下的 DQN 智能体")
    print(f"{'='*80}")
    print(f"总训练步数: {total_timesteps:,}")
    print(f"评估频率: 每 {eval_freq:,} 步评估一次")
    print(f"TensorBoard 日志目录: {os.path.abspath(tb_log_dir)}")
    print(f"模型保存目录: {os.path.abspath(model_dir)}")
    print(f"{'='*80}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=10,
        progress_bar=True
    )

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "dqn_dynamic_uav_final")
    model.save(final_model_path)

    print(f"\n{'='*80}")
    print("训练完成！")
    print(f"{'='*80}")
    print(f"最终模型已保存到: {final_model_path}")
    print(f"最佳模型已保存到: {os.path.join(model_dir, 'best_model')}")
    print(f"\nTensorBoard 查看命令:")
    print(f"  tensorboard --logdir={tb_log_dir}")
    print(f"{'='*80}\n")

    return model, env, final_model_path

def plot_training_rewards(log_dir):
    """
    绘制训练过程中的奖励曲线
    
    参数:
        log_dir: 日志目录
    """
    # 读取监控日志
    monitor_file = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_file):
        print(f"找不到监控日志文件: {monitor_file}")
        return
    
    try:
        # 使用pandas读取CSV文件，更可靠
        import pandas as pd
        
        # 读取CSV文件，跳过前两行（注释行）
        data = pd.read_csv(monitor_file, skiprows=2, names=['r', 'l', 't'])
        
        # 检查数据是否为空
        if len(data) == 0:
            print("监控日志文件为空，无法绘制奖励曲线")
            return
        
        print(f"数据形状: {data.shape}")
        print(f"列名: {data.columns.tolist()}")
        
        # 获取奖励数据
        rewards = data['r'].values
        print(f"奖励数据范围: {rewards.min():.2f} 到 {rewards.max():.2f}")
        
        # 绘制奖励曲线
        plt.figure(figsize=(12, 8))
        
        # 子图1：原始奖励曲线
        plt.subplot(2, 1, 1)
        plt.plot(rewards)
        plt.title("训练奖励曲线 (原始)")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        
        # 子图2：移动平均奖励曲线
        plt.subplot(2, 1, 2)
        window_size = min(100, len(rewards) // 10)  # 移动平均窗口大小
        if window_size > 1:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg)
            plt.title(f"train reward (pingjun move, window={window_size})")
        else:
            plt.plot(rewards)
            plt.title("训练奖励曲线")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "training_rewards.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print(f"\n训练统计信息:")
        print(f"总episodes: {len(rewards)}")
        print(f"平均奖励: {rewards.mean():.2f}")
        print(f"最终奖励: {rewards[-1]:.2f}")
        print(f"最高奖励: {rewards.max():.2f}")
        print(f"最低奖励: {rewards.min():.2f}")
        
        # 分析奖励趋势
        if len(rewards) > 10:
            recent_avg = rewards[-10:].mean()
            early_avg = rewards[:10].mean()
            improvement = recent_avg - early_avg
            print(f"最近10个episodes平均奖励: {recent_avg:.2f}")
            print(f"前10个episodes平均奖励: {early_avg:.2f}")
            print(f"奖励改善: {improvement:.2f}")
            
    except ImportError:
        # 如果没有pandas，使用numpy的替代方法
        print("pandas不可用，使用numpy替代方法...")
        
        # 手动读取CSV文件
        with open(monitor_file, 'r') as f:
            lines = f.readlines()
        
        # 跳过前两行，解析数据
        data_lines = lines[2:]  # 跳过注释行和列名行
        rewards = []
        
        for line in data_lines:
            if line.strip():  # 跳过空行
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    try:
                        reward = float(parts[0])
                        rewards.append(reward)
                    except ValueError:
                        continue
        
        if len(rewards) == 0:
            print("无法解析奖励数据")
            return
        
        rewards = np.array(rewards)
        print(f"数据形状: {rewards.shape}")
        print(f"奖励数据范围: {rewards.min():.2f} 到 {rewards.max():.2f}")
        
        # 绘制奖励曲线
        plt.figure(figsize=(12, 8))
        
        # 子图1：原始奖励曲线
        plt.subplot(2, 1, 1)
        plt.plot(rewards)
        plt.title("训练奖励曲线 (原始)")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        
        # 子图2：移动平均奖励曲线
        plt.subplot(2, 1, 2)
        window_size = min(100, len(rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg)
            plt.title(f"训练奖励曲线 (移动平均, 窗口={window_size})")
        else:
            plt.plot(rewards)
            plt.title("训练奖励曲线")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "training_rewards.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print(f"\n训练统计信息:")
        print(f"总episodes: {len(rewards)}")
        print(f"平均奖励: {rewards.mean():.2f}")
        print(f"最终奖励: {rewards[-1]:.2f}")
        print(f"最高奖励: {rewards.max():.2f}")
        print(f"最低奖励: {rewards.min():.2f}")
        
    except Exception as e:
        print(f"绘制奖励曲线时出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        
        # 尝试直接读取文件内容来调试
        try:
            with open(monitor_file, 'r') as f:
                lines = f.readlines()
                print(f"\n文件前10行内容:")
                for i, line in enumerate(lines[:10]):
                    print(f"第{i+1}行: {line.strip()}")
                print(f"总行数: {len(lines)}")
        except Exception as e2:
            print(f"读取文件内容时出错: {e2}")

def compare_traffic_patterns():
    """
    比较不同流量模式下的训练效果
    """
    # 训练不同流量模式下的模型
    patterns = ["constant", "dynamic", "burst"]
    models = {}
    
    for pattern in patterns:
        print(f"\n{'='*50}")
        print(f"训练 {pattern} 流量模式")
        print(f"{'='*50}")
        model, env, model_path = train_dqn(traffic_pattern=pattern, total_timesteps=300000)
        models[pattern] = {
            "model": model,
            "env": env,
            "model_path": model_path
        }
    
    # 比较结果
    print("\n比较不同流量模式下的训练结果:")
    for pattern, data in models.items():
        env = data["env"]
        print(f"\n{pattern} 流量模式:")
        print(f"  - 传输率: {env.get_wrapper_attr('packets_transmitted') / (env.get_wrapper_attr('packets_transmitted') + env.get_wrapper_attr('packets_dropped') + 1):.3f}")
        print(f"  - 平均奖励: {np.mean(env.get_wrapper_attr('rewards')):.2f}")
    
    return models

if __name__ == "__main__":
    # 训练混合流量模式（多环境：dynamic / extreme_fluctuation_1 / extreme_fluctuation_2），与 SAC 一致
    model, env, model_path = train_dqn(
        traffic_pattern="constant",
        total_timesteps=2000000,
        eval_freq=5000,
        use_mixed_env=True
    )

    # 绘制训练奖励曲线
    log_dir = "dynamic_uav_logs"
    if os.path.exists(log_dir):
        subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if subdirs:
            latest_dir = max(subdirs, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
            plot_training_rewards(os.path.join(log_dir, latest_dir))

    # 单环境训练示例:
    # model, env, model_path = train_dqn(traffic_pattern="dynamic", total_timesteps=800000)
    # 比较不同流量模式:
    # models = compare_traffic_patterns() 