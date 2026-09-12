import os
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    BaseCallback, 
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from dynamic_traffic_uav import DynamicTrafficUAVEnv
import matplotlib.pyplot as plt


class TemperatureScheduleCallback(BaseCallback):
    """
    温度参数调度回调：训练初期使用较高温度（增加探索），逐渐降低温度（专注利用）
    """
    def __init__(self, initial_temp=0.3, final_temp=0.1, total_steps=500000, verbose=0):
        super(TemperatureScheduleCallback, self).__init__(verbose)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_steps = total_steps
        self.current_temp = initial_temp
        self.update_freq = 1000  # 每1000步更新一次温度参数
        
    def _on_step(self) -> bool:
        # 每update_freq步更新一次温度参数
        if self.num_timesteps % self.update_freq == 0:
            # 线性衰减温度参数
            progress = min(self.num_timesteps / self.total_steps, 1.0)
            self.current_temp = self.initial_temp - (self.initial_temp - self.final_temp) * progress
            
            # 更新模型的温度参数
            try:
                if hasattr(self.model, 'ent_coef'):
                    if hasattr(self.model.ent_coef, 'data'):
                        self.model.ent_coef.data.fill_(self.current_temp)
                    else:
                        self.model.ent_coef = self.current_temp
                    
                    if self.verbose > 0 and self.num_timesteps % (self.update_freq * 10) == 0:
                        print(f"Step {self.num_timesteps}: 温度参数更新为 {self.current_temp:.3f}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"警告: 无法更新温度参数: {e}")
        
        return True


def make_env(traffic_pattern, rank, seed=0, max_steps=3000, reward_scale=0.1):
    """创建带指定流量模式和种子的环境（用于向量化并行）。"""
    def _init():
        env = DynamicTrafficUAVEnv(
            traffic_pattern=traffic_pattern,
            max_steps=max_steps,
            reward_scale=reward_scale,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train_sac(traffic_pattern="dynamic", total_timesteps=500000, eval_freq=5000, use_mixed_env=False,
              seed=None, log_dir=None, model_dir=None, tb_log_dir=None,
              max_steps=3000, reward_scale=0.1, gamma=0.995, learning_rate=1e-4):
    """
    训练SAC智能体以适应动态流量环境。

    若 eval/mean_reward 曲线不收敛、剧烈震荡，常见原因与对策：
    1) 奖励缩放过小：env 内 reward*0.01 会使梯度信号很弱 → 使用 reward_scale=0.1（或更大）
    2) episode 过长：max_steps=10000 导致单 episode 方差大、信用分配难 → 训练时用较短 max_steps（如 3000）
    3) gamma 过小：长 horizon 下 0.99 对远期回报几乎为 0 → 可试 gamma=0.995
    4) 学习率偏大：高方差环境下易震荡 → 可试 learning_rate=1e-4
    5) 环境随机性：流量与信道随机导致同一策略回报波动大 → 多跑几次或略增 eval 的 n_eval_episodes

    多流量模式 (use_mixed_env=True) 下的训练流程简述：
    1. 创建 n 个子环境（如 3 个），每个子环境一种流量模式，分别在独立进程中运行（SubprocVecEnv）。
    2. 每一轮训练：SAC 对当前 n 个观测各选一个动作，vec_env.step(actions) 一次会在 n 个环境中并行各执行一步，
       得到 n 份 (obs, reward, done, info)，即一次 step 产生 n 个“环境步”，这些转移都写入同一个 replay buffer。
    3. total_timesteps 表示“环境步”总数；若 n_envs=3，则实际 step 调用次数约 total_timesteps/3，但总经验量约 total_timesteps 条。
    4. SAC 从 buffer 中随机抽样做梯度更新，因此策略会同时学习多种流量模式，提升泛化。
    5. 评估仍用单一环境（如 dynamic）的 EvalCallback，便于用同一标准比较不同训练配置。

    参数:
        traffic_pattern: 流量模式/标签（单环境训练时生效，例如 "dynamic", "constant", "burst"）
        total_timesteps: 总训练步数（环境步）
        eval_freq: 评估频率（每多少步评估一次）
        use_mixed_env: 是否使用多流量模式的向量化环境进行训练
        seed: 随机种子（用于可复现；None 则不固定）
        max_steps: 每个 episode 最大步数，较短(如3000)利于收敛与信用分配
        reward_scale: 环境奖励缩放，过小(如0.01)易导致梯度信号弱
        gamma: 折扣因子，长 episode 时可适当提高(如0.995)
        learning_rate: 学习率，收敛不稳时可降低(如1e-4)
        log_dir / model_dir / tb_log_dir: 日志与模型目录

    返回:
        model, env, final_model_path, log_dir
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if log_dir is None:
        log_dir = f"SAC_dynamic_uav_logs/{traffic_pattern}_{timestamp}"
    if model_dir is None:
        model_dir = f"SAC_dynamic_uav_models/{traffic_pattern}_{timestamp}"
    if tb_log_dir is None:
        tb_log_dir = f"dynamic_uav_tensorboard/{traffic_pattern}_{timestamp}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # 创建环境（较短 max_steps + 较大 reward_scale 有利于收敛）
    env_kw = dict(max_steps=max_steps, reward_scale=reward_scale)
    if use_mixed_env:
        env_patterns = ["dynamic", "extreme_fluctuation_1", "extreme_fluctuation_2"]
        base_seed = (seed if seed is not None else 0)
        env_fns = [
            make_env(p, i, seed=base_seed + i, max_steps=max_steps, reward_scale=reward_scale)
            for i, p in enumerate(env_patterns)
        ]
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env, filename=log_dir)
    else:
        env = DynamicTrafficUAVEnv(traffic_pattern=traffic_pattern, **env_kw)
        if seed is not None:
            env.reset(seed=seed)
        env = Monitor(env, log_dir)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建SAC模型
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # 策略网络
            qf=[256, 256, 128]   # Q函数网络
        )
    )
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=500000,
        learning_starts= 5000,  # 至少 10% 用于 warm start
        batch_size=512,
        tau=0.005,
        gamma=gamma,
        train_freq=1,
        gradient_steps=2,  # 每步多一次更新，提高样本利用率
        action_noise=None,
        ent_coef='auto',
        target_update_interval=1,
        verbose=1,
        tensorboard_log=tb_log_dir,
        policy_kwargs=policy_kwargs,
        device=device,
    )
    
    # 创建检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="sac_dynamic_uav"
    )
    
    # 评估环境：与训练同 max_steps/reward_scale，便于对比
    eval_env = DynamicTrafficUAVEnv(traffic_pattern="dynamic", **env_kw)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=os.path.join(log_dir, "evaluations"),
        eval_freq=eval_freq,  # 每eval_freq步评估一次
        deterministic=True,
        render=False,
        n_eval_episodes=10,  # 每次评估运行5个episode
        verbose=1  # 显示评估信息
    )
    
    # 注意：当ent_coef='auto'时，不需要手动温度调度回调
    # 因为SAC会自动调整温度参数
    
    # 组合所有回调
    callback_list = CallbackList([
        checkpoint_callback,
        eval_callback
    ])
    
    # 打印训练信息
    print(f"\n{'='*80}")
    print(f"开始训练 {traffic_pattern} 流量模式下的SAC智能体")
    print(f"{'='*80}")
    print(f"总训练步数: {total_timesteps:,}")
    print(f"评估频率: 每 {eval_freq:,} 步评估一次")
    print(f"max_steps={max_steps}, reward_scale={reward_scale}, gamma={gamma}, lr={learning_rate}")
    print(f"温度参数: auto (自动调整)")
    print(f"TensorBoard日志目录: {os.path.abspath(tb_log_dir)}")
    print(f"模型保存目录: {os.path.abspath(model_dir)}")
    print(f"{'='*80}\n")
    
    # 开始训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=10,  # 每10步记录一次到TensorBoard
        progress_bar=True  # 显示进度条
    )
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "sac_dynamic_uav_final")
    model.save(final_model_path)
    
    print(f"\n{'='*80}")
    print(f"训练完成！")
    print(f"{'='*80}")
    print(f"最终模型已保存到: {final_model_path}")
    print(f"最佳模型已保存到: {os.path.join(model_dir, 'best_model')}")
    print(f"\nTensorBoard查看命令:")
    print(f"  tensorboard --logdir={tb_log_dir}")
    print(f"{'='*80}\n")
    
    return model, env, final_model_path, log_dir


def plot_training_rewards(log_dir):
    """
    绘制训练过程中的奖励曲线
    """
    monitor_file = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_file):
        print(f"找不到监控日志文件: {monitor_file}")
        return
    
    # 读取监控日志
    data = np.genfromtxt(monitor_file, delimiter=',', skip_header=2, names=True)
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(data['r'])
    plt.title("训练奖励曲线")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "training_rewards.png"))
    print(f"奖励曲线已保存到: {os.path.join(log_dir, 'training_rewards.png')}")


def compare_traffic_patterns():
    """
    比较不同流量模式下的训练效果
    """
    patterns = ["constant", "dynamic", "burst"]
    models = {}
    
    for pattern in patterns:
        print(f"\n{'='*80}")
        print(f"训练 {pattern} 流量模式")
        print(f"{'='*80}")
        model, env, model_path, _ = train_sac(traffic_pattern=pattern, total_timesteps=300000)
        models[pattern] = {
            "model": model,
            "env": env,
            "model_path": model_path
        }
    
    # 比较结果
    print(f"\n{'='*80}")
    print("不同流量模式下的训练结果比较:")
    print(f"{'='*80}")
    for pattern, data in models.items():
        env = data["env"]
        env_unwrapped = env.env if hasattr(env, 'env') else env
        print(f"\n{pattern} 流量模式:")
        if hasattr(env_unwrapped, 'packets_transmitted'):
            total = env_unwrapped.packets_transmitted + env_unwrapped.packets_dropped
            if total > 0:
                tx_rate = env_unwrapped.packets_transmitted / total
                print(f"  - 传输率: {tx_rate:.3f}")
                print(f"  - 传输包数: {env_unwrapped.packets_transmitted}")
                print(f"  - 丢弃包数: {env_unwrapped.packets_dropped}")
    
    return models


if __name__ == "__main__":
    # 训练混合流量模式；默认 max_steps=3000, reward_scale=0.1, gamma=0.995, lr=1e-4 利于收敛
    model, env, model_path, log_dir = train_sac(
        traffic_pattern="high_load",
        total_timesteps=2000000,  # 可改为 2000000 做完整训练
        eval_freq=10000,
        use_mixed_env=True,
        # max_steps=3000, reward_scale=0.1, gamma=0.995, learning_rate=1e-4 使用默认
    )
    
    # 绘制训练奖励曲线
    log_dir = "SAC_dynamic_uav_logs"
    if os.path.exists(log_dir):
        subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if subdirs:
            latest_dir = max(subdirs, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
            plot_training_rewards(os.path.join(log_dir, latest_dir))
    
    # 或者比较不同流量模式
    # models = compare_traffic_patterns()
