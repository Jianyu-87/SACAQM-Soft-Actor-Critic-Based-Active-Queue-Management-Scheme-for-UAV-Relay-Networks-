# SACAQM — Code for the paper

    "SACAQM: Soft Actor-Critic-Based Active Queue Management Scheme for UAV Relay Networks"
    

Reinforcement-learning-based active queue management (AQM) for a
dynamic-traffic UAV relay. **SACAQM** is a Soft Actor-Critic agent that,
at each time slot, emits a continuous *(energy threshold, priority weight,
urgency weight)* triple; packets are scheduled by a priority/urgency
dynamic score and the link is filled up to the instantaneous Shannon
capacity. It is compared against a **DQN** agent and the classic **CoDel**
AQM baseline under several bursty / fluctuating traffic patterns.



### Environments

| File | Role |
|---|---|
| `dynamic_traffic_uav.py` | **SACAQM environment (core).** Three-priority sensor traffic (Poisson arrivals with sinusoidal / burst modulation), Rayleigh block-fading channel (AR(1)-correlated) mapped to a per-slot packet budget via the Shannon formula. Continuous 3-D action = `(energy_threshold, w_priority, w_urgency)`; 7-D observation = per-priority queue occupancy + per-priority most-urgent delay ratio + normalized channel capacity. Scheduling uses the dynamic score `w_priority·priority + w_urgency·(delay/max_delay)`, then fills the link to capacity. Also defines the `Packet` class. |
| `dqn_dynamic_uav.py` | **DQN baseline environment.** Discrete 3-action space, `3 + max_queue`-D observation (153-D at the default `max_queue=150`). |
| `codel_dynamic_uav.py` | **CoDel baseline** (`CoDelDynamicUAVEnv`): classic controlled-delay AQM on the same traffic/channel model; no learning required. |

### Training

| File | Role |
|---|---|
| `train_dynamic_uav.py` | Train SACAQM (Stable-Baselines3 SAC). `__main__` trains on a **mixed** traffic env (`dynamic` / `extreme_fluctuation_1` / `extreme_fluctuation_2`) for generalization. Models → `SAC_dynamic_uav_models/`, TensorBoard → `dynamic_uav_tensorboard/`. |
| `dqn_train_dynamic_uav.py` | Train the DQN baseline (same mixed-env scheme). Models → `dynamic_uav_models/`. Includes `EpisodeSuccessRateLogger` for per-episode success-rate CSVs. |

### Evaluation

| File | Role |
|---|---|
| `sac_robustness_test.py` | Run a trained SACAQM model under a given traffic pattern; per-episode / per-step stats + per-priority success rates. |
| `dqn_robustness_test.py` | Same for the DQN baseline. |
| `codel_robustness_test.py` | Same for CoDel (no model needed). |
| `compare_dqn_sac_codel.py` | **Main comparison entry point.** Runs DQN / SAC / CoDel through the three robustness modules and writes comparison CSVs to `algorithm_comparison_<timestamp>/`. Set `dqn_model_path` / `sac_model_path` at the bottom before running. |
| `convergence_experiment.py` | Convergence study: train per traffic pattern and plot evaluation-return vs. training-step learning curves. Full CLI (see below). |
| `sac_simulate.py` / `dqn_simulate.py` | Stand-alone single-model rollout + statistics (earlier evaluation scripts, still runnable). |
| `calculate_channel_capacity.py` | Sanity-check script: prints the Shannon capacity → packet-budget table for the channel parameters used in the paper. |

### Plotting

| File | Role |
|---|---|
| `plot_success.py` | Grouped bar chart of per-priority (P1/P2/P3) packet success rate from a comparison CSV. |
| `plot_transmisson_efficiency.py` | Transmission-efficiency / energy bar chart from a comparison CSV. |
| `plot_convercy.py` | SAC reward-convergence curves read directly from TensorBoard event files (four load levels). |
| `plot_packet_success_rate.py` | Packet-success-rate convergence curves from SB3 `monitor.csv` (CLI). |
| `plot_high_priority_success_by_load.py` | High-priority (P3) success rate vs. traffic load, line or bar (CLI). |

### Supplementary

| Path | Role |
|---|---|
| `supplementary_traffic_experiments/` | Self-contained supplementary study: **per-priority independent** traffic-arrival patterns for the DQN / SAC / CoDel comparison. See its own `README.md`. |

## Requirements

Python ≥ 3.9, `numpy`, `pandas`, `torch`, `gymnasium`,
`stable-baselines3`, `matplotlib`, `scienceplots`,
`tensorboard` (optional, for `plot_convercy.py`).

```bash
pip install -r requirements.txt
```

## Reproducing the paper

```bash
# 1. Train SACAQM (SAC) on mixed traffic
python train_dynamic_uav.py
#    -> SAC_dynamic_uav_models/<pattern>_<timestamp>/best_model/best_model.zip

# 2. Train the DQN baseline
python dqn_train_dynamic_uav.py
#    -> dynamic_uav_models/<pattern>_<timestamp>/best_model/best_model.zip
#    (CoDel needs no training.)

# 3. Convergence study (learning curves)
python convergence_experiment.py --run --total_timesteps 500000
python convergence_experiment.py --plot_only --results_dir convergence_study/exp_<timestamp>
#    optional: --patterns constant dynamic --seeds 0 1 2 --eval_freq 10000

# 4. Compare the three algorithms
#    First edit dqn_model_path / sac_model_path near the bottom of the file.
python compare_dqn_sac_codel.py
#    -> algorithm_comparison_<timestamp>/*.csv

# 5. Figures (edit the CSV_PATH / RUNS paths inside each script first)
python plot_success.py
python plot_transmisson_efficiency.py
python plot_convercy.py
python plot_packet_success_rate.py --logs path/to/monitor.csv
```

## Model conventions

- **Time.** One `step` = one time slot; `time_slot_duration = 1.0`.
- **Channel.** Rayleigh block fading, average SNR ≈ 14.5 dB, coherence =
  3 steps, bandwidth = 1.15e4 Hz. Per-slot packet budget =
  `η · C · T / packet_size_bits` with `η = 0.7`, `packet_size = 8000` bits.
- **Traffic.** Three priorities — high (3) / medium (2) / low (1) — with
  `max_delay = 5 / 10 / 15` steps and base arrival rates scaled per
  priority (see `_generate_*_priority_packets`). Traffic pattern is set
  via the `traffic_pattern` argument (`dynamic`, `burst`, `constant`,
  `extreme_fluctuation_1/2`, `Pattern Shift`, `Extreme Congestion`, ...).
- **Success rate.** `transmitted / (transmitted + expired)`; per-priority
  success uses delay-constraint satisfaction.
