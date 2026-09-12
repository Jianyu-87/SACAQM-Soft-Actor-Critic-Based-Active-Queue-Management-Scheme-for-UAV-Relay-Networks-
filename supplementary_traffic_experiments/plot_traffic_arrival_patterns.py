# -*- coding: utf-8 -*-
"""
Visualize independent per-priority arrival patterns (three figures, English labels).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from independent_traffic_env import IndependentTrafficUAVEnv

plt.style.use(["science", "ieee", "no-latex"])

TOTAL_STEPS = 1000
RNG_SEED = 42

EXPERIMENTS = [
    {
        "patterns": {3: "burst", 2: "dynamic", 1: "dynamic"},
        "mode_label": "Sinusoidal (dynamic)",
        "plot_title": "Exp.1: High=Burst, Med/Low=Sinusoidal",
        "pattern_desc": "dynamic",
    },
    {
        "patterns": {3: "burst", 2: "Pattern Shift", 1: "Pattern Shift"},
        "mode_label": "State Switching (Pattern Shift)",
        "plot_title": "Exp.2: High=Burst, Med/Low=State Switching",
        "pattern_desc": "Pattern Shift",
    },
    {
        "patterns": {3: "burst", 2: "Extreme Congestion", 1: "Extreme Congestion"},
        "mode_label": "Near-Saturated (Extreme Congestion)",
        "plot_title": "Exp.3: High=Burst, Med/Low=Near-Saturated",
        "pattern_desc": "Extreme Congestion",
    },
]


def simulate_experiment(patterns, num_steps, seed):
    env = IndependentTrafficUAVEnv(
        max_steps=num_steps,
        max_queue=10000,
        traffic_pattern=patterns[2],
        priority_traffic_patterns=patterns,
    )
    return env.simulate_arrival_counts(num_steps, seed=seed)


def plot_one_experiment(steps, high, medium, low, plot_title, mode_label, save_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(steps)

    ax.plot(t, high, color="#E94F37", linewidth=1.0, alpha=0.85,
            label="High priority (burst)")
    ax.plot(t, medium, color="#2E86AB", linewidth=1.0, alpha=0.85,
            label=f"Medium priority ({mode_label})")
    ax.plot(t, low, color="#44AF69", linewidth=1.0, alpha=0.85,
            label=f"Low priority ({mode_label})")

    ax.set_xlabel("Time slot / step", fontsize=12)
    ax.set_ylabel("Arrivals per slot (packets/slot)", fontsize=12)
    ax.set_title(f"Independent Arrival Patterns\n{plot_title}", fontsize=13)
    ax.set_xlim(0, steps - 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(save_dir, exist_ok=True)

    for idx, cfg in enumerate(EXPERIMENTS, start=1):
        counts = simulate_experiment(cfg["patterns"], TOTAL_STEPS, seed=RNG_SEED + idx)
        high, medium, low = counts[:, 0], counts[:, 1], counts[:, 2]
        save_path = os.path.join(save_dir, f"traffic_arrival_exp{idx}.png")
        plot_one_experiment(
            TOTAL_STEPS, high, medium, low,
            plot_title=cfg["plot_title"],
            mode_label=cfg["mode_label"],
            save_path=save_path,
        )
        print(
            f"  Exp{idx} [{cfg['pattern_desc']}] mean arrivals  "
            f"High={high.mean():.2f}  Med={medium.mean():.2f}  Low={low.mean():.2f}"
        )

    print(f"\nDone. {len(EXPERIMENTS)} figures in: {save_dir}")


if __name__ == "__main__":
    main()
