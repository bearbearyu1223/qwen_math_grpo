#!/usr/bin/env python3
"""
Plot training metrics from GRPO training.

Usage:
    python scripts/plot_training.py --input outputs/grpo_model/training_history.json
    python scripts/plot_training.py --input outputs/grpo_model/training_history.json --output training_plot.png
"""

import argparse
import json
from pathlib import Path


def plot_training_metrics(history_path: str, output_path: str | None = None):
    """
    Plot training metrics from a training history JSON file.

    Args:
        history_path: Path to training_history.json
        output_path: Optional path to save the plot (if None, displays interactively)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    # Load training history
    with open(history_path) as f:
        history = json.load(f)

    if not history:
        print("No training data found in history file.")
        return

    # Extract metrics
    steps = [h["grpo_step"] for h in history]
    reward_mean = [h["reward_mean"] for h in history]
    answer_reward = [h["answer_reward_mean"] for h in history]
    loss = [h.get("loss", 0) for h in history]
    val_reward = [h.get("val_reward") for h in history]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("GRPO Training Metrics", fontsize=14)

    # Plot 1: Reward Mean
    ax1 = axes[0, 0]
    ax1.plot(steps, reward_mean, "b-", linewidth=1.5, label="Reward Mean")
    ax1.set_xlabel("GRPO Step")
    ax1.set_ylabel("Reward")
    ax1.set_title("Average Reward per Step")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Answer Reward
    ax2 = axes[0, 1]
    ax2.plot(steps, answer_reward, "g-", linewidth=1.5, label="Answer Reward")
    # Add validation reward if available
    val_steps = [s for s, v in zip(steps, val_reward) if v is not None]
    val_values = [v for v in val_reward if v is not None]
    if val_values:
        ax2.plot(val_steps, val_values, "r--", linewidth=2, marker="o", markersize=4, label="Val Reward")
    ax2.set_xlabel("GRPO Step")
    ax2.set_ylabel("Answer Reward")
    ax2.set_title("Answer Reward (Train vs Val)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Loss
    ax3 = axes[1, 0]
    ax3.plot(steps, loss, "r-", linewidth=1.5, label="Policy Loss")
    ax3.set_xlabel("GRPO Step")
    ax3.set_ylabel("Loss")
    ax3.set_title("Policy Gradient Loss")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Reward Statistics
    ax4 = axes[1, 1]
    reward_max = [h["reward_max"] for h in history]
    reward_min = [h["reward_min"] for h in history]
    ax4.fill_between(steps, reward_min, reward_max, alpha=0.3, color="blue", label="Min-Max Range")
    ax4.plot(steps, reward_mean, "b-", linewidth=1.5, label="Mean")
    ax4.set_xlabel("GRPO Step")
    ax4.set_ylabel("Reward")
    ax4.set_title("Reward Range (Min/Max/Mean)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def print_summary(history_path: str):
    """Print a summary of training metrics."""
    with open(history_path) as f:
        history = json.load(f)

    if not history:
        print("No training data found.")
        return

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    # First and last step metrics
    first = history[0]
    last = history[-1]

    print(f"\nTotal GRPO steps: {len(history)}")
    print(f"\nInitial metrics (step 0):")
    print(f"  Reward Mean: {first['reward_mean']:.4f}")
    print(f"  Answer Reward: {first['answer_reward_mean']:.4f}")
    print(f"  Loss: {first.get('loss', 'N/A')}")

    print(f"\nFinal metrics (step {last['grpo_step']}):")
    print(f"  Reward Mean: {last['reward_mean']:.4f}")
    print(f"  Answer Reward: {last['answer_reward_mean']:.4f}")
    print(f"  Loss: {last.get('loss', 'N/A')}")

    # Improvement
    if first["answer_reward_mean"] != 0:
        improvement = (last["answer_reward_mean"] - first["answer_reward_mean"]) / first["answer_reward_mean"] * 100
        print(f"\nAnswer Reward Improvement: {improvement:+.1f}%")
    else:
        improvement = last["answer_reward_mean"] - first["answer_reward_mean"]
        print(f"\nAnswer Reward Change: {improvement:+.4f}")

    # Best validation reward
    val_rewards = [(h["grpo_step"], h["val_reward"]) for h in history if h.get("val_reward") is not None]
    if val_rewards:
        best_step, best_val = max(val_rewards, key=lambda x: x[1])
        print(f"\nBest Validation Reward: {best_val:.4f} (step {best_step})")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO training metrics")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="outputs/grpo_model/training_history.json",
        help="Path to training_history.json",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save plot (if not specified, displays interactively)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, don't plot",
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return

    # Print summary
    print_summary(args.input)

    # Plot if not summary-only
    if not args.summary_only:
        plot_training_metrics(args.input, args.output)


if __name__ == "__main__":
    main()
