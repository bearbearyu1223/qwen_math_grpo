# GRPO Training for Qwen2.5-Math-1.5B

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bearbearyu1223/qwen_math_grpo/blob/main/notebooks/grpo_training_colab.ipynb)

Train math reasoning models using Group Relative Policy Optimization (GRPO). GRPO is a policy gradient method from [DeepSeekMath](https://arxiv.org/abs/2402.03300) that uses group-normalized rewards as advantages without requiring a separate value function.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- **NVIDIA GPUs**: 2 GPUs recommended (~16GB VRAM each)
- **Apple Silicon**: Single-GPU mode with MPS backend (16GB+ unified memory)

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/bearbearyu1223/qwen_math_grpo.git
cd qwen_math_grpo

# Install dependencies
uv sync --extra vllm  # NVIDIA GPUs
uv sync               # Apple Silicon (no vLLM)

# Download dataset
uv run python scripts/download_dataset.py

# Train (see examples below)
```

## Training

### NVIDIA GPUs (2 GPUs)

```bash
uv run python scripts/run_grpo.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --n-grpo-steps 200 \
    --group-size 8
```

### Single GPU / Apple Silicon

```bash
uv run python scripts/run_grpo.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --single-gpu \
    --policy-device mps \
    --rollout-batch-size 8 \
    --train-batch-size 8 \
    --gradient-accumulation-steps 8 \
    --n-grpo-steps 100
```

> **Note**: For Apple Silicon, use `--policy-device mps`. Reduce batch sizes if you encounter memory issues. The `train-batch-size` must be >= `gradient-accumulation-steps` and divisible by it.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-grpo-steps` | 200 | Number of training steps |
| `--learning-rate` | 1e-5 | Learning rate |
| `--group-size` | 8 | Rollouts per question |
| `--loss-type` | `reinforce_with_baseline` | `no_baseline`, `reinforce_with_baseline`, or `grpo_clip` |
| `--wandb-project` | None | Enable W&B logging |

## Evaluation

```bash
# Evaluate trained model
uv run python scripts/run_math_eval.py \
    --model-name-or-path outputs/grpo_model/final \
    --backend transformers

# Compare with base model
uv run python scripts/run_math_eval.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --output-path outputs/base_eval.jsonl
```

Use `--backend vllm` for faster inference on NVIDIA GPUs, `--num-samples N` for quick tests.

## Download Scripts

```bash
# Download MATH dataset
uv run python scripts/download_dataset.py

# Download model (optional, for offline use)
uv run python scripts/download_model.py
uv run python scripts/download_model.py --output-dir ./models/qwen-math
```

## Project Structure

```
qwen_math_grpo/
├── cs336_alignment/
│   ├── grpo.py              # Core GRPO implementation
│   ├── drgrpo_grader.py     # Reward functions
│   └── evaluate_math.py     # Evaluation utilities
├── scripts/
│   ├── run_grpo.py          # Training script
│   ├── run_math_eval.py     # Evaluation script
│   ├── download_dataset.py  # Download MATH dataset
│   └── download_model.py    # Download Qwen model
├── notebooks/
│   └── grpo_training_colab.ipynb
└── data/math/               # Dataset (after download)
```

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948)

## License

MIT License
