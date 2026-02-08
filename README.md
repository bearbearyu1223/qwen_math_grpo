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
uv sync --extra vllm              # NVIDIA GPUs
uv sync --extra vllm --extra cuda # A100/H100 (with Flash Attention)
uv sync                           # Apple Silicon (no vLLM)

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
# Evaluate trained model (vLLM - fast, requires NVIDIA GPU)
uv run python scripts/run_math_eval.py \
    --model-name-or-path outputs/grpo_model/final \
    --backend vllm

# Evaluate trained model (transformers - slower, works on CPU/MPS)
uv run python scripts/run_math_eval.py \
    --model-name-or-path outputs/grpo_model/final \
    --backend transformers

# Compare with base model
uv run python scripts/run_math_eval.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --output-path outputs/base_eval.jsonl \
    --backend vllm
```

Use `--num-samples N` for quick tests (e.g., `--num-samples 100`).

## Lambda Cloud

Run GRPO training on Lambda Cloud with NVIDIA GPUs.

### 1. Launch Instance

Choose an instance type on Lambda Cloud:

- **2+ GPUs** (recommended): 2x A100, 2x H100, etc. - enables vLLM acceleration
- **1 GPU**: Works but slower (uses transformers for inference)

### 2. Setup

```bash
# SSH into your Lambda instance
ssh ubuntu@<your-instance-ip>

# Check available GPUs
nvidia-smi --list-gpus

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/bearbearyu1223/qwen_math_grpo.git
cd qwen_math_grpo
uv sync --extra vllm

# Download dataset
uv run python scripts/download_dataset.py
```

### 3. Training

**With 2+ GPUs** (vLLM acceleration):

```bash
uv run python scripts/run_grpo.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --n-grpo-steps 200 \
    --group-size 8 \
    --output-dir outputs/grpo_model
```

**With 1 GPU** (single-GPU mode):

```bash
uv run python scripts/run_grpo.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --single-gpu \
    --policy-device cuda:0 \
    --rollout-batch-size 32 \
    --train-batch-size 32 \
    --gradient-accumulation-steps 8 \
    --n-grpo-steps 200 \
    --output-dir outputs/grpo_model
```

**With W&B logging** (optional):

```bash
uv run python scripts/run_grpo.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --n-grpo-steps 200 \
    --wandb-project qwen-math-grpo \
    --output-dir outputs/grpo_model
```

### 4. Evaluate

```bash
# Evaluate trained model
uv run python scripts/run_math_eval.py \
    --model-name-or-path outputs/grpo_model/final \
    --backend vllm

# Evaluate base model for comparison
uv run python scripts/run_math_eval.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --output-path outputs/base_eval.jsonl \
    --backend vllm
```

### 5. Download Results

```bash
# From your local machine
scp -r ubuntu@<your-instance-ip>:~/qwen_math_grpo/outputs ./lambda_outputs
```

## Download Scripts

```bash
# Download MATH dataset
uv run python scripts/download_dataset.py

# Download model (optional, for offline use)
uv run python scripts/download_model.py
uv run python scripts/download_model.py --output-dir ./models/qwen-math
```

## Plotting Training Metrics

Training metrics are automatically saved to `outputs/grpo_model/training_history.json`. To plot:

```bash
# Install matplotlib
uv sync --extra plot

# Plot training metrics
uv run python scripts/plot_training.py --input outputs/grpo_model/training_history.json

# Save plot to file
uv run python scripts/plot_training.py --input outputs/grpo_model/training_history.json --output training_plot.png

# Print summary only
uv run python scripts/plot_training.py --input outputs/grpo_model/training_history.json --summary-only
```

The plot shows:

- Average reward per step
- Answer reward (train vs validation)
- Policy gradient loss
- Reward range (min/max/mean)

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
│   ├── plot_training.py     # Plot training metrics
│   ├── download_dataset.py  # Download MATH dataset
│   └── download_model.py    # Download Qwen model
├── notebooks/
│   └── grpo_training_colab.ipynb
└── data/math/               # Dataset (after download)
```

## Troubleshooting

### CUDA Out of Memory

Gradient checkpointing is enabled by default to reduce memory usage. If you still get `torch.OutOfMemoryError: CUDA out of memory`, reduce batch sizes:

```bash
uv run python scripts/run_grpo.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --rollout-batch-size 64 \
    --train-batch-size 64 \
    --gradient-accumulation-steps 32 \
    --n-grpo-steps 200 \
    --group-size 8
```

For severe OOM, use even smaller values:

```bash
--rollout-batch-size 32 \
--train-batch-size 32 \
--gradient-accumulation-steps 16
```

**Batch size parameters**:

- `rollout-batch-size`: Total responses generated per step (prompts × group_size)
- `train-batch-size`: Samples processed per optimizer step
- `gradient-accumulation-steps`: Micro-batch size = train_batch_size / gradient_accumulation_steps

Reducing batch sizes while increasing gradient accumulation maintains training quality.

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948)

## License

MIT License
