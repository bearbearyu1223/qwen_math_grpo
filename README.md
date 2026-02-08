# GRPO Training for Qwen2.5-Math-1.5B

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bearbearyu1223/qwen_math_grpo/blob/main/notebooks/grpo_training_colab.ipynb)

This repository implements Group Relative Policy Optimization (GRPO) for training math reasoning models. GRPO is a policy gradient method that uses group-normalized rewards as advantages to improve model performance without requiring a separate value function.

## Overview

GRPO (from DeepSeekMath and DeepSeek R1) works by:
1. Generating multiple rollout responses for each math question
2. Computing rewards based on answer correctness and format compliance
3. Normalizing rewards within each group (per-question) to get advantages
4. Training the policy using policy gradient methods (REINFORCE or PPO-style clipping)

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- **NVIDIA GPUs**: 2 GPUs recommended (one for policy training, one for vLLM inference), ~16GB VRAM per GPU
- **Apple Silicon (M4/M3/M2)**: Single-GPU mode with MPS backend, 16GB+ unified memory recommended

## Installation

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with Homebrew
brew install uv
```

### Install the Project

```bash
# Clone the repository
git clone https://github.com/yourusername/qwen_math_grpo.git
cd qwen_math_grpo

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install with vLLM support (NVIDIA GPUs)
uv pip install -e ".[vllm]"

# Install without vLLM (Apple Silicon)
uv pip install -e .
```

## Download Resources

### Download the MATH Dataset

Use the provided utility script to download the hendrycks-MATH benchmark:

```bash
# Basic usage (downloads to data/math/)
uv run python scripts/download_dataset.py

# Specify a custom output directory
uv run python scripts/download_dataset.py --output-dir ./my_data/math

# Download only specific splits
uv run python scripts/download_dataset.py --splits train

# Force overwrite existing files
uv run python scripts/download_dataset.py --force
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-id` | `nlile/hendrycks-MATH-benchmark` | HuggingFace dataset ID |
| `--output-dir` | `data/math` | Output directory for JSONL files |
| `--splits` | `train test` | Splits to download |
| `--force` | `false` | Overwrite existing files |

### Download the Qwen Model

Pre-download the Qwen2.5-Math-1.5B model for offline use or faster loading:

```bash
# Download to HuggingFace cache (default)
uv run python scripts/download_model.py

# Download to a specific local directory
uv run python scripts/download_model.py --output-dir ./models/qwen-math

# Download a different model
uv run python scripts/download_model.py --model-id Qwen/Qwen2.5-Math-7B

# Download only safetensors files (smaller download)
uv run python scripts/download_model.py --include "*.safetensors" "*.json"

# Use a specific revision
uv run python scripts/download_model.py --revision v1.0.0
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | `Qwen/Qwen2.5-Math-1.5B` | HuggingFace model ID |
| `--output-dir` | HuggingFace cache | Local directory to save model |
| `--revision` | `main` | Branch, tag, or commit hash |
| `--token` | `None` | HuggingFace token for gated models |
| `--include` | `None` | File patterns to include |
| `--exclude` | `None` | File patterns to exclude |

## Usage

### Basic Training (2 GPUs)

```bash
uv run python scripts/run_grpo.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --train-data-path data/math/train.jsonl \
    --output-dir outputs/grpo_model \
    --n-grpo-steps 200 \
    --group-size 8
```

### Single GPU Mode (slower)

```bash
uv run python scripts/run_grpo.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --single-gpu \
    --rollout-batch-size 32 \
    --train-batch-size 32 \
    --gradient-accumulation-steps 16
```

### With Weights & Biases Logging

```bash
uv run python scripts/run_grpo.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --wandb-project qwen-math-grpo
```

### Apple Silicon (M4/M3/M2)

Running on Apple Silicon requires single-GPU mode with the MPS (Metal Performance Shaders) backend:

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

**Notes for Apple Silicon:**

- vLLM is not available, so inference uses HuggingFace's `model.generate()` (slower)
- `train-batch-size` must be >= `gradient-accumulation-steps` and divisible by it
- Reduce batch sizes if you encounter memory pressure
- M4 Pro/Max with 18GB+ unified memory works best
- Training will be slower compared to NVIDIA GPUs with vLLM acceleration
- You may see MPS-related warnings which can typically be ignored

**Troubleshooting:**

```bash
# If you encounter MPS memory issues, try smaller batch sizes
uv run python scripts/run_grpo.py \
    --single-gpu \
    --policy-device mps \
    --rollout-batch-size 4 \
    --train-batch-size 4 \
    --gradient-accumulation-steps 4

# For very limited memory (8GB), use minimal settings
uv run python scripts/run_grpo.py \
    --single-gpu \
    --policy-device mps \
    --rollout-batch-size 2 \
    --train-batch-size 2 \
    --gradient-accumulation-steps 2 \
    --group-size 2
```

## Evaluation

After training, evaluate your model on the MATH test set:

### Evaluate GRPO-Trained Model

```bash
# Apple Silicon / CPU (transformers backend)
uv run python scripts/run_math_eval.py \
    --model-name-or-path outputs/grpo_model/final \
    --input-path data/math/test.jsonl \
    --output-path outputs/eval_results.jsonl \
    --backend transformers

# NVIDIA GPU (vLLM backend - faster)
uv run python scripts/run_math_eval.py \
    --model-name-or-path outputs/grpo_model/final \
    --backend vllm
```

### Compare with Base Model

```bash
# Evaluate base model for comparison
uv run python scripts/run_math_eval.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --output-path outputs/base_model_eval.jsonl \
    --backend transformers
```

### Quick Test (Limited Samples)

```bash
# Run on 10 samples for quick testing
uv run python scripts/run_math_eval.py \
    --model-name-or-path outputs/grpo_model/final \
    --backend transformers \
    --num-samples 10
```

**Evaluation Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--model-name-or-path` | `Qwen/Qwen2.5-Math-1.5B` | Model to evaluate |
| `--input-path` | `data/math/test.jsonl` | Test dataset path |
| `--output-path` | `outputs/math_eval_results.jsonl` | Results output path |
| `--backend` | `transformers` | `vllm` (NVIDIA) or `transformers` (CPU/MPS) |
| `--num-samples` | `None` | Limit samples (None = all) |
| `--temperature` | `0.0` | Sampling temperature (0.0 = greedy) |
| `--max-tokens` | `2048` | Max tokens to generate |

**Output Files:**

- `{output_path}`: Per-example results in JSONL format
- `{output_path}_analysis.txt`: Detailed analysis report with examples

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-name-or-path` | `Qwen/Qwen2.5-Math-1.5B` | HuggingFace model ID |
| `--n-grpo-steps` | 200 | Number of GRPO training steps |
| `--learning-rate` | 1e-5 | Learning rate |
| `--rollout-batch-size` | 256 | Total rollouts per step |
| `--group-size` | 8 | Rollouts per question |
| `--loss-type` | `reinforce_with_baseline` | Loss type: `no_baseline`, `reinforce_with_baseline`, `grpo_clip` |
| `--cliprange` | 0.2 | PPO-style clip parameter (for `grpo_clip`) |
| `--sampling-temperature` | 1.0 | Temperature for rollout sampling |

## Loss Types

1. **`no_baseline`**: Naive policy gradient with raw rewards (high variance)
2. **`reinforce_with_baseline`**: Uses group-normalized advantages (recommended)
3. **`grpo_clip`**: PPO-style clipping for off-policy stability

## Project Structure

```
qwen_math_grpo/
├── cs336_alignment/
│   ├── grpo.py              # Core GRPO implementation
│   ├── utils.py             # Tokenization and log-prob utilities
│   ├── drgrpo_grader.py     # Reward functions and answer grading
│   ├── evaluate_math.py     # Evaluation utilities
│   └── prompts/
│       └── r1_zero.prompt
├── scripts/
│   ├── run_grpo.py          # Training script
│   ├── run_math_eval.py     # Evaluation script
│   ├── download_dataset.py  # Download MATH dataset
│   └── download_model.py    # Download Qwen model
├── notebooks/
│   └── grpo_training_colab.ipynb  # Colab notebook
├── data/
│   └── math/
│       ├── train.jsonl      # 12,000 training examples
│       └── test.jsonl       # 500 test examples
├── pyproject.toml
└── README.md
```

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

## License

MIT License
