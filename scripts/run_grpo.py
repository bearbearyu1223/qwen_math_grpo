#!/usr/bin/env python3
"""
GRPO Training Script for Qwen2.5-Math-1.5B.

This script trains a math reasoning model using Group Relative Policy Optimization
(GRPO) on the MATH dataset. It uses vLLM for fast rollout generation.

Usage:
    # Basic training (requires 2 GPUs: cuda:0 for policy, cuda:1 for vLLM)
    python scripts/run_grpo.py

    # Custom configuration
    python scripts/run_grpo.py \
        --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
        --train-data-path data/math/train.jsonl \
        --output-dir outputs/grpo_model \
        --n-grpo-steps 200 \
        --group-size 8

    # Single GPU mode (slower, uses HuggingFace generate instead of vLLM)
    python scripts/run_grpo.py --single-gpu

References:
    - DeepSeekMath: https://arxiv.org/abs/2402.03300
    - DeepSeek R1: https://arxiv.org/abs/2501.12948
"""

# IMPORTANT: Parse arguments and initialize vLLM BEFORE importing torch
# This allows us to set CUDA_VISIBLE_DEVICES for vLLM before CUDA is initialized

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model using GRPO on MATH dataset")

    # Model and data
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="data/math/train.jsonl",
        help="Path to training data (JSONL)",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default="data/math/test.jsonl",
        help="Path to validation data (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/grpo_model",
        help="Directory to save model and logs",
    )

    # GRPO hyperparameters
    parser.add_argument(
        "--n-grpo-steps",
        type=int,
        default=200,
        help="Number of GRPO training steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=256,
        help="Total number of rollouts per GRPO step",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of rollouts per question",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=128,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--epochs-per-rollout-batch",
        type=int,
        default=1,
        help="Number of epochs per rollout batch",
    )

    # Loss configuration
    parser.add_argument(
        "--loss-type",
        type=str,
        default="reinforce_with_baseline",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        help="Policy gradient loss type",
    )
    parser.add_argument(
        "--cliprange",
        type=float,
        default=0.2,
        help="Clip parameter for GRPO-Clip",
    )
    parser.add_argument(
        "--no-std-normalization",
        action="store_true",
        help="Disable std normalization in advantages (Dr. GRPO style)",
    )

    # Sampling parameters
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for rollouts",
    )
    parser.add_argument(
        "--sampling-max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per rollout",
    )

    # Hardware configuration
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Use single GPU mode (slower, no vLLM)",
    )
    parser.add_argument(
        "--policy-device",
        type=str,
        default="cuda:0",
        help="Device for policy model",
    )
    parser.add_argument(
        "--vllm-device",
        type=str,
        default="cuda:1",
        help="Device for vLLM inference",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--max-seq-length-train",
        type=int,
        default=512,
        help="Max sequence length during training (truncate longer sequences)",
    )

    # Other settings
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=None,
        help="Limit number of training samples (for debugging)",
    )
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=500,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Steps between checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (optional)",
    )

    return parser.parse_args()


def init_vllm_on_device(model_id: str, device: str, seed: int, gpu_memory_utilization: float):
    """
    Initialize vLLM on a specific GPU device.

    This must be called BEFORE importing torch to ensure CUDA_VISIBLE_DEVICES
    takes effect.
    """
    # Extract GPU index from device string (e.g., "cuda:1" -> "1")
    if ":" in device:
        gpu_id = device.split(":")[1]
    else:
        gpu_id = "0"

    logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_id} for vLLM")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Now import and initialize vLLM - it will only see the specified GPU
    from vllm import LLM

    logger.info(f"Initializing vLLM with model {model_id}...")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        seed=seed,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

    # Restore CUDA_VISIBLE_DEVICES to see all GPUs
    # This is needed so that torch can use the policy device
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    logger.info("vLLM initialized successfully")
    return llm


def main():
    # Parse arguments first (before any CUDA operations)
    args = parse_args()

    # Initialize vLLM BEFORE importing torch if using 2-GPU mode
    # This ensures CUDA_VISIBLE_DEVICES takes effect
    vllm_instance = None
    if not args.single_gpu:
        logger.info(f"Initializing vLLM on {args.vllm_device} (before loading policy)...")
        vllm_instance = init_vllm_on_device(
            model_id=args.model_name_or_path,
            device=args.vllm_device,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    # NOW import torch and other dependencies that use CUDA
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cs336_alignment.grpo import GRPOConfig, grpo_train_loop
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    # Set random seed
    torch.manual_seed(args.seed)

    # Load prompt template
    prompt_path = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    with open(prompt_path) as f:
        r1_zero_prompt_template = f.read()

    def load_math_data(data_path: str, num_samples: int | None = None) -> tuple[list[str], list[str]]:
        """Load MATH dataset and format prompts."""
        prompts = []
        answers = []
        with open(data_path) as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                example = json.loads(line)
                prompt = r1_zero_prompt_template.format(question=example["problem"])
                prompts.append(prompt)
                answers.append(example["answer"])
        logger.info(f"Loaded {len(prompts)} examples from {data_path}")
        return prompts, answers

    # Load data
    logger.info("Loading training data...")
    train_prompts, train_answers = load_math_data(
        args.train_data_path,
        num_samples=args.num_train_samples,
    )

    val_prompts, val_answers = None, None
    if os.path.exists(args.val_data_path):
        logger.info("Loading validation data...")
        val_prompts, val_answers = load_math_data(
            args.val_data_path,
            num_samples=args.num_val_samples,
        )

    # Create config
    config = GRPOConfig(
        model_name_or_path=args.model_name_or_path,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        loss_type=args.loss_type,
        cliprange=args.cliprange,
        use_std_normalization=not args.no_std_normalization,
        sampling_temperature=args.sampling_temperature,
        sampling_max_tokens=args.sampling_max_tokens,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_seq_length_train=args.max_seq_length_train,
    )

    logger.info("=" * 60)
    logger.info("GRPO TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"Training examples: {len(train_prompts)}")
    logger.info(f"GRPO steps: {config.n_grpo_steps}")
    logger.info(f"Rollout batch size: {config.rollout_batch_size}")
    logger.info(f"Group size: {config.group_size}")
    logger.info(f"Loss type: {config.loss_type}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info("=" * 60)

    # Load policy model
    logger.info(f"Loading policy model from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    policy = policy.to(args.policy_device)
    logger.info(f"Policy model loaded on {args.policy_device}")

    if args.single_gpu:
        logger.info("Running in single-GPU mode (using transformers for inference)")

    # Initialize wandb (optional)
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
            )
            logger.info(f"Weights & Biases initialized: {wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed, skipping logging")

    # Run training
    logger.info("Starting GRPO training...")
    trained_policy = grpo_train_loop(
        config=config,
        policy=policy,
        tokenizer=tokenizer,
        train_prompts=train_prompts,
        train_answers=train_answers,
        val_prompts=val_prompts,
        val_answers=val_answers,
        reward_fn=r1_zero_reward_fn,
        vllm_instance=vllm_instance,
        wandb_run=wandb_run,
    )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Model saved to: {config.output_dir}/final")
    logger.info("=" * 60)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
