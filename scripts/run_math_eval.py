#!/usr/bin/env python3
"""
Evaluate language models on MATH dataset using the r1_zero prompt.

Usage:
    # Evaluate GRPO-trained model on Apple Silicon (transformers backend)
    uv run python scripts/run_math_eval.py \
        --model-name-or-path outputs/grpo_model/final \
        --input-path data/math/test.jsonl \
        --output-path outputs/eval_results.jsonl \
        --backend transformers

    # Evaluate on NVIDIA GPU with vLLM (faster)
    uv run python scripts/run_math_eval.py \
        --model-name-or-path outputs/grpo_model/final \
        --backend vllm

    # Quick test with limited samples
    uv run python scripts/run_math_eval.py \
        --model-name-or-path outputs/grpo_model/final \
        --backend transformers \
        --num-samples 10

    # Evaluate base model for comparison
    uv run python scripts/run_math_eval.py \
        --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
        --output-path outputs/base_model_eval.jsonl \
        --backend transformers
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_alignment.evaluate_math import evaluate_math

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Evaluate model on MATH dataset with r1_zero prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HuggingFace model name or path to local model (e.g., outputs/grpo_model/final)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/math/test.jsonl",
        help="Path to MATH test examples (JSONL format)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/math_eval_results.jsonl",
        help="Path to write output results",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (vLLM backend only)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default="transformers",
        help="Inference backend: 'vllm' for NVIDIA GPUs, 'transformers' for CPU/MPS",
    )

    args = parser.parse_args()

    if args.backend == "transformers" and args.num_gpus > 1:
        logger.warning("--num-gpus is ignored with transformers backend")

    logger.info(f"Evaluating model: {args.model_name_or_path}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")

    metrics = evaluate_math(
        model_name_or_path=args.model_name_or_path,
        input_path=args.input_path,
        output_path=args.output_path,
        num_gpus=args.num_gpus,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        backend=args.backend,
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Format Accuracy: {metrics['format_reward']:.2%}")
    print(f"Answer Accuracy: {metrics['answer_reward']:.2%}")
    print(f"Overall Reward:  {metrics['reward']:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
