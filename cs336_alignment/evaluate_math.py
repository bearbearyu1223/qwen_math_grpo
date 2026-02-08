"""
Evaluate language models on MATH dataset using the r1_zero prompt.

This module provides utilities for evaluating models on the MATH dataset,
including prompt formatting, generation using multiple backends, and metric
computation with detailed analysis reports.

Supported backends:
    - vllm: Fast batched inference on NVIDIA GPUs (recommended)
    - transformers: Sequential inference for CPU/MPS (Mac M-series chips)

Example usage:
    >>> from cs336_alignment.evaluate_math import evaluate_math
    >>> metrics = evaluate_math(
    ...     model_name_or_path="outputs/grpo_model/final",
    ...     input_path="data/math/test.jsonl",
    ...     output_path="outputs/eval_results.jsonl",
    ...     backend="transformers",
    ... )
    >>> print(f"Answer accuracy: {metrics['answer_reward']:.2%}")
"""

import json
import logging
import os
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Callable, List, Literal

import torch
from tqdm import tqdm

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# Load the r1_zero prompt template
PROMPT_PATH = Path(__file__).parent / "prompts" / "r1_zero.prompt"
with open(PROMPT_PATH) as f:
    R1_ZERO_PROMPT_TEMPLATE = f.read()


def get_device() -> str:
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class TransformersModel:
    """HuggingFace transformers model wrapper for CPU/MPS inference."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or get_device()
        logger.info(f"Using device: {self.device}")

        if torch_dtype is None:
            if self.device in ("cuda", "mps"):
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        logger.info(f"Loading tokenizer from {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model from {model_name_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=self.device if self.device != "mps" else None,
        )
        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model loaded successfully")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> List[str]:
        """Generate responses for a list of prompts."""
        responses = []

        for prompt in tqdm(prompts, desc="Generating responses"):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                if temperature == 0.0:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)

        return responses


def format_r1_zero_prompt(question: str) -> str:
    """Format a math question using the r1_zero prompt template."""
    return R1_ZERO_PROMPT_TEMPLATE.format(question=question)


def compute_metrics_from_responses(
    responses: List[str],
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    input_examples: List[dict] | None = None,
) -> tuple[List[dict], dict[str, float]]:
    """Compute evaluation metrics from model-generated responses."""
    if input_examples is None:
        input_examples = [{} for _ in prompts]

    all_results = []
    all_metrics = []

    for input_example, prompt, response, ground_truth in tqdm(
        zip(input_examples, prompts, responses, ground_truths),
        total=len(prompts),
        desc="Computing metrics",
    ):
        metrics = reward_fn(response, ground_truth)
        all_metrics.append(metrics)

        result = {
            **input_example,
            "prompt": prompt,
            "output": response,
            "metrics": metrics,
        }
        all_results.append(result)

    aggregated_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            aggregated_metrics[key] = mean([m[key] for m in all_metrics])

    return all_results, aggregated_metrics


def evaluate_vllm(
    vllm_model: "LLM",
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: "SamplingParams",
    input_examples: List[dict] | None = None,
) -> tuple[List[dict], dict[str, float]]:
    """Evaluate using vLLM backend."""
    logger.info(f"Generating responses for {len(prompts)} prompts using vLLM...")
    raw_responses = vllm_model.generate(prompts, eval_sampling_params)

    responses = []
    for output in raw_responses:
        response = output.outputs[0].text
        responses.append(response)

    logger.info(f"Generated {len(responses)} responses")

    return compute_metrics_from_responses(
        responses=responses,
        prompts=prompts,
        ground_truths=ground_truths,
        reward_fn=reward_fn,
        input_examples=input_examples,
    )


def evaluate_transformers(
    model: TransformersModel,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    input_examples: List[dict] | None = None,
) -> tuple[List[dict], dict[str, float]]:
    """Evaluate using transformers backend."""
    logger.info(f"Generating responses for {len(prompts)} prompts using transformers...")
    responses = model.generate(
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    logger.info(f"Generated {len(responses)} responses")

    return compute_metrics_from_responses(
        responses=responses,
        prompts=prompts,
        ground_truths=ground_truths,
        reward_fn=reward_fn,
        input_examples=input_examples,
    )


def load_math_examples(input_path: str) -> tuple[List[dict], List[str], List[str]]:
    """Load MATH dataset examples and format them with r1_zero prompts."""
    input_examples = []
    with open(input_path) as f:
        for line in f:
            input_examples.append(json.loads(line))
    logger.info(f"Read {len(input_examples)} examples from {input_path}")

    prompts = []
    ground_truths = []
    for example in input_examples:
        question = example["problem"]
        prompt = format_r1_zero_prompt(question)
        prompts.append(prompt)
        ground_truths.append(example["answer"])

    return input_examples, prompts, ground_truths


def save_results(results: List[dict], output_path: str) -> None:
    """Save evaluation results to a JSONL file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as fout:
        for result in results:
            fout.write(json.dumps(result) + "\n")
    logger.info(f"Wrote {len(results)} results to {output_path}")


def categorize_results(results: List[dict]) -> dict[str, List[dict]]:
    """Categorize results based on format and answer correctness."""
    categories = {
        "correct": [],
        "format_only": [],
        "neither": [],
    }

    for result in results:
        metrics = result["metrics"]
        format_reward = metrics["format_reward"]
        answer_reward = metrics["answer_reward"]

        if format_reward == 1.0 and answer_reward == 1.0:
            categories["correct"].append(result)
        elif format_reward == 1.0 and answer_reward == 0.0:
            categories["format_only"].append(result)
        else:
            categories["neither"].append(result)

    return categories


def generate_analysis_report(results: List[dict], max_examples_per_category: int = 5) -> str:
    """Generate a detailed analysis report of evaluation results."""
    categories = categorize_results(results)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MATH EVALUATION ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    total = len(results)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Total examples: {total}")
    report_lines.append("")

    report_lines.append("Category Breakdown:")
    report_lines.append(
        f"  Correct (format=1, answer=1): {len(categories['correct'])} "
        f"({100*len(categories['correct'])/total:.1f}%)"
    )
    report_lines.append(
        f"  Format only (format=1, answer=0): {len(categories['format_only'])} "
        f"({100*len(categories['format_only'])/total:.1f}%)"
    )
    report_lines.append(
        f"  Neither (format=0): {len(categories['neither'])} "
        f"({100*len(categories['neither'])/total:.1f}%)"
    )
    report_lines.append("")

    # Show examples from each category
    for category_name, category_results in categories.items():
        if not category_results:
            continue

        report_lines.append("=" * 80)
        report_lines.append(f"EXAMPLES: {category_name.upper()}")
        report_lines.append("=" * 80)

        for i, example in enumerate(category_results[:max_examples_per_category], 1):
            report_lines.append(f"\n--- Example {i} ---")
            report_lines.append(f"Problem: {example.get('problem', 'N/A')[:200]}...")
            report_lines.append(f"Ground Truth: {example.get('answer', 'N/A')}")
            output = example.get("output", "N/A")
            report_lines.append(f"Model Output (first 500 chars): {output[:500]}...")
            report_lines.append("")

    return "\n".join(report_lines)


def save_analysis_report(report: str, output_path: str) -> None:
    """Save analysis report to file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"Wrote analysis report to {output_path}")
    print("\n" + report)


def evaluate_math(
    model_name_or_path: str,
    input_path: str = "data/math/test.jsonl",
    output_path: str = "outputs/math_eval_results.jsonl",
    num_gpus: int = 1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    num_samples: int | None = None,
    backend: Literal["vllm", "transformers"] = "vllm",
) -> dict[str, float]:
    """
    Evaluate a language model on the MATH dataset.

    Args:
        model_name_or_path: HuggingFace model ID or path to local model
        input_path: Path to MATH test examples (JSONL format)
        output_path: Path to write evaluation results
        num_gpus: Number of GPUs for vLLM (ignored for transformers backend)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        num_samples: Limit evaluation to N samples (None for all)
        backend: "vllm" for NVIDIA GPUs, "transformers" for CPU/MPS

    Returns:
        Dictionary of aggregated metrics (format_reward, answer_reward, reward)
    """
    input_examples, prompts, ground_truths = load_math_examples(input_path)

    if num_samples is not None and num_samples < len(prompts):
        logger.info(f"Limiting evaluation to {num_samples} samples")
        input_examples = input_examples[:num_samples]
        prompts = prompts[:num_samples]
        ground_truths = ground_truths[:num_samples]

    if backend == "vllm":
        from vllm import LLM, SamplingParams

        logger.info(f"Loading model {model_name_or_path} with vLLM backend...")
        model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=num_gpus,
            trust_remote_code=True,
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
        )

        all_results, aggregated_metrics = evaluate_vllm(
            vllm_model=model,
            reward_fn=r1_zero_reward_fn,
            prompts=prompts,
            ground_truths=ground_truths,
            eval_sampling_params=sampling_params,
            input_examples=input_examples,
        )
    elif backend == "transformers":
        logger.info(f"Loading model {model_name_or_path} with transformers backend...")
        model = TransformersModel(model_name_or_path)

        all_results, aggregated_metrics = evaluate_transformers(
            model=model,
            reward_fn=r1_zero_reward_fn,
            prompts=prompts,
            ground_truths=ground_truths,
            max_tokens=max_tokens,
            temperature=temperature,
            input_examples=input_examples,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    save_results(all_results, output_path)

    logger.info("=== Aggregated Metrics ===")
    for key, value in sorted(aggregated_metrics.items()):
        logger.info(f"{key}: {value:.4f}")

    analysis_report = generate_analysis_report(all_results)
    report_path = output_path.replace(".jsonl", "_analysis.txt")
    save_analysis_report(analysis_report, report_path)

    return aggregated_metrics
