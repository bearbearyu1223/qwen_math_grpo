"""
Group Relative Policy Optimization (GRPO) for math reasoning models.

This module implements GRPO as described in DeepSeekMath (Shao et al., 2024)
and DeepSeek R1 (DeepSeek-AI et al., 2025). GRPO is a policy gradient method
that uses group-normalized rewards as advantages to avoid learning a separate
value function.

Key components:
    - compute_group_normalized_rewards: Compute advantages using group normalization
    - compute_naive_policy_gradient_loss: Basic REINFORCE loss
    - compute_grpo_clip_loss: PPO-style clipped policy gradient loss
    - compute_policy_gradient_loss: Wrapper to select loss type
    - masked_mean: Utility for averaging over response tokens
    - grpo_microbatch_train_step: Single training microbatch step
    - grpo_train_loop: Full GRPO training loop (Algorithm 3)

References:
    - DeepSeekMath: https://arxiv.org/abs/2402.03300
    - DeepSeek R1: https://arxiv.org/abs/2501.12948
    - PPO: https://arxiv.org/abs/1707.06347
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# vLLM is optional - only required for multi-GPU mode
if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


# ==============================================================================
# Advantage/Reward Computation
# ==============================================================================


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by group statistics.

    This implements the advantage estimation from GRPO (Eq. 28 in the assignment):

        A^(i) = (r^(i) - mean(r^(1), ..., r^(G))) / (std(r^(1), ..., r^(G)) + eps)

    Or if normalize_by_std=False, the simplified Dr. GRPO variant (Eq. 31):

        A^(i) = r^(i) - mean(r^(1), ..., r^(G))

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Function that scores rollout responses against ground truths.
            Should return a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. Length = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            Ground truths repeated for each rollout. Same length as rollout_responses.
        group_size: int
            Number of responses per question (G in the notation).
        advantage_eps: float
            Small constant added to std to prevent division by zero.
        normalize_by_std: bool
            If True, divide by the per-group standard deviation.
            If False, only subtract the group mean (Dr. GRPO style).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            - advantages: shape (rollout_batch_size,), group-normalized rewards
            - raw_rewards: shape (rollout_batch_size,), unnormalized rewards
            - metadata: dict with statistics for logging (mean, std, max, min rewards, etc.)
    """
    rollout_batch_size = len(rollout_responses)
    assert len(repeated_ground_truths) == rollout_batch_size, (
        "rollout_responses and repeated_ground_truths must have same length"
    )
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )

    n_groups = rollout_batch_size // group_size

    # Compute raw rewards for each response
    raw_rewards = []
    format_rewards = []
    answer_rewards = []

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(response, ground_truth)
        raw_rewards.append(reward_info["reward"])
        format_rewards.append(reward_info.get("format_reward", 0.0))
        answer_rewards.append(reward_info.get("answer_reward", 0.0))

    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    format_rewards_tensor = torch.tensor(format_rewards, dtype=torch.float32)
    answer_rewards_tensor = torch.tensor(answer_rewards, dtype=torch.float32)

    # Reshape to (n_groups, group_size) for group-wise operations
    rewards_grouped = raw_rewards.view(n_groups, group_size)

    # Compute group-wise mean and std
    group_means = rewards_grouped.mean(dim=1, keepdim=True)  # (n_groups, 1)
    group_stds = rewards_grouped.std(dim=1, keepdim=True)    # (n_groups, 1)

    # Compute advantages (normalized rewards)
    if normalize_by_std:
        # Standard GRPO: A = (r - mean) / (std + eps)
        advantages_grouped = (rewards_grouped - group_means) / (group_stds + advantage_eps)
    else:
        # Dr. GRPO variant: A = r - mean (no std normalization)
        advantages_grouped = rewards_grouped - group_means

    # Flatten back to (rollout_batch_size,)
    advantages = advantages_grouped.view(-1)

    # Compute metadata for logging
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
        "format_reward_mean": format_rewards_tensor.mean().item(),
        "answer_reward_mean": answer_rewards_tensor.mean().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
        "group_mean_mean": group_means.mean().item(),
        "group_std_mean": group_stds.mean().item(),
    }

    return advantages, raw_rewards, metadata


# ==============================================================================
# Policy Gradient Losses
# ==============================================================================


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the per-token naive policy gradient loss.

    The naive policy gradient loss for each token is (Eq. 32 in assignment):

        -A_t * log p_θ(o_t | q, o_{<t})

    where A_t is the advantage (or raw reward) and log p_θ is the log probability
    under the current policy.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), log-probs for each token.

    Returns:
        torch.Tensor: Shape (batch_size, sequence_length), per-token policy gradient loss.
            This should be aggregated (summed/averaged) with masking in the training loop.
    """
    # Ensure rewards/advantages can broadcast over sequence dimension
    # raw_rewards_or_advantages: (batch_size, 1)
    # policy_log_probs: (batch_size, sequence_length)

    # The policy gradient loss: -A * log(p)
    # We negate because we want to maximize the objective, but optimizers minimize
    per_token_loss = -raw_rewards_or_advantages * policy_log_probs

    return per_token_loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the per-token GRPO-Clip loss (PPO-style clipping).

    The GRPO-Clip per-token loss is (Eq. 33 in assignment):

        -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    where:
        ratio = π_θ(o_t | q, o_{<t}) / π_θ_old(o_t | q, o_{<t})
              = exp(log p_θ - log p_θ_old)
        A = advantage
        ε = cliprange

    The clipping prevents the policy from moving too far from the old policy,
    which is important for off-policy training stability.

    Args:
        advantages: torch.Tensor
            Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), per-token log probs from current policy.
        old_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), per-token log probs from old policy.
        cliprange: float
            Clip parameter ε (e.g., 0.2).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss: Shape (batch_size, sequence_length), per-token clipped loss
            - metadata: dict with "clipped" tensor indicating which tokens were clipped
    """
    # Compute probability ratio: π_θ / π_θ_old = exp(log π_θ - log π_θ_old)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)  # (batch_size, sequence_length)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # Compute both unclipped and clipped objective terms
    # advantages: (batch_size, 1) -> broadcasts to (batch_size, sequence_length)
    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages

    # PPO-style clipping: take the minimum (pessimistic bound)
    # This ensures we don't get credit for moving too far from the old policy
    per_token_objective = torch.min(unclipped_objective, clipped_objective)

    # Convert to loss (negate because we minimize loss but want to maximize objective)
    per_token_loss = -per_token_objective

    # Track which tokens were clipped (for logging)
    # A token is "clipped" when the clipped objective was used instead of unclipped
    clipped = (clipped_objective < unclipped_objective).float()

    metadata = {
        "clipped": clipped,
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio.std().detach(),
        "ratio_max": ratio.max().detach(),
        "ratio_min": ratio.min().detach(),
    }

    return per_token_loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    This is a convenience wrapper that dispatches to the correct loss routine:
    - "no_baseline": Naive policy gradient with raw rewards (no variance reduction)
    - "reinforce_with_baseline": Naive policy gradient with normalized advantages
    - "grpo_clip": GRPO-Clip (PPO-style clipping for off-policy stability)

    Args:
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), per-token log-probabilities
            from the policy being trained.
        loss_type: str
            One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: torch.Tensor | None
            Required if loss_type == "no_baseline". Shape (batch_size, 1).
        advantages: torch.Tensor | None
            Required for "reinforce_with_baseline" and "grpo_clip". Shape (batch_size, 1).
        old_log_probs: torch.Tensor | None
            Required for "grpo_clip". Shape (batch_size, sequence_length).
        cliprange: float | None
            Required for "grpo_clip". Scalar ε used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss: Shape (batch_size, sequence_length), per-token loss
            - metadata: dict with statistics from the underlying routine
    """
    metadata = {}

    if loss_type == "no_baseline":
        # Validate arguments
        assert raw_rewards is not None, (
            "raw_rewards is required for loss_type='no_baseline'"
        )

        # Use raw rewards directly (no baseline subtraction)
        per_token_loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )

    elif loss_type == "reinforce_with_baseline":
        # Validate arguments
        assert advantages is not None, (
            "advantages is required for loss_type='reinforce_with_baseline'"
        )

        # Use group-normalized advantages as baseline
        per_token_loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )

    elif loss_type == "grpo_clip":
        # Validate arguments
        assert advantages is not None, (
            "advantages is required for loss_type='grpo_clip'"
        )
        assert old_log_probs is not None, (
            "old_log_probs is required for loss_type='grpo_clip'"
        )
        assert cliprange is not None, (
            "cliprange is required for loss_type='grpo_clip'"
        )

        per_token_loss, clip_metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        metadata.update(clip_metadata)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. "
                        f"Expected one of: 'no_baseline', 'reinforce_with_baseline', 'grpo_clip'")

    return per_token_loss, metadata


# ==============================================================================
# Masking Utilities
# ==============================================================================


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only
    elements where mask == 1.

    This is useful for averaging losses or entropies only over response tokens,
    excluding prompt and padding tokens.

    Args:
        tensor: torch.Tensor
            The data to be averaged.
        mask: torch.Tensor
            Same shape as tensor. Positions with value 1 are included in the mean,
            positions with value 0 are excluded.
        dim: int | None
            Dimension over which to average. If None, compute the mean over all
            masked elements (returns a scalar).

    Returns:
        torch.Tensor: The masked mean. Shape follows tensor.mean(dim) semantics.

    Example:
        >>> tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        >>> masked_mean(tensor, mask, dim=1)  # Average over sequence
        tensor([1.5000, 5.0000])
        >>> masked_mean(tensor, mask)  # Global average
        tensor(3.4000)
    """
    # Convert mask to float for computation
    mask_float = mask.float()

    # Apply mask to tensor
    masked_tensor = tensor * mask_float

    if dim is None:
        # Global mean over all masked elements
        total_sum = masked_tensor.sum()
        num_elements = mask_float.sum()
        # Avoid division by zero
        return total_sum / torch.clamp(num_elements, min=1e-8)
    else:
        # Mean along specified dimension
        sum_along_dim = masked_tensor.sum(dim=dim)
        count_along_dim = mask_float.sum(dim=dim)
        # Avoid division by zero
        return sum_along_dim / torch.clamp(count_along_dim, min=1e-8)


# ==============================================================================
# Training Steps
# ==============================================================================


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a single microbatch for GRPO.

    This function:
    1. Computes the per-token policy gradient loss
    2. Averages over response tokens using masked_mean
    3. Averages over the batch dimension
    4. Scales by gradient_accumulation_steps for proper gradient averaging
    5. Calls backward() to populate gradients

    Args:
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), per-token log-probabilities
            from the policy being trained. Must have requires_grad=True.
        response_mask: torch.Tensor
            Shape (batch_size, sequence_length), 1 for response tokens,
            0 for prompt/padding tokens.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step. Loss is scaled by
            1/gradient_accumulation_steps for proper gradient averaging.
        loss_type: str
            One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: torch.Tensor | None
            Required if loss_type == "no_baseline". Shape (batch_size, 1).
        advantages: torch.Tensor | None
            Required for other loss types. Shape (batch_size, 1).
        old_log_probs: torch.Tensor | None
            Required for "grpo_clip". Shape (batch_size, sequence_length).
        cliprange: float | None
            Required for "grpo_clip". Clip parameter ε.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss: Scalar tensor. The microbatch loss after scaling for
                   gradient accumulation (for logging purposes).
            - metadata: Dict with statistics from the loss computation.
    """
    # Compute per-token loss
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Average over sequence dimension (only response tokens)
    # per_example_loss: (batch_size,)
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)

    # Average over batch dimension
    batch_loss = per_example_loss.mean()

    # Scale for gradient accumulation
    # This ensures that gradients accumulated over multiple microbatches
    # average correctly
    scaled_loss = batch_loss / gradient_accumulation_steps

    # Backward pass
    scaled_loss.backward()

    # Add additional metadata
    metadata["batch_loss"] = batch_loss.detach()
    metadata["num_response_tokens"] = response_mask.sum().detach()

    # Compute clip fraction if applicable
    if "clipped" in metadata:
        clipped = metadata["clipped"]
        clip_fraction = masked_mean(clipped, response_mask).detach()
        metadata["clip_fraction"] = clip_fraction

    return scaled_loss.detach(), metadata


# ==============================================================================
# GRPO Training Loop (Algorithm 3 in the assignment)
# ==============================================================================


@dataclass
class GRPOConfig:
    """
    Configuration for GRPO training.

    Attributes:
        model_name_or_path: Path to the base model
        train_data_path: Path to training data (JSONL with "prompt" and "answer" fields)
        output_dir: Directory to save checkpoints and logs

        # GRPO hyperparameters
        n_grpo_steps: Number of outer GRPO steps
        learning_rate: Learning rate for optimizer
        advantage_eps: Small constant for numerical stability in advantage normalization
        rollout_batch_size: Total number of rollouts per GRPO step
        group_size: Number of rollouts per question (G)
        sampling_temperature: Temperature for sampling rollouts
        sampling_min_tokens: Minimum tokens to generate
        sampling_max_tokens: Maximum tokens to generate
        epochs_per_rollout_batch: Number of epochs of gradient steps per rollout batch
        train_batch_size: Batch size for training
        gradient_accumulation_steps: Number of microbatches per optimizer step

        # Loss configuration
        loss_type: One of "no_baseline", "reinforce_with_baseline", "grpo_clip"
        cliprange: Clip parameter for GRPO-Clip
        use_std_normalization: Whether to normalize by group std

        # Other settings
        max_grad_norm: Gradient clipping norm
        eval_steps: Steps between evaluations
        save_steps: Steps between checkpoints
        seed: Random seed
    """
    model_name_or_path: str = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    train_data_path: str = "/data/a5-alignment/MATH/train.jsonl"
    output_dir: str = "outputs/grpo_model"

    # GRPO hyperparameters
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128

    # Loss configuration
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    cliprange: float = 0.2
    use_std_normalization: bool = True

    # Other settings
    max_grad_norm: float = 1.0
    eval_steps: int = 10
    save_steps: int = 50
    seed: int = 42
    gpu_memory_utilization: float = 0.85
    max_seq_length_train: int = 512  # Truncate sequences longer than this during training


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> "LLM":
    """
    Initialize a vLLM instance for fast inference.

    Note: CUDA_VISIBLE_DEVICES cannot be used to place vLLM on a specific GPU
    after PyTorch/CUDA has been initialized. vLLM will use GPU 0 by default.
    The gpu_memory_utilization parameter controls how much GPU memory vLLM uses,
    leaving room for the policy model and training.

    Args:
        model_id: HuggingFace model ID or path
        device: Device string (ignored - vLLM uses GPU 0 after CUDA init)
        seed: Random seed for reproducibility
        gpu_memory_utilization: Fraction of GPU memory to use (default 0.85)

    Returns:
        vLLM LLM instance
    """
    from vllm import LLM

    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Note: vLLM will use GPU 0 since CUDA is already initialized
    # We limit memory usage to leave room for policy model and training
    logger.info(f"Initializing vLLM with gpu_memory_utilization={gpu_memory_utilization}")

    return LLM(
        model=model_id,
        dtype="bfloat16",
        seed=seed,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,  # Disable CUDA graphs to save memory
    )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: "LLM") -> None:
    """
    Load policy model weights into a vLLM instance for inference.

    This allows using the updated policy weights with vLLM's fast inference.
    Supports vLLM v0.6.x and v0.7.x (pinned in pyproject.toml).

    Args:
        policy: The HuggingFace policy model with updated weights
        llm: The vLLM instance to update
    """
    state_dict = policy.state_dict()

    # Try different paths for different vLLM versions
    model = None

    # vLLM v0.6.x - v0.7.x path (primary)
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass

    # vLLM v0.5.x path (fallback)
    if model is None:
        try:
            model = llm.llm_engine.driver_worker.model_runner.model
        except AttributeError:
            pass

    if model is not None:
        model.load_weights(state_dict.items())
    else:
        raise RuntimeError(
            "Cannot load weights into vLLM instance - internal API not found. "
            "This may be due to vLLM version incompatibility. Options:\n"
            "  1. Use single-GPU mode: --single-gpu\n"
            "  2. Reinstall vLLM: uv sync --extra vllm --reinstall\n"
            "  3. Check that vLLM version is 0.6.x or 0.7.x"
        )


def grpo_train_loop(
    config: GRPOConfig,
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_prompts: list[str],
    train_answers: list[str],
    val_prompts: list[str] | None = None,
    val_answers: list[str] | None = None,
    reward_fn: Callable[[str, str], dict[str, float]] | None = None,
    vllm_instance: "LLM | None" = None,
    wandb_run: Any | None = None,
) -> PreTrainedModel:
    """
    Run the full GRPO training loop (Algorithm 3 from the assignment).

    Algorithm overview:
    1. For each GRPO step:
       a. Sample a batch of questions from the training data
       b. Generate G rollouts per question using the current policy
       c. Compute rewards for each rollout
       d. Compute group-normalized advantages
       e. For each training epoch on the rollout batch:
          - Update the policy using the chosen loss function

    This implementation supports:
    - On-policy training (epochs_per_rollout_batch=1, train_batch_size=rollout_batch_size)
    - Off-policy training (multiple epochs, GRPO-Clip loss)
    - Periodic validation and checkpoint saving

    Args:
        config: GRPOConfig with all hyperparameters
        policy: The policy model to train (HuggingFace model)
        tokenizer: Tokenizer for the policy model
        train_prompts: List of training prompts
        train_answers: List of ground-truth answers for training
        val_prompts: Optional validation prompts
        val_answers: Optional validation answers
        reward_fn: Reward function. Defaults to r1_zero_reward_fn
        vllm_instance: Optional pre-initialized vLLM instance
        wandb_run: Optional wandb run for logging

    Returns:
        The trained policy model
    """
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    from cs336_alignment.utils import get_response_log_probs, tokenize_prompt_and_output

    # Default reward function
    if reward_fn is None:
        reward_fn = r1_zero_reward_fn

    # Setup
    device = next(policy.parameters()).device
    policy.train()

    # Enable gradient checkpointing to reduce memory usage
    # This trades computation for memory by recomputing activations during backprop
    if hasattr(policy, 'gradient_checkpointing_enable'):
        policy.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # Compute derived constants
    assert config.rollout_batch_size % config.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = config.rollout_batch_size // config.group_size

    assert config.train_batch_size % config.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    assert config.train_batch_size >= config.group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )

    # Setup sampling parameters for vLLM (only if vLLM is available)
    sampling_params = None
    if vllm_instance is not None:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=config.sampling_temperature,
            max_tokens=config.sampling_max_tokens,
            min_tokens=config.sampling_min_tokens,
            n=config.group_size,
            seed=config.seed,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Training metrics
    train_step = 0
    training_history = []  # Store metrics for plotting

    # Main GRPO loop
    for grpo_step in tqdm(range(config.n_grpo_steps), desc="GRPO Steps"):
        # ====================================================================
        # Step 1: Sample a batch of questions
        # ====================================================================
        # Randomly sample n_prompts_per_rollout_batch questions
        indices = torch.randperm(len(train_prompts))[:n_prompts_per_rollout_batch].tolist()
        batch_prompts = [train_prompts[i] for i in indices]
        batch_answers = [train_answers[i] for i in indices]

        # ====================================================================
        # Step 2: Generate rollouts using vLLM
        # ====================================================================
        if vllm_instance is not None:
            # Load current policy weights into vLLM
            load_policy_into_vllm_instance(policy, vllm_instance)

            # Generate rollouts
            outputs = vllm_instance.generate(batch_prompts, sampling_params)

            # Extract responses (flatten groups)
            rollout_responses = []
            repeated_ground_truths = []
            for output, answer in zip(outputs, batch_answers):
                for completion in output.outputs:
                    rollout_responses.append(completion.text)
                    repeated_ground_truths.append(answer)
        else:
            # Fallback: use HuggingFace generate (slower)
            logger.warning("No vLLM instance provided, using HuggingFace generate")
            rollout_responses = []
            repeated_ground_truths = []

            policy.eval()
            with torch.no_grad():
                for prompt, answer in zip(batch_prompts, batch_answers):
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    for _ in range(config.group_size):
                        outputs = policy.generate(
                            **inputs,
                            max_new_tokens=config.sampling_max_tokens,
                            min_new_tokens=config.sampling_min_tokens,
                            temperature=config.sampling_temperature,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        )
                        response = tokenizer.decode(
                            outputs[0, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=False
                        )
                        # Stop at </answer>
                        if "</answer>" in response:
                            response = response.split("</answer>")[0] + "</answer>"
                        rollout_responses.append(response)
                        repeated_ground_truths.append(answer)
            policy.train()

        # ====================================================================
        # Step 3: Compute rewards and advantages
        # ====================================================================
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization,
        )

        # Log reward statistics
        if wandb_run is not None:
            wandb_run.log({
                "train/reward_mean": reward_metadata["reward_mean"],
                "train/reward_std": reward_metadata["reward_std"],
                "train/answer_reward_mean": reward_metadata["answer_reward_mean"],
                "train/format_reward_mean": reward_metadata["format_reward_mean"],
                "train/advantage_mean": reward_metadata["advantage_mean"],
                "train/advantage_std": reward_metadata["advantage_std"],
                "train_step": train_step,
            })

        logger.info(
            f"GRPO Step {grpo_step}: "
            f"reward_mean={reward_metadata['reward_mean']:.4f}, "
            f"answer_reward={reward_metadata['answer_reward_mean']:.4f}"
        )

        # Initialize step metrics for this GRPO step
        step_metrics = {
            "grpo_step": grpo_step,
            "reward_mean": reward_metadata["reward_mean"],
            "reward_std": reward_metadata["reward_std"],
            "reward_max": reward_metadata["reward_max"],
            "reward_min": reward_metadata["reward_min"],
            "answer_reward_mean": reward_metadata["answer_reward_mean"],
            "format_reward_mean": reward_metadata["format_reward_mean"],
            "advantage_mean": reward_metadata["advantage_mean"],
            "advantage_std": reward_metadata["advantage_std"],
        }

        # ====================================================================
        # Step 4: Compute old log-probs (for off-policy training)
        # ====================================================================
        # Tokenize prompts and responses
        tokenized = tokenize_prompt_and_output(
            prompt_strs=batch_prompts * config.group_size,  # Repeat prompts
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )

        # Note: Need to reorder so that rollouts are grouped by prompt
        # Currently rollout_responses is ordered by completion within each prompt
        # We need: [prompt0_comp0, prompt0_comp1, ..., prompt1_comp0, ...]
        # This is already the correct order from vLLM

        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        response_mask = tokenized["response_mask"]

        # Truncate long sequences to save memory during training
        max_len = config.max_seq_length_train
        if input_ids.shape[1] > max_len:
            logger.info(f"Truncating sequences from {input_ids.shape[1]} to {max_len} tokens")
            input_ids = input_ids[:, :max_len]
            labels = labels[:, :max_len]
            response_mask = response_mask[:, :max_len]

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        response_mask = response_mask.to(device)

        # Reshape advantages and rewards to (rollout_batch_size, 1) for broadcasting
        advantages = advantages.to(device).unsqueeze(1)
        raw_rewards = raw_rewards.to(device).unsqueeze(1)

        # Get old log-probs (no gradient)
        old_log_probs = None
        if config.loss_type == "grpo_clip":
            policy.eval()
            with torch.no_grad():
                old_result = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                old_log_probs = old_result["log_probs"]
            policy.train()

        # ====================================================================
        # Step 5: Training loop on rollout batch
        # ====================================================================
        rollout_batch_size = len(rollout_responses)

        for epoch in range(config.epochs_per_rollout_batch):
            # Shuffle indices for this epoch
            perm = torch.randperm(rollout_batch_size)

            # Process in train batches
            for batch_start in range(0, rollout_batch_size, config.train_batch_size):
                batch_end = min(batch_start + config.train_batch_size, rollout_batch_size)
                batch_indices = perm[batch_start:batch_end]

                # Process microbatches
                optimizer.zero_grad()

                accumulated_loss = 0.0
                accumulated_clip_fraction = 0.0
                n_microbatches = 0

                for mb_start in range(0, len(batch_indices), micro_train_batch_size):
                    mb_end = min(mb_start + micro_train_batch_size, len(batch_indices))
                    mb_indices = batch_indices[mb_start:mb_end]

                    # Get microbatch data
                    mb_input_ids = input_ids[mb_indices]
                    mb_labels = labels[mb_indices]
                    mb_response_mask = response_mask[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_raw_rewards = raw_rewards[mb_indices]
                    mb_old_log_probs = old_log_probs[mb_indices] if old_log_probs is not None else None

                    # Forward pass to get current log probs
                    # NOTE: return_token_entropy=False to save memory (entropy not used)
                    log_prob_result = get_response_log_probs(
                        model=policy,
                        input_ids=mb_input_ids,
                        labels=mb_labels,
                        return_token_entropy=False,
                    )
                    mb_policy_log_probs = log_prob_result["log_probs"]

                    # GRPO microbatch step
                    loss, step_metadata = grpo_microbatch_train_step(
                        policy_log_probs=mb_policy_log_probs,
                        response_mask=mb_response_mask,
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        loss_type=config.loss_type,
                        raw_rewards=mb_raw_rewards,
                        advantages=mb_advantages,
                        old_log_probs=mb_old_log_probs,
                        cliprange=config.cliprange,
                    )

                    accumulated_loss += loss.item()
                    if "clip_fraction" in step_metadata:
                        accumulated_clip_fraction += step_metadata["clip_fraction"].item()
                    n_microbatches += 1

                    # Free memory between microbatches
                    del log_prob_result, mb_policy_log_probs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    config.max_grad_norm
                )

                # Optimizer step
                optimizer.step()
                train_step += 1

                # Log training metrics
                avg_loss = accumulated_loss / n_microbatches
                avg_clip_fraction = accumulated_clip_fraction / n_microbatches if config.loss_type == "grpo_clip" else 0.0

                if wandb_run is not None:
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/grad_norm": grad_norm.item(),
                        "train/learning_rate": config.learning_rate,
                        "train_step": train_step,
                    }
                    if config.loss_type == "grpo_clip":
                        log_dict["train/clip_fraction"] = avg_clip_fraction
                    wandb_run.log(log_dict)

        # Add loss metrics to step_metrics (use last batch's values)
        step_metrics["loss"] = avg_loss
        step_metrics["grad_norm"] = grad_norm.item()
        if config.loss_type == "grpo_clip":
            step_metrics["clip_fraction"] = avg_clip_fraction

        # ====================================================================
        # Step 6: Validation (periodic)
        # ====================================================================
        if val_prompts is not None and val_answers is not None:
            if (grpo_step + 1) % config.eval_steps == 0:
                # Create sampling params only if vLLM is available
                eval_sampling_params = None
                if vllm_instance is not None:
                    from vllm import SamplingParams
                    eval_sampling_params = SamplingParams(
                        temperature=config.sampling_temperature,
                        max_tokens=config.sampling_max_tokens,
                        min_tokens=config.sampling_min_tokens,
                        n=1,
                        seed=config.seed,
                        stop=["</answer>"],
                        include_stop_str_in_output=True,
                    )

                val_reward = evaluate_policy(
                    policy=policy,
                    tokenizer=tokenizer,
                    prompts=val_prompts[:1024],  # Evaluate on subset
                    answers=val_answers[:1024],
                    reward_fn=reward_fn,
                    vllm_instance=vllm_instance,
                    sampling_params=eval_sampling_params,
                    max_tokens=config.sampling_max_tokens,
                    temperature=config.sampling_temperature,
                )

                logger.info(f"Validation reward: {val_reward:.4f}")

                if wandb_run is not None:
                    wandb_run.log({
                        "eval/answer_reward": val_reward,
                        "eval_step": grpo_step + 1,
                    })

                # Add validation metrics
                step_metrics["val_reward"] = val_reward

        # Append step metrics to training history
        training_history.append(step_metrics)

        # Save training history periodically (every 10 steps)
        if (grpo_step + 1) % 10 == 0:
            import json
            history_path = os.path.join(config.output_dir, "training_history.json")
            with open(history_path, "w") as f:
                json.dump(training_history, f, indent=2)

        # ====================================================================
        # Step 7: Save checkpoint (periodic)
        # ====================================================================
        if (grpo_step + 1) % config.save_steps == 0:
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{grpo_step + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # Save final model
    final_dir = os.path.join(config.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved final model to {final_dir}")

    # Save final training history
    import json
    history_path = os.path.join(config.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    return policy


def evaluate_policy(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    answers: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    vllm_instance: "LLM | None" = None,
    sampling_params: "SamplingParams | None" = None,
    max_tokens: int = 1024,
    temperature: float = 1.0,
) -> float:
    """
    Evaluate the policy on a set of prompts and compute average reward.

    Args:
        policy: Policy model
        tokenizer: Tokenizer
        prompts: Evaluation prompts
        answers: Ground-truth answers
        reward_fn: Reward function
        vllm_instance: Optional vLLM instance for fast inference
        sampling_params: Sampling parameters

    Returns:
        Average answer reward
    """
    if vllm_instance is not None:
        load_policy_into_vllm_instance(policy, vllm_instance)
        outputs = vllm_instance.generate(prompts, sampling_params)

        total_reward = 0.0
        for output, answer in zip(outputs, answers):
            response = output.outputs[0].text
            reward_info = reward_fn(response, answer)
            total_reward += reward_info["answer_reward"]

        return total_reward / len(prompts)
    else:
        # Fallback to HuggingFace generate
        device = next(policy.parameters()).device
        policy.eval()

        total_reward = 0.0
        with torch.no_grad():
            for prompt, answer in tqdm(zip(prompts, answers), total=len(prompts), desc="Evaluating"):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = policy.generate(
                    **inputs,
                    max_new_tokens=sampling_params.max_tokens if sampling_params else max_tokens,
                    temperature=sampling_params.temperature if sampling_params else temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                response = tokenizer.decode(
                    outputs[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=False
                )
                if "</answer>" in response:
                    response = response.split("</answer>")[0] + "</answer>"
                reward_info = reward_fn(response, answer)
                total_reward += reward_info["answer_reward"]

        policy.train()
        return total_reward / len(prompts)
