"""
Minimal utilities for GRPO training.

This module provides the essential tokenization and log-probability computation
functions needed for GRPO training, extracted from the full SFT module.

Key components:
    - tokenize_prompt_and_output: Tokenize prompts and outputs with response masks
    - get_response_log_probs: Compute per-token log probabilities from a causal LM
    - compute_entropy: Compute entropy over vocabulary dimension
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_size = len(prompt_strs)
    assert len(output_strs) == batch_size, "prompt_strs and output_strs must have same length"

    all_input_ids = []
    all_labels = []
    all_response_masks = []
    prompt_and_output_lens = []

    for prompt, output in zip(prompt_strs, output_strs):
        # Tokenize prompt and output separately (no special tokens)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)

        # Concatenate prompt and output
        full_ids = prompt_ids + output_ids
        prompt_and_output_lens.append(len(full_ids))

        # Create input_ids and labels (shifted by 1 for next token prediction)
        # input_ids: all tokens except the last
        # labels: all tokens except the first
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # Create response mask: 1 for response tokens in labels, 0 for prompt tokens
        # Since labels are shifted, the response starts at position (prompt_len - 1)
        response_mask = [0] * len(labels)
        response_start = max(0, len(prompt_ids) - 1)
        for i in range(response_start, len(labels)):
            response_mask[i] = 1

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_response_masks.append(response_mask)

    # Pad to max length (max(prompt_and_output_lens) - 1)
    max_len = max(prompt_and_output_lens) - 1
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input_ids = []
    padded_labels = []
    padded_response_masks = []

    for input_ids, labels, response_mask in zip(all_input_ids, all_labels, all_response_masks):
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_token_id] * pad_len
            labels = labels + [-100] * pad_len  # -100 is ignored in loss
            response_mask = response_mask + [0] * pad_len

        padded_input_ids.append(input_ids)
        padded_labels.append(labels)
        padded_response_masks.append(response_mask)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "response_mask": torch.tensor(padded_response_masks, dtype=torch.float),
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Uses numerically stable computation via log_softmax.

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length). The entropy for each
        next-token prediction.
    """
    # Use log_softmax for numerical stability
    # log_softmax(x) = x - logsumexp(x)
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (batch_size, seq_length, vocab_size)

    # Convert to probabilities for entropy calculation
    probs = torch.exp(log_probs)

    # Entropy: H(p) = -sum(p(x) * log(p(x)))
    # Using the numerically stable form: -sum(probs * log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch_size, seq_length)

    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    seq_chunk_size: int = 256,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities (given the previous tokens) from
    a causal language model, and optionally the entropy of the model's next-token
    distribution.

    Memory-optimized: processes logsumexp in chunks along sequence dimension to
    avoid materializing the full (batch, seq, vocab) tensor at once.

    Args:
        model: PreTrainedModel, HuggingFace model used for scoring (placed on the
            correct device and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor of shape (batch_size, sequence_length), concatenated
            prompt + response tokens as produced by your tokenization method.
        labels: torch.Tensor of shape (batch_size, sequence_length), labels as
            produced by your tokenization method.
        return_token_entropy: bool, If True, also return per-token entropy by
            calling compute_entropy. WARNING: This is memory-intensive!
        seq_chunk_size: int, chunk size for processing logsumexp along sequence dim.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": shape (batch_size, sequence_length), conditional log-probabilities
                log p_θ(x_t | x_{<t}).
            "token_entropy": optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if return_token_entropy=True).
    """
    # Get logits from model
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (batch_size, seq_length, vocab_size)

    batch_size, seq_length, vocab_size = logits.shape

    # Handle -100 labels (padding) - temporarily replace with 0
    labels_for_gather = labels.clone()
    labels_for_gather[labels == -100] = 0

    # Gather logits at label positions: (batch_size, seq_length)
    gathered_logits = torch.gather(
        logits, dim=-1, index=labels_for_gather.unsqueeze(-1)
    ).squeeze(-1)

    # Memory-efficient logsumexp: process in chunks along sequence dimension
    # This avoids materializing the full (batch, seq, vocab) tensor in float32
    logsumexp_chunks = []
    for seq_start in range(0, seq_length, seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, seq_length)
        chunk_logits = logits[:, seq_start:seq_end, :]  # (batch, chunk_size, vocab)
        chunk_logsumexp = torch.logsumexp(chunk_logits.float(), dim=-1)
        logsumexp_chunks.append(chunk_logsumexp)

    logsumexp = torch.cat(logsumexp_chunks, dim=1)  # (batch_size, seq_length)

    # log_softmax(x)[i] = x[i] - logsumexp(x)
    log_probs = gathered_logits.float() - logsumexp  # (batch_size, seq_length)

    result = {"log_probs": log_probs}

    if return_token_entropy:
        # WARNING: This is memory-intensive as it requires full vocab probabilities
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy

    # Explicitly delete logits to free memory
    del logits, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
