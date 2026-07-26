# tableseq/training/utils.py

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor


def pad_sequences_to_length(
    sequences: Sequence[Tensor],
    max_length: int,
    pad_token_id: int,
) -> Tensor:
    """
    Pad or truncate 1D token sequences to a fixed length.

    Args:
        sequences:
            List of 1D LongTensor sequences.
        max_length:
            Final sequence length.
        pad_token_id:
            Padding token id.

    Returns:
        Tensor of shape (B, max_length).
    """
    batch_size = len(sequences)
    out = torch.full(
        (batch_size, max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=sequences[0].device,
    )

    for i, seq in enumerate(sequences):
        seq = seq[:max_length]
        out[i, : seq.numel()] = seq

    return out


def build_decoder_inputs_and_labels(
    labels: Tensor,
    labels_len: Sequence[int],
    token_prompts: Sequence[Tensor],
    pad_token_id: int,
    max_length: int,
    mask_prompt_loss: bool = True,
    legacy_prompt_masking: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Build decoder input_ids and target labels from full token sequences.

    This reproduces the old training logic:

        input_ids = y[:seq_len]
        labels    = y[1:seq_len]

    The prompt part is masked with pad_token_id so it does not contribute to
    the decoder loss.

    Args:
        labels:
            Tensor of shape (B, L), usually containing <s> + prompt + target + </s>.
        labels_len:
            True sequence lengths before padding.
        token_prompts:
            Prompt tokens for each sample, for example [<html>].
        pad_token_id:
            Token id ignored by the decoder loss.
        max_length:
            Maximum decoder length.
        mask_prompt_loss:
            Whether to mask prompt tokens in the loss.
        legacy_prompt_masking:
            If True, reproduces the original code exactly:
                y[i, :prompt_len] = pad
            If False, masks start token + prompt:
                y[i, :prompt_len + 1] = pad

    Returns:
        input_ids:
            Decoder input ids, shape (B, max_length).
        target_labels:
            Shifted decoder labels, shape (B, max_length).
    """
    y = labels.clone()

    if mask_prompt_loss:
        for i, prompt in enumerate(token_prompts):
            prompt_len = int(prompt.numel())

            if legacy_prompt_masking:
                end = prompt_len
            else:
                end = prompt_len + 1

            y[i, :end] = pad_token_id

    input_sequences = [
        labels[i, : int(seq_len)]
        for i, seq_len in enumerate(labels_len)
    ]

    target_sequences = [
        y[i, 1 : int(seq_len)]
        for i, seq_len in enumerate(labels_len)
    ]

    input_ids = pad_sequences_to_length(
        input_sequences,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )

    target_labels = pad_sequences_to_length(
        target_sequences,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )

    return input_ids, target_labels


def apply_teacher_forcing_noise(
    labels: Tensor,
    labels_len: Sequence[int],
    token_prompts: Sequence[Tensor],
    pad_token_id: int,
    vocab_size: int,
    error_rate: float,
) -> Tensor:
    """
    Simple version of the old teacher-forcing corruption.

    It randomly replaces target tokens after the prompt. This can be disabled
    during fine-tuning by setting error_rate=0.
    """
    if error_rate <= 0:
        return labels

    noisy = labels.clone()

    for b, seq_len in enumerate(labels_len):
        prompt_len = int(token_prompts[b].numel())
        start = prompt_len + 1

        for t in range(start, int(seq_len)):
            if noisy[b, t].item() == pad_token_id:
                continue

            if torch.rand(1).item() < error_rate:
                noisy[b, t] = torch.randint(
                    low=0,
                    high=vocab_size,
                    size=(1,),
                    device=labels.device,
                )

    return noisy