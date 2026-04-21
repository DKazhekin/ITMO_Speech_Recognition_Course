"""Greedy CTC decoding.

Implements argmax-collapse-strip-blank decoding for batched log-probability
tensors. Does NOT import from wav2vec2decoder.py; the collapse logic is
re-implemented locally to respect per-sample output_lengths.

References:
  - CONTRACTS.md §7: greedy_decode signature
  - Graves et al. (2006) CTC: collapse consecutive duplicate tokens, then
    remove blank tokens.
"""

from __future__ import annotations

import logging

import torch

from gp1.text.vocab import CharVocab

log = logging.getLogger(__name__)

BLANK_ID: int = 0


def greedy_decode(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    vocab: CharVocab,
) -> list[str]:
    """Greedy CTC decoding over a batch of log-probability tensors.

    Algorithm per sample:
      1. Truncate to output_lengths[i] timesteps.
      2. Argmax over the vocabulary dimension.
      3. Collapse consecutive duplicate token ids.
      4. Remove blank tokens (id 0).
      5. Delegate to vocab.decode (handles id -> char mapping).

    Args:
        log_probs: [B, T, V] float tensor of log probabilities (or logits;
            only relative order matters because we take argmax).
        output_lengths: [B] int64 tensor, actual sequence lengths before padding.
        vocab: CharVocab instance used for decoding ids to characters.

    Returns:
        List of B decoded strings, one per sample in the batch.
    """
    assert log_probs.ndim == 3, (
        f"log_probs must be [B, T, V], got shape {tuple(log_probs.shape)}"
    )
    assert output_lengths.ndim == 1, (
        f"output_lengths must be [B], got shape {tuple(output_lengths.shape)}"
    )
    B = log_probs.size(0)
    assert output_lengths.size(0) == B, (
        f"Batch size mismatch: log_probs B={B}, output_lengths B={output_lengths.size(0)}"
    )

    results: list[str] = []
    for i in range(B):
        length = int(output_lengths[i].item())
        # Truncate to valid timesteps
        sample_lp = log_probs[i, :length, :]  # [T_i, V]

        # Argmax over vocabulary dimension
        argmax_ids: list[int] = sample_lp.argmax(dim=-1).tolist()

        # CTC collapse: remove consecutive duplicates, then strip blanks
        # Delegating directly to vocab.decode which already implements this.
        decoded = vocab.decode(argmax_ids)
        results.append(decoded)

    log.debug("greedy_decode: batch_size=%d decoded %d strings", B, len(results))
    return results
