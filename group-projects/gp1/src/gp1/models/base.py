"""ASR encoder Protocol and shared output dataclass.

CONTRACTS.md §5: every concrete encoder implementation (quartznet,
crdnn, efficient_conformer, fast_conformer_bpe) conforms to
``ASREncoder`` and returns ``EncoderOutput`` from forward().

References:
- QuartzNet: Kriman et al., https://arxiv.org/abs/1910.10261
- Conformer: Gulati et al., https://arxiv.org/abs/2005.08100
- InterCTC: Lee & Watanabe, https://arxiv.org/abs/2102.03216
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class EncoderOutput:
    """Output contract for all acoustic encoders used in GP1.

    Fields
    ------
    log_probs : torch.Tensor
        Log-softmax over the vocabulary, shape ``[B, T', V]`` float32.
        Time dimension ``T'`` equals ``ceil(T / subsample_factor)`` for
        fixed-stride encoders.
    output_lengths : torch.Tensor
        ``[B]`` int64 - the effective (non-padded) length of each
        sequence along ``T'`` after subsampling. Used by CTC loss and
        beam search decoders.
    intermediate : torch.Tensor | None
        ``[B, T_mid, D_mid]`` mid-encoder features for InterCTC
        auxiliary loss, or ``None`` when the encoder does not expose a
        mid-layer tap.
    """

    log_probs: torch.Tensor
    output_lengths: torch.Tensor
    intermediate: torch.Tensor | None = None


class ASREncoder(Protocol):
    """Structural interface every GP1 acoustic encoder must satisfy.

    Implementations expose ``vocab_size`` and ``subsample_factor`` as
    attributes so the trainer can validate CTC shape invariants
    (``T' >= 2 * U_max`` per CONTRACTS.md §5) before the first epoch.
    """

    vocab_size: int
    subsample_factor: int

    def forward(
        self,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> EncoderOutput:
        """Compute encoder log-probabilities.

        Parameters
        ----------
        mel : torch.Tensor
            ``[B, n_mels, T]`` float32 log-mel spectrogram.
        mel_lengths : torch.Tensor
            ``[B]`` int64 - unpadded frame count per sample.
        """
        ...
