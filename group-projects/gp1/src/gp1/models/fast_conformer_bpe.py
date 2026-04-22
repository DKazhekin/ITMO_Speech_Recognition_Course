"""Fast Conformer BPE acoustic encoder for GP1 Russian spoken numbers ASR.

Baseline #5 of 5. Architecture (CONTRACTS.md §5, baselines table):

    Input  [B, 80, T]       log-mel spectrogram
    SubsampleConv(80 -> d_model, factor=4)    # strided 2D conv frontend
    16 x ConformerBlock(d_model, n_heads, ff_ratio, conv_kernel, dropout)
    Linear(d_model -> vocab_size)
    .float() + log_softmax(dim=-1)
    Output [B, T/4, vocab_size]

Deviation from paper
--------------------
Rekesh et al. (2023) use ``subsample_factor=8`` (two 4x steps), which
yields T'/U_max ≈ 0.75 for the GP1 worst-case 3.5 s sample — a ratio
below 1, which breaks CTC.  Per gp1_subsample_and_kernel_constraints.md
(project memory), s=8 is **hard-forbidden** in this codebase.  We use
``subsample_factor=4`` (T'/U_max ≈ 2.9 for BPE U_max~30), which retains
the "fast" character of the architecture (single striding block, no
separate time-reduction layers) while satisfying the CTC invariant
``T' >= 2 * U_max``.

Hyperparameter choices for ~4.2M params
-----------------------------------------
    d_model      = 96
    n_blocks     = 16
    n_heads      = 4       (head_dim = 24)
    ff_ratio     = 4
    conv_kernel  = 9       (odd, shorter than Conformer default 31 — faster)
    dropout      = 0.1
    vocab_size   = 257     (BPE-256 + 1 blank via BPEVocab id-shift)

Approximate param breakdown:
    SubsampleConv(80→96, factor=4)  : ~0.30M
    16 × ConformerBlock(96)          : ~3.86M
    Linear head (96→256)            : ~0.025M
    Total                           : ~4.19M  (< 5M budget)

References
----------
- Rekesh et al., Fast Conformer (2023): https://arxiv.org/abs/2305.05084
- NeMo FastConformerEncoder:
  https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py
- Gulati et al., Conformer (2020): https://arxiv.org/abs/2005.08100
- CONTRACTS.md §5: ASREncoder Protocol, EncoderOutput
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gp1.models.base import EncoderOutput
from gp1.models.common import ConformerBlock, SubsampleConv

log = logging.getLogger(__name__)

# Allowed subsample factor — s=8 is hard-forbidden (CTC breaks on GP1).
_ALLOWED_SUBSAMPLE_FACTOR: int = 4


class FastConformerBPE(nn.Module):
    """Fast Conformer encoder with BPE vocabulary for GP1 CTC ASR.

    Conforms to ``ASREncoder`` Protocol (CONTRACTS.md §5).

    Parameters
    ----------
    vocab_size : int
        Output vocabulary size including CTC blank at index 0.
        Matches BPEVocab.vocab_size (default 256 for BPE-256).
    d_model : int
        Model hidden dimension.  Default 144.
    n_blocks : int
        Number of ConformerBlocks.  Default 16.
    n_heads : int
        Number of attention heads in each ConformerBlock.  Default 4.
    ff_ratio : int
        Feed-forward expansion ratio.  Default 4.
    conv_kernel : int
        Depthwise conv kernel size inside ConvModule (must be odd).  Default 9.
    dropout : float
        Dropout probability applied in SubsampleConv projection, each
        ConformerBlock, and the head.  Default 0.1.
    subsample_factor : int
        Time-axis subsampling.  MUST be 4.  Any other value raises ValueError.
    """

    vocab_size: int
    subsample_factor: int

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        n_blocks: int = 16,
        n_heads: int = 4,
        ff_ratio: int = 4,
        conv_kernel: int = 9,
        dropout: float = 0.1,
        subsample_factor: int = 4,
    ) -> None:
        super().__init__()

        if subsample_factor != _ALLOWED_SUBSAMPLE_FACTOR:
            raise ValueError(
                f"subsample_factor must be {_ALLOWED_SUBSAMPLE_FACTOR} "
                f"(s=8 is forbidden — CTC breaks on GP1 worst-case 3.5s sample; "
                f"see gp1_subsample_and_kernel_constraints.md). "
                f"Got subsample_factor={subsample_factor}."
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.vocab_size = vocab_size
        self.subsample_factor = subsample_factor

        # --- Frontend: strided 2D conv, /4 time reduction -------------------
        # SubsampleConv with factor=4 applies two stride-2 stages internally.
        # Output: [B, T/4, d_model].
        self.subsample = SubsampleConv(
            n_mels=80, d_out=d_model, factor=subsample_factor
        )
        self.subsample_dropout = nn.Dropout(dropout)

        # --- Conformer encoder stack ----------------------------------------
        # Reuse ConformerBlock from common.py (depthwise conv module already
        # implemented with GLU + BN — "fast" depthwise-sep variant).
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_ratio=ff_ratio,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        # --- CTC projection head --------------------------------------------
        # Linear + float() + log_softmax forms the fp32 island.
        self.head = nn.Linear(d_model, vocab_size)

        n_params = sum(p.numel() for p in self.parameters())
        _PARAM_BUDGET = 5_000_000
        if n_params > _PARAM_BUDGET:
            raise ValueError(
                f"FastConformerBPE exceeds 5M param budget: "
                f"{n_params / 1e6:.2f}M > {_PARAM_BUDGET / 1e6:.1f}M. "
                f"Reduce d_model or n_blocks."
            )
        log.info(
            "FastConformerBPE initialised: d_model=%d, n_blocks=%d, "
            "vocab_size=%d, subsample_factor=%d, params=%d",
            d_model,
            n_blocks,
            vocab_size,
            subsample_factor,
            n_params,
        )

    def forward(self, mel: Tensor, mel_lengths: Tensor) -> EncoderOutput:
        """Compute CTC log-probabilities.

        Parameters
        ----------
        mel : Tensor
            ``[B, 80, T]`` float32 log-mel spectrogram.
        mel_lengths : Tensor
            ``[B]`` int64 — unpadded frame counts before subsampling.

        Returns
        -------
        EncoderOutput
            ``log_probs``      : ``[B, T', V]`` float32
            ``output_lengths`` : ``[B]`` int64, T' = ceil(ceil(T/2)/2) (two ceil-div passes)
            ``intermediate``   : ``None`` (no InterCTC tap for this encoder)
        """
        if mel.dim() != 3:
            raise ValueError(f"mel must be [B, n_mels, T], got {tuple(mel.shape)}")
        if mel_lengths.dim() != 1:
            raise ValueError(f"mel_lengths must be [B], got {tuple(mel_lengths.shape)}")

        # Frontend: [B, 80, T] -> [B, T/4, d_model]
        x: Tensor = self.subsample(mel)
        x = self.subsample_dropout(x)

        # Conformer encoder stack: each block [B, T', d_model] -> [B, T', d_model]
        for block in self.blocks:
            x = block(x)

        # CTC head — fp32 island: always cast to float32 before log_softmax
        # to ensure CTC loss stability even when model runs under fp16 autocast.
        logits = self.head(x).float()  # [B, T', V] float32
        log_probs = F.log_softmax(logits, dim=-1)

        # Subsampled lengths: SubsampleConv uses two stride-2 stages internally,
        # each producing ceil(T/2) frames. Apply ceil-div twice (H5 fix).
        # ceil(T/2) = (T + 1) // 2; apply twice for factor=4.
        lengths_stage1 = (mel_lengths + 1) // 2
        output_lengths = ((lengths_stage1 + 1) // 2).to(torch.long)

        return EncoderOutput(
            log_probs=log_probs,
            output_lengths=output_lengths,
            intermediate=None,
        )
