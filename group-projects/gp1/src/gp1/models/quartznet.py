"""QuartzNet-10x4 acoustic encoder for GP1 Russian spoken numbers ASR.

Architecture (CONTRACTS.md §5, plan "Baselines" table #1):

    Input  [B, 80, T]        log-mel spectrogram
    Prologue: TCSConvBlock(80  -> 256, k=33, stride=2)  # s=2 subsample
    Block B1: 4 x TCSConvBlock(256 -> 256, k=33, residual=True)
    Block B2: 4 x TCSConvBlock(256 -> 256, k=39, residual=True)  <- intermediate tap
    Block B3: 4 x TCSConvBlock(256 -> 512, k=51, residual=True)
    Block B4: 4 x TCSConvBlock(512 -> 512, k=63, residual=True)
    Block B5: 4 x TCSConvBlock(512 -> 512, k=75, residual=True)
    Epilogue: TCSConvBlock(512 -> 512, k=87, dilation=2)
    Head: Conv1d(512 -> vocab_size, k=1) -> LogSoftmax(dim=1)
    Output [B, T/2, vocab_size]

Intermediate output is tapped after B2 (depth = 2/5 ≈ 0.4, closest
practical mid-point) for the InterCTC auxiliary loss head.

Param count target: ~4.0M (well under 5M budget).

Subsample invariant: s=2 only. s=8 is FORBIDDEN (CTC breaks at T'/U_max
< 1 for 3.5s worst-case sample; see gp1_subsample_and_kernel_constraints).

References:
- Kriman et al., QuartzNet (2019): https://arxiv.org/abs/1910.10261
- NeMo reference impl:
  https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conv_asr.py
- CONTRACTS.md §5: ASREncoder Protocol, EncoderOutput
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from torch import Tensor, nn

from gp1.models.base import EncoderOutput
from gp1.models.common import TCSConvBlock

log = logging.getLogger(__name__)

# Supported subsample factor — hard constraint from CTC alignment theory.
_ALLOWED_SUBSAMPLE_FACTOR = 2


def _make_quartznet_block(
    c_in: int,
    c_out: int,
    kernel: int,
    n_repeats: int,
    dropout: float,
) -> nn.Sequential:
    """Build a QuartzNet B_R block: n_repeats TCSConvBlocks.

    Residual is enabled only when c_in == c_out AND stride == 1 (enforced
    inside TCSConvBlock). The first sub-block handles the channel transition
    and never has a residual; subsequent sub-blocks (same c) do.
    """
    layers: list[nn.Module] = []
    for i in range(n_repeats):
        in_ch = c_in if i == 0 else c_out
        layers.append(
            TCSConvBlock(
                c_in=in_ch,
                c_out=c_out,
                kernel_size=kernel,
                stride=1,
                dropout=dropout,
                residual=True,
            )
        )
    return nn.Sequential(*layers)


class QuartzNet10x4(nn.Module):
    """QuartzNet-10x4 CTC acoustic encoder.

    Conforms to ``ASREncoder`` Protocol (CONTRACTS.md §5).

    Parameters
    ----------
    vocab_size : int
        Output vocabulary size (including CTC blank at index 0).
        Default: 35 (blank + space + 33 Russian chars).
    d_model : int
        Base channel width for B1/B2 blocks. Default: 256.
    dropout : float
        Dropout probability applied inside every TCSConvBlock. Default: 0.1.
    subsample_factor : int
        Time-axis subsampling. MUST be 2. Any other value raises ValueError.
    """

    vocab_size: int
    subsample_factor: int

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        dropout: float = 0.1,
        subsample_factor: int = 2,
    ) -> None:
        super().__init__()
        if subsample_factor != _ALLOWED_SUBSAMPLE_FACTOR:
            raise ValueError(
                f"subsample_factor must be {_ALLOWED_SUBSAMPLE_FACTOR} "
                f"(s=8 forbidden — CTC breaks; see gp1_subsample_and_kernel_constraints). "
                f"Got subsample_factor={subsample_factor}."
            )

        self.vocab_size = vocab_size
        self.subsample_factor = subsample_factor

        # Prologue: stride=2 → halves the time dimension.
        # residual=False because c_in (80) != c_out (d_model) and stride != 1.
        self.prologue = TCSConvBlock(
            c_in=80,
            c_out=d_model,
            kernel_size=33,
            stride=2,
            dropout=dropout,
            residual=False,
        )

        # Five QuartzNet blocks (B1–B5), each with 4 sub-blocks (R=4).
        self.block1 = _make_quartznet_block(
            d_model, d_model, kernel=33, n_repeats=4, dropout=dropout
        )
        self.block2 = _make_quartznet_block(
            d_model, d_model, kernel=39, n_repeats=4, dropout=dropout
        )

        # After B2 we tap the intermediate features for InterCTC.
        # Stored as a buffer to expose shape without a separate projection.
        self._d_mid = d_model  # 256

        d_wide = 512  # channels for B3–B5 and epilogue
        self.block3 = _make_quartznet_block(
            d_model, d_wide, kernel=51, n_repeats=4, dropout=dropout
        )
        self.block4 = _make_quartznet_block(
            d_wide, d_wide, kernel=63, n_repeats=4, dropout=dropout
        )
        self.block5 = _make_quartznet_block(
            d_wide, d_wide, kernel=75, n_repeats=4, dropout=dropout
        )

        # Epilogue: dilated TCS conv (dilation=2) for wider receptive field.
        self.epilogue = TCSConvBlock(
            c_in=d_wide,
            c_out=d_wide,
            kernel_size=87,
            stride=1,
            dropout=dropout,
            residual=False,
            dilation=2,
        )

        # CTC head: pointwise conv → log-softmax.
        self.head = nn.Conv1d(
            in_channels=d_wide,
            out_channels=vocab_size,
            kernel_size=1,
            bias=True,
        )

        n_params = sum(p.numel() for p in self.parameters())
        log.info("QuartzNet10x4 initialised: %d params (target ~4.0M)", n_params)

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
            ``output_lengths`` : ``[B]`` int64, T' = ceil(T / 2) = (T + 1) // 2
            ``intermediate``   : ``[B, T', 256]`` float32 (after block2)
        """
        if mel.dim() != 3:
            raise ValueError(f"mel must be [B, n_mels, T], got {tuple(mel.shape)}")
        if mel_lengths.dim() != 1:
            raise ValueError(f"mel_lengths must be [B], got {tuple(mel_lengths.shape)}")

        # Prologue: [B, 80, T] -> [B, 256, T/2]
        x = self.prologue(mel)

        # B1 + B2
        x = self.block1(x)
        x = self.block2(x)

        # Tap intermediate BEFORE B3: [B, 256, T'] -> [B, T', 256]
        intermediate = x.transpose(1, 2)

        # B3 – B5
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Epilogue + head: [B, 512, T'] -> [B, V, T']
        x = self.epilogue(x)
        x = self.head(x)

        # log-softmax along vocab dimension, then transpose to [B, T', V]
        log_probs = torch.nn.functional.log_softmax(x, dim=1).transpose(1, 2)

        # Compute subsampled lengths using ceil-div to match strided conv output:
        # Conv1d with stride=s produces ceil(T/s) frames, not floor(T/s).
        # ceil(T/s) = (T + s - 1) // s  (H5 fix).
        output_lengths = (
            (mel_lengths + self.subsample_factor - 1) // self.subsample_factor
        ).to(torch.long)

        return EncoderOutput(
            log_probs=log_probs,
            output_lengths=output_lengths,
            intermediate=intermediate,
        )
