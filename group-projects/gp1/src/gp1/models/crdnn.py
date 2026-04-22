"""CRDNN (Convolutional-Recurrent Deep Neural Network) acoustic encoder.

Architecture baseline #3 for GP1 Russian spoken numbers ASR (CONTRACTS.md §5).

Research summary (Phase 0):
- SpeechBrain CRDNN (Ravanelli et al. 2021): 2D-conv frontend + BiGRU + linear head;
  canonical ref: https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/CRDNN.py
- ESPnet2 RNN encoder: similar Conv2D → BLSTM stack;
  ref: https://github.com/espnet/espnet/tree/master/espnet2/asr/encoder
- Both projects use stride on the frequency axis only (keeping time intact) for the
  Conv2D frontend when CTC is the objective, which is the approach adopted here.

Design notes:
- subsample_factor=1: stride=(1,2) halves frequency axis, NOT time.
  Meets CTC constraint: T'/U_max = 350/59 ≈ 5.9 >> 2 (safe for any speech speed).
- subsample_factor=2: first Conv2D block uses stride=(2,2), halving both axes once.
  T'/U_max = 175/59 ≈ 2.97 (still safe, matches QuartzNet baseline).
- Target param count: ~3.6M with d_cnn=64, rnn_hidden=256, rnn_layers=2.
- fp32 island: head output cast to float32 before log_softmax (CONTRACTS.md §11).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gp1.models.base import EncoderOutput

log = logging.getLogger(__name__)

# Hard constraint: s=8 forbidden (CTC breaks at T'/U_max < 1 for 3.5s worst-case).
# s=4 is tight; only 1 and 2 are unconditionally safe.
_ALLOWED_SUBSAMPLE_FACTORS = frozenset({1, 2})


@dataclass(frozen=True)
class CRDNNConfig:
    """Hyper-parameters for CRDNN encoder.

    Parameters
    ----------
    vocab_size : int
        Output vocabulary size (CTC blank at index 0). Default: 35.
    d_cnn : int
        Number of output channels in each Conv2D block. Default: 64.
    rnn_hidden : int
        Per-direction hidden size of the BiGRU. Default: 256.
    rnn_layers : int
        Number of BiGRU layers. Default: 2.
    dropout : float
        Dropout probability applied after each BiGRU layer and after conv blocks.
        Default: 0.15.
    subsample_factor : int
        Time-axis subsampling factor. Must be 1 or 2. Default: 1.
    n_mels : int
        Number of input mel filter-bank channels. Default: 80.
    """

    vocab_size: int = 35
    d_cnn: int = 64
    rnn_hidden: int = 256
    rnn_layers: int = 2
    dropout: float = 0.15
    subsample_factor: int = 1
    n_mels: int = 80


class _Conv2DBlock(nn.Module):
    """Single 2D-conv block: Conv2d → BatchNorm2d → ReLU → Dropout.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
        Square kernel applied to (freq, time) dimensions. Default: 3.
    stride : tuple[int, int]
        Stride on (freq, time). Default: (2, 1) — halves frequency only.
    dropout : float
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: tuple[int, int] = (2, 1),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, C_in, F, T] -> [B, C_out, F', T']."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


def _build_conv_frontend(
    d_cnn: int,
    subsample_factor: int,
    dropout: float,
) -> tuple[nn.Sequential, int]:
    """Build the 2-block Conv2D frontend.

    Returns the ``nn.Sequential`` module and the flattened feature dimension
    after the two conv blocks applied to 80 mel bins.

    The time axis is only downsampled when subsample_factor==2, in which case
    the first block uses stride=(2, 2) (both freq and time halved once).
    For subsample_factor==1 both blocks use stride=(2, 1) (freq only).
    """
    if subsample_factor == 2:
        stride_block1: tuple[int, int] = (2, 2)
    else:
        stride_block1 = (2, 1)

    block1 = _Conv2DBlock(
        in_channels=1,
        out_channels=d_cnn,
        kernel_size=3,
        stride=stride_block1,
        dropout=dropout,
    )
    block2 = _Conv2DBlock(
        in_channels=d_cnn,
        out_channels=d_cnn,
        kernel_size=3,
        stride=(2, 1),  # always halve freq only in block 2
        dropout=dropout,
    )

    # Compute frequency dimension after two Conv2D blocks on 80 mel bins.
    # Conv2d with k=3, p=1, s_freq: out_freq = ceil(in_freq / s_freq)
    # which equals (in_freq + 1) // 2 for s=2 (integer ceiling).
    freq_after_block1 = (80 + 1) // 2  # stride_freq=2 -> 40
    freq_after_block2 = (freq_after_block1 + 1) // 2  # stride_freq=2 -> 20
    flat_dim = d_cnn * freq_after_block2  # 64 * 20 = 1280

    frontend = nn.Sequential(block1, block2)
    return frontend, flat_dim


class CRDNN(nn.Module):
    """CRDNN acoustic encoder: Conv2D frontend + BiGRU + CTC head.

    Conforms to ``ASREncoder`` Protocol (CONTRACTS.md §5).

    Parameters
    ----------
    vocab_size : int
        Output vocabulary size (CTC blank at index 0). Default: 35.
    d_cnn : int
        Number of channels in each Conv2D block. Default: 64.
    rnn_hidden : int
        Per-direction hidden size of the BiGRU. Default: 256.
    rnn_layers : int
        Number of BiGRU layers stacked. Default: 2.
    dropout : float
        Dropout probability. Default: 0.15.
    subsample_factor : int
        Time-axis downsampling; must be 1 or 2. Default: 1.
    """

    vocab_size: int
    subsample_factor: int

    def __init__(
        self,
        vocab_size: int = 35,
        d_cnn: int = 64,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.15,
        subsample_factor: int = 1,
    ) -> None:
        super().__init__()
        if subsample_factor not in _ALLOWED_SUBSAMPLE_FACTORS:
            raise ValueError(
                f"subsample_factor must be one of {sorted(_ALLOWED_SUBSAMPLE_FACTORS)} "
                f"(s>=4 is forbidden — CTC alignment breaks; "
                f"see gp1_subsample_and_kernel_constraints). "
                f"Got subsample_factor={subsample_factor}."
            )

        self.vocab_size = vocab_size
        self.subsample_factor = subsample_factor

        # Conv2D frontend: [B, 1, 80, T] -> [B, d_cnn, 20, T']
        self.conv_frontend, flat_dim = _build_conv_frontend(
            d_cnn=d_cnn,
            subsample_factor=subsample_factor,
            dropout=dropout,
        )

        # Bidirectional GRU: input [B, T', flat_dim] -> [B, T', rnn_hidden*2]
        self.rnn = nn.GRU(
            input_size=flat_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        rnn_out_dim = rnn_hidden * 2  # bidirectional

        # Post-RNN normalisation + dropout
        self.layer_norm = nn.LayerNorm(rnn_out_dim)
        self.out_dropout = nn.Dropout(dropout)

        # CTC head: linear projection to vocab
        self.head = nn.Linear(rnn_out_dim, vocab_size)

        n_params = sum(p.numel() for p in self.parameters())
        log.info(
            "CRDNN initialised: %d params (target <5M), subsample_factor=%d",
            n_params,
            subsample_factor,
        )

    def _run_conv_frontend(self, mel: Tensor) -> Tensor:
        """Apply Conv2D frontend.

        Parameters
        ----------
        mel : Tensor
            ``[B, n_mels, T]`` float32

        Returns
        -------
        Tensor
            ``[B, T', flat_dim]`` float32
        """
        # [B, n_mels, T] -> [B, 1, n_mels, T]
        x = mel.unsqueeze(1)
        # [B, 1, n_mels, T] -> [B, d_cnn, freq', T']
        x = self.conv_frontend(x)
        # [B, d_cnn, freq', T'] -> [B, T', d_cnn * freq']
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        return x

    def _run_rnn(self, x: Tensor) -> Tensor:
        """Apply BiGRU + LayerNorm + Dropout.

        Parameters
        ----------
        x : Tensor
            ``[B, T', flat_dim]``

        Returns
        -------
        Tensor
            ``[B, T', rnn_hidden * 2]``
        """
        x, _ = self.rnn(x)
        x = self.layer_norm(x)
        x = self.out_dropout(x)
        return x

    def forward(self, mel: Tensor, mel_lengths: Tensor) -> EncoderOutput:
        """Compute CTC log-probabilities.

        Parameters
        ----------
        mel : Tensor
            ``[B, n_mels, T]`` float32 log-mel spectrogram.
        mel_lengths : Tensor
            ``[B]`` int64 — unpadded frame counts before subsampling.

        Returns
        -------
        EncoderOutput
            ``log_probs``      : ``[B, T', V]`` float32
            ``output_lengths`` : ``[B]`` int64, T' = ceil(T / subsample_factor)
            ``intermediate``   : ``None`` (CRDNN has no InterCTC tap)
        """
        if mel.dim() != 3:
            raise ValueError(f"mel must be [B, n_mels, T], got {tuple(mel.shape)}")
        if mel_lengths.dim() != 1:
            raise ValueError(f"mel_lengths must be [B], got {tuple(mel_lengths.shape)}")

        # Conv2D frontend: [B, 80, T] -> [B, T', flat_dim]
        x = self._run_conv_frontend(mel)

        # BiGRU: [B, T', flat_dim] -> [B, T', rnn_hidden*2]
        x = self._run_rnn(x)

        # CTC head with fp32 island (CONTRACTS.md §11)
        logits = self.head(x).float()  # [B, T', vocab_size] float32
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T', V] float32

        # Ceil-div: strided conv with stride=s produces ceil(T/s) frames (H5 fix).
        output_lengths = (
            (mel_lengths + self.subsample_factor - 1) // self.subsample_factor
        ).to(torch.long)

        return EncoderOutput(
            log_probs=log_probs,
            output_lengths=output_lengths,
            intermediate=None,
        )
