"""Shared NN building blocks for GP1 acoustic encoders.

This module defines:

- ``TCSConvBlock``  - QuartzNet-style time-channel-separable 1D conv.
- ``ConformerBlock`` - macaron FFN -> MHSA -> ConvModule -> FFN.
- ``SubsampleConv`` - striding 2D conv frontend that reduces time by
  ``factor`` and projects channels to ``d_out`` (channel-last output).

References:
- QuartzNet (TCS conv, kernel 33 depthwise + 1x1 pointwise):
  Kriman et al., https://arxiv.org/abs/1910.10261
- Conformer (macaron FFN + conv module with GLU + depthwise conv):
  Gulati et al., https://arxiv.org/abs/2005.08100
- torchaudio.models.Conformer reference implementation:
  https://pytorch.org/audio/stable/generated/torchaudio.models.Conformer.html
- NeMo QuartzNet building blocks:
  https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conv_asr.py
"""
from __future__ import annotations

from torch import Tensor, nn


def _same_pad(kernel_size: int, dilation: int = 1) -> int:
    """Return the 'same' padding for a 1D conv with odd kernel size."""
    return (kernel_size - 1) // 2 * dilation


class TCSConvBlock(nn.Module):
    """Time-Channel-Separable 1D conv block used in QuartzNet.

    Layout: depthwise Conv1d (groups=c_in) -> pointwise Conv1d (1x1) ->
    BatchNorm1d -> ReLU -> Dropout. A parameter-free residual is added
    from the input when ``residual=True`` AND ``stride == 1`` AND
    ``c_in == c_out``.

    Input:  ``[B, c_in, T]`` float32
    Output: ``[B, c_out, T_out]``   where ``T_out = ceil(T / stride)``.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"TCSConvBlock requires odd kernel_size for 'same' padding, "
                f"got {kernel_size}"
            )
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.depthwise = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=_same_pad(kernel_size, dilation),
            groups=c_in,
            dilation=dilation,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = residual and stride == 1 and c_in == c_out

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"TCSConvBlock expects [B, C, T], got shape {tuple(x.shape)}"
            )
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        if self.use_residual:
            out = out + x
        out = self.act(out)
        out = self.dropout(out)
        return out


class _FeedForwardModule(nn.Module):
    """Macaron feed-forward submodule used inside ``ConformerBlock``."""

    def __init__(self, d_model: int, ff_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * ff_ratio)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * ff_ratio, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.linear1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.linear2(y)
        y = self.drop2(y)
        return y


class _ConvModule(nn.Module):
    """Conformer convolution module: pointwise + GLU + depthwise + BN."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"Conformer conv_kernel must be odd, got {kernel_size}"
            )
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=_same_pad(kernel_size),
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D] -> LayerNorm -> transpose to [B, D, T] for convs
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.pointwise_in(y)
        y = self.glu(y)
        y = self.depthwise(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.pointwise_out(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)  # back to [B, T, D]
        return y


class ConformerBlock(nn.Module):
    """Conformer encoder block (macaron FFN / MHSA / ConvModule / FFN).

    Input:  ``[B, T, d_model]`` float32
    Output: ``[B, T, d_model]`` float32

    Uses absolute positional information implicit in the input
    embedding and ``nn.MultiheadAttention`` for the self-attention
    block; the SOTA Conformer paper additionally uses relative
    positional encoding, which is left as a Phase-4 tuning upgrade.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ff_ratio: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} not divisible by n_heads={n_heads}"
            )
        self.ffn1 = _FeedForwardModule(d_model, ff_ratio, dropout)
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mhsa_dropout = nn.Dropout(dropout)
        self.conv_module = _ConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = _FeedForwardModule(d_model, ff_ratio, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"ConformerBlock expects [B, T, D], got {tuple(x.shape)}"
            )
        # Macaron FFN (half residual)
        x = x + 0.5 * self.ffn1(x)
        # Multi-head self-attention with pre-norm
        y = self.mhsa_norm(x)
        attn_out, _ = self.mhsa(
            y, y, y,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.mhsa_dropout(attn_out)
        # Convolution module
        x = x + self.conv_module(x)
        # Macaron FFN (half residual)
        x = x + 0.5 * self.ffn2(x)
        # Final LayerNorm
        x = self.final_norm(x)
        return x


class SubsampleConv(nn.Module):
    """Striding 2D conv frontend that reduces time by ``factor`` and
    projects to ``d_out``.

    ``factor`` must be a power of 2 (1, 2, 4, 8). Implemented as a stack
    of ``log2(factor)`` stride-2 Conv2d layers followed by a linear
    projection from flattened ``(C_conv * mels_after)`` to ``d_out``.

    Input:  ``[B, n_mels, T]`` float32 log-mel
    Output: ``[B, T_out, d_out]`` float32, where ``T_out = T / factor``.

    Reference: torchaudio ``_SubsamplingConv`` and ESPnet2 Conformer
    frontend:
    https://github.com/espnet/espnet/blob/master/espnet2/asr/encoder/conformer_encoder.py
    """

    def __init__(self, n_mels: int, d_out: int, factor: int = 4) -> None:
        super().__init__()
        if factor < 1 or (factor & (factor - 1)) != 0:
            raise ValueError(
                f"SubsampleConv factor must be power of two, got {factor}"
            )
        self.n_mels = n_mels
        self.d_out = d_out
        self.factor = factor

        num_stages = 0
        while (1 << num_stages) < factor:
            num_stages += 1

        convs: list[nn.Module] = []
        in_c = 1
        mid_c = d_out
        cur_mels = n_mels
        for _ in range(num_stages):
            convs.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=mid_c,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            convs.append(nn.ReLU(inplace=False))
            in_c = mid_c
            # output size with k=3, p=1, s=2 is floor((L + 1) / 2)
            cur_mels = (cur_mels + 1) // 2
        self.convs = nn.Sequential(*convs)
        # When factor=1, there are no convs and the frontend is a pure
        # linear projection from the mel axis into d_out.
        flat_dim = mid_c * cur_mels if num_stages > 0 else n_mels
        self.proj = nn.Linear(flat_dim, d_out)
        self._num_stages = num_stages

    def forward(self, mel: Tensor) -> Tensor:
        if mel.dim() != 3:
            raise ValueError(
                f"SubsampleConv expects [B, n_mels, T], got {tuple(mel.shape)}"
            )
        if self._num_stages == 0:
            # factor=1: just transpose and linearly project
            y = mel.transpose(1, 2)  # [B, T, n_mels]
            return self.proj(y)
        # [B, n_mels, T] -> [B, 1, n_mels, T]
        y = mel.unsqueeze(1)
        y = self.convs(y)
        # [B, C_conv, mels', T']
        b, c, f, t = y.shape
        # rearrange to [B, T', C_conv * mels']
        y = y.permute(0, 3, 1, 2).reshape(b, t, c * f)
        y = self.proj(y)  # [B, T', d_out]
        return y
