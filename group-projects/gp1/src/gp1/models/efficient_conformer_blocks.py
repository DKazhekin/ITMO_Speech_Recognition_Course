"""Helper blocks specific to the EfficientConformer encoder.

These are NOT in common.py because they are tightly coupled to the
EfficientConformer progressive-downsampling design and would pollute the
shared block vocabulary used by QuartzNet and other encoders.

References:
- Burchi & Vielzeuf (2021): https://arxiv.org/abs/2109.01163
- Author reference impl: https://github.com/burchim/EfficientConformer
"""

from __future__ import annotations

import logging

from torch import Tensor, nn

from gp1.models.common import ConformerBlock

log = logging.getLogger(__name__)


class DownsampleBlock(nn.Module):
    """Stride-2 Conv1d downsampler between EfficientConformer stages.

    Reduces the time dimension by 2 and optionally projects channels
    from ``d_in`` to ``d_out``. BatchNorm + ReLU are applied after the
    convolution; no residual (dimensions change).

    Input:  ``[B, T, d_in]``  (channel-last, conformer convention)
    Output: ``[B, T//2, d_out]``
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_in,
            out_channels=d_out,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(d_out)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            ``[B, T, d_in]`` channel-last.

        Returns
        -------
        Tensor
            ``[B, T//2, d_out]`` channel-last.
        """
        if x.dim() != 3:
            raise ValueError(f"DownsampleBlock expects [B, T, D], got {tuple(x.shape)}")
        # [B, T, D] -> [B, D, T] for Conv1d
        y = x.transpose(1, 2)
        y = self.conv(y)  # [B, d_out, T//2]
        y = self.bn(y)
        y = self.act(y)
        # back to channel-last: [B, T//2, d_out]
        return y.transpose(1, 2)


class ConformerStage(nn.Module):
    """A stack of ``n_blocks`` ConformerBlocks at a fixed ``d_model``.

    Wraps the blocks as an ``nn.ModuleList`` so each block receives the
    optional ``key_padding_mask`` argument that ``nn.Sequential`` cannot
    forward.

    Input:  ``[B, T, d_model]``
    Output: ``[B, T, d_model]``
    """

    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        n_heads: int,
        ff_ratio: int,
        conv_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
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

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:  # [B, T, d_model]
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x
