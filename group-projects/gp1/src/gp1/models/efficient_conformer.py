"""EfficientConformer acoustic encoder for GP1 Russian spoken numbers ASR.

Research summary (Phase 0)
--------------------------
1. Burchi & Vielzeuf 2021 (https://arxiv.org/abs/2109.01163) groups
   ConformerBlocks into three stages and inserts a stride-2 Conv1d
   downsampler between stages 1→2, so later stages run on T/4 frames.
   This cuts MHSA cost O(T²) → O((T/4)²) for the majority of blocks.

2. The author reference impl (https://github.com/burchim/EfficientConformer)
   uses grouped multi-head attention (GMHSA) and relative positional
   encoding in later stages. Both are OPTIONAL upgrades; this
   implementation uses standard MHSA from ``common.ConformerBlock``
   to stay within the shared block vocabulary and the 5M param budget.
   GMHSA would reduce MHSA params further but add complexity without
   measurable quality gain on a closed-vocabulary 6-digit task.

3. NeMo ConformerEncoder
   (https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py)
   confirms the SubsampleConv-2D→ConformerBlocks pattern; we use a
   lightweight TCSConvBlock prologue instead to avoid the expensive
   flat_dim = d_out × ceil(n_mels/2) linear layer that blows the
   parameter budget (≈160K overhead at d_out=64).

Architecture
------------
    Input  [B, 80, T]              log-mel spectrogram

    Prologue:   TCSConvBlock(80 → 64, k=11, stride=2)  → [B, 64, T/2]
    Proj:       Conv1d(64 → 96, k=1)                   → [B, 96, T/2]
    Transpose:  channel-last                            → [B, T/2, 96]
    Stage 1:    4 × ConformerBlock(d=96, heads=4, k=15) → [B, T/2, 96]
    Downsample: DownsampleBlock(96 → 128, stride=2)     → [B, T/4, 128]
    Stage 2:    4 × ConformerBlock(d=128, heads=4, k=15)→ [B, T/4, 128]
    Stage 3:    4 × ConformerBlock(d=128, heads=4, k=15)→ [B, T/4, 128]
    Head:       Linear(128 → vocab_size) → fp32 → log_softmax → [B, T/4, V]

    Total subsample_factor = 4  (prologue /2 × inter-stage downsample /2).

CTC alignment warning
---------------------
subsample_factor=4 gives T'/U_max ≈ 87/59 ≈ 1.47 for the worst-case
sample (350 frames, U_max=59 chars for "999999" in Russian). This is
below the standard 2× safety margin. It is ACCEPTABLE for this
closed-vocabulary short-utterance task because:
  - The vocabulary is only 6-digit Russian number words (≤59 chars max).
  - Faster phoneme emission is observed on shorter, well-articulated
    utterances in the competition corpus.
  - s=4 is TIGHT; s=8 is FORBIDDEN (T'/U_max < 1 → CTC diverges).
See: gp1_subsample_and_kernel_constraints.md

Param count target: ~4.2M (well under 5M budget).

Grouped attention / relative PE
---------------------------------
Omitted (see point 2 above). If a Phase-4 upgrade is needed:
  - Replace ``ConformerBlock.mhsa`` with a grouped-query variant.
  - Add sinusoidal or learnable relative positional bias in stage 2/3.
  These would reduce stage-2/3 MHSA params further and improve long-form
  performance but are not necessary at T/4 ≤ 256 frames.

References:
- Burchi & Vielzeuf (2021): https://arxiv.org/abs/2109.01163
- Author reference impl: https://github.com/burchim/EfficientConformer
- NeMo ConformerEncoder: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py
- CONTRACTS.md §5: ASREncoder Protocol, EncoderOutput
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gp1.models.base import EncoderOutput
from gp1.models.common import TCSConvBlock
from gp1.models.efficient_conformer_blocks import ConformerStage, DownsampleBlock

log = logging.getLogger(__name__)

# The only allowed total subsample factor for this encoder (prologue /2
# × inter-stage downsampler /2). s=8 is FORBIDDEN — see module docstring.
_SUBSAMPLE_FACTOR = 4


class EfficientConformer(nn.Module):
    """EfficientConformer CTC acoustic encoder with progressive downsampling.

    Conforms to ``ASREncoder`` Protocol (CONTRACTS.md §5).

    Parameters
    ----------
    vocab_size : int
        Output vocabulary size (CTC blank at index 0).
        Default: 35 (blank + space + 33 Russian chars).
    d_model_stages : tuple[int, int, int]
        Hidden dimension per stage. Default: (96, 128, 128).
        Both stage-2 and stage-3 dimensions must be equal because they
        share the same ``DownsampleBlock`` output projection.
    n_blocks_per_stage : tuple[int, int, int]
        Number of ConformerBlocks per stage. Default: (4, 4, 4).
    n_heads : int
        Attention heads (must divide all d_model_stages). Default: 4.
    ff_ratio : int
        Feed-forward expansion ratio. Default: 4.
    conv_kernel : int
        Depthwise conv kernel size inside ConformerBlocks (odd). Default: 15.
    dropout : float
        Dropout probability. Default: 0.1.
    """

    vocab_size: int
    subsample_factor: int

    def __init__(
        self,
        vocab_size: int = 35,
        d_model_stages: tuple[int, int, int] = (96, 128, 128),
        n_blocks_per_stage: tuple[int, int, int] = (4, 4, 4),
        n_heads: int = 4,
        ff_ratio: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        d_s1, d_s2, d_s3 = d_model_stages
        n_b1, n_b2, n_b3 = n_blocks_per_stage

        for stage_idx, (d, n) in enumerate(
            zip(d_model_stages, n_blocks_per_stage), start=1
        ):
            if d % n_heads != 0:
                raise ValueError(
                    f"Stage {stage_idx}: d_model={d} not divisible by n_heads={n_heads}"
                )
            if n < 1:
                raise ValueError(
                    f"Stage {stage_idx}: n_blocks_per_stage must be >= 1, got {n}"
                )

        self.vocab_size = vocab_size
        self.subsample_factor = _SUBSAMPLE_FACTOR

        # ------------------------------------------------------------------
        # Prologue: TCSConvBlock with stride=2 halves the time dimension.
        # c_in=80 (n_mels), c_out=64, k=11 — sub-phoneme level receptive
        # field (~110 ms), lightweight (depthwise + pointwise).
        # residual=False because c_in != c_out and stride != 1.
        # ------------------------------------------------------------------
        _PROLOGUE_CHANNELS = 64
        self.prologue = TCSConvBlock(
            c_in=80,
            c_out=_PROLOGUE_CHANNELS,
            kernel_size=11,
            stride=2,
            dropout=dropout,
            residual=False,
        )

        # Channel projection: prologue output (64) → stage-1 d_model (d_s1).
        # Conv1d(k=1) is pointwise; LayerNorm is applied after transposing to
        # channel-last. No activation — stage 1 will apply FFN/MHSA norms.
        self.input_conv = nn.Conv1d(_PROLOGUE_CHANNELS, d_s1, kernel_size=1, bias=False)
        self.input_norm = nn.LayerNorm(d_s1)

        # ------------------------------------------------------------------
        # Stage 1: n_b1 ConformerBlocks at d_s1 (no downsampling here).
        # ------------------------------------------------------------------
        self.stage1 = ConformerStage(
            n_blocks=n_b1,
            d_model=d_s1,
            n_heads=n_heads,
            ff_ratio=ff_ratio,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Inter-stage downsampler: /2 via stride-2 Conv1d + channel change.
        # This is the defining operation of the EfficientConformer arch.
        # ------------------------------------------------------------------
        self.downsample = DownsampleBlock(d_in=d_s1, d_out=d_s2)

        # ------------------------------------------------------------------
        # Stage 2: n_b2 ConformerBlocks at d_s2.
        # ------------------------------------------------------------------
        self.stage2 = ConformerStage(
            n_blocks=n_b2,
            d_model=d_s2,
            n_heads=n_heads,
            ff_ratio=ff_ratio,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Stage 3: n_b3 ConformerBlocks at d_s3 (no further downsampling).
        # d_s3 must equal d_s2 (same DownsampleBlock output feeds both).
        # ------------------------------------------------------------------
        if d_s3 != d_s2:
            raise ValueError(
                f"d_model_stages[2]={d_s3} must equal d_model_stages[1]={d_s2} "
                f"because both stages follow the same DownsampleBlock."
            )
        self.stage3 = ConformerStage(
            n_blocks=n_b3,
            d_model=d_s3,
            n_heads=n_heads,
            ff_ratio=ff_ratio,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # CTC head: linear projection to vocab, fp32 log-softmax.
        # The fp32 cast and log_softmax are applied inside forward() so
        # they remain outside any fp16 autocast context (CONTRACTS §11).
        # ------------------------------------------------------------------
        self.head = nn.Linear(d_s3, vocab_size, bias=True)

        n_params = sum(p.numel() for p in self.parameters())
        log.info(
            "EfficientConformer initialised: %d params (target ~4.2M, budget <5M). "
            "Stages: d=%s, blocks=%s, heads=%d, ff_ratio=%d, conv_k=%d.",
            n_params,
            d_model_stages,
            n_blocks_per_stage,
            n_heads,
            ff_ratio,
            conv_kernel,
        )
        if n_params >= 5_000_000:
            raise RuntimeError(
                f"EfficientConformer param count {n_params:,} >= 5M budget. "
                f"Reduce d_model_stages or n_blocks_per_stage."
            )

    def forward(self, mel: Tensor, mel_lengths: Tensor) -> EncoderOutput:
        """Compute CTC log-probabilities.

        Parameters
        ----------
        mel : Tensor
            ``[B, 80, T]`` float32 log-mel spectrogram.
        mel_lengths : Tensor
            ``[B]`` int64 — unpadded frame counts before any subsampling.

        Returns
        -------
        EncoderOutput
            ``log_probs``      : ``[B, T/4, V]`` float32
            ``output_lengths`` : ``[B]`` int64
            ``intermediate``   : ``None`` (no InterCTC tap in this encoder)
        """
        if mel.dim() != 3:
            raise ValueError(f"mel must be [B, n_mels, T], got {tuple(mel.shape)}")
        if mel_lengths.dim() != 1:
            raise ValueError(f"mel_lengths must be [B], got {tuple(mel_lengths.shape)}")

        # ---- Prologue: [B, 80, T] → [B, 64, T/2] -------------------------
        x = self.prologue(mel)  # [B, 64, T/2]

        # ---- Input projection: [B, 64, T/2] → [B, T/2, d_s1] ------------
        x = self.input_conv(x)  # [B, d_s1, T/2]
        x = x.transpose(1, 2)  # [B, T/2, d_s1]  channel-last
        x = self.input_norm(x)  # LayerNorm along last dim

        # ---- Stage 1: [B, T/2, d_s1] (no length change) ------------------
        x = self.stage1(x)  # [B, T/2, d_s1]

        # ---- Inter-stage downsampler: [B, T/2, d_s1] → [B, T/4, d_s2] --
        x = self.downsample(x)  # [B, T/4, d_s2]

        # ---- Stage 2 + 3: [B, T/4, d_s{2,3}] ----------------------------
        x = self.stage2(x)  # [B, T/4, d_s2]
        x = self.stage3(x)  # [B, T/4, d_s3]

        # ---- CTC head (fp32 island) ---------------------------------------
        # Cast to float32 before log_softmax to satisfy fp32-island
        # requirement (CONTRACTS.md §11), even under fp16 autocast.
        logits = self.head(x).float()  # [B, T/4, V] fp32
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T/4, V] fp32

        # ---- Output lengths: progressive ceil-div for two stride-2 stages --
        # Prologue uses stride=2:  T_1 = ceil(T   / 2) = (T   + 1) // 2
        # DownsampleBlock stride=2: T'  = ceil(T_1 / 2) = (T_1 + 1) // 2
        # This matches actual Conv1d output shape for odd T (H5 fix).
        lengths_after_prologue = (mel_lengths + 1) // 2
        output_lengths = ((lengths_after_prologue + 1) // 2).to(torch.long)

        return EncoderOutput(
            log_probs=log_probs,
            output_lengths=output_lengths,
            intermediate=None,
        )
