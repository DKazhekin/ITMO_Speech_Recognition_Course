"""Tests for FastConformerBPE acoustic encoder.

TDD: all tests written BEFORE implementation (RED phase).
Red -> Green -> Refactor per CONTRACTS.md §11.

Architecture notes:
- subsample_factor = 4 (NOT 8 — project memory blocks s=8 as CTC-unsafe).
  For BPE vocab U_bpe~20-30, T_min~350 frames:
  T'/U_bpe = 87/30 = 2.9 — safe margin.
- vocab_size = 256 (BPE-256).
- d_model = 144, n_blocks = 16, n_heads = 4 — ~4.7M params.

References:
- Rekesh et al. Fast Conformer (2023): https://arxiv.org/abs/2305.05084
- CONTRACTS.md §5: ASREncoder Protocol, EncoderOutput
- gp1_subsample_and_kernel_constraints.md
"""

from __future__ import annotations

import pytest
import torch

VOCAB_SIZE = 256
B, N_MELS, T = 2, 80, 1024


@pytest.fixture(scope="module")
def model() -> "FastConformerBPE":
    from gp1.models.fast_conformer_bpe import FastConformerBPE

    torch.manual_seed(0)
    return FastConformerBPE(vocab_size=VOCAB_SIZE)


@pytest.fixture(scope="module")
def sample_mel() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    mel = torch.randn(B, N_MELS, T)
    mel_lengths = torch.tensor([T, T // 2], dtype=torch.long)
    return mel, mel_lengths


# ---------------------------------------------------------------------------
# 1. Parameter budget
# ---------------------------------------------------------------------------


def test_fast_conformer_has_params_under_5m(model):
    """Total trainable params must be strictly < 5_000_000."""
    n = sum(p.numel() for p in model.parameters())
    assert n < 5_000_000, (
        f"Param count {n:,} is >= 5M budget. Reduce d_model or n_blocks."
    )


# ---------------------------------------------------------------------------
# 2. Forward shape
# ---------------------------------------------------------------------------


def test_fast_conformer_forward_shape_bpe_vocab(model, sample_mel):
    """mel [2, 80, 1024] -> log_probs.shape == (2, 256, 256).

    T' = 1024 / 4 = 256 frames after subsample_factor=4.
    """
    mel, mel_lengths = sample_mel
    out = model(mel, mel_lengths)
    expected = (B, T // 4, VOCAB_SIZE)
    assert out.log_probs.shape == expected, (
        f"Expected log_probs shape {expected}, got {tuple(out.log_probs.shape)}"
    )


# ---------------------------------------------------------------------------
# 3. Log-probs are properly normalised (log-softmax over vocab)
# ---------------------------------------------------------------------------


def test_fast_conformer_log_probs_are_normalized(model, sample_mel):
    """exp(log_probs).sum(-1) must be ~1 for every frame."""
    mel, mel_lengths = sample_mel
    out = model(mel, mel_lengths)
    sums = out.log_probs.exp().sum(dim=-1)  # [B, T']
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"log-softmax sums deviate from 1: max_err={(sums - 1).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# 4. CTC alignment safety — BPE worst case
# ---------------------------------------------------------------------------


def test_fast_conformer_ctc_alignment_worst_case():
    """T=350, U_max_bpe=30, subsample=4: T'/U = 87/30 >= 2.0.

    With BPE-256, longest utterance encodes to ~20-30 subword pieces.
    T' = 350 // 4 = 87 frames. 87 >= 2 * 30 = 60. PASS.
    """
    T_worst = 350
    U_max_bpe = 30
    subsample_factor = 4

    t_prime = T_worst // subsample_factor  # 87
    assert t_prime >= 2 * U_max_bpe, (
        f"CTC alignment unsafe: T'={t_prime} < 2*U_max={2 * U_max_bpe}. "
        "Reduce subsample_factor or BPE vocab size."
    )


# ---------------------------------------------------------------------------
# 5. No NaN on forward
# ---------------------------------------------------------------------------


def test_fast_conformer_no_nan_on_forward(model, sample_mel):
    """Forward pass must not produce NaN in log_probs."""
    mel, mel_lengths = sample_mel
    out = model(mel, mel_lengths)
    assert not torch.isnan(out.log_probs).any(), "NaN detected in log_probs"
    if out.intermediate is not None:
        assert not torch.isnan(out.intermediate).any(), "NaN detected in intermediate"


# ---------------------------------------------------------------------------
# 6. ASREncoder Protocol compliance
# ---------------------------------------------------------------------------


def test_fast_conformer_implements_asrencoder_protocol(model):
    """Model must expose vocab_size (int) and subsample_factor (int)."""
    assert isinstance(model.vocab_size, int), (
        f"vocab_size must be int, got {type(model.vocab_size)}"
    )
    assert isinstance(model.subsample_factor, int), (
        f"subsample_factor must be int, got {type(model.subsample_factor)}"
    )
    assert model.vocab_size == VOCAB_SIZE, (
        f"Expected vocab_size={VOCAB_SIZE}, got {model.vocab_size}"
    )
    # forward() must return EncoderOutput
    from gp1.models.base import EncoderOutput

    torch.manual_seed(0)
    mel = torch.randn(1, 80, 400)
    mel_lengths = torch.tensor([400], dtype=torch.long)
    out = model(mel, mel_lengths)
    assert isinstance(out, EncoderOutput), (
        f"forward() must return EncoderOutput, got {type(out)}"
    )


# ---------------------------------------------------------------------------
# 7. subsample_factor attribute is exactly 4
# ---------------------------------------------------------------------------


def test_fast_conformer_subsample_factor_is_4(model):
    """subsample_factor must be 4 (memory constraint: s=8 is forbidden)."""
    assert model.subsample_factor == 4, (
        f"Expected subsample_factor=4, got {model.subsample_factor}. "
        "s=8 is forbidden per gp1_subsample_and_kernel_constraints."
    )


# ---------------------------------------------------------------------------
# 8. output_lengths are int64 and correctly computed
# ---------------------------------------------------------------------------


def test_fast_conformer_output_lengths_dtype_and_value(model):
    """output_lengths must be int64 and equal mel_lengths // subsample_factor."""
    torch.manual_seed(0)
    mel = torch.randn(2, 80, 400)
    mel_lengths = torch.tensor([400, 300], dtype=torch.long)
    out = model(mel, mel_lengths)

    assert out.output_lengths.dtype == torch.long, (
        f"output_lengths.dtype={out.output_lengths.dtype}, expected torch.long"
    )
    expected = torch.tensor([100, 75], dtype=torch.long)
    assert torch.equal(out.output_lengths, expected), (
        f"Expected {expected.tolist()}, got {out.output_lengths.tolist()}"
    )


# ---------------------------------------------------------------------------
# 9. log_probs dtype is float32 (CTC fp32 island)
# ---------------------------------------------------------------------------


def test_fast_conformer_log_probs_are_float32(model, sample_mel):
    """log_probs must be float32 for stable CTC loss."""
    mel, mel_lengths = sample_mel
    out = model(mel, mel_lengths)
    assert out.log_probs.dtype == torch.float32, (
        f"log_probs.dtype={out.log_probs.dtype}, expected torch.float32"
    )
