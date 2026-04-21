"""Tests for QuartzNet-10x4 acoustic encoder.

TDD: all tests written BEFORE implementation.
Red -> Green -> Refactor cycle per CONTRACTS.md §11.

STFT config: hop_length=160, sr=16kHz -> 1 frame = 10 ms.
U_max = 59 chars (longest Russian 6-digit number).
Worst-case T_min ~ 350 frames @ 3.5 s.
After /2 subsample: T' = 175 -> T'/U_max = 2.97 (safe).

References:
- CONTRACTS.md §5 (ASREncoder, EncoderOutput)
- gp1 memory: gp1_subsample_and_kernel_constraints.md
"""

from __future__ import annotations

import pytest
import torch

VOCAB_SIZE = 35  # blank + space + 33 Russian chars


@pytest.fixture(scope="module")
def model() -> "QuartzNet10x4":
    from gp1.models.quartznet import QuartzNet10x4

    torch.manual_seed(0)
    return QuartzNet10x4(vocab_size=VOCAB_SIZE)


@pytest.fixture(scope="module")
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Returns mel [2, 80, 400] and mel_lengths [400, 300]."""
    torch.manual_seed(0)
    mel = torch.randn(2, 80, 400)
    mel_lengths = torch.tensor([400, 300], dtype=torch.long)
    return mel, mel_lengths


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_forward_produces_expected_shape(model, sample_batch):
    """log_probs shape must be [B, T/2, vocab_size]."""
    mel, mel_lengths = sample_batch
    out = model(mel, mel_lengths)
    assert out.log_probs.shape == (2, 200, VOCAB_SIZE), (
        f"Expected [2, 200, {VOCAB_SIZE}], got {tuple(out.log_probs.shape)}"
    )


def test_forward_log_probs_sum_to_one_along_vocab(model, sample_batch):
    """exp(log_probs).sum(-1) must be ~1 for every frame (log-softmax)."""
    mel, mel_lengths = sample_batch
    out = model(mel, mel_lengths)
    sums = out.log_probs.exp().sum(dim=-1)  # [B, T']
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"log-softmax sums deviate from 1: max_err={(sums - 1).abs().max():.2e}"
    )


def test_output_lengths_equal_input_over_subsample(model):
    """output_lengths = ceil(mel_lengths / subsample_factor)."""
    from gp1.models.quartznet import QuartzNet10x4

    torch.manual_seed(0)
    m = QuartzNet10x4(vocab_size=VOCAB_SIZE)
    mel_lengths = torch.tensor([400, 300], dtype=torch.long)
    mel = torch.randn(2, 80, 400)
    out = m(mel, mel_lengths)
    expected = torch.tensor([200, 150], dtype=torch.long)
    assert torch.equal(out.output_lengths, expected), (
        f"Expected {expected.tolist()}, got {out.output_lengths.tolist()}"
    )


def test_intermediate_not_none_and_has_expected_midchannel(model, sample_batch):
    """intermediate tensor must be [B, T_mid, D] for InterCTC at mid-depth."""
    mel, mel_lengths = sample_batch
    out = model(mel, mel_lengths)
    assert out.intermediate is not None, "intermediate must not be None"
    assert out.intermediate.dim() == 3, (
        f"intermediate must be 3-D [B, T_mid, D], got {out.intermediate.dim()}-D"
    )
    B, T_mid, D_mid = out.intermediate.shape
    assert B == 2, f"batch dim mismatch: {B}"
    # T_mid must equal T' (same subsampled length)
    assert T_mid == out.log_probs.shape[1], (
        f"intermediate T_mid={T_mid} != log_probs T'={out.log_probs.shape[1]}"
    )
    # D_mid should be 256 (d_model of B1/B2 blocks)
    assert D_mid == 256, f"Expected D_mid=256, got {D_mid}"


# ---------------------------------------------------------------------------
# Attribute / invariant tests
# ---------------------------------------------------------------------------


def test_subsample_factor_is_2(model):
    """subsample_factor attribute must be 2 (CONTRACTS + memory constraint)."""
    assert model.subsample_factor == 2


def test_rejects_subsample_factor_not_2():
    """Constructor with invalid subsample_factor must raise ValueError."""
    from gp1.models.quartznet import QuartzNet10x4

    with pytest.raises(ValueError, match="subsample_factor"):
        QuartzNet10x4(vocab_size=VOCAB_SIZE, subsample_factor=4)


# ---------------------------------------------------------------------------
# Parameter budget
# ---------------------------------------------------------------------------


def test_quartznet_param_count_le_5M(model):
    """Total trainable params must not exceed 5,000,000."""
    n = sum(p.numel() for p in model.parameters())
    assert n <= 5_000_000, f"Param count {n:,} exceeds 5M budget"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_end_to_end():
    """Backward pass must produce non-zero gradient on first Conv weight."""
    from gp1.models.quartznet import QuartzNet10x4

    torch.manual_seed(0)
    m = QuartzNet10x4(vocab_size=VOCAB_SIZE)
    mel = torch.randn(1, 80, 200)
    mel_lengths = torch.tensor([200], dtype=torch.long)

    out = m(mel, mel_lengths)
    loss = out.log_probs.sum()
    loss.backward()

    # Find the first Conv1d weight in the prologue block
    first_conv = None
    for module in m.modules():
        if isinstance(module, torch.nn.Conv1d):
            first_conv = module
            break

    assert first_conv is not None, "No Conv1d found in model"
    assert first_conv.weight.grad is not None, "Gradient is None on first Conv"
    assert first_conv.weight.grad.abs().max() > 0, "Gradient is all-zero"
