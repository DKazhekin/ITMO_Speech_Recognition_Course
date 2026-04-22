"""Tests for CRDNN (Conv2D + BiGRU) acoustic encoder.

TDD: all tests written BEFORE implementation.
Red -> Green -> Refactor cycle per CONTRACTS.md §11.

STFT config: hop_length=160, sr=16kHz -> 1 frame = 10 ms.
U_max = 59 chars (longest Russian 6-digit number).
Worst-case T_min ~ 350 frames @ 3.5 s.
subsample_factor=1: T' = T, ratio T'/U_max = 350/59 = 5.93 (safe).

References:
- CONTRACTS.md §5 (ASREncoder, EncoderOutput)
- gp1 memory: gp1_subsample_and_kernel_constraints.md
"""

from __future__ import annotations

import pytest
import torch

VOCAB_SIZE = 35  # blank + space + 33 Russian chars
U_MAX = 59  # worst-case transcription length


@pytest.fixture(scope="module")
def model() -> "CRDNN":
    from gp1.models.crdnn import CRDNN

    torch.manual_seed(0)
    return CRDNN(vocab_size=VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Test 1: parameter budget
# ---------------------------------------------------------------------------


def test_crdnn_has_params_under_5m(model):
    """Total trainable params must be strictly less than 5,000,000."""
    n = sum(p.numel() for p in model.parameters())
    assert n < 5_000_000, f"Param count {n:,} exceeds 5M budget"


# ---------------------------------------------------------------------------
# Test 2: forward shape with char vocab
# ---------------------------------------------------------------------------


def test_crdnn_forward_shape_char_vocab(model):
    """log_probs shape must be [B, T/s, vocab_size] and lengths correct.

    Uses mel [2, 80, 1024] and mel_lengths=[1024, 512].
    """
    torch.manual_seed(0)
    mel = torch.randn(2, 80, 1024)
    mel_lengths = torch.tensor([1024, 512], dtype=torch.long)
    out = model(mel, mel_lengths)

    s = model.subsample_factor
    expected_t = 1024 // s
    assert out.log_probs.shape == (2, expected_t, VOCAB_SIZE), (
        f"Expected [2, {expected_t}, {VOCAB_SIZE}], got {tuple(out.log_probs.shape)}"
    )
    expected_lengths = torch.tensor([1024 // s, 512 // s], dtype=torch.long)
    assert torch.equal(out.output_lengths, expected_lengths), (
        f"Expected output_lengths={expected_lengths.tolist()}, "
        f"got {out.output_lengths.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 3: log-softmax normalization
# ---------------------------------------------------------------------------


def test_crdnn_log_probs_are_normalized(model):
    """exp(log_probs).sum(-1) must be ~1.0 for every frame (log-softmax)."""
    torch.manual_seed(1)
    mel = torch.randn(2, 80, 400)
    mel_lengths = torch.tensor([400, 300], dtype=torch.long)
    out = model(mel, mel_lengths)
    sums = out.log_probs.exp().sum(dim=-1)  # [B, T']
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"log-softmax sums deviate from 1: max_err={(sums - 1).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4: CTC alignment worst-case
# ---------------------------------------------------------------------------


def test_crdnn_ctc_alignment_worst_case(model):
    """T' / U_max >= 2 on worst-case T=350 sample with subsample_factor."""
    t_prime = 350 // model.subsample_factor
    ratio = t_prime / U_MAX
    assert ratio >= 2.0, (
        f"T'/U_max = {t_prime}/{U_MAX} = {ratio:.2f} < 2.0 — "
        f"CTC training will be unstable with subsample_factor={model.subsample_factor}"
    )


# ---------------------------------------------------------------------------
# Test 5: no NaN / Inf on forward
# ---------------------------------------------------------------------------


def test_crdnn_no_nan_on_forward(model):
    """Forward pass must not produce NaN or Inf in log_probs."""
    torch.manual_seed(42)
    mel = torch.randn(3, 80, 500)
    mel_lengths = torch.tensor([500, 400, 300], dtype=torch.long)
    out = model(mel, mel_lengths)
    assert not torch.isnan(out.log_probs).any(), "NaN detected in log_probs"
    assert not torch.isinf(out.log_probs).any(), "Inf detected in log_probs"


# ---------------------------------------------------------------------------
# Test 6: ASREncoder Protocol compliance
# ---------------------------------------------------------------------------


def test_crdnn_implements_asrencoder_protocol(model):
    """CRDNN must satisfy ASREncoder Protocol attributes and return EncoderOutput."""
    from gp1.models.base import EncoderOutput

    assert hasattr(model, "vocab_size"), "Missing attribute: vocab_size"
    assert hasattr(model, "subsample_factor"), "Missing attribute: subsample_factor"
    assert hasattr(model, "forward"), "Missing attribute: forward"

    torch.manual_seed(0)
    mel = torch.randn(1, 80, 200)
    mel_lengths = torch.tensor([200], dtype=torch.long)
    out = model(mel, mel_lengths)
    assert isinstance(out, EncoderOutput), (
        f"forward() must return EncoderOutput, got {type(out)}"
    )
    assert out.log_probs.dtype == torch.float32, (
        f"log_probs.dtype must be float32, got {out.log_probs.dtype}"
    )
    assert out.output_lengths.dtype == torch.long, (
        f"output_lengths.dtype must be int64, got {out.output_lengths.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 7: subsample_factor consistency
# ---------------------------------------------------------------------------


def test_crdnn_respects_subsample_factor(model):
    """output_lengths must equal mel_lengths // subsample_factor."""
    torch.manual_seed(0)
    mel_lengths = torch.tensor([400, 300, 200], dtype=torch.long)
    mel = torch.randn(3, 80, 400)
    out = model(mel, mel_lengths)
    expected = mel_lengths // model.subsample_factor
    assert torch.equal(out.output_lengths, expected), (
        f"Expected {expected.tolist()}, got {out.output_lengths.tolist()}"
    )
