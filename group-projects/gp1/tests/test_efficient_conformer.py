"""Tests for EfficientConformer acoustic encoder.

TDD: all tests written BEFORE implementation (RED → GREEN → REFACTOR).

Architecture context
--------------------
Burchi & Vielzeuf 2021 (https://arxiv.org/abs/2109.01163) introduces
progressive downsampling across conformer stages so later stages operate
on shorter sequences, reducing MHSA quadratic cost.

This encoder uses subsample_factor=4 (SubsampleConv /2 in prologue +
stride-2 conv between stage 1 and stage 2, stage 3 at /1).

CTC alignment note: s=4 gives T'/U_max ≈ 1.47 for worst-case
T=350 frames and U_max=59 chars — tight but acceptable for this
closed-vocabulary short-number task. Documented in
gp1_subsample_and_kernel_constraints.md and the model docstring.

STFT config: hop_length=160, sr=16kHz → 1 frame = 10 ms.
U_max = 59 chars ("девятьсот девяносто девять тысяч девятьсот девяносто девять").
T_min ~ 350 frames at 3.5 s fast speech.

References:
- Burchi & Vielzeuf (2021): https://arxiv.org/abs/2109.01163
- Author reference impl: https://github.com/burchim/EfficientConformer
- NeMo ConformerEncoder: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py
- CONTRACTS.md §5: ASREncoder Protocol, EncoderOutput
"""

from __future__ import annotations

import pytest
import torch

VOCAB_SIZE = 35  # blank + space + 33 Russian chars
U_MAX = 59  # worst-case char sequence length
T_WORST = 350  # worst-case frame count (3.5 s @ hop=160, sr=16kHz)


@pytest.fixture(scope="module")
def model() -> "EfficientConformer":
    from gp1.models.efficient_conformer import EfficientConformer

    torch.manual_seed(0)
    return EfficientConformer(vocab_size=VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Test 1: parameter budget
# ---------------------------------------------------------------------------


def test_efficient_conformer_has_params_under_5m(model):
    """Total trainable params must be strictly under 5,000,000."""
    n = sum(p.numel() for p in model.parameters())
    assert n < 5_000_000, (
        f"Param count {n:,} >= 5M budget. Reduce d_model_stages or n_blocks_per_stage."
    )


# ---------------------------------------------------------------------------
# Test 2: forward shape with char vocab
# ---------------------------------------------------------------------------


def test_efficient_conformer_forward_shape_char_vocab(model):
    """mel [2, 80, 1024] → log_probs [2, 256, 35], output_lengths [256, 128].

    subsample_factor=4: T'=T//4. Lengths [1024, 512] → [256, 128].
    """
    torch.manual_seed(0)
    mel = torch.randn(2, 80, 1024)
    mel_lengths = torch.tensor([1024, 512], dtype=torch.long)

    out = model(mel, mel_lengths)

    expected_t_prime = 1024 // 4  # 256
    assert out.log_probs.shape == (2, expected_t_prime, VOCAB_SIZE), (
        f"Expected [2, {expected_t_prime}, {VOCAB_SIZE}], "
        f"got {tuple(out.log_probs.shape)}"
    )
    expected_lengths = torch.tensor([1024 // 4, 512 // 4], dtype=torch.long)
    assert torch.equal(out.output_lengths, expected_lengths), (
        f"Expected output_lengths {expected_lengths.tolist()}, "
        f"got {out.output_lengths.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 3: log-probs are properly normalised
# ---------------------------------------------------------------------------


def test_efficient_conformer_log_probs_are_normalized(model):
    """exp(log_probs).sum(-1) must be ~1 for every (batch, frame) position."""
    torch.manual_seed(1)
    mel = torch.randn(2, 80, 400)
    mel_lengths = torch.tensor([400, 300], dtype=torch.long)

    out = model(mel, mel_lengths)
    sums = out.log_probs.exp().sum(dim=-1)  # [B, T']
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"log-softmax sums deviate from 1: max_err={(sums - 1).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4: CTC alignment worst-case (documented as tight)
# ---------------------------------------------------------------------------


def test_efficient_conformer_ctc_alignment_worst_case(model):
    """T'/U_max >= 1.4*U_max / U_max = 1.4 on worst-case 3.5 s sample.

    With s=4: T' = 350//4 = 87, ratio = 87/59 ≈ 1.47.
    This is tight (not the usual 2x), documented as acceptable for this
    closed-vocabulary short-utterance task. See gp1_subsample_and_kernel_constraints.md.
    """
    assert model.subsample_factor == 4, "This test assumes subsample_factor=4"
    t_prime = T_WORST // model.subsample_factor  # 87
    ratio = t_prime / U_MAX  # ~1.47
    assert ratio >= 1.4, (
        f"T'/U_max = {t_prime}/{U_MAX} = {ratio:.3f} < 1.4 — "
        f"CTC alignment too tight even for this closed-vocab task."
    )


# ---------------------------------------------------------------------------
# Test 5: no NaN on forward
# ---------------------------------------------------------------------------


def test_efficient_conformer_no_nan_on_forward(model):
    """Forward pass must not produce NaN in log_probs."""
    torch.manual_seed(42)
    mel = torch.randn(3, 80, 500)
    mel_lengths = torch.tensor([500, 400, 300], dtype=torch.long)

    out = model(mel, mel_lengths)
    assert not torch.isnan(out.log_probs).any(), "NaN detected in log_probs"


# ---------------------------------------------------------------------------
# Test 6: ASREncoder Protocol compliance
# ---------------------------------------------------------------------------


def test_efficient_conformer_implements_asrencoder_protocol(model):
    """EfficientConformer must satisfy ASREncoder Protocol attributes."""
    assert hasattr(model, "vocab_size"), "Missing vocab_size attribute"
    assert hasattr(model, "subsample_factor"), "Missing subsample_factor attribute"
    assert isinstance(model.vocab_size, int), (
        f"vocab_size must be int, got {type(model.vocab_size)}"
    )
    assert isinstance(model.subsample_factor, int), (
        f"subsample_factor must be int, got {type(model.subsample_factor)}"
    )
    # forward() must accept (mel, mel_lengths) and return EncoderOutput
    from gp1.models.base import EncoderOutput

    torch.manual_seed(0)
    mel = torch.randn(1, 80, 200)
    mel_lengths = torch.tensor([200], dtype=torch.long)
    out = model(mel, mel_lengths)
    assert isinstance(out, EncoderOutput), (
        f"forward() must return EncoderOutput, got {type(out)}"
    )


# ---------------------------------------------------------------------------
# Test 7: subsample_factor is 4
# ---------------------------------------------------------------------------


def test_efficient_conformer_subsample_factor_is_4(model):
    """subsample_factor attribute must be exactly 4 (progressive /2+/2)."""
    assert model.subsample_factor == 4, (
        f"Expected subsample_factor=4, got {model.subsample_factor}"
    )


# ---------------------------------------------------------------------------
# Bonus: output_lengths dtype is int64 (CTCLoss requirement)
# ---------------------------------------------------------------------------


def test_efficient_conformer_output_lengths_are_int64(model):
    """output_lengths must be int64 for torch.nn.CTCLoss."""
    torch.manual_seed(0)
    mel = torch.randn(2, 80, 400)
    mel_lengths = torch.tensor([400, 300], dtype=torch.long)

    out = model(mel, mel_lengths)
    assert out.output_lengths.dtype == torch.long, (
        f"output_lengths.dtype={out.output_lengths.dtype}, expected torch.long"
    )


# ---------------------------------------------------------------------------
# Bonus: log_probs are float32 (fp32 island requirement)
# ---------------------------------------------------------------------------


def test_efficient_conformer_log_probs_are_float32(model):
    """log_probs must be float32 for stable CTC loss (fp32 island)."""
    torch.manual_seed(0)
    mel = torch.randn(2, 80, 400)
    mel_lengths = torch.tensor([400, 300], dtype=torch.long)

    out = model(mel, mel_lengths)
    assert out.log_probs.dtype == torch.float32, (
        f"log_probs.dtype={out.log_probs.dtype}, expected torch.float32"
    )
