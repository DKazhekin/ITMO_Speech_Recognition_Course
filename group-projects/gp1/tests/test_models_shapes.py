"""Generic shape invariant tests for ASREncoder implementations.

Documents and enforces the CTC safety margin:
  T' / U_max >= 2 on the worst-case sample.

STFT config: hop_length=160, sr=16kHz -> 1 frame = 10 ms.
U_max = 59 chars: "девятьсот девяносто девять тысяч девятьсот девяносто девять"
T_min ~ 350 frames for a 3.5 s fast-spoken 6-digit utterance.

See: gp1_subsample_and_kernel_constraints.md
"""

from __future__ import annotations

import pytest
import torch

# Worst-case transcription of "999999" in Russian:
# "девятьсот девяносто девять тысяч девятьсот девяносто девять" = 59 chars
U_MAX = 59

# 3.5s fast speech at 16kHz / hop=160 -> 350 frames
T_WORST_CASE = 350

VOCAB_SIZE = 35


def test_T_prime_over_U_max_geq_2_on_worst_case():
    """After /2 subsample, T'/U_max >= 2 on worst-case 3.5 s sample.

    This documents the chosen safety margin and must pass for any encoder
    with subsample_factor=2 before training starts.
    """
    from gp1.models.quartznet import QuartzNet10x4

    torch.manual_seed(0)
    model = QuartzNet10x4(vocab_size=VOCAB_SIZE)
    assert model.subsample_factor == 2, "This test assumes s=2"

    t_prime = T_WORST_CASE // model.subsample_factor  # 175
    ratio = t_prime / U_MAX

    assert ratio >= 2.0, (
        f"T'/U_max = {t_prime}/{U_MAX} = {ratio:.2f} < 2.0 — "
        f"CTC training will be unstable with subsample_factor={model.subsample_factor}"
    )


def test_quartznet_output_lengths_are_int64():
    """output_lengths must be int64 (required by torch.nn.CTCLoss)."""
    from gp1.models.quartznet import QuartzNet10x4

    torch.manual_seed(0)
    model = QuartzNet10x4(vocab_size=VOCAB_SIZE)
    mel = torch.randn(2, 80, 300)
    mel_lengths = torch.tensor([300, 200], dtype=torch.long)
    out = model(mel, mel_lengths)
    assert out.output_lengths.dtype == torch.long, (
        f"output_lengths.dtype={out.output_lengths.dtype}, expected torch.long"
    )


def test_quartznet_log_probs_are_float32():
    """log_probs must be float32 for stable CTC loss computation."""
    from gp1.models.quartznet import QuartzNet10x4

    torch.manual_seed(0)
    model = QuartzNet10x4(vocab_size=VOCAB_SIZE)
    mel = torch.randn(2, 80, 300)
    mel_lengths = torch.tensor([300, 200], dtype=torch.long)
    out = model(mel, mel_lengths)
    assert out.log_probs.dtype == torch.float32, (
        f"log_probs.dtype={out.log_probs.dtype}, expected torch.float32"
    )


def test_quartznet_batch_size_1_works():
    """Model must handle batch_size=1 without errors (BN eval quirks)."""
    from gp1.models.quartznet import QuartzNet10x4

    torch.manual_seed(0)
    model = QuartzNet10x4(vocab_size=VOCAB_SIZE)
    model.eval()
    with torch.no_grad():
        mel = torch.randn(1, 80, 200)
        mel_lengths = torch.tensor([200], dtype=torch.long)
        out = model(mel, mel_lengths)
    assert out.log_probs.shape == (1, 100, VOCAB_SIZE)


def test_quartznet_no_nan_in_forward():
    """Forward pass must not produce NaN in log_probs or intermediate."""
    from gp1.models.quartznet import QuartzNet10x4

    torch.manual_seed(42)
    model = QuartzNet10x4(vocab_size=VOCAB_SIZE)
    mel = torch.randn(3, 80, 500)
    mel_lengths = torch.tensor([500, 400, 300], dtype=torch.long)
    out = model(mel, mel_lengths)
    assert not torch.isnan(out.log_probs).any(), "NaN detected in log_probs"
    if out.intermediate is not None:
        assert not torch.isnan(out.intermediate).any(), "NaN in intermediate"
