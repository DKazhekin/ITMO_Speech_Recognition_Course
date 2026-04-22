"""Tests for H5: output_lengths must use ceil-div not floor-div.

For any stride-2 op: floor(T/2) vs ceil(T/2) differs by 1 when T is odd.
PyTorch Conv1d with stride=2 produces ceil(T/2) frames by default
(same as: (T + 1) // 2).

TDD: tests FAIL before H5 fix (models use floor-div mel_lengths // s).
After fix, all models must use: (mel_lengths + s - 1) // s.

Also tests EfficientConformer progressive stride: each /2 stage applies
(T + 1) // 2, so overall 4x subsample is ceil(ceil(T/2) / 2).
"""

from __future__ import annotations

import pytest
import torch

VOCAB_SIZE = 35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _odd_mel(T: int, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Create an odd-T mel tensor with matching lengths."""
    assert T % 2 == 1, f"T must be odd, got {T}"
    mel = torch.randn(batch, 80, T)
    mel_lengths = torch.tensor(
        [T, T - 2], dtype=torch.long
    )  # second item also reasonable
    return mel, mel_lengths


def _ceil_div(x: torch.Tensor, s: int) -> torch.Tensor:
    return (x + s - 1) // s


# ---------------------------------------------------------------------------
# QuartzNet10x4 (s=2)
# ---------------------------------------------------------------------------


def test_quartznet_output_lengths_match_actual_log_probs_shape_odd_T():
    """For odd T, QuartzNet output_lengths must match log_probs.shape[1]."""
    from gp1.models.quartznet import QuartzNet10x4

    torch.manual_seed(0)
    model = QuartzNet10x4(vocab_size=VOCAB_SIZE).eval()

    T = 201  # odd
    mel, mel_lengths = _odd_mel(T)

    with torch.no_grad():
        out = model(mel, mel_lengths)

    actual_T_prime = out.log_probs.shape[1]
    reported_lengths = out.output_lengths

    # Each item in output_lengths must not exceed actual_T_prime
    assert (reported_lengths <= actual_T_prime).all(), (
        f"QuartzNet: output_lengths {reported_lengths.tolist()} > actual T'={actual_T_prime}"
    )

    # The first (full-length) item must equal the actual output time dimension
    assert reported_lengths[0].item() == actual_T_prime, (
        f"QuartzNet odd T={T}: output_lengths[0]={reported_lengths[0].item()} "
        f"!= actual T'={actual_T_prime}. "
        f"Expected ceil({T}/2)={(T + 1) // 2}, got floor={T // 2}."
    )


def test_quartznet_ceil_div_formula_for_various_odd_T():
    """QuartzNet output_lengths[0] must equal ceil(T/2) for multiple odd T values."""
    from gp1.models.quartznet import QuartzNet10x4

    torch.manual_seed(1)
    model = QuartzNet10x4(vocab_size=VOCAB_SIZE).eval()

    for T in [101, 201, 301, 401, 501]:
        mel = torch.randn(1, 80, T)
        mel_lengths = torch.tensor([T], dtype=torch.long)

        with torch.no_grad():
            out = model(mel, mel_lengths)

        expected = (T + 1) // 2  # ceil(T/2)
        actual_T_prime = out.log_probs.shape[1]
        reported = out.output_lengths[0].item()

        assert reported == expected, (
            f"QuartzNet T={T}: expected ceil(T/2)={expected}, "
            f"got output_lengths={reported} (floor would be {T // 2})"
        )
        assert reported == actual_T_prime, (
            f"QuartzNet T={T}: output_lengths={reported} != actual shape T'={actual_T_prime}"
        )


# ---------------------------------------------------------------------------
# CRDNN (s=1 or s=2)
# ---------------------------------------------------------------------------


def test_crdnn_subsample1_output_lengths_match_actual_shape_odd_T():
    """CRDNN with subsample_factor=1 — output_lengths must match actual T' for odd T."""
    from gp1.models.crdnn import CRDNN

    torch.manual_seed(2)
    model = CRDNN(vocab_size=VOCAB_SIZE, subsample_factor=1).eval()

    T = 201
    mel, mel_lengths = _odd_mel(T)

    with torch.no_grad():
        out = model(mel, mel_lengths)

    actual_T_prime = out.log_probs.shape[1]
    reported = out.output_lengths[0].item()

    # subsample_factor=1 → T' = T (no time striding in CRDNN with s=1)
    # The CNN blocks use stride=(2,1) — halve freq only, not time.
    assert reported == actual_T_prime, (
        f"CRDNN s=1 T={T}: output_lengths[0]={reported} != actual T'={actual_T_prime}"
    )


def test_crdnn_subsample2_output_lengths_match_actual_shape_odd_T():
    """CRDNN with subsample_factor=2 — output_lengths must match actual T' for odd T."""
    from gp1.models.crdnn import CRDNN

    torch.manual_seed(3)
    model = CRDNN(vocab_size=VOCAB_SIZE, subsample_factor=2).eval()

    T = 201  # odd
    mel, mel_lengths = _odd_mel(T)

    with torch.no_grad():
        out = model(mel, mel_lengths)

    actual_T_prime = out.log_probs.shape[1]
    reported = out.output_lengths[0].item()

    expected = (T + 1) // 2  # ceil(T/2)

    assert reported == expected, (
        f"CRDNN s=2 T={T}: expected ceil(T/2)={expected}, "
        f"got output_lengths={reported} (floor would be {T // 2})"
    )
    assert reported == actual_T_prime, (
        f"CRDNN s=2 T={T}: output_lengths={reported} != actual T'={actual_T_prime}"
    )


# ---------------------------------------------------------------------------
# EfficientConformer (progressive stride: /2 + /2 = /4)
# ---------------------------------------------------------------------------


def test_efficient_conformer_output_lengths_match_actual_shape_odd_T():
    """EfficientConformer output_lengths must match actual T' for odd T input."""
    from gp1.models.efficient_conformer import EfficientConformer

    torch.manual_seed(4)
    model = EfficientConformer(vocab_size=VOCAB_SIZE).eval()

    T = 201  # odd
    mel, mel_lengths = _odd_mel(T)

    with torch.no_grad():
        out = model(mel, mel_lengths)

    actual_T_prime = out.log_probs.shape[1]
    reported = out.output_lengths[0].item()

    # Two stride-2 stages: ceil(ceil(T/2)/2)
    after_prologue = (T + 1) // 2
    expected = (after_prologue + 1) // 2

    assert reported == expected, (
        f"EfficientConformer T={T}: expected progressive-ceil={expected}, "
        f"got output_lengths={reported} (flat floor would be {T // 4})"
    )
    assert reported == actual_T_prime, (
        f"EfficientConformer T={T}: output_lengths={reported} != actual T'={actual_T_prime}"
    )


def test_efficient_conformer_ceil_div_for_various_odd_T():
    """EfficientConformer: output_lengths == ceil(ceil(T/2)/2) for multiple odd T."""
    from gp1.models.efficient_conformer import EfficientConformer

    torch.manual_seed(5)
    model = EfficientConformer(vocab_size=VOCAB_SIZE).eval()

    for T in [101, 201, 301, 401]:
        mel = torch.randn(1, 80, T)
        mel_lengths = torch.tensor([T], dtype=torch.long)

        with torch.no_grad():
            out = model(mel, mel_lengths)

        after_prologue = (T + 1) // 2
        expected = (after_prologue + 1) // 2
        actual_T_prime = out.log_probs.shape[1]
        reported = out.output_lengths[0].item()

        assert reported == expected, (
            f"EfficientConformer T={T}: expected {expected}, got {reported}"
        )
        assert reported == actual_T_prime, (
            f"EfficientConformer T={T}: output_lengths={reported} != shape T'={actual_T_prime}"
        )


# ---------------------------------------------------------------------------
# FastConformerBPE (s=4 via SubsampleConv)
# ---------------------------------------------------------------------------


def test_fast_conformer_bpe_output_lengths_match_actual_shape_odd_T():
    """FastConformerBPE output_lengths must match actual T' for odd T."""
    pytest.importorskip("sentencepiece")  # BPEVocab requires sentencepiece
    from gp1.models.fast_conformer_bpe import FastConformerBPE

    torch.manual_seed(6)
    model = FastConformerBPE(vocab_size=256).eval()

    T = 201  # odd
    mel, mel_lengths = _odd_mel(T)

    with torch.no_grad():
        out = model(mel, mel_lengths)

    actual_T_prime = out.log_probs.shape[1]
    reported = out.output_lengths[0].item()

    # SubsampleConv factor=4: two stride-2 stages, so ceil(ceil(T/2)/2)
    after_s1 = (T + 1) // 2
    expected = (after_s1 + 1) // 2

    assert reported == expected, (
        f"FastConformerBPE T={T}: expected {expected}, got {reported}"
    )
    assert reported == actual_T_prime, (
        f"FastConformerBPE T={T}: output_lengths={reported} != actual T'={actual_T_prime}"
    )


def test_fast_conformer_bpe_output_lengths_no_sentencepiece():
    """FastConformerBPE output_lengths must match actual T' for odd T (no BPEVocab needed)."""
    from gp1.models.fast_conformer_bpe import FastConformerBPE

    torch.manual_seed(7)
    # vocab_size=35 works without BPEVocab
    model = FastConformerBPE(vocab_size=35).eval()

    T = 201  # odd
    mel = torch.randn(2, 80, T)
    mel_lengths = torch.tensor([T, T - 2], dtype=torch.long)

    with torch.no_grad():
        out = model(mel, mel_lengths)

    actual_T_prime = out.log_probs.shape[1]
    reported = out.output_lengths[0].item()

    # SubsampleConv factor=4: two stride-2 stages
    after_s1 = (T + 1) // 2
    expected = (after_s1 + 1) // 2

    assert reported == expected, (
        f"FastConformerBPE T={T}: expected {expected}, got {reported}"
    )
    assert reported == actual_T_prime, (
        f"FastConformerBPE T={T}: output_lengths={reported} != actual T'={actual_T_prime}"
    )


# ---------------------------------------------------------------------------
# Cross-model: even T should also be consistent
# ---------------------------------------------------------------------------


def test_all_models_even_T_output_lengths_match_shape():
    """Even T — ceil-div == floor-div, so all models must still be consistent."""
    from gp1.models.quartznet import QuartzNet10x4
    from gp1.models.crdnn import CRDNN
    from gp1.models.efficient_conformer import EfficientConformer
    from gp1.models.fast_conformer_bpe import FastConformerBPE

    torch.manual_seed(99)

    T = 200  # even

    cases = [
        (QuartzNet10x4(vocab_size=VOCAB_SIZE).eval(), T),
        (CRDNN(vocab_size=VOCAB_SIZE, subsample_factor=2).eval(), T),
        (EfficientConformer(vocab_size=VOCAB_SIZE).eval(), T),
        (FastConformerBPE(vocab_size=VOCAB_SIZE).eval(), T),
    ]

    for model, t in cases:
        mel = torch.randn(1, 80, t)
        mel_lengths = torch.tensor([t], dtype=torch.long)
        with torch.no_grad():
            out = model(mel, mel_lengths)
        actual = out.log_probs.shape[1]
        reported = out.output_lengths[0].item()
        assert reported == actual, (
            f"{type(model).__name__} T={t}: output_lengths={reported} != actual T'={actual}"
        )
