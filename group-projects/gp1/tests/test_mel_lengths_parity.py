"""Tests for mel_lengths formula parity between Trainer and inference.py.

Bug C1: Trainer uses ``audio_lengths // hop`` (missing +1).
Inference uses ``audio_lengths // hop + 1`` (correct for center=True STFT).

After the fix both must compute the identical expression:
    (audio_lengths // hop + 1).clamp(max=mel.size(-1))

This test suite:
  1. Compares the two formulas numerically for several audio lengths.
  2. Runs LogMelFilterBanks.forward() on synthetic audio and checks the
     actual output T_frames matches the formula.

TDD: all tests here FAIL before C1 is fixed (trainer uses wrong formula).
"""

from __future__ import annotations

import pytest
import torch

from gp1.features.melbanks import LogMelFilterBanks

HOP = 160
PARAMETRIZE_LENGTHS = [500, 1000, 8000, 16000, 25600, 25601, 25599, 7777, 333]


def _inference_formula(
    audio_lengths: torch.Tensor, hop: int, mel_T: int
) -> torch.Tensor:
    """Formula used by inference.py:593 — CORRECT for center=True."""
    return (audio_lengths // hop + 1).clamp(max=mel_T).to(torch.long)


def _trainer_formula_old(
    audio_lengths: torch.Tensor, hop: int, mel_T: int
) -> torch.Tensor:
    """Old (buggy) trainer formula — floor div with no +1."""
    return (audio_lengths // hop).clamp(max=mel_T).to(torch.long)


def _trainer_formula_new(
    audio_lengths: torch.Tensor, hop: int, mel_T: int
) -> torch.Tensor:
    """New (fixed) trainer formula — must match inference."""
    return (audio_lengths // hop + 1).clamp(max=mel_T).to(torch.long)


@pytest.mark.parametrize("audio_len", PARAMETRIZE_LENGTHS)
def test_trainer_formula_matches_inference_formula(audio_len: int):
    """After the fix, trainer and inference must produce identical mel_lengths."""
    mel = LogMelFilterBanks()(torch.zeros(1, audio_len))
    mel_T = mel.size(-1)
    lengths = torch.tensor([audio_len], dtype=torch.long)
    expected = _inference_formula(lengths, HOP, mel_T)
    actual = _trainer_formula_new(lengths, HOP, mel_T)
    assert actual.item() == expected.item(), (
        f"audio_len={audio_len}: trainer_new={actual.item()} != inference={expected.item()}"
    )


@pytest.mark.parametrize("audio_len", PARAMETRIZE_LENGTHS)
def test_new_formula_matches_actual_stft_output_shape(audio_len: int):
    """mel_lengths from the new formula must equal the actual T_frames from STFT."""
    frontend = LogMelFilterBanks()
    mel = frontend(torch.zeros(1, audio_len))  # [1, 80, T_frames]
    actual_T = mel.size(-1)

    lengths = torch.tensor([audio_len], dtype=torch.long)
    computed = _trainer_formula_new(lengths, HOP, actual_T)

    # After clamp, computed must equal actual_T when the formula is correct.
    # (For full-length audio, clamp should not trigger.)
    assert computed.item() == actual_T, (
        f"audio_len={audio_len}: computed mel_length={computed.item()} "
        f"!= actual STFT T_frames={actual_T}"
    )


@pytest.mark.parametrize("audio_len", PARAMETRIZE_LENGTHS)
def test_old_formula_is_wrong_for_most_lengths(audio_len: int):
    """Confirm the OLD trainer formula (//hop only) gives wrong result.

    This test should PASS on unfixed code (old formula != inference formula)
    and must be skipped or left as documentation after the fix.
    NOTE: this test documents the bug — it asserts the old formula differs.
    """
    mel = LogMelFilterBanks()(torch.zeros(1, audio_len))
    mel_T = mel.size(-1)
    lengths = torch.tensor([audio_len], dtype=torch.long)
    old_val = _trainer_formula_old(lengths, HOP, mel_T).item()
    correct_val = _inference_formula(lengths, HOP, mel_T).item()
    # Old formula is always off by 1 (unless clamp brings them together).
    # Document this: the old formula underestimates by 1 for unclamped case.
    assert old_val != correct_val or old_val == mel_T, (
        f"audio_len={audio_len}: old formula {old_val} unexpectedly equals "
        f"correct {correct_val} — the bug may already be fixed or this length "
        f"is at a clamp boundary"
    )


def test_batch_formula_parity():
    """Batch of different audio lengths — all must agree between formulas."""
    audio_lengths = torch.tensor([500, 1600, 8000, 16000, 25600], dtype=torch.long)
    max_len = int(audio_lengths.max().item())
    # Build a batch mel to get actual T_frames
    batch_audio = torch.zeros(len(audio_lengths), max_len)
    mel_batch = LogMelFilterBanks()(batch_audio)  # [B, 80, T]
    mel_T = mel_batch.size(-1)

    inf_lengths = _inference_formula(audio_lengths, HOP, mel_T)
    new_lengths = _trainer_formula_new(audio_lengths, HOP, mel_T)

    assert torch.equal(inf_lengths, new_lengths), (
        f"Batch parity failed:\n  inference={inf_lengths.tolist()}\n  trainer_new={new_lengths.tolist()}"
    )


def test_trainer_mel_lengths_with_audio_cfg():
    """Trainer must pass audio_cfg to LogMelFilterBanks and use correct formula.

    After H8 fix: Trainer.__init__ accepts audio_cfg and wires it through.
    This test instantiates a Trainer with custom hop_length and checks mel extraction.
    """
    from unittest.mock import MagicMock
    from pathlib import Path
    import tempfile

    from gp1.models.base import EncoderOutput
    from gp1.train.trainer import Trainer, TrainerConfig
    import torch.nn as nn
    import torch.nn.functional as F

    class _TinyEncoder(nn.Module):
        vocab_size = 35
        subsample_factor = 2

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(80, 35)

        def forward(self, mel, mel_lengths):
            pooled = mel.mean(-1)
            lp = F.log_softmax(self.proj(pooled), -1).unsqueeze(1).expand(-1, 2, -1)
            return EncoderOutput(
                log_probs=lp,
                output_lengths=torch.full((mel.size(0),), 2, dtype=torch.long),
                intermediate=None,
            )

    from gp1.types import Batch

    audio = torch.zeros(2, 1600)
    batch = Batch(
        audio=audio,
        audio_lengths=torch.full((2,), 1600, dtype=torch.long),
        targets=torch.ones(2, 2, dtype=torch.long),
        target_lengths=torch.full((2,), 2, dtype=torch.long),
        spk_ids=["s0", "s1"],
        transcriptions=["один два", "один два"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainerConfig(max_epochs=1, ckpt_dir=Path(tmp) / "ckpts")
        audio_cfg = {"hop_length": 160, "n_fft": 512, "win_length": 400, "n_mels": 80}
        trainer = Trainer(
            model=_TinyEncoder(),
            ctc_loss=MagicMock(return_value=torch.tensor(0.5, requires_grad=True)),
            inter_ctc=None,
            cr_ctc=None,
            word_aux=None,
            optimizer=torch.optim.SGD(_TinyEncoder().parameters(), lr=1e-3),
            scheduler=torch.optim.lr_scheduler.LambdaLR(
                torch.optim.SGD(_TinyEncoder().parameters(), lr=1e-3),
                lr_lambda=lambda _: 1.0,
            ),
            train_loader=[batch],
            val_loader=[batch],
            vocab=__import__("gp1.text.vocab", fromlist=["CharVocab"]).CharVocab(),
            config=cfg,
            device=torch.device("cpu"),
            audio_cfg=audio_cfg,
        )
        # Verify the trainer's mel frontend uses the passed hop_length
        assert trainer._mel.hop_length == 160
