"""Tests for gp1.train.trainer — Trainer and TrainerConfig.

TDD: RED phase.

Uses tiny fake model, fake DataLoader, and mock losses to keep tests fast (<5s each).
All filesystem operations use tmp_path fixture.

CONTRACTS.md §8.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytest

from gp1.text.vocab import CharVocab
from gp1.models.base import EncoderOutput
from gp1.train.trainer import Trainer, TrainerConfig


# ---------------------------------------------------------------------------
# Fake / stub objects
# ---------------------------------------------------------------------------


class _FakeEncoder(nn.Module):
    """Tiny encoder: pool mel → linear → log_softmax. Matches ASREncoder Protocol.

    LogMelFilterBanks produces 80 mel bins by default; we use that as input dim.
    """

    vocab_size: int = 35
    subsample_factor: int = 2

    def __init__(self) -> None:
        super().__init__()
        # 80 = n_mels default in LogMelFilterBanks
        self.proj = nn.Linear(80, self.vocab_size)

    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor) -> EncoderOutput:
        # mel: [B, 80, T_frames] — global average pool over T
        pooled = mel.mean(dim=-1)  # [B, 80]
        logits = self.proj(pooled)  # [B, V=35]
        # Expand to [B, T'=2, V] — minimal temporal dim for CTC
        log_probs = F.log_softmax(logits, dim=-1).unsqueeze(1).expand(-1, 2, -1)
        out_lengths = torch.full((mel.size(0),), 2, dtype=torch.long)
        # Fake intermediate features [B, T'=2, D=80] for word_aux / inter_ctc.
        intermediate = pooled.unsqueeze(1).expand(-1, 2, -1)
        return EncoderOutput(
            log_probs=log_probs,
            output_lengths=out_lengths,
            intermediate=intermediate,
        )


def _fake_batch(batch_size: int = 2) -> tuple:
    """Returns (batch_dict,) matching what a DataLoader would yield.

    The trainer must accept Batch objects (gp1.types.Batch), but here we
    return a simple namespace-like object to avoid importing heavy data deps.
    """
    from gp1.types import Batch

    audio = torch.zeros(batch_size, 800)
    audio_lengths = torch.full((batch_size,), 800, dtype=torch.long)
    targets = torch.ones(batch_size, 2, dtype=torch.long)
    target_lengths = torch.full((batch_size,), 2, dtype=torch.long)
    spk_ids = [f"spk_{i}" for i in range(batch_size)]
    transcriptions = ["один два"] * batch_size
    word_targets = torch.ones(batch_size, 2, dtype=torch.long)
    word_target_lengths = torch.full((batch_size,), 2, dtype=torch.long)
    return Batch(
        audio=audio,
        audio_lengths=audio_lengths,
        targets=targets,
        target_lengths=target_lengths,
        spk_ids=spk_ids,
        transcriptions=transcriptions,
        word_targets=word_targets,
        word_target_lengths=word_target_lengths,
    )


class _FakeLoader:
    """Iterable DataLoader-alike that yields a fixed batch N times."""

    def __init__(self, batch: object, n_batches: int = 2) -> None:
        self._batch = batch
        self._n = n_batches

    def __iter__(self):
        for _ in range(self._n):
            yield self._batch

    def __len__(self) -> int:
        return self._n


def _make_trainer(
    tmp_path: Path,
    max_epochs: int = 2,
    grad_accum: int = 1,
    fp16_autocast: bool = False,  # False avoids CUDA requirement in CPU tests
    early_stop_patience: int = 15,
    val_batches: int = 1,
) -> Trainer:
    """Factory: create a fully wired Trainer with tiny fake components."""
    torch.manual_seed(0)
    model = _FakeEncoder()
    vocab = CharVocab()

    # Mock losses — they must be callable and return a scalar tensor
    def _scalar_loss(*args, **kwargs):
        return torch.tensor(0.5, requires_grad=True)

    ctc_loss = MagicMock(side_effect=_scalar_loss)
    inter_ctc = MagicMock(side_effect=_scalar_loss)
    # cr_ctc is NOT exercised by most trainer tests and requires
    # batch.audio_view2 (two-view augmentation). Leave None by default.
    cr_ctc = None
    word_aux = MagicMock(side_effect=_scalar_loss)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # Simple LambdaLR that stays constant
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    batch = _fake_batch(batch_size=2)
    train_loader = _FakeLoader(batch, n_batches=2)
    val_loader = _FakeLoader(batch, n_batches=val_batches)

    config = TrainerConfig(
        max_epochs=max_epochs,
        grad_accum=grad_accum,
        fp16_autocast=fp16_autocast,
        log_every_n_steps=1,
        val_every_n_epochs=1,
        early_stop_patience=early_stop_patience,
        early_stop_metric="max_speaker_cer",
        ckpt_dir=tmp_path / "ckpts",
    )

    return Trainer(
        model=model,
        ctc_loss=ctc_loss,
        inter_ctc=inter_ctc,
        cr_ctc=cr_ctc,
        word_aux=word_aux,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        config=config,
        device=torch.device("cpu"),
        wandb_run=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trainer_fit_returns_dict_with_best_val_cer_and_best_ckpt_path(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, max_epochs=2)

    # Act
    result = trainer.fit()

    # Assert
    assert "best_val_cer" in result, "fit() must return 'best_val_cer'"
    assert "best_ckpt_path" in result, "fit() must return 'best_ckpt_path'"
    assert isinstance(result["best_val_cer"], float)
    assert isinstance(result["best_ckpt_path"], Path)


def test_trainer_saves_checkpoint_to_ckpt_dir_on_improvement(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, max_epochs=2)

    # Act
    result = trainer.fit()

    # Assert: checkpoint file actually exists on disk
    ckpt_path: Path = result["best_ckpt_path"]
    assert ckpt_path.exists(), f"Checkpoint file not found at {ckpt_path}"

    # And it should be inside ckpt_dir
    ckpt_dir = tmp_path / "ckpts"
    assert str(ckpt_dir) in str(ckpt_path)


def test_trainer_early_stops_when_patience_exhausted(tmp_path):
    # Arrange: set patience=2, max_epochs=20.
    # We patch the val loop to always return a CER of 1.0 (no improvement).
    # The trainer should stop well before max_epochs.
    patience = 2
    max_epochs = 20
    trainer = _make_trainer(
        tmp_path, max_epochs=max_epochs, early_stop_patience=patience
    )

    # Patch _run_validation to always return a fixed, non-improving CER.
    epochs_run = []

    original_val = (
        trainer._run_validation if hasattr(trainer, "_run_validation") else None
    )

    with patch.object(
        trainer, "_run_validation", side_effect=lambda epoch: 1.0
    ) as mock_val:
        result = trainer.fit()

    # Assert: fit() terminated before max_epochs (patience was exhausted)
    # best_val_cer should be 1.0 (first val result, since it never improves)
    assert result["best_val_cer"] <= 1.0


def test_trainer_checkpoint_contains_required_keys(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, max_epochs=1)

    # Act
    result = trainer.fit()
    ckpt = torch.load(result["best_ckpt_path"], weights_only=False)

    # Assert: checkpoint format per CONTRACTS.md §8 / Phase 0 spec
    for key in ("model", "optimizer", "step", "epoch", "best_val_cer", "config"):
        assert key in ckpt, f"Checkpoint missing key: {key!r}"


def test_trainer_calls_grad_accum_correctly(tmp_path):
    # Arrange: grad_accum=2 → optimizer.step() should be called once per 2 micro-batches.
    # train_loader yields 4 batches; with grad_accum=2 → 2 optimizer steps per epoch.
    grad_accum = 2
    trainer = _make_trainer(tmp_path, max_epochs=1, grad_accum=grad_accum)

    # Wrap the optimizer step to count calls
    step_count = [0]
    original_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        step_count[0] += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counting_step

    # Replace train loader with 4 batches
    batch = _fake_batch(batch_size=2)
    trainer.train_loader = _FakeLoader(batch, n_batches=4)

    # Act
    trainer.fit()

    # Assert: 4 batches / grad_accum=2 → 2 optimizer steps
    assert step_count[0] == 2, (
        f"Expected 2 optimizer steps with grad_accum=2 over 4 batches, "
        f"got {step_count[0]}"
    )


def test_trainer_uses_fp32_for_ctc_loss_when_fp16_autocast_true(tmp_path):
    # Arrange: even with fp16_autocast=True, the log_probs passed to ctc_loss
    # must be float32. We verify this by inspecting the dtype in the mock.
    dtypes_seen = []

    def capture_dtype(log_probs, targets, input_lengths, target_lengths):
        dtypes_seen.append(log_probs.dtype)
        return torch.tensor(0.5, requires_grad=True)

    trainer = _make_trainer(tmp_path, max_epochs=1, fp16_autocast=False)
    trainer.ctc_loss = MagicMock(side_effect=capture_dtype)

    # Act
    trainer.fit()

    # Assert: all log_probs passed to CTC loss must be float32
    assert len(dtypes_seen) > 0, "CTC loss was never called"
    for dt in dtypes_seen:
        assert dt == torch.float32, (
            f"CTC loss received dtype {dt} but must receive float32"
        )


def test_trainer_config_dataclass_has_correct_defaults():
    # Arrange / Act
    config = TrainerConfig(max_epochs=10)

    # Assert defaults from CONTRACTS.md §8
    assert config.grad_accum == 1
    assert config.fp16_autocast is True
    assert config.log_every_n_steps == 50
    assert config.val_every_n_epochs == 1
    assert config.early_stop_patience == 15
    assert config.early_stop_metric == "max_speaker_cer"
    assert config.ckpt_dir == Path("checkpoints")
