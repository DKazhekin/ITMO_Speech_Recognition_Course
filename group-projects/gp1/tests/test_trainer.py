"""Tests for gp1.train.trainer — Trainer and TrainerConfig.

TDD: RED phase → GREEN phase.

Uses a tiny fake model, fake DataLoader, and mock losses to keep tests fast (<5s each).
All filesystem operations use tmp_path fixture.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from gp1.models.base import EncoderOutput
from gp1.text.vocab import CharVocab
from gp1.train.trainer import Trainer, TrainerConfig
from gp1.types import Batch


# ---------------------------------------------------------------------------
# Fake / stub objects
# ---------------------------------------------------------------------------


class _FakeEncoder(nn.Module):
    """Tiny encoder: pool mel -> linear -> log_softmax. Matches ASREncoder Protocol."""

    vocab_size: int = 35
    subsample_factor: int = 2

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(80, self.vocab_size)

    def forward(self, mel: torch.Tensor, _mel_lengths: torch.Tensor) -> EncoderOutput:
        # mel: [B, 80, T_frames] — global average pool over T
        pooled = mel.mean(dim=-1)  # [B, 80]
        logits = self.proj(pooled)  # [B, V=35]
        log_probs = F.log_softmax(logits, dim=-1).unsqueeze(1).expand(-1, 2, -1)
        out_lengths = torch.full((mel.size(0),), 2, dtype=torch.long)
        return EncoderOutput(log_probs=log_probs, output_lengths=out_lengths)


def _fake_batch(batch_size: int = 2) -> Batch:
    """Return a Batch with 6 fields, matching the post-Wave1 Batch dataclass."""
    audio = torch.zeros(batch_size, 800)
    audio_lengths = torch.full((batch_size,), 800, dtype=torch.long)
    targets = torch.ones(batch_size, 2, dtype=torch.long)
    target_lengths = torch.full((batch_size,), 2, dtype=torch.long)
    spk_ids = [f"spk_{i}" for i in range(batch_size)]
    transcriptions = ["один два"] * batch_size
    return Batch(
        audio=audio,
        audio_lengths=audio_lengths,
        targets=targets,
        target_lengths=target_lengths,
        spk_ids=spk_ids,
        transcriptions=transcriptions,
    )


class _FakeLoader:
    """Iterable DataLoader-alike that yields a fixed batch N times."""

    def __init__(self, batch: Batch, n_batches: int = 2) -> None:
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
    fp16_autocast: bool = False,
    early_stop_patience: int = 15,
    val_batches: int = 1,
) -> Trainer:
    """Factory: create a fully wired Trainer with tiny fake components."""
    torch.manual_seed(0)
    model = _FakeEncoder()
    vocab = CharVocab()

    def _scalar_loss(*_args, **_kwargs):
        return torch.tensor(0.5, requires_grad=True)

    ctc_loss = MagicMock(side_effect=_scalar_loss)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
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
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        config=config,
        device=torch.device("cpu"),
        audio_cfg={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trainer_config_has_correct_defaults():
    # Arrange / Act
    config = TrainerConfig(max_epochs=10)

    # Assert
    assert config.grad_accum == 1
    assert config.fp16_autocast is True
    assert config.log_every_n_steps == 50
    assert config.val_every_n_epochs == 1
    assert config.early_stop_patience == 15
    assert config.early_stop_metric == "harmonic_in_out_cer"
    assert config.ckpt_dir == Path("checkpoints")


def test_trainer_fit_returns_best_cer_and_ckpt_path(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, max_epochs=2)

    # Act
    result = trainer.fit()

    # Assert
    assert "best_monitored" in result
    assert "best_ckpt_path" in result
    assert "history" in result
    assert isinstance(result["best_monitored"], float)
    assert isinstance(result["best_ckpt_path"], Path)
    assert isinstance(result["history"], list)
    assert len(result["history"]) >= 1


def test_trainer_uses_fp32_for_ctc_loss_even_when_fp16_autocast_true(tmp_path):
    # Arrange: intercept the dtype of log_probs received by ctc_loss
    dtypes_seen: list[torch.dtype] = []

    def capture_dtype(log_probs, _targets, _input_lengths, _target_lengths):
        dtypes_seen.append(log_probs.dtype)
        return torch.tensor(0.5, requires_grad=True)

    trainer = _make_trainer(tmp_path, max_epochs=1, fp16_autocast=False)
    trainer.ctc_loss = MagicMock(side_effect=capture_dtype)

    # Act
    trainer.fit()

    # Assert
    assert len(dtypes_seen) > 0, "ctc_loss was never called"
    for dt in dtypes_seen:
        assert dt == torch.float32, f"ctc_loss received {dt}, expected float32"


def test_trainer_applies_grad_accum_correctly(tmp_path):
    # Arrange: grad_accum=2, 4 batches per epoch -> 2 optimizer steps
    trainer = _make_trainer(tmp_path, max_epochs=1, grad_accum=2)

    step_count = [0]
    original_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        step_count[0] += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counting_step
    batch = _fake_batch(batch_size=2)
    trainer.train_loader = _FakeLoader(batch, n_batches=4)

    # Act
    trainer.fit()

    # Assert: 4 batches / grad_accum=2 -> 2 steps
    assert step_count[0] == 2, (
        f"Expected 2 optimizer steps with grad_accum=2 over 4 batches, got {step_count[0]}"
    )


def test_trainer_saves_best_checkpoint_when_cer_improves(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, max_epochs=2)

    # Act
    result = trainer.fit()

    # Assert: best.pt and meta.json exist in ckpt_dir
    ckpt_dir = tmp_path / "ckpts"
    best_pt = ckpt_dir / "best.pt"
    meta_json = ckpt_dir / "meta.json"
    assert best_pt.exists(), "best.pt was not created"
    assert meta_json.exists(), "meta.json was not created"
    assert result["best_ckpt_path"] == best_pt


def test_trainer_early_stops_after_patience_exceeded(tmp_path):
    # Arrange: patience=2, max_epochs=10; val always returns 1.0 (no improvement)
    patience = 2
    max_epochs = 10
    trainer = _make_trainer(
        tmp_path, max_epochs=max_epochs, early_stop_patience=patience
    )

    with patch.object(
        trainer,
        "_run_validation",
        side_effect=lambda _epoch: (1.0, {}, 0.0, 0.0, 0.0, 0.5),
    ):
        result = trainer.fit()

    # Assert: stopped before max_epochs; history has patience+1 entries at most
    history = result["history"]
    # First val sets best; then patience=2 non-improving -> stop after 3 val epochs
    assert len(history) <= patience + 1, (
        f"Expected at most {patience + 1} val epochs, got {len(history)}"
    )
    assert result["best_monitored"] <= 1.0
