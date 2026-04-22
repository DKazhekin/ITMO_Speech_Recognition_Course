"""Integration tests for CR-CTC two-view wiring across the pipeline.

Covers:
  - Batch dataclass accepts optional audio_view2/audio_view2_lengths fields.
  - collate_fn propagates audio_view2 from items (padded like audio).
  - Trainer._forward_batch runs a second encoder pass on batch.audio_view2
    and adds 0.2 * cr_ctc(log_probs_a, log_probs_b, output_lengths).
  - Missing audio_view2 when cr_ctc head is active raises ValueError.

Follows the same pattern as tests/test_word_aux_wiring.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from gp1.data.collate import collate_fn
from gp1.losses.ctc import CTCLoss
from gp1.models.quartznet import QuartzNet10x4
from gp1.text.vocab import CharVocab
from gp1.train.trainer import Trainer, TrainerConfig
from gp1.types import Batch


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------


class TestBatchAudioView2Fields:
    def test_batch_accepts_audio_view2_kwargs(self) -> None:
        # Arrange
        audio_view2 = torch.zeros(2, 160, dtype=torch.float32)
        audio_view2_lengths = torch.tensor([160, 80], dtype=torch.int64)

        # Act
        batch = Batch(
            audio=torch.zeros(2, 160),
            audio_lengths=torch.tensor([160, 80]),
            targets=torch.zeros(2, 3, dtype=torch.int64),
            target_lengths=torch.tensor([3, 3]),
            spk_ids=["a", "b"],
            transcriptions=["1", "2"],
            audio_view2=audio_view2,
            audio_view2_lengths=audio_view2_lengths,
        )

        # Assert
        assert torch.equal(batch.audio_view2, audio_view2)
        assert torch.equal(batch.audio_view2_lengths, audio_view2_lengths)

    def test_batch_defaults_view2_fields_to_none(self) -> None:
        # Arrange / Act
        batch = Batch(
            audio=torch.zeros(1, 160),
            audio_lengths=torch.tensor([160]),
            targets=torch.zeros(1, 3, dtype=torch.int64),
            target_lengths=torch.tensor([3]),
            spk_ids=["a"],
            transcriptions=["1"],
        )

        # Assert
        assert batch.audio_view2 is None
        assert batch.audio_view2_lengths is None


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _item(
    audio_len: int,
    target_len: int,
    idx: int,
    audio_view2: torch.Tensor | None = None,
) -> dict:
    d: dict = {
        "audio": torch.zeros(audio_len, dtype=torch.float32),
        "target": torch.arange(target_len, dtype=torch.int64),
        "spk_id": f"spk_{idx}",
        "transcription": str(10000 + idx),
    }
    if audio_view2 is not None:
        d["audio_view2"] = audio_view2
    return d


class TestCollateAudioView2:
    def test_collate_without_view2_leaves_view2_fields_none(self) -> None:
        # Arrange
        items = [_item(160, 5, 0), _item(320, 6, 1)]

        # Act
        batch = collate_fn(items)

        # Assert
        assert batch.audio_view2 is None
        assert batch.audio_view2_lengths is None

    def test_collate_pads_audio_view2_to_multiple_of_hop(self) -> None:
        # Arrange
        view2_a = torch.ones(160, dtype=torch.float32)
        view2_b = torch.ones(320, dtype=torch.float32)
        items = [
            _item(160, 5, 0, audio_view2=view2_a),
            _item(320, 6, 1, audio_view2=view2_b),
        ]

        # Act
        batch = collate_fn(items)

        # Assert
        assert batch.audio_view2 is not None
        assert batch.audio_view2_lengths is not None
        # Padded to max length (already multiples of hop=160)
        assert batch.audio_view2.shape == (2, 320)
        assert batch.audio_view2.dtype == torch.float32
        assert batch.audio_view2_lengths.tolist() == [160, 320]
        # Content preserved with zero-padding
        assert torch.equal(batch.audio_view2[0, :160], torch.ones(160))
        assert torch.all(batch.audio_view2[0, 160:] == 0)


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


def _build_batch_with_view2() -> Batch:
    audio = torch.zeros(2, 16000, dtype=torch.float32)
    audio_lengths = torch.tensor([16000, 16000], dtype=torch.int64)
    # A second, independent augmented view — here simply random noise.
    audio_view2 = torch.randn(2, 16000, dtype=torch.float32) * 0.01
    audio_view2_lengths = torch.tensor([16000, 16000], dtype=torch.int64)
    char_targets = torch.randint(1, 30, (2, 3), dtype=torch.int64)
    char_target_lengths = torch.tensor([3, 3], dtype=torch.int64)
    return Batch(
        audio=audio,
        audio_lengths=audio_lengths,
        targets=char_targets,
        target_lengths=char_target_lengths,
        spk_ids=["a", "b"],
        transcriptions=["100000", "100000"],
        audio_view2=audio_view2,
        audio_view2_lengths=audio_view2_lengths,
    )


def _make_trainer(cr_ctc=None):
    vocab = CharVocab()
    model = QuartzNet10x4(vocab_size=vocab.vocab_size, d_model=64)
    ctc_loss = CTCLoss(blank_id=vocab.blank_id)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    cfg = TrainerConfig(max_epochs=1, fp16_autocast=False, ckpt_dir=Path("/tmp"))
    return Trainer(
        model=model,
        ctc_loss=ctc_loss,
        inter_ctc=None,
        cr_ctc=cr_ctc,
        word_aux=None,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=[],
        val_loader=[],
        vocab=vocab,
        config=cfg,
        device=torch.device("cpu"),
    )


class TestTrainerCrCtcPath:
    def test_forward_batch_calls_cr_ctc_with_two_log_probs(self) -> None:
        # Arrange — mock cr_ctc to capture the call
        captured: dict = {}

        def _fake_cr_ctc(log_probs_a, log_probs_b, input_lengths):
            captured["lp_a_shape"] = tuple(log_probs_a.shape)
            captured["lp_b_shape"] = tuple(log_probs_b.shape)
            captured["in_lens"] = input_lengths.clone()
            return torch.tensor(0.7, requires_grad=True)

        cr_ctc = MagicMock(side_effect=_fake_cr_ctc)
        trainer = _make_trainer(cr_ctc=cr_ctc)
        batch = _build_batch_with_view2()

        # Act
        loss = trainer._forward_batch(batch)

        # Assert
        assert cr_ctc.call_count == 1
        assert captured["lp_a_shape"] == captured["lp_b_shape"]
        assert captured["lp_a_shape"][0] == 2  # batch dim
        assert torch.isfinite(loss).item()
        assert loss.item() > 0.0

    def test_forward_batch_raises_when_cr_ctc_set_but_view2_missing(self) -> None:
        # Arrange
        cr_ctc = MagicMock()
        trainer = _make_trainer(cr_ctc=cr_ctc)
        batch = Batch(
            audio=torch.zeros(2, 16000),
            audio_lengths=torch.tensor([16000, 16000]),
            targets=torch.randint(1, 30, (2, 3), dtype=torch.int64),
            target_lengths=torch.tensor([3, 3]),
            spk_ids=["a", "b"],
            transcriptions=["100000", "100000"],
            audio_view2=None,
            audio_view2_lengths=None,
        )

        # Act / Assert
        with pytest.raises(ValueError, match="audio_view2"):
            trainer._forward_batch(batch)
        cr_ctc.assert_not_called()

    def test_forward_batch_skips_cr_ctc_when_head_is_none(self) -> None:
        # Arrange — no cr_ctc head; batch still has view2 but should be ignored
        trainer = _make_trainer(cr_ctc=None)
        batch = _build_batch_with_view2()

        # Act — must not crash, must produce a finite loss
        loss = trainer._forward_batch(batch)

        # Assert
        assert torch.isfinite(loss).item()
        assert loss.item() > 0.0
