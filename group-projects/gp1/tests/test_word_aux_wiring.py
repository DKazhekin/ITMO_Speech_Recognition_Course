"""Integration tests for word-aux CTC wiring across the data+trainer pipeline.

Covers:
  - Batch dataclass accepts optional word_targets/word_target_lengths fields.
  - SpokenNumbersDataset with word_vocab returns word_target in items.
  - collate_fn propagates and pads word_targets.
  - Trainer._forward_batch feeds batch.word_targets (not char targets) into
    WordAuxCTCHead and the combined loss stays finite+positive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from gp1.data.collate import collate_fn
from gp1.data.dataset import SpokenNumbersDataset
from gp1.text.vocab import CharVocab
from gp1.text.vocab_word import WordVocab
from gp1.types import Batch, ManifestRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(path: Path, samplerate: int = 16000, duration_s: float = 0.5) -> None:
    n_samples = int(samplerate * duration_s)
    sf.write(str(path), np.zeros(n_samples, dtype=np.float32), samplerate)


def _make_record(path: Path, transcription: str = "139473") -> ManifestRecord:
    return ManifestRecord(
        audio_path=path,
        transcription=transcription,
        spk_id="spk_A",
        gender="male",
        ext="wav",
        samplerate=16000,
    )


@pytest.fixture()
def records(tmp_path: Path) -> list[ManifestRecord]:
    out: list[ManifestRecord] = []
    for i, trans in enumerate(["100000", "259341", "700005"]):
        wav_path = tmp_path / f"{i:03d}.wav"
        _write_wav(wav_path)
        out.append(_make_record(wav_path, transcription=trans))
    return out


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------


class TestBatchWordFields:
    def test_batch_accepts_word_targets_kwargs(self) -> None:
        # Arrange
        word_targets = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.int64)
        word_target_lengths = torch.tensor([3, 2], dtype=torch.int64)

        # Act
        batch = Batch(
            audio=torch.zeros(2, 160),
            audio_lengths=torch.tensor([160, 80]),
            targets=torch.zeros(2, 3, dtype=torch.int64),
            target_lengths=torch.tensor([3, 3]),
            spk_ids=["a", "b"],
            transcriptions=["1", "2"],
            word_targets=word_targets,
            word_target_lengths=word_target_lengths,
        )

        # Assert
        assert torch.equal(batch.word_targets, word_targets)
        assert torch.equal(batch.word_target_lengths, word_target_lengths)

    def test_batch_defaults_word_fields_to_none(self) -> None:
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
        assert batch.word_targets is None
        assert batch.word_target_lengths is None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestDatasetWordTarget:
    def test_dataset_without_word_vocab_has_no_word_target_key(
        self, records: list[ManifestRecord]
    ) -> None:
        # Arrange
        ds = SpokenNumbersDataset(records, CharVocab())

        # Act
        item = ds[0]

        # Assert
        assert "word_target" not in item

    def test_dataset_with_word_vocab_emits_word_target_tensor(
        self, records: list[ManifestRecord]
    ) -> None:
        # Arrange
        ds = SpokenNumbersDataset(records, CharVocab(), word_vocab=WordVocab())

        # Act
        item = ds[0]

        # Assert
        assert "word_target" in item
        assert item["word_target"].dtype == torch.int64
        assert item["word_target"].dim() == 1
        assert item["word_target"].numel() > 0
        # 100000 -> "сто тысяч" -> 2 word ids
        assert item["word_target"].numel() == 2

    def test_word_target_encodes_expected_ids(
        self, records: list[ManifestRecord]
    ) -> None:
        # Arrange
        ds = SpokenNumbersDataset(records, CharVocab(), word_vocab=WordVocab())
        word_vocab = WordVocab()

        # Act
        item = ds[0]  # "100000" -> "сто тысяч"

        # Assert
        expected = word_vocab.encode("сто тысяч")
        assert item["word_target"].tolist() == expected


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _item(
    audio_len: int,
    target_len: int,
    idx: int,
    word_target: torch.Tensor | None = None,
) -> dict:
    d: dict = {
        "audio": torch.zeros(audio_len, dtype=torch.float32),
        "target": torch.arange(target_len, dtype=torch.int64),
        "spk_id": f"spk_{idx}",
        "transcription": str(10000 + idx),
    }
    if word_target is not None:
        d["word_target"] = word_target
    return d


class TestCollateWordTargets:
    def test_collate_without_word_target_leaves_word_fields_none(self) -> None:
        # Arrange
        items = [_item(160, 5, 0), _item(320, 6, 1)]

        # Act
        batch = collate_fn(items)

        # Assert
        assert batch.word_targets is None
        assert batch.word_target_lengths is None

    def test_collate_pads_word_targets_to_longest(self) -> None:
        # Arrange
        items = [
            _item(160, 5, 0, word_target=torch.tensor([1, 2], dtype=torch.int64)),
            _item(320, 6, 1, word_target=torch.tensor([3, 4, 5], dtype=torch.int64)),
        ]

        # Act
        batch = collate_fn(items)

        # Assert
        assert batch.word_targets is not None
        assert batch.word_target_lengths is not None
        assert batch.word_targets.shape == (2, 3)
        assert batch.word_targets.dtype == torch.int64
        assert batch.word_targets[0].tolist() == [1, 2, 0]
        assert batch.word_targets[1].tolist() == [3, 4, 5]
        assert batch.word_target_lengths.tolist() == [2, 3]


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


def _build_mini_trainer_batch(
    word_vocab: WordVocab,
) -> Batch:
    """Produce a Batch with both char targets and word targets for a trainer pass."""
    audio = torch.zeros(2, 16000, dtype=torch.float32)
    audio_lengths = torch.tensor([16000, 16000], dtype=torch.int64)
    # Char targets length 3 each (not semantically meaningful for shape test).
    char_targets = torch.randint(1, 30, (2, 3), dtype=torch.int64)
    char_target_lengths = torch.tensor([3, 3], dtype=torch.int64)
    # Word targets: 2 words each from the vocab.
    word_target_lengths = torch.tensor([2, 2], dtype=torch.int64)
    # Use known word ids (stay within vocab).
    word_targets = torch.tensor(
        [word_vocab.encode("сто тысяч"), word_vocab.encode("сто тысяч")],
        dtype=torch.int64,
    )
    return Batch(
        audio=audio,
        audio_lengths=audio_lengths,
        targets=char_targets,
        target_lengths=char_target_lengths,
        spk_ids=["a", "b"],
        transcriptions=["100000", "100000"],
        word_targets=word_targets,
        word_target_lengths=word_target_lengths,
    )


class TestTrainerWordAuxPath:
    def test_forward_batch_uses_word_targets_when_word_aux_present(self) -> None:
        # Arrange
        from gp1.losses.ctc import CTCLoss
        from gp1.losses.word_aux import WordAuxCTCHead
        from gp1.models.quartznet import QuartzNet10x4
        from gp1.train.trainer import Trainer, TrainerConfig

        vocab = CharVocab()
        word_vocab = WordVocab()
        model = QuartzNet10x4(vocab_size=vocab.vocab_size, d_model=64)
        ctc_loss = CTCLoss(blank_id=vocab.blank_id)
        word_aux = WordAuxCTCHead(d_enc=64, word_vocab_size=word_vocab.size, blank_id=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
        cfg = TrainerConfig(max_epochs=1, fp16_autocast=False, ckpt_dir=Path("/tmp"))

        trainer = Trainer(
            model=model,
            ctc_loss=ctc_loss,
            inter_ctc=None,
            cr_ctc=None,
            word_aux=word_aux,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=[],
            val_loader=[],
            vocab=vocab,
            config=cfg,
            device=torch.device("cpu"),
        )
        batch = _build_mini_trainer_batch(word_vocab)

        # Act
        loss = trainer._forward_batch(batch)

        # Assert
        assert torch.isfinite(loss).item()
        assert loss.item() > 0.0

    def test_forward_batch_raises_when_word_aux_set_but_word_targets_missing(
        self,
    ) -> None:
        # Arrange
        from gp1.losses.ctc import CTCLoss
        from gp1.losses.word_aux import WordAuxCTCHead
        from gp1.models.quartznet import QuartzNet10x4
        from gp1.train.trainer import Trainer, TrainerConfig

        vocab = CharVocab()
        word_vocab = WordVocab()
        model = QuartzNet10x4(vocab_size=vocab.vocab_size, d_model=64)
        ctc_loss = CTCLoss(blank_id=vocab.blank_id)
        word_aux = WordAuxCTCHead(d_enc=64, word_vocab_size=word_vocab.size, blank_id=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
        cfg = TrainerConfig(max_epochs=1, fp16_autocast=False, ckpt_dir=Path("/tmp"))
        trainer = Trainer(
            model=model,
            ctc_loss=ctc_loss,
            inter_ctc=None,
            cr_ctc=None,
            word_aux=word_aux,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=[],
            val_loader=[],
            vocab=vocab,
            config=cfg,
            device=torch.device("cpu"),
        )
        batch = Batch(
            audio=torch.zeros(2, 16000),
            audio_lengths=torch.tensor([16000, 16000]),
            targets=torch.randint(1, 30, (2, 3), dtype=torch.int64),
            target_lengths=torch.tensor([3, 3]),
            spk_ids=["a", "b"],
            transcriptions=["100000", "100000"],
            word_targets=None,
            word_target_lengths=None,
        )

        # Act / Assert
        with pytest.raises(ValueError, match="word_aux"):
            trainer._forward_batch(batch)
