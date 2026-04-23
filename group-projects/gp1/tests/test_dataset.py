"""Tests for gp1.data.dataset (CONTRACTS.md §4).

TDD RED->GREEN->REFACTOR. Tests written before implementation.
AAA (Arrange-Act-Assert) pattern throughout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from helpers import write_wav
from gp1.data.dataset import SpokenNumbersDataset
from gp1.text.vocab import CharVocab
from gp1.types import ManifestRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    path: Path,
    transcription: str = "10000",
    spk_id: str = "spk_A",
    gender: str = "male",
    samplerate: int = 16000,
) -> ManifestRecord:
    return ManifestRecord(
        audio_path=path,
        transcription=transcription,
        spk_id=spk_id,
        gender=gender,
        ext="wav",
        samplerate=samplerate,
    )


@pytest.fixture()
def vocab() -> CharVocab:
    return CharVocab()


@pytest.fixture()
def simple_records(tmp_path: Path) -> list[ManifestRecord]:
    """Three 0.5s 16 kHz WAV records."""
    records = []
    for i, (trans, spk) in enumerate(
        [("10000", "spk_A"), ("20000", "spk_B"), ("30000", "spk_C")]
    ):
        wav_path = tmp_path / f"{i:03d}.wav"
        write_wav(wav_path, samplerate=16000, duration_s=0.5)
        records.append(_make_record(wav_path, transcription=trans, spk_id=spk))
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpokenNumbersDatasetLen:
    def test_len_equals_records_count(
        self, simple_records: list[ManifestRecord], vocab: CharVocab
    ) -> None:
        """__len__ returns the number of records passed to the constructor."""
        # Arrange
        dataset = SpokenNumbersDataset(simple_records, vocab)

        # Act
        length = len(dataset)

        # Assert
        assert length == len(simple_records)

    def test_len_zero_on_empty_records(self, vocab: CharVocab) -> None:
        """Empty records list -> len == 0."""
        # Arrange & Act
        dataset = SpokenNumbersDataset([], vocab)

        # Assert
        assert len(dataset) == 0


class TestSpokenNumbersDatasetGetItem:
    def test_returns_required_keys(
        self, simple_records: list[ManifestRecord], vocab: CharVocab
    ) -> None:
        """__getitem__ returns dict with keys: audio, target, spk_id, transcription."""
        # Arrange
        dataset = SpokenNumbersDataset(simple_records, vocab)

        # Act
        item = dataset[0]

        # Assert
        assert "audio" in item
        assert "target" in item
        assert "spk_id" in item
        assert "transcription" in item

    def test_audio_is_float32_1d(
        self, simple_records: list[ManifestRecord], vocab: CharVocab
    ) -> None:
        """audio tensor is 1-D float32."""
        # Arrange
        dataset = SpokenNumbersDataset(simple_records, vocab)

        # Act
        item = dataset[0]

        # Assert
        assert item["audio"].dtype == torch.float32
        assert item["audio"].ndim == 1

    def test_audio_resampled_to_target_samplerate(
        self, tmp_path: Path, vocab: CharVocab
    ) -> None:
        """22050 Hz native WAV gets resampled; length reflects 16 kHz."""
        # Arrange
        wav_path = tmp_path / "22k.wav"
        native_sr = 22050
        target_sr = 16000
        duration_s = 0.5
        write_wav(wav_path, samplerate=native_sr, duration_s=duration_s)
        record = _make_record(wav_path, samplerate=native_sr)
        dataset = SpokenNumbersDataset([record], vocab, target_samplerate=target_sr)

        # Act
        item = dataset[0]

        # Assert — within 5% of expected length
        expected_len = int(duration_s * target_sr)
        actual_len = item["audio"].shape[0]
        assert abs(actual_len - expected_len) < expected_len * 0.05, (
            f"Expected ~{expected_len} samples, got {actual_len}"
        )

    def test_target_is_int64(
        self, simple_records: list[ManifestRecord], vocab: CharVocab
    ) -> None:
        """target tensor dtype is int64."""
        # Arrange
        dataset = SpokenNumbersDataset(simple_records, vocab)

        # Act
        item = dataset[0]

        # Assert
        assert item["target"].dtype == torch.int64

    def test_target_is_encoded_via_vocab(
        self, tmp_path: Path, vocab: CharVocab
    ) -> None:
        """Transcription '10000' is encoded as vocab.encode(digits_to_words('10000'))."""
        # Arrange
        from gp1.text.normalize import digits_to_words

        wav_path = tmp_path / "a.wav"
        write_wav(wav_path, samplerate=16000)
        record = _make_record(wav_path, transcription="10000")
        dataset = SpokenNumbersDataset([record], vocab)

        expected_text = digits_to_words("10000")
        expected_ids = torch.tensor(vocab.encode(expected_text), dtype=torch.int64)

        # Act
        item = dataset[0]

        # Assert
        assert torch.equal(item["target"], expected_ids), (
            f"Expected {expected_ids.tolist()}, got {item['target'].tolist()}"
        )

    def test_spk_id_and_transcription_pass_through(
        self, simple_records: list[ManifestRecord], vocab: CharVocab
    ) -> None:
        """spk_id and transcription match the ManifestRecord fields."""
        # Arrange
        dataset = SpokenNumbersDataset(simple_records, vocab)

        # Act & Assert
        for i, record in enumerate(simple_records):
            item = dataset[i]
            assert item["spk_id"] == record.spk_id
            assert item["transcription"] == record.transcription

    def test_no_augmenter_returns_clean_audio(
        self, tmp_path: Path, vocab: CharVocab
    ) -> None:
        """Without augmenter, audio is a zero tensor (silent WAV)."""
        # Arrange
        wav_path = tmp_path / "silent.wav"
        write_wav(wav_path, samplerate=16000, duration_s=0.25)
        record = _make_record(wav_path)
        dataset = SpokenNumbersDataset([record], vocab, augmenter=None)

        # Act
        item = dataset[0]

        # Assert — silent WAV should produce all-zero tensor
        assert item["audio"].abs().max().item() == pytest.approx(0.0, abs=1e-6)
