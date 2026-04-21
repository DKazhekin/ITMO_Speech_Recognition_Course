"""Tests for gp1.data.dataset (CONTRACTS.md §4).

TDD RED->GREEN->REFACTOR. Tests written before implementation.
AAA (Arrange-Act-Assert) pattern throughout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from gp1.data.audio_aug import AudioAugmenter
from gp1.data.dataset import SpokenNumbersDataset
from gp1.text.vocab import CharVocab
from gp1.types import AugConfig, ManifestRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_wav(path: Path, samplerate: int, duration_s: float = 0.5) -> None:
    """Write a silent WAV file at given sample rate."""
    n_samples = int(samplerate * duration_s)
    data = np.zeros(n_samples, dtype=np.float32)
    sf.write(str(path), data, samplerate)


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
        _write_wav(wav_path, samplerate=16000, duration_s=0.5)
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
        _write_wav(wav_path, samplerate=native_sr, duration_s=duration_s)
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
        _write_wav(wav_path, samplerate=16000)
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
        _write_wav(wav_path, samplerate=16000, duration_s=0.25)
        record = _make_record(wav_path)
        dataset = SpokenNumbersDataset([record], vocab, augmenter=None)

        # Act
        item = dataset[0]

        # Assert — silent WAV should produce all-zero tensor
        assert item["audio"].abs().max().item() == pytest.approx(0.0, abs=1e-6)


class TestSpokenNumbersDatasetTwoViews:
    def test_returns_audio_view2_when_enabled(
        self, tmp_path: Path, vocab: CharVocab
    ) -> None:
        """return_two_views=True adds 'audio_view2' key to the returned dict."""
        # Arrange
        wav_path = tmp_path / "a.wav"
        _write_wav(wav_path, samplerate=16000, duration_s=0.5)
        record = _make_record(wav_path)
        augmenter = AudioAugmenter(AugConfig(seed=42))
        dataset = SpokenNumbersDataset(
            [record], vocab, augmenter=augmenter, return_two_views=True
        )

        # Act
        item = dataset[0]

        # Assert
        assert "audio_view2" in item
        assert item["audio_view2"].dtype == torch.float32
        assert item["audio_view2"].ndim == 1

    def test_two_views_differ_with_stochastic_aug(
        self, tmp_path: Path, vocab: CharVocab
    ) -> None:
        """With stochastic augmentation the two views should differ.

        We use gain augmentation (prob=1.0) which perturbs amplitude,
        and check that both views are float tensors of similar length
        (speed perturb may change length, both should be non-trivially different).
        """
        # Arrange — use a non-silent WAV so gain has effect
        n_samples = 8000
        wav_path = tmp_path / "signal.wav"
        rng = np.random.default_rng(0)
        data = rng.uniform(-0.1, 0.1, n_samples).astype(np.float32)
        sf.write(str(wav_path), data, 16000)

        record = _make_record(wav_path)
        # High gain and speed probs guarantee at least one of the augmentations fires
        cfg = AugConfig(
            speed_prob=0.0,
            pitch_prob=0.0,
            vtlp_prob=0.0,
            gain_prob=1.0,
            gain_db_range=(-8.0, 8.0),
            noise_prob=0.0,
            rir_prob=0.0,
            seed=None,  # non-deterministic so views differ
        )
        augmenter = AudioAugmenter(cfg)
        dataset = SpokenNumbersDataset(
            [record], vocab, augmenter=augmenter, return_two_views=True
        )

        # Act — repeated calls to get() to see variety
        differences_found = False
        for _ in range(20):
            item = dataset[0]
            v1 = item["audio"]
            v2 = item["audio_view2"]
            if not torch.allclose(v1, v2, atol=1e-5):
                differences_found = True
                break

        # Assert
        assert differences_found, (
            "Expected view1 and view2 to differ with stochastic aug after 20 tries"
        )

    def test_no_view2_key_when_disabled(
        self, simple_records: list[ManifestRecord], vocab: CharVocab
    ) -> None:
        """return_two_views=False (default) -> 'audio_view2' not in returned dict."""
        # Arrange
        dataset = SpokenNumbersDataset(simple_records, vocab, return_two_views=False)

        # Act
        item = dataset[0]

        # Assert
        assert "audio_view2" not in item
