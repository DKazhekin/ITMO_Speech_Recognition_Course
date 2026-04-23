"""Tests for gp1.data.manifest.

Follows TDD RED->GREEN->REFACTOR. AAA (Arrange-Act-Assert) pattern throughout.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import soundfile as sf

from helpers import write_wav
from gp1.data.manifest import (
    leave_n_speakers_out_split,
    records_from_csv,
)
from gp1.types import ManifestRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(
    csv_path: Path,
    audio_root: Path,
    rows: list[dict],
    *,
    samplerate: int = 16000,
) -> None:
    """Write a Kaggle-style CSV and create corresponding WAV files."""
    fieldnames = ["filename", "transcription", "spk_id", "gender"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
            wav_path = audio_root / row["filename"]
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sr = row.get("_sr", samplerate)
            write_wav(wav_path, samplerate=sr)


def _make_record(
    audio_path: Path,
    transcription: str = "12345",
    spk_id: str = "spk_A",
    gender: str = "male",
    ext: str = "wav",
    samplerate: int = 16000,
) -> ManifestRecord:
    return ManifestRecord(
        audio_path=audio_path,
        transcription=transcription,
        spk_id=spk_id,
        gender=gender,
        ext=ext,
        samplerate=samplerate,
    )


# ---------------------------------------------------------------------------
# records_from_csv tests
# ---------------------------------------------------------------------------


class TestRecordsFromCsv:
    def test_record_count_matches_csv_rows(self, tmp_path: Path) -> None:
        """Number of returned records equals the number of CSV data rows."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        csv_path = tmp_path / "train.csv"
        rows = [
            {
                "filename": "spk_A/001.wav",
                "transcription": "10000",
                "spk_id": "spk_A",
                "gender": "male",
            },
            {
                "filename": "spk_B/002.wav",
                "transcription": "20000",
                "spk_id": "spk_B",
                "gender": "female",
            },
            {
                "filename": "spk_C/003.wav",
                "transcription": "30000",
                "spk_id": "spk_C",
                "gender": "male",
            },
        ]
        _make_csv(csv_path, audio_root, rows)

        # Act
        records = records_from_csv(csv_path, audio_root)

        # Assert
        assert len(records) == 3

    def test_audio_path_is_absolute(self, tmp_path: Path) -> None:
        """audio_path in each ManifestRecord is absolute."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        csv_path = tmp_path / "train.csv"
        rows = [
            {
                "filename": "spk_A/001.wav",
                "transcription": "15000",
                "spk_id": "spk_A",
                "gender": "male",
            },
        ]
        _make_csv(csv_path, audio_root, rows)

        # Act
        records = records_from_csv(csv_path, audio_root)

        # Assert
        assert records[0].audio_path.is_absolute(), (
            f"Expected absolute path, got {records[0].audio_path}"
        )

    def test_ext_is_lowercase_without_dot(self, tmp_path: Path) -> None:
        """ext field is lowercase and has no leading dot."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        csv_path = tmp_path / "train.csv"
        rows = [
            {
                "filename": "spk_A/001.wav",
                "transcription": "10000",
                "spk_id": "spk_A",
                "gender": "male",
            },
        ]
        _make_csv(csv_path, audio_root, rows)

        # Act
        records = records_from_csv(csv_path, audio_root)

        # Assert
        assert records[0].ext == "wav"
        assert not records[0].ext.startswith(".")

    def test_samplerate_matches_native_wav(self, tmp_path: Path) -> None:
        """samplerate field matches the native sample rate read via sf.info."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        wav_22k = audio_root / "22k.wav"
        wav_8k = audio_root / "8k.wav"
        write_wav(wav_22k, samplerate=22050)
        write_wav(wav_8k, samplerate=8000)

        csv_path = tmp_path / "train.csv"
        fieldnames = ["filename", "transcription", "spk_id", "gender"]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "22k.wav",
                    "transcription": "11000",
                    "spk_id": "spk_A",
                    "gender": "male",
                }
            )
            writer.writerow(
                {
                    "filename": "8k.wav",
                    "transcription": "22000",
                    "spk_id": "spk_B",
                    "gender": "female",
                }
            )

        # Act
        records = records_from_csv(csv_path, audio_root)

        # Assert
        by_name = {r.audio_path.name: r for r in records}
        assert by_name["22k.wav"].samplerate == 22050
        assert by_name["8k.wav"].samplerate == 8000

    def test_duration_s_matches_sf_info(self, tmp_path: Path) -> None:
        """duration_s equals frames / samplerate as reported by sf.info."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        wav_path = audio_root / "clip.wav"
        target_duration = 0.25
        write_wav(wav_path, samplerate=16000, duration_s=target_duration)

        csv_path = tmp_path / "train.csv"
        fieldnames = ["filename", "transcription", "spk_id", "gender"]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "clip.wav",
                    "transcription": "10000",
                    "spk_id": "spk_A",
                    "gender": "male",
                }
            )

        # Act
        records = records_from_csv(csv_path, audio_root)

        # Assert
        info = sf.info(str(wav_path))
        expected_duration = info.frames / info.samplerate
        assert records[0].duration_s == pytest.approx(expected_duration, rel=1e-5)

    def test_metadata_fields_populated_correctly(self, tmp_path: Path) -> None:
        """transcription, spk_id, gender are taken verbatim from the CSV row."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        csv_path = tmp_path / "train.csv"
        rows = [
            {
                "filename": "spk_F/001.wav",
                "transcription": "55555",
                "spk_id": "spk_F",
                "gender": "female",
            },
        ]
        _make_csv(csv_path, audio_root, rows)

        # Act
        records = records_from_csv(csv_path, audio_root)

        # Assert
        r = records[0]
        assert r.transcription == "55555"
        assert r.spk_id == "spk_F"
        assert r.gender == "female"

    def test_missing_audio_file_raises(self, tmp_path: Path) -> None:
        """records_from_csv raises an error when an audio file does not exist."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        csv_path = tmp_path / "train.csv"
        fieldnames = ["filename", "transcription", "spk_id", "gender"]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "missing.wav",
                    "transcription": "10000",
                    "spk_id": "spk_A",
                    "gender": "male",
                }
            )

        # Act / Assert
        with pytest.raises(Exception):
            records_from_csv(csv_path, audio_root)


# ---------------------------------------------------------------------------
# leave_n_speakers_out_split tests
# ---------------------------------------------------------------------------


class TestLeaveNSpeakersOutSplit:
    def _make_records(self, tmp_path: Path) -> list[ManifestRecord]:
        return [
            _make_record(tmp_path / "a.wav", spk_id="spk_A", transcription="10000"),
            _make_record(tmp_path / "b.wav", spk_id="spk_B", transcription="20000"),
            _make_record(tmp_path / "c.wav", spk_id="spk_C", transcription="30000"),
            _make_record(tmp_path / "d.wav", spk_id="spk_E", transcription="40000"),
            _make_record(tmp_path / "e.wav", spk_id="spk_F", transcription="50000"),
            _make_record(tmp_path / "f.wav", spk_id="spk_A", transcription="60000"),
        ]

    def test_partitions_by_spk_id(self, tmp_path: Path) -> None:
        """Records with holdout spk_ids go to dev; others to train."""
        # Arrange
        records = self._make_records(tmp_path)
        holdout = ["spk_E", "spk_F"]

        # Act
        train, dev = leave_n_speakers_out_split(records, holdout)

        # Assert
        train_spk_ids = {r.spk_id for r in train}
        dev_spk_ids = {r.spk_id for r in dev}
        assert "spk_E" not in train_spk_ids
        assert "spk_F" not in train_spk_ids
        assert dev_spk_ids == {"spk_E", "spk_F"}
        assert len(train) + len(dev) == len(records)

    def test_preserves_order_within_splits(self, tmp_path: Path) -> None:
        """train and dev splits preserve relative input order."""
        # Arrange
        records = self._make_records(tmp_path)
        holdout = ["spk_E", "spk_F"]

        # Act
        train, dev = leave_n_speakers_out_split(records, holdout)

        # Assert — verify monotonically increasing original indices
        original_indices = {id(r): i for i, r in enumerate(records)}

        train_indices = [original_indices[id(r)] for r in train]
        assert train_indices == sorted(train_indices), (
            "Train split is not order-preserving"
        )

        dev_indices = [original_indices[id(r)] for r in dev]
        assert dev_indices == sorted(dev_indices), "Dev split is not order-preserving"

    def test_empty_holdout_yields_all_train(self, tmp_path: Path) -> None:
        """Empty holdout_speakers -> train = all records, dev = empty."""
        # Arrange
        records = self._make_records(tmp_path)

        # Act
        train, dev = leave_n_speakers_out_split(records, [])

        # Assert
        assert len(train) == len(records)
        assert len(dev) == 0

    def test_all_holdout_yields_all_dev(self, tmp_path: Path) -> None:
        """Holding out all speaker ids -> train = empty, dev = all records."""
        # Arrange
        records = self._make_records(tmp_path)
        all_spk_ids = list({r.spk_id for r in records})

        # Act
        train, dev = leave_n_speakers_out_split(records, all_spk_ids)

        # Assert
        assert len(train) == 0
        assert len(dev) == len(records)

    def test_returns_new_lists_not_mutated_input(self, tmp_path: Path) -> None:
        """Returned splits are new lists; the input list is not mutated."""
        # Arrange
        records = self._make_records(tmp_path)
        original_len = len(records)
        original_ids = [id(r) for r in records]

        # Act
        train, dev = leave_n_speakers_out_split(records, ["spk_E"])

        # Assert
        assert len(records) == original_len, "Input list was mutated"
        assert [id(r) for r in records] == original_ids, "Input records were reordered"
        assert train is not records
        assert dev is not records
