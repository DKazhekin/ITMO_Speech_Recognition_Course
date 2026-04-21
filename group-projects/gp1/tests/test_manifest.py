"""Tests for gp1.data.manifest (CONTRACTS.md §4).

Follows TDD RED->GREEN->REFACTOR. All tests written before implementation.
AAA (Arrange-Act-Assert) pattern throughout.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from gp1.data.manifest import (
    build_manifest,
    leave_n_speakers_out_split,
    read_jsonl,
    write_jsonl,
)
from gp1.types import ManifestRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_wav(path: Path, samplerate: int, duration_s: float = 0.5) -> None:
    """Write a silent WAV file at the given sample rate."""
    n_samples = int(samplerate * duration_s)
    data = np.zeros(n_samples, dtype=np.float32)
    sf.write(str(path), data, samplerate)


def _make_csv(
    csv_path: Path,
    audio_root: Path,
    rows: list[dict],
) -> None:
    """Write a Kaggle-style CSV and create corresponding WAV files."""
    fieldnames = ["filename", "transcription", "spk_id", "gender"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            wav_path = audio_root / row["filename"]
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sr = row.get("_sr", 16000)
            _write_wav(wav_path, samplerate=sr)


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
# build_manifest tests
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_writes_correct_record_count(self, tmp_path: Path) -> None:
        """build_manifest returns count == number of CSV rows and JSONL has same count."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        csv_path = tmp_path / "train.csv"
        out_path = tmp_path / "manifest.jsonl"
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
        count = build_manifest(csv_path, audio_root, out_path)

        # Assert
        assert count == 3
        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_resolves_absolute_paths(self, tmp_path: Path) -> None:
        """audio_path in each JSONL record is absolute."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        csv_path = tmp_path / "train.csv"
        out_path = tmp_path / "manifest.jsonl"
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
        build_manifest(csv_path, audio_root, out_path)

        # Assert
        record_dict = json.loads(out_path.read_text().strip().splitlines()[0])
        audio_path = Path(record_dict["audio_path"])
        assert audio_path.is_absolute(), f"Expected absolute path, got {audio_path}"

    def test_reads_native_samplerate(self, tmp_path: Path) -> None:
        """JSONL samplerate field matches the actual native sample rate of each WAV."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        csv_path = tmp_path / "train.csv"
        out_path = tmp_path / "manifest.jsonl"

        # Create WAVs with different sample rates
        wav_22k = audio_root / "22k.wav"
        wav_8k = audio_root / "8k.wav"
        _write_wav(wav_22k, samplerate=22050)
        _write_wav(wav_8k, samplerate=8000)

        # Write CSV manually (rows reference the exact filenames)
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
        build_manifest(csv_path, audio_root, out_path)

        # Assert
        lines = out_path.read_text().strip().splitlines()
        records_by_filename = {
            Path(json.loads(line)["audio_path"]).name: json.loads(line)
            for line in lines
        }
        assert records_by_filename["22k.wav"]["samplerate"] == 22050
        assert records_by_filename["8k.wav"]["samplerate"] == 8000

    def test_stores_correct_metadata_fields(self, tmp_path: Path) -> None:
        """Each JSONL record has all required ManifestRecord fields."""
        # Arrange
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        csv_path = tmp_path / "train.csv"
        out_path = tmp_path / "manifest.jsonl"
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
        build_manifest(csv_path, audio_root, out_path)

        # Assert
        record_dict = json.loads(out_path.read_text().strip().splitlines()[0])
        assert record_dict["transcription"] == "55555"
        assert record_dict["spk_id"] == "spk_F"
        assert record_dict["gender"] == "female"
        assert record_dict["ext"] == "wav"
        assert "samplerate" in record_dict
        assert "audio_path" in record_dict


# ---------------------------------------------------------------------------
# read_jsonl / write_jsonl round-trip tests
# ---------------------------------------------------------------------------


class TestReadWriteJsonl:
    def test_round_trip_preserves_all_fields(self, tmp_path: Path) -> None:
        """write_jsonl then read_jsonl returns an equal list of ManifestRecord."""
        # Arrange
        records = [
            ManifestRecord(
                audio_path=tmp_path / "a.wav",
                transcription="10001",
                spk_id="spk_A",
                gender="male",
                ext="wav",
                samplerate=16000,
            ),
            ManifestRecord(
                audio_path=tmp_path / "b.wav",
                transcription="999999",
                spk_id="spk_B",
                gender="female",
                ext="wav",
                samplerate=22050,
            ),
        ]
        out_path = tmp_path / "manifest.jsonl"

        # Act
        write_jsonl(records, out_path)
        loaded = read_jsonl(out_path)

        # Assert
        assert len(loaded) == len(records)
        for orig, restored in zip(records, loaded):
            assert restored.audio_path == orig.audio_path
            assert restored.transcription == orig.transcription
            assert restored.spk_id == orig.spk_id
            assert restored.gender == orig.gender
            assert restored.ext == orig.ext
            assert restored.samplerate == orig.samplerate

    def test_write_produces_one_json_object_per_line(self, tmp_path: Path) -> None:
        """Each line in the output file is a valid JSON object."""
        # Arrange
        records = [
            ManifestRecord(
                audio_path=tmp_path / "c.wav",
                transcription="12000",
                spk_id="spk_C",
                gender="male",
                ext="wav",
                samplerate=16000,
            ),
        ]
        out_path = tmp_path / "manifest.jsonl"

        # Act
        write_jsonl(records, out_path)

        # Assert
        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert isinstance(parsed, dict)

    def test_read_restores_path_as_path_object(self, tmp_path: Path) -> None:
        """read_jsonl returns ManifestRecord with audio_path as pathlib.Path."""
        # Arrange
        record = ManifestRecord(
            audio_path=tmp_path / "d.wav",
            transcription="10000",
            spk_id="spk_D",
            gender="male",
            ext="wav",
            samplerate=16000,
        )
        out_path = tmp_path / "manifest.jsonl"
        write_jsonl([record], out_path)

        # Act
        loaded = read_jsonl(out_path)

        # Assert
        assert isinstance(loaded[0].audio_path, Path)


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
