"""Tests for gp1.submit.inference_utils.

TDD RED->GREEN->REFACTOR. Tests written before implementation.
AAA (Arrange-Act-Assert) pattern throughout.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from helpers import write_wav
from gp1.text.vocab import CharVocab
from gp1.types import ManifestRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_records(tmp_path: Path, n: int = 5) -> list[ManifestRecord]:
    """Create *n* silent 16 kHz WAV records in *tmp_path*."""
    records = []
    for i in range(n):
        wav_path = tmp_path / f"sample_{i:03d}.wav"
        write_wav(wav_path, samplerate=16000, duration_s=0.5)
        records.append(
            ManifestRecord(
                audio_path=wav_path,
                transcription=str(10000 + i),
                spk_id=f"spk_{i}",
                gender="male",
                ext="wav",
                samplerate=16000,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Test 1 — build_test_dataloader preserves record order
# ---------------------------------------------------------------------------


def test_build_test_dataloader_preserves_order(tmp_path: Path) -> None:
    """DataLoader iterates records in the same order as the input list."""
    from gp1.submit.inference_utils import build_test_dataloader

    # Arrange
    vocab = CharVocab()
    records = _make_records(tmp_path, n=5)

    # Act
    loader = build_test_dataloader(records, vocab, batch_size=2, num_workers=0)
    collected_paths: list[Path] = []
    for batch in loader:
        # batch is a Batch dataclass; collect audio_lengths as a proxy for order
        # We need transcriptions to match — stored in Batch.transcriptions
        collected_paths.extend(batch.transcriptions)

    # Assert — transcriptions must come out in input order
    expected = [r.transcription for r in records]
    assert collected_paths == expected, (
        f"Order mismatch: expected {expected}, got {collected_paths}"
    )


# ---------------------------------------------------------------------------
# Test 2 — no augmentation: identical waveforms on two calls
# ---------------------------------------------------------------------------


def test_build_test_dataloader_no_aug(tmp_path: Path) -> None:
    """Two passes over the loader return identical waveforms (no randomness)."""
    from gp1.submit.inference_utils import build_test_dataloader

    # Arrange
    vocab = CharVocab()
    records = _make_records(tmp_path, n=3)
    loader = build_test_dataloader(records, vocab, batch_size=3, num_workers=0)

    # Act — collect first batch twice
    batch1 = next(iter(loader))
    batch2 = next(iter(loader))

    # Assert
    assert torch.equal(batch1.audio, batch2.audio), (
        "Waveforms differ between two passes — augmentation must be disabled"
    )


# ---------------------------------------------------------------------------
# Test 3 — write_submission writes header and rows
# ---------------------------------------------------------------------------


def test_write_submission_writes_header_and_rows(tmp_path: Path) -> None:
    """CSV contains the header row followed by data rows in input order."""
    from gp1.submit.inference_utils import write_submission

    # Arrange
    out = tmp_path / "submission.csv"
    pairs = [("file_a.wav", "12345"), ("file_b.wav", "99999")]

    # Act
    write_submission(pairs, out)

    # Assert
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    assert rows[0] == ["filename", "transcription"]
    assert rows[1] == ["file_a.wav", "12345"]
    assert rows[2] == ["file_b.wav", "99999"]
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# Test 4 — write_submission creates missing parent directories
# ---------------------------------------------------------------------------


def test_write_submission_creates_parent_dir(tmp_path: Path) -> None:
    """Parent directories are created if they do not exist."""
    from gp1.submit.inference_utils import write_submission

    # Arrange — nested path that does not exist yet
    out = tmp_path / "level1" / "level2" / "sub.csv"

    # Act
    write_submission([("a.wav", "1")], out)

    # Assert
    assert out.exists(), "Output file was not created"
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[1] == ["a.wav", "1"]


# ---------------------------------------------------------------------------
# Test 5 — write_submission overwrites existing file (no append)
# ---------------------------------------------------------------------------


def test_write_submission_overwrites_existing(tmp_path: Path) -> None:
    """Calling write_submission twice produces only the second write's rows."""
    from gp1.submit.inference_utils import write_submission

    # Arrange
    out = tmp_path / "dup.csv"
    write_submission([("old.wav", "00000")], out)

    # Act — overwrite with different data
    write_submission([("new.wav", "11111")], out)

    # Assert — only the second write's content is present
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    assert len(rows) == 2, f"Expected 2 rows (header + 1), got {len(rows)}: {rows}"
    assert rows[1][0] == "new.wav", "Old row was not overwritten"
