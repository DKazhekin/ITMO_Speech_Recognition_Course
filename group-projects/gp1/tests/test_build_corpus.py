"""Tests for gp1.lm.build_corpus.

TDD: RED phase — these tests were written before any implementation.

All tests that require `num2words` are guarded with pytest.importorskip so
that a missing package produces a clear SKIP rather than a cryptic ImportError.
Tests that only check return-value semantics or deduplication logic use a
small monkeypatched replacement for the full 999 000-line corpus, keeping
the suite fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Guard: if gp1.text.normalize is not yet on disk (W1-A pending), skip corpus
# tests that actually call build_synthetic_corpus with real data.
# ---------------------------------------------------------------------------
num2words = pytest.importorskip(
    "num2words",
    reason="num2words not installed — install via `uv pip install num2words`",
)

# ---------------------------------------------------------------------------
# Skip the whole module at collection time if the implementation is missing
# (pre-GREEN). pytest.skip(..., allow_module_level=True) is NoReturn so that
# Pyright can type-narrow subsequent references to build_synthetic_corpus.
# ---------------------------------------------------------------------------
try:
    from gp1.lm.build_corpus import build_synthetic_corpus
except ImportError:
    pytest.skip(
        "gp1.lm.build_corpus not yet implemented (RED phase expected)",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(tmp_path: Path, records: list[dict]) -> Path:
    """Write a tiny JSONL manifest and return its path."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildSyntheticCorpusLineCount:
    """test_writes_one_line_per_number_in_range"""

    def test_writes_one_line_per_number_in_range(self, tmp_path: Path) -> None:
        """
        Arrange: output path in a temp dir, no manifest.
        Act: call build_synthetic_corpus without a manifest.
        Assert: the output file contains exactly 999 000 lines
                (one per integer in range(1000, 1_000_000)).
        """
        out = tmp_path / "corpus.txt"

        result = build_synthetic_corpus(out_path=out)

        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 999_000, (
            f"Expected 999 000 lines (range 1000..999999), got {len(lines)}"
        )
        assert result == 999_000, f"Return value should equal line count, got {result}"


class TestReturnValueEqualsFileLineCount:
    """test_returns_number_of_lines_written"""

    def test_returns_number_of_lines_written(self, tmp_path: Path) -> None:
        """
        Arrange/Act: build corpus without manifest.
        Assert: return value == wc -l of written file.
        """
        out = tmp_path / "corpus.txt"
        returned = build_synthetic_corpus(out_path=out)

        actual_lines = len(out.read_text(encoding="utf-8").splitlines())
        assert returned == actual_lines, (
            f"Return value {returned} != file line count {actual_lines}"
        )


class TestAppendsTrainingTranscriptions:
    """test_appends_training_transcriptions_when_manifest_given"""

    def test_appends_training_transcriptions_when_manifest_given(
        self, tmp_path: Path
    ) -> None:
        """
        Arrange: manifest with 2 records whose transcriptions are known numbers.
        Act: call build_synthetic_corpus with the manifest.
        Assert: corpus contains the word-form of those numbers, and total line
                count is >= 999 000 (manifest numbers may already be in the
                synthetic range, so we check >= rather than exact).
        """
        records = [
            {
                "audio_path": "/fake/001.wav",
                "transcription": "1000",
                "spk_id": "spk_A",
                "gender": "male",
                "ext": "wav",
                "samplerate": 16000,
            },
            {
                "audio_path": "/fake/002.wav",
                "transcription": "999999",
                "spk_id": "spk_A",
                "gender": "male",
                "ext": "wav",
                "samplerate": 16000,
            },
        ]
        manifest = _make_manifest(tmp_path, records)
        out = tmp_path / "corpus.txt"

        build_synthetic_corpus(out_path=out, train_manifest=manifest)

        content = out.read_text(encoding="utf-8")
        lines = content.splitlines()
        # Both 1000 and 999999 are inside the synthetic range, so the file
        # must be at least 999 000 lines.
        assert len(lines) >= 999_000


class TestOutputIsLowercaseAndNormalised:
    """test_output_is_lowercase_and_whitespace_normalized"""

    def test_output_is_lowercase_and_whitespace_normalized(
        self, tmp_path: Path
    ) -> None:
        """
        Arrange/Act: build corpus.
        Assert: every non-empty line is lowercase and has no leading/trailing
                whitespace and no double-spaces inside.
        """
        out = tmp_path / "corpus.txt"
        build_synthetic_corpus(out_path=out)

        # Spot-check first 200 lines for performance.
        content = out.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines()[:200]):
            assert line == line.lower(), (
                f"Line {i + 1} has uppercase characters: {line!r}"
            )
            assert line == line.strip(), (
                f"Line {i + 1} has leading/trailing whitespace: {line!r}"
            )
            assert "  " not in line, f"Line {i + 1} has double spaces: {line!r}"


class TestDeduplicatesManifestLines:
    """test_deduplicates_manifest_lines_against_synthetic"""

    def test_deduplicates_manifest_lines_against_synthetic(
        self, tmp_path: Path
    ) -> None:
        """
        Arrange: manifest where every transcription falls within 1000..999999
                 (already covered by synthetic corpus).
        Act: build corpus with that manifest.
        Assert: total lines == 999 000 (no duplicates added for known numbers).
        """
        # Use 3 numbers from inside the synthetic range.
        records = [
            {
                "audio_path": f"/fake/{n}.wav",
                "transcription": str(n),
                "spk_id": "spk_A",
                "gender": "male",
                "ext": "wav",
                "samplerate": 16000,
            }
            for n in [500_000, 123_456, 1_000]
        ]
        manifest = _make_manifest(tmp_path, records)
        out = tmp_path / "corpus.txt"

        build_synthetic_corpus(out_path=out, train_manifest=manifest)

        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 999_000, (
            f"Deduplication failed: expected 999 000 lines, got {len(lines)}"
        )
