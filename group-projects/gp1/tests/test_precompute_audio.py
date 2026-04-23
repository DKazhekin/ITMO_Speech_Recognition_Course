"""Tests for scripts/precompute_audio.py (Phase B audio cache).

TDD RED->GREEN->REFACTOR.  Tests written before implementation.
AAA (Arrange-Act-Assert) pattern throughout.

References:
  - soundfile.write PCM_16 subtype:
      https://python-soundfile.readthedocs.io/en/latest/#soundfile.write
  - concurrent.futures.ProcessPoolExecutor:
      https://docs.python.org/3/library/concurrent.futures.html
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Make scripts/ importable without installing.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav_fixture(
    path: Path,
    samplerate: int,
    duration_s: float = 0.3,
) -> None:
    """Write a synthetic sine-wave WAV file as a test fixture."""
    n = int(samplerate * duration_s)
    t = np.linspace(0, duration_s, n, dtype=np.float32)
    data = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), data, samplerate)


def _write_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """Write a Kaggle-format CSV: filename,transcription,spk_id,gender."""
    with open(str(path), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "transcription", "spk_id", "gender"])
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Tests — precompute_audio CLI function
# ---------------------------------------------------------------------------


class TestPrecomputeAudioCreatesFiles:
    def test_creates_wav_files_in_output_dir(self, tmp_path: Path) -> None:
        """Creates one .wav per CSV row in output_dir."""
        # Arrange
        from precompute_audio import precompute_audio

        root = tmp_path / "root"
        root.mkdir()
        out = tmp_path / "cache"

        rows = []
        for i in range(3):
            fname = f"spk_A/audio_{i:03d}.wav"
            (root / "spk_A").mkdir(exist_ok=True)
            _write_wav_fixture(root / fname, samplerate=16000)
            rows.append((fname, "10000", "spk_A", "male"))

        csv_path = tmp_path / "train.csv"
        _write_csv(csv_path, rows)

        # Act
        stats = precompute_audio(
            csv_path=csv_path,
            root=root,
            output_dir=out,
            target_samplerate=16000,
        )

        # Assert — cache uses flat basename layout (matches Dataset lookup
        # which reads `cache_dir / record.audio_path.name`).
        assert stats["processed"] == 3
        assert stats["skipped"] == 0
        for i in range(3):
            expected = out / f"audio_{i:03d}.wav"
            assert expected.exists(), f"Expected {expected} to exist"

    def test_output_is_16khz_wav(self, tmp_path: Path) -> None:
        """Output files have samplerate == target_samplerate."""
        from precompute_audio import precompute_audio

        root = tmp_path / "root"
        root.mkdir()
        out = tmp_path / "cache"

        fname = "a.wav"
        _write_wav_fixture(root / fname, samplerate=24000)
        _write_csv(tmp_path / "t.csv", [(fname, "10000", "spk_A", "male")])

        precompute_audio(
            csv_path=tmp_path / "t.csv",
            root=root,
            output_dir=out,
            target_samplerate=16000,
        )

        info = sf.info(str(out / "a.wav"))
        assert info.samplerate == 16000


class TestPrecomputeAudioResample:
    def test_resamples_24khz_to_16khz(self, tmp_path: Path) -> None:
        """Input at 24 kHz is resampled; output at 16 kHz."""
        from precompute_audio import precompute_audio

        root = tmp_path / "root"
        root.mkdir()
        out = tmp_path / "cache"
        fname = "hi_sr.wav"
        _write_wav_fixture(root / fname, samplerate=24000, duration_s=0.5)
        _write_csv(tmp_path / "t.csv", [(fname, "10000", "spk_A", "male")])

        precompute_audio(
            csv_path=tmp_path / "t.csv",
            root=root,
            output_dir=out,
            target_samplerate=16000,
        )

        info = sf.info(str(out / fname))
        assert info.samplerate == 16000
        # Duration should be approximately preserved (within 5 %)
        assert abs(info.duration - 0.5) < 0.5 * 0.05


class TestPrecomputeAudioIdempotency:
    def test_skips_existing_files_by_default(self, tmp_path: Path) -> None:
        """Second call with overwrite=False skips all already-cached files."""
        from precompute_audio import precompute_audio

        root = tmp_path / "root"
        root.mkdir()
        out = tmp_path / "cache"
        rows = []
        for i in range(3):
            fname = f"f{i}.wav"
            _write_wav_fixture(root / fname, samplerate=16000)
            rows.append((fname, "10000", "spk_A", "male"))
        csv_path = tmp_path / "t.csv"
        _write_csv(csv_path, rows)

        # First run — should process 3
        s1 = precompute_audio(csv_path, root, out, target_samplerate=16000)
        assert s1["processed"] == 3

        # Second run — should skip 3
        s2 = precompute_audio(csv_path, root, out, target_samplerate=16000)
        assert s2["skipped"] == 3
        assert s2["processed"] == 0

    def test_overwrite_flag_rewrites_existing(self, tmp_path: Path) -> None:
        """Second call with overwrite=True re-writes all files."""
        from precompute_audio import precompute_audio

        root = tmp_path / "root"
        root.mkdir()
        out = tmp_path / "cache"
        fname = "x.wav"
        _write_wav_fixture(root / fname, samplerate=16000)
        _write_csv(tmp_path / "t.csv", [(fname, "10000", "spk_A", "male")])

        precompute_audio(tmp_path / "t.csv", root, out, target_samplerate=16000)
        mtime1 = (out / fname).stat().st_mtime

        # Sleep briefly to ensure mtime changes
        import time

        time.sleep(0.05)

        precompute_audio(
            tmp_path / "t.csv", root, out, target_samplerate=16000, overwrite=True
        )
        mtime2 = (out / fname).stat().st_mtime

        assert mtime2 > mtime1, "File should have been re-written with overwrite=True"


class TestPrecomputeAudioPcm16:
    def test_output_subtype_is_pcm_16(self, tmp_path: Path) -> None:
        """Cache WAV must use PCM_16 (int16) encoding to save disk space."""
        from precompute_audio import precompute_audio

        root = tmp_path / "root"
        root.mkdir()
        out = tmp_path / "cache"
        fname = "s.wav"
        _write_wav_fixture(root / fname, samplerate=16000)
        _write_csv(tmp_path / "t.csv", [(fname, "10000", "spk_A", "male")])

        precompute_audio(tmp_path / "t.csv", root, out, target_samplerate=16000)

        info = sf.info(str(out / fname))
        # soundfile reports subtype as e.g. "PCM_16"
        assert "PCM_16" in info.subtype.upper()


class TestPrecomputeAudioParallelWorkers:
    def test_parallel_workers_produce_same_output(self, tmp_path: Path) -> None:
        """With num_workers=2, all files are written without crash."""
        from precompute_audio import precompute_audio

        root = tmp_path / "root"
        root.mkdir()
        out = tmp_path / "cache"
        rows = []
        for i in range(3):
            fname = f"w{i}.wav"
            _write_wav_fixture(root / fname, samplerate=16000)
            rows.append((fname, "10000", "spk_A", "male"))
        csv_path = tmp_path / "t.csv"
        _write_csv(csv_path, rows)

        stats = precompute_audio(
            csv_path, root, out, target_samplerate=16000, num_workers=2
        )

        assert stats["processed"] == 3
        for i in range(3):
            assert (out / f"w{i}.wav").exists()


# ---------------------------------------------------------------------------
# Tests — Dataset cache-aware loading
# ---------------------------------------------------------------------------


class TestDatasetCacheAware:
    def _make_record(self, audio_path: Path, samplerate: int = 16000):
        from gp1.types import ManifestRecord

        return ManifestRecord(
            audio_path=audio_path,
            transcription="10000",
            spk_id="spk_A",
            gender="male",
            ext="wav",
            samplerate=samplerate,
        )

    def test_uses_cache_when_audio_cache_dir_set(self, tmp_path: Path) -> None:
        """dataset[0] reads from cache even if source file does not exist."""
        import torch

        from gp1.data.dataset import SpokenNumbersDataset
        from gp1.text.vocab import CharVocab

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Write a real cached WAV at 16 kHz
        cached_wav = cache_dir / "fake_audio.wav"
        n = 8000
        data = np.zeros(n, dtype=np.float32)
        sf.write(str(cached_wav), data, 16000)

        # Source file intentionally does NOT exist
        fake_src = tmp_path / "fake_audio.wav"
        record = self._make_record(fake_src, samplerate=16000)

        vocab = CharVocab()
        dataset = SpokenNumbersDataset([record], vocab, audio_cache_dir=cache_dir)

        # Act — must not raise FileNotFoundError
        item = dataset[0]

        # Assert
        assert "audio" in item
        assert item["audio"].dtype == torch.float32
        assert item["audio"].ndim == 1

    def test_falls_back_to_source_on_cache_miss(self, tmp_path: Path) -> None:
        """When cache_dir is set but cache file is missing, load from source."""
        import torch

        from gp1.data.dataset import SpokenNumbersDataset
        from gp1.text.vocab import CharVocab

        # Source file exists
        src = tmp_path / "real_audio.wav"
        sf.write(str(src), np.zeros(8000, dtype=np.float32), 16000)

        # Cache dir exists but no cached file
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        record = self._make_record(src, samplerate=16000)
        vocab = CharVocab()
        dataset = SpokenNumbersDataset([record], vocab, audio_cache_dir=cache_dir)

        # Act
        item = dataset[0]

        # Assert — fallback path works
        assert "audio" in item
        assert isinstance(item["audio"], torch.Tensor)

    def test_no_cache_works_unchanged(self, tmp_path: Path) -> None:
        """Default (audio_cache_dir=None) behaves exactly as before."""
        import torch

        from gp1.data.dataset import SpokenNumbersDataset
        from gp1.text.vocab import CharVocab

        src = tmp_path / "normal.wav"
        sf.write(str(src), np.zeros(8000, dtype=np.float32), 16000)

        record = self._make_record(src, samplerate=16000)
        vocab = CharVocab()
        dataset = SpokenNumbersDataset([record], vocab)

        item = dataset[0]

        assert "audio" in item
        assert isinstance(item["audio"], torch.Tensor)
        assert item["audio"].dtype == torch.float32
