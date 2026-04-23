"""One-shot audio precompute CLI for GP1.

Walks a Kaggle-style CSV (filename, transcription, spk_id, gender),
decodes each audio file, resamples it to *target_samplerate* Hz, and
writes a PCM_16 WAV to *output_dir* preserving the relative path structure.

Usage::

    python scripts/precompute_audio.py \\
        --csv data/train/train.csv \\
        --root data/train \\
        --output-dir data/_cache/train \\
        --target-samplerate 16000 \\
        --num-workers 4

The script is idempotent: files that already exist in *output_dir* are
skipped unless ``--overwrite`` is passed.

References:
  - torchaudio.transforms.Resample:
      https://pytorch.org/audio/stable/transforms.html#resample
  - soundfile.write PCM_16 subtype:
      https://python-soundfile.readthedocs.io/en/latest/#soundfile.write
  - concurrent.futures.ProcessPoolExecutor:
      https://docs.python.org/3/library/concurrent.futures.html
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import logging
import os
import sys
from pathlib import Path
from typing import TypedDict

import soundfile as sf
import torch
import torchaudio.transforms as T

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("gp1.precompute_audio")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class PrecomputeStats(TypedDict):
    processed: int
    skipped: int
    total_audio_seconds: float
    total_bytes: int


# ---------------------------------------------------------------------------
# Per-file worker (must be importable at module level for multiprocessing)
# ---------------------------------------------------------------------------


def _process_one(
    src_path: str,
    dst_path: str,
    orig_sr: int,
    target_sr: int,
    overwrite: bool,
) -> tuple[bool, float]:
    """Decode, resample, and write one cached WAV.

    Returns:
        (was_processed, audio_duration_seconds).
        was_processed is False when the file was skipped (already exists).
    """
    dst = Path(dst_path)
    if dst.exists() and not overwrite:
        data, sr = sf.read(str(dst), dtype="float32", always_2d=False)
        duration = len(data) / sr
        return False, duration

    # Decode source audio.
    data, sr = sf.read(src_path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)

    wav = torch.from_numpy(data.copy()).float()

    # Resample when needed.
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav.unsqueeze(0)).squeeze(0)

    duration = wav.shape[0] / target_sr

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), wav.numpy(), target_sr, subtype="PCM_16")

    return True, duration


def _process_one_args(args: tuple) -> tuple[bool, float]:
    """Unpacking wrapper for ProcessPoolExecutor.map."""
    return _process_one(*args)


# ---------------------------------------------------------------------------
# Public API (callable from tests without subprocess)
# ---------------------------------------------------------------------------


def precompute_audio(
    csv_path: Path,
    root: Path,
    output_dir: Path,
    target_samplerate: int = 16000,
    num_workers: int = 0,
    overwrite: bool = False,
) -> PrecomputeStats:
    """Precompute resampled 16-kHz WAV cache from a Kaggle-style CSV.

    Args:
        csv_path: Path to CSV with columns: filename, transcription, spk_id, gender.
        root: Root directory that ``filename`` values are relative to.
        output_dir: Where to write cached WAV files (same relative layout).
        target_samplerate: Target sample rate in Hz.
        num_workers: Number of parallel worker processes (0 = sequential).
        overwrite: Re-write files that already exist in output_dir.

    Returns:
        Dict with counts: processed, skipped, total_audio_seconds, total_bytes.
    """
    csv_path = Path(csv_path)
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV rows.
    rows: list[dict[str, str]] = []
    with open(str(csv_path), newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        return PrecomputeStats(
            processed=0, skipped=0, total_audio_seconds=0.0, total_bytes=0
        )

    # Build (src, dst, orig_sr, target_sr, overwrite) tuples.
    # We probe the source sample rate once to fill orig_sr correctly; using
    # sf.info avoids a full decode just to get the header.
    task_args: list[tuple[str, str, int, int, bool]] = []
    for row in rows:
        filename: str = row["filename"]
        src = root / filename
        # Cache uses a flat basename layout so Dataset can look up by
        # `record.audio_path.name` without relying on the original CSV
        # path structure. See SpokenNumbersDataset._load_audio_maybe_cached.
        dst = (output_dir / Path(filename).name).with_suffix(".wav")
        try:
            orig_sr = sf.info(str(src)).samplerate
        except Exception:
            # If probing fails (e.g. mp3 without libsndfile codec), fall back
            # to target_samplerate to let _process_one handle the error.
            orig_sr = target_samplerate
        task_args.append((str(src), str(dst), orig_sr, target_samplerate, overwrite))

    # Execute — parallel or sequential.
    results: list[tuple[bool, float]] = []
    if num_workers > 0:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            results = list(executor.map(_process_one_args, task_args))
    else:
        for args in task_args:
            results.append(_process_one(*args))

    processed = sum(1 for was_proc, _ in results if was_proc)
    skipped = len(results) - processed
    total_seconds = sum(dur for _, dur in results)
    total_bytes = sum(
        Path(args[1]).stat().st_size for args in task_args if Path(args[1]).exists()
    )

    log.info(
        "precompute_audio done: %d processed, %d skipped, %.1f s total audio, %.1f MB",
        processed,
        skipped,
        total_seconds,
        total_bytes / 1024 / 1024,
    )

    return PrecomputeStats(
        processed=processed,
        skipped=skipped,
        total_audio_seconds=total_seconds,
        total_bytes=total_bytes,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute resampled 16-kHz PCM_16 WAV cache from a Kaggle-style CSV.\n"
            "Idempotent: existing files are skipped unless --overwrite is passed."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Kaggle CSV with columns: filename, transcription, spk_id, gender",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory that filename values in the CSV are relative to",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for cached WAV files",
    )
    parser.add_argument(
        "--target-samplerate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (0 = sequential, default: 4)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-write files that already exist in output-dir",
    )
    args = parser.parse_args()

    stats = precompute_audio(
        csv_path=args.csv,
        root=args.root,
        output_dir=args.output_dir,
        target_samplerate=args.target_samplerate,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )

    print(
        f"Done: {stats['processed']} processed, {stats['skipped']} skipped, "
        f"{stats['total_audio_seconds']:.1f}s audio, "
        f"{stats['total_bytes'] / 1024 / 1024:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
