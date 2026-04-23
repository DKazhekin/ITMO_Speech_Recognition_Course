"""Manifest I/O for the GP1 data pipeline.

Reads a Kaggle-style CSV (filename, transcription, spk_id, gender),
resolves absolute audio paths, queries native sample rates and durations
via soundfile.info (metadata only, no decode), and returns a list of
ManifestRecord objects.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import soundfile as sf

from gp1.types import ManifestRecord

log = logging.getLogger(__name__)


def records_from_csv(csv_path: Path, audio_root: Path) -> list[ManifestRecord]:
    """Read a Kaggle-style CSV and return one ManifestRecord per row.

    Each row's ``filename`` is resolved as ``audio_root / filename``. Sample
    rate and duration are read via ``soundfile.info`` (fast metadata-only).
    """
    csv_path = Path(csv_path)
    audio_root = Path(audio_root)
    records: list[ManifestRecord] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            filename: str = row["filename"]
            audio_path = (audio_root / filename).resolve()
            ext = audio_path.suffix.lstrip(".").lower()

            info = sf.info(str(audio_path))
            samplerate: int = info.samplerate
            duration_s: float = info.frames / info.samplerate

            records.append(
                ManifestRecord(
                    audio_path=audio_path,
                    transcription=row["transcription"],
                    spk_id=row["spk_id"],
                    gender=row["gender"],
                    ext=ext,
                    samplerate=samplerate,
                    duration_s=duration_s,
                )
            )

    log.info("records_from_csv: loaded %d records from %s", len(records), csv_path)
    return records


def leave_n_speakers_out_split(
    records: list[ManifestRecord],
    holdout_speakers: list[str],
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    """Partition records into train and dev splits by speaker id."""
    holdout_set = set(holdout_speakers)
    train: list[ManifestRecord] = []
    dev: list[ManifestRecord] = []

    for record in records:
        if record.spk_id in holdout_set:
            dev.append(record)
        else:
            train.append(record)

    log.info(
        "leave_n_speakers_out_split: train=%d, dev=%d (holdout=%s)",
        len(train),
        len(dev),
        sorted(holdout_set),
    )
    return train, dev
