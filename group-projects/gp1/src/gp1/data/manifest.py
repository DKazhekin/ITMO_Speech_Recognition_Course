"""Manifest building and I/O for the GP1 data pipeline.

Reads a Kaggle-style CSV (filename, transcription, spk_id, gender),
resolves absolute audio paths, reads native sample rates via soundfile.info
(fast metadata-only read — no full decode), and writes a JSONL manifest of
ManifestRecord dicts.

References:
  - soundfile.info: https://python-soundfile.readthedocs.io/en/latest/#soundfile.info
  - CONTRACTS.md §4 "Data pipeline — manifest.py"
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import soundfile as sf

from gp1.types import ManifestRecord

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# TODO: как будто избыточный код, зачем создавать промежуточный манифест, если он все равно попадает в JSONL в виде словаря ?
def build_manifest(csv_path: Path, audio_root: Path, out_path: Path) -> int:
    """Build a JSONL manifest from a Kaggle-style CSV.

    Reads each row from *csv_path*, resolves the audio file to an absolute
    path under *audio_root*, queries its native sample rate via
    ``soundfile.info()`` (metadata only — no decode), and writes one JSON
    object per line to *out_path*.

    Args:
        csv_path: Path to the Kaggle CSV with columns:
            filename, transcription, spk_id, gender.
        audio_root: Root directory under which audio files live.
        out_path: Destination JSONL path (created or overwritten).

    Returns:
        Number of records written.
    """
    csv_path = Path(csv_path)
    audio_root = Path(audio_root)
    out_path = Path(out_path)

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

            record = ManifestRecord(
                audio_path=audio_path,
                transcription=row["transcription"],
                spk_id=row["spk_id"],
                gender=row["gender"],
                ext=ext,
                samplerate=samplerate,
                duration_s=duration_s,
            )
            records.append(record)

    write_jsonl(records, out_path)
    log.info("build_manifest: wrote %d records to %s", len(records), out_path)
    return len(records)


def write_jsonl(records: list[ManifestRecord], path: Path) -> None:
    """Serialise a list of ManifestRecord objects to a JSONL file.

    Each record is written as a single JSON object on its own line.
    ``audio_path`` is stored as a string (absolute posix path).

    Args:
        records: List of ManifestRecord dataclasses to serialise.
        path: Destination file path (created or overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            obj = {
                "audio_path": str(record.audio_path),
                "transcription": record.transcription,
                "spk_id": record.spk_id,
                "gender": record.gender,
                "ext": record.ext,
                "samplerate": record.samplerate,
                "duration_s": record.duration_s,
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    log.debug("write_jsonl: wrote %d records to %s", len(records), path)


def read_jsonl(path: Path) -> list[ManifestRecord]:
    """Deserialise a JSONL manifest into a list of ManifestRecord objects.

    Args:
        path: Path to the JSONL manifest file produced by ``write_jsonl``.

    Returns:
        List of ManifestRecord instances in file order.
    """
    path = Path(path)
    records: list[ManifestRecord] = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # duration_s may be absent in old JSONL files; fall back gracefully
            # to a conservative 2-second estimate matching the old train.py behaviour.
            raw_duration = obj.get("duration_s", 2.0)
            record = ManifestRecord(
                audio_path=Path(obj["audio_path"]),
                transcription=obj["transcription"],
                spk_id=obj["spk_id"],
                gender=obj["gender"],
                ext=obj["ext"],
                samplerate=int(obj["samplerate"]),
                duration_s=float(raw_duration),
            )
            records.append(record)

    log.debug("read_jsonl: loaded %d records from %s", len(records), path)
    return records


def leave_n_speakers_out_split(
    records: list[ManifestRecord],
    holdout_speakers: list[str],
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    """Partition records into train and dev splits by speaker id.

    Records whose ``spk_id`` is in *holdout_speakers* are placed in the dev
    split; all others go to train. Input order is preserved within each
    split. The input list is not mutated.

    Args:
        records: Full list of ManifestRecord objects.
        holdout_speakers: Speaker ids to reserve for the dev split.

    Returns:
        Tuple of (train_records, dev_records).
    """
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
