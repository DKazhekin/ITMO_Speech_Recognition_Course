"""Architecture-neutral inference utilities for GP1 submission notebooks.

Provides:
- build_test_dataloader: deterministic, shuffle-free DataLoader from ManifestRecords
- write_submission: write (filename, prediction) pairs to a Kaggle CSV
"""

from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.data import DataLoader

from gp1.data.collate import collate_fn
from gp1.data.dataset import SpokenNumbersDataset, VocabProtocol
from gp1.types import ManifestRecord


def build_test_dataloader(
    records: list[ManifestRecord],
    vocab: VocabProtocol,
    batch_size: int = 32,
    num_workers: int = 0,
    target_samplerate: int = 16000,
) -> DataLoader:
    """Build a deterministic, shuffle-free DataLoader for test/inference.

    Uses SpokenNumbersDataset (without augmenter) + standard collate_fn.
    Sampler is a SequentialSampler to preserve record order.
    """
    dataset = SpokenNumbersDataset(
        records=records,
        vocab=vocab,
        target_samplerate=target_samplerate,
        augmenter=None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def write_submission(
    pairs: list[tuple[str, str]],
    out_path: Path,
    header: tuple[str, str] = ("filename", "transcription"),
) -> None:
    """Write (filename, prediction) pairs to a Kaggle-style submission CSV.

    Overwrites out_path. Creates parent dirs if missing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(pairs)
