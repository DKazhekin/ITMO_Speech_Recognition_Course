"""Collation utilities for GP1 CTC batches.

collate_fn: zero-pads audio to a multiple of hop_length (160) and
    targets to the longest in the batch, then returns a Batch dataclass.
"""

from __future__ import annotations

import logging
import math

import torch

from gp1.types import Batch

log = logging.getLogger(__name__)


def collate_fn(
    batch: list[dict],
    pad_audio_to_multiple: int = 160,
) -> Batch:
    """Collate a list of dataset items into a padded Batch.

    Audio tensors are zero-padded on the right to the smallest multiple of
    *pad_audio_to_multiple* that fits the longest audio in the batch.
    Target sequences are zero-padded to the longest target in the batch
    (CTC ignores padding via ``target_lengths``).

    Args:
        batch: List of dicts from SpokenNumbersDataset.__getitem__.
            Each dict must have keys: audio [T], target [U], spk_id, transcription.
        pad_audio_to_multiple: Pad audio width to a multiple of this value.
            Defaults to 160 (hop_length so mel frames align perfectly).

    Returns:
        Batch dataclass with tensors on CPU.
    """
    assert len(batch) > 0, "collate_fn received an empty batch"

    audios: list[torch.Tensor] = [item["audio"] for item in batch]
    targets: list[torch.Tensor] = [item["target"] for item in batch]
    spk_ids: list[str] = [item["spk_id"] for item in batch]
    transcriptions: list[str] = [item["transcription"] for item in batch]

    # Record actual (unpadded) lengths before padding.
    audio_lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.int64)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.int64)

    # Pad audio to a multiple of hop_length.
    max_audio_len = int(audio_lengths.max().item())
    padded_max = (
        math.ceil(max_audio_len / pad_audio_to_multiple) * pad_audio_to_multiple
    )

    audio_padded = torch.zeros(len(batch), padded_max, dtype=torch.float32)
    for i, wav in enumerate(audios):
        audio_padded[i, : wav.shape[0]] = wav

    # Pad targets with 0 (CTC loss uses target_lengths to ignore padding).
    max_target_len = int(target_lengths.max().item())
    targets_padded = torch.zeros(len(batch), max_target_len, dtype=torch.int64)
    for i, tgt in enumerate(targets):
        targets_padded[i, : tgt.shape[0]] = tgt

    return Batch(
        audio=audio_padded,
        audio_lengths=audio_lengths,
        targets=targets_padded,
        target_lengths=target_lengths,
        spk_ids=spk_ids,
        transcriptions=transcriptions,
    )
