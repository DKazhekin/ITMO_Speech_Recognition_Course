"""Collation and dynamic-bucket sampling for GP1 CTC batches.

collate_fn: zero-pads audio to a multiple of hop_length (160) and
    targets to the longest in the batch, then returns a Batch dataclass.

DynamicBucketSampler: groups indices by length into buckets, then packs
    each bucket into mini-batches whose total length fits within
    max_tokens_per_batch.

References:
  - CONTRACTS.md §4 "Data pipeline — collate.py"
  - Dynamic bucketing concept: SpeechBrain DataLoader / Lhotse bucketing
    https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/sampling/dynamic_bucketing.py
"""

from __future__ import annotations

import logging
import math
import random

import torch
import torch.utils.data

from gp1.types import Batch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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

    # Word-level targets are optional — the Dataset only populates them
    # when constructed with a WordVocab (enables the word-aux CTC head).
    has_word_targets = "word_target" in batch[0]
    word_targets_list: list[torch.Tensor] = (
        [item["word_target"] for item in batch] if has_word_targets else []
    )

    # Second augmented view is optional — populated by the Dataset only
    # when ``return_two_views=True`` (enables the CR-CTC consistency loss).
    has_view2 = "audio_view2" in batch[0]
    view2_list: list[torch.Tensor] = (
        [item["audio_view2"] for item in batch] if has_view2 else []
    )

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

    word_targets_padded: torch.Tensor | None = None
    word_target_lengths: torch.Tensor | None = None
    if has_word_targets:
        wtl = torch.tensor([w.shape[0] for w in word_targets_list], dtype=torch.int64)
        max_word_len = int(wtl.max().item())
        wtp = torch.zeros(len(batch), max_word_len, dtype=torch.int64)
        for i, w in enumerate(word_targets_list):
            wtp[i, : w.shape[0]] = w
        word_target_lengths = wtl
        word_targets_padded = wtp

    audio_view2_padded: torch.Tensor | None = None
    audio_view2_lengths: torch.Tensor | None = None
    if has_view2:
        v2_lengths = torch.tensor([v.shape[0] for v in view2_list], dtype=torch.int64)
        v2_max = int(v2_lengths.max().item())
        v2_padded_max = (
            math.ceil(v2_max / pad_audio_to_multiple) * pad_audio_to_multiple
        )
        v2_padded = torch.zeros(len(batch), v2_padded_max, dtype=torch.float32)
        for i, v in enumerate(view2_list):
            v2_padded[i, : v.shape[0]] = v
        audio_view2_padded = v2_padded
        audio_view2_lengths = v2_lengths

    return Batch(
        audio=audio_padded,
        audio_lengths=audio_lengths,
        targets=targets_padded,
        target_lengths=target_lengths,
        spk_ids=spk_ids,
        transcriptions=transcriptions,
        word_targets=word_targets_padded,
        word_target_lengths=word_target_lengths,
        audio_view2=audio_view2_padded,
        audio_view2_lengths=audio_view2_lengths,
    )


class DynamicBucketSampler(torch.utils.data.Sampler):
    """Length-aware batch sampler that keeps total tokens per batch bounded.

    Algorithm:
      1. Sort all indices by length (ascending).
      2. Split sorted indices into *num_buckets* roughly-equal contiguous groups.
      3. Within each bucket, greedily pack indices into mini-batches whose
         cumulative length <= *max_tokens_per_batch*. A single item whose
         length exceeds the budget forms a 1-item batch on its own.
      4. If *shuffle* is True: shuffle bucket order and shuffle within-bucket
         batches each epoch using an internal ``random.Random`` seeded from
         ``self._seed``.

    Attributes:
        _seed: Integer seed for reproducible shuffling. Set after construction
            to override (e.g., in tests: ``sampler._seed = 42``).

    Args:
        lengths: Sequence lengths (e.g. audio sample counts) for each dataset index.
        max_tokens_per_batch: Upper bound on the sum of lengths in one batch.
        num_buckets: Number of length-sorted buckets.
        shuffle: Whether to shuffle bucket order and intra-bucket batch order.
    """

    def __init__(
        self,
        lengths: list[int],
        max_tokens_per_batch: int,
        num_buckets: int = 20,
        shuffle: bool = True,
    ) -> None:
        self._lengths = list(lengths)
        self._max_tokens = max_tokens_per_batch
        self._num_buckets = max(1, num_buckets)
        self._shuffle = shuffle
        self._seed: int = 0
        self._epoch: int = (
            0  # incremented each __iter__ call for distinct per-epoch ordering
        )

        # Pre-compute batch lists (shuffle-independent structure).
        self._batches: list[list[int]] = self._build_batches(shuffle=False)

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(list(self._build_batches(shuffle=self._shuffle, epoch=self._epoch)))

    def __iter__(self):
        batches = self._build_batches(shuffle=self._shuffle, epoch=self._epoch)
        self._epoch += 1
        yield from batches

    def set_epoch(self, epoch: int) -> None:
        """Explicitly set the epoch counter for reproducible per-epoch shuffling.

        Mirrors the ``torch.utils.data.DistributedSampler.set_epoch`` API.
        Calling ``set_epoch(n)`` before iterating ensures that epoch *n* always
        produces the same shuffle regardless of how many times ``__iter__``
        was called before.

        Args:
            epoch: Zero-based epoch index.
        """
        self._epoch = epoch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_batches(self, shuffle: bool, epoch: int = 0) -> list[list[int]]:
        """Build the list of mini-batches from scratch.

        Sorting + bucketing is deterministic regardless of *shuffle*; only
        the order of the resulting batches changes when *shuffle* is True.
        The effective seed is ``self._seed + epoch`` so each epoch sees a
        distinct (but reproducible) permutation.
        """
        n = len(self._lengths)
        if n == 0:
            return []

        # Sort indices by length ascending.
        sorted_indices = sorted(range(n), key=lambda i: self._lengths[i])

        # Split into num_buckets roughly-equal groups.
        bucket_size = math.ceil(n / self._num_buckets)
        buckets: list[list[int]] = [
            sorted_indices[start : start + bucket_size]
            for start in range(0, n, bucket_size)
        ]

        # Within each bucket, greedily pack into mini-batches.
        all_batches: list[list[int]] = []
        for bucket in buckets:
            bucket_batches = self._pack_bucket(bucket)
            all_batches.extend(bucket_batches)

        if shuffle:
            rng = random.Random(self._seed + epoch)
            rng.shuffle(all_batches)
            for mb in all_batches:
                rng.shuffle(mb)

        return all_batches

    def _pack_bucket(self, indices: list[int]) -> list[list[int]]:
        """Greedily pack *indices* into batches within the token budget."""
        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_sum = 0

        for idx in indices:
            length = self._lengths[idx]
            if current_batch and current_sum + length > self._max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_sum = 0
            current_batch.append(idx)
            current_sum += length

        if current_batch:
            batches.append(current_batch)

        return batches
