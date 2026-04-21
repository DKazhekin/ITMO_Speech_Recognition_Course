"""Tests for gp1.data.collate (CONTRACTS.md §4).

TDD RED->GREEN->REFACTOR. Tests written before implementation.
AAA (Arrange-Act-Assert) pattern throughout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gp1.data.collate import DynamicBucketSampler, collate_fn
from gp1.types import Batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(audio_len: int, target_len: int, idx: int = 0) -> dict:
    """Create a minimal dataset item dict for testing collate_fn."""
    return {
        "audio": torch.zeros(audio_len, dtype=torch.float32),
        "target": torch.arange(target_len, dtype=torch.int64),
        "spk_id": f"spk_{idx}",
        "transcription": str(10000 + idx),
    }


# ---------------------------------------------------------------------------
# collate_fn tests
# ---------------------------------------------------------------------------


class TestCollateFn:
    def test_pads_audio_to_multiple_of_160(self) -> None:
        """After collation audio.shape[1] is divisible by 160."""
        # Arrange — three items with lengths not divisible by 160
        items = [
            _make_item(audio_len=100, target_len=5, idx=0),
            _make_item(audio_len=250, target_len=7, idx=1),
            _make_item(audio_len=310, target_len=6, idx=2),
        ]

        # Act
        batch = collate_fn(items, pad_audio_to_multiple=160)

        # Assert
        assert batch.audio.shape[1] % 160 == 0, (
            f"Audio width {batch.audio.shape[1]} is not a multiple of 160"
        )

    def test_audio_shape_is_batch_by_max_padded_length(self) -> None:
        """batch.audio has shape [B, T_max_padded]."""
        # Arrange
        items = [
            _make_item(audio_len=160, target_len=3, idx=0),
            _make_item(audio_len=320, target_len=4, idx=1),
        ]

        # Act
        batch = collate_fn(items, pad_audio_to_multiple=160)

        # Assert
        assert batch.audio.shape[0] == 2
        assert batch.audio.shape[1] >= 320
        assert batch.audio.shape[1] % 160 == 0

    def test_audio_lengths_match_pre_pad_lengths(self) -> None:
        """audio_lengths[i] equals the actual unpadded audio length for item i."""
        # Arrange
        lengths = [100, 250, 310]
        items = [
            _make_item(audio_len=l, target_len=5, idx=i) for i, l in enumerate(lengths)
        ]

        # Act
        batch = collate_fn(items, pad_audio_to_multiple=160)

        # Assert
        for i, expected_len in enumerate(lengths):
            assert batch.audio_lengths[i].item() == expected_len, (
                f"Item {i}: expected audio_lengths={expected_len}, "
                f"got {batch.audio_lengths[i].item()}"
            )

    def test_audio_lengths_dtype_is_int64(self) -> None:
        """audio_lengths tensor dtype is int64."""
        # Arrange
        items = [_make_item(audio_len=160, target_len=4)]

        # Act
        batch = collate_fn(items)

        # Assert
        assert batch.audio_lengths.dtype == torch.int64

    def test_targets_padded_to_longest(self) -> None:
        """targets are zero-padded to max target length; target_lengths reflect actuals."""
        # Arrange
        target_lens = [3, 7, 5]
        items = [
            _make_item(audio_len=160, target_len=l, idx=i)
            for i, l in enumerate(target_lens)
        ]

        # Act
        batch = collate_fn(items)

        # Assert
        assert batch.targets.shape == (3, max(target_lens))
        for i, expected_len in enumerate(target_lens):
            assert batch.target_lengths[i].item() == expected_len

    def test_target_padding_uses_zero(self) -> None:
        """Padding positions in targets tensor are filled with 0 (not vocab.BLANK_ID
        semantics differ — 0 is just the pad value; CTC ignores via target_lengths)."""
        # Arrange — one short and one long target
        items = [
            _make_item(audio_len=160, target_len=2, idx=0),
            _make_item(audio_len=160, target_len=5, idx=1),
        ]

        # Act
        batch = collate_fn(items)

        # Assert — the short target's trailing positions should be 0
        short_target = batch.targets[0]
        assert short_target[2:].sum().item() == 0, (
            "Expected trailing positions of short target to be padded with 0"
        )

    def test_transcriptions_pass_through_in_order(self) -> None:
        """transcriptions list preserves order from input batch."""
        # Arrange
        items = [
            _make_item(audio_len=160, target_len=3, idx=0),
            _make_item(audio_len=160, target_len=4, idx=1),
            _make_item(audio_len=160, target_len=5, idx=2),
        ]

        # Act
        batch = collate_fn(items)

        # Assert
        assert batch.transcriptions == ["10000", "10001", "10002"]

    def test_spk_ids_pass_through_in_order(self) -> None:
        """spk_ids list preserves order from input batch."""
        # Arrange
        items = [
            _make_item(audio_len=160, target_len=3, idx=0),
            _make_item(audio_len=160, target_len=4, idx=1),
        ]

        # Act
        batch = collate_fn(items)

        # Assert
        assert batch.spk_ids == ["spk_0", "spk_1"]

    def test_returns_batch_dataclass(self) -> None:
        """collate_fn returns an instance of Batch."""
        # Arrange
        items = [_make_item(audio_len=160, target_len=3)]

        # Act
        batch = collate_fn(items)

        # Assert
        assert isinstance(batch, Batch)

    def test_audio_dtype_float32(self) -> None:
        """batch.audio dtype is float32."""
        # Arrange
        items = [_make_item(audio_len=320, target_len=4)]

        # Act
        batch = collate_fn(items)

        # Assert
        assert batch.audio.dtype == torch.float32

    def test_custom_multiple_respected(self) -> None:
        """pad_audio_to_multiple parameter is respected for non-default values."""
        # Arrange
        items = [_make_item(audio_len=100, target_len=3)]

        # Act
        batch = collate_fn(items, pad_audio_to_multiple=64)

        # Assert
        assert batch.audio.shape[1] % 64 == 0

    def test_single_item_batch(self) -> None:
        """Single-item batch works correctly."""
        # Arrange
        items = [_make_item(audio_len=160, target_len=5)]

        # Act
        batch = collate_fn(items)

        # Assert
        assert batch.audio.shape[0] == 1
        assert batch.targets.shape[0] == 1
        assert len(batch.spk_ids) == 1


# ---------------------------------------------------------------------------
# DynamicBucketSampler tests
# ---------------------------------------------------------------------------


class TestDynamicBucketSampler:
    def test_every_batch_within_token_budget(self) -> None:
        """Every yielded batch has sum(lengths) <= max_tokens_per_batch."""
        # Arrange
        lengths = [100, 200, 300, 400, 500]
        max_tokens = 600

        sampler = DynamicBucketSampler(
            lengths=lengths,
            max_tokens_per_batch=max_tokens,
            num_buckets=2,
            shuffle=False,
        )

        # Act
        batches = list(sampler)

        # Assert
        assert len(batches) > 0, "Sampler yielded no batches"
        for batch_indices in batches:
            batch_sum = sum(lengths[i] for i in batch_indices)
            assert batch_sum <= max_tokens, (
                f"Batch {batch_indices} has token sum {batch_sum} > {max_tokens}"
            )

    def test_all_indices_covered_exactly_once(self) -> None:
        """Every index appears exactly once across all batches."""
        # Arrange
        lengths = [100, 200, 150, 300, 250, 400, 350, 500]
        sampler = DynamicBucketSampler(
            lengths=lengths,
            max_tokens_per_batch=700,
            num_buckets=3,
            shuffle=False,
        )

        # Act
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        # Assert
        assert sorted(all_indices) == list(range(len(lengths))), (
            f"Expected each index 0..{len(lengths) - 1} exactly once, "
            f"got: {sorted(all_indices)}"
        )

    def test_same_seed_produces_identical_order(self) -> None:
        """Two samplers with the same seed yield identical batch sequences."""
        # Arrange
        lengths = [100, 200, 300, 400, 500, 150, 250, 350]
        kwargs = dict(
            lengths=lengths, max_tokens_per_batch=600, num_buckets=3, shuffle=True
        )

        sampler_a = DynamicBucketSampler(**kwargs)
        sampler_a._seed = 42
        sampler_b = DynamicBucketSampler(**kwargs)
        sampler_b._seed = 42

        # Act
        batches_a = list(sampler_a)
        batches_b = list(sampler_b)

        # Assert
        assert batches_a == batches_b, "Same seed should produce identical batch order"

    def test_different_seeds_produce_different_order(self) -> None:
        """Two samplers with different seeds should (very likely) yield different orders."""
        # Arrange
        lengths = list(range(100, 600, 50))  # 10 lengths
        kwargs = dict(
            lengths=lengths, max_tokens_per_batch=500, num_buckets=3, shuffle=True
        )

        sampler_a = DynamicBucketSampler(**kwargs)
        sampler_a._seed = 0
        sampler_b = DynamicBucketSampler(**kwargs)
        sampler_b._seed = 999

        # Act
        batches_a = list(sampler_a)
        batches_b = list(sampler_b)

        # Assert — with 10 lengths this almost certainly differs
        assert batches_a != batches_b, (
            "Different seeds should (almost certainly) produce different batch orders"
        )

    def test_len_returns_number_of_batches(self) -> None:
        """__len__ returns the number of batches that __iter__ would yield."""
        # Arrange
        lengths = [100, 200, 300, 400, 500]
        sampler = DynamicBucketSampler(
            lengths=lengths,
            max_tokens_per_batch=600,
            num_buckets=2,
            shuffle=False,
        )

        # Act
        declared_len = len(sampler)
        actual_len = len(list(sampler))

        # Assert
        assert declared_len == actual_len, (
            f"__len__ returned {declared_len} but iter yielded {actual_len} batches"
        )

    def test_empty_lengths_raises_or_yields_nothing(self) -> None:
        """Empty lengths list: sampler yields no batches."""
        # Arrange & Act
        sampler = DynamicBucketSampler(
            lengths=[],
            max_tokens_per_batch=1000,
            num_buckets=5,
            shuffle=False,
        )

        # Assert
        batches = list(sampler)
        assert batches == [], f"Expected no batches, got {batches}"

    def test_single_item_too_large_still_included(self) -> None:
        """A single item whose length exceeds max_tokens still appears as a 1-item batch."""
        # Arrange — single item larger than budget
        lengths = [1000]
        sampler = DynamicBucketSampler(
            lengths=lengths,
            max_tokens_per_batch=500,
            num_buckets=1,
            shuffle=False,
        )

        # Act
        batches = list(sampler)

        # Assert
        assert len(batches) == 1
        assert batches[0] == [0]

    def test_shuffle_false_no_seed_required(self) -> None:
        """shuffle=False produces a deterministic order without needing _seed."""
        # Arrange
        lengths = [300, 100, 200, 400, 150]
        sampler = DynamicBucketSampler(
            lengths=lengths,
            max_tokens_per_batch=600,
            num_buckets=2,
            shuffle=False,
        )

        # Act — two iterations should yield identical results
        run1 = list(sampler)
        run2 = list(sampler)

        # Assert
        assert run1 == run2
