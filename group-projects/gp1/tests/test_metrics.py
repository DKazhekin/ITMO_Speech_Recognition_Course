"""Tests for gp1.train.metrics — compute_cer and compute_per_speaker_cer.

TDD: RED phase — all tests must fail before implementation.
CER is corpus-level Levenshtein / sum(reference lengths), NOT mean of per-sample CERs.

CONTRACTS.md §8.
"""

from __future__ import annotations

import logging
import math

import pytest

from gp1.train.metrics import (
    compute_cer,
    compute_digit_cer_in_out_harmonic,
    compute_per_speaker_cer,
)


# ---------------------------------------------------------------------------
# compute_cer
# ---------------------------------------------------------------------------


def test_cer_identical_strings_returns_zero():
    # Arrange
    refs = ["привет мир", "тест"]
    hyps = ["привет мир", "тест"]

    # Act
    cer = compute_cer(refs, hyps)

    # Assert
    assert cer == 0.0


def test_cer_single_substitution_in_6_char_string_returns_one_sixth():
    # Arrange
    refs = ["абвгде"]  # 6 chars
    hyps = ["абвгда"]  # last char differs → 1 edit

    # Act
    cer = compute_cer(refs, hyps)

    # Assert
    assert abs(cer - 1.0 / 6.0) < 1e-9


def test_cer_empty_reference_and_empty_hypothesis_returns_zero():
    # Arrange: both empty → 0 edits / 0 ref chars; convention: 0.0
    refs = [""]
    hyps = [""]

    # Act
    cer = compute_cer(refs, hyps)

    # Assert
    assert cer == 0.0


def test_cer_empty_reference_with_nonempty_hypothesis_contributes_zero_to_corpus():
    # Arrange: empty ref contributes 0 to numerator AND denominator.
    # Corpus CER = total_edits / total_ref_len.
    # Only pair with non-empty ref drives the metric.
    refs = ["", "аб"]
    hyps = ["вгд", "аб"]

    # Act
    cer = compute_cer(refs, hyps)

    # Assert: 0 edits / 2 ref chars = 0.0
    assert cer == 0.0


def test_cer_corpus_level_weighted_by_reference_length():
    # Arrange: two pairs:
    #   pair 1: ref="аааааааааа" (10 chars), hyp="аааааааааа" → 0 edits
    #   pair 2: ref="аа" (2 chars),          hyp="бб"         → 2 edits
    # Corpus CER = (0 + 2) / (10 + 2) = 2/12 ≈ 0.1667
    # Mean-of-per-sample would give (0/10 + 2/2) / 2 = 0.5 — DIFFERENT
    refs = ["аааааааааа", "аа"]
    hyps = ["аааааааааа", "бб"]

    # Act
    cer = compute_cer(refs, hyps)

    # Assert
    expected = 2.0 / 12.0
    assert abs(cer - expected) < 1e-9


def test_cer_full_deletion_returns_one():
    # Arrange
    refs = ["абв"]
    hyps = [""]

    # Act
    cer = compute_cer(refs, hyps)

    # Assert
    assert abs(cer - 1.0) < 1e-9


def test_cer_with_all_insertions_returns_greater_than_one():
    # Arrange: hypothesis has more chars than reference
    refs = ["аб"]
    hyps = ["абвгд"]

    # Act
    cer = compute_cer(refs, hyps)

    # Assert: 3 insertions / 2 ref chars = 1.5
    assert abs(cer - 1.5) < 1e-9


def test_cer_raises_value_error_when_lists_have_mismatched_lengths():
    # Arrange
    refs = ["а", "б"]
    hyps = ["а"]

    # Act / Assert
    with pytest.raises(ValueError, match="mismatched"):
        compute_cer(refs, hyps)


# ---------------------------------------------------------------------------
# compute_per_speaker_cer
# ---------------------------------------------------------------------------


def test_per_speaker_cer_groups_by_spk_id_and_returns_one_entry_per_speaker():
    # Arrange
    refs = ["аб", "вг", "де"]
    hyps = ["аб", "вг", "де"]
    spk_ids = ["spk_A", "spk_B", "spk_A"]

    # Act
    result = compute_per_speaker_cer(refs, hyps, spk_ids)

    # Assert
    assert set(result.keys()) == {"spk_A", "spk_B"}
    assert result["spk_A"] == 0.0
    assert result["spk_B"] == 0.0


def test_per_speaker_cer_respects_same_weighting_as_compute_cer_within_group():
    # Arrange: spk_A has two pairs; CER within group is corpus-weighted.
    #   spk_A: ref="аааааааааа" (10), hyp="аааааааааа" → 0 edits
    #          ref="аа" (2),           hyp="бб"         → 2 edits
    #          expected CER = 2/12
    #   spk_B: ref="ааа" (3), hyp="ааа" → 0 edits; expected CER = 0
    refs = ["аааааааааа", "аа", "ааа"]
    hyps = ["аааааааааа", "бб", "ааа"]
    spk_ids = ["spk_A", "spk_A", "spk_B"]

    # Act
    result = compute_per_speaker_cer(refs, hyps, spk_ids)

    # Assert
    assert abs(result["spk_A"] - 2.0 / 12.0) < 1e-9
    assert result["spk_B"] == 0.0


def test_per_speaker_cer_raises_value_error_on_length_mismatch():
    # Arrange
    refs = ["а"]
    hyps = ["а", "б"]
    spk_ids = ["spk_A"]

    # Act / Assert
    with pytest.raises(ValueError):
        compute_per_speaker_cer(refs, hyps, spk_ids)


def test_per_speaker_cer_single_speaker_matches_compute_cer():
    # Arrange
    refs = ["тест", "привет"]
    hyps = ["тест", "приvet"]  # last 2 chars wrong in second
    spk_ids = ["X", "X"]

    # Act
    per_spk = compute_per_speaker_cer(refs, hyps, spk_ids)
    corpus = compute_cer(refs, hyps)

    # Assert: single speaker result == corpus result
    assert abs(per_spk["X"] - corpus) < 1e-9


# ---------------------------------------------------------------------------
# Harmonic in/out digit-CER
# ---------------------------------------------------------------------------


def test_harmonic_in_out_digit_cer_happy_path_non_empty_subgroups():
    # Arrange: in-domain = {spk_A}, out = {spk_B}.
    #   in:  ref="123" (3), hyp="124" -> 1 edit; in_cer = 1/3
    #   out: ref="56"  (2), hyp="78"  -> 2 edits; out_cer = 2/2 = 1.0
    refs = ["123", "56"]
    hyps = ["124", "78"]
    spk_ids = ["spk_A", "spk_B"]
    in_domain = {"spk_A"}

    # Act
    in_cer, out_cer, hm = compute_digit_cer_in_out_harmonic(
        refs, hyps, spk_ids, in_domain
    )

    # Assert
    expected_in = 1.0 / 3.0
    expected_out = 1.0
    expected_hm = 2.0 * expected_in * expected_out / (expected_in + expected_out)
    assert abs(in_cer - expected_in) < 1e-9
    assert abs(out_cer - expected_out) < 1e-9
    assert abs(hm - expected_hm) < 1e-9


def test_harmonic_in_out_digit_cer_swapping_groups_yields_swapped_cer_and_same_harmonic():
    # Arrange
    refs = ["123", "56"]
    hyps = ["124", "78"]
    spk_ids = ["spk_A", "spk_B"]

    # Act
    in_a, out_a, hm_a = compute_digit_cer_in_out_harmonic(
        refs, hyps, spk_ids, {"spk_A"}
    )
    in_b, out_b, hm_b = compute_digit_cer_in_out_harmonic(
        refs, hyps, spk_ids, {"spk_B"}
    )

    # Assert: swap in/out, harmonic mean is symmetric.
    assert abs(in_a - out_b) < 1e-9
    assert abs(out_a - in_b) < 1e-9
    assert abs(hm_a - hm_b) < 1e-9


def test_harmonic_in_out_digit_cer_in_subgroup_empty_returns_out_cer_and_warns(caplog):
    # Arrange: none of the speakers are in the in_domain set.
    refs = ["123", "56"]
    hyps = ["124", "78"]
    spk_ids = ["spk_A", "spk_B"]
    in_domain: set[str] = {"spk_UNKNOWN"}

    # Act
    with caplog.at_level(logging.WARNING, logger="gp1.train.metrics"):
        in_cer, out_cer, hm = compute_digit_cer_in_out_harmonic(
            refs, hyps, spk_ids, in_domain
        )

    # Assert: in is empty -> in_cer = 0.0 (compute_cer on [] returns 0.0),
    # hm falls back to max(in_cer, out_cer) = out_cer.
    expected_out = (1 + 2) / (3 + 2)
    assert in_cer == 0.0
    assert abs(out_cer - expected_out) < 1e-9
    assert abs(hm - max(in_cer, out_cer)) < 1e-9
    assert any("empty" in rec.message.lower() for rec in caplog.records)


def test_harmonic_in_out_digit_cer_out_subgroup_empty_returns_in_cer_and_warns(caplog):
    # Arrange: every speaker is in the in_domain set.
    refs = ["123", "56"]
    hyps = ["124", "78"]
    spk_ids = ["spk_A", "spk_B"]
    in_domain = {"spk_A", "spk_B"}

    # Act
    with caplog.at_level(logging.WARNING, logger="gp1.train.metrics"):
        in_cer, out_cer, hm = compute_digit_cer_in_out_harmonic(
            refs, hyps, spk_ids, in_domain
        )

    # Assert
    expected_in = (1 + 2) / (3 + 2)
    assert abs(in_cer - expected_in) < 1e-9
    assert out_cer == 0.0
    assert abs(hm - max(in_cer, out_cer)) < 1e-9
    assert any("empty" in rec.message.lower() for rec in caplog.records)


def test_harmonic_in_out_digit_cer_both_subgroups_empty_returns_zeros():
    # Arrange
    refs: list[str] = []
    hyps: list[str] = []
    spk_ids: list[str] = []
    in_domain: set[str] = set()

    # Act
    in_cer, out_cer, hm = compute_digit_cer_in_out_harmonic(
        refs, hyps, spk_ids, in_domain
    )

    # Assert
    assert in_cer == 0.0
    assert out_cer == 0.0
    assert hm == 0.0


def test_harmonic_in_out_digit_cer_both_zero_cer_returns_zero_not_nan():
    # Arrange: both subgroups have identical refs/hyps -> 0 errors each.
    refs = ["12", "34"]
    hyps = ["12", "34"]
    spk_ids = ["spk_A", "spk_B"]
    in_domain = {"spk_A"}

    # Act
    in_cer, out_cer, hm = compute_digit_cer_in_out_harmonic(
        refs, hyps, spk_ids, in_domain
    )

    # Assert
    assert in_cer == 0.0
    assert out_cer == 0.0
    assert hm == 0.0
    assert not math.isnan(hm)


def test_harmonic_in_out_digit_cer_raises_on_length_mismatch():
    # Arrange
    refs = ["12", "34"]
    hyps = ["12"]  # shorter
    spk_ids = ["spk_A", "spk_B"]
    in_domain: set[str] = {"spk_A"}

    # Act / Assert
    with pytest.raises(ValueError, match="mismatched"):
        compute_digit_cer_in_out_harmonic(refs, hyps, spk_ids, in_domain)


def test_harmonic_in_out_digit_cer_raises_on_spk_ids_length_mismatch():
    # Arrange
    refs = ["12", "34"]
    hyps = ["12", "34"]
    spk_ids = ["spk_A"]  # shorter
    in_domain: set[str] = {"spk_A"}

    # Act / Assert
    with pytest.raises(ValueError, match="mismatched"):
        compute_digit_cer_in_out_harmonic(refs, hyps, spk_ids, in_domain)
