"""Tests for digits_to_words — num2words(lang='ru') wrapper.

Contract (CONTRACTS.md §2): round-trip `n in range(1000, 1_000_000)` via
`digits_to_words -> words_to_digits` must equal `str(n)`. The full sweep
is opt-in via `-m slow`; the always-on suite uses a stratified sample
plus three random seeds to keep CI quick.
"""

from __future__ import annotations

import random

import pytest

from gp1.text.denormalize import words_to_digits
from gp1.text.normalize import digits_to_words

# Stratified sample — touches every decimal band and boundary condition.
_STRATA: tuple[int, ...] = (
    1000, 1001, 1005, 1010, 1100, 1234, 1999,
    2000, 2001, 2025, 2100,
    9999, 10000, 10001, 10500,
    12345, 21001, 21021,
    99999, 100000, 100001, 100025,
    123456, 139473, 250000,
    500000, 500500, 555555,
    700000, 888888, 999000, 999999,
)


@pytest.mark.parametrize("n", _STRATA)
def test_digits_to_words_stratified_sample_round_trips(n: int) -> None:
    # Arrange / Act
    words = digits_to_words(n)
    back = words_to_digits(words)

    # Assert
    assert back == str(n), f"{n=} -> {words!r} -> {back!r}"


def test_digits_to_words_accepts_string_input() -> None:
    # Arrange / Act
    out = digits_to_words("1005")

    # Assert
    assert out == "одна тысяча пять"


def test_digits_to_words_rejects_below_1000() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        digits_to_words(999)


def test_digits_to_words_rejects_above_999999() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        digits_to_words(1_000_000)


def test_digits_to_words_rejects_malformed_string() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        digits_to_words("abc")


def test_digits_to_words_output_is_lowercase_and_no_hyphens() -> None:
    # Arrange / Act
    out = digits_to_words(123456)

    # Assert
    assert out == out.lower()
    assert "-" not in out
    assert "  " not in out


def test_digits_to_words_1005_uses_feminine_odna() -> None:
    """Regression: num2words(1005, 'ru') == 'одна тысяча пять'."""
    # Arrange / Act
    out = digits_to_words(1005)

    # Assert
    assert out == "одна тысяча пять"


def test_digits_to_words_2000_uses_feminine_dve() -> None:
    # Arrange / Act
    out = digits_to_words(2000)

    # Assert
    assert out == "две тысячи"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_round_trip_random_sample(seed: int) -> None:
    """Random sample across the full range for each seed; smoke confidence."""
    # Arrange
    rng = random.Random(seed)
    sample = [rng.randint(1000, 999_999) for _ in range(500)]

    # Act / Assert
    for n in sample:
        assert words_to_digits(digits_to_words(n)) == str(n), f"failed on {n}"


@pytest.mark.slow
def test_round_trip_full_range_1000_to_999999() -> None:
    """Exhaustive round-trip over every integer in 1000..999999. Opt-in via -m slow."""
    # Arrange / Act / Assert
    for n in range(1000, 1_000_000):
        words = digits_to_words(n)
        assert words_to_digits(words) == str(n), f"failed on {n}: {words!r}"
