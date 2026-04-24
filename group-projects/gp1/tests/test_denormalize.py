"""Tests for words_to_digits — deterministic reducer over Russian number words."""

from __future__ import annotations

import pytest

from gp1.text import denormalize as denorm_module
from gp1.text.denormalize import (
    fsa_constrained_best,
    safe_words_to_digits,
    words_to_digits,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("одна тысяча", "1000"),
        ("одна тысяча пять", "1005"),
        ("две тысячи", "2000"),
        ("сто двадцать тысяч", "120000"),
        ("сто двадцать три", "123"),
        ("сто двадцать три тысячи", "123000"),
        ("сто тридцать девять тысяч четыреста семьдесят три", "139473"),
        ("пятьсот тысяч", "500000"),
        ("девятьсот девяносто девять тысяч девятьсот девяносто девять", "999999"),
        ("двадцать одна тысяча один", "21001"),
        ("сто тысяч", "100000"),
    ],
)
def test_known_fixtures(text: str, expected: str) -> None:
    # Arrange / Act
    out = words_to_digits(text)

    # Assert
    assert out == expected


def test_case_insensitive_input() -> None:
    # Arrange / Act
    out = words_to_digits("Сто Двадцать Три")

    # Assert
    assert out == "123"


def test_collapses_extra_whitespace() -> None:
    # Arrange / Act
    out = words_to_digits("  сто   двадцать\tтри  ")

    # Assert
    assert out == "123"


def test_replaces_hyphens_with_spaces() -> None:
    # Arrange / Act
    out = words_to_digits("сто-двадцать-три")

    # Assert
    assert out == "123"


def test_raises_on_empty_string() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        words_to_digits("")


def test_raises_on_unknown_word() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        words_to_digits("сто абракадабра три")


def test_raises_on_standalone_tysyacha_without_head() -> None:
    """'тысяча' alone has no numeric head → malformed."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        words_to_digits("тысяча")


def test_one_masculine_for_units() -> None:
    # Arrange / Act
    out = words_to_digits("двести сорок один")

    # Assert
    assert out == "241"


def test_one_feminine_odna_before_tysyacha() -> None:
    # Arrange / Act
    out = words_to_digits("одна тысяча")

    # Assert
    assert out == "1000"


def test_two_feminine_dve_before_tysyachi() -> None:
    # Arrange / Act
    out = words_to_digits("две тысячи")

    # Assert
    assert out == "2000"


def test_two_masculine_dva_for_units() -> None:
    # Arrange / Act
    out = words_to_digits("сто два")

    # Assert
    assert out == "102"


# ---------------------------------------------------------------------------
# safe_words_to_digits
# ---------------------------------------------------------------------------


def test_safe_words_to_digits_happy_path_returns_digit_string() -> None:
    # Arrange
    text = "сто двадцать три"

    # Act
    out = safe_words_to_digits(text)

    # Assert
    assert out == "123"


@pytest.mark.parametrize(
    "text,fallback,expected",
    [
        ("сто абракадабра три", "", ""),
        ("сто абракадабра три", "<ERR>", "<ERR>"),
        ("", "EMPTY", "EMPTY"),
        ("foobar", "X", "X"),
    ],
)
def test_safe_words_to_digits_returns_fallback_on_value_error(
    text: str, fallback: str, expected: str
) -> None:
    # Arrange / Act
    out = safe_words_to_digits(text, fallback=fallback)

    # Assert
    assert out == expected


def test_safe_words_to_digits_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Must catch ValueError only — KeyboardInterrupt must propagate."""

    # Arrange
    def _raise_ki(_: str) -> str:
        raise KeyboardInterrupt("simulated")

    monkeypatch.setattr(denorm_module, "words_to_digits", _raise_ki)

    # Act / Assert
    with pytest.raises(KeyboardInterrupt):
        safe_words_to_digits("anything")


# ---------------------------------------------------------------------------
# fsa_constrained_best
# ---------------------------------------------------------------------------


def _make_beam(
    text: str,
    logit_score: float,
) -> tuple[str, object, list[tuple[int, int]], float, float]:
    """Mimic a pyctcdecode beam tuple: (text, last_lm_state, text_frames, logit_score, lm_score)."""
    return (text, None, [], logit_score, 0.0)


def test_fsa_constrained_best_empty_beams_returns_empty_string() -> None:
    # Arrange / Act
    out = fsa_constrained_best([])

    # Assert
    assert out == ""


def test_fsa_constrained_best_all_invalid_parses_returns_empty_string() -> None:
    # Arrange
    beams = [
        _make_beam("foo bar", 0.9),
        _make_beam("abracadabra", 0.5),
    ]

    # Act
    out = fsa_constrained_best(beams)

    # Assert
    assert out == ""


def test_fsa_constrained_best_all_valid_but_out_of_length_range_returns_empty_string() -> (
    None
):
    # Arrange: "сто" -> "100" (3 digits), "один" -> "1" (1 digit) — both outside default (4, 6)
    beams = [
        _make_beam("сто", 0.9),
        _make_beam("один", 0.8),
    ]

    # Act
    out = fsa_constrained_best(beams)

    # Assert
    assert out == ""


def test_fsa_constrained_best_mixed_picks_valid_with_max_logit_score() -> None:
    # Arrange
    beams = [
        _make_beam("foo bar", 100.0),  # invalid parse
        _make_beam("одна тысяча", -2.0),  # "1000" (4 digits, in range)
        _make_beam("одна тысяча пять", 1.0),  # "1005" (4 digits, best score)
        _make_beam("abracadabra", 50.0),  # invalid
        _make_beam("сто двадцать три", 0.0),  # "123" (3 digits, out of range)
    ]

    # Act
    out = fsa_constrained_best(beams)

    # Assert
    assert out == "1005"


def test_fsa_constrained_best_tie_on_logit_score_picks_first_occurrence() -> None:
    # Arrange — both valid, both in range, same score. max() is stable → first wins.
    beams = [
        _make_beam("одна тысяча", 0.5),  # "1000"
        _make_beam("две тысячи", 0.5),  # "2000"
    ]

    # Act
    out = fsa_constrained_best(beams)

    # Assert
    assert out == "1000"


def test_fsa_constrained_best_length_range_lower_bound_inclusive() -> None:
    # Arrange: "1000" has exactly 4 digits; default lower bound is 4.
    beams = [_make_beam("одна тысяча", 0.0)]

    # Act
    out = fsa_constrained_best(beams)

    # Assert
    assert out == "1000"


def test_fsa_constrained_best_length_range_upper_bound_inclusive() -> None:
    # Arrange: "999999" has exactly 6 digits; default upper bound is 6.
    beams = [
        _make_beam(
            "девятьсот девяносто девять тысяч девятьсот девяносто девять",
            0.0,
        )
    ]

    # Act
    out = fsa_constrained_best(beams)

    # Assert
    assert out == "999999"


def test_fsa_constrained_best_custom_length_range_respected() -> None:
    # Arrange: restrict to exactly 3 digits — only "сто двадцать три" -> "123" qualifies.
    beams = [
        _make_beam("одна тысяча", 10.0),  # "1000" — 4 digits, rejected
        _make_beam("сто двадцать три", 1.0),  # "123" — 3 digits, accepted
    ]

    # Act
    out = fsa_constrained_best(beams, length_range=(3, 3))

    # Assert
    assert out == "123"
