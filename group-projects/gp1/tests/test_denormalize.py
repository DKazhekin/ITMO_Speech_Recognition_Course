"""Tests for words_to_digits — deterministic reducer over Russian number words."""

from __future__ import annotations

import pytest

from gp1.text.denormalize import words_to_digits


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
