"""Text denormalization: Russian number words to digit strings.

Contract (CONTRACTS.md §2): deterministic reducer over closed 42-word vocabulary.
Inverse of digits_to_words. Raises ValueError on malformed input.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Word → value lookup tables (shared module-level constants)
# ---------------------------------------------------------------------------

_UNITS: dict[str, int] = {
    "ноль": 0,
    "один": 1,
    "одна": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
}

_TEENS: dict[str, int] = {
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
}

_TENS: dict[str, int] = {
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
}

_HUNDREDS: dict[str, int] = {
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
}

_THOUSANDS: frozenset[str] = frozenset({"тысяча", "тысячи", "тысяч"})

# All known words (for validation)
_ALL_KNOWN: frozenset[str] = frozenset(
    _UNITS.keys() | _TEENS.keys() | _TENS.keys() | _HUNDREDS.keys() | _THOUSANDS
)


def _parse_chunk(tokens: list[str], context: str) -> int:
    """Parse a sequence of Russian number words into an integer value.

    Handles: hundreds + (tens | teens) + units, in that order.
    Does NOT handle thousands — caller splits on thousand markers.

    Args:
        tokens: List of lowercase Russian number words.
        context: Human-readable label for error messages.

    Returns:
        Integer value of the parsed chunk.

    Raises:
        ValueError: If tokens are empty or contain unknown/malformed words.
    """
    if not tokens:
        raise ValueError(f"Empty token sequence in {context}.")

    total = 0
    idx = 0

    # Hundreds
    if idx < len(tokens) and tokens[idx] in _HUNDREDS:
        total += _HUNDREDS[tokens[idx]]
        idx += 1

    # Teens (десять..девятнадцать) or tens (двадцать..девяносто)
    if idx < len(tokens) and tokens[idx] in _TEENS:
        total += _TEENS[tokens[idx]]
        idx += 1
    elif idx < len(tokens) and tokens[idx] in _TENS:
        total += _TENS[tokens[idx]]
        idx += 1
        # Units after tens
        if idx < len(tokens) and tokens[idx] in _UNITS:
            total += _UNITS[tokens[idx]]
            idx += 1
    elif idx < len(tokens) and tokens[idx] in _UNITS:
        total += _UNITS[tokens[idx]]
        idx += 1

    if idx < len(tokens):
        raise ValueError(
            f"Unexpected token {tokens[idx]!r} at position {idx} in {context}. "
            f"Remaining: {tokens[idx:]}"
        )

    return total


def words_to_digits(text: str) -> str:
    """Convert Russian number words to a digit string.

    Inverse of digits_to_words. Handles the range produced by
    num2words(n, lang='ru') for n in 1000..999999.

    Args:
        text: Russian number words (case-insensitive, hyphens treated as spaces).

    Returns:
        Digit string without leading zeros or spaces.
        Example: "сто тридцать девять тысяч четыреста семьдесят три" -> "139473"

    Raises:
        ValueError: On empty input, unknown words, or malformed structure.
    """
    # Normalise input
    normalised = text.lower().replace("-", " ")
    tokens = normalised.split()

    if not tokens:
        raise ValueError("Empty input string. Cannot convert to digits.")

    # Validate all tokens are known
    for token in tokens:
        if token not in _ALL_KNOWN:
            raise ValueError(
                f"Unknown word {token!r}. Expected one of the closed Russian number vocabulary."
            )

    # Find the thousand marker (at most one allowed)
    thousand_positions = [i for i, t in enumerate(tokens) if t in _THOUSANDS]

    if len(thousand_positions) > 1:
        raise ValueError(
            f"Multiple thousand markers found at positions {thousand_positions}. "
            f"Input: {text!r}"
        )

    if not thousand_positions:
        # No thousands: parse the whole sequence as a single chunk
        value = _parse_chunk(tokens, context=f"input {text!r}")
        return str(value)

    thou_pos = thousand_positions[0]
    thousands_tokens = tokens[:thou_pos]
    units_tokens = tokens[thou_pos + 1 :]

    # Must have at least one token before the thousand marker
    if not thousands_tokens:
        raise ValueError(
            f"Thousand marker {tokens[thou_pos]!r} has no numeric head. Input: {text!r}"
        )

    thousands_value = _parse_chunk(
        thousands_tokens, context=f"thousands part of {text!r}"
    )
    thousands_part = thousands_value * 1000

    if units_tokens:
        units_value = _parse_chunk(units_tokens, context=f"units part of {text!r}")
    else:
        units_value = 0

    total = thousands_part + units_value
    return str(total)
