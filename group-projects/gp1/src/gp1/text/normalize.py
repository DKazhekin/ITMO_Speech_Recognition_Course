"""Text normalization: digit strings to Russian number words.

Contract (CONTRACTS.md §2): wraps num2words(n, lang='ru').
Output is lowercase, whitespace-normalized, no hyphens.
Valid range: 1000..999999 inclusive.
"""

from __future__ import annotations

import logging

from num2words import num2words

logger = logging.getLogger(__name__)

_MIN_VALUE: int = 1000
_MAX_VALUE: int = 999_999


def digits_to_words(n: int | str) -> str:
    """Convert an integer (or digit string) to Russian number words.

    Args:
        n: Integer or string representation of an integer in range 1000..999999.

    Returns:
        Lowercase, whitespace-normalized Russian word string without hyphens.
        Example: 139473 -> "сто тридцать девять тысяч четыреста семьдесят три"

    Raises:
        ValueError: If n is out of range 1000..999999 or is not a valid integer.
    """
    if isinstance(n, str):
        try:
            value = int(n)
        except ValueError:
            raise ValueError(
                f"Cannot convert {n!r} to integer. Expected a digit string."
            )
    else:
        value = n

    if value < _MIN_VALUE or value > _MAX_VALUE:
        raise ValueError(f"Value {value} is out of range [{_MIN_VALUE}, {_MAX_VALUE}].")

    raw: str = num2words(value, lang="ru")
    # Normalize: lowercase, replace hyphens with spaces, collapse whitespace
    normalized = raw.lower().replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized
