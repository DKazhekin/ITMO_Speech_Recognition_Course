"""Text normalization: digit strings to Russian number words.

Wraps num2words(n, lang='ru'). Output is lowercase, whitespace-normalized,
no hyphens. Accepts any non-negative integer supported by num2words.
"""

from __future__ import annotations

import logging

from num2words import num2words

logger = logging.getLogger(__name__)


def digits_to_words(n: int | str) -> str:
    """Convert a non-negative integer (or digit string) to Russian number words.

    Example: 139473 -> "сто тридцать девять тысяч четыреста семьдесят три"

    Raises ValueError on malformed input or negative values.
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

    if value < 0:
        raise ValueError(f"Value {value} must be non-negative.")

    raw: str = num2words(value, lang="ru")
    normalized = raw.lower().replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized
