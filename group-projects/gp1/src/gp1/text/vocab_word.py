"""Word-level vocabulary for Russian number ASR.

Contract (CONTRACTS.md §2):
  - Closed vocabulary covering all words emitted by num2words(n, lang='ru')
    for n in range(1000, 1_000_000).
  - BLANK_ID = 0
  - vocab_size = 1 + len(NUMBER_WORDS)
  - encode raises ValueError on out-of-vocabulary words
  - decode strips blank ids
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Exhaustive set of Russian number words for range 1000..999999.
# Verified via: num2words(n, lang='ru') for all n in range(1000, 1_000_000).
# Includes "ноль" for completeness (not emitted in target range but in contract).
NUMBER_WORDS: tuple[str, ...] = (
    "ноль",
    "один",
    "одна",
    "два",
    "две",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
    "тысяча",
    "тысячи",
    "тысяч",
)

_WORD_TO_ID: dict[str, int] = {word: idx + 1 for idx, word in enumerate(NUMBER_WORDS)}
_ID_TO_WORD: dict[int, str] = {v: k for k, v in _WORD_TO_ID.items()}


class WordVocab:
    """Closed word vocabulary for Russian number words.

    Token ids:
      0 = blank
      1..len(NUMBER_WORDS) = number words in NUMBER_WORDS order
    """

    BLANK_ID: int = 0

    @property
    def size(self) -> int:
        return 1 + len(NUMBER_WORDS)

    def encode(self, text: str) -> list[int]:
        """Encode a space-separated sequence of number words to token ids.

        Args:
            text: Space-separated Russian number words.

        Returns:
            List of token ids. Empty list for empty string.

        Raises:
            ValueError: If any word is not in NUMBER_WORDS.
        """
        if not text:
            return []

        result: list[int] = []
        for word in text.split():
            if word not in _WORD_TO_ID:
                raise ValueError(
                    f"Word {word!r} is not in the closed vocabulary. "
                    f"Expected one of the {len(NUMBER_WORDS)} Russian number words."
                )
            result.append(_WORD_TO_ID[word])
        return result

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids to a space-separated word string.

        Blank ids (0) are stripped.

        Args:
            ids: List of token ids (may include blanks).

        Returns:
            Space-separated Russian number words.
        """
        if not ids:
            return ""

        words: list[str] = []
        for token_id in ids:
            if token_id == self.BLANK_ID:
                continue
            words.append(_ID_TO_WORD[token_id])

        return " ".join(words)
