"""Character-level vocabulary for Russian ASR.

Contract (CONTRACTS.md §2):
  - BLANK_ID = 0
  - SPACE_ID = 1
  - 33 Russian lowercase letters occupy ids 2..34
  - vocab_size = 35
  - encode raises ValueError on unknown chars
  - decode performs CTC-collapse (consecutive duplicates → one) then strips blanks
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

RUSSIAN_ALPHABET_LOWER: str = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"  # 33 letters


class CharVocab:
    """Immutable character vocabulary for Russian number ASR."""

    BLANK_ID: int = 0
    SPACE_ID: int = 1
    vocab_size: int = 35  # blank + space + 33 letters

    def __init__(self) -> None:
        # ids: 0=blank, 1=space, 2..34=letters
        self._char_to_id: dict[str, int] = {
            ch: idx + 2 for idx, ch in enumerate(RUSSIAN_ALPHABET_LOWER)
        }
        self._id_to_char: dict[int, str] = {v: k for k, v in self._char_to_id.items()}

    @property
    def blank_id(self) -> int:
        return self.BLANK_ID

    @property
    def size(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode a Russian text string to a list of token ids.

        Args:
            text: Lowercase Russian string (letters and spaces only).

        Returns:
            List of integer token ids. Empty list for empty string.

        Raises:
            ValueError: If any character is not in the vocabulary.
        """
        if not text:
            return []

        result: list[int] = []
        for ch in text:
            if ch == " ":
                result.append(self.SPACE_ID)
            elif ch in self._char_to_id:
                result.append(self._char_to_id[ch])
            else:
                raise ValueError(
                    f"Character {ch!r} is not in the vocabulary. "
                    f"Expected Russian lowercase letters and spaces."
                )
        return result

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids to a string with CTC-collapse.

        CTC-collapse: consecutive duplicate ids are merged into one,
        then blank ids (0) are removed.

        Args:
            ids: List of integer token ids (may include blanks and duplicates).

        Returns:
            Decoded Russian string.
        """
        if not ids:
            return ""

        # CTC-collapse: merge consecutive duplicates
        collapsed: list[int] = []
        prev: int | None = None
        for token_id in ids:
            if token_id != prev:
                collapsed.append(token_id)
                prev = token_id

        # Strip blanks
        chars: list[str] = []
        for token_id in collapsed:
            if token_id == self.BLANK_ID:
                continue
            if token_id == self.SPACE_ID:
                chars.append(" ")
            else:
                chars.append(self._id_to_char[token_id])

        return "".join(chars)
