"""Tests for WordVocab — closed Russian number-word vocabulary."""

from __future__ import annotations

import pytest

from gp1.text.vocab_word import NUMBER_WORDS, WordVocab


def test_every_number_word_has_unique_id() -> None:
    # Arrange
    vocab = WordVocab()

    # Act
    ids = [vocab.encode(word)[0] for word in NUMBER_WORDS]

    # Assert
    assert len(set(ids)) == len(NUMBER_WORDS)
    assert WordVocab.BLANK_ID not in ids


def test_blank_id_is_zero() -> None:
    # Arrange / Act / Assert
    assert WordVocab.BLANK_ID == 0


def test_vocab_size_is_blank_plus_number_words() -> None:
    # Arrange
    vocab = WordVocab()

    # Act / Assert
    assert vocab.size == 1 + len(NUMBER_WORDS)


def test_encode_oov_raises_value_error() -> None:
    # Arrange
    vocab = WordVocab()

    # Act / Assert
    with pytest.raises(ValueError):
        vocab.encode("абракадабра")


def test_encode_multiword_transcription() -> None:
    # Arrange
    vocab = WordVocab()

    # Act
    ids = vocab.encode("сто двадцать три")

    # Assert
    assert len(ids) == 3
    assert all(i != WordVocab.BLANK_ID for i in ids)


def test_encode_decode_round_trip() -> None:
    # Arrange
    vocab = WordVocab()
    text = "одна тысяча пять"

    # Act
    ids = vocab.encode(text)
    decoded = vocab.decode(ids)

    # Assert
    assert decoded == text


def test_decode_strips_blanks() -> None:
    # Arrange
    vocab = WordVocab()
    ids = vocab.encode("две тысячи")
    ids_with_blanks = [WordVocab.BLANK_ID] + ids + [WordVocab.BLANK_ID]

    # Act
    decoded = vocab.decode(ids_with_blanks)

    # Assert
    assert decoded == "две тысячи"


def test_number_words_is_exhaustive_for_1000_to_999999() -> None:
    """Every word emitted by num2words in the target range must be in NUMBER_WORDS."""
    # Arrange
    from num2words import num2words

    emitted: set[str] = set()
    for n in range(1000, 1_000_000):
        for w in num2words(n, lang="ru").replace("-", " ").split():
            emitted.add(w)

    # Act / Assert
    missing = emitted - set(NUMBER_WORDS)
    assert not missing, f"missing words in NUMBER_WORDS: {missing}"


def test_encode_empty_string_returns_empty_list() -> None:
    # Arrange
    vocab = WordVocab()

    # Act
    ids = vocab.encode("")

    # Assert
    assert ids == []


def test_decode_empty_list_returns_empty_string() -> None:
    # Arrange
    vocab = WordVocab()

    # Act
    text = vocab.decode([])

    # Assert
    assert text == ""
