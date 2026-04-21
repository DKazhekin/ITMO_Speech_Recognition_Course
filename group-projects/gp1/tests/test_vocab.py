"""Tests for CharVocab — Russian char-level vocab with blank + space + 33 letters."""

from __future__ import annotations

import pytest

from gp1.text.vocab import RUSSIAN_ALPHABET_LOWER, CharVocab


def test_vocab_size_is_35_per_contract() -> None:
    # Arrange
    vocab = CharVocab()

    # Act / Assert
    assert vocab.size == 35
    assert CharVocab.vocab_size == 35


def test_blank_id_is_zero() -> None:
    # Arrange
    vocab = CharVocab()

    # Act / Assert
    assert vocab.blank_id == 0
    assert CharVocab.BLANK_ID == 0


def test_space_id_is_one() -> None:
    # Arrange / Act / Assert
    assert CharVocab.SPACE_ID == 1


def test_russian_alphabet_has_33_letters() -> None:
    # Arrange / Act / Assert
    assert len(RUSSIAN_ALPHABET_LOWER) == 33
    # ё is in the alphabet
    assert "ё" in RUSSIAN_ALPHABET_LOWER


def test_encode_decode_round_trip_on_sto_dvadtsat_tri() -> None:
    # Arrange
    vocab = CharVocab()
    text = "сто двадцать три"

    # Act
    ids = vocab.encode(text)
    decoded = vocab.decode(ids)

    # Assert
    assert decoded == text
    assert all(0 < i < vocab.size for i in ids)


def test_encode_space_maps_to_space_id() -> None:
    # Arrange
    vocab = CharVocab()

    # Act
    ids = vocab.encode("а б")

    # Assert
    assert ids[1] == CharVocab.SPACE_ID


def test_decode_strips_blanks() -> None:
    # Arrange
    vocab = CharVocab()
    ids_clean = vocab.encode("да")
    ids_with_blanks = [CharVocab.BLANK_ID] + ids_clean + [CharVocab.BLANK_ID, CharVocab.BLANK_ID]

    # Act
    decoded = vocab.decode(ids_with_blanks)

    # Assert
    assert decoded == "да"


def test_encode_rejects_unknown_character() -> None:
    # Arrange
    vocab = CharVocab()

    # Act / Assert
    with pytest.raises(ValueError):
        vocab.encode("hello")  # Latin letters are out-of-vocab


def test_encode_empty_string_returns_empty_list() -> None:
    # Arrange
    vocab = CharVocab()

    # Act
    ids = vocab.encode("")

    # Assert
    assert ids == []


def test_decode_empty_list_returns_empty_string() -> None:
    # Arrange
    vocab = CharVocab()

    # Act
    text = vocab.decode([])

    # Assert
    assert text == ""


def test_encode_all_russian_letters_produces_unique_ids() -> None:
    # Arrange
    vocab = CharVocab()

    # Act
    ids = vocab.encode(RUSSIAN_ALPHABET_LOWER)

    # Assert
    assert len(set(ids)) == 33
    assert CharVocab.BLANK_ID not in ids
    assert CharVocab.SPACE_ID not in ids
