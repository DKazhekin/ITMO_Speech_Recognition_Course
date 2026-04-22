"""Tests for BPEVocab — SentencePiece BPE-256 vocabulary for ASR.

TDD: all tests written BEFORE implementation (RED phase).
Uses pytest.importorskip("sentencepiece") to skip if not installed.

The tiny fixture SP model is trained inline on synthetic Russian number
words (num2words output) so the test suite is fully self-contained.

References:
- SentencePiece Python API: https://github.com/google/sentencepiece
- CONTRACTS.md §2: vocab API — encode, decode, blank_id, size / vocab_size
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# Skip entire module if sentencepiece is not installed.
spm = pytest.importorskip("sentencepiece")


# Must be small enough for the tiny corpus (200 lines).
# SentencePiece raises RuntimeError if vocab_size > unique_pieces in corpus.
# 100 is safely within the ~120 unique Cyrillic subword pieces in our corpus.
TINY_VOCAB_SIZE = 100


@pytest.fixture(scope="module")
def tiny_sp_model_path() -> Path:
    """Train a tiny SentencePiece BPE model on synthetic Russian corpus.

    Uses a small range of numbers (1000..1100) converted to Russian words
    via num2words so the corpus contains all basic lexical forms.
    The model is written to a TemporaryDirectory that persists for the
    module scope.
    """
    num2words = pytest.importorskip("num2words")

    tmpdir = tempfile.mkdtemp(prefix="gp1_bpe_test_")
    corpus_path = Path(tmpdir) / "corpus.txt"
    model_prefix = str(Path(tmpdir) / "test_bpe")

    # Build small corpus: 1000..1200 Russian number words, one per line.
    lines: list[str] = []
    for n in range(1000, 1200):
        text: str = num2words.num2words(n, lang="ru")
        # num2words uses hyphens; replace with spaces to keep char vocab clean.
        text = text.replace("-", " ").lower()
        lines.append(text)

    corpus_path.write_text("\n".join(lines), encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=TINY_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=3,
    )
    return Path(model_prefix + ".model")


# ---------------------------------------------------------------------------
# 1. vocab_size matches the trained model
# ---------------------------------------------------------------------------


def test_bpe_vocab_size_matches_model(tiny_sp_model_path):
    """BPEVocab.vocab_size must equal the piece count in the SP model."""
    from gp1.text.vocab_bpe import BPEVocab

    vocab = BPEVocab(tiny_sp_model_path)
    # SP vocab_size includes special tokens trained into the model.
    # We assert it matches the model's own GetPieceSize.
    sp = spm.SentencePieceProcessor()
    sp.load(str(tiny_sp_model_path))
    # BPEVocab shifts ids by +1 (blank=0, SP pieces at 1..N)
    # so vocab_size = sp.GetPieceSize() + 1
    assert vocab.vocab_size == sp.get_piece_size() + 1, (
        f"vocab_size={vocab.vocab_size} != sp.get_piece_size()+1={sp.get_piece_size() + 1}"
    )


# ---------------------------------------------------------------------------
# 2. encode/decode round-trip
# ---------------------------------------------------------------------------


def test_bpe_encode_decode_round_trip(tiny_sp_model_path):
    """A Russian number phrase must round-trip through encode → decode."""
    from gp1.text.vocab_bpe import BPEVocab

    vocab = BPEVocab(tiny_sp_model_path)
    text = "тысяча сто один"

    ids = vocab.encode(text)
    decoded = vocab.decode(ids)

    assert decoded == text, f"Round-trip failed: '{text}' -> {ids} -> '{decoded}'"


# ---------------------------------------------------------------------------
# 3. blank_id is zero
# ---------------------------------------------------------------------------


def test_bpe_blank_id_is_zero(tiny_sp_model_path):
    """BPEVocab.blank_id must be 0 per CONTRACTS.md §2."""
    from gp1.text.vocab_bpe import BPEVocab

    vocab = BPEVocab(tiny_sp_model_path)
    assert vocab.blank_id == 0, f"blank_id={vocab.blank_id}, expected 0"


# ---------------------------------------------------------------------------
# 4. Missing model file raises a clear error
# ---------------------------------------------------------------------------


def test_bpe_missing_model_raises_clear_error():
    """FileNotFoundError or ValueError when model_path does not exist."""
    from gp1.text.vocab_bpe import BPEVocab

    with pytest.raises((FileNotFoundError, ValueError)):
        BPEVocab(Path("/nonexistent/path/to/model.model"))


# ---------------------------------------------------------------------------
# 5. decode strips blank tokens
# ---------------------------------------------------------------------------


def test_bpe_decode_strips_blanks(tiny_sp_model_path):
    """decode() must silently drop id=0 (blank) from the sequence."""
    from gp1.text.vocab_bpe import BPEVocab

    vocab = BPEVocab(tiny_sp_model_path)
    text = "тысяча"
    ids = vocab.encode(text)
    assert 0 not in ids, "encode() must not emit blank ids"

    # Inject blanks and verify they are stripped on decode.
    ids_with_blanks = [0] + ids + [0, 0]
    decoded = vocab.decode(ids_with_blanks)
    assert decoded == text, (
        f"decode with blanks failed: expected '{text}', got '{decoded}'"
    )


# ---------------------------------------------------------------------------
# 6. size property is an alias for vocab_size
# ---------------------------------------------------------------------------


def test_bpe_size_is_alias_for_vocab_size(tiny_sp_model_path):
    """size property must equal vocab_size."""
    from gp1.text.vocab_bpe import BPEVocab

    vocab = BPEVocab(tiny_sp_model_path)
    assert vocab.size == vocab.vocab_size, (
        f"size={vocab.size} != vocab_size={vocab.vocab_size}"
    )


# ---------------------------------------------------------------------------
# 7. encode returns list of positive ints (no blanks, no out-of-range)
# ---------------------------------------------------------------------------


def test_bpe_encode_returns_valid_ids(tiny_sp_model_path):
    """All ids from encode() must be in range [1, vocab_size)."""
    from gp1.text.vocab_bpe import BPEVocab

    vocab = BPEVocab(tiny_sp_model_path)
    text = "двести пятьдесят три"
    ids = vocab.encode(text)

    assert len(ids) > 0, "encode() must return non-empty list for non-empty text"
    for i, token_id in enumerate(ids):
        assert 1 <= token_id < vocab.vocab_size, (
            f"id[{i}]={token_id} out of range [1, {vocab.vocab_size})"
        )
