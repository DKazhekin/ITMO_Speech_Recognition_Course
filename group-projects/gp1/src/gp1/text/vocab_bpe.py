"""BPE vocabulary backed by a SentencePiece model for Russian ASR.

Wraps a trained ``*.model`` file and exposes the same interface as
``CharVocab`` / ``WordVocab`` (CONTRACTS.md §2):

    encode(text: str)  -> list[int]   — text to shifted SP piece ids
    decode(ids: list[int]) -> str     — ids back to text (blank=0 stripped)
    blank_id  : int = 0
    vocab_size : int = sp.get_piece_size() + 1   (1-indexed, 0 reserved for blank)
    size       : int  alias for vocab_size

Id space
--------
    0           = CTC blank (never emitted by encode, silently dropped by decode)
    1 .. N      = SentencePiece piece ids shifted +1 from SP's native 0-based range

The shift is intentional: the CTC blank (id=0) must not overlap any real
subword piece, and SentencePiece uses 0 for its own <unk> token.  Shifting
by +1 keeps the blank at index 0 consistent with CharVocab and WordVocab.

Optional helper
---------------
``train_bpe_model(corpus_path, model_prefix, vocab_size)`` wraps
``sentencepiece.SentencePieceTrainer.train`` for reproducible offline
training of BPE models on the GP1 Russian number-word corpus.

SentencePiece lazy import
-------------------------
``sentencepiece`` is not in the project's hard requirements.  The module-level
import is deferred to the first use of ``BPEVocab`` so that the rest of the
codebase can import this module without triggering an ImportError when SP is
absent.  Tests that need SP are gated with ``pytest.importorskip("sentencepiece")``.

References
----------
- SentencePiece Python API: https://github.com/google/sentencepiece
- BPE subword tokenization: Sennrich et al. (2016), https://arxiv.org/abs/1508.07909
- CONTRACTS.md §2: vocab API contract
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _import_sentencepiece():
    """Lazy import of sentencepiece; raises ImportError with a clear message."""
    try:
        import sentencepiece as spm  # type: ignore[import]

        return spm
    except ImportError as exc:
        raise ImportError(
            "sentencepiece is required for BPEVocab but is not installed. "
            "Install it with: uv pip install sentencepiece"
        ) from exc


class BPEVocab:
    """BPE vocabulary loaded from a SentencePiece ``*.model`` file.

    Parameters
    ----------
    model_path : Path
        Path to a trained SentencePiece binary model file (``*.model``).
        Raises ``FileNotFoundError`` if the file does not exist.
        Raises ``ImportError`` if ``sentencepiece`` is not installed.

    Attributes
    ----------
    blank_id   : int = 0  (CTC blank, never emitted, stripped on decode)
    vocab_size : int      sp.get_piece_size() + 1
    size       : int      alias for vocab_size
    """

    BLANK_ID: int = 0

    def __init__(self, model_path: Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {model_path}. "
                "Train a model first with vocab_bpe.train_bpe_model() "
                "or provide a pre-trained *.model file."
            )

        spm = _import_sentencepiece()
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(model_path))
        self._model_path = model_path

        # vocab_size: reserve index 0 for CTC blank; SP pieces occupy 1..N.
        self._vocab_size: int = self._sp.get_piece_size() + 1

        log.info(
            "BPEVocab loaded: model=%s, vocab_size=%d (incl. blank)",
            model_path,
            self._vocab_size,
        )

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def blank_id(self) -> int:
        """CTC blank token id (always 0)."""
        return self.BLANK_ID

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including blank at index 0."""
        return self._vocab_size

    @property
    def size(self) -> int:
        """Alias for ``vocab_size`` (mirrors CharVocab.size)."""
        return self._vocab_size

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of shifted SP piece ids.

        Parameters
        ----------
        text : str
            Input text (Russian number words, lowercase, space-separated).

        Returns
        -------
        list[int]
            Piece ids in range ``[1, vocab_size)``.  Never contains 0.
            Empty list for empty input.
        """
        if not text:
            return []
        # sp.encode returns 0-based ids; shift by +1 to reserve 0 for blank.
        native_ids: list[int] = self._sp.encode(text, out_type=int)
        return [idx + 1 for idx in native_ids]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of shifted piece ids back to text.

        Blank tokens (id=0) are silently dropped before decoding.

        Parameters
        ----------
        ids : list[int]
            Sequence of piece ids (may include blanks).

        Returns
        -------
        str
            Decoded text string.
        """
        if not ids:
            return ""
        # Strip blanks, then un-shift back to SP native ids.
        native_ids = [idx - 1 for idx in ids if idx != self.BLANK_ID]
        if not native_ids:
            return ""
        return self._sp.decode(native_ids)


# ---------------------------------------------------------------------------
# Optional training helper
# ---------------------------------------------------------------------------


def train_bpe_model(
    corpus_path: Path,
    model_prefix: str,
    vocab_size: int = 256,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
) -> Path:
    """Train a SentencePiece BPE model and return the path to the model file.

    Parameters
    ----------
    corpus_path : Path
        Plain-text file with one utterance per line (UTF-8).
    model_prefix : str
        Output model prefix; the trainer writes ``<prefix>.model`` and
        ``<prefix>.vocab``.
    vocab_size : int
        Number of BPE pieces.  Default 256 matches the GP1 BPE-256 baseline.
    model_type : str
        SentencePiece model type (``"bpe"`` or ``"unigram"``).
    character_coverage : float
        Character coverage for the SP model.  Use 1.0 for Russian so that
        all Cyrillic characters are guaranteed in-vocabulary.

    Returns
    -------
    Path
        Path to the written ``*.model`` file.

    References
    ----------
    - SentencePiece training: https://github.com/google/sentencepiece#train-sentencepiece-model
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    spm = _import_sentencepiece()
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=3,
    )
    model_file = Path(f"{model_prefix}.model")
    log.info("BPE model trained: %s (vocab_size=%d)", model_file, vocab_size)
    return model_file
