"""Build a text corpus for KenLM training.

The corpus covers every Russian spoken-number form in the range 1 000..999 999
via ``num2words``, optionally augmented with word-forms extracted from a
training manifest (JSONL, same schema as ``gp1.data.manifest``).

SOTA reference
--------------
A common pattern for building KenLM corpora from num2words is used in e.g.
wav2letter / flashlight recipes:
https://github.com/facebookresearch/flashlight/tree/main/recipes/lm_corpus

Algorithm
---------
1. For each n in range(1000, 1_000_000), call ``digits_to_words(n)`` from
   ``gp1.text.normalize`` (which wraps ``num2words(n, lang='ru')``).
   Collect the result in a ``set[str]`` to allow O(1) deduplication.
2. If ``train_manifest`` is given, read each JSON line, extract
   ``transcription`` (a digit string), convert via ``digits_to_words``,
   and add to the set only if not already present.
3. Write the set to ``out_path`` in insertion order (synthetic first, then
   novel manifest lines), one line per entry.
4. Return total line count.

Why ``set`` + ordered writes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using a ``set`` for deduplication and writing in a deterministic pass keeps
memory usage bounded (the set holds ~999 k unique strings, each roughly
30-60 bytes  → ≈30-60 MB) while preventing duplicate lines that would inflate
KenLM unigram counts and distort perplexity estimates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency: digits_to_words from W1-A (gp1.text.normalize).
# We attempt the import at module level and surface a clear message if the
# sibling module is not yet on disk (parallel Wave-1 agents).
# ---------------------------------------------------------------------------
try:
    from gp1.text.normalize import digits_to_words as _digits_to_words

    _NORMALIZE_AVAILABLE = True
except ImportError:
    _NORMALIZE_AVAILABLE = False
    _digits_to_words = None  # type: ignore[assignment]

# Also guard num2words directly, used as the fallback below.
from importlib.util import find_spec

_NUM2WORDS_AVAILABLE = find_spec("num2words") is not None


def _convert(n: int) -> str:
    """Convert integer to lowercase, whitespace-normalised Russian word form.

    Prefers ``gp1.text.normalize.digits_to_words`` (W1-A).  Falls back to a
    minimal inline implementation using ``num2words`` directly so that this
    module stays functional even while W1-A is in progress.

    Raises ``ImportError`` if neither dependency is available.
    """
    if _NORMALIZE_AVAILABLE and _digits_to_words is not None:
        return _digits_to_words(n)

    if _NUM2WORDS_AVAILABLE:
        import num2words

        raw: str = num2words.num2words(int(n), lang="ru")
        # num2words uses hyphens for compound words; replace with spaces to
        # match the digits_to_words contract (no hyphens).
        normalised = raw.replace("-", " ").lower()
        # Collapse multiple spaces.
        return " ".join(normalised.split())

    raise ImportError(
        "Neither gp1.text.normalize nor num2words is available. "
        "Install via `uv pip install num2words` or wait for W1-A to land."
    )


def build_synthetic_corpus(
    out_path: Path,
    train_manifest: Path | None = None,
) -> int:
    """Write a text corpus for KenLM training and return total line count.

    Parameters
    ----------
    out_path:
        Destination file.  Parent directory must exist.
    train_manifest:
        Optional path to a JSONL manifest (one JSON object per line).  Each
        record must have a ``"transcription"`` field containing a digit string
        (e.g. ``"139473"``).  Unique word-forms not already present in the
        synthetic corpus will be appended.

    Returns
    -------
    int
        Total number of lines written to ``out_path``.
    """
    log.info("Building synthetic corpus → %s", out_path)

    # Phase 1: synthetic lines (range 1000..999999, deduplicated by set).
    seen: set[str] = set()
    ordered_lines: list[str] = []

    for n in range(1000, 1_000_000):
        word_form = _convert(n)
        if word_form not in seen:
            seen.add(word_form)
            ordered_lines.append(word_form)

    log.info("Synthetic corpus: %d unique word forms", len(ordered_lines))

    # Phase 2: augment with training transcriptions.
    if train_manifest is not None:
        log.info("Augmenting corpus from manifest: %s", train_manifest)
        manifest_lines = _load_manifest_word_forms(train_manifest)
        added = 0
        for word_form in manifest_lines:
            if word_form not in seen:
                seen.add(word_form)
                ordered_lines.append(word_form)
                added += 1
        log.info("Added %d novel word forms from manifest", added)

    # Phase 3: write to disk.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ordered_lines) + "\n", encoding="utf-8")

    total = len(ordered_lines)
    log.info("Corpus written: %d lines → %s", total, out_path)
    return total


def _load_manifest_word_forms(manifest_path: Path) -> list[str]:
    """Read manifest JSONL and return deduplicated word forms in order of first occurrence.

    Each line must be a JSON object with a ``"transcription"`` field
    (digit string, e.g. ``"500000"``).  Lines that cannot be parsed are
    skipped with a warning.
    """
    word_forms: list[str] = []
    seen_local: set[str] = set()

    for lineno, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            log.warning("Skipping manifest line %d (JSON error): %s", lineno, exc)
            continue

        transcription: str | None = record.get("transcription")
        if transcription is None:
            log.warning("Manifest line %d has no 'transcription' field", lineno)
            continue

        try:
            word_form = _convert(int(transcription))
        except (ValueError, ImportError) as exc:
            log.warning(
                "Cannot convert transcription %r on manifest line %d: %s",
                transcription,
                lineno,
                exc,
            )
            continue

        if word_form not in seen_local:
            seen_local.add(word_form)
            word_forms.append(word_form)

    return word_forms
