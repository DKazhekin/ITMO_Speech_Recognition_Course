"""Build a Russian word-form text corpus for KenLM training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from gp1.text.normalize import digits_to_words

log = logging.getLogger(__name__)


def build_synthetic_corpus(
    out_path: Path,
    train_manifest: Path | None = None,
) -> int:
    """Write a text corpus for KenLM training and return total line count.

    Parameters
    ----------
    out_path:
        Destination file.  Parent directory will be created if missing.
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

    # Phase 1: synthetic lines (range 1..999999, deduplicated).
    seen: set[str] = set()
    ordered_lines: list[str] = []

    for n in range(1, 1_000_000):
        word_form = digits_to_words(n)
        if word_form not in seen:
            seen.add(word_form)
            ordered_lines.append(word_form)

    log.info("Synthetic corpus: %d unique word forms", len(ordered_lines))

    # Phase 2: augment with training transcriptions.
    if train_manifest is not None:
        log.info("Augmenting corpus from manifest: %s", train_manifest)
        added = 0
        for lineno, raw_line in enumerate(
            train_manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
                transcription = record["transcription"]
                word_form = digits_to_words(int(transcription))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("Skipping manifest line %d: %s", lineno, exc)
                continue

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
