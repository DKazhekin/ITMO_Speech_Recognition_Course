"""Train a KenLM n-gram language model from a text corpus.

Wraps the ``lmplz`` and ``build_binary`` command-line tools that ship with
KenLM (https://github.com/kpu/kenlm).

Typical usage::

    from pathlib import Path
    from gp1.lm.train_kenlm import train_kenlm

    train_kenlm(
        corpus_path=Path("data/lm/corpus.txt"),
        out_binary=Path("data/lm/lm.bin"),
        order=4,
        vocab_limit_path=Path("data/lm/vocab.txt"),
    )

KenLM installation
------------------
- macOS (Homebrew): ``brew install kenlm``
- pip wheel (Linux/macOS): ``pip install kenlm``  — bundles ``lmplz``/``build_binary``.
- Build from source: https://github.com/kpu/kenlm#compiling

SOTA reference
--------------
Pattern for subprocess wrapping adopted from wav2letter lm recipes:
https://github.com/facebookresearch/flashlight/tree/main/recipes/lm_corpus

Bash hygiene note
-----------------
Per project convention stderr and stdout are kept on separate file handles;
``stderr=subprocess.PIPE`` is used for lmplz/build_binary so that error
output is captured and logged without mixing with stdin/stdout.
``check=True`` ensures a non-zero exit code raises ``subprocess.CalledProcessError``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_LMPLZ_NOT_FOUND_MSG = (
    "lmplz not found on PATH — install kenlm "
    "(brew install kenlm or build from source: https://github.com/kpu/kenlm)"
)
_BUILD_BINARY_NOT_FOUND_MSG = (
    "build_binary not found on PATH — it ships with kenlm alongside lmplz; "
    "reinstall kenlm (brew install kenlm or build from source)"
)


def _run_shell_capture(cmd: list[str], *, stdin_path: Path | None = None) -> None:
    """Execute *cmd* via ``subprocess.run``, capturing stderr separately.

    Parameters
    ----------
    cmd:
        Argument list passed directly to ``subprocess.run``.
    stdin_path:
        If given, open this file and pass it as stdin.

    Raises
    ------
    subprocess.CalledProcessError
        When the process exits with a non-zero status.
    """
    stdin_fh = None
    try:
        if stdin_path is not None:
            stdin_fh = stdin_path.open("rb")

        log.debug("Running: %s", " ".join(str(x) for x in cmd))
        subprocess.run(
            cmd,
            stdin=stdin_fh,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    finally:
        if stdin_fh is not None:
            stdin_fh.close()


def train_kenlm(
    corpus_path: Path,
    out_binary: Path,
    order: int = 4,
    vocab_limit_path: Path | None = None,
) -> None:
    """Train a KenLM binary LM from *corpus_path* and write it to *out_binary*.

    Steps
    -----
    1. Run ``lmplz`` to produce an ARPA file in a temporary location.
    2. Run ``build_binary trie <arpa> <out_binary>`` to compile the binary.
    3. Delete the temporary ARPA file.

    Parameters
    ----------
    corpus_path:
        Plain-text corpus, one sentence per line, UTF-8.
    out_binary:
        Destination for the compiled ``.bin`` file.
    order:
        n-gram order (default 4).
    vocab_limit_path:
        If given, passed as ``--limit_vocab_file`` to ``lmplz``.  Only n-grams
        containing words in this file will be kept, producing a much smaller
        model for closed-vocabulary tasks.

    Raises
    ------
    FileNotFoundError
        If ``lmplz`` or ``build_binary`` is not found on PATH.
    subprocess.CalledProcessError
        If either tool exits with a non-zero status.
    """
    _check_binary("lmplz", _LMPLZ_NOT_FOUND_MSG)
    _check_binary("build_binary", _BUILD_BINARY_NOT_FOUND_MSG)

    out_binary.parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary file for the intermediate ARPA so that cleanup is
    # guaranteed even on error (the finally block removes it).
    tmp_arpa = out_binary.parent / f"_tmp_{out_binary.stem}.arpa"

    try:
        _run_lmplz(
            corpus_path=corpus_path,
            arpa_path=tmp_arpa,
            order=order,
            vocab_limit_path=vocab_limit_path,
        )
        _run_build_binary(arpa_path=tmp_arpa, out_binary=out_binary)
        log.info("KenLM binary written: %s", out_binary)
    finally:
        if tmp_arpa.exists():
            tmp_arpa.unlink()
            log.debug("Removed temporary ARPA: %s", tmp_arpa)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_binary(name: str, message: str) -> None:
    """Raise ``FileNotFoundError`` with *message* if *name* is not on PATH."""
    if shutil.which(name) is None:
        raise FileNotFoundError(message)


def _run_lmplz(
    corpus_path: Path,
    arpa_path: Path,
    order: int,
    vocab_limit_path: Path | None,
) -> None:
    """Invoke ``lmplz`` to estimate an n-gram LM and write ARPA to *arpa_path*."""
    cmd: list[str] = [
        shutil.which("lmplz"),  # type: ignore[list-item]
        "-o",
        str(order),
        "--discount_fallback",
    ]

    if vocab_limit_path is not None:
        cmd += ["--limit_vocab_file", str(vocab_limit_path)]

    # lmplz reads corpus from stdin and writes ARPA to stdout.
    # We redirect stdout to the arpa file via subprocess pipe + manual write.
    log.info(
        "Running lmplz -o %d%s < %s",
        order,
        " --limit_vocab_file ..." if vocab_limit_path else "",
        corpus_path,
    )

    with corpus_path.open("rb") as stdin_fh:
        log.debug("Running: %s", " ".join(str(x) for x in cmd))
        result = subprocess.run(
            cmd,
            stdin=stdin_fh,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    arpa_path.write_bytes(result.stdout)


def _run_build_binary(arpa_path: Path, out_binary: Path) -> None:
    """Invoke ``build_binary trie <arpa> <out_binary>``."""
    cmd: list[str] = [
        shutil.which("build_binary"),  # type: ignore[list-item]
        "trie",
        str(arpa_path),
        str(out_binary),
    ]
    log.info("Running build_binary trie %s → %s", arpa_path, out_binary)
    _run_shell_capture(cmd)
