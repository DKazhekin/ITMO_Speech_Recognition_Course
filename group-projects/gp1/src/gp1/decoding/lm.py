"""KenLM language model wrapper.

Lazily imports the ``kenlm`` C-extension so that the rest of the codebase
can be imported on machines where kenlm is not installed (tests will skip
via pytest.importorskip).

References:
  - CONTRACTS.md §7: KenLMWrapper signature
  - KenLM documentation: https://github.com/kpu/kenlm
    kenlm.Model.score(text, bos, eos) returns log10 probability.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class KenLMWrapper:
    """Thin wrapper around a compiled KenLM binary (.bin) model.

    The kenlm native extension is imported lazily inside __init__ so that
    modules that import KenLMWrapper can still be loaded on systems without
    the extension (tests mock it via unittest.mock.patch).

    Args:
        binary_path: Path to the compiled KenLM binary (``*.bin``).

    Raises:
        FileNotFoundError: If binary_path does not exist on disk, with a
            message that includes the full path for easy debugging.
    """

    def __init__(self, binary_path: Path) -> None:
        if not binary_path.exists():
            raise FileNotFoundError(
                f"KenLM binary not found: {binary_path}. "
                "Build it with: lmplz -o 4 | build_binary trie lm.arpa lm.bin"
            )

        import kenlm  # lazy import — not available in all envs

        self._model = kenlm.Model(str(binary_path))
        log.info("KenLMWrapper: loaded model from %s", binary_path)

    def score(self, text: str, bos: bool = True, eos: bool = True) -> float:
        """Score a text string with the KenLM model.

        Args:
            text: Whitespace-tokenised text string.
            bos: Whether to prepend a <s> token (begin-of-sentence context).
            eos: Whether to append a </s> token (end-of-sentence normalisation).

        Returns:
            Log10 probability of the text under the language model.
            (KenLM natively returns log10 scores.)
        """
        return self._model.score(text, bos=bos, eos=eos)
