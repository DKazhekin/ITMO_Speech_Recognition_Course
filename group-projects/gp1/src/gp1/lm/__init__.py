"""gp1.lm — language model utilities.

Provides:
- build_corpus.build_synthetic_corpus: generates a num2words-based text
  corpus for KenLM training from the full Russian spoken-numbers range
  (1 000 .. 999 999).
- train_kenlm.train_kenlm: wraps the ``lmplz`` and ``build_binary`` CLI
  tools shipped with KenLM to train a 4-gram binary LM.

KenLM availability
------------------
The ``kenlm`` Python wheel (https://pypi.org/project/kenlm/) includes the
``lmplz`` binary on Linux x86-64 and macOS arm64 (via Homebrew).  On other
platforms users may need to build KenLM from source::

    brew install kenlm          # macOS
    # or
    git clone https://github.com/kpu/kenlm && cmake -DCMAKE_BUILD_TYPE=Release ..

See https://github.com/kpu/kenlm for authoritative instructions.
"""

from gp1.lm.build_corpus import build_synthetic_corpus
from gp1.lm.train_kenlm import train_kenlm

__all__ = ["build_synthetic_corpus", "train_kenlm"]
