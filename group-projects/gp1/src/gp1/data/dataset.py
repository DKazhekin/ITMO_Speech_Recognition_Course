"""PyTorch Dataset for the GP1 Russian spoken-numbers ASR pipeline.

Loads audio files via soundfile (avoids torchaudio backend dependency
on torchcodec in torchaudio >= 2.9), resamples to target sample rate via
torchaudio.transforms.Resample, applies optional AudioAugmenter, encodes the
transcription (digit string -> Russian words -> CharVocab ids), and returns
per-sample dicts.

References:
  - soundfile.read: https://python-soundfile.readthedocs.io/
  - torchaudio.transforms.Resample:
      https://pytorch.org/audio/stable/transforms.html#resample
  - CONTRACTS.md §4 "Data pipeline — dataset.py"
"""

from __future__ import annotations

import logging

import soundfile as sf
import torch
import torchaudio.transforms

from gp1.data.audio_aug import AudioAugmenter
from gp1.text.normalize import digits_to_words
from gp1.text.vocab import CharVocab
from gp1.text.vocab_word import WordVocab
from gp1.types import ManifestRecord

log = logging.getLogger(__name__)


class SpokenNumbersDataset(torch.utils.data.Dataset):
    """Dataset of Russian spoken-number utterances for CTC training.

    Each item loads one audio file, resamples it to *target_samplerate*,
    optionally augments it, and encodes the transcription as a sequence of
    character ids via *vocab*.

    Args:
        records: Ordered list of ManifestRecord objects.
        vocab: CharVocab instance for encoding transcriptions.
        target_samplerate: All audio is resampled to this rate (Hz).
        augmenter: Optional AudioAugmenter applied in ``__getitem__``.
        return_two_views: When True, also returns ``audio_view2`` produced
            by a second independent augmenter call (used for CR-CTC loss).
        word_vocab: When provided, each item also exposes ``word_target``
            — a 1-D int64 tensor of word-level token ids — for the
            auxiliary word-level CTC head.
    """

    def __init__(
        self,
        records: list[ManifestRecord],
        vocab: CharVocab,
        target_samplerate: int = 16000,
        augmenter: AudioAugmenter | None = None,
        return_two_views: bool = False,
        word_vocab: WordVocab | None = None,
    ) -> None:
        self._records = list(records)  # defensive copy — immutable inputs
        self._vocab = vocab
        self._target_sr = target_samplerate
        self._augmenter = augmenter
        self._return_two_views = return_two_views
        self._word_vocab = word_vocab

        # Pre-build resamplers keyed by native sample rate.
        # torchaudio.transforms.Resample is stateful but thread-safe for
        # inference; we cache to avoid re-allocating the filter on every call.
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        """Load and process one sample.

        Returns:
            dict with keys:
              - ``audio`` [T_audio] float32 at target_samplerate.
              - ``target`` [U] int64 char ids (no blanks, no pad).
              - ``spk_id`` str.
              - ``transcription`` str (original digit string).
              - ``audio_view2`` [T_audio] float32 (only if return_two_views).
        """
        record = self._records[idx]

        wav = self._load_wav(str(record.audio_path))
        wav = self._resample(wav, record.samplerate)

        # Augment — view 1.
        if self._augmenter is not None:
            wav = self._augmenter(wav, samplerate=self._target_sr)

        # Encode transcription: digit string -> Russian words -> char ids.
        words = digits_to_words(record.transcription)
        target = torch.tensor(self._vocab.encode(words), dtype=torch.int64)

        item: dict = {
            "audio": wav,
            "target": target,
            "spk_id": record.spk_id,
            "transcription": record.transcription,
        }

        if self._word_vocab is not None:
            item["word_target"] = torch.tensor(
                self._word_vocab.encode(words), dtype=torch.int64
            )

        # Augment — view 2 (independent call; augmenter's internal RNG advances).
        if self._return_two_views:
            # Reload clean audio for view 2 so the augmenter sees the original.
            wav2 = self._load_wav(str(record.audio_path))
            wav2 = self._resample(wav2, record.samplerate)
            if self._augmenter is not None:
                wav2 = self._augmenter(wav2, samplerate=self._target_sr)
            item["audio_view2"] = wav2

        return item

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_wav(path: str) -> torch.Tensor:
        """Load a WAV/audio file using soundfile and return a 1-D float32 tensor."""
        data, _ = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return torch.from_numpy(data.copy()).float()

    def _resample(self, wav: torch.Tensor, native_sr: int) -> torch.Tensor:
        """Resample *wav* from *native_sr* to target_samplerate if needed."""
        if native_sr == self._target_sr:
            return wav

        if native_sr not in self._resamplers:
            self._resamplers[native_sr] = torchaudio.transforms.Resample(
                orig_freq=native_sr,
                new_freq=self._target_sr,
            )
            log.debug(
                "SpokenNumbersDataset: created Resample %d -> %d",
                native_sr,
                self._target_sr,
            )

        resampler = self._resamplers[native_sr]
        return resampler(wav.unsqueeze(0)).squeeze(0)
