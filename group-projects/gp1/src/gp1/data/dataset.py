"""PyTorch Dataset for the GP1 Russian spoken-numbers ASR pipeline.

Loads audio files via soundfile (avoids torchaudio backend dependency
on torchcodec in torchaudio >= 2.9), resamples to target sample rate via
torchaudio.transforms.Resample, applies optional AudioAugmenter, encodes the
transcription (digit string -> Russian words -> CharVocab ids), and returns
per-sample dicts.

Optional audio cache (Phase B speed-up):
  Pass ``audio_cache_dir`` to enable the pre-resampled WAV cache.  When a
  cached file is found the raw soundfile decode + torchaudio resample are
  skipped entirely — only one fast sf.read of a smaller int16 WAV is done.

References:
  - soundfile.read: https://python-soundfile.readthedocs.io/
  - torchaudio.transforms.Resample:
      https://pytorch.org/audio/stable/transforms.html#resample
  - CONTRACTS.md §4 "Data pipeline — dataset.py"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import soundfile as sf
import torch
import torchaudio.transforms

from gp1.data.audio_aug import AudioAugmenter
from gp1.text.normalize import digits_to_words
from gp1.text.vocab import CharVocab
from gp1.text.vocab_word import WordVocab
from gp1.types import ManifestRecord

if TYPE_CHECKING:
    # BPEVocab is an optional dep (sentencepiece); import only for type-checking.
    from gp1.text.vocab_bpe import BPEVocab


@runtime_checkable
class VocabProtocol(Protocol):
    """Structural interface required by SpokenNumbersDataset.

    Any vocabulary object must expose:
    - ``encode(text: str) -> list[int]``   — text to token ids
    - ``vocab_size: int``                  — total vocabulary size
    - ``blank_id: int``                    — CTC blank token id (always 0)
    """

    def encode(self, text: str) -> list[int]: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def blank_id(self) -> int: ...


log = logging.getLogger(__name__)


class SpokenNumbersDataset(torch.utils.data.Dataset):
    """Dataset of Russian spoken-number utterances for CTC training.

    Each item loads one audio file, resamples it to *target_samplerate*,
    optionally augments it, and encodes the transcription as a sequence of
    character ids via *vocab*.

    Args:
        records: Ordered list of ManifestRecord objects.
        vocab: Vocabulary for encoding transcriptions.  Accepts any object
            satisfying ``VocabProtocol`` (``CharVocab``, ``BPEVocab``, or
            ``WordVocab``).  Duck-typed at runtime; the annotation broadens the
            formerly ``CharVocab``-only hint to reflect actual usage.
        target_samplerate: All audio is resampled to this rate (Hz).
        augmenter: Optional AudioAugmenter applied in ``__getitem__``.
        return_two_views: When True, also returns ``audio_view2`` produced
            by a second independent augmenter call (used for CR-CTC loss).
        word_vocab: When provided, each item also exposes ``word_target``
            — a 1-D int64 tensor of word-level token ids — for the
            auxiliary word-level CTC head.
        audio_cache_dir: Optional path to a pre-resampled 16-kHz PCM_16 WAV
            cache produced by ``scripts/precompute_audio.py``.  When a cached
            file for a record is found (matched by filename stem), the source
            soundfile decode and torchaudio resample are skipped entirely.
            Falls back to the uncached path on cache miss.
    """

    def __init__(
        self,
        records: list[ManifestRecord],
        vocab: "CharVocab | BPEVocab | VocabProtocol",
        target_samplerate: int = 16000,
        augmenter: AudioAugmenter | None = None,
        return_two_views: bool = False,
        word_vocab: WordVocab | None = None,
        audio_cache_dir: Path | None = None,
    ) -> None:
        self._records = list(records)  # defensive copy — immutable inputs
        self._vocab = vocab
        self._target_sr = target_samplerate
        self._augmenter = augmenter
        self._return_two_views = return_two_views
        self._word_vocab = word_vocab
        self._audio_cache_dir = (
            Path(audio_cache_dir) if audio_cache_dir is not None else None
        )
        # Guards one-shot WARNING log if cache is set but misses keep happening.
        self._cache_miss_warned: bool = False

        # Pre-build resamplers keyed by native sample rate.
        # torchaudio.transforms.Resample is stateful but thread-safe for
        # inference; we cache to avoid re-allocating the filter on every call.
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

        if self._audio_cache_dir is not None:
            _cached_count = sum(
                1
                for r in self._records
                if (
                    self._audio_cache_dir / Path(r.audio_path.name).with_suffix(".wav")
                ).exists()
            )
            log.info(
                "SpokenNumbersDataset: audio_cache_dir=%s, %d/%d files cached",
                self._audio_cache_dir,
                _cached_count,
                len(self._records),
            )

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

        wav = self._load_audio_maybe_cached(record)

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
            wav2 = self._load_audio_maybe_cached(record)
            if self._augmenter is not None:
                wav2 = self._augmenter(wav2, samplerate=self._target_sr)
            item["audio_view2"] = wav2

        return item

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_audio_maybe_cached(self, record: ManifestRecord) -> torch.Tensor:
        """Return waveform, reading from cache when available.

        Cache hit path: one sf.read of a small PCM_16 WAV at target_samplerate
        — no resample step needed.

        Cache miss / no cache: fall through to ``_load_and_resample_from_source``.
        """
        if self._audio_cache_dir is not None:
            cached = self._audio_cache_dir / Path(record.audio_path.name).with_suffix(
                ".wav"
            )
            if cached.exists():
                data, _ = sf.read(str(cached), dtype="float32", always_2d=False)
                if data.ndim == 2:
                    data = data.mean(axis=1)
                return torch.from_numpy(data.copy()).float()
            # Warn once on first cache miss so an incomplete or mismatched
            # cache does not silently degrade training to the slow path.
            if not self._cache_miss_warned:
                log.warning(
                    "audio_cache_dir set but cache miss for %s — falling back to "
                    "source load+resample. Run scripts/precompute_audio.py to "
                    "populate the cache, or unset --audio-cache-dir.",
                    cached,
                )
                self._cache_miss_warned = True
        return self._load_and_resample_from_source(record)

    def _load_and_resample_from_source(self, record: ManifestRecord) -> torch.Tensor:
        """Load from the original source file and resample to target_samplerate."""
        wav = self._load_wav(str(record.audio_path))
        return self._resample(wav, record.samplerate)

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
