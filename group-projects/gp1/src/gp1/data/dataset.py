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
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import soundfile as sf
import torch
import torchaudio.transforms

from gp1.data.audio_aug import AudioAugmenter
from gp1.text.normalize import digits_to_words
from gp1.types import ManifestRecord


def preload_audio_cache(
    records: list[ManifestRecord], target_samplerate: int = 16000
) -> dict[str, torch.Tensor]:
    """Pre-load and resample all audio files into a RAM cache.

    One-time cost (minutes); subsequent Dataset iterations skip disk I/O
    and Resample entirely. Typical total size: ~4 bytes * 16000 * mean_duration
    * len(records). For GP1 (~15k files x 2.3s avg x 16kHz): ~2 GB.
    """
    from tqdm import tqdm

    cache: dict[str, torch.Tensor] = {}
    resamplers: dict[int, torchaudio.transforms.Resample] = {}
    for rec in tqdm(records, desc="preload audio"):
        path_str = str(rec.audio_path)
        if path_str in cache:
            continue
        data, sr = sf.read(path_str, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data.copy()).float()
        if sr != target_samplerate:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=target_samplerate
                )
            wav = resamplers[sr](wav.unsqueeze(0)).squeeze(0)
        cache[path_str] = wav
    total_bytes = sum(w.numel() * 4 for w in cache.values())
    log.info(
        "preload_audio_cache: %d tensors, %.2f GB RAM",
        len(cache),
        total_bytes / 1e9,
    )
    return cache


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
        vocab: Any object satisfying ``VocabProtocol`` (e.g. ``CharVocab``, ``BPEVocab``).
        target_samplerate: All audio is resampled to this rate (Hz).
        augmenter: Optional AudioAugmenter applied in ``__getitem__``.
        audio_cache: Optional {str(audio_path) -> Tensor} dict from
            ``preload_audio_cache``. When provided, skips disk read and resample.
    """

    def __init__(
        self,
        records: list[ManifestRecord],
        vocab: VocabProtocol,
        target_samplerate: int = 16000,
        augmenter: AudioAugmenter | None = None,
        audio_cache: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self._records = list(records)  # defensive copy — immutable inputs
        self._vocab = vocab
        self._target_sr = target_samplerate
        self._augmenter = augmenter
        self._audio_cache = audio_cache

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
        """
        record = self._records[idx]

        path_str = str(record.audio_path)
        if self._audio_cache is not None and path_str in self._audio_cache:
            wav = self._audio_cache[path_str]
        else:
            wav = self._resample(self._load_wav(path_str), record.samplerate)

        # Augment.
        if self._augmenter is not None:
            wav = self._augmenter(wav, samplerate=self._target_sr)

        # Encode transcription: digit string -> Russian words -> char ids.
        words = digits_to_words(record.transcription)
        target = torch.tensor(self._vocab.encode(words), dtype=torch.int64)

        return {
            "audio": wav,
            "target": target,
            "spk_id": record.spk_id,
            "transcription": record.transcription,
        }

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
