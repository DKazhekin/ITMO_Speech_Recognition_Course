"""Inference pipeline for GP1 Russian spoken-numbers ASR (§10, CONTRACTS.md).

Loads a QuartzNet10x4 checkpoint, builds the log-mel frontend, and runs
batched CTC decoding over a manifest. Returns ``list[tuple[str, str]]``
where each tuple is ``(filename, digit_string)`` in manifest order.

Decoding strategy
-----------------
- Default: greedy CTC (always available, no optional deps).
- Optional: beam search via pyctcdecode + KenLM when ``lm_binary_path``
  is set *and* both ``pyctcdecode`` and ``kenlm`` are importable.
  Falls back silently to greedy if either dep is missing.

Checkpoint format
-----------------
``torch.load(checkpoint_path)`` must return a dict with at least a
``"model"`` key whose value is a ``state_dict``.  PyTorch >= 2.6 sets
``weights_only=True`` by default, but training checkpoints that serialise
Python dataclasses (e.g. ``TrainerConfig``) require ``weights_only=False``.
We use ``weights_only=False`` and document this choice explicitly.
See: https://pytorch.org/docs/stable/generated/torch.load.html

Config YAML format
------------------
Parsed via ``pyyaml`` (lazy import).  If pyyaml is not installed, the
code falls back to a hand-rolled line-parser for the specific fields
needed (``model.*`` and ``audio.*``).  If even the fallback cannot parse
the file, an ``ImportError`` is raised with a clear message.

Padding for batched inference
------------------------------
Audio samples are right-zero-padded to the longest sample in each batch.
``audio_lengths`` carry the true per-sample sample counts.  The mel
frontend receives the padded ``[B, T_max]`` tensor; ``mel_lengths`` are
derived as ``audio_lengths // hop_length + 1`` (matching torchaudio's
framing convention with ``center=True``).
See: https://pytorch.org/audio/stable/transforms.html

References
----------
- NeMo CTC inference:
  https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_models.py
- soundfile batched loading:
  https://python-soundfile.readthedocs.io/en/0.13.1/
- PyTorch autocast CPU/CUDA:
  https://pytorch.org/docs/stable/amp.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio.transforms

from gp1.decoding.greedy import greedy_decode
from gp1.features.melbanks import LogMelFilterBanks
from gp1.models.quartznet import QuartzNet10x4
from gp1.text.denormalize import words_to_digits
from gp1.text.vocab import CharVocab
from gp1.types import ManifestRecord

log = logging.getLogger(__name__)

TARGET_SR: int = 16000


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceConfig:
    """Immutable configuration for ``run_inference``.

    Parameters
    ----------
    checkpoint_path:
        Path to ``.pt`` checkpoint file saved by ``Trainer``.
        Expected dict layout: ``{"model": <state_dict>, ...}``.
    config_path:
        Path to YAML file describing model architecture and audio frontend.
        Required keys: ``model.vocab_size`` (int), optionally ``model.d_model``,
        ``model.dropout``, ``model.subsample_factor``, ``audio.*``.
    lm_binary_path:
        Optional path to a compiled KenLM ``.bin`` binary.  When provided
        and pyctcdecode + kenlm are installed, beam search is used instead
        of greedy decoding.
    batch_size:
        Number of utterances per forward pass. Default: 32.
    device:
        PyTorch device string (e.g. ``"cuda"``, ``"cpu"``). Default: ``"cuda"``.
    """

    checkpoint_path: Path
    config_path: Path
    lm_binary_path: Path | None
    batch_size: int = 32
    device: str = "cuda"


# ---------------------------------------------------------------------------
# YAML loading (lazy, with fallback)
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file.  Prefers pyyaml; falls back to a tiny hand-rolled parser.

    The hand-rolled fallback only handles the simple ``key: value`` /
    ``  sub_key: value`` subset used in GP1 model configs.  If it cannot
    parse the file, it raises ``ImportError`` directing the user to install
    pyyaml.
    """
    try:
        import yaml  # type: ignore[import-not-found]

        with path.open() as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        pass

    # -- hand-rolled fallback for simple nested YAML --------------------------
    try:
        return _parse_simple_yaml(path.read_text())
    except Exception as exc:
        raise ImportError(
            "pyyaml is not installed and the hand-rolled YAML fallback failed. "
            "Install pyyaml with: uv pip install pyyaml\n"
            f"Fallback error: {exc}"
        ) from exc


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a minimal nested YAML string (no lists, no multi-line scalars).

    Handles the structure produced by ``yaml.safe_dump`` for GP1 configs:

        model:
          vocab_size: 35
          d_model: 256
        audio:
          n_fft: 512
    """
    result: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    current_key: str | None = None

    for raw_line in text.splitlines():
        # Skip blank lines and comments.
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        if ":" not in stripped:
            continue

        key_part, _, val_part = stripped.partition(":")
        key = key_part.strip()
        val_str = val_part.strip()

        if indent == 0:
            # Top-level key
            current_key = key
            if val_str:
                result[key] = _coerce_scalar(val_str)
                current_section = None
            else:
                result[key] = {}
                current_section = result[key]
        else:
            # Nested key under current_section
            if current_section is not None:
                current_section[key] = _coerce_scalar(val_str) if val_str else {}

    return result


def _coerce_scalar(s: str) -> Any:
    """Convert a YAML scalar string to int, float, bool, None, or str."""
    if s in ("null", "~", ""):
        return None
    if s in ("true", "True", "yes"):
        return True
    if s in ("false", "False", "no"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_model(
    checkpoint_path: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> QuartzNet10x4:
    """Instantiate QuartzNet10x4 from config and load weights from checkpoint.

    Notes
    -----
    ``weights_only=False`` is used because training checkpoints may embed
    Python dataclasses (``TrainerConfig``) that are not serialisation-safe
    under ``weights_only=True``.  PyTorch >= 2.6 changed the default to
    ``True``, so we must be explicit.
    See: https://pytorch.org/docs/stable/generated/torch.load.html
    """
    model_cfg = cfg.get("model", {}) or {}
    vocab_size: int = int(model_cfg.get("vocab_size", CharVocab.vocab_size))
    d_model: int = int(model_cfg.get("d_model", 256))
    dropout: float = float(model_cfg.get("dropout", 0.1))
    subsample_factor: int = int(model_cfg.get("subsample_factor", 2))

    model = QuartzNet10x4(
        vocab_size=vocab_size,
        d_model=d_model,
        dropout=dropout,
        subsample_factor=subsample_factor,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,  # see docstring — training ckpts embed Python objects
    )
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    log.info(
        "Loaded QuartzNet10x4 from %s (vocab_size=%d, d_model=%d)",
        checkpoint_path,
        vocab_size,
        d_model,
    )
    return model


# ---------------------------------------------------------------------------
# Audio loading and preprocessing
# ---------------------------------------------------------------------------


def _load_and_preprocess_batch(
    records: list[ManifestRecord],
    target_sr: int = TARGET_SR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of audio files, resample, and right-zero-pad.

    Follows the same soundfile.read + torchaudio.transforms.Resample pattern
    used in ``src/gp1/data/dataset.py`` to keep a single canonical audio path.

    Parameters
    ----------
    records:
        Batch of manifest records (all files must exist).
    target_sr:
        Target sample rate in Hz (default 16000).

    Returns
    -------
    audio : torch.Tensor
        ``[B, T_max]`` float32 padded waveform.
    audio_lengths : torch.Tensor
        ``[B]`` int64 actual sample counts before padding.
    """
    resamplers: dict[int, torchaudio.transforms.Resample] = {}
    waveforms: list[torch.Tensor] = []

    for record in records:
        data, file_sr = sf.read(
            str(record.audio_path), dtype="float32", always_2d=False
        )
        if data.ndim == 2:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data.copy()).float()

        # Resample if native SR differs from target.
        if file_sr != target_sr:
            if file_sr not in resamplers:
                resamplers[file_sr] = torchaudio.transforms.Resample(
                    orig_freq=file_sr,
                    new_freq=target_sr,
                )
            wav = resamplers[file_sr](wav.unsqueeze(0)).squeeze(0)

        waveforms.append(wav)

    # Right-zero-pad to longest sample in the batch.
    audio_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    t_max = int(audio_lengths.max().item())
    padded = torch.zeros(len(waveforms), t_max, dtype=torch.float32)
    for i, wav in enumerate(waveforms):
        padded[i, : wav.shape[0]] = wav

    return padded, audio_lengths


# ---------------------------------------------------------------------------
# Decoder selection
# ---------------------------------------------------------------------------


def _try_build_beam_decoder(
    vocab: CharVocab,
    lm_binary_path: Path,
) -> Any | None:
    """Attempt to build a BeamSearchDecoder.  Returns None if deps missing."""
    try:
        from gp1.decoding.beam_pyctc import BeamSearchConfig, BeamSearchDecoder
        from gp1.text.vocab_word import NUMBER_WORDS

        unigrams = list(NUMBER_WORDS)
        beam_cfg = BeamSearchConfig()
        decoder = BeamSearchDecoder(
            vocab=vocab,
            kenlm_path=lm_binary_path,
            unigrams=unigrams,
            config=beam_cfg,
        )
        log.info("Beam search decoder initialised with LM: %s", lm_binary_path)
        return decoder
    except Exception as exc:
        log.warning(
            "Beam search decoder unavailable (lm=%s), falling back to greedy: %s",
            lm_binary_path,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_inference(
    manifest: list[ManifestRecord],
    config: InferenceConfig,
) -> list[tuple[str, str]]:
    """Run batched CTC inference and return (filename, digit_string) pairs.

    Parameters
    ----------
    manifest:
        Ordered list of manifest records.  The order is preserved in the output.
    config:
        Immutable inference configuration (see ``InferenceConfig``).

    Returns
    -------
    list[tuple[str, str]]
        ``[(filename, digit_string), ...]`` in the same order as *manifest*.

        - ``filename`` is ``record.audio_path.name`` (e.g. ``"abc_001.wav"``).
        - ``digit_string`` is the CTC prediction converted from Russian number
          words back to digits via ``words_to_digits``.  Empty string when
          decoding yields only blanks/spaces or conversion fails.

    Notes
    -----
    - Model runs in ``eval()`` mode, under ``torch.no_grad()``.
    - On CUDA: ``torch.autocast`` with ``dtype=torch.float16`` is applied.
    - On CPU: no autocast (float32 throughout).
    - Mel lengths are derived as ``audio_lengths // hop_length + 1`` matching
      torchaudio's ``center=True`` framing.

    References
    ----------
    - NeMo CTC inference:
      https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_models.py
    - icefall ASR inference:
      https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/transducer/decode.py
    """
    if not manifest:
        return []

    device = torch.device(config.device)

    # -- Load config and model ------------------------------------------------
    cfg = _load_yaml(config.config_path)
    model = _load_model(config.checkpoint_path, cfg, device)

    # -- Build mel frontend ---------------------------------------------------
    audio_cfg: dict[str, Any] = cfg.get("audio", {}) or {}
    frontend = LogMelFilterBanks(**audio_cfg).to(device)
    frontend.eval()
    hop_length: int = int(audio_cfg.get("hop_length", 160))

    # -- Vocabulary and decoder -----------------------------------------------
    vocab = CharVocab()

    beam_decoder = None
    if config.lm_binary_path is not None:
        beam_decoder = _try_build_beam_decoder(vocab, config.lm_binary_path)

    # -- Inference loop -------------------------------------------------------
    results: list[tuple[str, str]] = []
    batch_size = config.batch_size
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for start in range(0, len(manifest), batch_size):
            batch_records = manifest[start : start + batch_size]

            # Load and pad audio.
            audio, audio_lengths = _load_and_preprocess_batch(batch_records, TARGET_SR)
            audio = audio.to(device)
            audio_lengths = audio_lengths.to(device)

            # Derive mel lengths (center=True adds n_fft//2 padding on both sides).
            mel_lengths = (audio_lengths // hop_length + 1).to(torch.long)

            # Forward pass — fp16 on CUDA, fp32 on CPU.
            with torch.autocast(
                device_type=device.type, enabled=is_cuda, dtype=torch.float16
            ):
                enc_out = model(frontend(audio), mel_lengths)

            # Decode.
            log_probs = enc_out.log_probs.float()  # ensure fp32 for decoders
            output_lengths = enc_out.output_lengths

            if beam_decoder is not None:
                raw_texts = beam_decoder.decode_batch(log_probs, output_lengths)
            else:
                raw_texts = greedy_decode(log_probs, output_lengths, vocab)

            # Convert Russian number words → digit string.
            for record, raw_text in zip(batch_records, raw_texts):
                filename = record.audio_path.name
                digit_str = _words_to_digits_safe(raw_text)
                results.append((filename, digit_str))

            log.debug(
                "Processed batch [%d:%d] on %s",
                start,
                start + len(batch_records),
                device,
            )

    return results


def _words_to_digits_safe(text: str) -> str:
    """Convert Russian number words to a digit string, returning '' on failure.

    ``words_to_digits`` raises ``ValueError`` on malformed input (e.g. blank
    output from greedy CTC).  We catch that here so one bad sample does not
    abort the entire inference pass.
    """
    if not text or not text.strip():
        return ""
    try:
        return words_to_digits(text.strip())
    except (ValueError, KeyError, Exception) as exc:
        log.debug("words_to_digits failed for %r: %s", text, exc)
        return ""
