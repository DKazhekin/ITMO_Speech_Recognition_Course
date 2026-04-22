"""Inference pipeline for GP1 Russian spoken-numbers ASR (§10, CONTRACTS.md).

Supports four model architectures (dispatched via ``cfg["model"]["name"]``):
- ``quartznet_10x4``     → QuartzNet10x4      (char vocab)
- ``crdnn``              → CRDNN              (char vocab)
- ``efficient_conformer``→ EfficientConformer (char vocab)
- ``fast_conformer_bpe`` → FastConformerBPE   (BPE vocab)

Decoding strategy
-----------------
- Default: greedy CTC (always available, no optional deps).
- Optional: beam search via pyctcdecode + KenLM when ``lm_binary_path``
  is set *and* both ``pyctcdecode`` and ``kenlm`` are importable.
  Falls back silently to greedy if either dep is missing.
  BPE vocab falls back to greedy with a warning even if LM path is given,
  because the beam decoder only supports CharVocab today.

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
import torch.nn as nn
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
    vocab_size: int | None = None,
) -> nn.Module:
    """Instantiate a model from config and load weights from checkpoint.

    Dispatches on ``cfg["model"]["name"]`` (default: ``"quartznet_10x4"``).

    Supported names
    ---------------
    - ``quartznet_10x4``
    - ``crdnn``
    - ``efficient_conformer``
    - ``fast_conformer_bpe``

    Raises ``ValueError`` for unknown names.

    Parameters
    ----------
    checkpoint_path:
        Path to ``.pt`` checkpoint saved by ``Trainer``.
    cfg:
        Full config dict (parsed from YAML).
    device:
        Target device for the model.
    vocab_size:
        When provided, overrides ``cfg["model"]["vocab_size"]``.  Pass
        ``vocab.vocab_size`` here to avoid the off-by-one drift between the
        YAML value (SP piece count) and the actual head size used at training
        time (SP piece count + 1 blank).  When absent, falls back to the
        config value for backward compatibility with char-vocab models.

    Notes
    -----
    ``weights_only=False`` is used because training checkpoints may embed
    Python dataclasses (``TrainerConfig``) that are not serialisation-safe
    under ``weights_only=True``.  PyTorch >= 2.6 changed the default to
    ``True``, so we must be explicit.
    See: https://pytorch.org/docs/stable/generated/torch.load.html
    """
    model_cfg = cfg.get("model", {}) or {}
    model_name: str = str(model_cfg.get("name", "quartznet_10x4"))

    # vocab_size resolution: kwarg (from vocab object) takes priority over cfg.
    # The kwarg path is the C2 fix: BPE config says 256 but the checkpoint head
    # was built with BPEVocab.vocab_size = 257 (sp.get_piece_size() + 1).
    if vocab_size is None:
        cfg_vocab_size = int(model_cfg.get("vocab_size", CharVocab.vocab_size))
        vocab_size = cfg_vocab_size
    else:
        cfg_vocab_size = int(model_cfg.get("vocab_size", CharVocab.vocab_size))
        if cfg_vocab_size != vocab_size:
            log.warning(
                "vocab_size from config (%d) differs from vocab object (%d); "
                "using vocab object value. This is expected for BPE models where "
                "the YAML stores SP piece count but the checkpoint head includes "
                "an extra blank token.",
                cfg_vocab_size,
                vocab_size,
            )

    model: nn.Module
    if model_name == "quartznet_10x4":
        d_model: int = int(model_cfg.get("d_model", 256))
        dropout: float = float(model_cfg.get("dropout", 0.1))
        subsample_factor: int = int(model_cfg.get("subsample_factor", 2))
        model = QuartzNet10x4(
            vocab_size=vocab_size,
            d_model=d_model,
            dropout=dropout,
            subsample_factor=subsample_factor,
        )
        log.info(
            "Loading QuartzNet10x4 from %s (vocab_size=%d, d_model=%d)",
            checkpoint_path,
            vocab_size,
            d_model,
        )

    elif model_name == "crdnn":
        from gp1.models.crdnn import CRDNN  # lazy — avoid circular at module level

        d_cnn: int = int(model_cfg.get("d_cnn", 64))
        rnn_hidden: int = int(model_cfg.get("rnn_hidden", 256))
        rnn_layers: int = int(model_cfg.get("rnn_layers", 2))
        dropout = float(model_cfg.get("dropout", 0.15))
        subsample_factor = int(model_cfg.get("subsample_factor", 1))
        model = CRDNN(
            vocab_size=vocab_size,
            d_cnn=d_cnn,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            dropout=dropout,
            subsample_factor=subsample_factor,
        )
        log.info(
            "Loading CRDNN from %s (vocab_size=%d, d_cnn=%d, rnn_hidden=%d)",
            checkpoint_path,
            vocab_size,
            d_cnn,
            rnn_hidden,
        )

    elif model_name == "efficient_conformer":
        from gp1.models.efficient_conformer import EfficientConformer

        raw_stages = model_cfg.get("d_model_stages", [96, 128, 128])
        raw_blocks = model_cfg.get("n_blocks_per_stage", [4, 4, 4])
        d_model_stages: tuple[int, int, int] = tuple(int(x) for x in raw_stages)  # type: ignore[assignment]
        n_blocks_per_stage: tuple[int, int, int] = tuple(int(x) for x in raw_blocks)  # type: ignore[assignment]
        n_heads: int = int(model_cfg.get("n_heads", 4))
        ff_ratio: int = int(model_cfg.get("ff_ratio", 4))
        conv_kernel: int = int(model_cfg.get("conv_kernel", 15))
        dropout = float(model_cfg.get("dropout", 0.1))
        model = EfficientConformer(
            vocab_size=vocab_size,
            d_model_stages=d_model_stages,
            n_blocks_per_stage=n_blocks_per_stage,
            n_heads=n_heads,
            ff_ratio=ff_ratio,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )
        log.info(
            "Loading EfficientConformer from %s (vocab_size=%d, stages=%s)",
            checkpoint_path,
            vocab_size,
            d_model_stages,
        )

    elif model_name == "fast_conformer_bpe":
        from gp1.models.fast_conformer_bpe import FastConformerBPE

        d_model = int(model_cfg.get("d_model", 96))
        n_blocks: int = int(model_cfg.get("n_blocks", 16))
        n_heads = int(model_cfg.get("n_heads", 4))
        ff_ratio = int(model_cfg.get("ff_ratio", 4))
        conv_kernel = int(model_cfg.get("conv_kernel", 9))
        dropout = float(model_cfg.get("dropout", 0.1))
        subsample_factor = int(model_cfg.get("subsample_factor", 4))
        model = FastConformerBPE(
            vocab_size=vocab_size,
            d_model=d_model,
            n_blocks=n_blocks,
            n_heads=n_heads,
            ff_ratio=ff_ratio,
            conv_kernel=conv_kernel,
            dropout=dropout,
            subsample_factor=subsample_factor,
        )
        log.info(
            "Loading FastConformerBPE from %s (vocab_size=%d, d_model=%d)",
            checkpoint_path,
            vocab_size,
            d_model,
        )

    else:
        _known = (
            "quartznet_10x4",
            "crdnn",
            "efficient_conformer",
            "fast_conformer_bpe",
        )
        raise ValueError(
            f"Unknown model name {model_name!r}. "
            f"Known architectures: {_known}. "
            "Check cfg['model']['name'] in your YAML config."
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
    return model


# ---------------------------------------------------------------------------
# Vocabulary dispatch
# ---------------------------------------------------------------------------


def _build_vocab(cfg: dict[str, Any], config_path: Path) -> Any:
    """Build the correct vocabulary object based on ``cfg["text"]["vocab_type"]``.

    Parameters
    ----------
    cfg:
        Full config dict (may lack a ``"text"`` section — falls back to char).
    config_path:
        Path of the YAML config file.  Used to resolve a relative
        ``bpe_model_path`` relative to ``config_path.parent``.

    Returns
    -------
    CharVocab | BPEVocab
    """
    text_cfg: dict[str, Any] = cfg.get("text", {}) or {}
    vocab_type: str = str(text_cfg.get("vocab_type", "char"))

    if vocab_type == "bpe":
        from gp1.text.vocab_bpe import BPEVocab  # lazy — sentencepiece optional

        raw_path = text_cfg.get("bpe_model_path", "")
        if not raw_path:
            raise ValueError(
                "cfg['text']['bpe_model_path'] is missing or empty. "
                "Provide the path to a trained SentencePiece *.model file."
            )
        bpe_path = Path(raw_path)
        if not bpe_path.is_absolute():
            bpe_path = config_path.parent / bpe_path
        if not bpe_path.exists():
            raise FileNotFoundError(
                f"BPE model file not found: {bpe_path}. "
                "Train it first with scripts/train_bpe.py or provide an existing file."
            )
        return BPEVocab(model_path=bpe_path)

    # Default: character vocab
    return CharVocab()


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
    """Attempt to build a BeamSearchDecoder.  Returns None if deps missing.

    Only ``ImportError`` (pyctcdecode / kenlm not installed) and
    ``FileNotFoundError`` (LM binary path does not exist) are treated as
    expected missing-dependency conditions and silenced.  Any other exception
    indicates a real programming bug and is re-raised so it surfaces clearly.
    """
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
    except (ImportError, FileNotFoundError) as exc:
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

    # -- Load config ----------------------------------------------------------
    cfg = _load_yaml(config.config_path)

    # -- Build vocabulary FIRST (C2 fix) --------------------------------------
    # vocab.vocab_size is the single source of truth for the model head size.
    # For BPE, BPEVocab.vocab_size = sp.get_piece_size() + 1, which is one
    # more than the raw YAML value (which stores the SP piece count).
    # Building vocab before _load_model lets us pass the correct vocab_size
    # into the model constructor, preventing load_state_dict shape mismatches.
    vocab = _build_vocab(cfg, config.config_path)

    # -- Load model (C2 fix: pass vocab_size from vocab object) ---------------
    model = _load_model(
        config.checkpoint_path, cfg, device, vocab_size=vocab.vocab_size
    )

    # -- Build mel frontend ---------------------------------------------------
    audio_cfg: dict[str, Any] = cfg.get("audio", {}) or {}
    frontend = LogMelFilterBanks(**audio_cfg).to(device)
    frontend.eval()
    hop_length: int = int(audio_cfg.get("hop_length", 160))

    beam_decoder = None
    if config.lm_binary_path is not None:
        text_cfg: dict[str, Any] = cfg.get("text", {}) or {}
        vocab_type: str = str(text_cfg.get("vocab_type", "char"))
        if vocab_type == "bpe":
            log.warning(
                "Beam search decoder is not supported for BPE vocab; "
                "falling back to greedy decoding."
            )
        else:
            # Beam decoder only works with CharVocab.
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
    output from greedy CTC, unknown vocabulary token).  We catch only that
    exception here so one bad sample does not abort the entire inference pass.

    Any other exception type (``AttributeError``, ``ImportError``, etc.)
    indicates a real bug and must propagate to the caller.
    """
    if not text or not text.strip():
        return ""
    try:
        return words_to_digits(text.strip())
    except ValueError as exc:
        log.debug("words_to_digits failed for %r: %s", text, exc)
        return ""
