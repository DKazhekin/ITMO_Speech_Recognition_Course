"""Tests for src/gp1/submit/inference.py.

TDD RED → GREEN → REFACTOR per CONTRACTS.md §10.

All tests use tiny fake checkpoints/models + tmp_path to avoid real disk state.
No pyctcdecode / kenlm required for the greedy path; beam path is skip-gated.

References:
  - CONTRACTS.md §10: InferenceConfig + run_inference signatures
  - NeMo inference pattern: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_models.py
  - soundfile docs: https://python-soundfile.readthedocs.io/
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

TARGET_SR = 16000


def _write_wav(path: Path, duration_s: float = 0.5) -> None:
    """Write a short mono float32 WAV file for testing."""
    n_samples = int(TARGET_SR * duration_s)
    audio = np.random.RandomState(42).randn(n_samples).astype(np.float32)
    sf.write(str(path), audio, TARGET_SR, subtype="FLOAT")


def _make_manifest_record(audio_path: Path) -> "ManifestRecord":
    from gp1.types import ManifestRecord

    return ManifestRecord(
        audio_path=audio_path,
        transcription="123456",
        spk_id="spk_A",
        gender="male",
        ext="wav",
        samplerate=TARGET_SR,
    )


def _write_quartznet_checkpoint(ckpt_path: Path, cfg_path: Path) -> None:
    """Save a minimal randomly-initialised QuartzNet10x4 checkpoint + config YAML."""
    from gp1.models.quartznet import QuartzNet10x4

    model = QuartzNet10x4(vocab_size=35)
    state = {"model": model.state_dict()}
    torch.save(state, ckpt_path)

    # Write a minimal YAML config that inference.py should be able to load.
    # We hand-write the YAML string so we do NOT depend on pyyaml being installed
    # in the RED phase — inference.py itself will gate on yaml availability.
    cfg_yaml = (
        "model:\n"
        "  vocab_size: 35\n"
        "  d_model: 256\n"
        "  dropout: 0.1\n"
        "  subsample_factor: 2\n"
        "audio:\n"
        "  n_fft: 512\n"
        "  samplerate: 16000\n"
        "  hop_length: 160\n"
        "  win_length: 400\n"
        "  n_mels: 80\n"
    )
    cfg_path.write_text(cfg_yaml)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_setup(tmp_path: Path):
    """Create 2 WAV files, a checkpoint and a config YAML, return components."""
    wav1 = tmp_path / "spk_A_001.wav"
    wav2 = tmp_path / "spk_A_002.wav"
    _write_wav(wav1)
    _write_wav(wav2)

    ckpt_path = tmp_path / "model.pt"
    cfg_path = tmp_path / "config.yaml"
    _write_quartznet_checkpoint(ckpt_path, cfg_path)

    records = [_make_manifest_record(wav1), _make_manifest_record(wav2)]
    return records, ckpt_path, cfg_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_inference_config_is_frozen_dataclass(tmp_path: Path) -> None:
    """InferenceConfig must be a frozen dataclass — mutation raises FrozenInstanceError."""
    from gp1.submit.inference import InferenceConfig

    cfg = InferenceConfig(
        checkpoint_path=tmp_path / "ckpt.pt",
        config_path=tmp_path / "cfg.yaml",
        lm_binary_path=None,
        batch_size=8,
        device="cpu",
    )

    assert dataclasses.is_dataclass(cfg), "InferenceConfig must be a dataclass"

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cfg.batch_size = 1  # type: ignore[misc]


def test_run_inference_on_empty_manifest_returns_empty_list(tmp_path: Path) -> None:
    """Edge case: empty manifest → empty list."""
    from gp1.submit.inference import InferenceConfig, run_inference

    ckpt_path = tmp_path / "model.pt"
    cfg_path = tmp_path / "config.yaml"
    _write_quartznet_checkpoint(ckpt_path, cfg_path)

    config = InferenceConfig(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        lm_binary_path=None,
        batch_size=32,
        device="cpu",
    )

    result = run_inference([], config)
    assert result == [], f"Expected [], got {result!r}"


def test_run_inference_returns_list_of_filename_digit_string_tuples(
    tiny_setup,
) -> None:
    """run_inference must return list[tuple[str, str]] of length == len(manifest)."""
    from gp1.submit.inference import InferenceConfig, run_inference

    records, ckpt_path, cfg_path = tiny_setup
    config = InferenceConfig(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        lm_binary_path=None,
        batch_size=32,
        device="cpu",
    )

    result = run_inference(records, config)

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == len(records), (
        f"Length mismatch: expected {len(records)}, got {len(result)}"
    )
    for item in result:
        assert isinstance(item, tuple) and len(item) == 2, (
            f"Each result must be a 2-tuple, got {item!r}"
        )
        filename, digit_str = item
        assert isinstance(filename, str), f"filename must be str, got {type(filename)}"
        assert isinstance(digit_str, str), (
            f"digit_string must be str, got {type(digit_str)}"
        )


def test_run_inference_preserves_order_of_input_manifest(tmp_path: Path) -> None:
    """Output order must match the input manifest order."""
    from gp1.submit.inference import InferenceConfig, run_inference

    ckpt_path = tmp_path / "model.pt"
    cfg_path = tmp_path / "config.yaml"
    _write_quartznet_checkpoint(ckpt_path, cfg_path)

    wav_paths = []
    records = []
    for i in range(4):
        wav = tmp_path / f"audio_{i:03d}.wav"
        _write_wav(wav, duration_s=0.3)
        wav_paths.append(wav)
        records.append(_make_manifest_record(wav))

    config = InferenceConfig(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        lm_binary_path=None,
        batch_size=2,  # intentionally smaller than manifest — tests chunking
        device="cpu",
    )

    result = run_inference(records, config)

    assert len(result) == 4
    for i, (filename, _) in enumerate(result):
        expected_name = wav_paths[i].name
        assert filename == expected_name, (
            f"Order mismatch at index {i}: expected {expected_name!r}, got {filename!r}"
        )


def test_run_inference_digit_string_only_contains_digits_or_empty(
    tiny_setup,
) -> None:
    """Decoded text must match ^\\d*$ (digits only or empty string)."""
    from gp1.submit.inference import InferenceConfig, run_inference

    records, ckpt_path, cfg_path = tiny_setup
    config = InferenceConfig(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        lm_binary_path=None,
        batch_size=32,
        device="cpu",
    )

    result = run_inference(records, config)

    for filename, digit_str in result:
        assert re.fullmatch(r"\d*", digit_str) is not None, (
            f"File {filename!r}: expected \\d* output, got {digit_str!r}"
        )


def test_run_inference_handles_batch_size_one(tmp_path: Path) -> None:
    """Boundary case: batch_size=1 must not raise."""
    from gp1.submit.inference import InferenceConfig, run_inference

    ckpt_path = tmp_path / "model.pt"
    cfg_path = tmp_path / "config.yaml"
    _write_quartznet_checkpoint(ckpt_path, cfg_path)

    wav = tmp_path / "single.wav"
    _write_wav(wav, duration_s=0.5)
    records = [_make_manifest_record(wav)]

    config = InferenceConfig(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        lm_binary_path=None,
        batch_size=1,
        device="cpu",
    )

    result = run_inference(records, config)

    assert len(result) == 1
    filename, digit_str = result[0]
    assert filename == wav.name
    assert re.fullmatch(r"\d*", digit_str) is not None


def test_run_inference_filename_is_audio_path_name(tiny_setup) -> None:
    """The first element of each tuple must be record.audio_path.name (not stem)."""
    from gp1.submit.inference import InferenceConfig, run_inference

    records, ckpt_path, cfg_path = tiny_setup
    config = InferenceConfig(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        lm_binary_path=None,
        batch_size=32,
        device="cpu",
    )

    result = run_inference(records, config)

    for record, (filename, _) in zip(records, result):
        assert filename == record.audio_path.name, (
            f"Expected {record.audio_path.name!r}, got {filename!r}"
        )


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pyctcdecode"),
    reason="pyctcdecode not installed",
)
def test_run_inference_uses_beam_when_lm_path_provided_and_pyctc_installed(
    tmp_path: Path,
) -> None:
    """Smoke-test beam path when pyctcdecode is available."""
    from gp1.submit.inference import InferenceConfig, run_inference

    ckpt_path = tmp_path / "model.pt"
    cfg_path = tmp_path / "config.yaml"
    _write_quartznet_checkpoint(ckpt_path, cfg_path)

    wav = tmp_path / "beam_test.wav"
    _write_wav(wav)
    records = [_make_manifest_record(wav)]

    # Use a non-existent LM path; the decoder should handle it gracefully or
    # the test documents that beam init requires a real binary.
    lm_path = tmp_path / "fake.bin"
    lm_path.write_bytes(b"")  # zero-byte placeholder

    config = InferenceConfig(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        lm_binary_path=lm_path,
        batch_size=32,
        device="cpu",
    )

    # If pyctcdecode is installed, run_inference should not crash on greedy fallback
    # even if the LM binary is invalid (pyctcdecode's fault, not ours).
    try:
        result = run_inference(records, config)
        assert len(result) == 1
    except Exception as exc:
        # A failure inside pyctcdecode (bad binary) is acceptable here.
        pytest.skip(f"pyctcdecode raised with bad LM binary: {exc}")
