"""Tests for GPUAudioAugmenter (TDD: RED phase)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from gp1.data.audio_aug_gpu import GPUAudioAugmenter


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (audio [4, 32000], audio_lengths [4]) on CPU."""
    torch.manual_seed(0)
    audio = torch.randn(4, 32000, dtype=torch.float32)
    lengths = torch.tensor([32000, 28000, 30000, 16000], dtype=torch.long)
    return audio, lengths


def test_gpu_augmenter_preserves_shape(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """VTLP with prob=1.0 must preserve batch shape [4, 32000]."""
    audio, lengths = batch
    aug = GPUAudioAugmenter(vtlp_prob=1.0, noise_prob=0.0, rir_prob=0.0)
    out = aug(audio, lengths)
    assert out.shape == audio.shape


def test_gpu_augmenter_is_noop_when_disabled(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """All probs=0.0 and no pools — output equals input exactly."""
    audio, lengths = batch
    aug = GPUAudioAugmenter(
        vtlp_prob=0.0,
        noise_prob=0.0,
        rir_prob=0.0,
        musan_root=None,
        rir_root=None,
    )
    out = aug(audio, lengths)
    assert torch.equal(out, audio)


def test_gpu_augmenter_handles_missing_pools(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """noise_prob/rir_prob > 0 but no pools — must not crash, shape preserved."""
    audio, lengths = batch
    aug = GPUAudioAugmenter(
        vtlp_prob=0.0,
        noise_prob=0.5,
        rir_prob=0.5,
        musan_root=None,
        rir_root=None,
    )
    out = aug(audio, lengths)
    assert out.shape == audio.shape


def test_apply_add_noise_batched_changes_signal(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """With a synthetic noise pool _apply_add_noise_batched alters the signal."""
    audio, lengths = batch
    aug = GPUAudioAugmenter(vtlp_prob=0.0, noise_prob=1.0, rir_prob=0.0)
    torch.manual_seed(1)
    aug._noise_pool = [torch.randn(16000)]  # synthetic noise

    out = aug._apply_add_noise_batched(audio.clone(), lengths)

    assert out.shape == audio.shape
    assert not torch.equal(out, audio), "Expected signal to change after noise addition"


def test_apply_rir_batched_preserves_length(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """_apply_rir_batched returns tensor of same shape [B, T]."""
    audio, _ = batch
    aug = GPUAudioAugmenter(vtlp_prob=0.0, noise_prob=0.0, rir_prob=1.0)
    ir = torch.randn(4096)
    aug._rir_pool = [ir / (ir.norm() + 1e-9)]

    out = aug._apply_rir_batched(audio.clone())

    assert out.shape == audio.shape


def test_load_pool_truncates_long_irs_to_half_second(tmp_path: Path) -> None:
    """_load_pool truncates IRs to <=0.5s at samplerate."""
    samplerate = 16000
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(32000).astype(np.float32)
    sf.write(tmp_path / "long_ir.wav", arr, samplerate)

    aug = GPUAudioAugmenter(
        samplerate=samplerate,
        vtlp_prob=0.0,
        noise_prob=0.0,
        rir_prob=0.0,
        rir_root=tmp_path,
    )

    max_samples = int(0.5 * samplerate)
    assert len(aug._rir_pool) == 1
    for ir in aug._rir_pool:
        assert ir.numel() <= max_samples


def test_load_pool_normalizes_irs_at_load_time(tmp_path: Path) -> None:
    """_load_pool returns unit-norm IRs."""
    samplerate = 16000
    rng = np.random.default_rng(1)
    for i, n in enumerate((4000, 6000, 8000)):
        arr = rng.standard_normal(n).astype(np.float32) * (i + 1)
        sf.write(tmp_path / f"ir_{i}.wav", arr, samplerate)

    aug = GPUAudioAugmenter(
        samplerate=samplerate,
        vtlp_prob=0.0,
        noise_prob=0.0,
        rir_prob=0.0,
        rir_root=tmp_path,
    )

    assert len(aug._rir_pool) == 3
    for ir in aug._rir_pool:
        assert torch.allclose(ir.norm(), torch.tensor(1.0), atol=1e-3)


def test_apply_rir_batched_uses_conv1d_no_fft(
    monkeypatch: pytest.MonkeyPatch,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """RIR path must NOT call torchaudio.functional.fftconvolve."""
    import torchaudio.functional as taF

    def _fail(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("fftconvolve called")

    monkeypatch.setattr(taF, "fftconvolve", _fail)

    audio, lengths = batch
    aug = GPUAudioAugmenter(vtlp_prob=0.0, noise_prob=0.0, rir_prob=1.0)
    ir = torch.randn(2048)
    aug._rir_pool = [ir / (ir.norm() + 1e-9)]

    out = aug(audio, lengths)
    assert out.shape == audio.shape


def test_apply_rir_batched_preserves_dtype(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """forward with RIR enabled keeps fp32 dtype and shape."""
    audio, lengths = batch
    assert audio.dtype == torch.float32
    aug = GPUAudioAugmenter(vtlp_prob=0.0, noise_prob=0.0, rir_prob=1.0)
    ir = torch.randn(2048)
    aug._rir_pool = [ir / (ir.norm() + 1e-9)]

    out = aug(audio, lengths)

    assert out.dtype == torch.float32
    assert out.shape == audio.shape


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
def test_gpu_augmenter_runs_on_cuda() -> None:
    """GPUAudioAugmenter.to('cuda') processes a CUDA batch and returns CUDA tensor."""
    torch.manual_seed(42)
    audio = torch.randn(2, 16000, dtype=torch.float32, device="cuda")
    lengths = torch.tensor([16000, 12000], dtype=torch.long, device="cuda")

    aug = GPUAudioAugmenter(vtlp_prob=1.0, noise_prob=0.0, rir_prob=0.0).to("cuda")
    out = aug(audio, lengths)

    assert out.device.type == "cuda"
    assert out.shape == audio.shape
