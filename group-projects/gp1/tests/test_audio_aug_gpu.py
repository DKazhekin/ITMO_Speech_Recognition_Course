"""Tests for GPUAudioAugmenter (TDD: RED phase)."""

from __future__ import annotations

import pytest
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
    aug._rir_pool = [torch.randn(4096)]  # synthetic IR

    out = aug._apply_rir_batched(audio.clone())

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
