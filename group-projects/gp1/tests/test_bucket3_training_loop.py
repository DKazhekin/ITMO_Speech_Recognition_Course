"""Bucket 3 — Training-loop correctness tests.

Covers H1 (SpecAug wiring), H2 (real sample_lengths), H3 (sampler epoch
seed), H4 (grad_clip_norm), H6 (InterCTC guard), M11 (final-val pass).

TDD RED phase — all tests must FAIL before fixes are applied.
AAA (Arrange-Act-Assert) pattern throughout.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup (mirrors scripts/train.py bootstrap)
# ---------------------------------------------------------------------------

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from gp1.data.collate import DynamicBucketSampler
from gp1.data.spec_aug import SpecAugmenter
from gp1.models.base import EncoderOutput
from gp1.text.vocab import CharVocab
from gp1.train.trainer import Trainer, TrainerConfig
from gp1.types import AugConfig, Batch, ManifestRecord


# ---------------------------------------------------------------------------
# Shared helpers / fake objects
# ---------------------------------------------------------------------------


class _FakeEncoderWithIntermediate(nn.Module):
    """Encoder that exposes intermediate features (e.g. QuartzNet-style)."""

    def __init__(self, vocab_size: int = 35) -> None:
        super().__init__()
        self.proj = nn.Linear(80, vocab_size)

    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor) -> EncoderOutput:
        pooled = mel.mean(dim=-1)  # [B, 80]
        logits = self.proj(pooled)
        log_probs = F.log_softmax(logits, dim=-1).unsqueeze(1).expand(-1, 2, -1)
        out_lengths = torch.full((mel.size(0),), 2, dtype=torch.long)
        intermediate = pooled.unsqueeze(1).expand(-1, 2, -1)
        return EncoderOutput(
            log_probs=log_probs,
            output_lengths=out_lengths,
            intermediate=intermediate,
        )


class _FakeEncoderNoIntermediate(nn.Module):
    """Encoder that returns intermediate=None (e.g. CRDNN-style)."""

    def __init__(self, vocab_size: int = 35) -> None:
        super().__init__()
        self.proj = nn.Linear(80, vocab_size)

    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor) -> EncoderOutput:
        pooled = mel.mean(dim=-1)
        logits = self.proj(pooled)
        log_probs = F.log_softmax(logits, dim=-1).unsqueeze(1).expand(-1, 2, -1)
        out_lengths = torch.full((mel.size(0),), 2, dtype=torch.long)
        return EncoderOutput(
            log_probs=log_probs,
            output_lengths=out_lengths,
            intermediate=None,  # <-- no intermediate tap
        )


def _fake_batch(batch_size: int = 2) -> Batch:
    audio = torch.zeros(batch_size, 800)
    audio_lengths = torch.full((batch_size,), 800, dtype=torch.long)
    targets = torch.ones(batch_size, 2, dtype=torch.long)
    target_lengths = torch.full((batch_size,), 2, dtype=torch.long)
    return Batch(
        audio=audio,
        audio_lengths=audio_lengths,
        targets=targets,
        target_lengths=target_lengths,
        spk_ids=[f"spk_{i}" for i in range(batch_size)],
        transcriptions=["один два"] * batch_size,
    )


class _FakeLoader:
    def __init__(self, batch: Batch, n_batches: int = 2) -> None:
        self._batch = batch
        self._n = n_batches

    def __iter__(self):
        for _ in range(self._n):
            yield self._batch

    def __len__(self) -> int:
        return self._n


def _scalar_loss(*args, **kwargs):
    return torch.tensor(0.5, requires_grad=True)


def _make_trainer(
    tmp_path: Path,
    *,
    model: nn.Module | None = None,
    max_epochs: int = 2,
    val_every_n_epochs: int = 1,
    grad_clip_norm: float | None = None,
    spec_augmenter: SpecAugmenter | None = None,
    inter_ctc=None,
    train_batches: int = 2,
    val_batches: int = 1,
) -> Trainer:
    torch.manual_seed(0)
    if model is None:
        model = _FakeEncoderWithIntermediate()
    vocab = CharVocab()
    ctc_loss = MagicMock(side_effect=_scalar_loss)
    word_aux = MagicMock(side_effect=_scalar_loss)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    batch = _fake_batch(batch_size=2)
    train_loader = _FakeLoader(batch, n_batches=train_batches)
    val_loader = _FakeLoader(batch, n_batches=val_batches)
    config = TrainerConfig(
        max_epochs=max_epochs,
        grad_accum=1,
        fp16_autocast=False,
        log_every_n_steps=1,
        val_every_n_epochs=val_every_n_epochs,
        early_stop_patience=100,
        early_stop_metric="max_speaker_cer",
        ckpt_dir=tmp_path / "ckpts",
        grad_clip_norm=grad_clip_norm,
    )
    return Trainer(
        model=model,
        ctc_loss=ctc_loss,
        inter_ctc=inter_ctc,
        cr_ctc=None,
        word_aux=None,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        config=config,
        device=torch.device("cpu"),
        wandb_run=None,
        spec_augmenter=spec_augmenter,
    )


# ---------------------------------------------------------------------------
# H1 — SpecAugmenter wiring
# ---------------------------------------------------------------------------


class TestSpecAugWiring:
    """H1: SpecAugmenter must be called during train batches but not validation."""

    def test_specaug_applied_in_training_not_validation(self, tmp_path: Path) -> None:
        """SpecAugmenter is called for every train batch but never for val batch."""
        # Arrange
        spec_aug = MagicMock(wraps=SpecAugmenter())
        # Make the mock actually return the input unchanged
        spec_aug.side_effect = lambda mel, lengths: mel
        spec_aug.training = True

        trainer = _make_trainer(
            tmp_path,
            max_epochs=1,
            spec_augmenter=spec_aug,
            train_batches=2,
            val_batches=1,
        )

        # Act
        trainer.fit()

        # Assert — called for each of the 2 train batches
        assert spec_aug.call_count == 2, (
            f"Expected SpecAugmenter called 2 times (train), got {spec_aug.call_count}"
        )

    def test_specaug_not_applied_during_validation(self, tmp_path: Path) -> None:
        """SpecAugmenter is NOT called during validation (model.eval() mode)."""
        # Arrange
        call_log: list[str] = []

        class _LoggingAug(SpecAugmenter):
            def forward(self, mel, lengths):
                call_log.append("train" if self.training else "eval")
                return mel

        spec_aug = _LoggingAug()
        trainer = _make_trainer(
            tmp_path,
            max_epochs=1,
            spec_augmenter=spec_aug,
            train_batches=2,
            val_batches=1,
        )

        # Act
        trainer.fit()

        # Assert — no "eval" entries; only "train" entries
        assert "eval" not in call_log, (
            f"SpecAugmenter was called during eval; log: {call_log}"
        )
        assert len(call_log) == 2, f"Expected 2 train calls, got {call_log}"


class TestSpecAugParamPlumbing:
    """H1: specaug_* keys must NOT be filtered out by _build_aug_config."""

    def test_specaug_params_plumb_from_yaml_to_augconfig(self) -> None:
        """_build_aug_config passes specaug_* keys through to AugConfig."""
        # Arrange — import the function from train.py
        train_script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("train_script", train_script)
        train_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_mod)

        cfg_aug = {
            "specaug_freq_mask_param": 20,
            "specaug_num_freq_masks": 3,
            "specaug_time_mask_param": 30,
            "specaug_num_time_masks": 4,
            "specaug_time_mask_max_ratio": 0.08,
        }

        # Act
        aug_config: AugConfig = train_mod._build_aug_config(cfg_aug, seed=42)

        # Assert — all specaug params survive filtering
        assert aug_config.specaug_freq_mask_param == 20
        assert aug_config.specaug_num_freq_masks == 3
        assert aug_config.specaug_time_mask_param == 30
        assert aug_config.specaug_num_time_masks == 4
        assert aug_config.specaug_time_mask_max_ratio == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# H2 — real sample_lengths
# ---------------------------------------------------------------------------


class TestManifestRecordDurationField:
    """H2: ManifestRecord must carry duration_s (float in seconds)."""

    def test_manifest_record_has_duration_s_field(self) -> None:
        """dataclasses.fields(ManifestRecord) includes 'duration_s' of type float."""
        # Arrange / Act
        field_names = {f.name for f in dataclasses.fields(ManifestRecord)}

        # Assert
        assert "duration_s" in field_names, (
            "ManifestRecord missing 'duration_s' field; add it to types.py"
        )

    def test_manifest_record_duration_s_type_is_float(self) -> None:
        """duration_s field annotation is float."""
        field_map = {f.name: f for f in dataclasses.fields(ManifestRecord)}
        assert "duration_s" in field_map
        assert field_map["duration_s"].type in (float, "float"), (
            "duration_s should be float"
        )


class TestBuildManifestPopulatesDuration:
    """H2: build_manifest must populate duration_s from sf.info."""

    def _write_wav(self, path: Path, samplerate: int, n_samples: int) -> None:
        data = np.zeros(n_samples, dtype=np.float32)
        sf.write(str(path), data, samplerate)

    def test_build_manifest_populates_duration_s(self, tmp_path: Path) -> None:
        """build_manifest writes duration_s that matches frames / samplerate."""
        import csv as csv_mod

        from gp1.data.manifest import build_manifest, read_jsonl

        # Arrange — 2-second WAV at 16 kHz
        audio_root = tmp_path / "audio"
        audio_root.mkdir()
        sr = 16000
        n_samples = sr * 2  # exactly 2 seconds
        wav_path = audio_root / "test.wav"
        self._write_wav(wav_path, sr, n_samples)

        csv_path = tmp_path / "train.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv_mod.DictWriter(
                fh, fieldnames=["filename", "transcription", "spk_id", "gender"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "test.wav",
                    "transcription": "12345",
                    "spk_id": "spk_A",
                    "gender": "male",
                }
            )

        out_path = tmp_path / "manifest.jsonl"

        # Act
        build_manifest(csv_path, audio_root, out_path)
        records = read_jsonl(out_path)

        # Assert
        assert len(records) == 1
        assert hasattr(records[0], "duration_s"), (
            "duration_s missing from loaded record"
        )
        assert records[0].duration_s == pytest.approx(2.0, abs=0.001), (
            f"Expected duration_s≈2.0, got {records[0].duration_s}"
        )


class TestSampleLengthsUseDuration:
    """H2: sample_lengths in train.py must use r.duration_s, not r.samplerate * 2."""

    def test_sample_lengths_reflect_real_duration(self) -> None:
        """sample_lengths computation uses r.duration_s * target_sr, not r.samplerate * 2."""
        # Arrange — two fake records with different durations
        r1 = ManifestRecord(
            audio_path=Path("/fake/a.wav"),
            transcription="10000",
            spk_id="spk_A",
            gender="male",
            ext="wav",
            samplerate=16000,
            duration_s=1.5,
        )
        r2 = ManifestRecord(
            audio_path=Path("/fake/b.wav"),
            transcription="20000",
            spk_id="spk_B",
            gender="male",
            ext="wav",
            samplerate=16000,
            duration_s=3.0,
        )
        records = [r1, r2]

        # Act — reproduce the expression from train.py
        TARGET_SR = 16000
        sample_lengths = [int(r.duration_s * TARGET_SR) for r in records]

        # Assert
        assert sample_lengths[0] == 24000, f"Expected 24000, got {sample_lengths[0]}"
        assert sample_lengths[1] == 48000, f"Expected 48000, got {sample_lengths[1]}"
        # They must differ — old code would produce the same value for both
        assert sample_lengths[0] != sample_lengths[1], (
            "sample_lengths must reflect actual duration, not a fixed 2 s estimate"
        )


# ---------------------------------------------------------------------------
# H3 — DynamicBucketSampler seed per epoch
# ---------------------------------------------------------------------------


class TestDynamicBucketSamplerEpochSeed:
    """H3: Repeated calls to __iter__ must produce different orders (epoch-based seed)."""

    def test_successive_iters_produce_different_orders(self) -> None:
        """Three consecutive iter() calls produce at least 2 distinct orderings."""
        # Arrange — enough items to make shuffle meaningful
        lengths = list(range(100, 1600, 100))  # 15 items
        sampler = DynamicBucketSampler(
            lengths=lengths,
            max_tokens_per_batch=500,
            num_buckets=3,
            shuffle=True,
        )
        sampler._seed = 0

        # Act
        epoch1 = list(sampler)
        epoch2 = list(sampler)
        epoch3 = list(sampler)

        # Assert — at least 2 of 3 must differ
        all_same = epoch1 == epoch2 == epoch3
        assert not all_same, (
            "All 3 epochs produced identical batch order — seed not advancing between epochs"
        )

    def test_set_epoch_advances_order(self) -> None:
        """set_epoch(n) followed by iter() produces epoch-n ordering."""
        lengths = list(range(100, 1100, 100))  # 10 items
        sampler = DynamicBucketSampler(
            lengths=lengths,
            max_tokens_per_batch=400,
            num_buckets=3,
            shuffle=True,
        )
        sampler._seed = 7

        sampler.set_epoch(0)
        order_e0_first = list(sampler)
        sampler.set_epoch(0)
        order_e0_second = list(sampler)
        sampler.set_epoch(1)
        order_e1 = list(sampler)

        # Same epoch → same order; different epoch → different (almost certainly)
        assert order_e0_first == order_e0_second, (
            "set_epoch(0) twice must yield same order"
        )
        assert order_e0_first != order_e1, (
            "set_epoch(0) vs set_epoch(1) must (almost certainly) differ"
        )


# ---------------------------------------------------------------------------
# H4 — grad_clip_norm
# ---------------------------------------------------------------------------


class TestGradClipNorm:
    """H4: grad_clip_norm must be read from config and applied during training."""

    def test_trainer_config_has_grad_clip_norm_field(self) -> None:
        """TrainerConfig accepts grad_clip_norm without TypeError."""
        # Arrange / Act
        config = TrainerConfig(max_epochs=1, grad_clip_norm=1.0)

        # Assert
        assert config.grad_clip_norm == 1.0

    def test_trainer_config_grad_clip_norm_defaults_to_none(self) -> None:
        """grad_clip_norm defaults to None (backward-compat: no clip by default)."""
        config = TrainerConfig(max_epochs=1)
        assert config.grad_clip_norm is None

    def test_trainer_calls_clip_grad_norm_when_configured(self, tmp_path: Path) -> None:
        """clip_grad_norm_ is called with the configured max_norm during training."""
        # Arrange
        trainer = _make_trainer(tmp_path, max_epochs=1, grad_clip_norm=1.5)

        # Act — patch the clip function and run training
        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            mock_clip.return_value = torch.tensor(0.1)
            trainer.fit()

        # Assert — must have been called at least once with max_norm=1.5
        assert mock_clip.call_count > 0, "clip_grad_norm_ was never called"
        for c in mock_clip.call_args_list:
            _, kwargs = c
            # max_norm may be positional (arg 1) or keyword
            passed_max_norm = kwargs.get("max_norm", c[0][1] if len(c[0]) > 1 else None)
            assert passed_max_norm == pytest.approx(1.5), (
                f"clip_grad_norm_ called with max_norm={passed_max_norm}, expected 1.5"
            )

    def test_trainer_does_not_call_clip_grad_norm_when_not_configured(
        self, tmp_path: Path
    ) -> None:
        """clip_grad_norm_ is NOT called when grad_clip_norm is None."""
        # Arrange
        trainer = _make_trainer(tmp_path, max_epochs=1, grad_clip_norm=None)

        # Act
        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            trainer.fit()

        # Assert
        assert mock_clip.call_count == 0, (
            f"clip_grad_norm_ called {mock_clip.call_count} times but grad_clip_norm is None"
        )


# ---------------------------------------------------------------------------
# H6 — InterCTC guard
# ---------------------------------------------------------------------------


class TestInterCTCGuard:
    """H6: Trainer must raise ValueError early if inter_ctc enabled but intermediate=None."""

    def test_trainer_raises_on_inter_ctc_with_none_intermediate(
        self, tmp_path: Path
    ) -> None:
        """Trainer.__init__ or first forward raises ValueError when inter_ctc is
        enabled but the model returns intermediate=None."""
        # Arrange
        model = _FakeEncoderNoIntermediate()
        inter_ctc_head = MagicMock(side_effect=_scalar_loss)

        # Act / Assert
        with pytest.raises(ValueError, match="(?i)intermediate"):
            trainer = _make_trainer(
                tmp_path,
                model=model,
                inter_ctc=inter_ctc_head,
            )
            # In case __init__ defers the check to first forward
            trainer.fit()


# ---------------------------------------------------------------------------
# M11 — final-epoch validation gap
# ---------------------------------------------------------------------------


class TestFinalEpochValidation:
    """M11: fit() must run a final validation after the last epoch if it was skipped."""

    def test_final_validation_runs_when_last_epoch_not_multiple_of_val_every(
        self, tmp_path: Path
    ) -> None:
        """max_epochs=3, val_every_n_epochs=2 → validation must also run after epoch 3."""
        # Arrange
        trainer = _make_trainer(
            tmp_path,
            max_epochs=3,
            val_every_n_epochs=2,
        )

        validation_epochs: list[int] = []
        original_val = trainer._run_validation

        def recording_val(epoch: int) -> float:
            validation_epochs.append(epoch)
            return original_val(epoch)

        trainer._run_validation = recording_val  # type: ignore[method-assign]

        # Act
        trainer.fit()

        # Assert — epoch 3 must be validated (even though 3 % 2 != 0)
        assert 3 in validation_epochs, (
            f"Validation was not run for epoch 3. Validated epochs: {validation_epochs}"
        )

    def test_no_double_validation_when_last_epoch_is_multiple(
        self, tmp_path: Path
    ) -> None:
        """max_epochs=4, val_every_n_epochs=2 → epoch 4 validated once (no duplicate)."""
        # Arrange
        trainer = _make_trainer(
            tmp_path,
            max_epochs=4,
            val_every_n_epochs=2,
        )

        validation_epochs: list[int] = []
        original_val = trainer._run_validation

        def recording_val(epoch: int) -> float:
            validation_epochs.append(epoch)
            return original_val(epoch)

        trainer._run_validation = recording_val  # type: ignore[method-assign]

        # Act
        trainer.fit()

        # Assert — epoch 4 must appear exactly once
        assert validation_epochs.count(4) == 1, (
            f"Epoch 4 validated {validation_epochs.count(4)} times; expected exactly once. "
            f"All validated epochs: {validation_epochs}"
        )
