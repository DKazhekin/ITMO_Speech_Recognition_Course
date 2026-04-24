"""Slim CTC-only training loop for GP1 ASR system."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from gp1.data.spec_aug import SpecAugmenter
from gp1.decoding.greedy import greedy_decode
from gp1.features.melbanks import LogMelFilterBanks
from gp1.losses.ctc import CTCLoss
from gp1.text.denormalize import safe_words_to_digits
from gp1.text.normalize import digits_to_words
from gp1.train.checkpoint import save_best
from gp1.train.metrics import (
    compute_cer,
    compute_digit_cer_in_out_harmonic,
    compute_per_speaker_cer,
)
from gp1.types import Batch

logger = logging.getLogger(__name__)
_INF_CER = float("inf")


@dataclass
class TrainerConfig:
    """Hyperparameters and bookkeeping settings for the Trainer."""

    max_epochs: int
    grad_accum: int = 1
    fp16_autocast: bool = True
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1
    early_stop_patience: int = 15
    early_stop_metric: str = "harmonic_in_out_digit_cer"
    ckpt_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    grad_clip_norm: float | None = 1.0
    amp_dtype: torch.dtype = torch.bfloat16
    in_domain_speakers: set[str] | None = None


class Trainer:
    """CTC-only training loop with tqdm progress, fp16 autocast, grad accum, early stop."""

    def __init__(
        self,
        model: nn.Module,
        ctc_loss: CTCLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: Any,
        val_loader: Any,
        vocab: Any,
        config: TrainerConfig,
        device: torch.device,
        audio_cfg: dict,
        spec_augmenter: SpecAugmenter | None = None,
        feature_extractor: LogMelFilterBanks | None = None,
    ) -> None:
        if (
            config.early_stop_metric == "harmonic_in_out_digit_cer"
            and config.in_domain_speakers is None
        ):
            logger.warning(
                "early_stop_metric='harmonic_in_out_digit_cer' requires "
                "config.in_domain_speakers to be set; falling back to "
                "'max_speaker_cer' for this Trainer instance."
            )
            config.early_stop_metric = "max_speaker_cer"
        self.model = model.to(device)
        self.ctc_loss = ctc_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.config = config
        self.device = device
        self._mel = (
            feature_extractor
            if feature_extractor is not None
            else LogMelFilterBanks(**(audio_cfg or {}))
        ).to(device)
        self._spec_augmenter: SpecAugmenter | None = spec_augmenter
        if self._spec_augmenter is not None and isinstance(
            self._spec_augmenter, nn.Module
        ):
            self._spec_augmenter = self._spec_augmenter.to(device)
        self._global_step: int = 0
        self._best_monitored: float = _INF_CER
        self._best_ckpt_path: Path | None = None
        self._no_improve_epochs: int = 0
        config.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> dict[str, Any]:
        """Run training; return dict(best_monitored, best_ckpt_path, history)."""
        self._best_monitored = _INF_CER
        self._best_ckpt_path = None
        self._no_improve_epochs = 0
        history: list[dict] = []

        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="epochs"):
            avg_loss = self._train_epoch(epoch)

            if epoch % self.config.val_every_n_epochs == 0:
                (
                    val_cer,
                    per_spk,
                    in_cer,
                    out_cer,
                    harmonic_cer,
                    val_loss,
                ) = self._run_validation(epoch)
                max_spk_cer = max(per_spk.values()) if per_spk else val_cer
                if self.config.early_stop_metric == "harmonic_in_out_digit_cer":
                    monitored = harmonic_cer
                elif self.config.early_stop_metric == "max_speaker_cer":
                    monitored = max_spk_cer
                else:
                    monitored = val_cer
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss_avg": avg_loss,
                        "val_loss": val_loss,
                        "val_cer": val_cer,
                        "per_speaker_cer": per_spk,
                        "max_speaker_cer": max_spk_cer,
                        "in_cer": in_cer,
                        "out_cer": out_cer,
                        "harmonic_cer": harmonic_cer,
                    }
                )

                if monitored < self._best_monitored:
                    self._best_monitored = monitored
                    self._no_improve_epochs = 0
                    self._best_ckpt_path = save_best(
                        self.model,
                        {
                            "epoch": epoch,
                            "val_cer": val_cer,
                            "max_speaker_cer": max_spk_cer,
                            "best_monitored": self._best_monitored,
                        },
                        self.config.ckpt_dir,
                    )
                else:
                    self._no_improve_epochs += 1

                tqdm.write(
                    f"[Epoch {epoch}/{self.config.max_epochs}] train  | "
                    f"loss={avg_loss:.4f}"
                )
                if self.config.in_domain_speakers is None:
                    tqdm.write(
                        f"[Epoch {epoch}/{self.config.max_epochs}] val    | "
                        f"loss={val_loss:.4f}  cer={val_cer:.4f}  "
                        f"max_spk={max_spk_cer:.4f}  "
                        f"best={self._best_monitored:.4f}  "
                        f"no_improve={self._no_improve_epochs}/"
                        f"{self.config.early_stop_patience}"
                    )
                else:
                    tqdm.write(
                        f"[Epoch {epoch}/{self.config.max_epochs}] val    | "
                        f"loss={val_loss:.4f}  hm_cer={harmonic_cer:.4f}  "
                        f"(in={in_cer:.4f}  out={out_cer:.4f})  "
                        f"best={self._best_monitored:.4f}  "
                        f"no_improve={self._no_improve_epochs}/"
                        f"{self.config.early_stop_patience}"
                    )

                if self._no_improve_epochs >= self.config.early_stop_patience:
                    logger.info(
                        "Early stopping after %d epochs without improvement.",
                        self._no_improve_epochs,
                    )
                    break

        if self._best_ckpt_path is None:
            self._best_ckpt_path = save_best(
                self.model,
                {
                    "epoch": 0,
                    "val_cer": self._best_monitored,
                    "max_speaker_cer": self._best_monitored,
                    "best_monitored": self._best_monitored,
                },
                self.config.ckpt_dir,
            )

        return {
            "best_monitored": self._best_monitored,
            "best_ckpt_path": self._best_ckpt_path,
            "history": history,
        }

    def _train_epoch(self, epoch: int) -> float:
        """One pass over train_loader; return average loss."""
        self.model.train()
        self.optimizer.zero_grad()
        loss_sum = 0.0
        count = 0

        for micro_idx, batch in enumerate(
            tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        ):
            loss = self._forward_batch(batch)
            (loss / self.config.grad_accum).backward()
            loss_sum += loss.item()
            count += 1

            if (micro_idx + 1) % self.config.grad_accum == 0:
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self._global_step += 1

        if len(self.train_loader) % self.config.grad_accum != 0:
            if self.config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self._global_step += 1

        return loss_sum / max(count, 1)

    def _mel_features(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract mel spectrogram and compute frame lengths."""
        mel = self._mel(audio)
        mel_lengths = (
            (audio_lengths // self._mel.hop_length + 1).clamp(max=mel.size(-1)).long()
        )
        return mel, mel_lengths

    def _forward_batch(self, batch: Batch) -> torch.Tensor:
        """Compute CTC loss for one micro-batch. log_probs always cast to fp32."""
        audio = batch.audio.to(self.device)
        targets = batch.targets.to(self.device)
        target_lengths = batch.target_lengths.to(self.device)

        with torch.no_grad():
            mel, mel_lengths = self._mel_features(
                audio, batch.audio_lengths.to(self.device)
            )

        if self._spec_augmenter is not None and self.model.training:
            self._spec_augmenter.train()
            mel = self._spec_augmenter(mel, mel_lengths)

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.config.fp16_autocast,
            dtype=self.config.amp_dtype,
        ):
            encoder_out = self.model(mel, mel_lengths)

        log_probs_fp32 = encoder_out.log_probs.float()
        with torch.autocast(device_type=self.device.type, enabled=False):
            loss = self.ctc_loss(
                log_probs_fp32, targets, encoder_out.output_lengths, target_lengths
            )
        return loss

    def _run_validation(
        self, epoch: int
    ) -> tuple[float, dict[str, float], float, float, float, float]:
        """Run greedy-decode validation.

        Returns:
            (corpus_cer, per_speaker_cer, in_cer, out_cer, harmonic_cer, val_loss).
            in_cer/out_cer/harmonic_cer are zeros when config.in_domain_speakers is None.
        """
        self.model.eval()
        all_refs: list[str] = []
        all_hyps: list[str] = []
        all_spks: list[str] = []
        all_refs_digits: list[str] = []
        val_loss_sum = 0.0
        val_loss_count = 0

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                enabled=self.config.fp16_autocast,
                dtype=self.config.amp_dtype,
            ),
        ):
            for batch in self.val_loader:
                audio = batch.audio.to(self.device)
                audio_lengths = batch.audio_lengths.to(self.device)
                targets = batch.targets.to(self.device)
                target_lengths = batch.target_lengths.to(self.device)

                mel, mel_lengths = self._mel_features(audio, audio_lengths)
                encoder_out = self.model(mel, mel_lengths)

                log_probs_fp32 = encoder_out.log_probs.float()
                with torch.autocast(device_type=self.device.type, enabled=False):
                    loss = self.ctc_loss(
                        log_probs_fp32,
                        targets,
                        encoder_out.output_lengths,
                        target_lengths,
                    )
                val_loss_sum += loss.item()
                val_loss_count += 1

                decoded = greedy_decode(
                    encoder_out.log_probs, encoder_out.output_lengths, self.vocab
                )
                all_refs.extend(digits_to_words(t) for t in batch.transcriptions)
                all_hyps.extend(decoded)
                all_spks.extend(batch.spk_ids)
                all_refs_digits.extend(batch.transcriptions)

        corpus_cer = compute_cer(all_refs, all_hyps)
        per_spk = compute_per_speaker_cer(all_refs, all_hyps, all_spks)
        val_loss = val_loss_sum / max(val_loss_count, 1)

        if self.config.in_domain_speakers is None:
            return corpus_cer, per_spk, 0.0, 0.0, 0.0, val_loss

        all_hyps_digits = [safe_words_to_digits(h, fallback="") for h in all_hyps]
        in_cer, out_cer, harmonic_cer = compute_digit_cer_in_out_harmonic(
            all_refs_digits,
            all_hyps_digits,
            all_spks,
            self.config.in_domain_speakers,
        )
        return corpus_cer, per_spk, in_cer, out_cer, harmonic_cer, val_loss
