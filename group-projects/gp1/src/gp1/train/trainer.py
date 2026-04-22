"""Training loop for GP1 ASR system.

CONTRACTS.md §8:
  TrainerConfig — frozen-ish dataclass with training hyperparameters.
  Trainer — orchestrates forward pass, loss computation, gradient accumulation,
    checkpointing, early stopping, and optional wandb logging.

Key design decisions:
  - CTC loss is computed in fp32 even when fp16_autocast=True.
    Pattern: exit autocast context → cast log_probs to float() → compute.
    Reference: NeMo ctc_models.py
    https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/ctc_models.py
  - Checkpoint format: {"model", "optimizer", "step", "epoch",
    "best_val_cer", "config"} per CONTRACTS.md §8 / Phase 0 spec.
  - Early stopping monitors max_speaker_cer (worst speaker CER in the batch).
    Lower is better. Patience counts epochs without strict improvement.
  - No hardcoded .cuda(); all tensors moved via self.device.
  - No print() — logging.getLogger(__name__) only.

mel-feature extraction is done inside the Trainer using the
``gp1.features.melbanks.LogMelFilterBanks`` module so the DataLoader
only needs to supply raw audio waveforms (Batch.audio).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from gp1.data.spec_aug import SpecAugmenter
from gp1.decoding.greedy import greedy_decode
from gp1.features.melbanks import LogMelFilterBanks
from gp1.text.vocab import CharVocab
from gp1.train.metrics import compute_cer, compute_per_speaker_cer
from gp1.types import Batch

logger = logging.getLogger(__name__)

_INF_CER = float("inf")


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Hyperparameters and bookkeeping settings for the Trainer.

    Args:
        max_epochs: Maximum number of training epochs.
        grad_accum: Gradient accumulation steps. Optimizer.step() is called
            once per ``grad_accum`` micro-batches.
        fp16_autocast: Enable ``torch.autocast`` for the forward pass.
            CTC loss is always computed in fp32 regardless of this flag.
        log_every_n_steps: Log training metrics every N global steps.
        val_every_n_epochs: Run validation every N epochs.
        early_stop_patience: Stop training if val metric does not improve
            for this many consecutive validation epochs.
        early_stop_metric: Which metric to monitor for early stopping.
            "max_speaker_cer" = worst per-speaker CER (lower is better).
        ckpt_dir: Directory to save checkpoints. Created automatically.
    """

    max_epochs: int
    grad_accum: int = 1
    fp16_autocast: bool = True
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1
    early_stop_patience: int = 15
    early_stop_metric: str = "max_speaker_cer"
    ckpt_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    grad_clip_norm: float | None = None  # H4: max gradient norm; None = no clipping


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Full training loop for GP1 ASR system.

    The trainer orchestrates:
    1. Mel-spectrogram extraction (LogMelFilterBanks, applied to raw audio).
    2. Forward pass through the acoustic encoder.
    3. Multi-loss computation: CTC + InterCTC + CR-CTC + word-aux CTC.
    4. Gradient accumulation with optimizer and scheduler stepping.
    5. Periodic validation + greedy decoding → CER.
    6. Checkpoint saving on metric improvement.
    7. Early stopping.
    8. Optional wandb logging.

    CTC fp32 island:
        Even when fp16_autocast=True, the log_probs passed to ctc_loss
        are converted to float32 before the loss call via ``.float()``.
        The autocast context is NOT exited explicitly; instead we rely on
        CTCLoss's internal cast (see gp1.losses.ctc.CTCLoss).

    Args:
        model: ASR encoder implementing the ASREncoder protocol.
        ctc_loss: CTCLoss instance.
        inter_ctc: InterCTCHead (or None to skip).
        cr_ctc: CRCTCLoss (or None to skip).
        word_aux: WordAuxCTCHead (or None to skip).
        optimizer: Torch optimizer.
        scheduler: LR scheduler (called scheduler.step() every global step).
        train_loader: Iterable DataLoader yielding ``gp1.types.Batch``.
        val_loader: Iterable DataLoader yielding ``gp1.types.Batch``.
        vocab: CharVocab for greedy decoding during validation.
        config: TrainerConfig.
        device: torch.device (CPU or CUDA). All tensors are moved here.
        wandb_run: Optional wandb run object for metric logging.
    """

    def __init__(
        self,
        model: nn.Module,
        ctc_loss: nn.Module,
        inter_ctc: nn.Module | None,
        cr_ctc: nn.Module | None,
        word_aux: nn.Module | None,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader,
        val_loader,
        vocab: CharVocab,
        config: TrainerConfig,
        device: torch.device,
        wandb_run=None,
        audio_cfg: dict[str, Any] | None = None,
        spec_augmenter: SpecAugmenter | None = None,
    ) -> None:
        self.model = model.to(device)
        self.ctc_loss = ctc_loss
        self.inter_ctc = inter_ctc
        self.cr_ctc = cr_ctc
        self.word_aux = word_aux
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.config = config
        self.device = device
        self.wandb_run = wandb_run

        # Mel-feature extractor lives on the same device as the model.
        # audio_cfg is passed through from the YAML config (H8 fix).
        self._mel = LogMelFilterBanks(**(audio_cfg or {})).to(device)

        # H1: optional SpecAugmenter — applied to mel after extraction,
        # before the encoder forward pass, training mode only.
        self._spec_augmenter: SpecAugmenter | None = spec_augmenter
        if self._spec_augmenter is not None:
            # Only call .to(device) on real nn.Module instances; a MagicMock
            # passed in tests would return a different mock object from .to().
            if isinstance(self._spec_augmenter, nn.Module):
                self._spec_augmenter = self._spec_augmenter.to(device)
            logger.info(
                "SpecAugmenter enabled: freq_mask_param=%d, time_mask_param=%d",
                self._spec_augmenter.freq_mask_param,
                self._spec_augmenter.time_mask_param,
            )
        else:
            logger.info("SpecAugmenter disabled (spec_augmenter=None).")

        # H6: fail fast if inter_ctc is configured but the model never
        # returns intermediate features. We run a single dry-run forward.
        if self.inter_ctc is not None:
            self._validate_inter_ctc_model(model)

        self._global_step: int = 0
        self._best_val_cer: float = _INF_CER
        self._best_ckpt_path: Path | None = None
        self._no_improve_epochs: int = 0

        config.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _validate_inter_ctc_model(self, model: nn.Module) -> None:
        """Dry-run the model with a minimal dummy input to check intermediate."""
        dummy_mel = torch.zeros(1, 80, 16, device=self.device)
        dummy_lengths = torch.tensor([16], dtype=torch.long, device=self.device)
        try:
            model.eval()
            with torch.no_grad():
                out = model(dummy_mel, dummy_lengths)
        finally:
            # Restore train mode regardless of outcome.
            model.train()

        if out.intermediate is None:
            raise ValueError(
                "InterCTC is enabled but the model returns intermediate=None. "
                "Use a model with an intermediate tap (e.g. QuartzNet10x4). "
                "CRDNN and FastConformerBPE do not support InterCTC."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, Any]:
        """Run the training loop for up to ``config.max_epochs`` epochs.

        Returns:
            dict with keys:
              "best_val_cer" (float) — best validation CER achieved.
              "best_ckpt_path" (Path) — path to the best checkpoint file.
        """
        try:
            from tqdm.auto import tqdm as _tqdm

            epoch_iter = _tqdm(
                range(1, self.config.max_epochs + 1),
                desc="epochs",
                dynamic_ncols=True,
            )
        except ImportError:
            epoch_iter = range(1, self.config.max_epochs + 1)

        last_executed_epoch = 0
        last_validated_epoch = 0

        for epoch in epoch_iter:
            self._train_epoch(epoch)
            last_executed_epoch = epoch

            if epoch % self.config.val_every_n_epochs == 0:
                val_cer = self._run_validation(epoch)
                last_validated_epoch = epoch
                improved = val_cer < self._best_val_cer

                if improved:
                    self._best_val_cer = val_cer
                    self._no_improve_epochs = 0
                    self._save_checkpoint(epoch, val_cer)
                else:
                    self._no_improve_epochs += 1

                logger.info(
                    "Epoch %d | val_cer=%.4f | best=%.4f | no_improve=%d/%d",
                    epoch,
                    val_cer,
                    self._best_val_cer,
                    self._no_improve_epochs,
                    self.config.early_stop_patience,
                )

                if self._wandb_active():
                    self.wandb_run.log(
                        {"val/cer": val_cer, "epoch": epoch},
                        step=self._global_step,
                    )

                if self._no_improve_epochs >= self.config.early_stop_patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        self._no_improve_epochs,
                    )
                    break

        # M11: run a final validation pass for the actual last executed epoch
        # when it was not already validated (early-stop at a non-val epoch, or
        # max_epochs not a multiple of val_every_n_epochs).
        if last_executed_epoch > last_validated_epoch:
            logger.info(
                "Running final validation for epoch %d (not covered by periodic val).",
                last_executed_epoch,
            )
            val_cer = self._run_validation(last_executed_epoch)
            improved = val_cer < self._best_val_cer
            if improved:
                self._best_val_cer = val_cer
                self._save_checkpoint(last_executed_epoch, val_cer)

        # If we never improved (e.g. model immediately returns 1.0 CER),
        # create an initial checkpoint so callers always get a valid path.
        if self._best_ckpt_path is None:
            self._save_checkpoint(epoch=0, val_cer=self._best_val_cer)

        return {
            "best_val_cer": self._best_val_cer,
            "best_ckpt_path": self._best_ckpt_path,
        }

    # ------------------------------------------------------------------
    # Internal: training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> None:
        """Run one full pass over train_loader."""
        self.model.train()
        self.optimizer.zero_grad()

        try:
            from tqdm.auto import tqdm as _tqdm

            loader_iter = enumerate(
                _tqdm(
                    self.train_loader,
                    desc=f"epoch {epoch} train",
                    leave=False,
                    dynamic_ncols=True,
                )
            )
        except ImportError:
            loader_iter = enumerate(self.train_loader)

        for micro_idx, batch in loader_iter:
            loss = self._forward_batch(batch)

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.grad_accum
            scaled_loss.backward()

            is_last_micro = (micro_idx + 1) % self.config.grad_accum == 0

            if is_last_micro:
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.grad_clip_norm,
                    )
                    logger.debug(
                        "Gradient clipped to max_norm=%.4f", self.config.grad_clip_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self._global_step += 1

                if self._global_step % self.config.log_every_n_steps == 0:
                    logger.info(
                        "Epoch %d | step %d | loss=%.4f | lr=%.2e",
                        epoch,
                        self._global_step,
                        loss.item(),
                        self.scheduler.get_last_lr()[0],
                    )
                    if self._wandb_active():
                        self.wandb_run.log(
                            {
                                "train/loss": loss.item(),
                                "train/lr": self.scheduler.get_last_lr()[0],
                            },
                            step=self._global_step,
                        )

        # Handle leftover micro-batches (when len(loader) % grad_accum != 0)
        # We do a final step if there are any accumulated gradients.
        remainder = len(self.train_loader) % self.config.grad_accum
        if remainder != 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self._global_step += 1

    def _forward_batch(self, batch: Batch) -> torch.Tensor:
        """Compute combined loss for one micro-batch.

        CTC log_probs are always cast to float32 before being passed to
        ctc_loss, ensuring the fp32 island invariant from CONTRACTS.md §11.

        Args:
            batch: A ``gp1.types.Batch`` instance.

        Returns:
            Combined scalar loss tensor (with grad).
        """
        # Move audio to device
        audio = batch.audio.to(self.device)  # [B, T_audio]
        audio_lengths = batch.audio_lengths.to(self.device)
        targets = batch.targets.to(self.device)
        target_lengths = batch.target_lengths.to(self.device)

        # Mel extraction: [B, T_audio] → [B, n_mels, T_frames]
        with torch.no_grad():
            mel = self._mel(audio)  # [B, 80, T_frames]
            # Compute mel frame lengths: center=True STFT pads by n_fft//2 on
            # both sides, so T_frames = L // hop + 1. Clamp to guard against
            # any frontend edge-case (C1 fix).
            hop = self._mel.hop_length
            mel_lengths = (audio_lengths // hop + 1).clamp(max=mel.size(-1)).long()

        # H1: apply SpecAugmenter in training mode (after mel, before encoder).
        # SpecAugmenter itself respects self.training; the guard here is an
        # extra safety belt to avoid masking validation batches.
        if self._spec_augmenter is not None and self.model.training:
            self._spec_augmenter.train()
            mel = self._spec_augmenter(mel, mel_lengths)

        autocast_ctx = torch.autocast(
            device_type=self.device.type,
            enabled=self.config.fp16_autocast,
        )

        with autocast_ctx:
            encoder_out = self.model(mel, mel_lengths)

        log_probs = encoder_out.log_probs  # [B, T', V]
        output_lengths = encoder_out.output_lengths

        # --- CTC loss in fp32 ---
        # CTCLoss internally casts to float, but we make the fp32 island
        # explicit here by calling .float() before the loss.
        ctc_lp_fp32 = log_probs.float()
        loss_ctc = self.ctc_loss(ctc_lp_fp32, targets, output_lengths, target_lengths)

        total_loss = loss_ctc

        # --- CR-CTC consistency loss (two-view) ---
        # Requires a second augmented view (batch.audio_view2) and computes
        # the symmetric KL between the two log-prob outputs. Reference:
        # "Consistency Regularisation for CTC" (Yao et al. 2024).
        if self.cr_ctc is not None:
            if batch.audio_view2 is None or batch.audio_view2_lengths is None:
                raise ValueError(
                    "cr_ctc head is active but batch.audio_view2 is None — "
                    "construct the Dataset with return_two_views=True to populate it."
                )
            audio2 = batch.audio_view2.to(self.device)
            audio2_lengths = batch.audio_view2_lengths.to(self.device)
            with torch.no_grad():
                mel2 = self._mel(audio2)
                mel2_lengths = (
                    (audio2_lengths // hop + 1).clamp(max=mel2.size(-1)).long()
                )
            with autocast_ctx:
                encoder_out2 = self.model(mel2, mel2_lengths)
            log_probs2 = encoder_out2.log_probs.float()
            # Align time dimension: CR-CTC requires both views to have the
            # same T'. Truncate to the min (both from the same underlying
            # utterance so lengths are typically identical up to rounding).
            t_min = min(log_probs.size(1), log_probs2.size(1))
            loss_cr = self.cr_ctc(
                ctc_lp_fp32[:, :t_min, :],
                log_probs2[:, :t_min, :],
                torch.clamp(output_lengths, max=t_min),
            )
            total_loss = total_loss + 0.2 * loss_cr

        # --- InterCTC auxiliary loss ---
        if self.inter_ctc is not None and encoder_out.intermediate is not None:
            loss_inter = self.inter_ctc(
                encoder_out.intermediate, output_lengths, targets, target_lengths
            )
            total_loss = total_loss + 0.3 * loss_inter

        # --- Word-aux CTC ---
        if self.word_aux is not None:
            if batch.word_targets is None or batch.word_target_lengths is None:
                raise ValueError(
                    "word_aux head is active but batch.word_targets is None — "
                    "construct the Dataset with word_vocab=WordVocab() to populate it."
                )
            if encoder_out.intermediate is None:
                raise ValueError(
                    "word_aux head requires encoder_out.intermediate features "
                    "[B, T', D_mid]; the current encoder does not expose them."
                )
            word_targets = batch.word_targets.to(self.device)
            word_target_lengths = batch.word_target_lengths.to(self.device)
            loss_word = self.word_aux(
                encoder_out.intermediate,
                output_lengths,
                word_targets,
                word_target_lengths,
            )
            total_loss = total_loss + 0.1 * loss_word

        return total_loss

    # ------------------------------------------------------------------
    # Internal: validation
    # ------------------------------------------------------------------

    def _run_validation(self, epoch: int) -> float:
        """Run greedy-decode validation and compute CER / per-speaker CER.

        Args:
            epoch: Current epoch (used for logging only).

        Returns:
            Scalar val CER (lower is better). If ``early_stop_metric`` is
            "max_speaker_cer", returns the worst per-speaker CER.
        """
        self.model.eval()
        all_refs: list[str] = []
        all_hyps: list[str] = []
        all_spks: list[str] = []

        with torch.no_grad():
            for batch in self.val_loader:
                audio = batch.audio.to(self.device)
                audio_lengths = batch.audio_lengths.to(self.device)

                mel = self._mel(audio)
                hop = self._mel.hop_length
                # center=True STFT: T_frames = L // hop + 1. Clamp to guard
                # against edge cases. Same formula as _forward_batch (C1 fix).
                mel_lengths = (audio_lengths // hop + 1).clamp(max=mel.size(-1)).long()

                encoder_out = self.model(mel, mel_lengths)
                decoded = greedy_decode(
                    encoder_out.log_probs,
                    encoder_out.output_lengths,
                    self.vocab,
                )

                all_refs.extend(batch.transcriptions)
                all_hyps.extend(decoded)
                all_spks.extend(batch.spk_ids)

        corpus_cer = compute_cer(all_refs, all_hyps)
        per_spk = compute_per_speaker_cer(all_refs, all_hyps, all_spks)

        if per_spk:
            max_spk_cer = max(per_spk.values())
        else:
            max_spk_cer = corpus_cer

        if self.config.early_stop_metric == "max_speaker_cer":
            monitored = max_spk_cer
        else:
            monitored = corpus_cer

        logger.info(
            "Validation epoch %d | corpus_cer=%.4f | max_spk_cer=%.4f",
            epoch,
            corpus_cer,
            max_spk_cer,
        )

        if self._wandb_active():
            self.wandb_run.log(
                {
                    "val/corpus_cer": corpus_cer,
                    "val/max_speaker_cer": max_spk_cer,
                    **{f"val/cer_{spk}": cer for spk, cer in per_spk.items()},
                },
                step=self._global_step,
            )

        return monitored

    # ------------------------------------------------------------------
    # Internal: checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_cer: float) -> None:
        """Save model + optimizer state to disk.

        Checkpoint format (CONTRACTS.md §8 / Phase 0):
          {
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "step":          int,
            "epoch":         int,
            "best_val_cer":  float,
            "config":        dict (asdict of TrainerConfig),
          }

        Args:
            epoch: Current epoch.
            val_cer: Validation CER to embed in the filename and checkpoint.
        """
        ckpt_name = f"epoch{epoch:04d}_cer{val_cer:.4f}.pt"
        ckpt_path = self.config.ckpt_dir / ckpt_name

        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._global_step,
            "epoch": epoch,
            "best_val_cer": val_cer,
            "config": asdict(self.config),
        }
        torch.save(payload, ckpt_path)
        self._best_ckpt_path = ckpt_path
        logger.info("Checkpoint saved: %s", ckpt_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wandb_active(self) -> bool:
        return self.wandb_run is not None
