"""Training entrypoint for GP1 Russian spoken-numbers ASR.

Wires CONTRACTS.md §4 (data pipeline) + §5 (model registry) + §6 (losses)
+ §8 (Trainer) together into a CLI.

Typical invocation (local):

    python scripts/train.py \\
        --config configs/quartznet_10x4.yaml \\
        --train-csv /path/to/train.csv --train-root /path/to/train/ \\
        --dev-csv   /path/to/dev.csv   --dev-root   /path/to/dev/ \\
        --output-dir runs/quartznet_baseline

Kaggle-style invocation (single manifest pre-built):

    python scripts/train.py \\
        --config configs/quartznet_10x4.yaml \\
        --train-manifest data/train.jsonl \\
        --dev-manifest   data/dev.jsonl \\
        --output-dir runs/quartznet_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import torch
from torch.utils.data import DataLoader

from gp1.config import load_config as _load_config
from gp1.data.audio_aug import AudioAugmenter
from gp1.data.collate import DynamicBucketSampler, collate_fn
from gp1.data.dataset import SpokenNumbersDataset
from gp1.data.manifest import ManifestRecord, build_manifest, read_jsonl
from gp1.data.spec_aug import SpecAugmenter
from gp1.losses.cr_ctc import CRCTCLoss
from gp1.losses.ctc import CTCLoss
from gp1.losses.inter_ctc import InterCTCHead
from gp1.losses.word_aux import WordAuxCTCHead
from gp1.models.crdnn import CRDNN
from gp1.models.efficient_conformer import EfficientConformer
from gp1.models.fast_conformer_bpe import FastConformerBPE
from gp1.models.quartznet import QuartzNet10x4
from gp1.text.vocab import CharVocab
from gp1.text.vocab_word import WordVocab
from gp1.train.optim import build_adamw, build_novograd
from gp1.train.schedulers import build_cosine_warmup, build_noam
from gp1.train.trainer import Trainer, TrainerConfig
from gp1.types import AugConfig

# BPEVocab is imported lazily in _build_vocab to avoid hard dep on sentencepiece.
# The symbol is aliased here so tests can patch 'train.BPEVocab'.
try:
    from gp1.text.vocab_bpe import BPEVocab  # type: ignore[assignment]
except ImportError:  # pragma: no cover — sentencepiece not installed
    BPEVocab = None  # type: ignore[assignment,misc]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("gp1.train")


def _resolve_manifest(
    manifest_arg: Path | None, csv_arg: Path | None, root_arg: Path | None, cache: Path
) -> Path:
    """Return a jsonl manifest path, building from CSV if manifest_arg is None."""
    if manifest_arg is not None:
        return manifest_arg
    if csv_arg is None or root_arg is None:
        raise ValueError(
            "Provide either --{split}-manifest OR both --{split}-csv and --{split}-root"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    build_manifest(csv_arg, root_arg, cache)
    return cache


def _build_aug_config(cfg_aug: dict[str, Any], seed: int | None) -> AugConfig:
    """Project YAML aug block onto AugConfig, passing through fields we know."""
    allowed = {
        "speed_factors",
        "speed_prob",
        "vtlp_prob",
        "pitch_prob",
        "pitch_range_semitones",
        "gain_prob",
        "gain_db_range",
        "noise_prob",
        "noise_snr_db_range",
        "rir_prob",
        # H1: specaug parameters were previously filtered out; include them now.
        "specaug_freq_mask_param",
        "specaug_num_freq_masks",
        "specaug_time_mask_param",
        "specaug_num_time_masks",
        "specaug_time_mask_max_ratio",
    }
    kwargs = {k: v for k, v in (cfg_aug or {}).items() if k in allowed}
    if "speed_factors" in kwargs and isinstance(kwargs["speed_factors"], list):
        kwargs["speed_factors"] = tuple(kwargs["speed_factors"])
    return AugConfig(seed=seed, **kwargs)


def _build_spec_augmenter(aug_config: AugConfig) -> SpecAugmenter | None:
    """Build a SpecAugmenter from the resolved AugConfig, or None if all masks are 0."""
    # Only instantiate when at least one masking dimension is active.
    if (
        aug_config.specaug_num_freq_masks == 0
        and aug_config.specaug_num_time_masks == 0
    ):
        return None
    return SpecAugmenter(
        freq_mask_param=aug_config.specaug_freq_mask_param,
        num_freq_masks=aug_config.specaug_num_freq_masks,
        time_mask_param=aug_config.specaug_time_mask_param,
        num_time_masks=aug_config.specaug_num_time_masks,
        time_mask_max_ratio=aug_config.specaug_time_mask_max_ratio,
        seed=aug_config.seed,
    )


def _build_vocab(cfg: dict[str, Any], config_path: Path) -> CharVocab | Any:
    """Return the appropriate vocab based on cfg['text']['vocab_type'].

    Defaults to CharVocab when the 'text' section is absent or vocab_type == 'char'.
    For 'bpe', resolves bpe_model_path relative to the config file's parent directory
    when the path is not absolute, then constructs BPEVocab.
    """
    text_cfg = cfg.get("text") or {}
    vocab_type = text_cfg.get("vocab_type", "char").lower()
    if vocab_type == "char":
        return CharVocab()
    if vocab_type == "bpe":
        if BPEVocab is None:  # pragma: no cover
            raise ImportError(
                "sentencepiece is required for BPE vocab. "
                "Install it with: uv pip install sentencepiece"
            )
        raw_path = text_cfg.get("bpe_model_path")
        if raw_path is None:
            raise ValueError(
                "cfg.text.bpe_model_path is required when vocab_type == 'bpe'"
            )
        bpe_path = Path(raw_path)
        if not bpe_path.is_absolute():
            bpe_path = config_path.parent / bpe_path
        return BPEVocab(bpe_path)
    raise ValueError(f"Unknown vocab_type: {vocab_type!r}. Expected 'char' or 'bpe'.")


def _build_model(cfg: dict[str, Any], vocab_size: int) -> torch.nn.Module:
    """Instantiate the encoder model specified in cfg['model']['name'].

    Supported names: quartznet_10x4, crdnn, efficient_conformer, fast_conformer_bpe.
    Raises ValueError for unknown names.
    """
    model_cfg = cfg.get("model") or {}
    name = model_cfg.get("name", "quartznet_10x4").lower()

    if name == "quartznet_10x4":
        return QuartzNet10x4(
            vocab_size=vocab_size,
            d_model=int(model_cfg.get("d_model", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            subsample_factor=int(model_cfg.get("subsample_factor", 2)),
        )

    if name == "crdnn":
        return CRDNN(
            vocab_size=vocab_size,
            d_cnn=int(model_cfg.get("d_cnn", 64)),
            rnn_hidden=int(model_cfg.get("rnn_hidden", 256)),
            rnn_layers=int(model_cfg.get("rnn_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            subsample_factor=int(model_cfg.get("subsample_factor", 1)),
        )

    if name == "efficient_conformer":
        d_model_stages = model_cfg.get("d_model_stages", [96, 128, 128])
        n_blocks_per_stage = model_cfg.get("n_blocks_per_stage", [4, 4, 4])
        return EfficientConformer(
            vocab_size=vocab_size,
            d_model_stages=tuple(d_model_stages),
            n_blocks_per_stage=tuple(n_blocks_per_stage),
            n_heads=int(model_cfg.get("n_heads", 4)),
            ff_ratio=int(model_cfg.get("ff_ratio", 4)),
            conv_kernel=int(model_cfg.get("conv_kernel", 15)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )

    if name == "fast_conformer_bpe":
        return FastConformerBPE(
            vocab_size=vocab_size,
            d_model=int(model_cfg.get("d_model", 96)),
            n_blocks=int(model_cfg.get("n_blocks", 16)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            ff_ratio=int(model_cfg.get("ff_ratio", 4)),
            conv_kernel=int(model_cfg.get("conv_kernel", 9)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            subsample_factor=int(model_cfg.get("subsample_factor", 4)),
        )

    raise ValueError(
        f"Unknown model name: {name!r}. "
        "Expected one of: quartznet_10x4, crdnn, efficient_conformer, fast_conformer_bpe."
    )


def _build_inter_ctc(
    cfg: dict[str, Any], d_mid: int, vocab_size: int, blank_id: int
) -> InterCTCHead | None:
    """Return InterCTCHead when cfg['inter_ctc']['enabled'] is True, else None."""
    inter_cfg = cfg.get("inter_ctc") or {}
    if not inter_cfg.get("enabled", False):
        return None
    return InterCTCHead(d_mid=d_mid, vocab_size=vocab_size, blank_id=blank_id)


def _build_cr_ctc(cfg: dict[str, Any]) -> CRCTCLoss | None:
    """Return CRCTCLoss when cfg['cr_ctc']['enabled'] is True, else None."""
    cr_cfg = cfg.get("cr_ctc") or {}
    if not cr_cfg.get("enabled", False):
        return None
    return CRCTCLoss(
        temperature=float(cr_cfg.get("temperature", 1.0)),
        min_prob=float(cr_cfg.get("min_prob", 0.1)),
    )


def _build_train_dataset(
    records: list[ManifestRecord],
    vocab: Any,
    target_sr: int,
    augmenter: AudioAugmenter | None,
    word_vocab: WordVocab | None,
    return_two_views: bool = False,
    audio_cache_dir: Path | None = None,
) -> SpokenNumbersDataset:
    """Build the training dataset, enabling two-view audio when cr_ctc is active."""
    return SpokenNumbersDataset(
        records,
        vocab,
        target_samplerate=target_sr,
        augmenter=augmenter,
        word_vocab=word_vocab,
        return_two_views=return_two_views,
        audio_cache_dir=audio_cache_dir,
    )


def _build_dataloader(
    dataset: SpokenNumbersDataset,
    *,
    batch_size: int | None,
    sample_lengths: list[int] | None,
    max_tokens_per_batch: int | None,
    num_buckets: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    if max_tokens_per_batch and sample_lengths:
        sampler = DynamicBucketSampler(
            lengths=sample_lengths,
            max_tokens_per_batch=max_tokens_per_batch,
            num_buckets=num_buckets,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )
    assert batch_size is not None, (
        "either max_tokens_per_batch or batch_size must be set"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def _build_optimizer(cfg_opt: dict[str, Any], params):
    name = cfg_opt.get("name", "adamw").lower()
    lr = float(cfg_opt.get("lr", 1e-3))
    wd = float(cfg_opt.get("weight_decay", 1e-3))
    if name == "novograd":
        betas = tuple(cfg_opt.get("betas", (0.95, 0.5)))
        return build_novograd(params, lr=lr, betas=betas, weight_decay=wd)
    if name == "adamw":
        return build_adamw(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(
    cfg_sched: dict[str, Any], optimizer, d_model: int, total_steps: int
):
    name = cfg_sched.get("name", "cosine").lower()
    warmup = int(cfg_sched.get("warmup_steps", 1000))
    if name == "noam":
        return build_noam(optimizer, d_model=d_model, warmup_steps=warmup)
    if name == "cosine":
        min_ratio = float(cfg_sched.get("min_lr_ratio", 0.01))
        return build_cosine_warmup(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup,
            min_lr_ratio=min_ratio,
        )
    raise ValueError(f"Unknown scheduler: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="GP1 ASR training")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--dev-manifest", type=Path, default=None)
    parser.add_argument("--dev-csv", type=Path, default=None)
    parser.add_argument("--dev-root", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--audio-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional pre-resampled WAV cache directory produced by "
            "scripts/precompute_audio.py. When set, dataset items are loaded "
            "from cached 16-kHz PCM_16 WAVs instead of the original source files."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="If set, initialise wandb.init(project=..., name=output_dir.name)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = _load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    train_manifest_path = _resolve_manifest(
        args.train_manifest,
        args.train_csv,
        args.train_root,
        args.output_dir / "_manifest_train.jsonl",
    )
    dev_manifest_path = _resolve_manifest(
        args.dev_manifest,
        args.dev_csv,
        args.dev_root,
        args.output_dir / "_manifest_dev.jsonl",
    )
    train_records = read_jsonl(train_manifest_path)
    dev_records = read_jsonl(dev_manifest_path)
    log.info("Loaded %d train / %d dev records", len(train_records), len(dev_records))

    vocab = _build_vocab(cfg, args.config)
    aug_config = _build_aug_config(cfg.get("aug", {}), seed=args.seed)
    augmenter = AudioAugmenter(aug_config)
    # H1: build SpecAugmenter from the resolved aug config.
    spec_augmenter = _build_spec_augmenter(aug_config)

    word_aux_cfg = cfg.get("word_aux") or {}
    word_aux_enabled = bool(word_aux_cfg.get("enabled", False))
    word_vocab = WordVocab() if word_aux_enabled else None
    if word_aux_enabled:
        log.info("Word-aux CTC head enabled (word_vocab_size=%d)", word_vocab.size)

    # Build cr_ctc early — needed to set return_two_views on train_ds.
    cr_ctc_head = _build_cr_ctc(cfg)

    target_sr = int(cfg.get("audio", {}).get("samplerate", 16000))
    train_ds = _build_train_dataset(
        records=train_records,
        vocab=vocab,
        target_sr=target_sr,
        augmenter=augmenter,
        word_vocab=word_vocab,
        return_two_views=(cr_ctc_head is not None),
        audio_cache_dir=args.audio_cache_dir,
    )
    dev_ds = SpokenNumbersDataset(
        dev_records,
        vocab,
        target_samplerate=target_sr,
        augmenter=None,
        word_vocab=word_vocab,
        audio_cache_dir=args.audio_cache_dir,
    )

    data_cfg = cfg.get("data", {})
    # H2: use real audio durations instead of a fixed 2-second proxy.
    _target_sr: int = int(cfg.get("audio", {}).get("samplerate", 16000))
    train_loader = _build_dataloader(
        train_ds,
        batch_size=data_cfg.get("train_batch_size"),
        sample_lengths=[int(r.duration_s * _target_sr) for r in train_records],
        max_tokens_per_batch=data_cfg.get("max_tokens_per_batch"),
        num_buckets=int(data_cfg.get("num_buckets", 20)),
        shuffle=True,
        num_workers=args.num_workers,
    )
    dev_loader = _build_dataloader(
        dev_ds,
        batch_size=data_cfg.get("dev_batch_size", 16),
        sample_lengths=None,
        max_tokens_per_batch=None,
        num_buckets=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ----------------------------------------------------------------- model
    model_cfg = cfg.get("model", {})
    model = _build_model(cfg, vocab_size=vocab.vocab_size)
    model_name = model_cfg.get("name", "quartznet_10x4")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("%s trainable params: %d (%.2fM)", model_name, n_params, n_params / 1e6)

    ctc_loss = CTCLoss(blank_id=vocab.blank_id)

    # inter_ctc: derive d_mid from model attribute or fall back to cfg d_model.
    d_mid = getattr(model, "_d_mid", int(model_cfg.get("d_model", 256)))
    inter_ctc_head = _build_inter_ctc(
        cfg, d_mid=d_mid, vocab_size=vocab.vocab_size, blank_id=vocab.blank_id
    )

    word_aux_head: WordAuxCTCHead | None = None
    if word_aux_enabled:
        assert word_vocab is not None
        d_enc_word = int(word_aux_cfg.get("d_enc", model_cfg.get("d_model", 256)))
        word_aux_head = WordAuxCTCHead(
            d_enc=d_enc_word,
            word_vocab_size=word_vocab.size,
            blank_id=WordVocab.BLANK_ID,
        )

    # -------------------------------------------------------------- optimizer
    train_cfg = cfg.get("train", {})
    trainable_params: list = list(model.parameters())
    if inter_ctc_head is not None:
        trainable_params += list(inter_ctc_head.parameters())
    if word_aux_head is not None:
        trainable_params += list(word_aux_head.parameters())
    # CRCTCLoss has no learnable parameters; skip.
    optimizer = _build_optimizer(train_cfg.get("optimizer", {}), trainable_params)

    max_epochs = int(train_cfg.get("max_epochs", 50))
    steps_per_epoch = max(1, len(train_loader) // int(train_cfg.get("grad_accum", 1)))
    total_steps = max_epochs * steps_per_epoch
    scheduler = _build_scheduler(
        train_cfg.get("scheduler", {}),
        optimizer,
        d_model=int(model_cfg.get("d_model", 256)),
        total_steps=total_steps,
    )

    # -------------------------------------------------------------- trainer
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Training on device: %s", device)

    # H4: read grad_clip_norm from config (may be absent in old configs → None).
    _raw_grad_clip = train_cfg.get("grad_clip_norm")
    _grad_clip_norm: float | None = (
        float(_raw_grad_clip) if _raw_grad_clip is not None else None
    )

    trainer_config = TrainerConfig(
        max_epochs=max_epochs,
        grad_accum=int(train_cfg.get("grad_accum", 1)),
        fp16_autocast=bool(train_cfg.get("fp16_autocast", True))
        and device.type == "cuda",
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 50)),
        val_every_n_epochs=int(train_cfg.get("val_every_n_epochs", 1)),
        early_stop_patience=int(train_cfg.get("early_stop_patience", 15)),
        early_stop_metric=str(train_cfg.get("early_stop_metric", "max_speaker_cer")),
        ckpt_dir=args.output_dir / "checkpoints",
        grad_clip_norm=_grad_clip_norm,
    )

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb  # type: ignore[import-not-found]

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.output_dir.name,
                config={**cfg, "args": vars(args)},
            )
        except Exception as err:  # pragma: no cover — diagnostic only
            log.warning("wandb init failed (%s); continuing without wandb", err)

    if inter_ctc_head is not None:
        inter_ctc_head = inter_ctc_head.to(device)
    if cr_ctc_head is not None:
        cr_ctc_head = cr_ctc_head.to(device)
    if word_aux_head is not None:
        word_aux_head = word_aux_head.to(device)

    trainer = Trainer(
        model=model,
        ctc_loss=ctc_loss,
        inter_ctc=inter_ctc_head,
        cr_ctc=cr_ctc_head,
        word_aux=word_aux_head,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=dev_loader,
        vocab=vocab,
        config=trainer_config,
        device=device,
        wandb_run=wandb_run,
        audio_cfg=cfg.get("audio", {}),  # H8: wire audio config to LogMelFilterBanks
        spec_augmenter=spec_augmenter,  # H1: apply SpecAugment during training
    )

    result = trainer.fit()
    log.info(
        "Training finished. best_val_cer=%.4f best_ckpt=%s",
        result["best_val_cer"],
        result["best_ckpt_path"],
    )

    # RFC 8259 §6: JSON does not allow Infinity or NaN.  Represent an
    # "never-improved" CER (float("inf") initial sentinel) as JSON null so
    # that result.json is always valid standard JSON.  Downstream readers
    # (export.py, publish_release.sh) must treat null as "no CER available".
    _raw_cer = float(result["best_val_cer"])
    _cer_for_json: float | None = (
        None if (math.isinf(_raw_cer) or math.isnan(_raw_cer)) else _raw_cer
    )

    summary_path = args.output_dir / "result.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_val_cer": _cer_for_json,
                "best_ckpt_path": str(result["best_ckpt_path"]),
                "config_path": str(args.config),
                "n_params": n_params,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
