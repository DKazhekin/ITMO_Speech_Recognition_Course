"""Training entrypoint for GP1 Russian spoken-numbers ASR.

Wires CONTRACTS.md §4 (data pipeline) + §5 (QuartzNet10x4) + §6 (losses)
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
import sys
from pathlib import Path
from typing import Any

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import torch
import yaml
from torch.utils.data import DataLoader

from gp1.data.audio_aug import AudioAugmenter
from gp1.data.collate import DynamicBucketSampler, collate_fn
from gp1.data.dataset import SpokenNumbersDataset
from gp1.data.manifest import build_manifest, read_jsonl
from gp1.losses.ctc import CTCLoss
from gp1.models.quartznet import QuartzNet10x4
from gp1.text.vocab import CharVocab
from gp1.train.optim import build_adamw, build_novograd
from gp1.train.schedulers import build_cosine_warmup, build_noam
from gp1.train.trainer import Trainer, TrainerConfig
from gp1.types import AugConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("gp1.train")


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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
    }
    kwargs = {k: v for k, v in (cfg_aug or {}).items() if k in allowed}
    if "speed_factors" in kwargs and isinstance(kwargs["speed_factors"], list):
        kwargs["speed_factors"] = tuple(kwargs["speed_factors"])
    return AugConfig(seed=seed, **kwargs)


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
            pin_memory=True,
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
        pin_memory=True,
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
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
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

    vocab = CharVocab()
    aug_config = _build_aug_config(cfg.get("aug", {}), seed=args.seed)
    augmenter = AudioAugmenter(aug_config)

    target_sr = int(cfg.get("audio", {}).get("samplerate", 16000))
    train_ds = SpokenNumbersDataset(
        train_records, vocab, target_samplerate=target_sr, augmenter=augmenter
    )
    dev_ds = SpokenNumbersDataset(
        dev_records, vocab, target_samplerate=target_sr, augmenter=None
    )

    data_cfg = cfg.get("data", {})
    train_loader = _build_dataloader(
        train_ds,
        batch_size=data_cfg.get("train_batch_size"),
        sample_lengths=[int(r.samplerate * 2.0) for r in train_records],
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
    model = QuartzNet10x4(
        vocab_size=vocab.vocab_size,
        d_model=int(model_cfg.get("d_model", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("QuartzNet10x4 trainable params: %d (%.2fM)", n_params, n_params / 1e6)

    ctc_loss = CTCLoss(blank_id=vocab.blank_id)

    # -------------------------------------------------------------- optimizer
    train_cfg = cfg.get("train", {})
    optimizer = _build_optimizer(train_cfg.get("optimizer", {}), model.parameters())

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

    trainer = Trainer(
        model=model,
        ctc_loss=ctc_loss,
        inter_ctc=None,
        cr_ctc=None,
        word_aux=None,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=dev_loader,
        vocab=vocab,
        config=trainer_config,
        device=device,
        wandb_run=wandb_run,
    )

    result = trainer.fit()
    log.info(
        "Training finished. best_val_cer=%.4f best_ckpt=%s",
        result["best_val_cer"],
        result["best_ckpt_path"],
    )

    summary_path = args.output_dir / "result.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_val_cer": float(result["best_val_cer"]),
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
