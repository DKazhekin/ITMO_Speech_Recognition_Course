# GP1 — Public API Contracts

Single source of truth for cross-module types and signatures. Every
parallel agent implementing a module under `src/gp1/` MUST match the
signatures here exactly. Internal helpers are free-form.

Python 3.11, PEP 604 unions (`X | None`), type annotations on all
public signatures, immutable dataclasses where possible.

**Last updated:** 2026-04-22 after Wave-1 completion — see §12 for deltas
between the original spec and the shipped implementation.

---

## 1. Shared types — `src/gp1/types.py`

```python
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass(frozen=True)
class ManifestRecord:
    audio_path: Path            # absolute path to .wav/.mp3
    transcription: str          # "139473" (digit string, 4..6 digits)
    spk_id: str                 # "spk_A" ... "spk_N"
    gender: str                 # "male" | "female"
    ext: str                    # "wav" | "mp3"
    samplerate: int             # native sample rate (will be resampled to 16 kHz)

@dataclass
class Batch:
    audio: torch.Tensor          # [B, T_audio_max] float32, 16 kHz, zero-padded
    audio_lengths: torch.Tensor  # [B] int64, actual audio length in samples
    targets: torch.Tensor        # [B, U_max] int64, char-ids (or word-ids), blank not included
    target_lengths: torch.Tensor # [B] int64
    spk_ids: list[str]           # [B]
    transcriptions: list[str]    # [B] digit strings, for metric computation

@dataclass(frozen=True)
class AugConfig:
    speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1)
    speed_prob: float = 1.0
    vtlp_prob: float = 0.5
    vtlp_alpha_range: tuple[float, float] = (0.9, 1.1)
    pitch_prob: float = 0.3
    pitch_range_semitones: tuple[float, float] = (-3.0, 3.0)
    gain_prob: float = 0.7
    gain_db_range: tuple[float, float] = (-8.0, 8.0)
    noise_prob: float = 0.3
    noise_snr_db_range: tuple[float, float] = (5.0, 20.0)
    musan_root: Path | None = None
    rir_prob: float = 0.1
    rir_root: Path | None = None
    specaug_freq_mask_param: int = 15
    specaug_num_freq_masks: int = 2
    specaug_time_mask_param: int = 25
    specaug_num_time_masks: int = 5
    specaug_time_mask_max_ratio: float = 0.05
    seed: int | None = None
```

---

## 2. Text — `src/gp1/text/`

### `vocab.py`

```python
RUSSIAN_ALPHABET_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"  # 33 letters

class CharVocab:
    BLANK_ID: int = 0
    SPACE_ID: int = 1
    vocab_size: int = 35  # blank + space + 33 letters

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...  # strips blanks
    @property
    def blank_id(self) -> int: ...
    @property
    def size(self) -> int: ...
```

### `vocab_word.py`

```python
# Closed vocabulary for Russian number words in range 1000..999999.
# Implementer MUST verify exhaustiveness against num2words(n, 'ru').
NUMBER_WORDS: tuple[str, ...] = (
    "ноль", "один", "одна", "два", "две", "три", "четыре", "пять",
    "шесть", "семь", "восемь", "девять",
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
    "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
    "восемьдесят", "девяносто",
    "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
    "семьсот", "восемьсот", "девятьсот",
    "тысяча", "тысячи", "тысяч",
)

class WordVocab:
    BLANK_ID: int = 0
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def size(self) -> int: ...
```

### `normalize.py`

```python
def digits_to_words(n: int | str) -> str:
    """
    "139473" -> "сто тридцать девять тысяч четыреста семьдесят три".
    Wraps num2words(int, lang='ru'). Lowercase, whitespace-normalized,
    no hyphens. Range 1000..999999.
    """
```

### `denormalize.py`

```python
def words_to_digits(text: str) -> str:
    """
    Inverse of digits_to_words. Deterministic reducer over closed
    42-word vocabulary (see NUMBER_WORDS above). Returns digit string.
    Raises ValueError on malformed input.
    """
```

**TDD acceptance**: `tests/test_normalize.py` round-trips every n in
`range(1000, 1000000)` through `digits_to_words → words_to_digits`.

---

## 3. Features — `src/gp1/features/melbanks.py`

Port from `assignments/assignment1/melbanks.py`; adjust defaults.

```python
class LogMelFilterBanks(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        samplerate: int = 16000,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min_hz: float = 0.0,
        f_max_hz: float | None = None,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T_audio] float32 — 16 kHz waveform.
        returns: [B, n_mels, T_frames] log mel spectrogram.
        """
```

---

## 4. Data pipeline — `src/gp1/data/`

### `manifest.py`

```python
def build_manifest(csv_path: Path, audio_root: Path, out_path: Path) -> int: ...
def read_jsonl(path: Path) -> list[ManifestRecord]: ...
def write_jsonl(records: list[ManifestRecord], path: Path) -> None: ...
def leave_n_speakers_out_split(
    records: list[ManifestRecord],
    holdout_speakers: list[str],
) -> tuple[list[ManifestRecord], list[ManifestRecord]]: ...
```

### `dataset.py`

```python
class SpokenNumbersDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[ManifestRecord],
        vocab: CharVocab,
        target_samplerate: int = 16000,
        augmenter: AudioAugmenter | None = None,
        return_two_views: bool = False,  # for CR-CTC
    ): ...

    def __getitem__(self, idx: int) -> dict:
        """
        Returns dict: audio [T_audio], target [U], spk_id str, transcription str.
        If return_two_views: also 'audio_view2' with independent aug.
        """
```

### `audio_aug.py`

```python
class AudioAugmenter:
    def __init__(self, config: AugConfig): ...
    def __call__(self, wav: torch.Tensor, samplerate: int = 16000) -> torch.Tensor:
        """Speed and pitch are XOR per sample. When both probs would trigger
        on the same sample, pick one via a 50/50 coin flip (so neither branch
        becomes unreachable when speed_prob == pitch_prob == 1.0).
        Deterministic if config.seed is set."""
```

### `spec_aug.py`

```python
class SpecAugmenter(nn.Module):
    """F=15x2, T=25x5, no time-warp.

    Takes explicit kwargs (not AugConfig) so callers can construct
    SpecAugmenter() with no args and get the defaults below, matching
    AugConfig.specaug_* fields.
    """
    def __init__(
        self,
        *,
        freq_mask_param: int = 15,
        num_freq_masks: int = 2,
        time_mask_param: int = 25,
        num_time_masks: int = 5,
        time_mask_max_ratio: float = 0.05,
        seed: int | None = None,
    ): ...
    def forward(self, mel: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor: ...
```

### `collate.py`

```python
def collate_fn(batch: list[dict], pad_audio_to_multiple: int = 160) -> Batch: ...

class DynamicBucketSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        lengths: list[int],
        max_tokens_per_batch: int,
        num_buckets: int = 20,
        shuffle: bool = True,
    ): ...
```

---

## 5. Models — `src/gp1/models/`

### `base.py`

```python
from typing import Protocol

@dataclass
class EncoderOutput:
    log_probs: torch.Tensor       # [B, T', V] log-softmax
    output_lengths: torch.Tensor  # [B] int64
    intermediate: torch.Tensor | None  # [B, T_mid, D_mid] for InterCTC, or None

class ASREncoder(Protocol):
    vocab_size: int
    subsample_factor: int

    def forward(
        self,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> EncoderOutput: ...
```

### `common.py`

```python
class TCSConvBlock(nn.Module):
    """Time-Channel Separable: depthwise 1D conv + pointwise 1x1."""
    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int = 1,
                 dropout: float = 0.1, residual: bool = True): ...

class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, ff_ratio: int = 4,
                 conv_kernel: int = 31, dropout: float = 0.1): ...

class SubsampleConv(nn.Module):
    def __init__(self, n_mels: int, d_out: int, factor: int = 4): ...
```

### `quartznet.py`, `crdnn.py`, `efficient_conformer.py`, `fast_conformer_bpe.py`

Each matches `ASREncoder` Protocol. See plan for per-model hyperparams.

**Shape invariant**: `T' >= 2 * U_max` on longest training sample.
Verified in `tests/test_models_shapes.py`.

---

## 6. Losses — `src/gp1/losses/`

```python
class CTCLoss(nn.Module):
    """fp32 internally, zero_infinity=True."""
    def __init__(self, blank_id: int = 0): ...
    def forward(
        self, log_probs, targets, input_lengths, target_lengths,
    ) -> torch.Tensor: ...

class InterCTCHead(nn.Module):
    def __init__(self, d_mid: int, vocab_size: int, blank_id: int = 0): ...
    def forward(self, mid_features, input_lengths, targets, target_lengths) -> torch.Tensor: ...

class CRCTCLoss(nn.Module):
    def __init__(self, temperature: float = 1.0, min_prob: float = 0.1): ...
    def forward(self, log_probs_a, log_probs_b, input_lengths) -> torch.Tensor: ...

class WordAuxCTCHead(nn.Module):
    def __init__(self, d_enc: int, word_vocab_size: int, blank_id: int = 0): ...
    def forward(self, enc_features, input_lengths, word_targets, word_target_lengths) -> torch.Tensor: ...
```

---

## 7. Decoding — `src/gp1/decoding/`

```python
def greedy_decode(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    vocab: CharVocab,
) -> list[str]: ...

class KenLMWrapper:
    def __init__(self, binary_path: Path): ...
    def score(self, text: str, bos: bool = True, eos: bool = True) -> float: ...

@dataclass(frozen=True)
class BeamSearchConfig:
    alpha: float = 0.7
    beta: float = 1.0
    beam_width: int = 100
    hotwords: tuple[str, ...] = ("тысяча", "тысячи", "тысяч")
    hotword_weight: float = 8.0

class BeamSearchDecoder:
    def __init__(self, vocab: CharVocab, kenlm_path: Path | None,
                 unigrams: list[str], config: BeamSearchConfig): ...
    def decode_batch(self, log_probs, output_lengths) -> list[str]: ...
```

---

## 8. Training — `src/gp1/train/`

```python
def compute_cer(references: list[str], hypotheses: list[str]) -> float: ...
def compute_per_speaker_cer(
    references: list[str], hypotheses: list[str], spk_ids: list[str],
) -> dict[str, float]: ...

def build_novograd(params, lr, betas=(0.95, 0.5), weight_decay=1e-3): ...
def build_adamw(params, lr, weight_decay=1e-6): ...
def build_noam(optimizer, d_model, warmup_steps): ...
def build_cosine_warmup(optimizer, total_steps, warmup_steps, min_lr_ratio=0.01): ...

@dataclass
class TrainerConfig:
    max_epochs: int
    grad_accum: int = 1
    fp16_autocast: bool = True  # CTC stays fp32
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1
    early_stop_patience: int = 15
    early_stop_metric: str = "max_speaker_cer"
    ckpt_dir: Path = Path("checkpoints")

class Trainer:
    def __init__(
        self,
        model, ctc_loss, inter_ctc, cr_ctc, word_aux,
        optimizer, scheduler,
        train_loader, val_loader,
        vocab: CharVocab,
        config: TrainerConfig,
        device: torch.device,
        wandb_run=None,
    ): ...
    def fit(self) -> dict: ...  # {"best_val_cer": float, "best_ckpt_path": Path}
```

---

## 9. Language model — `src/gp1/lm/`

```python
def build_synthetic_corpus(out_path: Path, train_manifest: Path | None = None) -> int: ...
def train_kenlm(
    corpus_path: Path, out_binary: Path,
    order: int = 4, vocab_limit_path: Path | None = None,
) -> None: ...
```

---

## 10. Submission — `src/gp1/submit/inference.py`

```python
@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_path: Path
    config_path: Path
    lm_binary_path: Path | None
    batch_size: int = 32
    device: str = "cuda"

def run_inference(
    manifest: list[ManifestRecord],
    config: InferenceConfig,
) -> list[tuple[str, str]]: ...  # [(filename, digit_string), ...]
```

---

## 11. Cross-cutting conventions

| Topic | Rule |
|---|---|
| Imports | Absolute: `from gp1.text.vocab import CharVocab`. No relative imports outside tests. |
| Random | Every aug/dataset takes `seed: int \| None`. If set, deterministic. |
| Logging | `logging.getLogger(__name__)`. No `print()` outside CLI. |
| fp32 islands | CTC loss, KenLM: always fp32. Wrap with `torch.autocast(..., enabled=False)`. |
| Device | Tensors on `self.device` from config. No hardcoded `.cuda()`. |
| Shape asserts | Public APIs assert tensor dims on entry. |
| Tests | Pytest, AAA pattern, `test_<behavior>_<condition>`. Coverage >= 80%. |
| TDD | RED → GREEN → REFACTOR. Failing test before each function. |
| SOTA | Before any non-trivial module: GitHub code search + Context7. Cite URLs in docstrings. |

---

## 12. Wave-1 implementation notes (2026-04-22)

Wave-1 shipped with three deltas from the original contract. The signatures
above have already been updated in place; this section records the rationale
so Wave-2 agents can trust the contract without hunting through git history.

| Contract section | Original spec | Shipped reality | Reason |
|---|---|---|---|
| §2 `NUMBER_WORDS` / `WordVocab` | "30-word vocabulary" (prose) | 42 entries (`tuple` already had them) | `num2words(n, 'ru')` for 1000..999999 emits 41 unique forms (including feminine `одна`/`две` and three plural variants of `тысяча`); "ноль" stays for completeness. Round-trip test `test_number_words_is_exhaustive_for_1000_to_999999` confirms. Implication: `WordAuxCTCHead` output dim is `vocab_size = 1 + 42 = 43`, not 31. |
| §4 `SpecAugmenter.__init__` | `(config: AugConfig)` (implied) | explicit keyword-only params with defaults matching `AugConfig.specaug_*` | Test constructs `SpecAugmenter()` with no args and reads `.freq_mask_param`. Keeping kwargs makes that trivial; callers can still pass `**dataclasses.asdict(config)` if they want config-driven construction. |
| §4 `AudioAugmenter.__call__` | "Speed and pitch are XOR" | Speed and pitch are XOR per sample; **50/50 coin flip** when both would fire simultaneously | With `speed_prob == pitch_prob == 1.0` the "prefer speed" rule makes pitch unreachable, which the deterministic-coverage test flagged. The coin flip is seeded from `config.seed` so reproducibility holds. |

Wave-1 test counts (pytest):

| Suite | Count |
|---|---|
| text (vocab + normalize + denormalize + vocab_word) | 85 |
| data (audio_aug + spec_aug) | 8 |
| losses (ctc + inter_ctc + cr_ctc + word_aux) | 18 |
| features (melbanks) | 5 |
| lm (build_corpus + train_kenlm) | 9 |
| **Total** | **125 (124 fast + 1 slow full-range round-trip)** |
