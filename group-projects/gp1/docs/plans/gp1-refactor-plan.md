# План: унификация ноутбуков gp1 + KenLM-декодинг с эвристиками

## Context

В проекте `gp1` (Russian spoken-numbers ASR, ITMO course) пользователь обучил модель локально через `notebooks/01_quartznet.ipynb`. Этот ноутбук содержит «канонический» стиль data-setup: env-hardening (PYTORCH_CUDA_ALLOC_CONF, TF32, cudnn.benchmark=False), путь до `notebooks/data`, AUDIO_CACHE preload + `share_memory_()`, `_worker_init` (per-worker BLAS+RNG), DataLoader с `persistent_workers`/`prefetch_factor`. Остальные ноутбуки (02–04, 05_predict, и `experiments/01a-c`, `experiments/06`) написаны менее аккуратно — нет env-фиксов, нет `worker_init_fn`, нет `share_memory_()`, размер батча/воркеров hardcoded и расходится.

Цели:
1. Перенести канонический стиль 01 в **02, 03, 04, 05_predict, 01a, 01b, 01c, 06** (inline, без shared utility).
2. Создать новый ноутбук **`notebooks/07_decode_kenlm.ipynb`**, который берёт чекпоинт CharVocab-модели, тренирует KenLM, запускает beam-search с максимальным набором эвристик (hard lexicon + hotwords + alpha/beta grid + закрытый FSA-rescoring через `words_to_digits`-валидатор) и считает финальные метрики (CER на словах + CER на цифрах + per-speaker).
3. **Добавить новую training-валидационную метрику**: гармоническое среднее digit-CER (после `safe_words_to_digits`) по двум подгруппам dev — **in-domain спикерам** (spk_id ∈ train) и **out-of-domain спикерам** (spk_id ∉ train). Сделать её **дефолтной** для `early_stop_metric` (no_improve) в `TrainerConfig`. Прокинуть в **все training-ноутбуки** (01, 02, 03, 04, 01a, 01b, 01c).
4. **Аккуратное per-epoch логирование** в две строки: train (loss) и val (loss + harmonic CER + in/out breakdown + best + no_improve). Без тестов.

Данные уже распакованы в `notebooks/data/` — шаг скачивания делаем **idempotent**: gdown+unzip только если целевая папка не существует.

## Часть 1 — Унификация data-setup в ноутбуках

### Канонический шаблон (копируется in-place в каждый ноутбук)

**Cell 1 — Idempotent download:**
```python
import os
from pathlib import Path

DATA_ROOT = Path("notebooks/data")  # или абсолютный путь от корня репо
if not DATA_ROOT.exists() or not any(DATA_ROOT.iterdir()):
    import gdown, zipfile
    zip_path = Path("data.zip")
    if not zip_path.exists():
        gdown.download(
            url="https://drive.google.com/file/d/1WOubhQ4LtPYEZTOHNkZiDqIobfOQEWBW/view?usp=share_link",
            output=str(zip_path), quiet=False, fuzzy=True,
        )
    zipfile.ZipFile(zip_path).extractall(DATA_ROOT)
```

**Cell 2 — Env hardening (до `import torch`):**
```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
torch.backends.cudnn.benchmark = False           # variable-length batches
torch.set_float32_matmul_precision("high")       # TF32 для matmul
```

**Cell 3 — Paths (локальные):**
```python
TRAIN_ROOT = DATA_ROOT / "train"
DEV_ROOT   = DATA_ROOT / "dev"
TEST_ROOT  = DATA_ROOT / "test"          # may not exist
TRAIN_CSV  = TRAIN_ROOT / "train.csv"
DEV_CSV    = DEV_ROOT / "dev.csv"
CKPT_ROOT  = Path("checkpoints") / "<arch_name>"
for p in (TRAIN_ROOT, DEV_ROOT, TRAIN_CSV, DEV_CSV):
    assert p.exists(), p
CKPT_ROOT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Cell 4 — FIXED dict:** объединить `samplerate/n_fft/n_mels/hop_length/win_length/max_epochs/grad_accum/subsample_factor` в `FIXED` (как в 01).

**Cell 5 — Manifest + AUDIO_CACHE:**
```python
from gp1.data.manifest import records_from_csv
from gp1.data.dataset import preload_audio_cache
train_records = records_from_csv(TRAIN_CSV, TRAIN_ROOT)
dev_records   = records_from_csv(DEV_CSV, DEV_ROOT)
AUDIO_CACHE = preload_audio_cache(train_records + dev_records,
                                  target_samplerate=FIXED["samplerate"])
for k in list(AUDIO_CACHE.keys()):
    AUDIO_CACHE[k] = AUDIO_CACHE[k].contiguous().share_memory_()
```

**Cell 6 — `_worker_init` + DataLoader:**
```python
import random
def _worker_init(worker_id: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    info = torch.utils.data.get_worker_info()
    aug = getattr(info.dataset, "_augmenter", None) if info else None
    if aug is not None and hasattr(aug, "_rng"):
        aug._rng = random.Random(info.seed & 0xFFFFFFFF)

BATCH_SIZE = 64        # 02/01a/01b/01c=64, 03/04=16
DL_WORKERS = 4

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, num_workers=DL_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2, worker_init_fn=_worker_init)
```

### Скоуп изменений по ноутбукам

| Notebook | BATCH_SIZE | DL_WORKERS | Доп. правки |
|---|---|---|---|
| `02_crdnn.ipynb` | 64 | 4 | Заменить `os.cpu_count()` на 4 |
| `03_efficient_conformer.ipynb` | 16 | 4 | то же |
| `04_fast_conformer_bpe.ipynb` | 16 | 4 | BPE-vocab сохранить, data-plumbing идентично; **+ inline LM-блок** (см. Часть 2-bis) |
| `05_predict.ipynb` | — | — | только env-cell + paths-cell + idempotent download; AUDIO_CACHE не нужен |
| `experiments/01a_quartznet_inter_ctc.ipynb` | 32 | 4 | manual training loop оставить |
| `experiments/01b_quartznet_word_aux.ipynb` | 32 | 4 | то же |
| `experiments/01c_quartznet_cr_ctc.ipynb` | 32 | 4 | то же |
| `experiments/06_kenlm_beam_rescore.ipynb` | — | — | env+paths правки; основная логика будет в новом 07 |

### Файлы к редактированию (Часть 1)
- `notebooks/02_crdnn.ipynb`
- `notebooks/03_efficient_conformer.ipynb`
- `notebooks/04_fast_conformer_bpe.ipynb`
- `notebooks/05_predict.ipynb`
- `notebooks/experiments/01a_quartznet_inter_ctc.ipynb`
- `notebooks/experiments/01b_quartznet_word_aux.ipynb`
- `notebooks/experiments/01c_quartznet_cr_ctc.ipynb`
- `notebooks/experiments/06_kenlm_beam_rescore.ipynb`

---

## Часть 1-bis — Новая training-валидационная метрика (harmonic in/out digit-CER)

### Контракт

```python
# src/gp1/train/metrics.py (extend)
def compute_digit_cer_in_out_harmonic(
    refs_digits: list[str],            # original digit strings, e.g. ["123456", ...]
    hyps_digits: list[str],            # safe_words_to_digits applied beforehand
    spk_ids: list[str],
    in_domain_speakers: set[str],
) -> tuple[float, float, float]:
    """
    Returns (in_domain_cer, out_of_domain_cer, harmonic_mean).

    - in_domain_cer  = compute_cer(refs[spk in in_domain], hyps[spk in in_domain])
    - out_of_domain_cer = compute_cer(refs[spk not in in_domain], hyps[spk not in in_domain])
    - harmonic_mean = 2*a*b/(a+b)  (∞ if a==b==0; 0 if either subset empty — see fallback)
    """
```

Edge cases:
- Если одна из подгрупп пустая → возвращаем `(in_cer, out_cer, max(in_cer, out_cer))` (т.е. деградируем к доступной части без harmonic), и логируем warning.
- Если обе подгруппы пустые → `(0.0, 0.0, 0.0)`.
- Если один из CER = 0 → harmonic = 0 (это в нашу пользу — perfect score).

### Wiring в `Trainer._run_validation`

Добавить:
1. Поле в `TrainerConfig`:
   ```python
   in_domain_speakers: set[str] | None = None
   ```
2. Поддержка нового значения `early_stop_metric: str = "harmonic_in_out_digit_cer"` (рядом с существующими `"max_speaker_cer"`, `"val_cer"`).
3. В `_run_validation` после greedy_decode:
   ```python
   from gp1.text.denormalize import safe_words_to_digits
   refs_digits = list(batch.transcriptions)  # уже digit strings
   hyps_digits = [safe_words_to_digits(h, fallback="") for h in decoded]
   if self.config.in_domain_speakers is not None:
       in_cer, out_cer, hm = compute_digit_cer_in_out_harmonic(
           all_refs_digits, all_hyps_digits, all_spks, self.config.in_domain_speakers
       )
       # log + add to history
   ```
4. В `fit()` ветка выбора `monitored`:
   ```python
   if self.config.early_stop_metric == "harmonic_in_out_digit_cer":
       monitored = hm
   elif self.config.early_stop_metric == "max_speaker_cer":
       monitored = max_spk_cer
   else:
       monitored = val_cer
   ```

### Wiring в training-ноутбуки

В каждом тренировочном ноутбуке (01, 02, 03, 04, 01a, 01b, 01c) после загрузки `train_records`/`dev_records`:
```python
in_domain_speakers = {r.spk_id for r in train_records}
out_domain_count = sum(1 for r in dev_records if r.spk_id not in in_domain_speakers)
print(f"dev: in-domain spks={len(set(r.spk_id for r in dev_records) & in_domain_speakers)}, "
      f"out-of-domain={out_domain_count} samples")
```
И прокинуть `in_domain_speakers=in_domain_speakers` + `early_stop_metric="harmonic_in_out_digit_cer"` в `TrainerConfig`.

### Файлы (Часть 1-bis)
- **Edit**: `src/gp1/train/metrics.py` (+ функция, + tests-ready)
- **Edit**: `src/gp1/train/trainer.py` (TrainerConfig поля, _run_validation hook, fit() ветка)
- **Edit**: `tests/test_metrics.py` (новые тесты для harmonic функции — пустые подгруппы, симметрия, корректность HM)
- **Edit**: каждый training-ноутбук (01, 02, 03, 04, 01a, 01b, 01c) — добавить `in_domain_speakers` + новый `early_stop_metric`

---

## Часть 1-ter — Per-epoch логирование (два чистых ряда)

### Текущее состояние (`src/gp1/train/trainer.py`)

- `_train_epoch`: print через tqdm + `logger.info("epoch %d step %d loss=%.4f", ...)` каждые `log_every_n_steps`
- `fit()`: `logger.info("epoch %d val_cer=%.4f max_spk=%.4f best=%.4f no_improve=%d/%d", ...)`
- Validation **не считает CTC loss** — только CER. Нужно добавить val_loss.

### Что меняем

1. **Добавить val_loss в `_run_validation`**: при greedy-проходе также вызывать `self.ctc_loss(...)` под `no_grad`+autocast и аккумулировать (среднее по батчам). Возвращать кортеж `(corpus_cer, per_spk, val_loss, in_cer, out_cer, harmonic_cer)`.
2. **Удалить (или приглушить до DEBUG) per-step `logger.info`** с loss каждые N шагов — оставить только tqdm progress-bar внутри эпохи.
3. **В `fit()` после каждой эпохи печатать ровно 2 строки** через `tqdm.write()` (чтобы не ломать прогресс-бары):
   ```
   [Epoch 12/60] train  | loss=0.347
   [Epoch 12/60] val    | loss=0.412  hm_cer=0.0871  (in=0.0432  out=0.1827)  best=0.0784  no_improve=2/15
   ```
   Формат:
   - `loss` — 4 знака после точки
   - `hm_cer`, `in`, `out`, `best` — 4 знака
   - Поля выровнены пробелами, разделители `|` для train, лёгкие двойные пробелы между парами в val
   - Если `in_domain_speakers is None` → выводим только `cer=...` без in/out splits (backwards-compat)
4. **Дефолт `TrainerConfig.early_stop_metric`** меняем на `"harmonic_in_out_digit_cer"` (если `in_domain_speakers` задан, иначе fallback на `"max_speaker_cer"` с warning).
5. Удалить устаревшее имя поля `_best_val_cer` → переименовать в `_best_monitored` (это не val_cer, а значение monitored-метрики; misleading naming уже отмечен).

### Файлы (Часть 1-ter)
- **Edit**: `src/gp1/train/trainer.py` — единственный файл
- **Без новых тестов** (только cosmetic logging) — но существующие `test_trainer.py` (если есть) должны остаться зелёными

---

## Часть 2 — Новый ноутбук `notebooks/07_decode_kenlm.ipynb` (CharVocab)

Цель: загрузить чекпоинт **CharVocab**-модели (любой из 01/01a/01b/01c — BPE-модель 04 обрабатывается отдельно, см. Часть 2-bis), посчитать log-probs на dev-выборке, выдать максимально качественный декодинг с метриками **и сгенерировать предсказания для test-выборки** (`notebooks/data/test/`) в формате submission CSV.

### Эвристики (что входит)

Reusing уже реализованного:
- `gp1.lm.build_synthetic_corpus` — 999k уникальных Russian word-forms для 1000..999999
- `gp1.lm.train_kenlm` — wrapper над `lmplz` + `build_binary trie` (KenLM order=4 default)
- `gp1.decoding.BeamSearchDecoder` (`src/gp1/decoding/beam_pyctc.py`) — pyctcdecode + KenLM + hotwords + unigrams (lexicon)
- `gp1.text.denormalize.words_to_digits` (`src/gp1/text/denormalize.py`) — детерминированный парсер 42-словного словаря, инверсия `digits_to_words`. Raises на любую невалидную последовательность → идеален как **проверка валидности FSA-пути**
- `gp1.train.metrics.compute_cer` / `compute_per_speaker_cer`

Эвристики, которые будут включены в декодинг (от слабой к сильной):

1. **Greedy baseline** — `gp1.decoding.greedy.greedy_decode`, для контроля.

2. **KenLM shallow fusion** через `BeamSearchDecoder` (pyctcdecode):
   - `alpha` ∈ {0.3, 0.5, 0.7, 1.0, 1.3} — вес LM
   - `beta` ∈ {0.0, 0.5, 1.0, 1.5} — word insertion bonus
   - `beam_width` = 100
   - **Grid search** на dev → лучший (alpha, beta) по corpus-CER (на цифрах)

3. **Hard lexicon** (unigrams):
   - Передаём в `BeamSearchDecoder(unigrams=...)` все 42 слова из `denormalize._ALL_KNOWN` плюс синтетический 999k-корпус (даёт closed-vocab constraint в pyctcdecode)
   - Это эквивалент тонкого lexicon-FST (L в HLG) для closed vocab

4. **Hotwords** для thousand markers (`тысяча`, `тысячи`, `тысяч`) с весом 6–10 (sweep на 3 значениях). Эти токены критически важны и редко появляются в char-level пути.

5. **N-best + FSA-style validator rescoring** (главная эвристика для closed-form задачи):
   - Из `decoder._decoder.decode_beams(...)` берём top-N (N=10) гипотез
   - Для каждой пробуем `words_to_digits(hyp)` — если raises ⇒ кандидат отбрасывается
   - Среди валидных: фильтруем по длине цифровой строки ∈ [4, 6] (контракт задачи)
   - Среди оставшихся: **берём с лучшим logit_score** (или re-rank: `logit_score + α·lm_score`)
   - Если ни одного валидного — fallback к best-beam (или к greedy)
   - **Это эквивалент HLG-декодинга на closed FSA без зависимости от k2/icefall** — финальное множество допустимых путей вычисляется парсером, а beam-search играет роль приближения к Viterbi.

6. **Failure-aware fallback**: если N-best пустой / все невалидны → fallback chain:
   `constrained_best → unconstrained_beam_best → greedy → ""`

### О FSA / k2 / icefall (что слышал пользователь)

«FSA-декодинг» в Kaldi/k2/icefall означает построение композиции `H ∘ L ∘ G`:
- **H** — CTC topology (allow blank, allow self-loops)
- **L** — lexicon FST (char-sequence → word)
- **G** — word LM в виде FST (обычно из ARPA → fstcompile)

Полный FSA-pipeline даёт строжайшие гарантии («декодер физически не может выдать слово вне словаря»), но требует:
- `k2` или `pynini`/`OpenFST` (тяжёлые C++ зависимости с CUDA для k2)
- Конвертацию ARPA → G FST, написание L FST для русского алфавита
- Своё CTC-decoding ядро

**Лёгкий эквивалент для closed vocabulary** (наш случай — всего 42 слова, любая допустимая последовательность парсится `words_to_digits`):
- pyctcdecode `unigrams=lexicon` уже даёт лексикон-фильтр на уровне beam-search
- Финальная валидация через `words_to_digits` на N-best гарантирует FSA-acceptance (если строка проходит парсер — она лежит на допустимом пути FSA закрытой грамматики Russian 4-6 digit numbers)

⇒ Полноценный k2/HLG **не нужен** для этой задачи. N-best rescoring с детерминированным парсером даёт тот же гарант acceptance при ~10× меньшей сложности развёртки. Если после всех правок CER всё ещё неудовлетворителен — можно добавить k2 отдельным экспериментом.

### Структура ноутбука 07

1. **Setup**: env, imports, idempotent download.
2. **Config**: `CKPT_PATH = checkpoints/01_quartznet/<best_trial>/best.pt`, выбор арки через `meta.json["arch"]`, paths.
3. **Load checkpoint**: dispatcher по `meta["arch"]` → инстанцируем модель → `load_state_dict` (с unwrap `_orig_mod` если нужно).
4. **Build vocab + dev DataLoader** (по образцу 01).
5. **Forward pass**: пройти dev целиком, кэшировать `(log_probs, output_lengths, transcriptions, spk_ids)` в RAM (dev маленький).
6. **Build KenLM** (idempotent: skip если `lm/lm.bin` уже есть):
   ```python
   from gp1.lm import build_synthetic_corpus, train_kenlm
   build_synthetic_corpus(Path("lm/corpus.txt"), train_manifest=None)
   train_kenlm(Path("lm/corpus.txt"), Path("lm/lm.bin"), order=4,
               vocab_limit_path=None)
   ```
7. **Build lexicon unigrams** из `gp1.text.denormalize` модуля (42 слова + синтетический корпус).
8. **Greedy baseline**: greedy_decode → words_to_digits (safe wrapper) → CER.
9. **Beam baseline** (alpha=0.7, beta=1.0, beam_width=100, hotwords on).
10. **Alpha/beta grid search**: nested loop, кэшируем decode_beams (N-best) один раз, переоцениваем разными (α, β) на CPU. Вывод: heatmap или таблица.
11. **N-best + FSA-validator rescoring** с лучшими (α, β).
12. **Финальные метрики** в трёх вариантах:
    - **Word-level CER** (Russian text, как в trainer)
    - **Digit-level CER** (после `words_to_digits` с safe fallback) — главная метрика задачи
    - **Per-speaker CER** для двух предыдущих
13. **Comparison table** — markdown:
    | Method | word CER | digit CER | max-spk CER | invalid % |
    | greedy | … | … | … | … |
    | beam (α=0.7,β=1) | … | … | … | … |
    | beam + grid-best | … | … | … | … |
    | beam + lexicon + N-best validator | … | … | … | … |
14. **Diagnostics**: топ-10 worst examples (для каждого метода, с разбивкой по speaker), failure analysis (сколько hyps не парсятся / выходят за [4,6]).
15. **Test submission**: загрузить `TEST_ROOT = notebooks/data/test/`, прочитать `test.csv` (или просто список `*.wav`/`*.mp3`), пройти forward → beam+lexicon+N-best validator (лучшая конфигурация из шага 11), `safe_words_to_digits` → fallback `"0000"` для невалидных, записать `submission.csv` (колонки: `filename,transcription`) рядом с ноутбуком.

---

## Часть 2-bis — LM-блок внутри `notebooks/04_fast_conformer_bpe.ipynb`

BPE-словарь несовместим с CharVocab-декодером 07 (другие labels, другой lexicon-формат), поэтому LM-пайплайн встраивается **прямо в 04** в самом конце ноутбука после `Trainer.fit()`.

### Что добавить в 04 (в конец ноутбука)

1. **KenLM training** (тот же `build_synthetic_corpus` + `train_kenlm`, что и в 07 — корпус один и тот же, шаг идемпотентен).
2. **BPE-aware beam decoder**: pyctcdecode поддерживает любые labels, поэтому `build_ctcdecoder(labels=bpe_pieces, kenlm_model_path=lm.bin, alpha, beta)` где `bpe_pieces` — выгрузка из SentencePiece модели в порядке id (с blank id 0 как `""`). Lexicon (unigrams) — те же 999k word-forms из синтетического корпуса.
3. **Hotwords**: `("тысяча", "тысячи", "тысяч")` — pyctcdecode сам разложит их по BPE-токенам через свой матчер.
4. **Alpha/beta grid** на dev (тот же sweep, что и в 07).
5. **N-best + `words_to_digits` validator rescoring** (главная эвристика — работает независимо от BPE/Char, потому что текст на выходе декодера всё равно русский).
6. **Метрики** (word/digit-level CER, per-speaker) — те же helper'ы из `gp1.train.metrics` + `safe_words_to_digits`.
7. **Test submission** — аналогично шагу 15 в 07, генерация `submission.csv` для test-выборки, отдельный файл (например `submission_bpe.csv`).

Дублирование кода между 07 и 04 будет (alpha/beta grid loop, FSA-rescoring) — это OK для inline-стиля, который пользователь выбрал. Общие helper'ы лежат в `src/gp1/text/denormalize.py` (`safe_words_to_digits`, `fsa_constrained_best`).

### Safe wrappers (новые helpers — добавить в `src/gp1/text/denormalize.py`)

```python
def safe_words_to_digits(text: str, fallback: str = "") -> str:
    """Try words_to_digits; return fallback on ValueError."""
    try:
        return words_to_digits(text)
    except ValueError:
        return fallback

def fsa_constrained_best(beams, length_range: tuple[int, int] = (4, 6)) -> str:
    """From pyctcdecode top-N beams (each tuple (text, lm_state, frames, logit_score, lm_score)),
    return first hypothesis whose words_to_digits parses to a digit string within length_range.
    Among valid, pick max logit_score. Fallback to '' if none valid."""
    ...
```

### Новые/изменяемые исходники
- **Edit**: `src/gp1/text/denormalize.py` — добавить `safe_words_to_digits` + `fsa_constrained_best` рядом с существующим `words_to_digits`
- **Новый**: `notebooks/07_decode_kenlm.ipynb`
- **Tests**: расширить существующий `tests/test_denormalize.py` (добавить TestSafeWordsToDigits + TestFsaConstrainedBest)

### Зависимости
- `kenlm` (pip install kenlm) — НЕ в `pyproject.toml`, добавить в раздел `[project.optional-dependencies]` как `lm = ["kenlm>=0.2.0"]`
- `pyctcdecode>=0.5.0` — уже есть в `pyproject.toml`
- Установка: `uv pip install kenlm` или `pip install https://github.com/kpu/kenlm/archive/master.zip`

---

## Verification

### Часть 1
```bash
# Запуск pytest (никакого ноутбук-кода тестами не покрыто, но src должен быть зелёный)
uv run pytest -m "not slow" -q

# Sanity-check каждого изменённого ноутбука
uv run jupyter nbconvert --to script notebooks/02_crdnn.ipynb --stdout | python -c "import sys; sys.stdin.read()"
# (убедиться что синтаксис чистый; полный run требует GPU + данные)

# Manual: пройти первые 6 cells каждого ноутбука руками — env, paths, manifests, AUDIO_CACHE, DataLoader
# Проверить что DL шевелится: next(iter(train_loader)).audio.shape == [BATCH_SIZE, T_max]
```

### Часть 2
```bash
# Установить kenlm
uv pip install kenlm

# Тесты для нового модуля
uv run pytest tests/test_postprocess.py -v

# Smoke: build_synthetic_corpus + train_kenlm + KenLMWrapper.score
uv run python -c "
from gp1.lm import build_synthetic_corpus, train_kenlm
from gp1.decoding.lm import KenLMWrapper
from pathlib import Path
build_synthetic_corpus(Path('/tmp/c.txt'))
train_kenlm(Path('/tmp/c.txt'), Path('/tmp/lm.bin'), order=4)
wrap = KenLMWrapper(Path('/tmp/lm.bin'))
print(wrap.score('сто двадцать три'))
"

# Полный прогон ноутбука 07 на реальном чекпоинте (GPU нужен только для forward pass — beam decode на CPU)
uv run jupyter execute notebooks/07_decode_kenlm.ipynb
```

### Acceptance criteria
- Все 8 правленных ноутбуков синтаксически валидны, импорты не падают.
- `pytest -m "not slow"` зелёный.
- Ноутбук 07 проходит до конца на реальном чекпоинте, генерирует `submission.csv` для test-выборки.
- LM-блок в 04 проходит до конца, генерирует `submission_bpe.csv`.
- В comparison-таблице 07: digit-CER метода `beam + lexicon + N-best validator` ≤ digit-CER `greedy` (ожидаемое улучшение -15…-40% относительно greedy).
- `invalid %` (доля гипотез, не парсящихся `words_to_digits`) для финального метода ≤ 1%.
- `submission.csv` / `submission_bpe.csv` имеют ровно `len(test_records)` строк, все `transcription` — цифровые строки длины 4–6 (с fallback `"0000"` для невалидных).

## Ключевые файлы для модификации (полный список)

**Часть 1 (notebook edits)** — style alignment + harmonic-metric wiring:
- `notebooks/01_quartznet.ipynb` (только wiring новой метрики)
- `notebooks/02_crdnn.ipynb`
- `notebooks/03_efficient_conformer.ipynb`
- `notebooks/04_fast_conformer_bpe.ipynb`
- `notebooks/05_predict.ipynb` (без metric wiring — это inference-ноутбук)
- `notebooks/experiments/01a_quartznet_inter_ctc.ipynb`
- `notebooks/experiments/01b_quartznet_word_aux.ipynb`
- `notebooks/experiments/01c_quartznet_cr_ctc.ipynb`
- `notebooks/experiments/06_kenlm_beam_rescore.ipynb` (без metric wiring — нет тренировки)

**Часть 1-bis (training metric)**:
- `src/gp1/train/metrics.py` (extend) — `compute_digit_cer_in_out_harmonic`
- `src/gp1/train/trainer.py` (extend) — TrainerConfig field, _run_validation hook, fit() ветка
- `tests/test_metrics.py` (extend)

**Часть 1-ter (logging cosmetic, без тестов)**:
- `src/gp1/train/trainer.py` (further edit) — val_loss, 2-line summary через tqdm.write, rename `_best_val_cer` → `_best_monitored`

**Часть 2 (new code + notebook)**:
- `src/gp1/text/denormalize.py` (extend) — добавить `safe_words_to_digits` + `fsa_constrained_best`
- `tests/test_denormalize.py` (extend) — TestSafeWordsToDigits + TestFsaConstrainedBest
- `notebooks/07_decode_kenlm.ipynb` (new) — CharVocab pipeline + test submission
- `notebooks/04_fast_conformer_bpe.ipynb` (extend tail) — BPE inline LM-блок + test submission
- `pyproject.toml` (add `[project.optional-dependencies] lm = ["kenlm"]`)

---

## Оркестрация саб-агентов

### Логические блоки и зависимости

```
┌──────────────────────────────────────────────────────────────┐
│ Блок C1: Helpers в src/gp1/text/denormalize.py               │
│   → safe_words_to_digits, fsa_constrained_best + tests       │
│   ⚠ MUST finish first — все остальные импортируют helpers    │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Блок M: Harmonic in/out digit-CER метрика                    │
│   → metrics.py (+ compute_digit_cer_in_out_harmonic)         │
│   → trainer.py (TrainerConfig + _run_validation + fit ветка) │
│   → tests/test_metrics.py                                    │
│   зависит от C1 (использует safe_words_to_digits)            │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Блок L: Per-epoch logging (косметика, без тестов)            │
│   → trainer.py (val_loss + 2-line summary + rename _best_*)  │
│   зависит от M (использует harmonic CER в выводе)            │
└──────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────────────────────────────┐
        ▼                                                       ▼
┌──────────────────────────┐  ┌──────────────────────────────────┐
│ N1: Main training nbs    │  │ N2: Experiment training nbs      │
│ 01, 02, 03, 04 + 05/06   │  │ 01a, 01b, 01c                    │
│ — style alignment        │  │ — style alignment                │
│ — wire in_domain_speakers│  │ — wire in_domain_speakers        │
│ — wire harmonic metric   │  │ — wire harmonic metric           │
│ depends on C1+M+L        │  │ depends on C1+M+L                │
└──────────────────────────┘  └──────────────────────────────────┘
        │                          │
        ▼                          ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ C2: notebooks/07            │  │ C3: 04 inline LM block      │
│ CharVocab+KenLM+FSA+submit  │  │ BPE+KenLM+FSA+submit        │
│ depends on C1 (helpers)     │  │ depends on C1 (helpers) +   │
│ (не зависит от M/L)         │  │ N1 (04 style-alignment)     │
└─────────────────────────────┘  └─────────────────────────────┘
```

**Этапы исполнения** (последовательные, внутри этапа — параллель):
- **Этап 1**: C1 (один)
- **Этап 2**: M (один, depends on C1)
- **Этап 3**: L (один, depends на M; trainer.py — общий файл, последовательно)
- **Этап 4**: N1 + N2 + C2 в **параллели** (C2 ждёт только C1, но запустим параллельно для скорости)
- **Этап 5**: C3 (depends on N1, который обработает 04_fast_conformer_bpe.ipynb)

### Тройка агентов на каждый блок

```
[Iteration N]
  ┌──────────┐    spec       ┌────────────┐    diff      ┌──────────┐
  │ TDD Coder├──────────────▶│ Simplifier ├─────────────▶│ Reviewer │
  └──────────┘   (+ tests)   └────────────┘  (cleaned)   └─────┬────┘
                                                               │
              ┌─verdict: APPROVE → блок готов                  │
              ◀                                                │
              └─verdict: CHANGES_REQUESTED → feedback ─────────┤
                          (новая итерация с правками)          │
                                                               │
                                  N ≤ 4 итераций (hard cap, потом эскалация к user)
```

**Роли**:

| Роль | Subagent type | Задача |
|---|---|---|
| TDD Coder | `general-purpose` (промпт "RED→GREEN→REFACTOR" для src; для notebook — "apply canonical cells from 01") | Написать тесты (для src) или применить cell-template (для notebook). Сделать тесты/прогон зелёным. |
| Simplifier | `code-simplifier:code-simplifier` | Удалить дублирование, lишние комментарии/импорты, выровнять стиль с соседними модулями. Не ломать публичный API. |
| Reviewer | `python-reviewer` (src) / `general-purpose` с notebook-чеклистом (notebooks) | Проверить контракт блока (см. чеклисты ниже). Вердикт: APPROVE / CHANGES_REQUESTED + список проблем file:line. |

**Исключение для блока L** (cosmetic logging, без тестов): тройка вырождается в **дуэт** Coder→Reviewer:
- Coder: `general-purpose` — применяет правки в trainer.py согласно спеку (val_loss, 2-line tqdm.write, rename)
- Reviewer: `python-reviewer` — проверяет, что existing tests зелёные + чеклист L пройден
- Simplifier пропускается (нет тестируемого нового кода — только переформатирование вывода)

### Чеклисты ревьювера

**Для блока C1** (denormalize.py + tests):
- [ ] `safe_words_to_digits(text, fallback="")` ловит **только** `ValueError` (не bare `except`)
- [ ] `fsa_constrained_best(beams, length_range=(4,6))` корректно обрабатывает: пустые beams, все невалидные, mix валидных/невалидных
- [ ] Tie-break при равном logit_score детерминирован (берётся первый)
- [ ] Type hints на всех публичных функциях
- [ ] Существующий `words_to_digits` НЕ изменён (regression-safe)
- [ ] Все 23+ существующих теста `test_denormalize.py` зелёные
- [ ] Новые тесты покрывают: happy path, ValueError fallback, length out-of-range, empty beams

**Для блока L** (logging cosmetic) — упрощённый чеклист, тестов нет:
- [ ] `_run_validation` теперь возвращает val_loss
- [ ] Per-step `logger.info(loss=...)` в `_train_epoch` удалён (или DEBUG-level)
- [ ] `tqdm.write` используется для 2-line summary (не ломает прогресс-бары)
- [ ] Формат строго: `[Epoch X/Y] train  | loss=...` и `[Epoch X/Y] val    | loss=... hm_cer=... (in=... out=...) best=... no_improve=N/M`
- [ ] `_best_val_cer` переименован в `_best_monitored`
- [ ] Backwards-compat: если `in_domain_speakers is None` → val-строка без in/out splits
- [ ] Все существующие тесты trainer'а зелёные

**Для блока M** (metrics + trainer):
- [ ] `compute_digit_cer_in_out_harmonic` корректно обрабатывает: empty in-domain, empty out-of-domain, both empty, in==out==0
- [ ] Harmonic mean считается через `2*a*b/(a+b)` (не среднее арифметическое)
- [ ] `TrainerConfig.in_domain_speakers: set[str] | None` имеет дефолт `None` (backwards-compat)
- [ ] `TrainerConfig.early_stop_metric` поддерживает новое значение `"harmonic_in_out_digit_cer"` (3 валидных значения)
- [ ] `_run_validation` вызывает `safe_words_to_digits` на hyps (импорт из `gp1.text.denormalize`)
- [ ] Логирование: новый метрик пишется в history dict + `logger.info`
- [ ] Existing 8+ tests `test_metrics.py` зелёные (regression-safe)
- [ ] Existing trainer tests / smoke (если есть) зелёные

**Для блоков N1/N2/C2/C3** (notebooks):
- [ ] Cell 1: idempotent download присутствует и проверяет `DATA_ROOT.exists()` И `any(DATA_ROOT.iterdir())`
- [ ] Cell 2: env hardening идёт **до** `import torch` (порядок критичен для PYTORCH_CUDA_ALLOC_CONF)
- [ ] Cell 3: paths используют `DATA_ROOT` (не hardcoded `/path/to/...`)
- [ ] AUDIO_CACHE: `share_memory_()` вызван на каждом тензоре
- [ ] `_worker_init` присутствует, реcеёдит `info.dataset._augmenter._rng`
- [ ] DataLoader: `worker_init_fn=_worker_init`, `persistent_workers=True`, `prefetch_factor=2`, `pin_memory=True`
- [ ] BATCH_SIZE и DL_WORKERS соответствуют таблице из плана
- [ ] Никаких leftover `os.cpu_count()`, никаких незаменённых FILL_ME_IN
- [ ] **(N1/N2 only)** `in_domain_speakers = {r.spk_id for r in train_records}` присутствует
- [ ] **(N1/N2 only)** `TrainerConfig(..., in_domain_speakers=in_domain_speakers, early_stop_metric="harmonic_in_out_digit_cer")` прокинуто
- [ ] Smoke: `jupyter nbconvert --to script <nb> --stdout | python -c "compile(sys.stdin.read(), '<nb>', 'exec')"` проходит

**Дополнительно для C2/C3** (LM-pipeline notebooks):
- [ ] Импорт `safe_words_to_digits`, `fsa_constrained_best` из `gp1.text.denormalize`
- [ ] KenLM-build идемпотентен (skip если `lm/lm.bin` уже есть)
- [ ] Alpha/beta grid использует кэшированные beams (decode_beams вызывается один раз на dev)
- [ ] FSA-validator rescoring имеет fallback chain: `constrained → best_beam → greedy → ""`
- [ ] Test submission CSV содержит ровно `len(test_records)` строк, все transcription — цифровые 4–6 chars

### Политика консистентности (когда блоки пересекаются)

1. **Каждый агент сначала читает текущее состояние всех файлов, которые он трогает** — никаких допущений из промпта. Промпт даёт паспорт задачи (контракт + чеклист), не готовый код.
2. **Helpers в `denormalize.py`** — единственная общая точка между C2 и C3. C1 завершается до запуска C2/C3, поэтому состояние denormalize.py к моменту старта C2/C3 уже зафиксировано.
3. **Канонические cells в 01_quartznet.ipynb** — источник правды для блоков N1/N2. Каждый агент re-reads cells `7abac1c8`/`e35b1f82`/`K2S1Jj6JJB9k`/`b215b3c1` из 01 перед применением к целевому ноутбуку.
4. **`pyproject.toml`** правится только один раз (в C1 или в C2 — назначить C1).
5. **Между итерациями триплета**: feedback ревьювера передаётся в TDD Coder следующей итерации **в виде текста** (не diff'a) — coder сам решает, как править.

### Loop policy

- Hard cap: **4 итерации** на блок. После 4-й — эскалация: показать пользователю последний reviewer-verdict и спросить, что делать.
- Между итерациями: НЕ удалять промежуточные файлы. Если reviewer указал на конкретный тест — coder должен этот тест исправить, не удалять.
- Если simplifier вернул "no changes needed" — пропускаем сразу к reviewer.
- Если reviewer APPROVE на первой итерации — блок готов, не зацикливаемся.

### Telemetry / progress

После каждого блока в чат:
```
✓ Block C1 complete (2 iterations): denormalize.py +safe_words_to_digits, +fsa_constrained_best
✓ Block N1 complete (1 iteration): 02_crdnn.ipynb, 03_..., 05_..., 06_...
...
```

---

## Reused (НЕ трогаем)

- `src/gp1/lm/build_corpus.py` — уже упрощён, готов
- `src/gp1/lm/train_kenlm.py` — wrapper готов
- `src/gp1/decoding/beam_pyctc.py` — `BeamSearchDecoder` готов
- `src/gp1/decoding/lm.py` — `KenLMWrapper` готов
- `src/gp1/text/normalize.py`, `src/gp1/text/denormalize.py` — готовы
- `src/gp1/train/metrics.py` — `compute_cer`, `compute_per_speaker_cer` готовы
- `src/gp1/data/dataset.py::preload_audio_cache` — готов
