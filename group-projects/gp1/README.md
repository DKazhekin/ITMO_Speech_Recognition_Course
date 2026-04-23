
## Group Project 1. Automatic Speech Recognition - [30 pts]

In this exercise you will be building Automatic Speech Recognition system in Russian language. In particular, given the training and validation sets, you are required to train a small NN recognizing russian spoken numbers.


### Data Description

You are provided with training and validation (a.k.a. development) sets of pairs (audio, transcript) with a corresponding meta information in a form csv file as follows:

```
filename,transcription,spk_id,gender,ext,samplerate
train/0007c21c23.wav,139473,spk_E,female,wav,24000
train/000bee1b1d.wav,992597,spk_B,male,wav,24000
```

where **transcription** is a number from range `[1_000 .. 999_999]`, and **ext** is an extension of audio file (one of `wav`, `mp3`).

Overall there are 3 data splits of 14 unique `spk_id` from **spk_A** to **spk_N** with the following amount of audio samples in each split:
- `train/`: [**download link**](https://drive.google.com/file/d/15CpIWvVDA6mOlPxyI4-vicyXSqd-EcIb/view?usp=sharing)
    - 12,553 samples from 6 `spk_id` 
- `dev/`: [**download link**](https://drive.google.com/file/d/1Jlw09RSJjhJTxdN3VQj5Bph4zRNwOqSL/view?usp=sharing)
    - 2,265 samples from 10 `spk_id`
- `test/`: not available for local development: [Kaggle challenge evaluation page](https://www.kaggle.com/competitions/asr-2026-spoken-numbers-recognition-challenge/overview). Challenge invitation you'll receive via Google Classroom assignment
    - 2,265 samples from all 14 `spk_id`

> NOTE: `dev/` data CAN NOT be used for trainig, but for validation purposes only

Please keep in mind that samplerate and file extension are not constant across all the files in all data splits.


### Project Requirements

- You are required to train a model for ***16 kHz samplerate*** audio input (though you can see samplerate of 22.05 kHz and higher in training data), this way we balance a trade-off between input signals range support and a model accuracy

- Any architecture, algorithm or training baseline could be used (e.g. open-sourced pipeline or architecture)
    - However, initialization of a model from pre-trained weights is not allowed. ***Only training from scratch*** on provided training data:
      - additional training data is not allowed except the samples of noises for augmenation
      - validation split can not be concatenated to training data and is only provided to help you tracking the errors and overfitting similarly to what you can receive on the test set

- Keep the model small - up to **5M** parameters

- You can re-use your work from [personal assignments 1 and 2](../../assignments/), though this is not compulsory

- You can train a KenLM language model for LM fusion and rescoring

- Model training can be run offline using any available hardware resources ([Google Colab](https://colab.research.google.com/), [Kaggle](https://www.kaggle.com/))

- Try improving your metrics on validation split, because it is correlated with the test data


### Hints 

* When training, track the recognition error ([CER](https://lightning.ai/docs/torchmetrics/stable/text/char_error_rate.html)) per speaker `spk_id` - this will show you if the model overfits and performs really badly on unseen voice (maybe you overparameterized your model or forgot various regularizations)

* The labeling is not normalized, meaning that direct transcription may not provide you digits directly (unless you wanna try training such setup) - what you actually hear and what is given as a label differ. You can think of applying normalization and denormalization to transcriptions

* Try being creative with how you construct the vocabulary of symbols/words/subwords for recognition

* Note that word “тысяча” can highlight you that you will have to have three more symbols after it even if there is nothing spoken after it (e.g. “одна тысяча пять” -> 1_005)

* Don’t forget to use various audio augmentations techniques while training, as some samples in `dev/` and `test/` splits are noisy


### Evaluation

- Evaluation of models is held on the [Kaggle platform](https://www.kaggle.com/competitions/asr-2026-spoken-numbers-recognition-challenge/overview). Note that this is used for evaluation only, training of model can be performed in an offline fashion with the available hardware. Though you can still stick to the in-Kaggle training with data available on the competition page (duplicated)

- The model performance will be evaluated on the holdout testing set, containing extra out-of-domain test speakers `spk_id`

- All works will be ranked according to eval metrics. Primary metric is a **harmonic mean CER** for recognized numbers for inD and ooD `spk_id`. ooD CER will be considered as a secondary metric in case of equality of the results


### Deliverables

- Kaggle Competition submission and corresponding position on the leaderboard:
    - Public submission notebook has to import your model and weights from GitHub (e.g. github release) and run inference + decoding

- Public GitHub repository with source code of your training pipeline and model weights (weights as a release in order to be imported in Kaggle)

- Google Classroom PDF report describing your work, experiments and results (also your Kaggle team name, and a history of submissions) in free form


### Resources

- [Kaggle copmetition submission page](https://www.kaggle.com/competitions/asr-2026-spoken-numbers-recognition-challenge/overview)
- For text normalization and denormalization you can use [NeMo toolkit](https://github.com/NVIDIA/NeMo-text-processing/blob/main/tutorials/Text_(Inverse)_Normalization.ipynb) or [num2words](https://pypi.org/project/num2words/) library
- Making models smaller and more efficient with [different types of convolutions](https://animatedai.github.io/)

---

## Codebase layout (implementation)

```
src/gp1/               # Reusable, architecture-neutral library
├── types.py           # ManifestRecord, Batch, AugConfig
├── data/              # manifest, dataset, collate, audio_aug, spec_aug
├── features/          # LogMelFilterBanks
├── text/              # CharVocab, BPEVocab, digits<->words normalize/denormalize
├── losses/            # CTCLoss (the only in-library loss)
├── decoding/          # greedy_decode; optional BeamSearchDecoder (pyctcdecode+KenLM)
├── lm/                # build_synthetic_corpus, train_kenlm
├── models/            # QuartzNet10x4, CRDNN, EfficientConformer, FastConformerBPE + common blocks
├── train/             # slim Trainer (tqdm, fp16 autocast, grad accum, early stop), checkpoint save/load, metrics, optim, schedulers
└── submit/            # inference_utils: build_test_dataloader, write_submission

notebooks/
├── 01_quartznet.ipynb               # HP random search + train (NovoGrad + cosine)
├── 02_crdnn.ipynb                   # (AdamW + cosine)
├── 03_efficient_conformer.ipynb     # (AdamW + Noam, s=4)
├── 04_fast_conformer_bpe.ipynb      # (BPE-256 + Noam)
├── 05_predict.ipynb                 # load best checkpoint → submission.csv
└── experiments/
    ├── 01a_quartznet_inter_ctc.ipynb     # inline InterCTCHead + raw training loop
    ├── 01b_quartznet_word_aux.ipynb      # inline WordVocab + WordAuxCTCHead
    ├── 01c_quartznet_cr_ctc.ipynb        # inline CRCTCLoss with two SpecAug views
    └── 06_kenlm_beam_rescore.ipynb       # train KenLM + beam rescore

tests/                 # Reusable-core tests only (data, features, vocab, losses/ctc, models, trainer, decoding, metrics)
pyproject.toml         # uv-managed, Python 3.11, torch 2.5.1 + deps
```

## Workflow

1. **Setup.** `uv sync` locally. On Colab/Kaggle, the first two cells of every notebook contain platform-specific `!git clone` + `!pip install` blocks (commented out) — uncomment the one matching your platform.
2. **Fill paths.** The "Пути (заполните вручную)" cell at the top of every notebook declares explicit `TRAIN_ROOT`, `DEV_ROOT`, `TRAIN_CSV`, `DEV_CSV`, `CKPT_ROOT`, optional `TEST_ROOT`. Replace each `FILL_ME_IN` with a real `Path`.
3. **Pick an architecture.** Open one of the four main notebooks (`01_quartznet` / `02_crdnn` / `03_efficient_conformer` / `04_fast_conformer_bpe`). Each is a complete step-by-step pipeline: data → vocab → model → trainer → HP random search → best checkpoint.
4. **Tune HPs.** Each notebook has two dicts:
   - `FIXED` — audio/batching params you don't vary per trial.
   - `HP_GRID` — axes to randomize (lr, dropout, SpecAug, augmentation probs, ...).
   - `N_TRIALS` — how many random configurations to try.
5. **Run the notebook.** The HP loop saves the best checkpoint per trial under `<CKPT_ROOT>/<run_id>/trial_XX/best.pt` + `meta.json` describing `arch`, `hparams`, `val_cer`.
6. **Predict.** Open `05_predict.ipynb`, point `CKPT` to any `best.pt`, it rebuilds the matching model and writes `submission.csv`.
7. **Optional experiments.** `notebooks/experiments/` contains self-contained demos of auxiliary losses (`inter_ctc`, `word_aux`, `cr_ctc`) and a KenLM beam rescore recipe. They keep their extra classes inline — open one and see the whole picture.

## Testing

```
.venv/bin/python -m pytest tests/ -m "not slow"
```

Tests cover only architecture-neutral infrastructure (dataset, collate, features, vocab, CTC loss, trainer, decoding, metrics, per-architecture shape contracts). Architecture-specific experiments are validated by running their notebooks.

## Why this layout

Everything architecture-specific (loss heads for InterCTC / CR-CTC / WordAux, model constructors, optimizer choice, scheduler choice) lives in the notebooks — the reader sees the whole pipeline in one file and can step-debug any cell. The `src/` library is intentionally arch-neutral: no `if model_name == ...` branches, no dispatcher, no YAML configs, no auto-detection of platform — all hyperparameters and paths are Python constants at the top of each notebook, filled in by hand.
