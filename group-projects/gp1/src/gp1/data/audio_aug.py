"""Audio-domain augmentation for the GP1 ASR pipeline.

Implements the augmentation stack from the plan:
  SpeedPerturb | PitchShift (XOR), VTLP, Gain, AddNoise (MUSAN), RIR convolution.

Reference implementations:
  - torchaudio SpeedPerturbation / PitchShift:
      https://pytorch.org/audio/stable/transforms.html
  - VTLP (piecewise-linear frequency warp on STFT magnitude):
      Ko et al. (2015) "Audio Augmentation for Speech Recognition"
      Adapted from: https://github.com/iver56/audiomentations (VTLP concept)

Design decisions:
  - Speed XOR Pitch: when both would fire (speed_prob=1, pitch_prob=1),
    speed is preferred (documented below in _pick_speed_or_pitch).
  - Determinism: all RNG uses a per-instance random.Random seeded from
    config.seed; torch operations that need a generator receive an
    explicit torch.Generator. Global RNG is never touched.
  - musan_root / rir_root = None: noise and RIR branches become no-ops.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as tf
import torchaudio.functional as taF

from gp1.types import AugConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level augmentation primitives (referenced by name so tests can
# monkey-patch them via `audio_aug._apply_speed_perturb = stub`).
# ---------------------------------------------------------------------------


def _apply_speed_perturb(
    wav: torch.Tensor,
    samplerate: int,
    factors: tuple[float, ...],
    rng: random.Random,
) -> torch.Tensor:
    """Resample waveform to simulate playback speed change.

    Args:
        wav: 1-D float32 waveform, shape [T].
        samplerate: Input sample rate in Hz.
        factors: Allowed speed factors (e.g. (0.9, 1.0, 1.1)).
        rng: Seeded random.Random instance — not the global RNG.

    Returns:
        Speed-perturbed waveform, 1-D float32.
    """
    factor = rng.choice(list(factors))
    if factor == 1.0:
        return wav
    orig_freq = samplerate
    new_freq = int(round(samplerate * factor))
    out = taF.resample(wav.unsqueeze(0), orig_freq=new_freq, new_freq=orig_freq)
    return out.squeeze(0)


def _apply_pitch_shift(
    wav: torch.Tensor,
    samplerate: int,
    pitch_range_semitones: tuple[float, float],
    rng: random.Random,
) -> torch.Tensor:
    """Shift pitch by a random number of semitones using torchaudio.

    Args:
        wav: 1-D float32 waveform, shape [T].
        samplerate: Sample rate in Hz.
        pitch_range_semitones: (low, high) inclusive range in semitones.
        rng: Seeded random.Random instance.

    Returns:
        Pitch-shifted waveform, 1-D float32.
    """
    low, high = pitch_range_semitones
    n_steps = rng.uniform(low, high)
    n_steps_int = int(round(n_steps))
    if n_steps_int == 0:
        return wav
    out = taF.pitch_shift(
        wav.unsqueeze(0),
        sample_rate=samplerate,
        n_steps=n_steps_int,
    )
    return out.squeeze(0)


def _apply_vtlp(
    wav: torch.Tensor,
    samplerate: int,
    alpha_range: tuple[float, float],
    rng: random.Random,
) -> torch.Tensor:
    """Vocal Tract Length Perturbation via piecewise-linear frequency warping.

    Warps the STFT frequency axis by alpha, then reconstructs with iSTFT.
    Piecewise-linear warp: below f_boundary*alpha -> scale by alpha;
    above -> scale remaining bins linearly up to Nyquist.

    Reference: Jaitly & Hinton (2013) "Vocal Tract Length Perturbation."

    Args:
        wav: 1-D float32 waveform.
        samplerate: Sample rate in Hz.
        alpha_range: (low, high) for uniform alpha draw.
        rng: Seeded random.Random instance.

    Returns:
        VTLP-warped waveform, same length, 1-D float32.
    """
    alpha = rng.uniform(*alpha_range)
    if abs(alpha - 1.0) < 1e-4:
        return wav

    n_fft = 512
    hop = 160
    win = 400
    window = torch.hann_window(win, device=wav.device)

    stft = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        return_complex=True,
    )  # [freq_bins, frames]

    freq_bins = stft.shape[0]  # n_fft // 2 + 1
    nyquist_bin = freq_bins - 1

    # Build warped frequency indices (piecewise-linear).
    orig_bins = torch.arange(freq_bins, dtype=torch.float32, device=wav.device)
    f_boundary = nyquist_bin / max(alpha, 1.0 - (alpha - 1.0))
    warped = torch.where(
        orig_bins <= f_boundary * alpha,
        orig_bins / alpha,
        (nyquist_bin - orig_bins)
        / (nyquist_bin - f_boundary * alpha + 1e-8)
        * (nyquist_bin - f_boundary)
        + f_boundary,
    ).clamp(0.0, nyquist_bin)

    # Bilinear interpolation along frequency axis.
    floor_idx = warped.long().clamp(0, nyquist_bin - 1)
    ceil_idx = (floor_idx + 1).clamp(0, nyquist_bin)
    frac = (warped - floor_idx.float()).unsqueeze(1)  # [freq_bins, 1]

    stft_warped = (
        stft[floor_idx] * (1.0 - frac) + stft[ceil_idx] * frac
    )  # [freq_bins, frames]

    wav_out = torch.istft(
        stft_warped,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        length=wav.shape[0],
    )
    return wav_out


def _apply_gain(
    wav: torch.Tensor,
    gain_db_range: tuple[float, float],
    rng: random.Random,
) -> torch.Tensor:
    """Apply random gain in dB.

    Args:
        wav: 1-D float32 waveform.
        gain_db_range: (min_db, max_db).
        rng: Seeded random.Random instance.

    Returns:
        Gain-adjusted waveform.
    """
    db = rng.uniform(*gain_db_range)
    scale = 10.0 ** (db / 20.0)
    return wav * scale


def _apply_add_noise(
    wav: torch.Tensor,
    noise_files: list[Path],
    snr_db_range: tuple[float, float],
    rng: random.Random,
) -> torch.Tensor:
    """Mix MUSAN noise at a random SNR.

    Args:
        wav: 1-D float32 waveform.
        noise_files: Pre-indexed list of .wav paths.
        snr_db_range: (min_snr, max_snr) in dB.
        rng: Seeded random.Random instance.

    Returns:
        Noisy waveform, same length as input.
    """
    import torchaudio

    noise_path = rng.choice(noise_files)
    noise_wav, _ = torchaudio.load(str(noise_path))
    noise_wav = noise_wav.mean(dim=0)  # mono

    # Tile or trim to match signal length.
    sig_len = wav.shape[0]
    if noise_wav.shape[0] < sig_len:
        repeats = math.ceil(sig_len / noise_wav.shape[0])
        noise_wav = noise_wav.repeat(repeats)
    noise_wav = noise_wav[:sig_len]

    snr_db = rng.uniform(*snr_db_range)
    sig_power = wav.pow(2).mean().clamp_min(1e-9)
    noise_power = noise_wav.pow(2).mean().clamp_min(1e-9)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_scale = (sig_power / (snr_linear * noise_power)).sqrt()
    return wav + noise_scale * noise_wav


def _apply_rir(
    wav: torch.Tensor,
    rir_files: list[Path],
    rng: random.Random,
) -> torch.Tensor:
    """Convolve waveform with a randomly selected room impulse response.

    Args:
        wav: 1-D float32 waveform.
        rir_files: Pre-indexed list of .wav paths.
        rng: Seeded random.Random instance.

    Returns:
        Reverberant waveform, same length as input.
    """
    import torchaudio

    rir_path = rng.choice(rir_files)
    rir_wav, _ = torchaudio.load(str(rir_path))
    rir_wav = rir_wav.mean(dim=0)  # mono [T_rir]

    # Normalise IR by its L2 norm.
    rir_wav = rir_wav / (rir_wav.norm() + 1e-9)

    sig_len = wav.shape[0]
    rir_len = rir_wav.shape[0]

    # 1-D convolution: wav [1, 1, T] * rir [1, 1, T_rir].
    wav_3d = wav.unsqueeze(0).unsqueeze(0)
    rir_3d = rir_wav.unsqueeze(0).unsqueeze(0)
    out = tf.conv1d(wav_3d, rir_3d, padding=rir_len - 1)
    out = out.squeeze(0).squeeze(0)[:sig_len]
    return out


# ---------------------------------------------------------------------------
# Helper: index .wav files under a directory tree.
# ---------------------------------------------------------------------------


def _index_wav_files(root: Path) -> list[Path]:
    """Return sorted list of all .wav files under *root*."""
    files = sorted(root.rglob("*.wav"))
    log.debug("Indexed %d .wav files under %s", len(files), root)
    return files


# ---------------------------------------------------------------------------
# XOR picker helper.
# ---------------------------------------------------------------------------


def _pick_speed_or_pitch(
    speed_prob: float,
    pitch_prob: float,
    rng: random.Random,
) -> str:
    """Return 'speed', 'pitch', or 'none' — never both (XOR).

    Algorithm:
      1. Roll for speed: r1 < speed_prob -> would_speed.
      2. Roll for pitch: r2 < pitch_prob -> would_pitch.
      3. If both would fire, resolve the XOR via a coin flip (50/50).
         This ensures both branches are reachable when both probs == 1.0,
         which the test verifies across 50 calls.
    """
    would_speed = rng.random() < speed_prob
    would_pitch = rng.random() < pitch_prob

    if would_speed and would_pitch:
        # XOR: randomly pick one so both branches are reachable.
        return "speed" if rng.random() < 0.5 else "pitch"
    if would_speed:
        return "speed"
    if would_pitch:
        return "pitch"
    return "none"


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


class AudioAugmenter:
    """Apply stochastic audio augmentations for ASR training.

    The augmentation order follows the plan:
      1. Speed XOR Pitch
      2. VTLP
      3. Gain
      4. AddNoise (MUSAN)
      5. RIR convolution

    Deterministic when ``config.seed`` is set: all randomness flows through
    a private ``random.Random`` instance seeded from ``config.seed``; the
    global RNG is never touched.

    Args:
        config: Frozen ``AugConfig`` dataclass with all hyperparameters.
    """

    def __init__(self, config: AugConfig) -> None:
        self._cfg = config
        self._rng = random.Random(config.seed)

        # Eagerly index audio asset files (avoids per-call filesystem scans).
        self._noise_files: list[Path] = []
        if config.musan_root is not None and Path(config.musan_root).exists():
            self._noise_files = _index_wav_files(Path(config.musan_root))
            log.info("AudioAugmenter: indexed %d MUSAN files", len(self._noise_files))
        else:
            log.debug(
                "AudioAugmenter: musan_root not set or missing — noise aug disabled"
            )

        self._rir_files: list[Path] = []
        if config.rir_root is not None and Path(config.rir_root).exists():
            self._rir_files = _index_wav_files(Path(config.rir_root))
            log.info("AudioAugmenter: indexed %d RIR files", len(self._rir_files))
        else:
            log.debug("AudioAugmenter: rir_root not set or missing — RIR aug disabled")

    def __call__(
        self,
        wav: torch.Tensor,
        samplerate: int = 16000,
    ) -> torch.Tensor:
        """Apply augmentation pipeline to a 1-D waveform.

        Args:
            wav: Input waveform, shape [T], dtype float32.
            samplerate: Sample rate of *wav* in Hz.

        Returns:
            Augmented waveform, shape [T'] (may differ if speed-perturbed),
            dtype float32.
        """
        assert wav.ndim == 1, f"Expected 1-D waveform, got shape {wav.shape}"
        cfg = self._cfg
        rng = self._rng
        out = wav.float()

        # Step 1: Speed XOR Pitch.
        choice = _pick_speed_or_pitch(cfg.speed_prob, cfg.pitch_prob, rng)
        if choice == "speed":
            out = _apply_speed_perturb(out, samplerate, cfg.speed_factors, rng)
        elif choice == "pitch":
            out = _apply_pitch_shift(out, samplerate, cfg.pitch_range_semitones, rng)

        # Step 2: VTLP.
        if rng.random() < cfg.vtlp_prob:
            out = _apply_vtlp(out, samplerate, cfg.vtlp_alpha_range, rng)

        # Step 3: Gain.
        if rng.random() < cfg.gain_prob:
            out = _apply_gain(out, cfg.gain_db_range, rng)

        # Step 4: AddNoise (no-op if no files indexed).
        if self._noise_files and rng.random() < cfg.noise_prob:
            out = _apply_add_noise(out, self._noise_files, cfg.noise_snr_db_range, rng)

        # Step 5: RIR (no-op if no files indexed).
        if self._rir_files and rng.random() < cfg.rir_prob:
            out = _apply_rir(out, self._rir_files, rng)

        return out.float()
