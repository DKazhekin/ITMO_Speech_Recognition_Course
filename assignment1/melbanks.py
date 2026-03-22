from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
        ):
        super(LogMelFilterBanks, self).__init__()

        # general params and params defined by the exercise
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.register_buffer('window', torch.hann_window(self.window_length))
        
        # Do correct initialization of stft params below:
        self.hop_length = hop_length
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power

        # Do correct initialization of mel fbanks params below:
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.n_mels = n_mels
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        # finish parameters initialization
        self.register_buffer('mel_fbanks', self._init_melscale_fbanks())

    def _init_melscale_fbanks(self):
        # To access attributes, use self.<parameter_name>
        return F.melscale_fbanks(
            # Turns a normal STFT into a mel frequency STFT with triangular filter banks
            # make a full and correct function call
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz if self.f_max_hz is not None else self.samplerate // 2,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale
        )

    def spectrogram(self, x):
        # x - is an input signal
        return torch.stft(
            # make a full and correct function call
            x, self.n_fft, self.hop_length, self.window_length, self.window, self.center, self.pad_mode, self.normalize_stft, self.onesided, self.return_complex
        )

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        
        S = self.spectrogram(x)
        transpose_filter_banks = self.mel_fbanks.T
        power_spectrum = torch.pow(S.abs(), self.power)
        mel_S = torch.matmul(transpose_filter_banks, power_spectrum)
        return (mel_S + 1e-6).log()
