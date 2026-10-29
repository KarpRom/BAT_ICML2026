"""Waveform augmentations for the SSL student views (approach AUG).

From AudioMosaic (arXiv 2605.14231), whose ablation finds waveform augmentation is its largest
factor, restricted to the augmentations that keep every time-frequency patch in place: BAT's
local loss regresses each masked student patch onto the teacher's patch at the same position,
so time-stretch and pitch-shift (which move content) are left out. Polarity and gain are left
out too: the power mel + per-clip min-max normalization undoes them.

Kept, each drawn independently per waveform (= per view, so the 16 views of a clip differ):
- filtering: high-pass, low-pass (band-limited sources: 34% of AudioSet clips have a brickwall
  lowpass, study_audio) or band-stop, zero-phase Butterworth magnitude responses;
- additive white / pink / brown noise at a random SNR.
The teacher keeps the clean clip: the student learns to predict clean targets from degraded views.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class WaveformAugment(nn.Module):
    def __init__(self, sr: int = 16_000, p_filter: float = 0.5, p_noise: float = 0.5,
                 snr_db: Tuple[float, float] = (5., 40.)):
        super().__init__()
        self.sr, self.p_filter, self.p_noise, self.snr_db = sr, p_filter, p_noise, snr_db

    def _log_uniform(self, n: int, lo: float, hi: float, device) -> torch.Tensor:
        return torch.exp(torch.empty(n, 1, device=device).uniform_(math.log(lo), math.log(hi)))

    def _filters(self, n: int, f: torch.Tensor) -> torch.Tensor:
        """(n, F) magnitude responses: a third each of high-pass, low-pass and band-stop."""
        dev = f.device
        f = f.clamp(min=1.)[None]  # (1, F); avoids 0 ** negative
        order = torch.randint(2, 9, (n, 1), device=dev).float()
        hp = 1 / torch.sqrt(1 + (self._log_uniform(n, 50., 1000., dev) / f) ** (2 * order))
        lp = 1 / torch.sqrt(1 + (f / self._log_uniform(n, 2000., 7500., dev)) ** (2 * order))
        lo = self._log_uniform(n, 200., 5000., dev)
        hi = lo * self._log_uniform(n, 1.2, 2.5, dev)  # band of 0.25 to 1.3 octaves
        band = (1 / torch.sqrt(1 + (lo / f) ** (2 * order))) * (1 / torch.sqrt(1 + (f / hi) ** (2 * order)))
        bs = 1 - band
        kind = torch.randint(0, 3, (n, 1), device=dev)
        return torch.where(kind == 0, hp, torch.where(kind == 1, lp, bs))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, L) waveforms -> (N, L) augmented waveforms."""
        n, length = x.shape
        dev = x.device
        freqs = torch.fft.rfftfreq(length, d=1 / self.sr, device=dev)
        spec = torch.fft.rfft(x.float())

        do_filter = torch.rand(n, 1, device=dev) < self.p_filter
        h = torch.where(do_filter, self._filters(n, freqs), torch.ones(1, device=dev))
        y = torch.fft.irfft(spec * h, n=length)

        do_noise = torch.rand(n, 1, device=dev) < self.p_noise
        beta = torch.randint(0, 3, (n, 1), device=dev).float()  # 0 white, 1 pink, 2 brown
        shape = freqs.clamp(min=freqs[1])[None] ** (-beta / 2)
        noise_spec = torch.complex(torch.randn_like(h), torch.randn_like(h)) * shape
        noise = torch.fft.irfft(noise_spec, n=length)
        snr = torch.empty(n, 1, device=dev).uniform_(*self.snr_db)
        p_sig = y.pow(2).mean(dim=1, keepdim=True)
        p_noise = noise.pow(2).mean(dim=1, keepdim=True).clamp(min=1e-12)
        noise = noise * torch.sqrt(p_sig / p_noise / 10 ** (snr / 10))
        return (y + do_noise * noise).to(x.dtype)
