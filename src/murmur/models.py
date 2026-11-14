"""Three architectures sharing one preprocessing pipeline (independently designed for this study).

Input to every model: a batch of band-passed waveform segments (B, L) at 4 kHz.
Normalisation is fitted on the model's own training set and frozen as buffers (proposal: "normalisation
fitted on clean outer-train data and held fixed"), so corrupted inputs are not re-normalised away.
  wave_cnn : x * (1 / training RMS)                      -> 1-D CNN
  spec_cnn : log-power STFT, bins <= 1000 Hz, per-bin z-score from training set -> 2-D CNN
  crnn     : same spectrogram front-end -> 2-D conv (frequency pooling) -> BiGRU -> pooled -> linear
`input_transform(x)` returns exactly what the classifier body sees (used for effective-SNR measurement).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecFrontEnd(nn.Module):
    def __init__(self, fs=4000, n_fft=256, hop=64, fmax_hz=1000.0, log_eps=1e-8):
        super().__init__()
        self.n_fft, self.hop, self.log_eps = n_fft, hop, log_eps
        self.n_bins = int(fmax_hz * n_fft / fs) + 1
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("mean", torch.zeros(self.n_bins))
        self.register_buffer("std", torch.ones(self.n_bins))

    def logspec(self, x):
        X = torch.stft(x.float(), self.n_fft, self.hop, window=self.window, center=True, return_complex=True)
        P = X.real ** 2 + X.imag ** 2
        return torch.log(P[:, : self.n_bins] + self.log_eps)          # (B, F, T)

    def forward(self, x):
        S = self.logspec(x)
        return ((S - self.mean[None, :, None]) / self.std[None, :, None]).unsqueeze(1)   # (B,1,F,T)

    @torch.no_grad()
    def fit(self, segments_iter):
        """Streaming per-bin mean/std over training segments."""
        n, s1, s2 = 0, None, None
        for x in segments_iter:
            S = self.logspec(x).double()
            cnt = S.shape[0] * S.shape[2]
            a, b = S.sum((0, 2)), (S ** 2).sum((0, 2))
            s1 = a if s1 is None else s1 + a
            s2 = b if s2 is None else s2 + b
            n += cnt
        mean = s1 / n
        std = torch.sqrt(torch.clamp(s2 / n - mean ** 2, min=1e-12))
        self.mean.copy_(mean.float())
        self.std.copy_(std.float())


def _block1d(cin, cout, k=9, pool=4):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
        nn.Conv1d(cout, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
        nn.MaxPool1d(pool),
    )


def _block2d(cin, cout, pool):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class WaveCNN(nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.register_buffer("scale", torch.ones(()))
        self.stem = nn.Sequential(nn.Conv1d(1, 16, 63, stride=2, padding=31, bias=False), nn.BatchNorm1d(16), nn.ReLU(inplace=True))
        self.body = nn.Sequential(_block1d(16, 32), _block1d(32, 64), _block1d(64, 128), _block1d(128, 128))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

    def input_transform(self, x):
        return (x.float() * self.scale).unsqueeze(1)

    def forward(self, x):
        h = self.body(self.stem(self.input_transform(x)))
        return self.head(h.mean(-1)).squeeze(-1)

    @torch.no_grad()
    def fit_norm(self, segments_iter):
        n, s2 = 0, 0.0
        for x in segments_iter:
            s2 += float((x.double() ** 2).sum())
            n += x.numel()
        self.scale.fill_(1.0 / max((s2 / n) ** 0.5, 1e-8))


class SpecCNN(nn.Module):
    def __init__(self, spec: dict, fs=4000):
        super().__init__()
        self.fe = SpecFrontEnd(fs, **spec)
        self.body = nn.Sequential(_block2d(1, 16, 2), _block2d(16, 32, 2), _block2d(32, 64, 2), _block2d(64, 128, 2))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

    def input_transform(self, x):
        return self.fe(x)

    def forward(self, x):
        h = self.body(self.fe(x))
        return self.head(h.mean((-1, -2))).squeeze(-1)

    def fit_norm(self, segments_iter):
        self.fe.fit(segments_iter)


class CRNN(nn.Module):
    def __init__(self, spec: dict, fs=4000, hidden=64):
        super().__init__()
        self.fe = SpecFrontEnd(fs, **spec)
        self.conv = nn.Sequential(_block2d(1, 32, (2, 2)), _block2d(32, 64, (2, 2)), _block2d(64, 128, (4, 1)))
        f_out = self.fe.n_bins // 2 // 2 // 4
        self.gru = nn.GRU(128 * f_out, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * hidden, 1))

    def input_transform(self, x):
        return self.fe(x)

    def forward(self, x):
        h = self.conv(self.fe(x))                    # (B, C, F', T')
        B, C, Fp, T = h.shape
        h = h.permute(0, 3, 1, 2).reshape(B, T, C * Fp)
        h, _ = self.gru(h)
        pooled = torch.cat([h.mean(1), h.amax(1)], -1)
        return self.head(pooled).squeeze(-1)

    def fit_norm(self, segments_iter):
        self.fe.fit(segments_iter)


def build_model(arch: str, cfg: dict) -> nn.Module:
    if arch == "wave_cnn":
        return WaveCNN()
    if arch == "spec_cnn":
        return SpecCNN(cfg["spectrogram"])
    if arch == "crnn":
        return CRNN(cfg["spectrogram"])
    raise ValueError(arch)
