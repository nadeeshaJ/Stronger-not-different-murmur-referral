"""Controlled acoustic corruption (Study B).

Corruption is applied to the RAW signal (float, before band-pass), then the standard preprocessing runs.
Seeds are matched per (recording, condition), so every model sees exactly the same corrupted audio.

Effective SNR is measured where the model looks: 10*log10( P(proc(clean)) / P(proc(corrupt) - proc(clean)) ),
with proc = the architecture's actual input transform (filtered waveform here; spectrogram added in Phase 2).
"""
from __future__ import annotations

import hashlib

import numpy as np

CONDITIONS_DEFAULT = [
    {"name": "identity"},
    *[{"name": f"white_{s}dB", "type": "white", "snr_db": s} for s in (20, 10, 0)],
    *[{"name": f"pink_{s}dB", "type": "pink", "snr_db": s} for s in (20, 10, 0)],
    # Bursts: 2 or 5 Hann-windowed broadband bursts per 10 s at 0 dB burst SNR (handling/rubbing artefacts).
    *[{"name": f"burst_{r}per10s", "type": "burst", "rate_per_10s": r, "burst_snr_db": 0, "dur_s": (0.05, 0.2)} for r in (2, 5)],
    *[{"name": f"clip_p{q}", "type": "clip", "percentile": q} for q in (99.9, 99.5, 99)],
    *[{"name": f"dropout_{int(f*100)}pct", "type": "dropout", "fraction": f, "chunk_s": 0.1} for f in (0.05, 0.10, 0.20)],
]


def condition_seed(recording_id: str, condition: str, global_seed: int = 20261109) -> int:
    h = hashlib.sha256(f"{global_seed}|{recording_id}|{condition}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def _power(x):
    return float(np.mean(np.asarray(x, np.float64) ** 2))


def _scale_to_snr(signal, noise, snr_db):
    ps, pn = _power(signal), _power(noise)
    if pn == 0:
        return noise
    return noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def pink_noise(n: int, rng) -> np.ndarray:
    """1/f power spectrum via FFT shaping (DC removed)."""
    w = rng.standard_normal(n)
    X = np.fft.rfft(w)
    f = np.arange(len(X), dtype=float)
    f[0] = 1.0
    X = X / np.sqrt(f)
    X[0] = 0
    return np.fft.irfft(X, n)


def apply(x: np.ndarray, fs: int, cond: dict, seed: int) -> tuple[np.ndarray, dict]:
    """Return (corrupted raw signal, realised stats measured on the raw signal)."""
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    t = cond.get("type")
    info = {"condition": cond["name"]}
    if cond["name"] == "identity" or t is None:
        y = x.copy()
    elif t in ("white", "pink"):
        noise = rng.standard_normal(len(x)) if t == "white" else pink_noise(len(x), rng)
        noise = _scale_to_snr(x, noise, cond["snr_db"])
        y = x + noise
        info["nominal_snr_db"] = cond["snr_db"]
    elif t == "burst":
        y = x.copy()
        n_b = rng.poisson(cond["rate_per_10s"] * len(x) / fs / 10)
        lo, hi = cond["dur_s"]
        mask = np.zeros(len(x), bool)
        ps = _power(x)
        for _ in range(n_b):
            L = int(rng.uniform(lo, hi) * fs)
            s0 = int(rng.integers(0, max(1, len(x) - L)))
            b = rng.standard_normal(L) * np.hanning(L)
            b *= np.sqrt(ps / (_power(b) * 10 ** (cond["burst_snr_db"] / 10)))
            y[s0 : s0 + L] += b
            mask[s0 : s0 + L] = True
        info["n_bursts"] = int(n_b)
        info["burst_fraction"] = float(mask.mean())
        info["burst_energy_ratio"] = float(np.sum((y - x) ** 2) / max(np.sum(x ** 2), 1e-20))
    elif t == "clip":
        c = float(np.percentile(np.abs(x), cond["percentile"]))
        y = np.clip(x, -c, c)
        info["clip_level"] = c
        # clipping levels are set so that 0.1%, 0.5% and 1% of samples clip (percentile clipping at 95/90
        # removed most S1/S2 energy in the label-free check: median effective SNR 3.5 / 2.4 dB)
        info["clip_fraction_raw"] = float(np.mean(np.abs(x) > c))
    elif t == "dropout":
        L = max(1, int(cond["chunk_s"] * fs))
        n_chunks = int(np.ceil(len(x) / L))
        k = int(round(cond["fraction"] * n_chunks))
        drop = rng.choice(n_chunks, size=k, replace=False)
        y = x.copy()
        mask = np.zeros(len(x), bool)
        for c in drop:
            mask[c * L : (c + 1) * L] = True
        y[mask] = 0.0
        info["dropout_fraction_raw"] = float(mask.mean())
    else:
        raise ValueError(t)
    return y.astype(np.float32), info


def effective_snr_db(clean_proc: np.ndarray, corrupt_proc: np.ndarray) -> float:
    """SNR of what reaches the model: clean-processed power over power of the processed difference."""
    d = np.asarray(corrupt_proc, np.float64) - np.asarray(clean_proc, np.float64)
    pd_ = _power(d)
    if pd_ == 0:
        return float("inf")
    return 10 * np.log10(_power(clean_proc) / pd_)
