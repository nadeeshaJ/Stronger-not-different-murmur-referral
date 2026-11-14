"""Signal preprocessing, identical for every architecture.

raw int16 wav -> float32 in [-1, 1] -> (resample if fs != target; CirCor is already 4 kHz)
-> zero-phase Butterworth band-pass 20-1000 Hz -> fixed-length segments.

The waveform cache stores the band-passed signal of every recording, concatenated, in one float32
memmap plus an index (offset, length). Segmentation and any normalisation happen at load time, so the
cache is shared by all architectures and Study B can apply the *same* function to corrupted raw audio.
"""
from __future__ import annotations

import json
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt, sosfilt

from .hashing import sha256_file


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    x, fs = sf.read(str(path), dtype="int16", always_2d=False)
    return x.astype(np.float32) / 32768.0, fs


def design_bandpass(fs: int, band=(20.0, 1000.0), order: int = 4):
    return butter(order, band, btype="bandpass", fs=fs, output="sos")


def preprocess_signal(x: np.ndarray, fs: int, cfg: dict) -> np.ndarray:
    """Raw float signal -> band-passed float32 at target rate. Pure function; used for clean and corrupted audio."""
    target = cfg["sample_rate_hz"]
    if fs != target:
        g = gcd(fs, target)
        x = resample_poly(x, target // g, fs // g)
    sos = design_bandpass(target, tuple(cfg["bandpass_hz"]), cfg["butter_order"])
    y = sosfiltfilt(sos, x) if cfg["zero_phase"] else sosfilt(sos, x)
    return y.astype(np.float32)


def segment(y: np.ndarray, fs: int, seg_s: float, hop_s: float) -> np.ndarray:
    """Fixed windows of seg_s with hop hop_s. A final partial window is aligned to the end so the tail is
    covered; recordings shorter than one window are zero-padded on the right. Returns (n_seg, L)."""
    L, H = int(round(seg_s * fs)), int(round(hop_s * fs))
    n = len(y)
    if n <= L:
        out = np.zeros((1, L), dtype=np.float32)
        out[0, :n] = y
        return out
    starts = list(range(0, n - L + 1, H))
    if starts[-1] + L < n:
        starts.append(n - L)
    return np.stack([y[s : s + L] for s in starts]).astype(np.float32)


def build_cache(rec: pd.DataFrame, ds: Path, out_dir: Path, cfg: dict) -> dict:
    """Write waveforms.f32 (memmap) + index.csv + meta.json to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sigs, idx, off = [], [], 0
    for r in rec.itertuples(index=False):
        x, fs = read_wav(ds / r.wav)
        y = preprocess_signal(x, fs, cfg)
        sigs.append(y)
        idx.append({"recording_id": r.recording_id, "patient_id": r.patient_id, "location": r.location,
                    "offset": off, "length": len(y), "raw_fs": fs, "raw_len": len(x)})
        off += len(y)
    mm = np.lib.format.open_memmap(out_dir / "waveforms.npy", mode="w+", dtype=np.float32, shape=(off,))
    for s, i in zip(sigs, idx):
        mm[i["offset"] : i["offset"] + i["length"]] = s
    mm.flush()
    del mm
    pd.DataFrame(idx).to_csv(out_dir / "index.csv", index=False, lineterminator="\n")
    meta = {"preprocess": cfg, "n_recordings": len(idx), "n_samples": off,
            "waveforms_sha256": sha256_file(out_dir / "waveforms.npy"),
            "index_sha256": sha256_file(out_dir / "index.csv")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class WaveCache:
    """Read-only access to the cache: per-recording filtered signal."""

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.wave = np.load(self.dir / "waveforms.npy", mmap_mode="r")
        self.index = pd.read_csv(self.dir / "index.csv", dtype={"patient_id": str})
        self._pos = {r: i for i, r in enumerate(self.index["recording_id"])}

    def signal(self, recording_id: str) -> np.ndarray:
        r = self.index.iloc[self._pos[recording_id]]
        return np.asarray(self.wave[r.offset : r.offset + r.length])
