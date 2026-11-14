import numpy as np
import pytest

from murmur.preprocess import preprocess_signal, segment

CFG = {"sample_rate_hz": 4000, "bandpass_hz": [20.0, 1000.0], "butter_order": 4, "zero_phase": True}


def _gain_db(f, fs=4000, dur=10.0):
    t = np.arange(int(fs * dur)) / fs
    x = np.sin(2 * np.pi * f * t).astype(np.float32)
    y = preprocess_signal(x, fs, CFG)
    mid = slice(int(fs * 2), int(fs * 8))  # avoid edges
    return 20 * np.log10(np.std(y[mid]) / np.std(x[mid]))


def test_passband_flat():
    for f in (60, 150, 300, 600):
        assert abs(_gain_db(f)) < 0.5, f


def test_stopband_attenuated():
    assert _gain_db(8) < -40          # low-frequency drift removed
    assert _gain_db(1800) < -20


def test_resample_path():
    fs = 8000
    t = np.arange(fs * 5) / fs
    x = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    y = preprocess_signal(x, fs, CFG)
    assert len(y) == 4000 * 5


def test_identity_fs_no_resample():
    x = np.random.default_rng(0).standard_normal(4000 * 3).astype(np.float32)
    assert len(preprocess_signal(x, 4000, CFG)) == len(x)


def test_segment_cover_and_pad():
    fs = 4000
    y = np.arange(int(fs * 9.5), dtype=np.float32)
    s = segment(y, fs, 4.0, 2.0)
    assert s.shape[1] == 16000
    assert s[0, 0] == 0 and s[-1, -1] == y[-1]          # tail covered
    short = segment(np.ones(1000, np.float32), fs, 4.0, 2.0)
    assert short.shape == (1, 16000) and short[0, :1000].sum() == 1000 and short[0, 1000:].sum() == 0
    exact = segment(np.ones(16000, np.float32), fs, 4.0, 2.0)
    assert exact.shape == (1, 16000)


def test_cache_roundtrip(tmp_path, cohort):
    import pandas as pd
    from murmur.paths import dataset_dir
    from murmur.preprocess import WaveCache, build_cache, read_wav
    _, rec, _ = cohort
    sub = rec.head(5)
    ds = dataset_dir()
    build_cache(sub, ds, tmp_path, CFG)
    wc = WaveCache(tmp_path)
    for r in sub.itertuples():
        x, fs = read_wav(ds / r.wav)
        np.testing.assert_array_equal(wc.signal(r.recording_id), preprocess_signal(x, fs, CFG))
