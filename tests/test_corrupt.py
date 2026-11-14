import numpy as np
import pytest

from murmur.corrupt import CONDITIONS_DEFAULT, apply, condition_seed, effective_snr_db, pink_noise
from murmur.preprocess import preprocess_signal

CFG = {"sample_rate_hz": 4000, "bandpass_hz": [20.0, 1000.0], "butter_order": 4, "zero_phase": True}
FS = 4000


def sig(n=FS * 10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return (0.3 * np.sin(2 * np.pi * 50 * t) * (np.sin(2 * np.pi * 1.2 * t) > 0.6) + 0.01 * rng.standard_normal(n)).astype(np.float32)


C = {c["name"]: c for c in CONDITIONS_DEFAULT}


def test_identity_reproduces_clean():
    x = sig()
    y, _ = apply(x, FS, C["identity"], 1)
    np.testing.assert_array_equal(y, x)
    assert effective_snr_db(preprocess_signal(x, FS, CFG), preprocess_signal(y, FS, CFG)) == float("inf")


@pytest.mark.parametrize("name,snr", [("white_20dB", 20), ("white_0dB", 0), ("pink_10dB", 10)])
def test_nominal_snr_exact_on_raw(name, snr):
    x = sig()
    y, info = apply(x, FS, C[name], 7)
    noise = y.astype(np.float64) - x
    assert 10 * np.log10(np.mean(x.astype(np.float64) ** 2) / np.mean(noise ** 2)) == pytest.approx(snr, abs=0.01)


def test_effective_snr_differs_from_nominal():
    """Band-pass removes out-of-band white noise: effective SNR at the model input should exceed nominal."""
    x = sig()
    y, _ = apply(x, FS, C["white_10dB"], 3)
    eff = effective_snr_db(preprocess_signal(x, FS, CFG), preprocess_signal(y, FS, CFG))
    assert eff > 10.5


def test_pink_spectrum_slope():
    n = 2 ** 16
    p = np.mean([np.abs(np.fft.rfft(pink_noise(n, np.random.default_rng(s)))) ** 2 for s in range(20)], 0)
    f = np.arange(len(p))
    sel = (f > 10) & (f < len(p) // 2)
    slope = np.polyfit(np.log(f[sel]), np.log(p[sel]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.1)


def test_clip_and_dropout_fractions():
    x = sig()
    _, i1 = apply(x, FS, C["clip_p99"], 0)
    assert i1["clip_fraction_raw"] == pytest.approx(0.01, abs=0.002)
    y, i2 = apply(x, FS, C["dropout_10pct"], 0)
    assert i2["dropout_fraction_raw"] == pytest.approx(0.10, abs=0.01)
    assert np.mean(y == 0) >= 0.09


def test_bursts_present():
    x = sig(FS * 30)
    y, info = apply(x, FS, C["burst_5per10s"], 11)
    assert info["n_bursts"] > 0 and 0 < info["burst_fraction"] < 0.5


def test_matched_seeds():
    assert condition_seed("123_AV", "white_0dB") == condition_seed("123_AV", "white_0dB")
    assert condition_seed("123_AV", "white_0dB") != condition_seed("123_PV", "white_0dB")
    x = sig()
    a, _ = apply(x, FS, C["pink_0dB"], condition_seed("r", "pink_0dB"))
    b, _ = apply(x, FS, C["pink_0dB"], condition_seed("r", "pink_0dB"))
    np.testing.assert_array_equal(a, b)
