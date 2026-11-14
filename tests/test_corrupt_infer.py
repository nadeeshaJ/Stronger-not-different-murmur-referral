import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from murmur.corrupt import CONDITIONS_DEFAULT  # noqa: E402
from murmur.corrupt_infer import corrupted_cache, infer_condition  # noqa: E402
from murmur.partitions import make_partitions  # noqa: E402
from murmur.preprocess import preprocess_signal  # noqa: E402
from murmur.train import DeviceData, RunSpec, score_encounters, train_one  # noqa: E402

from test_train import CFG, synth  # noqa: E402

PP = {"sample_rate_hz": 4000, "bandpass_hz": [20.0, 1000.0], "butter_order": 4, "zero_phase": True}
C = {c["name"]: c for c in CONDITIONS_DEFAULT}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    enc, rec, raw = synth(n_enc=50, seed=3)
    raws = {r.recording_id: raw[r.offset: r.offset + r.length] for r in rec.itertuples()}
    clean = np.concatenate([preprocess_signal(raws[r], 4000, PP) for r in rec.recording_id])
    folds = make_partitions(enc, [5], 5, 5, extra_strata=None)
    dev = torch.device("cpu")
    data = DeviceData(clean, rec, enc, CFG, dev)
    root = tmp_path_factory.mktemp("runs")
    d = train_one(RunSpec(0, 0, (0,), "spec_cnn", 0), data, folds, CFG, root, [5], True, epochs=2, log=lambda m: None)
    return enc, rec, raws, clean, folds, data, d


def test_identity_reproduces_clean(trained):
    enc, rec, raws, clean, folds, data, d = trained
    wave, info = corrupted_cache(rec, lambda rid: (raws[rid], 4000), C["identity"], PP, clean)
    np.testing.assert_array_equal(wave, clean)
    dc = DeviceData(wave, rec, enc, CFG, torch.device("cpu"))
    res, snr = infer_condition("identity", [d], data, dc, folds, CFG, torch.device("cpu"), rep_runs={d.name})
    z = np.load(d / "preds.npz")
    np.testing.assert_allclose(res["prob"][0, : res["n"][0]], z["test_prob"][-1], atol=1e-6)
    assert np.isinf(snr.effective_snr_db_input).all()


def test_corruption_changes_scores_and_measures_snr(trained):
    enc, rec, raws, clean, folds, data, d = trained
    wave, info = corrupted_cache(rec, lambda rid: (raws[rid], 4000), C["white_0dB"], PP, clean)
    assert np.isfinite(info.effective_snr_db_filtered).all()
    dc = DeviceData(wave, rec, enc, CFG, torch.device("cpu"))
    res, snr = infer_condition("white_0dB", [d], data, dc, folds, CFG, torch.device("cpu"), rep_runs={d.name})
    z = np.load(d / "preds.npz")
    assert np.max(np.abs(res["prob"][0, : res["n"][0]] - z["test_prob"][-1])) > 1e-3
    assert len(snr) > 0 and np.isfinite(snr.effective_snr_db_input).all()
