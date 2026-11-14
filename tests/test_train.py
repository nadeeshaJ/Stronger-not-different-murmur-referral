"""CPU smoke test of the training harness on synthetic audio (murmur = band-limited noise between beats)."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from murmur.models import build_model  # noqa: E402
from murmur.partitions import make_partitions  # noqa: E402
from murmur.train import DeviceData, RunSpec, enumerate_runs, score_encounters, train_one  # noqa: E402

FS = 4000
CFG = {
    "archs": ["wave_cnn", "spec_cnn", "crnn"], "member_seeds": [0], "segment_s": 4.0, "hop_s": 2.0,
    "train_label": "location", "segment_to_recording": "mean_prob", "recording_to_encounter": "max",
    "epochs": 4, "batch_size": 32, "crops_per_recording": 3, "lr": 3e-3, "weight_decay": 1e-4,
    "class_weighting": "pos_weight", "amp": False, "deterministic": True,
    "spectrogram": {"n_fft": 256, "hop": 64, "fmax_hz": 1000.0, "log_eps": 1e-8},
}


def synth(n_enc=80, seed=0):
    rng = np.random.default_rng(seed)
    enc, rec, sig, off = [], [], [], 0
    for i in range(n_enc):
        pid = str(1000 + i)
        lab = int(rng.random() < 0.35)
        locs = ["AV", "PV", "TV", "MV"][: rng.integers(1, 4)]
        mlocs = "+".join(locs[:1]) if lab else ""
        enc.append({"patient_id": pid, "individual_id": pid if i % 10 else str(1000 + i - 1) if i else pid,
                    "label": lab, "age_group": "Child", "murmur_locations": mlocs})
        for loc in locs:
            n = int(rng.uniform(2.5, 9) * FS)
            t = np.arange(n) / FS
            beat = (np.sin(2 * np.pi * 1.3 * t) > 0.95).astype(float)
            x = 0.4 * beat * np.sin(2 * np.pi * 60 * t) + 0.01 * rng.standard_normal(n)
            if lab and loc in mlocs.split("+"):
                mur = rng.standard_normal(n)
                mur = np.convolve(mur, np.ones(5) / 5, "same") * (1 - beat) * 0.15
                x = x + mur
            sig.append(x.astype(np.float32))
            rec.append({"recording_id": f"{pid}_{loc}", "patient_id": pid, "location": loc, "offset": off, "length": n})
            off += n
    return pd.DataFrame(enc), pd.DataFrame(rec), np.concatenate(sig)


@pytest.fixture(scope="module")
def setup():
    enc, rec, wave = synth()
    folds = make_partitions(enc, [7], 5, 5, extra_strata=None)
    data = DeviceData(wave, rec, enc, CFG, torch.device("cpu"))
    return enc, rec, wave, folds, data


def test_location_labels(setup):
    enc, rec, wave, folds, data = setup
    pos_enc = set(enc.loc[enc.label == 1, "patient_id"])
    r = data.rec
    assert (r.loc[~r.patient_id.isin(pos_enc), "rec_label"] == 0).all()
    # in a positive encounter only the murmur location is positive
    assert r.loc[r.patient_id.isin(pos_enc)].groupby("patient_id").rec_label.sum().eq(1).all()


def test_gather_zero_pads_short(setup):
    _, rec, wave, _, data = setup
    r = int(np.argmin(rec.length.to_numpy()))
    st, ow = data.fixed_segments([r])
    x = data.gather(st, ow)
    n = int(rec.length[r])
    if n < data.L:
        assert float(x[0, n:].abs().sum()) == 0.0
        np.testing.assert_allclose(x[0, :n].numpy(), wave[rec.offset[r]: rec.offset[r] + n])


def test_enumerate_counts():
    assert len(enumerate_runs(5, 5, 5, ["a", "b", "c"], [0, 1, 2])) == 1125
    assert len(enumerate_runs(5, 5, 5, ["a", "b", "c"], [0, 1, 2], inner_inner_partitions=(0,))) == 1125 + 5 * 10 * 9


@pytest.mark.parametrize("arch", ["wave_cnn", "spec_cnn", "crnn"])
def test_train_learns_and_writes(setup, tmp_path, arch):
    enc, rec, wave, folds, data = setup
    spec = RunSpec(0, 0, (0,), arch, 0)
    out = train_one(spec, data, folds, CFG, tmp_path, [7], score_test=True, epochs=CFG["epochs"], log=lambda m: None)
    z = np.load(out / "preds.npz")
    m = json.loads((out / "manifest.json").read_text())
    assert z["oof_prob"].shape == (CFG["epochs"], len(z["oof_ids"]))
    assert z["test_prob"].shape[1] == (folds[(folds.partition == 0) & (folds.outer_fold == 0)].role == "test").sum()
    assert np.isfinite(z["oof_prob"]).all() and np.isfinite(z["test_prob"]).all()
    assert m["scored_test"] and (out / "final.pt").exists()
    # learns the obvious synthetic murmur: AUROC on test at last epoch
    from sklearn.metrics import roc_auc_score
    y = enc.set_index("patient_id").label[z["test_ids"]].to_numpy()
    assert roc_auc_score(y, z["test_prob"][-1]) > 0.8, arch


def test_pilot_never_scores_test(setup, tmp_path):
    enc, rec, wave, folds, data = setup
    out = train_one(RunSpec(0, 0, (1,), "wave_cnn", 0), data, folds, CFG, tmp_path, [7], score_test=False, epochs=1, log=lambda m: None)
    z = np.load(out / "preds.npz")
    assert len(z["test_ids"]) == 0 and z["test_prob"].size == 0


def test_inner_inner_scores_both_heldout(setup, tmp_path):
    enc, rec, wave, folds, data = setup
    out = train_one(RunSpec(0, 0, (1, 3), "wave_cnn", 0), data, folds, CFG, tmp_path, [7], score_test=True, epochs=1, log=lambda m: None)
    z = np.load(out / "preds.npz")
    assert set(np.unique(z["oof_heldout_fold"])) == {1, 3} and len(z["test_ids"]) == 0


def test_deterministic_cpu(setup, tmp_path):
    enc, rec, wave, folds, data = setup
    a = train_one(RunSpec(0, 1, (2,), "spec_cnn", 0), data, folds, CFG, tmp_path / "a", [7], False, epochs=1, log=lambda m: None)
    b = train_one(RunSpec(0, 1, (2,), "spec_cnn", 0), data, folds, CFG, tmp_path / "b", [7], False, epochs=1, log=lambda m: None)
    np.testing.assert_array_equal(np.load(a / "preds.npz")["oof_prob"], np.load(b / "preds.npz")["oof_prob"])


def test_norm_frozen_against_corruption(setup):
    """Normalisation buffers are fitted once; scaling the input must change what the body sees."""
    _, _, _, _, data = setup
    m = build_model("spec_cnn", CFG)
    st, ow = data.fixed_segments(np.arange(5))
    x = data.gather(st, ow)
    m.fit_norm([x])
    a = m.input_transform(x)
    b = m.input_transform(x * 2)
    assert not torch.allclose(a, b)
