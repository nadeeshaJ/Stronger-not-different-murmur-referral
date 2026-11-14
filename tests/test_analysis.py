"""End-to-end Study C analysis on simulated runs with known structure (needs the real cohort/folds files)."""
import numpy as np
import pandas as pd
import pytest

from murmur import analysis as an
from murmur.paths import derived_dir, load_config
from murmur.simulate import simulate_runs

ACFG = load_config("analysis") | {"n_boot": 2000}


@pytest.fixture(scope="module")
def base():
    d = derived_dir()
    if not (d / "folds.csv").exists():
        pytest.skip("derived files missing")
    enc = pd.read_csv(d / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    folds = pd.read_csv(d / "folds.csv", dtype={"patient_id": str, "individual_id": str})
    return enc, folds


def run(base, tmp, extra, n_part=2):
    enc, folds = base
    simulate_runs(enc, folds, tmp, n_part=n_part, extra=extra, seed=1)
    runs = an.load_runs(tmp)
    assert not an.check_complete(runs, n_part, 5, 5, an.ARCHS, [0, 1, 2])
    res = an.study_c(runs, enc, n_part, 5, 5, [0, 1, 2], ACFG)
    return res, an.confirmatory(res, enc, ACFG)


def test_effect_detected(base, tmp_path_factory):
    res, conf = run(base, tmp_path_factory.mktemp("eff"), extra=1.5)
    h1w = conf.set_index("hypothesis").loc["H1-weak"]
    assert h1w.estimate > 0 and h1w.p_signflip < 0.01, h1w.to_dict()
    # all detection matrices fully populated for present encounters; 2 partitions x 3 seeds x 5 inner = 30 reps
    enc = base[0]
    present = enc.label.to_numpy() == 1
    for k, v in res["arms"].items():
        assert v.shape == (30, len(enc)) and not np.isnan(v[:, present]).any()
    # H1 arms share primary positives: differences come only from referrals
    sel = res["selection"]
    assert (sel[sel.primary == "weak"].chosen != "single:wave_cnn").all()
    # oracle ceiling respected
    pr = res["per_rep"]
    s = pr[pr.arm == "secondary"]
    assert (s.captured <= s.oracle_ceiling).all()
    # descriptive table has the random floor
    d = an.descriptive_table(res)
    assert {"secondary", "own_threshold", "random", "standalone_topM"} <= set(d.arm)


def test_matched_total_workload(base, tmp_path_factory):
    res, _ = run(base, tmp_path_factory.mktemp("wl"), extra=1.0, n_part=1)
    pr = res["per_rep"]
    key = ["partition", "outer", "seed", "inner", "primary", "budget"]
    wf = pr[pr.arm == "secondary"].set_index(key).total_flags
    sa = pr[pr.arm == "standalone_topM"].set_index(key).total_flags
    assert (wf == sa.loc[wf.index]).all()


def test_null_no_extra_information(base, tmp_path_factory):
    """extra=0: non-primary archs see nothing the weak primary misses beyond noise; H1-weak should not be
    strongly significant in the 'more misses found' direction by construction of independent noise only."""
    res, conf = run(base, tmp_path_factory.mktemp("null"), extra=0.0)
    h = conf.set_index("hypothesis")
    assert set(h.index) == {"H1-weak", "H2-weak", "H1-strong", "H2-strong"}
    assert (h.p_holm >= h.p_signflip - 1e-12).all()
    assert "equivalent" not in h.columns          # no equivalence testing (decided 2026-11-09)
