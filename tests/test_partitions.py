import numpy as np
import pandas as pd

from murmur.partitions import make_partitions


def test_each_encounter_tested_once_per_partition(folds, cohort):
    enc = cohort[0]
    t = folds[folds.role == "test"]
    c = t.groupby(["partition", "patient_id"]).size()
    assert (c == 1).all()
    assert t.groupby("partition").size().eq(len(enc)).all()


def test_no_individual_crosses_outer_split(folds):
    for (p, o), g in folds.groupby(["partition", "outer_fold"]):
        te = set(g.loc[g.role == "test", "individual_id"])
        dev = set(g.loc[g.role == "dev", "individual_id"])
        assert not te & dev, (p, o)


def test_no_individual_crosses_inner_split(folds):
    dev = folds[folds.role == "dev"]
    n_folds_per_ind = dev.groupby(["partition", "outer_fold", "individual_id"])["inner_fold"].nunique()
    assert (n_folds_per_ind == 1).all()


def test_inner_folds_cover_dev(folds):
    dev = folds[folds.role == "dev"]
    assert (dev.inner_fold >= 0).all()
    assert (folds.loc[folds.role == "test", "inner_fold"] == -1).all()
    assert dev.groupby(["partition", "outer_fold"]).inner_fold.nunique().eq(5).all()


def test_stratification(folds, cohort):
    prev = cohort[0]["label"].mean()
    t = folds[folds.role == "test"].groupby(["partition", "outer_fold"]).label.mean()
    assert (t - prev).abs().max() < 0.015
    i = folds[folds.role == "dev"].groupby(["partition", "outer_fold", "inner_fold"]).label.mean()
    assert (i - prev).abs().max() < 0.03
    sizes = folds[folds.role == "test"].groupby(["partition", "outer_fold"]).size()
    assert sizes.max() - sizes.min() <= 3


def test_partitions_differ(folds):
    t = folds[folds.role == "test"].pivot_table(index="patient_id", columns="partition", values="outer_fold")
    # assignments are not identical across partitions
    for a in range(5):
        for b in range(a + 1, 5):
            assert (t[a] != t[b]).mean() > 0.5


def test_deterministic(cohort, cfg, folds):
    pc = cfg["partitions"]
    again = make_partitions(cohort[0], pc["seeds"], pc["n_outer"], pc["n_inner"])
    pd.testing.assert_frame_equal(folds, again)


def test_synthetic_grouping():
    rng = np.random.default_rng(0)
    n = 300
    enc = pd.DataFrame({
        "patient_id": [str(i) for i in range(n)],
        "individual_id": [str(i // 2) for i in range(n)],  # pairs share an individual
        "label": rng.integers(0, 2, n),
        "age_group": "Child",
    })
    f = make_partitions(enc, [1], 5, 5)
    for o, g in f.groupby("outer_fold"):
        assert not set(g.loc[g.role == "test", "individual_id"]) & set(g.loc[g.role == "dev", "individual_id"])


def test_age_balance(folds):
    t = folds[folds.role == "test"]
    pres = t[t.label == 1].groupby(["partition", "outer_fold", "age_group"]).size().unstack(fill_value=0)
    allc = t.groupby(["partition", "outer_fold", "age_group"]).size().unstack(fill_value=0)
    # Infant: 101 encounters / 25 present overall -> ~20 / ~5 per test fold
    assert allc["Infant"].max() - allc["Infant"].min() <= 4
    assert pres["Infant"].max() - pres["Infant"].min() <= 2
    assert pres["Adolescent"].max() - pres["Adolescent"].min() <= 3  # 16 present; linked individuals limit balance
