def test_expected_counts(cohort, cfg):
    enc, rec, log = cohort
    for k, v in cfg["expected_counts"].items():
        assert log[k] == v, (k, log[k], v)


def test_exclusions_logged(cohort):
    _, _, log = cohort
    assert log["raw_encounters"] == 942
    assert log["excluded_murmur_unknown"] == 68
    assert log["excluded_missing_age"] == 72
    assert log["raw_encounters"] - log["excluded_murmur_unknown"] - log["excluded_missing_age"] == log["encounters"]


def test_age_groups_match_proposal(cohort):
    _, _, log = cohort
    assert log["age_groups"] == {"Neonate": 5, "Infant": 101, "Child": 627, "Adolescent": 69}
    assert log["age_groups_present"] == {"Neonate": 1, "Infant": 25, "Child": 132, "Adolescent": 16}


def test_unique_encounters_and_recordings(cohort):
    enc, rec, _ = cohort
    assert enc["patient_id"].is_unique
    assert rec["recording_id"].is_unique
    assert set(rec["patient_id"]) == set(enc["patient_id"])
    assert (enc["n_recordings"] >= 1).all()


def test_individual_links_consistent(cohort):
    enc, _, _ = cohort
    # every individual id is one of its own encounters' patient ids or a linked raw id; individuals <= encounters
    assert enc["individual_id"].nunique() <= len(enc)
    # linked encounters share label-independent identity only; no constraint on labels, but sex must agree
    multi = enc.groupby("individual_id").filter(lambda g: len(g) > 1)
    sex_conflicts = multi.groupby("individual_id")["sex"].nunique()
    assert (sex_conflicts == 1).all(), "linked encounters with conflicting sex"
