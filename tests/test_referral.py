import numpy as np
import pytest

from murmur.referral import (budget_k, capture_curve, detection_vector, refer, standalone_top_m,
                             summarise, top_k)


def toy():
    # 10 encounters; primary threshold 0.5
    p = np.array([0.9, 0.8, 0.6, 0.45, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01])
    y = np.array([1,   0,   1,   1,    0,    1,    0,    1,    0,    0])
    s = np.array([0.1, 0.1, 0.1, 0.2,  0.9,  0.8,  0.1,  0.7,  0.3,  0.0])  # secondary
    return p, y, s


def test_budget():
    assert budget_k(160, 120, 0.10, "all") == 16
    assert budget_k(160, 120, 0.10, "pred_neg") == 12
    assert budget_k(160, 10, 0.10, "all") == 10          # capped at predicted negatives
    assert budget_k(161, 120, 0.10, "all") == 17          # ceil
    assert budget_k(100, 100, 0.10, "all") == 10          # exact product, no float overshoot


def test_top_k_and_ties():
    sc = np.array([1, 3, 3, 2, 5.0])
    el = np.array([1, 1, 1, 1, 0], bool)
    m = top_k(sc, el, 2, seed=1)
    assert m.sum() == 2 and m[1] and m[2] and not m[4]
    m1 = top_k(sc, el, 1, seed=1); m2 = top_k(sc, el, 1, seed=1)
    assert (m1 == m2).all() and (m1[1] or m1[2])          # deterministic tie-break among the 3s


def test_h1_arms_by_hand():
    p, y, s = toy()
    # predicted negatives: idx 3..9 (7). b=0.2, base=all -> K=2
    sec = refer(p, 0.5, s, 0.2, "all")
    own = refer(p, 0.5, p, 0.2, "all")
    assert sec.k == own.k == 2
    assert set(np.flatnonzero(sec.referred)) == {4, 5}    # secondary scores 0.9, 0.8
    assert set(np.flatnonzero(own.referred)) == {3, 4}    # next-highest primary probs
    a, b = summarise(sec, y), summarise(own, y)
    assert a["eligible_fn"] == 3 and a["oracle_ceiling"] == 2
    assert a["captured"] == 1 and b["captured"] == 1
    assert a["primary_tp"] == 2 and a["detected"] == 3 and a["missed"] == 2
    assert a["total_flags"] == 5
    assert a["random_floor_expected"] == pytest.approx(2 * 3 / 7)


def test_h2_matched_total_workload():
    p, y, s = toy()
    wf = refer(p, 0.5, s, 0.2, "all")
    m = int(wf.flagged.sum())
    alone = standalone_top_m(p, m)
    assert alone.sum() == m == 5
    det_wf = np.nansum(detection_vector(wf.flagged, y))
    det_alone = np.nansum(detection_vector(alone, y))
    assert det_wf == 3 and det_alone == 3


def test_imperfect_reviewer():
    p, y, s = toy()
    r = refer(p, 0.5, s, 0.2, "all")
    out = summarise(r, y, reviewer_sensitivity=0.8)
    assert out["detected"] == pytest.approx(2 + 0.8)
    dv = detection_vector(r.flagged, y, r.referred, 0.8)
    assert np.nansum(dv) == pytest.approx(2.8)


def test_oracle_and_monotone_curve():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.random(n) < 0.2
    p = np.clip(0.5 * y + rng.normal(0.3, 0.2, n), 0, 1)
    s = np.clip(0.5 * y + rng.normal(0.3, 0.2, n), 0, 1)
    cur = capture_curve(p, 0.6, s, y, [0.01, 0.05, 0.1, 0.2, 0.3])
    caps = [c["captured"] for c in cur]
    assert caps == sorted(caps)
    assert all(c["captured"] <= c["oracle_ceiling"] for c in cur)
    # perfect secondary (= label) hits the ceiling
    perf = capture_curve(p, 0.6, y.astype(float), y, [0.05, 0.1])
    assert all(c["captured"] == c["oracle_ceiling"] for c in perf)


def test_random_floor_matches_simulation():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.random(n) < 0.25
    p = rng.random(n)
    caps = [summarise(refer(p, 0.7, rng.random(n), 0.1), y)["captured"] for _ in range(4000)]
    floor = summarise(refer(p, 0.7, rng.random(n), 0.1), y)["random_floor_expected"]
    assert np.mean(caps) == pytest.approx(floor, rel=0.03)
