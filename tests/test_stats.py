import itertools

import numpy as np
import pytest
from scipy import stats as sps

from murmur.calibration import Platt, sensitivity_threshold, sigmoid, logit
from murmur.stats import (beta_binomial_posterior, bootstrap_indices, clopper_pearson, cluster_bootstrap_ci,
                          holm, mcnemar_exact, mean_diff, midp_interval, per_individual_diffs, sign_flip_test,
                          tost_ci)


def brute_sign_flip(d):
    d = np.asarray(d, float)
    t = abs(d.sum())
    cnt = 0
    for s in itertools.product([-1, 1], repeat=len(d)):
        cnt += abs(np.dot(s, d)) >= t - 1e-12
    return cnt / 2 ** len(d)


@pytest.mark.parametrize("d", [[1, 1, 1, 1, 1], [1, -1, 1, 0.5, 0.25, 1], [0.04, 0.2, -0.12, 1, 1, 1, 0.6, -0.4]])
def test_sign_flip_matches_brute_force(d):
    assert sign_flip_test(d) == pytest.approx(brute_sign_flip(d))


def test_sign_flip_known_values():
    assert sign_flip_test([0, 0, 0]) == 1.0                   # all zero differences
    assert sign_flip_test([1] * 10) == pytest.approx(2 / 1024)
    assert sign_flip_test([1] * 10, alternative="greater") == pytest.approx(1 / 1024)
    # 4 vs 1 discordant pairs -> equals exact McNemar
    assert sign_flip_test([1, 1, 1, 1, -1]) == pytest.approx(mcnemar_exact(4, 1)) == pytest.approx(0.375)


def test_sign_flip_rep_averaged_lattice():
    rng = np.random.default_rng(0)
    d = rng.integers(-25, 26, 60) / 25          # 25 reps
    p1 = sign_flip_test(d, resolution=25)
    p2 = sign_flip_test(d)
    assert p1 == pytest.approx(p2)
    # Monte Carlo cross-check
    s = rng.choice([-1, 1], size=(200_000, len(d)))
    mc = np.mean(np.abs(s @ d) >= abs(d.sum()) - 1e-12)
    assert p1 == pytest.approx(mc, abs=0.005)


def test_sign_flip_type1_error():
    rng = np.random.default_rng(3)
    rej = 0
    for _ in range(400):
        d = rng.choice([-1, 0, 0, 0, 1], 80)
        rej += sign_flip_test(d) < 0.05
    assert rej / 400 < 0.075


def test_per_individual_and_ratio():
    # 2 reps, 4 encounters, encounters 0 and 1 share individual 'a'; encounter 3 absent
    da = np.array([[1, 1, 0, np.nan], [1, 0, 1, np.nan]])
    db = np.array([[0, 1, 0, np.nan], [0, 1, 0, np.nan]])
    ids, d, n = per_individual_diffs(da, db, ["a", "a", "b", "c"], [1, 1, 1, 0])
    assert list(ids) == ["a", "b"] and list(n) == [2, 1]
    assert d == pytest.approx([1 + (-0.5), 0.5])
    assert mean_diff(d, n) == pytest.approx(1.0 / 3)


def test_bootstrap_coverage():
    rng = np.random.default_rng(5)
    true = 0.05
    cover = 0
    idx = bootstrap_indices(150, n_boot=2000, seed=1)
    for _ in range(200):
        n = rng.choice([1, 1, 1, 2], 150)
        d = np.array([rng.binomial(k, 0.5 + true / 2) - rng.binomial(k, 0.5 - true / 2) for k in n], float) / 2
        # E[d_i] = n_i * true / 2 ... use the per-individual mean-diff scale
        lo, hi = cluster_bootstrap_ci(d, n, idx)
        cover += lo <= true / 2 <= hi
    assert 0.90 <= cover / 200 <= 0.99


def test_holm():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_tost():
    rng = np.random.default_rng(2)
    n = np.ones(400)
    d = rng.choice([-1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 400).astype(float)
    assert tost_ci(d, n, margin=0.05)["equivalent"]
    assert not tost_ci(d + (rng.random(400) < 0.2), n, margin=0.05)["equivalent"]


def test_intervals():
    lo, hi = clopper_pearson(3, 16)
    assert lo == pytest.approx(0.0405, abs=1e-3) and hi == pytest.approx(0.4565, abs=1e-3)
    mlo, mhi = midp_interval(3, 16)
    assert lo < mlo < 3 / 16 < mhi < hi                    # mid-p is narrower
    assert midp_interval(0, 10)[0] == 0.0 and midp_interval(10, 10)[1] == 1.0


def test_beta_binomial():
    r = beta_binomial_posterior(4, 1, prior=(1, 1))
    assert r["p_a_better"] == pytest.approx(sps.beta(5, 2).sf(0.5))
    assert beta_binomial_posterior(3, 3)["p_a_better"] == pytest.approx(0.5)


def test_platt_recovers_parameters():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.01, 0.99, 20000)
    y = rng.random(20000) < sigmoid(1.7 * logit(s) - 0.6)
    m = Platt().fit(s, y)
    assert m.a == pytest.approx(1.7, abs=0.06) and m.b == pytest.approx(-0.6, abs=0.06)


def test_sensitivity_threshold():
    rng = np.random.default_rng(1)
    y = rng.random(500) < 0.2
    p = np.clip(0.4 * y + rng.random(500) * 0.6, 0, 1)
    t = sensitivity_threshold(p, y, 0.90)
    sens = np.mean(p[y] >= t)
    assert sens >= 0.90
    # largest such threshold: any higher candidate drops below target
    higher = np.sort(p[y][p[y] > t])
    if len(higher):
        assert np.mean(p[y] >= higher[0]) < 0.90
