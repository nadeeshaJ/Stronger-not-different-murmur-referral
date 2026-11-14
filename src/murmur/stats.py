"""Paired, individual-clustered inference for the confirmatory tests (proposal Section 7).

Estimand: mean over the fixed set of murmur-present encounters (denominator = all of them, e.g. 174)
of [detection under arm A - detection under arm B], where each encounter's detection is first averaged
over partitions x inner models. Shared individuals count once: per-individual sums d_i are the units.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import stats as sps


# ---------------- estimand ----------------

def per_individual_diffs(det_a, det_b, individual, present):
    """det_a, det_b: arrays (n_reps, n_encounters) of per-encounter detection (0..1, NaN allowed for absent).
    Returns (individual ids, d_i = sum over that individual's present encounters of rep-mean(A-B), n_i)."""
    det_a, det_b = np.atleast_2d(det_a).astype(float), np.atleast_2d(det_b).astype(float)
    present = np.asarray(present, bool)
    diff = np.nanmean(det_a[:, present] - det_b[:, present], axis=0)
    ind = np.asarray(individual)[present]
    ids, inv = np.unique(ind, return_inverse=True)
    d = np.bincount(inv, weights=diff)
    n = np.bincount(inv)
    return ids, d, n


def mean_diff(d, n):
    return float(np.sum(d) / np.sum(n))


# ---------------- exact sign-flip randomisation ----------------

def sign_flip_test(d, resolution: int | None = None, alternative: str = "two-sided") -> float:
    """Exact sign-flip randomisation test of H0: individual differences symmetric about 0.

    Statistic T = sum(d_i). Exact null distribution by convolution on an integer lattice:
    d_i are rep-averages, so d_i * R is (near-)integer for R = n_reps; pass `resolution=R`.
    If resolution is None, the smallest R <= 10_000 making all d_i integral (to 1e-9) is found."""
    d = np.asarray(d, float)
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 1.0
    if resolution is None:
        resolution = _find_resolution(d)
    v = np.rint(np.abs(d) * resolution).astype(np.int64)
    if np.max(np.abs(np.abs(d) * resolution - v)) > 1e-6:
        raise ValueError("differences not on the lattice; pass the correct resolution")
    t_obs = int(np.rint(np.sum(d) * resolution))
    total = int(v.sum())
    # distribution of S = sum s_i v_i, s_i = +-1; array index = S + total
    dist = np.zeros(2 * total + 1)
    dist[total] = 1.0
    for vi in v:
        new = np.zeros_like(dist)
        new[vi:] += 0.5 * dist[: len(dist) - vi]     # s_i = +1
        new[: len(dist) - vi] += 0.5 * dist[vi:]     # s_i = -1
        dist = new
    support = np.arange(-total, total + 1)
    if alternative == "two-sided":
        p = dist[np.abs(support) >= abs(t_obs)].sum()
    elif alternative == "greater":
        p = dist[support >= t_obs].sum()
    elif alternative == "less":
        p = dist[support <= t_obs].sum()
    else:
        raise ValueError(alternative)
    return float(min(1.0, p))


def _find_resolution(d):
    for R in range(1, 10_001):
        if np.all(np.abs(d * R - np.rint(d * R)) < 1e-9):
            return R
    raise ValueError("could not find lattice resolution")


# ---------------- individual-clustered bootstrap ----------------

def bootstrap_indices(n_individuals: int, n_boot: int = 10_000, seed: int = 20261109) -> np.ndarray:
    """Draw once; reuse for every arm/partition so intervals are comparable."""
    return np.random.default_rng(seed).integers(0, n_individuals, size=(n_boot, n_individuals))


def cluster_bootstrap_ci(d, n, idx=None, level=0.95, seed=20261109):
    """Percentile CI of the ratio estimator sum(d*)/sum(n*) over resampled individuals."""
    d, n = np.asarray(d, float), np.asarray(n, float)
    if idx is None:
        idx = bootstrap_indices(len(d), seed=seed)
    est = d[idx].sum(1) / n[idx].sum(1)
    a = (1 - level) / 2
    return float(np.quantile(est, a)), float(np.quantile(est, 1 - a))


# ---------------- multiplicity ----------------

def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------- equivalence ----------------

def tost_ci(d, n, margin, idx=None, alpha=0.05):
    """Equivalence by the CI method (= two one-sided tests at alpha): equivalent if the
    (1 - 2*alpha) clustered-bootstrap CI lies inside (-margin, +margin)."""
    lo, hi = cluster_bootstrap_ci(d, n, idx, level=1 - 2 * alpha)
    return {"ci90": (lo, hi), "margin": margin, "equivalent": bool(lo > -margin and hi < margin)}


# ---------------- descriptive / small-count tools ----------------

def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar on discordant counts b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(sps.binomtest(min(b, c), n, 0.5).pvalue)


def clopper_pearson(k: int, n: int, level=0.95):
    if n == 0:
        return (np.nan, np.nan)
    r = sps.binomtest(k, n).proportion_ci(confidence_level=level, method="exact")
    return float(r.low), float(r.high)


def midp_interval(k: int, n: int, level=0.95):
    """Mid-p exact binomial interval (less conservative than Clopper-Pearson)."""
    from scipy.optimize import brentq
    a = (1 - level) / 2
    def upper_tail(p):  # P(X > k) + 0.5 P(X = k)
        return sps.binom.sf(k, n, p) + 0.5 * sps.binom.pmf(k, n, p)
    def lower_tail(p):  # P(X < k) + 0.5 P(X = k)
        return sps.binom.cdf(k - 1, n, p) + 0.5 * sps.binom.pmf(k, n, p)
    lo = 0.0 if k == 0 else brentq(lambda p: upper_tail(p) - a, 1e-12, 1 - 1e-12)
    hi = 1.0 if k == n else brentq(lambda p: lower_tail(p) - a, 1e-12, 1 - 1e-12)
    return float(lo), float(hi)


def beta_binomial_posterior(n_a_only: float, n_b_only: float, prior=(2.0, 2.0)):
    """Discordant murmurs: theta = P(a discordant murmur is found by A). Returns P(theta > 0.5) and 95% CrI."""
    a, b = prior[0] + n_a_only, prior[1] + n_b_only
    post = sps.beta(a, b)
    return {"p_a_better": float(post.sf(0.5)), "cri95": (float(post.ppf(0.025)), float(post.ppf(0.975))), "alpha": a, "beta": b}
