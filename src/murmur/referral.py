"""Selective referral at matched workload (Study C).

All functions work on ONE scored test set (one inner model's predictions on one outer-test fold).
Labels `y` are only ever passed in by the frozen analysis script.

Definitions (proposal Sections 4-5):
  predicted negatives  : primary calibrated probability < primary threshold
  budget K             : ceil(b * N_base) referrals, capped at the number of predicted negatives.
                         N_base is configurable: "all" = all test encounters (Study C text),
                         "pred_neg" = primary predicted negatives (registered choice for H1, decided 2026-11-09).
  extra-review workload: K                                   (H1)
  total flagged        : primary positives + referred         (H2)
  oracle ceiling       : min(K, eligible false negatives)
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


def budget_k(n_test: int, n_pred_neg: int, b: float, base: str = "all") -> int:
    if base == "all":
        k = ceil(b * n_test - 1e-12)
    elif base == "pred_neg":
        k = ceil(b * n_pred_neg - 1e-12)
    else:
        raise ValueError(base)
    return int(min(k, n_pred_neg))


def tie_break_keys(n: int, seed: int) -> np.ndarray:
    """Deterministic random tie-break keys (label-free)."""
    return np.random.default_rng(seed).random(n)


def top_k(score: np.ndarray, eligible: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Boolean mask of the k highest-scoring eligible items. Ties broken by seeded random keys."""
    score = np.asarray(score, float)
    eligible = np.asarray(eligible, bool)
    out = np.zeros(len(score), bool)
    if k <= 0:
        return out
    idx = np.flatnonzero(eligible)
    tb = tie_break_keys(len(score), seed)[idx]
    order = np.lexsort((-tb, -score[idx]))  # primary: score desc, secondary: tie-break desc
    out[idx[order[:k]]] = True
    return out


@dataclass
class ReferralResult:
    k: int
    n_test: int
    n_pred_pos: int
    n_pred_neg: int
    referred: np.ndarray          # bool mask
    flagged: np.ndarray           # bool mask: primary positive OR referred


def refer(p_primary, thr_primary, ref_score, b, base="all", seed=0) -> ReferralResult:
    p = np.asarray(p_primary, float)
    pos = p >= thr_primary
    neg = ~pos
    k = budget_k(len(p), int(neg.sum()), b, base)
    referred = top_k(ref_score, neg, k, seed)
    return ReferralResult(k, len(p), int(pos.sum()), int(neg.sum()), referred, pos | referred)


def standalone_top_m(score, m: int, seed: int = 0) -> np.ndarray:
    """H2 comparator: the standalone model flags its top-M encounters (label-free matched total workload)."""
    return top_k(score, np.ones(len(score), bool), int(m), seed)


def standalone_threshold(score, thr) -> np.ndarray:
    """H2 sensitivity analysis: standalone model at a development-matched threshold (realised M may differ)."""
    return np.asarray(score, float) >= thr


# ---------- outcome summaries (need labels) ----------

def summarise(res: ReferralResult, y, reviewer_sensitivity: float = 1.0) -> dict:
    """Per-test-set counts. Imperfect reviewer: each referred murmur is detected with prob r
    (expected counts reported, so results are deterministic)."""
    y = np.asarray(y, int)
    pos = res.flagged & ~res.referred
    fn_eligible = int((~pos & (y == 1)).sum())           # primary false negatives
    captured = int((res.referred & (y == 1)).sum())
    r = reviewer_sensitivity
    tp_primary = int((pos & (y == 1)).sum())
    detected = tp_primary + r * captured
    n_present = int(y.sum())
    flagged_neg = int((res.flagged & (y == 0)).sum())
    return {
        "k": res.k, "n_test": res.n_test, "n_present": n_present,
        "primary_pos": res.n_pred_pos, "primary_neg": res.n_pred_neg,
        "primary_tp": tp_primary, "eligible_fn": fn_eligible,
        "captured": captured, "captured_expected": r * captured,
        "oracle_ceiling": min(res.k, fn_eligible),
        "random_floor_expected": (res.k * fn_eligible / res.n_pred_neg) if res.n_pred_neg else 0.0,
        "referral_precision": captured / res.k if res.k else np.nan,
        "detected": detected, "missed": n_present - detected,
        "total_flags": int(res.flagged.sum()),
        "sensitivity": detected / n_present if n_present else np.nan,
        "specificity": 1 - flagged_neg / int((y == 0).sum()) if (y == 0).any() else np.nan,
    }


def detection_vector(flagged, y, referred=None, reviewer_sensitivity: float = 1.0) -> np.ndarray:
    """Per-encounter expected detection (0..1) for murmur-present encounters; NaN for absent ones.
    Primary positives count 1; referred murmurs count r."""
    y = np.asarray(y, int)
    d = np.asarray(flagged, float)
    if referred is not None and reviewer_sensitivity != 1.0:
        d = np.where(np.asarray(referred, bool), reviewer_sensitivity, d)
    return np.where(y == 1, d, np.nan)


def capture_curve(p_primary, thr_primary, ref_score, y, budgets, base="all", seed=0) -> list[dict]:
    out = []
    for b in budgets:
        s = summarise(refer(p_primary, thr_primary, ref_score, b, base, seed), y)
        s["budget"] = b
        out.append(s)
    return out
