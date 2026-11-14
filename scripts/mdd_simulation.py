"""Minimum detectable difference (MDD) for the confirmatory paired tests, before the protocol freeze.

Model. 174 murmur-present encounters; R = 75 replicates (5 partitions x 3 seeds x 5 inner models).
In each replicate an encounter is concordant, "A-only" (found by arm A, missed by B) or "B-only".
Expected discordant murmurs per replicate D = a + b; effect Delta = a - b (murmurs per replicate).
Consistency c in (0, 1]: a fraction of encounters is discordance-prone and is discordant in a share c of
replicates (c = 1: the same murmurs are discordant in every replicate = single-split McNemar situation;
small c: discordance is mostly training noise that averaging over replicates removes).
Test: exact sign-flip on per-encounter replicate-averaged differences (the registered test).
MDD = smallest Delta with power >= 80%, at alpha 0.05 (unadjusted) and 0.0125 (Holm, most stringent step).

Archival anchors (proposal Section 1): strong primary at matched 405 flags, 5 discordant murmurs (4 vs 1).
Usage: python scripts/mdd_simulation.py [--sims 400]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from murmur.stats import sign_flip_test  # noqa: E402

N, R = 174, 75


def simulate_d(D, delta, c, rng):
    a, b = (D + delta) / 2, (D - delta) / 2
    # encounters prone to A-only / B-only discordance; each prone encounter is discordant in a share c of reps
    pa, pb = a / (N * c), b / (N * c)
    if pa + pb > 1:
        return None
    u = rng.random(N)
    prone_a = u < pa
    prone_b = (u >= pa) & (u < pa + pb)
    ka = rng.binomial(R, c, N) * prone_a
    kb = rng.binomial(R, c, N) * prone_b
    return (ka - kb) / R


def power(D, delta, c, alpha, sims, rng):
    hits, valid = 0, 0
    for _ in range(sims):
        d = simulate_d(D, delta, c, rng)
        if d is None:
            return np.nan
        valid += 1
        hits += sign_flip_test(d, resolution=R) < alpha
    return hits / valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--tost", action="store_true")
    a = ap.parse_args()
    if a.tost:
        return
    rng = np.random.default_rng(20261109)
    rows = []
    for D in (5, 10, 20, 30):
        for c in (1.0, 0.5, 0.25):
            for alpha in (0.05, 0.0125):
                mdd, curve = None, []
                for delta in np.arange(0.5, D + 0.01, 0.5):
                    pw = power(D, delta, c, alpha, a.sims, rng)
                    curve.append((delta, pw))
                    if pw >= 0.8:
                        mdd = delta
                        break
                rows.append({"discordant_per_rep_D": D, "consistency_c": c, "alpha": alpha,
                             "mdd_murmurs": mdd, "mdd_pp_sensitivity": None if mdd is None else 100 * mdd / N,
                             "mdd_as_A_vs_B": None if mdd is None else f"{(D + mdd) / 2:.1f} vs {(D - mdd) / 2:.1f}",
                             "power_at_proposal_4v1": power(5, 3, c, alpha, a.sims, rng) if D == 5 else None})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = REPO / "protocol"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "mdd.csv", index=False, lineterminator="\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()


def tost_feasibility(sims=200, n_boot=500):
    """Under no true difference: the equivalence margin the 90% clustered-bootstrap CI fits inside with 80% probability."""
    from murmur.stats import cluster_bootstrap_ci, bootstrap_indices
    rng = np.random.default_rng(7)
    idx = bootstrap_indices(N, n_boot, 1)
    rows = []
    for D in (5, 10, 20):
        for c in (1.0, 0.5, 0.25):
            w = []
            for _ in range(sims):
                d = simulate_d(D, 0.0, c, rng)
                lo, hi = cluster_bootstrap_ci(d, np.ones(N), idx, level=0.90)
                w.append(max(abs(lo), abs(hi)))
            m80 = float(np.quantile(w, 0.8))
            rows.append({"discordant_per_rep_D": D, "consistency_c": c, "margin_needed_pp_80pct": 100 * m80,
                         "margin_needed_murmurs": m80 * N, "p_equiv_at_1pp": float(np.mean(np.array(w) < 0.01))})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "protocol" / "tost_feasibility.csv", index=False, lineterminator="\n")
    return df


if __name__ == "__main__" and "--tost" in sys.argv:
    print(tost_feasibility().round(3).to_string(index=False))
