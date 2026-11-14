"""Individual-grouped, murmur-stratified nested partitions.

Output table (one row per partition x outer_fold x encounter):
  partition, outer_fold, patient_id, individual_id, label, age_group, role ('test'|'dev'), inner_fold (-1 for test)

Inner-inner sets for the ensemble-scoring arm are derived, not stored: for outer fold o, the model that
excludes inner folds {j,k} is trained on dev rows with inner_fold not in {j,k}.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def inner_seed(partition_seed: int, outer_fold: int) -> int:
    """Deterministic 32-bit seed for the inner split of one outer fold."""
    return int(np.random.SeedSequence([partition_seed, outer_fold]).generate_state(1)[0])


def _sgkf_assign(y, groups, n_splits, seed, strata=None) -> np.ndarray:
    """Grouped, stratified fold assignment.

    sklearn's StratifiedGroupKFold(shuffle=True) balanced prevalence poorly on this cohort (test-fold
    prevalence up to 0.31 vs 0.22 overall). Instead we stratify at the *individual* level on the
    individual's label pattern (n_encounters, n_present) with StratifiedKFold, then map back to
    encounters. Individuals are never split, and encounter prevalence stays near-exact.
    If `strata` (per-encounter labels, e.g. age group) is given, the individual's stratum (taken from
    its first encounter in sorted order) is added to the pattern, balancing that variable as well.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    n_enc = np.bincount(inv)
    n_pos = np.bincount(inv, weights=y).astype(int)
    pattern = np.array([f"{a}_{b}" for a, b in zip(n_enc, n_pos)], dtype=object)
    if strata is not None:
        strata = np.asarray(strata)
        first = np.full(len(ug), -1)
        for i in range(len(inv) - 1, -1, -1):
            first[inv[i]] = i
        pattern = pattern + "_" + strata[first].astype(str)
    pattern = pattern.astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gfold = np.full(len(ug), -1, dtype=int)
    with warnings.catch_warnings():
        # rare patterns (e.g. 2 encounters, 1 present: 4 individuals) have < n_splits members
        warnings.filterwarnings("ignore", message="The least populated class")
        for k, (_, te) in enumerate(skf.split(np.zeros(len(ug)), pattern)):
            gfold[te] = k
    assert (gfold >= 0).all()
    return gfold[inv]


def make_partitions(enc: pd.DataFrame, seeds, n_outer=5, n_inner=5, extra_strata: str | None = "age_group") -> pd.DataFrame:
    """Stratified on murmur pattern per individual, plus `extra_strata` (default age group)."""
    rows = []
    st = enc[extra_strata].to_numpy() if extra_strata else None
    y_all = enc["label"].to_numpy()
    g_all = enc["individual_id"].to_numpy()
    for p, seed in enumerate(seeds):
        outer = _sgkf_assign(y_all, g_all, n_outer, seed, st)
        for o in range(n_outer):
            te = outer == o
            dev_idx = np.flatnonzero(~te)
            # inner seed derived deterministically from partition seed and outer fold
            inner = _sgkf_assign(y_all[dev_idx], g_all[dev_idx], n_inner, inner_seed(seed, o),
                                 None if st is None else st[dev_idx])
            inner_full = np.full(len(enc), -1, dtype=int)
            inner_full[dev_idx] = inner
            part = enc[["patient_id", "individual_id", "label", "age_group"]].copy()
            part.insert(0, "outer_fold", o)
            part.insert(0, "partition", p)
            part["role"] = np.where(te, "test", "dev")
            part["inner_fold"] = inner_full
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def balance_report(folds: pd.DataFrame) -> pd.DataFrame:
    """Per partition x outer fold x split: n, n_present, and age-group counts."""
    test = folds[folds.role == "test"].assign(split="test")
    inner = folds[folds.role == "dev"].assign(split=lambda d: "inner" + d.inner_fold.astype(str))
    both = pd.concat([test, inner])
    agg = both.groupby(["partition", "outer_fold", "split"]).agg(
        n=("label", "size"), present=("label", "sum"), individuals=("individual_id", "nunique")
    )
    ages = both.pivot_table(index=["partition", "outer_fold", "split"], columns="age_group", values="label", aggfunc="size", fill_value=0)
    out = agg.join(ages).reset_index()
    out["prev"] = out["present"] / out["n"]
    return out
