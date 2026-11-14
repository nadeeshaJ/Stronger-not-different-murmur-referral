"""Study A (baseline reliability and error structure) and RQ5 (age strata) analyses.

Uses the same scoring mechanism as Study C: per rep (partition, seed, inner model j) every encounter has a
calibrated outer-test probability and a prediction at the development threshold of its (p, o, arch, seed).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import stats as st
from .analysis import ARCHS, build_context
from .calibration import calibration_slope_intercept

AGE_ORDER = ["Neonate", "Infant", "Child", "Adolescent"]


def assemble(runs, enc, n_part, n_outer, n_inner, seeds, sens_target, archs=ARCHS):
    """Returns reps list, prob[arch] (n_reps, N), pos[arch] (n_reps, N) bool, and contexts."""
    label_of = enc.set_index("patient_id").label.astype(int)
    order = enc.patient_id.astype(str).to_numpy()
    pos_of = {pid: i for i, pid in enumerate(order)}
    reps = [(p, s, j) for p in range(n_part) for s in seeds for j in range(n_inner)]
    R = {r: k for k, r in enumerate(reps)}
    prob = {a: np.full((len(reps), len(order)), np.nan) for a in archs}
    pos = {a: np.zeros((len(reps), len(order)), bool) for a in archs}
    ctx = {}
    for p in range(n_part):
        for o in range(n_outer):
            c = build_context(runs, p, o, n_inner, archs, seeds, label_of, sens_target)
            ctx[(p, o)] = c
            idx = np.array([pos_of[t] for t in c.test_ids])
            for a in archs:
                for s in seeds:
                    for j in range(n_inner):
                        k = R[(p, s, j)]
                        q = c.cal_test[(a, s, j)]
                        prob[a][k, idx] = q
                        pos[a][k, idx] = q >= c.thr[(a, s)]
    assert all(not np.isnan(v).any() for v in prob.values())
    return reps, prob, pos, ctx


def _ratio_ci(values, groups, idx_cache, n_boot=2000, seed=20261109):
    """Clustered bootstrap CI for the mean of per-encounter values (already rep-averaged)."""
    ids, inv = np.unique(groups, return_inverse=True)
    d = np.bincount(inv, weights=values)
    n = np.bincount(inv).astype(float)
    key = len(ids)
    if key not in idx_cache:
        idx_cache[key] = st.bootstrap_indices(len(ids), n_boot, seed)
    return st.cluster_bootstrap_ci(d, n, idx_cache[key])


def performance(reps, prob, pos, enc, n_boot=2000, seed=20261109, n_boot_auc=1000) -> pd.DataFrame:
    y = enc.label.to_numpy()
    ind = enc.individual_id.astype(str).to_numpy()
    cache = {}
    rows = []
    for a in prob:
        per = []
        for k in range(len(reps)):
            pk, qk = prob[a][k], pos[a][k]
            slope, inter = calibration_slope_intercept(pk, y)
            per.append({"auroc": roc_auc_score(y, pk), "prauc": average_precision_score(y, pk),
                        "sensitivity": qk[y == 1].mean(), "specificity": 1 - qk[y == 0].mean(),
                        "cal_slope": slope, "cal_intercept": inter, "brier": np.mean((pk - y) ** 2),
                        "fn": int((~qk & (y == 1)).sum()), "fp": int((qk & (y == 0)).sum())})
        per = pd.DataFrame(per)
        det = pos[a].mean(0)
        sl, sh = _ratio_ci(det[y == 1], ind[y == 1], cache, n_boot, seed)
        fl, fh = _ratio_ci(1 - det[y == 0], ind[y == 0], cache, n_boot, seed)
        # AUROC patient-sampling CI: resample individuals (multiplicity weights), recompute the SINGLE-MODEL
        # AUROC in every rep and average over reps. (AUROC of the rep-averaged score would be an ensemble AUROC.)
        ids, inv = np.unique(ind, return_inverse=True)
        rng = np.random.default_rng(seed)
        aucs = []
        for _ in range(n_boot_auc):
            w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[inv].astype(float)
            if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
                continue
            aucs.append(np.mean([roc_auc_score(y, prob[a][k], sample_weight=w) for k in range(len(reps))]))
        row = {"arch": a}
        for c in per.columns:
            row[f"{c}_mean"] = per[c].mean()
            row[f"{c}_sd_reps"] = per[c].std(ddof=1)
        row.update({"sens_ci95": (sl, sh), "spec_ci95": (fl, fh),
                    "auroc_ci95": tuple(np.quantile(aucs, [0.025, 0.975]))})
        rows.append(row)
    return pd.DataFrame(rows)


def reliability(prob, enc, n_bins=10) -> pd.DataFrame:
    y = enc.label.to_numpy()
    rows = []
    edges = np.linspace(0, 1, n_bins + 1)
    for a, P in prob.items():
        flat = P.ravel()
        yy = np.tile(y, P.shape[0])
        b = np.clip(np.digitize(flat, edges[1:-1]), 0, n_bins - 1)
        for i in range(n_bins):
            m = b == i
            if m.any():
                rows.append({"arch": a, "bin": i, "lo": edges[i], "hi": edges[i + 1], "mean_pred": flat[m].mean(),
                             "observed": yy[m].mean(), "n_per_rep": m.sum() / P.shape[0]})
    return pd.DataFrame(rows)


def error_structure(reps, pos, enc) -> pd.DataFrame:
    """Pairwise error overlap between architectures (same rep: same partition, seed, inner split)."""
    y = enc.label.to_numpy()
    rows = []
    for a, b in combinations(list(pos), 2):
        vals = []
        for k in range(len(reps)):
            ca, cb = pos[a][k] == (y == 1), pos[b][k] == (y == 1)       # correct indicators
            fa, fb = ~pos[a][k] & (y == 1), ~pos[b][k] & (y == 1)        # false negatives
            inter, union = (fa & fb).sum(), (fa | fb).sum()
            n11, n00 = np.sum(ca & cb), np.sum(~ca & ~cb)
            n10, n01 = np.sum(ca & ~cb), np.sum(~ca & cb)
            q = (n11 * n00 - n01 * n10) / max(n11 * n00 + n01 * n10, 1)
            ea, eb = (~ca).astype(float), (~cb).astype(float)
            corr = np.corrcoef(ea, eb)[0, 1] if ea.std() > 0 and eb.std() > 0 else np.nan
            vals.append({"fn_a": fa.sum(), "fn_b": fb.sum(), "fn_both": inter, "fn_jaccard": inter / union if union else np.nan,
                         "fn_nested": inter / min(fa.sum(), fb.sum()) if min(fa.sum(), fb.sum()) else np.nan,
                         "q_statistic": q, "error_correlation": corr})
        v = pd.DataFrame(vals)
        rows.append({"arch_a": a, "arch_b": b, **{f"{c}_mean": v[c].mean() for c in v}, **{f"{c}_sd_reps": v[c].std(ddof=1) for c in v}})
    return pd.DataFrame(rows)


def by_age(pos, enc, extra_arms: dict | None = None, n_boot=2000, seed=20261109) -> pd.DataFrame:
    """Sensitivity / specificity (and any detection matrices in extra_arms) per age group.
    Neonates are descriptive only (flagged). Clustered bootstrap intervals over individuals."""
    y = enc.label.to_numpy()
    ind = enc.individual_id.astype(str).to_numpy()
    age = enc.age_group.to_numpy()
    cache = {}
    rows = []
    arms = {f"model:{a}": m for a, m in pos.items()}
    for name, M in (extra_arms or {}).items():
        arms[name] = M
    for name, M in arms.items():
        is_det = not name.startswith("model:")
        with np.errstate(all="ignore"):
            det = np.nanmean(M, 0) if is_det else M.mean(0)
        for g in AGE_ORDER:
            mp = (age == g) & (y == 1)
            ma = (age == g) & (y == 0)
            r = {"arm": name, "age_group": g, "n_present": int(mp.sum()), "n_absent": int(ma.sum()),
                 "descriptive_only": g == "Neonate" or mp.sum() < 10}
            if mp.any():
                r["sensitivity"] = det[mp].mean()
                r["missed_mean"] = (1 - det[mp]).sum()
                if mp.sum() >= 2 and g != "Neonate":
                    r["sens_ci95"] = _ratio_ci(det[mp], ind[mp], cache, n_boot, seed)
            if ma.any() and not is_det:
                r["specificity"] = 1 - det[ma].mean()
                if ma.sum() >= 2 and g != "Neonate":
                    r["spec_ci95"] = _ratio_ci(1 - det[ma], ind[ma], cache, n_boot, seed)
            rows.append(r)
    return pd.DataFrame(rows)
