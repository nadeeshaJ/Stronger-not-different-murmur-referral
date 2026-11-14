"""Study B analysis: robustness under controlled corruption, referral under corruption, age x corruption.

Everything learnt on clean development data is frozen: calibrators, thresholds, the selected secondary ranker
and the H2 standalone model. Only the outer-test inputs are corrupted.
Every condition is labelled by what reached the model: median effective SNR at each architecture's input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as st
from .analysis import ARCHS, compare, study_c
from .analysis_a import AGE_ORDER, assemble, by_age, error_structure

SNR_CAP_DB = 40.0     # identity / clean coded as 40 dB in models (no finite effective SNR exists)


def load_condition(path: Path) -> dict:
    """{run_id: (test_ids, prob)} for one condition file."""
    from .corrupt_infer import unpack
    return unpack(np.load(path))


def with_corrupted_test(runs: dict, cond: dict) -> dict:
    """Copy of runs with outer-test probabilities replaced by the corrupted ones (OOF unchanged)."""
    out = {}
    for key, r in runs.items():
        p, o, j, a, s = key
        rid = f"p{p}_o{o}_i{j}_{a}_s{s}"
        ids, prob = cond[rid]
        assert (ids == r["test_ids"]).all(), rid
        out[key] = {**r, "test": prob}
    return out


def selection_dict(sel: pd.DataFrame) -> dict:
    return {(int(r.partition), int(r.outer), int(r.seed), r.primary, float(r.budget)): (r.chosen, r.standalone) for r in sel.itertuples()}


def snr_summary(corrupt_dir: Path, cond_name: str) -> dict:
    out = {"condition": cond_name}
    f = corrupt_dir / f"{cond_name}_recordings.csv"
    if f.exists():
        r = pd.read_csv(f).replace([np.inf], np.nan)
        if "effective_snr_db_filtered" in r:
            out["snr_filtered_median"] = r.effective_snr_db_filtered.median()
        for c in ("clip_fraction_processed", "dropout_fraction_processed", "burst_energy_ratio"):
            if c in r:
                out[f"{c}_median"] = r[c].median()
    f = corrupt_dir / f"{cond_name}_input_snr.csv"
    if f.exists():
        s = pd.read_csv(f).replace([np.inf], np.nan)
        for a, g in s.groupby("arch"):
            out[f"snr_input_median:{a}"] = g.effective_snr_db_input.median()
            out[f"snr_input_iqr:{a}"] = (g.effective_snr_db_input.quantile(0.25), g.effective_snr_db_input.quantile(0.75))
    return out


def study_b(runs, corrupt_dir: Path, conditions: list[str], enc, n_part, n_outer, n_inner, seeds, acfg, clean_selection: pd.DataFrame,
            n_boot=2000):
    y = enc.label.to_numpy()
    present = y == 1
    ind = enc.individual_id.astype(str).to_numpy()
    reps, prob0, pos0, _ = assemble(runs, enc, n_part, n_outer, n_inner, seeds, acfg["sensitivity_target"])
    sel = selection_dict(clean_selection[clean_selection.budget == acfg["budget_primary"]])
    b = acfg["budget_primary"]
    clean_c = study_c(runs, enc, n_part, n_outer, n_inner, seeds, acfg, fixed_selection=sel, budgets=[b])
    idx_p = st.bootstrap_indices(len(np.unique(ind[present])), n_boot, acfg["boot_seed"])
    idx_a = st.bootstrap_indices(len(np.unique(ind[~present])), n_boot, acfg["boot_seed"])
    rob, refl, ages, overlap = [], [], [], []
    for cn in conditions:
        cond = load_condition(corrupt_dir / f"{cn}.npz")
        rc = with_corrupted_test(runs, cond)
        _, prob1, pos1, _ = assemble(rc, enc, n_part, n_outer, n_inner, seeds, acfg["sensitivity_target"])
        snr = snr_summary(corrupt_dir, cn)
        for a in ARCHS:
            # paired change per encounter, rep-averaged; clustered CIs (present for sensitivity, absent for specificity)
            ds = (pos1[a].astype(float) - pos0[a].astype(float)).mean(0)
            _, d_p, n_p = st.per_individual_diffs(pos1[a][:, present].astype(float), pos0[a][:, present].astype(float), ind[present], np.ones(present.sum(), bool))
            _, d_a, n_a = st.per_individual_diffs(1 - pos1[a][:, ~present].astype(float), 1 - pos0[a][:, ~present].astype(float), ind[~present], np.ones((~present).sum(), bool))
            fn0, fn1 = ~pos0[a] & present, ~pos1[a] & present
            jac = np.nanmean([((fn0[k] & fn1[k]).sum() / max((fn0[k] | fn1[k]).sum(), 1)) for k in range(len(reps))])
            rob.append({"condition": cn, "arch": a, "snr_input_median": snr.get(f"snr_input_median:{a}", np.nan),
                        "snr_filtered_median": snr.get("snr_filtered_median", np.nan),
                        "sensitivity": pos1[a][:, present].mean(), "specificity": 1 - pos1[a][:, ~present].mean(),
                        "delta_sensitivity": ds[present].mean(), "delta_sens_ci95": st.cluster_bootstrap_ci(d_p, n_p, idx_p),
                        "delta_specificity": -ds[~present].mean(), "delta_spec_ci95": st.cluster_bootstrap_ci(d_a, n_a, idx_a),
                        "delta_fn": float((fn1.sum(1) - fn0.sum(1)).mean()), "delta_fp": float(((pos1[a] & ~present).sum(1) - (pos0[a] & ~present).sum(1)).mean()),
                        "prob_shift_present": float((prob1[a] - prob0[a])[:, present].mean()), "prob_shift_absent": float((prob1[a] - prob0[a])[:, ~present].mean()),
                        "fn_set_jaccard_vs_clean": jac})
        es = error_structure(reps, pos1, enc)
        es.insert(0, "condition", cn)
        overlap.append(es)
        # referral under corruption with frozen choices
        cc = study_c(rc, enc, n_part, n_outer, n_inner, seeds, acfg, fixed_selection=sel, budgets=[b])
        for plabel in acfg["primaries"]:
            pr1 = cc["per_rep"][(cc["per_rep"].primary == plabel) & (cc["per_rep"].arm == "secondary")]
            pr0 = clean_c["per_rep"][(clean_c["per_rep"].primary == plabel) & (clean_c["per_rep"].arm == "secondary")]
            agg = lambda d: d.groupby(["partition", "seed", "inner"])[["eligible_fn", "captured", "missed", "oracle_ceiling"]].sum().mean()
            a1, a0 = agg(pr1), agg(pr0)
            key = (plabel, b, "secondary")
            # which murmurs are missed by the primary changes -> eligible population; how well referral finds them -> capture rate
            r = {"condition": cn, "primary": plabel,
                 "eligible_fn_clean": a0.eligible_fn, "eligible_fn": a1.eligible_fn,
                 "capture_rate_clean": a0.captured / a0.eligible_fn if a0.eligible_fn else np.nan,
                 "capture_rate": a1.captured / a1.eligible_fn if a1.eligible_fn else np.nan,
                 "captured": a1.captured, "missed_total": a1.missed, "missed_total_clean": a0.missed,
                 "oracle_ceiling": a1.oracle_ceiling}
            h1 = compare(cc["arms"], key, (plabel, b, "own_threshold"), enc, n_boot=n_boot, boot_seed=acfg["boot_seed"], prior=acfg["beta_prior"], test=False)
            h2 = compare(cc["arms"], key, (plabel, b, "standalone_topM"), enc, n_boot=n_boot, boot_seed=acfg["boot_seed"], prior=acfg["beta_prior"], test=False)
            r.update({"H1_estimate_murmurs": h1["estimate_murmurs"], "H1_ci95": (h1["ci95_lo"] * h1["n_present"], h1["ci95_hi"] * h1["n_present"]),
                      "H2_estimate_murmurs": h2["estimate_murmurs"], "H2_ci95": (h2["ci95_lo"] * h2["n_present"], h2["ci95_hi"] * h2["n_present"])})
            refl.append(r)
        ag = by_age(pos1, enc, n_boot=n_boot)
        ag0 = by_age(pos0, enc, n_boot=n_boot).set_index(["arm", "age_group"])
        ag["delta_sensitivity"] = ag.sensitivity.to_numpy() - ag0.loc[list(zip(ag.arm, ag.age_group)), "sensitivity"].to_numpy()
        ag["delta_specificity"] = ag.specificity.to_numpy() - ag0.loc[list(zip(ag.arm, ag.age_group)), "specificity"].to_numpy()
        ag.insert(0, "condition", cn)
        ag["snr_input_median"] = [snr.get(f"snr_input_median:{x.split(':', 1)[1]}", np.nan) for x in ag.arm]
        ages.append(ag)
    return {"robustness": pd.DataFrame(rob), "referral": pd.DataFrame(refl), "age": pd.concat(ages, ignore_index=True),
            "overlap": pd.concat(overlap, ignore_index=True)}


def age_snr_gee(runs, corrupt_dir, conditions, enc, n_part, n_outer, n_inner, seeds, acfg, min_present=10):
    """Population-averaged logistic model (GEE, exchangeable within individual) of murmur detection
    (rep-averaged, fractional outcome) on age group (Infant, Adolescent vs Child), effective SNR at the input,
    and their interaction. Neonates and strata with < min_present murmur-present encounters are excluded.
    NOTE: the proposal names mixed-effects logistic models; GEE is the population-averaged analogue (decision for the protocol)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    y = enc.label.to_numpy()
    present = y == 1
    keep_ages = [g for g in ["Infant", "Child", "Adolescent"] if ((enc.age_group == g) & present).sum() >= min_present]
    m = present & enc.age_group.isin(keep_ages).to_numpy()
    _, _, pos0, _ = assemble(runs, enc, n_part, n_outer, n_inner, seeds, acfg["sensitivity_target"])
    rows = []
    for a in ARCHS:
        rows.append(pd.DataFrame({"arch": a, "condition": "clean", "snr": SNR_CAP_DB, "det": pos0[a][:, m].mean(0),
                                  "age": enc.age_group.to_numpy()[m], "individual": enc.individual_id.astype(str).to_numpy()[m]}))
    for cn in conditions:
        if cn == "identity":
            continue
        rc = with_corrupted_test(runs, load_condition(corrupt_dir / f"{cn}.npz"))
        _, _, pos1, _ = assemble(rc, enc, n_part, n_outer, n_inner, seeds, acfg["sensitivity_target"])
        snr = snr_summary(corrupt_dir, cn)
        for a in ARCHS:
            v = snr.get(f"snr_input_median:{a}", np.nan)
            rows.append(pd.DataFrame({"arch": a, "condition": cn, "snr": min(v, SNR_CAP_DB) if np.isfinite(v) else SNR_CAP_DB,
                                      "det": pos1[a][:, m].mean(0), "age": enc.age_group.to_numpy()[m],
                                      "individual": enc.individual_id.astype(str).to_numpy()[m]}))
    df = pd.concat(rows, ignore_index=True)
    df["age"] = pd.Categorical(df.age, categories=["Child"] + [g for g in keep_ages if g != "Child"])
    out = []
    for a, g in df.groupby("arch"):
        g = g.sort_values("individual")
        mod = smf.gee("det ~ C(age) * snr", groups="individual", data=g, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable())
        fit = mod.fit()
        ci = fit.conf_int()
        for term in fit.params.index:
            out.append({"arch": a, "term": term, "coef_logodds": fit.params[term], "ci95_lo": ci.loc[term, 0], "ci95_hi": ci.loc[term, 1],
                        "n_rows": len(g), "n_individuals": g.individual.nunique()})
    return pd.DataFrame(out), df
