"""Study C analysis: selective referral at matched workload (H1, H2), from sealed run outputs.

Scoring mechanism (proposal Section 6, primary design): each outer-test encounter is scored by each of the
five inner models separately; calibrator and threshold come from the pooled inner OOF scores of the same
architecture/seed. A replicate ("rep") is (partition, member seed, inner model j); over the 5 outer folds
a rep scores every encounter exactly once, so each arm yields a (n_reps x n_encounters) detection matrix.

Secondary models are used only to RANK the primary's predicted negatives, and the H2 standalone uses
label-free top-M, so neither needs an ensemble threshold (no inner-inner models are required).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import stats as st
from .calibration import Platt, sensitivity_threshold
from .referral import budget_k, detection_vector, refer, standalone_top_m, summarise, top_k

ARCHS = ["wave_cnn", "spec_cnn", "crnn"]


# ---------------------------------------------------------------- loading

def load_runs(root: Path, epoch: int = -1) -> dict:
    """{(p, o, j, arch, seed): dict(oof_ids, oof, test_ids, test)} for single-model runs."""
    out = {}
    for d in sorted(Path(root).glob("p*_o*_i*_*_s*")):
        if ".tmp" in d.name or ".stale" in d.name or not (d / "manifest.json").exists():
            continue
        p, o, i, *arch, s = d.name.split("_")
        z = np.load(d / "preds.npz")
        out[(int(p[1:]), int(o[1:]), int(i[1:]), "_".join(arch), int(s[1:]))] = {
            "oof_ids": z["oof_ids"].astype(str), "oof": z["oof_prob"][epoch],
            "test_ids": z["test_ids"].astype(str), "test": z["test_prob"][epoch] if z["test_prob"].size else None,
        }
    return out


def check_complete(runs: dict, n_part, n_outer, n_inner, archs, seeds) -> list:
    return [(p, o, j, a, s) for p in range(n_part) for o in range(n_outer) for j in range(n_inner)
            for a in archs for s in seeds if (p, o, j, a, s) not in runs]


# ---------------------------------------------------------------- per (p, o) scoring context

@dataclass
class FoldContext:
    """Calibrated scores for one (partition, outer fold): per arch/seed, per inner model j."""
    oof_ids: dict = field(default_factory=dict)       # j -> ids
    test_ids: np.ndarray | None = None
    cal_oof: dict = field(default_factory=dict)       # (arch, seed, j) -> calibrated OOF probs (fold j)
    cal_test: dict = field(default_factory=dict)      # (arch, seed, j) -> calibrated test probs
    thr: dict = field(default_factory=dict)           # (arch, seed) -> 90%-sensitivity threshold
    dev_auc: dict = field(default_factory=dict)       # (arch, seed) -> pooled OOF AUROC
    dev_neg_share: dict = field(default_factory=dict)  # (arch, seed) -> pooled OOF predicted-negative share
    platt: dict = field(default_factory=dict)


def build_context(runs, p, o, n_inner, archs, seeds, label_of: pd.Series, sens_target) -> FoldContext:
    c = FoldContext()
    for a in archs:
        for s in seeds:
            ids = np.concatenate([runs[(p, o, j, a, s)]["oof_ids"] for j in range(n_inner)])
            raw = np.concatenate([runs[(p, o, j, a, s)]["oof"] for j in range(n_inner)])
            y = label_of[ids].to_numpy()
            pl = Platt().fit(raw, y)
            cal = pl.transform(raw)
            c.platt[(a, s)] = pl.to_dict()
            c.thr[(a, s)] = sensitivity_threshold(cal, y, sens_target)
            c.dev_auc[(a, s)] = roc_auc_score(y, raw)
            c.dev_neg_share[(a, s)] = float(np.mean(cal < c.thr[(a, s)]))
            for j in range(n_inner):
                r = runs[(p, o, j, a, s)]
                c.oof_ids[j] = r["oof_ids"]
                c.cal_oof[(a, s, j)] = pl.transform(r["oof"])
                if r["test"] is not None:
                    c.cal_test[(a, s, j)] = pl.transform(r["test"])
                    if c.test_ids is None:
                        c.test_ids = r["test_ids"]
                    assert (c.test_ids == r["test_ids"]).all()
    return c


# ---------------------------------------------------------------- candidate secondary rankers

def candidate_scores(c: FoldContext, primary: str, seed: int, j: int, where: str, archs, seeds) -> dict:
    """Ranking scores among the primary's negatives. where = 'oof' | 'test'."""
    src = c.cal_oof if where == "oof" else c.cal_test
    S = {}
    for a in archs:
        if a != primary:
            S[f"single:{a}"] = src[(a, seed, j)]
    for a in archs:
        S[f"hom_mean:{a}"] = np.mean([src[(a, s, j)] for s in seeds], 0)
    arch_means = np.stack([np.mean([src[(a, s, j)] for s in seeds], 0) for a in archs])
    S["het_mean"] = arch_means.mean(0)
    S["het_disagree"] = arch_means.std(0)
    return S


def select_secondary(c: FoldContext, primary, seed, n_inner, label_of, b, base, archs, seeds) -> tuple[str, dict]:
    """Development-only selection: inner-OOF capture of primary false negatives at budget b."""
    totals = {}
    for j in range(n_inner):
        y = label_of[c.oof_ids[j]].to_numpy()
        pp = c.cal_oof[(primary, seed, j)]
        for name, sc in candidate_scores(c, primary, seed, j, "oof", archs, seeds).items():
            totals[name] = totals.get(name, 0) + summarise(refer(pp, c.thr[(primary, seed)], sc, b, base), y)["captured"]
    best = max(totals.items(), key=lambda kv: (kv[1], -list(totals).index(kv[0])))[0]   # ties -> earlier (simpler) candidate
    return best, totals


def strongest_standalone(c: FoldContext, seed, archs) -> str:
    return max(archs, key=lambda a: c.dev_auc[(a, seed)])


# ---------------------------------------------------------------- Study C

def study_c(runs, enc: pd.DataFrame, n_part, n_outer, n_inner, seeds, acfg, fixed_selection: dict | None = None,
            budgets: list | None = None) -> dict:
    """fixed_selection: {(p, o, seed, primary_label, budget): (chosen, standalone)} -- re-apply frozen choices (Study B)."""
    archs = ARCHS
    label_of = enc.set_index("patient_id").label.astype(int)
    order = enc.patient_id.astype(str).to_numpy()
    pos_of = {pid: i for i, pid in enumerate(order)}
    N = len(order)
    reps = [(p, s, j) for p in range(n_part) for s in seeds for j in range(n_inner)]
    R = {rep: k for k, rep in enumerate(reps)}
    budgets = budgets or [acfg["budget_primary"], *acfg["budgets_secondary"]]
    arms = {}       # (primary_label, budget, arm) -> (n_reps, N) detection
    rows, sel_rows, transfer_rows = [], [], []

    def mat(key):
        if key not in arms:
            arms[key] = np.full((len(reps), N), np.nan)
        return arms[key]

    for p in range(n_part):
        for o in range(n_outer):
            c = build_context(runs, p, o, n_inner, archs, seeds, label_of, acfg["sensitivity_target"])
            idx = np.array([pos_of[t] for t in c.test_ids])
            y_test = label_of[c.test_ids].to_numpy()
            for plabel, primary in acfg["primaries"].items():
                for s in seeds:
                    thr = c.thr[(primary, s)]
                    standalone = strongest_standalone(c, s, archs)
                    for b in budgets:
                        if fixed_selection is not None:
                            chosen, standalone = fixed_selection[(p, o, s, plabel, b)]
                            totals = {}
                        else:
                            chosen, totals = select_secondary(c, primary, s, n_inner, label_of, b, acfg["budget_base"], archs, seeds)
                        sel_rows.append({"partition": p, "outer": o, "seed": s, "primary": plabel, "budget": b,
                                         "chosen": chosen, "standalone": standalone, **{f"dev_cap:{k}": v for k, v in totals.items()}})
                        for j in range(n_inner):
                            k = R[(p, s, j)]
                            pp = c.cal_test[(primary, s, j)]
                            # label-free transfer check (before labels are used)
                            if b == acfg["budget_primary"]:
                                transfer_rows.append({"partition": p, "outer": o, "seed": s, "inner": j, "primary": plabel,
                                                      "dev_neg_share": c.dev_neg_share[(primary, s)],
                                                      "test_neg_share": float(np.mean(pp < thr))})
                            sc = candidate_scores(c, primary, s, j, "test", archs, seeds)[chosen]
                            wf = refer(pp, thr, sc, b, acfg["budget_base"], seed=k)
                            own = refer(pp, thr, pp, b, acfg["budget_base"], seed=k)
                            m = int(wf.flagged.sum())
                            alone = standalone_top_m(c.cal_test[(standalone, s, j)], m, seed=k)
                            mat((plabel, b, "secondary"))[k, idx] = detection_vector(wf.flagged, y_test)
                            mat((plabel, b, "own_threshold"))[k, idx] = detection_vector(own.flagged, y_test)
                            mat((plabel, b, "standalone_topM"))[k, idx] = detection_vector(alone, y_test)
                            for r in acfg["reviewer_sensitivities"]:
                                mat((plabel, b, f"secondary_r{r}"))[k, idx] = detection_vector(wf.flagged, y_test, wf.referred, r)
                                mat((plabel, b, f"own_threshold_r{r}"))[k, idx] = detection_vector(own.flagged, y_test, own.referred, r)
                            rnd = refer(pp, thr, np.random.default_rng(10_000 + k).random(len(pp)), b, acfg["budget_base"], seed=k)
                            for arm, res in (("secondary", wf), ("own_threshold", own), ("random", rnd)):
                                rows.append({"partition": p, "outer": o, "seed": s, "inner": j, "primary": plabel, "budget": b,
                                             "arm": arm, "chosen": chosen, **summarise(res, y_test)})
                            rows.append({"partition": p, "outer": o, "seed": s, "inner": j, "primary": plabel, "budget": b,
                                         "arm": "standalone_topM", "chosen": standalone, "total_flags": int(alone.sum()),
                                         "detected": int((alone & (y_test == 1)).sum()), "n_present": int(y_test.sum()),
                                         "missed": int(((~alone) & (y_test == 1)).sum()),
                                         "sensitivity": float((alone & (y_test == 1)).sum() / max(y_test.sum(), 1)),
                                         "specificity": float(1 - (alone & (y_test == 0)).sum() / max((y_test == 0).sum(), 1))})
    present = label_of[order].to_numpy() == 1
    for key, v in arms.items():
        assert not np.isnan(v[:, present]).any(), f"arm {key}: some rep did not score every murmur-present encounter"
    return {"arms": arms, "reps": reps, "order": order, "per_rep": pd.DataFrame(rows),
            "selection": pd.DataFrame(sel_rows), "transfer": pd.DataFrame(transfer_rows)}


# ---------------------------------------------------------------- inference

def compare(arms, a_key, b_key, enc, idx_boot=None, n_boot=10000, boot_seed=20261109, margin=None, prior=(2, 2), test=True):
    """Paired per-individual comparison A - B over all murmur-present encounters."""
    present = enc.label.to_numpy() == 1
    ids, d, n = st.per_individual_diffs(arms[a_key], arms[b_key], enc.individual_id.astype(str).to_numpy(), present)
    if idx_boot is None:
        idx_boot = st.bootstrap_indices(len(d), n_boot, boot_seed)
    est = st.mean_diff(d, n)
    lo, hi = st.cluster_bootstrap_ci(d, n, idx_boot)
    p = st.sign_flip_test(d) if test else np.nan
    A, B = arms[a_key][:, present], arms[b_key][:, present]
    a_only = float(np.mean(np.nansum(np.clip(A - B, 0, None), 1)))
    b_only = float(np.mean(np.nansum(np.clip(B - A, 0, None), 1)))
    out = {"A": "|".join(map(str, a_key)), "B": "|".join(map(str, b_key)), "n_present": int(present.sum()),
           "n_individuals": len(d), "estimate": est, "estimate_murmurs": est * present.sum(), "ci95_lo": lo, "ci95_hi": hi,
           "p_signflip": p, "mean_A_only": a_only, "mean_B_only": b_only,
           "detected_A": float(np.nanmean(A.sum(1))), "detected_B": float(np.nanmean(B.sum(1)))}
    bb = st.beta_binomial_posterior(a_only, b_only, prior)
    out["bayes_p_A_better"] = bb["p_a_better"]
    if margin is not None:
        t = st.tost_ci(d, n, margin, idx_boot)
        out.update({"tost_ci90_lo": t["ci90"][0], "tost_ci90_hi": t["ci90"][1], "tost_margin": margin, "equivalent": t["equivalent"]})
    return out


def partition_spread(arms, a_key, b_key, enc, reps):
    """Training/partition variability: estimate per partition (averaging seeds x inner models only)."""
    present = enc.label.to_numpy() == 1
    parts = sorted({r[0] for r in reps})
    vals = []
    for p in parts:
        k = [i for i, r in enumerate(reps) if r[0] == p]
        vals.append(float(np.mean(np.nanmean(arms[a_key][k][:, present] - arms[b_key][k][:, present], 0))))
    return {"per_partition": vals, "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "min": min(vals), "max": max(vals)}


def confirmatory(res, enc, acfg) -> pd.DataFrame:
    arms, b = res["arms"], acfg["budget_primary"]
    present = enc.label.to_numpy() == 1
    idx = st.bootstrap_indices(enc[present].individual_id.nunique(), acfg["n_boot"], acfg["boot_seed"])
    rows = []
    for plabel in acfg["primaries"]:
        h1 = compare(arms, (plabel, b, "secondary"), (plabel, b, "own_threshold"), enc, idx, prior=acfg["beta_prior"])
        h1["hypothesis"] = f"H1-{plabel}"
        h1.update({f"spread_{k}": v for k, v in partition_spread(arms, (plabel, b, "secondary"), (plabel, b, "own_threshold"), enc, res["reps"]).items() if k != "per_partition"})
        rows.append(h1)
        h2 = compare(arms, (plabel, b, "secondary"), (plabel, b, "standalone_topM"), enc, idx, prior=acfg["beta_prior"])
        h2["hypothesis"] = f"H2-{plabel}"
        h2.update({f"spread_{k}": v for k, v in partition_spread(arms, (plabel, b, "secondary"), (plabel, b, "standalone_topM"), enc, res["reps"]).items() if k != "per_partition"})
        rows.append(h2)
    df = pd.DataFrame(rows)
    df["p_holm"] = st.holm(df.p_signflip.to_numpy())
    df["reject_holm"] = df.p_holm < acfg["alpha"]
    return df[["hypothesis"] + [c for c in df.columns if c != "hypothesis"]]


def secondary_analyses(res, enc, acfg) -> pd.DataFrame:
    """Secondary budgets and imperfect-reviewer sensitivity (unadjusted, labelled as such)."""
    arms = res["arms"]
    present = enc.label.to_numpy() == 1
    idx = st.bootstrap_indices(enc[present].individual_id.nunique(), acfg["n_boot"], acfg["boot_seed"])
    rows = []
    for plabel in acfg["primaries"]:
        for b in acfg["budgets_secondary"]:
            for name, a, bb in ((f"H1-{plabel}", "secondary", "own_threshold"), (f"H2-{plabel}", "secondary", "standalone_topM")):
                r = compare(arms, (plabel, b, a), (plabel, b, bb), enc, idx, prior=acfg["beta_prior"])
                r.update({"analysis": f"{name} budget {b}", "adjusted": False})
                rows.append(r)
        b = acfg["budget_primary"]
        for rv in acfg["reviewer_sensitivities"]:
            r = compare(arms, (plabel, b, f"secondary_r{rv}"), (plabel, b, f"own_threshold_r{rv}"), enc, idx, prior=acfg["beta_prior"])
            r.update({"analysis": f"H1-{plabel} reviewer {rv}", "adjusted": False})
            rows.append(r)
    return pd.DataFrame(rows)


def descriptive_table(res) -> pd.DataFrame:
    """Mean counts per (primary, budget, arm) over reps: denominators, eligible FN, oracle ceiling, flags, etc."""
    df = res["per_rep"]
    num = [c for c in ["k", "n_present", "primary_pos", "eligible_fn", "captured", "oracle_ceiling", "random_floor_expected",
                       "referral_precision", "detected", "missed", "total_flags", "sensitivity", "specificity"] if c in df]
    # counts: sum over outer folds within a rep; rates: mean over outer folds; then average over reps
    keys = ["primary", "budget", "arm", "partition", "seed", "inner"]
    rates = [c for c in ("referral_precision", "sensitivity", "specificity") if c in num]
    counts = [c for c in num if c not in rates]
    per_rep = df.groupby(keys)[counts].sum(min_count=1).join(df.groupby(keys)[rates].mean())
    per_rep["sensitivity_pooled"] = per_rep["detected"] / per_rep["n_present"]
    return per_rep.groupby(["primary", "budget", "arm"]).mean().reset_index()
