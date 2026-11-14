"""POST-HOC, EXPLORATORY (deviation 8): does a large pretrained audio model change the second-opinion
picture? Frozen calibration, thresholds, referral and comparison functions; b = 10%; estimates and
individual-clustered 95% CIs only (no tests). Plan: protocol/deviation8_fm_pilot.md."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np, pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from murmur import analysis as an, stats as st  # noqa: E402
from murmur.referral import detection_vector, refer, standalone_top_m  # noqa: E402
from murmur.hashing import sha256_file  # noqa: E402
from murmur.paths import derived_dir, load_config  # noqa: E402

FM = "ast"


def main(fm_root=None, out_dir=None):
    t0 = time.time()
    acfg, tcfg, pc = load_config("analysis"), load_config("train"), load_config("phase1")["partitions"]
    seeds, n_inner, n_part, n_outer = tcfg["member_seeds"], pc["n_inner"], pc["n_partitions"], pc["n_outer"]
    archs = an.ARCHS + [FM]
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    label_of = enc.set_index("patient_id").label.astype(int)
    order = enc.patient_id.astype(str).to_numpy(); pos_of = {p: i for i, p in enumerate(order)}
    runs = an.load_runs(REPO / "runs", acfg["epoch"])
    runs.update(an.load_runs(Path(fm_root or REPO / "runs_fm"), -1))
    missing = an.check_complete(runs, n_part, n_outer, n_inner, archs, seeds)
    assert not missing, f"{len(missing)} runs missing, e.g. {missing[:3]}"
    sel = pd.read_csv(REPO / "results" / "studyC_selection.csv")
    b = acfg["budget_primary"]
    sel = sel[np.isclose(sel.budget, b)].set_index(["partition", "outer", "seed", "primary"])
    reps = [(p, s, j) for p in range(n_part) for s in seeds for j in range(n_inner)]
    R = {r: k for k, r in enumerate(reps)}
    arms, rows, auc_rows, corr_rows = {}, [], [], []

    def mat(key):
        if key not in arms:
            arms[key] = np.full((len(reps), len(order)), np.nan)
        return arms[key]

    def record(pl, arm, k, idx, y, flagged, p, o, s, j):
        mat((pl, arm))[k, idx] = detection_vector(flagged, y)
        rows.append({"partition": p, "outer": o, "seed": s, "inner": j, "primary": pl, "arm": arm,
                     "flags": int(flagged.sum()), "detected": int((flagged & (y == 1)).sum()),
                     "fp": int((flagged & (y == 0)).sum()), "n_present": int(y.sum()), "n_absent": int((y == 0).sum())})

    primaries = {**acfg["primaries"], "fm": FM}
    for p in range(n_part):
        for o in range(n_outer):
            c = an.build_context(runs, p, o, n_inner, archs, seeds, label_of, acfg["sensitivity_target"])
            idx = np.array([pos_of[t] for t in c.test_ids]); y = label_of[c.test_ids].to_numpy()
            # A. standalone discrimination; B. development error correlation
            for s in seeds:
                for a in archs:
                    for j in range(n_inner):
                        auc_rows.append({"partition": p, "outer": o, "seed": s, "inner": j, "arch": a,
                                         "test_auroc": roc_auc_score(y, c.cal_test[(a, s, j)]),
                                         "dev_auroc": c.dev_auc[(a, s)]})
                ids = np.concatenate([c.oof_ids[j] for j in range(n_inner)])
                yd = label_of[ids].to_numpy()
                fn = {a: (np.concatenate([c.cal_oof[(a, s, j)] for j in range(n_inner)]) < c.thr[(a, s)])[yd == 1] for a in archs}
                for a in archs:
                    for a2 in archs:
                        if a < a2:
                            u, v = fn[a].astype(float), fn[a2].astype(float)
                            r = np.corrcoef(u, v)[0, 1] if u.std() > 0 and v.std() > 0 else np.nan
                            corr_rows.append({"partition": p, "outer": o, "seed": s, "a": a, "b": a2, "fn_corr": r,
                                              "fn_rate_a": u.mean(), "fn_rate_b": v.mean()})
            # C, D. referral workflows
            for pl, prim in primaries.items():
                for s in seeds:
                    thr = c.thr[(prim, s)]
                    for j in range(n_inner):
                        k = R[(p, s, j)]
                        pp = c.cal_test[(prim, s, j)]
                        crnn_ens = np.mean([c.cal_test[("crnn", s2, j)] for s2 in seeds], 0)
                        fm_ens = np.mean([c.cal_test[(FM, s2, j)] for s2 in seeds], 0)
                        own = refer(pp, thr, pp, b, acfg["budget_base"], seed=k)
                        record(pl, "own_threshold", k, idx, y, own.flagged, p, o, s, j)
                        if pl == "fm":
                            wf = refer(pp, thr, crnn_ens, b, acfg["budget_base"], seed=k)
                            record(pl, "crnn_secondary", k, idx, y, wf.flagged, p, o, s, j)
                        else:
                            chosen = sel.loc[(p, o, s, pl), "chosen"]
                            sc = an.candidate_scores(c, prim, s, j, "test", an.ARCHS, seeds)[chosen]
                            fz = refer(pp, thr, sc, b, acfg["budget_base"], seed=k)
                            record(pl, "frozen_secondary", k, idx, y, fz.flagged, p, o, s, j)
                            wf = refer(pp, thr, fm_ens, b, acfg["budget_base"], seed=k)
                            record(pl, "ast_secondary", k, idx, y, wf.flagged, p, o, s, j)
                        m = int(wf.flagged.sum())
                        record(pl, "crnn_ensemble_topM", k, idx, y, standalone_top_m(crnn_ens, m, seed=k), p, o, s, j)
                        record(pl, "ast_ensemble_topM", k, idx, y, standalone_top_m(fm_ens, m, seed=k), p, o, s, j)
        print(f"partition {p} done ({time.time() - t0:.0f}s)", flush=True)

    present = enc.label.to_numpy() == 1
    for key, v in arms.items():
        assert not np.isnan(v[:, present]).any(), key
    idx_b = st.bootstrap_indices(enc[present].individual_id.nunique(), acfg["n_boot"], acfg["boot_seed"])
    contrasts = [(pl, "ast_secondary", cmp) for pl in ("weak", "strong")
                 for cmp in ("own_threshold", "frozen_secondary", "crnn_ensemble_topM")]
    contrasts += [("fm", "crnn_secondary", cmp) for cmp in ("own_threshold", "crnn_ensemble_topM", "ast_ensemble_topM")]
    contrasts += [("fm", "ast_ensemble_topM", "crnn_ensemble_topM")]
    out = []
    for pl, a_arm, b_arm in contrasts:
        r = an.compare(arms, (pl, a_arm), (pl, b_arm), enc, idx_b, prior=acfg["beta_prior"], test=False)
        r.update({"primary": pl, "arm": a_arm, "comparator": b_arm}); out.append(r)
    out = pd.DataFrame(out)
    d = pd.DataFrame(rows)
    per = d.groupby(["primary", "arm", "partition", "seed", "inner"])[["flags", "detected", "fp", "n_present", "n_absent"]].sum()
    desc = per.groupby(["primary", "arm"]).mean().reset_index()
    desc["sensitivity"] = desc.detected / desc.n_present; desc["specificity"] = 1 - desc.fp / desc.n_absent
    desc["flag_share"] = desc["flags"] / (desc.n_present + desc.n_absent)
    auc = pd.DataFrame(auc_rows)
    auc_sum = auc.groupby("arch").agg(test_auroc_mean=("test_auroc", "mean"), test_auroc_sd=("test_auroc", "std"),
                                      dev_auroc_mean=("dev_auroc", "mean")).reset_index()
    piv = auc.pivot_table(index=["partition", "outer", "seed", "inner"], columns="arch", values="test_auroc")
    for a in an.ARCHS:
        dd = piv[FM] - piv[a]
        auc_sum.loc[len(auc_sum)] = {"arch": f"{FM}_minus_{a}", "test_auroc_mean": dd.mean(), "test_auroc_sd": dd.std(),
                                     "dev_auroc_mean": np.nan}
    corr = pd.DataFrame(corr_rows).groupby(["a", "b"])[["fn_corr", "fn_rate_a", "fn_rate_b"]].mean().reset_index()
    od = Path(out_dir or REPO / "results" / "post_hoc"); od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / "fm_pilot_contrasts.csv", index=False, lineterminator="\n")
    desc.to_csv(od / "fm_pilot_descriptive.csv", index=False, lineterminator="\n")
    auc_sum.to_csv(od / "fm_pilot_auroc.csv", index=False, lineterminator="\n")
    corr.to_csv(od / "fm_pilot_error_correlation.csv", index=False, lineterminator="\n")
    (od / "fm_pilot_manifest.json").write_text(json.dumps({"post_hoc": True, "deviation": 8, "budget": b,
        "plan_sha256": sha256_file(REPO / "protocol" / "deviation8_fm_pilot.md"),
        "outputs": {f.name: sha256_file(f) for f in sorted(od.glob("fm_pilot_*.csv"))}}, indent=2))
    pd.set_option("display.width", 200)
    print(auc_sum.round(4).to_string(index=False))
    print(corr.round(3).to_string(index=False))
    print(out[["primary", "arm", "comparator", "estimate_murmurs", "ci95_lo", "ci95_hi"]].assign(
        lo=lambda x: x.ci95_lo * 174, hi=lambda x: x.ci95_hi * 174).round(3).to_string(index=False))
    print(desc[["primary", "arm", "flags", "detected", "sensitivity", "specificity", "flag_share"]].round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
