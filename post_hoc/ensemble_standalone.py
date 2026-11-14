"""POST-HOC, EXPLORATORY (deviation 4): is the H2 gain a 'second opinion' or a seed-ensemble effect?

The development-selected secondary is often a 3-seed mean (e.g. hom_mean:crnn), while the registered H2
standalone is a single model. This script adds two label-free comparators at the SAME total flags M:
  (a) ensemble standalone: 3-seed mean of the strongest architecture, top-M
  (b) the selected secondary itself used standalone, top-M
Frozen Study C functions and frozen selections are reused (no reselection). Estimates + 95% CIs only.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np, pandas as pd  # noqa: E402
from murmur import analysis as an, stats as st  # noqa: E402
from murmur.referral import detection_vector, refer, standalone_top_m  # noqa: E402
from murmur.hashing import sha256_file  # noqa: E402
from murmur.paths import derived_dir, load_config  # noqa: E402


def main():
    acfg, tcfg, p1 = load_config("analysis"), load_config("train"), load_config("phase1"); pc = p1["partitions"]
    seeds, n_inner, archs = tcfg["member_seeds"], pc["n_inner"], an.ARCHS
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    label_of = enc.set_index("patient_id").label.astype(int)
    order = enc.patient_id.astype(str).to_numpy(); pos_of = {p: i for i, p in enumerate(order)}
    runs = an.load_runs(REPO / "runs", acfg["epoch"])
    sel = pd.read_csv(REPO / "results" / "studyC_selection.csv")
    b = acfg["budget_primary"]
    sel = sel[np.isclose(sel.budget, b)].set_index(["partition", "outer", "seed", "primary"])
    reps = [(p, s, j) for p in range(pc["n_partitions"]) for s in seeds for j in range(n_inner)]
    R = {r: k for k, r in enumerate(reps)}
    arms = {}
    def mat(key):
        if key not in arms:
            arms[key] = np.full((len(reps), len(order)), np.nan)
        return arms[key]
    rows = []
    for p in range(pc["n_partitions"]):
        for o in range(pc["n_outer"]):
            c = an.build_context(runs, p, o, n_inner, archs, seeds, label_of, acfg["sensitivity_target"])
            idx = np.array([pos_of[t] for t in c.test_ids]); y = label_of[c.test_ids].to_numpy()
            for pl, prim in acfg["primaries"].items():
                for s in seeds:
                    chosen, standalone = sel.loc[(p, o, s, pl), ["chosen", "standalone"]]
                    thr = c.thr[(prim, s)]
                    for j in range(n_inner):
                        k = R[(p, s, j)]
                        pp = c.cal_test[(prim, s, j)]
                        sc = an.candidate_scores(c, prim, s, j, "test", archs, seeds)[chosen]
                        wf = refer(pp, thr, sc, b, acfg["budget_base"], seed=k)
                        m = int(wf.flagged.sum())
                        ens = np.mean([c.cal_test[(standalone, s2, j)] for s2 in seeds], 0)
                        a_ens = standalone_top_m(ens, m, seed=k)
                        a_sec = standalone_top_m(sc, m, seed=k)
                        a_single = standalone_top_m(c.cal_test[(standalone, s, j)], m, seed=k)
                        mat((pl, "secondary"))[k, idx] = detection_vector(wf.flagged, y)
                        mat((pl, "standalone_single"))[k, idx] = detection_vector(a_single, y)
                        mat((pl, "standalone_ensemble"))[k, idx] = detection_vector(a_ens, y)
                        mat((pl, "secondary_alone"))[k, idx] = detection_vector(a_sec, y)
                        for name, f in (("secondary", wf.flagged), ("standalone_single", a_single),
                                        ("standalone_ensemble", a_ens), ("secondary_alone", a_sec)):
                            rows.append({"partition": p, "outer": o, "seed": s, "inner": j, "primary": pl, "arm": name,
                                         "flags": int(f.sum()), "detected": int((f & (y == 1)).sum()),
                                         "fp": int((f & (y == 0)).sum()), "n_present": int(y.sum()), "n_absent": int((y == 0).sum())})
    present = enc.label.to_numpy() == 1
    idx_b = st.bootstrap_indices(enc[present].individual_id.nunique(), acfg["n_boot"], acfg["boot_seed"])
    out = []
    for pl in acfg["primaries"]:
        for bb in ("standalone_single", "standalone_ensemble", "secondary_alone"):
            r = an.compare(arms, (pl, "secondary"), (pl, bb), enc, idx_b, prior=acfg["beta_prior"], test=False)
            r.update({"primary": pl, "comparator": bb}); out.append(r)
    out = pd.DataFrame(out)
    d = pd.DataFrame(rows)
    per = d.groupby(["primary", "arm", "partition", "seed", "inner"])[["flags", "detected", "fp", "n_present", "n_absent"]].sum()
    desc = per.groupby(["primary", "arm"]).mean().reset_index()
    desc["sensitivity"] = desc.detected / desc.n_present; desc["specificity"] = 1 - desc.fp / desc.n_absent
    od = REPO / "results" / "post_hoc"; od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / "h2_ensemble_contrasts.csv", index=False, lineterminator="\n")
    desc.to_csv(od / "h2_ensemble_descriptive.csv", index=False, lineterminator="\n")
    (od / "h2_ensemble_manifest.json").write_text(json.dumps({"post_hoc": True, "deviation": 4, "budget": b,
        "outputs": {f.name: sha256_file(f) for f in sorted(od.glob("h2_ensemble_*.csv"))}}, indent=2))
    print(out[["primary", "comparator", "estimate_murmurs", "ci95_lo", "ci95_hi", "detected_A", "detected_B"]].assign(
        lo=lambda x: x.ci95_lo * 174, hi=lambda x: x.ci95_hi * 174).round(3).to_string(index=False))
    print(desc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
