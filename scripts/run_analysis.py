"""Frozen analysis script: the ONLY place outer-test labels are joined.

    python scripts/run_analysis.py --runs runs --out results              # real analysis (requires freeze)
    python scripts/run_analysis.py --runs runs_sim --out results_sim --simulated   # test on simulated runs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from murmur import analysis as an  # noqa: E402
from murmur.hashing import sha256_file  # noqa: E402
from murmur.paths import derived_dir, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--simulated", action="store_true")
    ap.add_argument("--corrupt", default="runs_corrupt", help="Study B inputs (skipped if missing)")
    ap.add_argument("--studies", nargs="*", default=["C", "A", "H3", "B"])
    a = ap.parse_args()
    runs_dir, out = REPO / a.runs, REPO / a.out
    if not a.simulated:
        from murmur.train import freeze_status
        ok, why = freeze_status(REPO)
        if not ok:
            sys.exit(f"REFUSING to join outer-test labels: {why}")
    out.mkdir(parents=True, exist_ok=True)
    acfg, tcfg, p1 = load_config("analysis"), load_config("train"), load_config("phase1")
    pc = p1["partitions"]
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    t = time.time()
    runs = an.load_runs(runs_dir, acfg["epoch"])
    missing = an.check_complete(runs, pc["n_partitions"], pc["n_outer"], pc["n_inner"], an.ARCHS, tcfg["member_seeds"])
    if missing:
        sys.exit(f"{len(missing)} runs missing, e.g. {missing[:3]}")
    res = an.study_c(runs, enc, pc["n_partitions"], pc["n_outer"], pc["n_inner"], tcfg["member_seeds"], acfg)
    conf = an.confirmatory(res, enc, acfg)
    sec = an.secondary_analyses(res, enc, acfg)
    desc = an.descriptive_table(res)
    tr = res["transfer"]
    # pooled per (partition, seed, primary) over outer folds x inner models (~800 x 5 scores): a single test fold
    # has ~160 encounters, so its predicted-negative share has SD ~0.04 from sampling alone.
    tr = tr.groupby(["partition", "seed", "primary"])[["dev_neg_share", "test_neg_share"]].mean().reset_index()
    tr["gap"] = (tr.test_neg_share - tr.dev_neg_share).abs()
    tr["transfer_failure"] = tr.gap > acfg["transfer_tolerance"]
    for name, df in (("studyC_confirmatory", conf), ("studyC_secondary", sec), ("studyC_descriptive", desc),
                     ("studyC_per_rep", res["per_rep"]), ("studyC_selection", res["selection"]), ("transfer_check", tr)):
        df.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    tables = {}
    if "A" in a.studies:
        from murmur import analysis_a as aa
        reps, prob, pos, _ = aa.assemble(runs, enc, pc["n_partitions"], pc["n_outer"], pc["n_inner"], tcfg["member_seeds"], acfg["sensitivity_target"])
        tables["studyA_performance"] = aa.performance(reps, prob, pos, enc, n_boot=2000, seed=acfg["boot_seed"])
        tables["studyA_reliability"] = aa.reliability(prob, enc)
        tables["studyA_error_overlap"] = aa.error_structure(reps, pos, enc)
        b = acfg["budget_primary"]
        extra = {f"C:{pl}:{arm}": res["arms"][(pl, b, arm)] for pl in acfg["primaries"] for arm in ("secondary", "own_threshold", "standalone_topM")}
        tables["age_strata"] = aa.by_age(pos, enc, extra, n_boot=2000, seed=acfg["boot_seed"])
    if "H3" in a.studies:
        from murmur import analysis_h3 as h3
        runs_e = h3.load_runs_all_epochs(runs_dir)
        tables["H3_primary"], tables["H3_matching_log"] = h3.h3_primary(runs_e, enc, pc["n_partitions"], pc["n_outer"], pc["n_inner"], tcfg["member_seeds"], acfg, margin=acfg["h3_margin"])
        tables["H3_adjusted"], tables["H3_combinations"] = h3.h3_adjusted(runs_e, enc, pc["n_partitions"], pc["n_outer"], pc["n_inner"], tcfg["member_seeds"], acfg, seed=acfg["boot_seed"])
    cdir = REPO / a.corrupt
    if "B" in a.studies and cdir.exists():
        from murmur import analysis_b as ab
        conds = [c for c in sorted(p.stem for p in cdir.glob("*.npz"))]
        if "identity" in conds and not (cdir / "identity_check.txt").exists() and not a.simulated:
            sys.exit("identity check missing - Study B not analysed")
        conds.sort(key=lambda c: c != "identity")
        rb = ab.study_b(runs, cdir, conds, enc, pc["n_partitions"], pc["n_outer"], pc["n_inner"], tcfg["member_seeds"], acfg, res["selection"])
        for k, v in rb.items():
            tables[f"studyB_{k}"] = v
        tables["studyB_age_snr_gee"], _ = ab.age_snr_gee(runs, cdir, conds, enc, pc["n_partitions"], pc["n_outer"], pc["n_inner"], tcfg["member_seeds"], acfg)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    manifest = {"simulated": a.simulated, "runs_dir": str(runs_dir), "n_runs": len(runs), "elapsed_s": round(time.time() - t, 1),
                "outputs": {p.name: sha256_file(p) for p in sorted(out.glob("*.csv"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.set_option("display.width", 200)
    print(conf[["hypothesis", "estimate_murmurs", "ci95_lo", "ci95_hi", "p_signflip", "p_holm", "reject_holm", "bayes_p_A_better"]].round(4).to_string(index=False))
    print(f"transfer failures: {int(tr.transfer_failure.sum())}/{len(tr)} | {manifest['elapsed_s']}s")


if __name__ == "__main__":
    main()
