"""POST-HOC (deviation 3): registered budget curve (protocol section 6: 'curve over 1-30%'), which the frozen
run_analysis.py did not write. Uses the frozen Study C functions unchanged; descriptive estimates + 95% CIs only
(no new tests). Outputs to results/post_hoc/ so the frozen results/ manifest is untouched.

    python post_hoc/budget_curve.py --runs runs --out results/post_hoc
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np, pandas as pd  # noqa: E402
from murmur import analysis as an, stats as st  # noqa: E402
from murmur.hashing import sha256_file  # noqa: E402
from murmur.paths import derived_dir, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="results/post_hoc")
    ap.add_argument("--budgets", nargs="*", type=float, default=None); a = ap.parse_args()
    acfg, tcfg, p1 = load_config("analysis"), load_config("train"), load_config("phase1"); pc = p1["partitions"]
    budgets = a.budgets or acfg["budget_curve"]
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    out = REPO / a.out; out.mkdir(parents=True, exist_ok=True); t = time.time()
    runs = an.load_runs(REPO / a.runs, acfg["epoch"])
    assert not an.check_complete(runs, pc["n_partitions"], pc["n_outer"], pc["n_inner"], an.ARCHS, tcfg["member_seeds"])
    res = an.study_c(runs, enc, pc["n_partitions"], pc["n_outer"], pc["n_inner"], tcfg["member_seeds"], acfg, budgets=budgets)
    present = enc.label.to_numpy() == 1
    idx = st.bootstrap_indices(enc[present].individual_id.nunique(), acfg["n_boot"], acfg["boot_seed"])
    rows = []
    for pl in acfg["primaries"]:
        for b in budgets:
            for name, bb in (("H1", "own_threshold"), ("H2", "standalone_topM"), ("vs_random", "random")):
                if (pl, b, bb) not in res["arms"]:
                    continue
                r = an.compare(res["arms"], (pl, b, "secondary"), (pl, b, bb), enc, idx, prior=acfg["beta_prior"], test=False)
                r.update({"primary": pl, "budget": b, "contrast": name}); rows.append(r)
    cmp_ = pd.DataFrame(rows)
    desc = an.descriptive_table(res)
    cmp_.to_csv(out / "budget_curve_contrasts.csv", index=False, lineterminator="\n")
    desc.to_csv(out / "budget_curve_descriptive.csv", index=False, lineterminator="\n")
    res["selection"].to_csv(out / "budget_curve_selection.csv", index=False, lineterminator="\n")
    man = {"post_hoc": True, "deviation": 3, "budgets": budgets, "elapsed_s": round(time.time() - t, 1),
           "outputs": {p.name: sha256_file(p) for p in sorted(out.glob("budget_curve_*.csv"))}}
    (out / "budget_curve_manifest.json").write_text(json.dumps(man, indent=2))
    print(cmp_[["primary", "budget", "contrast", "estimate_murmurs", "ci95_lo", "ci95_hi"]].round(3).to_string(index=False), man["elapsed_s"])


if __name__ == "__main__":
    main()
