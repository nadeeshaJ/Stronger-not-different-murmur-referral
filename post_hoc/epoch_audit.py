"""POST-HOC AUDIT (deviation 5): does the result depend on the globally fixed epoch (14), chosen on
partition 0 / outer fold 0 development data whose individuals reappear in other partitions' outer tests?

Nested re-selection: for every (partition, outer fold, architecture, seed) the epoch is chosen from the pooled
inner out-of-fold predictions of THAT outer development set only (outer test never used), by
  rule 'pilot'  : smallest epoch within 0.005 AUROC of the best (the registered pilot rule, applied per fold)
  rule 'argmax' : epoch with the best pooled inner-OOF AUROC
Then calibration, thresholds, secondary selection and Study C are re-run with the frozen functions.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np, pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from murmur import analysis as an  # noqa: E402
from murmur.analysis_h3 import load_runs_all_epochs  # noqa: E402
from murmur.hashing import sha256_file  # noqa: E402
from murmur.paths import derived_dir, load_config  # noqa: E402


def main():
    acfg, tcfg, p1 = load_config("analysis"), load_config("train"), load_config("phase1"); pc = p1["partitions"]
    seeds, archs, nI = tcfg["member_seeds"], an.ARCHS, pc["n_inner"]
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    label_of = enc.set_index("patient_id").label.astype(int)
    R = load_runs_all_epochs(REPO / "runs")
    E = next(iter(R.values()))["oof"].shape[0]
    choice = []
    for p in range(pc["n_partitions"]):
        for o in range(pc["n_outer"]):
            for a in archs:
                for s in seeds:
                    ids = np.concatenate([R[(p, o, j, a, s)]["oof_ids"] for j in range(nI)])
                    y = label_of[ids].to_numpy()
                    auc = [roc_auc_score(y, np.concatenate([R[(p, o, j, a, s)]["oof"][e] for j in range(nI)])) for e in range(E)]
                    best = max(auc)
                    choice.append({"partition": p, "outer": o, "arch": a, "seed": s, "auc_epoch14": auc[-1], "auc_best": best,
                                   "pilot": 1 + next(e for e in range(E) if auc[e] >= best - 0.005), "argmax": 1 + int(np.argmax(auc))})
    ch = pd.DataFrame(choice)
    out = REPO / "results" / "post_hoc"; out.mkdir(parents=True, exist_ok=True)
    ch.to_csv(out / "epoch_audit_choices.csv", index=False, lineterminator="\n")
    print(ch.groupby("arch")[["pilot", "argmax"]].describe().round(1).to_string())
    present = enc.label.to_numpy() == 1
    idx = __import__("murmur.stats", fromlist=["x"]).bootstrap_indices(enc[present].individual_id.nunique(), acfg["n_boot"], acfg["boot_seed"])
    rows, perf = [], []
    for rule in ("frozen14", "pilot", "argmax"):
        sel = ch.set_index(["partition", "outer", "arch", "seed"])
        runs = {}
        for (p, o, j, a, s), r in R.items():
            e = E - 1 if rule == "frozen14" else int(sel.loc[(p, o, a, s), rule]) - 1
            runs[(p, o, j, a, s)] = {"oof_ids": r["oof_ids"], "oof": r["oof"][e], "test_ids": r["test_ids"], "test": r["test"][e]}
        res = an.study_c(runs, enc, pc["n_partitions"], pc["n_outer"], nI, seeds, acfg, budgets=[acfg["budget_primary"]])
        conf = an.confirmatory(res, enc, acfg)
        conf["rule"] = rule
        rows.append(conf)
        # Study A: mean per-replicate single-model AUROC on the outer test
        for a in archs:
            v = []
            for p in range(pc["n_partitions"]):
                for s in seeds:
                    for j in range(nI):
                        ids = np.concatenate([runs[(p, o, j, a, s)]["test_ids"] for o in range(pc["n_outer"])])
                        sc = np.concatenate([runs[(p, o, j, a, s)]["test"] for o in range(pc["n_outer"])])
                        v.append(roc_auc_score(label_of[ids].to_numpy(), sc))
            perf.append({"rule": rule, "arch": a, "auroc_mean": float(np.mean(v))})
    conf = pd.concat(rows)
    conf.to_csv(out / "epoch_audit_confirmatory.csv", index=False, lineterminator="\n")
    pd.DataFrame(perf).to_csv(out / "epoch_audit_auroc.csv", index=False, lineterminator="\n")
    (out / "epoch_audit_manifest.json").write_text(json.dumps({"post_hoc": True, "deviation": 5, "outputs": {
        f.name: sha256_file(f) for f in sorted(out.glob("epoch_audit_*.csv"))}}, indent=2))
    conf["lo"], conf["hi"] = conf.ci95_lo * present.sum(), conf.ci95_hi * present.sum()
    print(conf[["rule", "hypothesis", "estimate_murmurs", "lo", "hi", "p_signflip", "p_holm"]].round(4).to_string(index=False))
    print(pd.DataFrame(perf).pivot(index="arch", columns="rule", values="auroc_mean").round(4))


if __name__ == "__main__":
    main()
