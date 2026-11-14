"""POST-HOC (deviation 7): H3 strength-matched comparison restricted to (partition, outer, seed) cells that matched
naturally (final checkpoints, no earlier-checkpoint adjustment). Same frozen functions; estimates + 95% CIs only."""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np, pandas as pd  # noqa: E402
from murmur.analysis import compare  # noqa: E402
from murmur.analysis_h3 import ARCHS, EpochContext, load_runs_all_epochs, match_pairs  # noqa: E402
from murmur.calibration import sensitivity_threshold  # noqa: E402
from murmur.referral import detection_vector, refer  # noqa: E402
from murmur.hashing import sha256_file  # noqa: E402
from murmur.paths import derived_dir, load_config  # noqa: E402


def main():
    acfg, tcfg, p1 = load_config("analysis"), load_config("train"), load_config("phase1"); pc = p1["partitions"]
    seeds, n_inner = tcfg["member_seeds"], pc["n_inner"]
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    runs = load_runs_all_epochs(REPO / "runs")
    label_of = enc.set_index("patient_id").label.astype(int)
    order = enc.patient_id.astype(str).to_numpy(); pos_of = {p: i for i, p in enumerate(order)}
    reps = [(p, s, j) for p in range(pc["n_partitions"]) for s in seeds for j in range(n_inner)]
    R = {r: k for k, r in enumerate(reps)}
    b, base, margin = acfg["budget_primary"], acfg["budget_base"], acfg["h3_margin"]
    arms, log = {}, []
    for p in range(pc["n_partitions"]):
        for o in range(pc["n_outer"]):
            c = EpochContext(runs, p, o, n_inner, label_of, ARCHS, seeds)
            E_last = len(next(iter(c.strength.values()))) - 1
            idx = np.array([pos_of[t] for t in c.test_ids]); y_test = label_of[c.test_ids].to_numpy()
            for plabel, P in acfg["primaries"].items():
                others = [a for a in ARCHS if a != P]
                for s in seeds:
                    thr = sensitivity_threshold(c.cal_oof_pooled(P, s, E_last), c.y_oof, acfg["sensitivity_target"])
                    het = [(others[0], s), (others[1], s)]
                    hom_arch = max(others, key=lambda a: c.strength[(a, s)][-1])
                    hom = [(hom_arch, s), (hom_arch, (s + 1) % len(seeds))]
                    ep, route, m_het, m_hom = match_pairs(c.strength, het, hom, margin)
                    log.append({"partition": p, "outer": o, "seed": s, "primary": plabel, "route": route, "strength_het": m_het, "strength_hom": m_hom})
                    if ep is None or route != "natural":
                        continue
                    for j in range(n_inner):
                        k = R[(p, s, j)]; pp = c.cal(P, s, E_last, j, "test")
                        for name, pair in (("het", het), ("hom", hom)):
                            sc = np.mean([c.cal(a, t, ep[(a, t)], j, "test") for a, t in pair], 0)
                            res = refer(pp, thr, sc, b, base, seed=k)
                            arms.setdefault((plabel, name), np.full((len(reps), len(order)), np.nan))[k, idx] = detection_vector(res.flagged, y_test)
    lg = pd.DataFrame(log); present = enc.label.to_numpy() == 1; rows = []
    for plabel in acfg["primaries"]:
        A, B = arms[(plabel, "het")], arms[(plabel, "hom")]
        defined = ~np.isnan(A).all(0) | ~present
        e2 = enc[defined].reset_index(drop=True)
        r = compare({(plabel, "het"): A[:, defined], (plabel, "hom"): B[:, defined]}, (plabel, "het"), (plabel, "hom"), e2,
                    n_boot=acfg["n_boot"], boot_seed=acfg["boot_seed"], prior=acfg["beta_prior"], test=False)
        g = lg[(lg.primary == plabel) & (lg.route == "natural")]
        r.update({"analysis": f"H3-{plabel}-natural", "n_cells": len(g), "n_present_covered": int((present & defined).sum()),
                  "mean_strength_het": g.strength_het.mean(), "mean_strength_hom": g.strength_hom.mean()})
        rows.append(r)
    df = pd.DataFrame(rows)
    od = REPO / "results" / "post_hoc"; df.to_csv(od / "h3_natural.csv", index=False, lineterminator="\n")
    (od / "h3_natural_manifest.json").write_text(json.dumps({"post_hoc": True, "deviation": 7, "sha256": sha256_file(od / "h3_natural.csv")}))
    df["lo"], df["hi"] = df.ci95_lo * df.n_present, df.ci95_hi * df.n_present
    print(df[["analysis", "n_cells", "n_present_covered", "estimate_murmurs", "lo", "hi", "mean_strength_het", "mean_strength_hom"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
