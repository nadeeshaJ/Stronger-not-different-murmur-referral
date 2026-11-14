"""POST-HOC AUDIT (deviation 6): type I error of the confirmatory inference under the real design.

Two exchangeable rankers A and B (null true by construction) rank the primary's predicted negatives on the real
CirCor encounters, individuals, outer folds and 75-replicate structure, at a fixed budget K = ceil(0.10 x negatives).
Scores share a latent encounter signal and have PERSISTENT model-specific components (fixed across replicates),
so individual-level differences are consistent across replicates, which is the case the reviewer worried about.
For each simulated dataset: exact sign-flip p-value over individuals and the individual-clustered bootstrap CI.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np, pandas as pd  # noqa: E402
from murmur import stats as st  # noqa: E402
from murmur.referral import detection_vector, refer  # noqa: E402
from murmur.paths import derived_dir, load_config  # noqa: E402


def main(n_sim=1000, n_boot=2000):
    acfg, p1 = load_config("analysis"), load_config("phase1"); pc = p1["partitions"]
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    folds = pd.read_csv(derived_dir() / "folds.csv", dtype={"patient_id": str})
    order = enc.patient_id.to_numpy(); pos = {p: i for i, p in enumerate(order)}
    y = enc.label.to_numpy().astype(int); present = y == 1; N = len(y)
    ind = enc.individual_id.astype(str).to_numpy()
    # outer test folds per partition (role == 'test' rows of folds.csv)
    te = folds[folds.role == "test"]
    test_sets = {p: [np.array([pos[i] for i in te[(te.partition == p) & (te.outer_fold == o)].patient_id])
                     for o in range(pc["n_outer"])] for p in range(pc["n_partitions"])}
    reps = [(p, s, j) for p in range(pc["n_partitions"]) for s in range(3) for j in range(pc["n_inner"])]
    rng = np.random.default_rng(20261112)
    idx_b = st.bootstrap_indices(enc[present].individual_id.nunique(), n_boot, 20261109)
    out = []
    t = time.time()
    for scen, (mu, su, se) in {"moderate": (1.9, 0.5, 0.5), "persistent": (1.9, 1.0, 0.3)}.items():
        for k_sim in range(n_sim):
            z = rng.normal(mu * y, 1.0)
            uP, uA, uB = (rng.normal(0, su, N) for _ in range(3))
            A = np.full((len(reps), N), np.nan); B = A.copy()
            for k, (p, s, j) in enumerate(reps):
                e = rng.normal(0, se, (3, N))
                P_, A_, B_ = z + uP + e[0], z + uA + e[1], z + uB + e[2]
                for ix in test_sets[p]:
                    yy = y[ix]
                    thr = np.quantile(P_[ix][yy == 1], 0.10)          # ~90% sensitivity, label-based (irrelevant under the null)
                    ra = refer(P_[ix], thr, A_[ix], acfg["budget_primary"], acfg["budget_base"], seed=k)
                    rb = refer(P_[ix], thr, B_[ix], acfg["budget_primary"], acfg["budget_base"], seed=k)
                    A[k, ix] = detection_vector(ra.flagged, yy); B[k, ix] = detection_vector(rb.flagged, yy)
            ids, d, n = st.per_individual_diffs(A, B, ind, present)
            pval = st.sign_flip_test(d)
            lo, hi = st.cluster_bootstrap_ci(d, n, idx_b)
            out.append({"scenario": scen, "sim": k_sim, "estimate_murmurs": st.mean_diff(d, n) * present.sum(), "p": pval,
                        "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": (lo > 0) or (hi < 0)})
            if k_sim % 100 == 0:
                print(scen, k_sim, round(time.time() - t), "s", flush=True)
    df = pd.DataFrame(out)
    od = REPO / "results" / "post_hoc"; df.to_csv(od / "null_simulation.csv", index=False, lineterminator="\n")
    summ = df.groupby("scenario").agg(n=("p", "size"), type1_05=("p", lambda v: (v < 0.05).mean()),
                                      type1_01=("p", lambda v: (v < 0.01).mean()), type1_0125=("p", lambda v: (v < 0.0125).mean()),
                                      ci_noncoverage=("ci_excludes_0", "mean"), mean_est=("estimate_murmurs", "mean"),
                                      sd_est=("estimate_murmurs", "std"))
    summ.to_csv(od / "null_simulation_summary.csv", lineterminator="\n")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main(n_sim=int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
