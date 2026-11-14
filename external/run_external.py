"""External transportability check (protocol addendum E1, registered 2026-11-12 11:21 UTC before data download).

Two stages, so labels never touch scoring:
  1) score   (GPU, laptop): every frozen model scores every external unit. No labels are read.
       python external/run_external.py score --manifest external/<name>_manifest.csv --name <name>
     manifest columns: unit_id, recording_id, path   (one row per recording; unit = participant or recording)
     -> results/external/<name>_scores.npz  (run_ids x units, raw probabilities)
  2) analyse (CPU): frozen calibrators/thresholds/selections applied; labels joined here only.
       python external/run_external.py analyse --name <name> --labels external/<name>_labels.csv
     labels columns: unit_id, label (0/1), plus optional subset / diagnosis / group columns used for subgroups
     -> results/external/<name>_*.csv + manifest
Frozen code in src/murmur is imported, never modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = REPO / "results" / "external"


# ------------------------------------------------------------------ stage 1: scoring (no labels)
def read_audio(path):
    import soundfile as sf
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(1).astype(np.float32), int(fs)          # mono


def build_external_cache(man: pd.DataFrame, pp_cfg: dict):
    from murmur.preprocess import preprocess_signal
    sigs, rows, off, bad = [], [], 0, []
    for r in man.itertuples(index=False):
        try:
            path = Path(r.path) if Path(r.path).is_absolute() else REPO / r.path
            x, fs = read_audio(path)
            if len(x) == 0:
                raise ValueError("empty")
        except Exception as e:      # unreadable files are excluded and counted (addendum section 2)
            bad.append({"unit_id": r.unit_id, "recording_id": r.recording_id, "path": r.path, "error": str(e)[:200]})
            continue
        y = preprocess_signal(x, fs, pp_cfg)
        sigs.append(y)
        rows.append({"recording_id": str(r.recording_id), "patient_id": str(r.unit_id), "location": "NA",
                     "offset": off, "length": len(y), "raw_fs": fs, "raw_len": len(x)})
        off += len(y)
    wave = np.concatenate(sigs).astype(np.float32) if sigs else np.zeros(0, np.float32)
    return wave, pd.DataFrame(rows), pd.DataFrame(bad)


def cmd_score(a):
    import torch
    from murmur.corrupt_infer import load_member
    from murmur.paths import load_config
    from murmur.train import DeviceData, score_encounters
    cfg, p1 = load_config("train"), load_config("phase1")
    man = pd.read_csv(a.manifest, dtype=str)
    assert {"unit_id", "recording_id", "path"} <= set(man.columns), "manifest needs unit_id, recording_id, path"
    assert "label" not in man.columns, "scoring stage must not see labels"
    t = time.time()
    wave, rec_index, bad = build_external_cache(man, p1["preprocess"])
    units = sorted(rec_index.patient_id.unique(), key=str)
    enc = pd.DataFrame({"patient_id": units, "label": 0, "murmur_locations": ""})   # dummy: labels unused for scoring
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DeviceData(wave, rec_index, enc, cfg, device)
    run_dirs = sorted(d for d in (REPO / a.runs).glob("p*_o*_i*_*_s*")
                      if ".tmp" not in d.name and ".stale" not in d.name and (d / "final.pt").exists())
    if a.limit:
        run_dirs = run_dirs[: a.limit]
    probs = np.full((len(run_dirs), len(units)), np.nan)
    for k, d in enumerate(run_dirs):
        man_r = json.loads((d / "manifest.json").read_text())
        model = load_member(d, man_r["arch"], cfg, device)
        probs[k] = score_encounters(model, data, units, cfg)
        if k % 50 == 0:
            print(f"{k}/{len(run_dirs)} runs scored ({time.time() - t:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{a.name}_scores.npz", run_ids=np.asarray([d.name for d in run_dirs]),
                        unit_ids=np.asarray(units), prob=probs)
    rec_index.to_csv(OUT / f"{a.name}_recordings.csv", index=False, lineterminator="\n")
    bad.to_csv(OUT / f"{a.name}_unreadable.csv", index=False, lineterminator="\n")
    print(f"done: {len(run_dirs)} runs x {len(units)} units; {len(bad)} unreadable files; {time.time() - t:.0f}s")


# ------------------------------------------------------------------ stage 2: analysis (labels joined here)
def auroc_ci(Q, y, rng, n_boot=2000):
    """Unit bootstrap of the replicate-mean AUROC (Mann-Whitney form, vectorised over replicates)."""
    from scipy.stats import rankdata
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy = y[i]
        n1 = yy.sum(); n0 = n - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(Q[:, i], axis=1)
        a = (r[:, yy == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
        vals.append(a.mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def cmd_analyse(a):
    from murmur import analysis as an
    from murmur.calibration import Platt, calibration_slope_intercept
    from murmur.hashing import sha256_file
    from murmur.paths import derived_dir, load_config
    from murmur.referral import refer, standalone_top_m
    acfg, tcfg, p1 = load_config("analysis"), load_config("train"), load_config("phase1")
    pc, seeds, archs, b = p1["partitions"], tcfg["member_seeds"], an.ARCHS, acfg["budget_primary"]
    z = np.load(OUT / f"{a.name}_scores.npz")
    run_ids, units, P = z["run_ids"].astype(str), z["unit_ids"].astype(str), z["prob"]
    lab = pd.read_csv(a.labels, dtype={"unit_id": str}).set_index("unit_id")
    keep = np.array([u in lab.index for u in units])
    units, P = units[keep], P[:, keep]
    lab = lab.loc[units]
    y = lab.label.astype(int).to_numpy()
    raw = {rid: P[k] for k, rid in enumerate(run_ids)}
    # frozen CirCor development quantities (dev labels only)
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    label_of = enc.set_index("patient_id").label.astype(int)
    runs = an.load_runs(REPO / a.runs, acfg["epoch"])
    sel = pd.read_csv(REPO / "results" / "studyC_selection.csv")
    sel = sel[np.isclose(sel.budget, b)].set_index(["partition", "outer", "seed", "primary"])
    perf, det, transfer, cal_all, flags_all = [], {}, [], {}, {}
    n_rep = 0
    for p in range(pc["n_partitions"]):
        for o in range(pc["n_outer"]):
            c = an.build_context(runs, p, o, pc["n_inner"], archs, seeds, label_of, acfg["sensitivity_target"])
            cal = {}
            for ar in archs:
                for s in seeds:
                    pl = Platt(); pl.a, pl.b = c.platt[(ar, s)]["a"], c.platt[(ar, s)]["b"]
                    for j in range(pc["n_inner"]):
                        rid = f"p{p}_o{o}_i{j}_{ar}_s{s}"
                        if rid not in raw:
                            continue
                        q = pl.transform(raw[rid])
                        cal[(ar, s, j)] = q
                        cal_all[(ar, p, o, s, j)] = q
                        thr = c.thr[(ar, s)]
                        flag = q >= thr
                        flags_all.setdefault(ar, []).append(flag.astype(float))
                        slope, icpt = calibration_slope_intercept(q, y) if 0 < y.sum() < len(y) else (np.nan, np.nan)
                        perf.append({"partition": p, "outer": o, "inner": j, "arch": ar, "seed": s, "auroc": auroc(y, q),
                                     "sensitivity": flag[y == 1].mean() if y.sum() else np.nan,
                                     "specificity": 1 - flag[y == 0].mean() if (y == 0).sum() else np.nan,
                                     "pos_share": flag.mean(), "dev_neg_share": c.dev_neg_share[(ar, s)],
                                     "cal_slope": slope, "cal_intercept": icpt, "brier": float(np.mean((q - y) ** 2))})
            for plabel, prim in acfg["primaries"].items():
                for s in seeds:
                    chosen, standalone = sel.loc[(p, o, s, plabel), ["chosen", "standalone"]]
                    for j in range(pc["n_inner"]):
                        if any((ar, s2, j) not in cal for ar in archs for s2 in seeds):
                            continue
                        k = n_rep = n_rep + 1
                        src = {key: v for key, v in cal.items() if key[2] == j}
                        # candidate scores, same definitions as murmur.analysis.candidate_scores
                        S = {f"single:{ar}": src[(ar, s, j)] for ar in archs if ar != prim}
                        for ar in archs:
                            S[f"hom_mean:{ar}"] = np.mean([src[(ar, s2, j)] for s2 in seeds], 0)
                        am = np.stack([np.mean([src[(ar, s2, j)] for s2 in seeds], 0) for ar in archs])
                        S["het_mean"], S["het_disagree"] = am.mean(0), am.std(0)
                        pp, thr = src[(prim, s, j)], c.thr[(prim, s)]
                        wf = refer(pp, thr, S[chosen], b, acfg["budget_base"], seed=k)
                        own = refer(pp, thr, pp, b, acfg["budget_base"], seed=k)
                        m = int(wf.flagged.sum())
                        arms = {"secondary": wf.flagged, "own_threshold": own.flagged,
                                "standalone_single": standalone_top_m(src[(standalone, s, j)], m, seed=k),
                                "standalone_ensemble": standalone_top_m(np.mean([src[(standalone, s2, j)] for s2 in seeds], 0), m, seed=k)}
                        for arm, f in arms.items():
                            det.setdefault((plabel, arm), []).append(f.astype(float))
                        transfer.append({"partition": p, "outer": o, "seed": s, "inner": j, "primary": plabel,
                                         "test_neg_share": float(np.mean(pp < thr)), "dev_neg_share": c.dev_neg_share[(prim, s)],
                                         "k": wf.k, "total_flags": m})
    perf = pd.DataFrame(perf)
    rng = np.random.default_rng(acfg["boot_seed"])
    n = len(y)
    summ = []
    for ar in archs:
        d = perf[perf.arch == ar]
        Q = np.stack([cal_all[kk] for kk in sorted(cal_all) if kk[0] == ar])      # replicates x units
        lo, hi = auroc_ci(Q, y, rng, n_boot=2000)
        summ.append({"arch": ar, "n_replicates": len(d), "auroc": d.auroc.mean(), "auroc_ci95_lo": lo, "auroc_ci95_hi": hi,
                     "sensitivity": d.sensitivity.mean(), "specificity": d.specificity.mean(), "pos_share": d.pos_share.mean(),
                     "dev_neg_share": d.dev_neg_share.mean(), "cal_slope": d.cal_slope.mean(),
                     "cal_intercept": d.cal_intercept.mean(), "brier": d.brier.mean()})
    summ = pd.DataFrame(summ)
    pos = y == 1
    idx = rng.integers(0, pos.sum(), size=(10000, pos.sum()))
    contr = []
    for plabel in acfg["primaries"]:
        A = np.mean(det[(plabel, "secondary")], 0)[pos]
        for bb in ("own_threshold", "standalone_single", "standalone_ensemble"):
            Bm = np.mean(det[(plabel, bb)], 0)[pos]
            dvec = A - Bm
            est = dvec.mean()
            boots = dvec[idx].mean(1)
            contr.append({"primary": plabel, "comparator": bb, "n_positive_units": int(pos.sum()),
                          "extra_detected": est * pos.sum(), "ci95_lo": np.percentile(boots, 2.5) * pos.sum(),
                          "ci95_hi": np.percentile(boots, 97.5) * pos.sum(), "detected_A": A.sum(), "detected_B": Bm.sum()})
    contr = pd.DataFrame(contr)
    tr = pd.DataFrame(transfer)
    tag = f"{a.name}_{a.tag}" if a.tag else a.name
    # descriptive subgroups: share flagged per unit (mean over replicates), by any grouping column in the labels file
    unit_flag = {f"model:{ar}": np.mean(v, 0) for ar, v in flags_all.items()}
    unit_flag.update({f"workflow:{pl}:{arm}": np.mean(v, 0) for (pl, arm), v in det.items()})
    sub = []
    for col in [c for c in ("subset", "diagnosis", "age_group") if c in lab.columns]:
        for g, idx_g in lab.reset_index().groupby(col).groups.items():
            ii = np.asarray(list(idx_g))
            for arm, uf in unit_flag.items():
                sub.append({"group_by": col, "group": g, "arm": arm, "n_units": len(ii), "n_positive": int(y[ii].sum()),
                            "flag_rate_positive": float(uf[ii][y[ii] == 1].mean()) if y[ii].sum() else np.nan,
                            "flag_rate_negative": float(uf[ii][y[ii] == 0].mean()) if (y[ii] == 0).sum() else np.nan})
    pd.DataFrame(sub).to_csv(OUT / f"{tag}_subgroups.csv", index=False, lineterminator="\n")
    perf.to_csv(OUT / f"{tag}_perf_per_replicate.csv", index=False, lineterminator="\n")
    summ.to_csv(OUT / f"{tag}_performance.csv", index=False, lineterminator="\n")
    contr.to_csv(OUT / f"{tag}_referral.csv", index=False, lineterminator="\n")
    tr.to_csv(OUT / f"{tag}_transfer.csv", index=False, lineterminator="\n")
    (OUT / f"{tag}_manifest.json").write_text(json.dumps({
        "addendum": "protocol/addendum_external.md", "n_units": int(n), "n_positive": int(pos.sum()),
        "outputs": {f.name: sha256_file(f) for f in sorted(OUT.glob(f"{tag}_*.csv"))}}, indent=2))
    print(summ.round(3).to_string(index=False)); print(contr.round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--manifest", required=True); s.add_argument("--name", required=True)
    s.add_argument("--runs", default="runs"); s.add_argument("--limit", type=int, default=0)
    t = sub.add_parser("analyse"); t.add_argument("--name", required=True); t.add_argument("--labels", required=True)
    t.add_argument("--tag", default="", help="output suffix for a registered subset, e.g. clean")
    t.add_argument("--runs", default="runs")
    a = ap.parse_args()
    {"score": cmd_score, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
