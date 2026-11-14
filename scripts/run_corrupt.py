"""Study B: re-score the outer test with frozen final checkpoints under every corruption condition (GPU).

    python scripts/run_corrupt.py                 # all conditions; skips conditions already written
    python scripts/run_corrupt.py --conditions identity white_0dB

Requires the protocol freeze and a complete set of full runs. Identity must reproduce clean predictions
within the tolerance before any other condition is written.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

IDENTITY_TOL = 1e-3   # max |identity prob - clean prob| (identical inputs; allows GPU kernel nondeterminism)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs_corrupt")
    a = ap.parse_args()
    import torch
    from murmur.corrupt import CONDITIONS_DEFAULT
    from murmur.corrupt_infer import corrupted_cache, infer_condition
    from murmur.paths import cache_dir, dataset_dir, derived_dir, load_config
    from murmur.preprocess import read_wav
    from murmur.train import DeviceData, freeze_status

    ok, why = freeze_status(REPO)
    if not ok:
        sys.exit(f"REFUSING: {why}")
    cfg, p1 = load_config("train"), load_config("phase1")
    run_root, out = REPO / a.runs, REPO / a.out
    out.mkdir(exist_ok=True)
    run_dirs = sorted(d for d in run_root.glob("p*_o*_i*_*_s*") if ".tmp" not in d.name and ".stale" not in d.name and (d / "manifest.json").exists())
    pc = p1["partitions"]
    expected = pc["n_partitions"] * pc["n_outer"] * pc["n_inner"] * len(cfg["archs"]) * len(cfg["member_seeds"])
    if len(run_dirs) != expected:
        sys.exit(f"{len(run_dirs)}/{expected} full runs present; finish training first")
    rep_runs = {d.name for d in run_dirs if "_i0_" in d.name and d.name.endswith("_s0")}   # one per (p, o, arch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdir = cache_dir()
    clean = np.load(cdir / "waveforms.npy", mmap_mode="r")
    rec_index = pd.read_csv(cdir / "index.csv", dtype={"patient_id": str})
    rec_meta = pd.read_csv(derived_dir() / "cohort_recordings.csv", dtype={"patient_id": str}).set_index("recording_id")
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    folds = pd.read_csv(derived_dir() / "folds.csv", dtype={"patient_id": str, "individual_id": str})
    data_clean = DeviceData(clean, rec_index, enc, cfg, device)
    ds = dataset_dir()
    loader = lambda rid: read_wav(ds / rec_meta.wav[rid])

    conds = [c for c in CONDITIONS_DEFAULT if a.conditions is None or c["name"] in a.conditions]
    conds.sort(key=lambda c: c["name"] != "identity")     # identity first
    for c in conds:
        if (out / f"{c['name']}.npz").exists():
            print("skip", c["name"]); continue
        t = time.time()
        wave, info = corrupted_cache(rec_index, loader, c, p1["preprocess"], clean)
        data_c = DeviceData(wave, rec_index, enc, cfg, device)
        res, snr = infer_condition(c["name"], run_dirs, data_clean, data_c, folds, cfg, device, rep_runs)
        if c["name"] == "identity":
            from murmur.corrupt_infer import unpack
            worst = 0.0
            for d, (tid, pr) in unpack(res).items():
                z = np.load(run_root / d / "preds.npz")
                assert (z["test_ids"].astype(str) == tid).all()
                worst = max(worst, float(np.max(np.abs(z["test_prob"][-1] - pr))))
            print(f"identity check: max |diff| = {worst:.2e} (tolerance {IDENTITY_TOL})")
            if worst > IDENTITY_TOL:
                sys.exit("IDENTITY CHECK FAILED - corrupted results must not be analysed")
            (out / "identity_check.txt").write_text(f"max_abs_diff {worst}\ntolerance {IDENTITY_TOL}\n")
        np.savez_compressed(out / f"{c['name']}.npz", **res)
        info.to_csv(out / f"{c['name']}_recordings.csv", index=False, lineterminator="\n")
        snr.to_csv(out / f"{c['name']}_input_snr.csv", index=False, lineterminator="\n")
        del data_c
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{c['name']}: {len(res['run_ids'])} runs scored in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
