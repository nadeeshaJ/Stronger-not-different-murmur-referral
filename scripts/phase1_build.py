"""Phase 1: verify data, build cohort, partitions and waveform cache.

Usage (from repo root):
    python scripts/phase1_build.py              # everything
    python scripts/phase1_build.py --no-cache   # skip the waveform cache
    python scripts/phase1_build.py --no-verify  # skip SHA256SUMS check
Env: MURMUR_DATA_ROOT (folder containing the extracted CirCor dir), MURMUR_DERIVED, MURMUR_CACHE.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from murmur.cohort import build_cohort  # noqa: E402
from murmur.hashing import sha256_file, verify_sha256sums  # noqa: E402
from murmur.partitions import balance_report, make_partitions  # noqa: E402
from murmur.paths import cache_dir, dataset_dir, derived_dir, load_config  # noqa: E402
from murmur.preprocess import build_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    cfg = load_config("phase1")
    ds = dataset_dir(cfg)
    out = derived_dir()
    report = {"dataset_dir": str(ds)}

    if not a.no_verify:
        t = time.time()
        ok, bad = verify_sha256sums(ds)
        report["sha256sums"] = {"ok": ok, "failed": len(bad), "first_failures": bad[:10]}
        print(f"SHA256SUMS: {ok} ok, {len(bad)} failed ({time.time()-t:.0f}s)")
        if bad:
            sys.exit("Checksum failures - re-extract the dataset.")

    enc, rec, log = build_cohort(ds, cfg)
    report["cohort"] = log
    exp = cfg["expected_counts"]
    for k, v in exp.items():
        assert log[k] == v, f"{k}: got {log[k]}, expected {v}"
    enc.to_csv(out / "cohort_encounters.csv", index=False, lineterminator="\n")
    rec.to_csv(out / "cohort_recordings.csv", index=False, lineterminator="\n")
    print("Cohort:", {k: log[k] for k in exp})

    pc = cfg["partitions"]
    folds = make_partitions(enc, pc["seeds"], pc["n_outer"], pc["n_inner"])
    folds.to_csv(out / "folds.csv", index=False, lineterminator="\n")
    balance_report(folds).to_csv(out / "fold_balance.csv", index=False, lineterminator="\n")
    report["folds_sha256"] = sha256_file(out / "folds.csv")
    report["cohort_encounters_sha256"] = sha256_file(out / "cohort_encounters.csv")
    report["cohort_recordings_sha256"] = sha256_file(out / "cohort_recordings.csv")
    print("folds.csv sha256:", report["folds_sha256"])

    if not a.no_cache:
        t = time.time()
        meta = build_cache(rec, ds, cache_dir(), cfg["preprocess"])
        report["cache"] = {k: meta[k] for k in ("n_recordings", "n_samples", "waveforms_sha256", "index_sha256")}
        print(f"Cache: {meta['n_recordings']} recordings, {meta['n_samples']} samples ({time.time()-t:.0f}s)")

    (out / "phase1_report.json").write_text(json.dumps(report, indent=2))
    print("Wrote", out / "phase1_report.json")


if __name__ == "__main__":
    main()
