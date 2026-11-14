"""Freeze the protocol: run all tests, check that no placeholders remain, hash everything, write protocol/FREEZE.json.

    python scripts/make_freeze.py --check     # report what still blocks the freeze (writes nothing)
    python scripts/make_freeze.py             # freeze (refuses while blockers remain)

After the freeze, ANY change to src/, scripts/, tests/ or configs/ changes the code hash, and the full
training run, corruption run and analysis all refuse to proceed. Changes after the freeze must be logged
as deviations in protocol/deviations.md and the freeze re-made with --reason.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from murmur.hashing import sha256_file  # noqa: E402
from murmur.paths import cache_dir, derived_dir  # noqa: E402
from murmur.train import code_files, code_hash  # noqa: E402

MARKERS = re.compile(r"PLACEHOLDER|PROVISIONAL|OPEN DECISION", re.I)


def blockers() -> list[str]:
    out = []
    for f in sorted((REPO / "configs").glob("*.json")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if MARKERS.search(line):
                out.append(f"{f.relative_to(REPO)}:{i}: {line.strip()[:110]}")
    for f in ("src/murmur/corrupt.py", "scripts/run_corrupt.py"):
        for i, line in enumerate((REPO / f).read_text().splitlines(), 1):
            if MARKERS.search(line):
                out.append(f"{f}:{i}: {line.strip()[:110]}")
    if not (REPO / "protocol" / "protocol.md").exists():
        out.append("protocol/protocol.md missing")
    if not (REPO / "runs_pilot" / "pilot_curves.csv").exists():
        out.append("pilot not run (runs_pilot/pilot_curves.csv missing): epoch count not fixed from development data")
    if (REPO / "runs").exists() and any((REPO / "runs").glob("*/manifest.json")):
        out.append("runs/ already contains full runs - outer-test predictions exist before the freeze!")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reason", default="initial freeze")
    a = ap.parse_args()
    b = blockers()
    if b:
        print("Freeze blockers:")
        for x in b:
            print("  -", x)
    if a.check:
        print(f"\ncode hash now: {code_hash()[:16]} over {len(code_files())} files")
        return
    if b:
        sys.exit("\nNot frozen. Resolve the blockers above (each PLACEHOLDER/PROVISIONAL/OPEN DECISION line must be decided).")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=REPO)
    if r.returncode != 0:
        sys.exit("tests failed - not frozen")
    rep = json.loads((derived_dir() / "phase1_report.json").read_text())
    freeze = {
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": a.reason,
        "code_hash": code_hash(),
        "files": {f: sha256_file(REPO / f) for f in code_files()},
        "folds_sha256": sha256_file(derived_dir() / "folds.csv"),
        "cohort_encounters_sha256": sha256_file(derived_dir() / "cohort_encounters.csv"),
        "cache_waveforms_sha256": rep.get("cache", {}).get("waveforms_sha256"),
        "protocol_sha256": sha256_file(REPO / "protocol" / "protocol.md"),
        "pilot_curves_sha256": sha256_file(REPO / "runs_pilot" / "pilot_curves.csv"),
        "osf_registration": None,
    }
    out = REPO / "protocol" / "FREEZE.json"
    if out.exists():
        prev = json.loads(out.read_text())
        (REPO / "protocol" / f"FREEZE_superseded_{prev['frozen_at_utc'].replace(':', '')}.json").write_text(json.dumps(prev, indent=2))
    out.write_text(json.dumps(freeze, indent=2))
    print(f"FROZEN. code hash {freeze['code_hash'][:16]}. Register protocol/protocol.md + FREEZE.json on OSF, then put the OSF link in FREEZE.json -> osf_registration.")


if __name__ == "__main__":
    main()
