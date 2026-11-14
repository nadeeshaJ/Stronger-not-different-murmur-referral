"""Resumable training runner.

    python scripts/run_train.py --mode pilot                 # Phase 4: partition 0, outer 0, inner models, NO test scoring
    python scripts/run_train.py --mode pilot --bench 1       # one run only, prints seconds/run + VRAM
    python scripts/run_train.py --mode full --workers 2      # Phase 6: everything (refuses unless protocol frozen)
    python scripts/run_train.py --status [--mode full]       # progress only

Finished runs (directory with manifest.json) are skipped, so a crash/reboot only loses the run in progress.
--workers N starts N processes on the same GPU sharing one job queue: each worker claims the next unclaimed
pending run (atomic directory creation under <runs>/.claims), so a slower worker never holds back the rest.
Operational only: which process trains a run has no effect on the run (seeds come from the run spec).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from murmur.paths import cache_dir, derived_dir, load_config  # noqa: E402


def claim(claims_dir: Path, run_id: str) -> bool:
    """Atomically claim a run for this worker (mkdir is atomic on Windows and POSIX).
    Each launch uses its own claims directory, so claims from interrupted launches never block runs."""
    d = Path(claims_dir)
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(d / run_id)
        return True
    except FileExistsError:
        return False


def no_power_throttling():
    """Windows: opt this process out of EcoQoS/efficiency-mode CPU throttling and raise priority slightly.
    Best effort; no effect on results."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class PPTS(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        st = PPTS(1, 0x1, 0x0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED, state off
        k32 = ctypes.windll.kernel32
        k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))  # ProcessPowerThrottling
    except Exception as e:  # noqa: BLE001
        print("power-throttling opt-out failed:", e, flush=True)
    try:
        import psutil
        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception as e:  # noqa: BLE001
        print("priority change failed:", e, flush=True)


def pending_runs(mode, cfg, p1, out_root, archs=None, partitions=None):
    from murmur.train import enumerate_runs
    pc = p1["partitions"]
    ii = cfg["inner_inner"]["partitions"] if cfg["inner_inner"]["enabled"] else ()
    if mode == "pilot":
        runs = [r for r in enumerate_runs(1, 1, pc["n_inner"], archs or cfg["archs"], cfg["member_seeds"])]
    else:
        runs = enumerate_runs(pc["n_partitions"], pc["n_outer"], pc["n_inner"], archs or cfg["archs"], cfg["member_seeds"], ii)
        if partitions is not None:
            runs = [r for r in runs if r.partition in partitions]
    done = {r.run_id for r in runs if (out_root / r.run_id / "manifest.json").exists()}
    return runs, [r for r in runs if r.run_id not in done]


def status(runs, todo, out_root):
    done = [r for r in runs if r not in todo]
    walls = []
    for r in done:
        try:
            walls.append(json.loads((out_root / r.run_id / "manifest.json").read_text())["wall_s"])
        except Exception:
            pass
    msg = f"{len(done)}/{len(runs)} runs done"
    if walls:
        msg += f" | median {np.median(walls):.0f}s/run | remaining ~{np.median(walls) * len(todo) / 3600:.1f} GPU-hours (1 worker)"
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--partitions", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override (pilot only)")
    ap.add_argument("--bench", type=int, default=0, help="run only this many pending runs")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--claims", default=None, help="claims directory for this launch (set by the launcher)")
    a = ap.parse_args()

    cfg = load_config("train")
    p1 = load_config("phase1")
    out_root = REPO / ("runs_pilot" if a.mode == "pilot" else "runs")
    out_root.mkdir(exist_ok=True)
    runs, todo = pending_runs(a.mode, cfg, p1, out_root, a.archs, a.partitions)

    if a.status:
        print(status(runs, todo, out_root))
        return

    from murmur.train import freeze_status
    if a.mode == "full":
        ok, why = freeze_status(REPO)
        if not ok:
            sys.exit(f"REFUSING full run: {why}. Outer-test scoring is only allowed after the protocol freeze.")
        if a.epochs is not None:
            sys.exit("--epochs override is not allowed in full mode")

    # ---- launcher: spawn workers
    if a.workers > 1 and a.worker is None:
        claims_dir = out_root / ".claims" / time.strftime("launch-%Y%m%dT%H%M%S")
        logs = out_root / "logs"
        logs.mkdir(exist_ok=True)
        procs = []
        for k in range(a.workers):
            cmd = [sys.executable, __file__, "--mode", a.mode, "--workers", str(a.workers), "--worker", str(k), "--claims", str(claims_dir)]
            for flag, val in (("--archs", a.archs), ("--partitions", a.partitions)):
                if val:
                    cmd += [flag, *map(str, val)]
            if a.epochs:
                cmd += ["--epochs", str(a.epochs)]
            if a.bench:
                cmd += ["--bench", str(a.bench)]
            fh = open(logs / f"worker{k}.log", "a")
            procs.append(subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT))
            print(f"worker {k} started -> {logs / f'worker{k}.log'}")
        while any(p.poll() is None for p in procs):
            time.sleep(60)
            print(time.strftime("%H:%M"), status(*pending_runs(a.mode, cfg, p1, out_root, a.archs, a.partitions), out_root), flush=True)
        codes = [p.returncode for p in procs]
        print("workers finished, exit codes", codes)
        sys.exit(max(codes))

    import torch
    from murmur.train import DeviceData, train_one

    overall = status(runs, todo, out_root)
    use_claims = a.claims is not None
    no_power_throttling()
    if a.bench:
        todo = todo[: a.bench]
    if not todo:
        print("nothing to do:", status(runs, todo, out_root))
        return
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device {device} | worker {a.worker} sharing a queue of {len(todo)} pending runs | overall: {overall}", flush=True)

    cdir = cache_dir()
    wave = np.load(cdir / "waveforms.npy", mmap_mode="r")
    rec_index = pd.read_csv(cdir / "index.csv", dtype={"patient_id": str})
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    folds = pd.read_csv(derived_dir() / "folds.csv", dtype={"patient_id": str, "individual_id": str})
    data = DeviceData(wave, rec_index, enc, cfg, device)
    score_test = a.mode == "full"
    for i, spec in enumerate(todo):
        if (out_root / spec.run_id / "manifest.json").exists():
            continue
        if use_claims and not claim(Path(a.claims), spec.run_id):
            continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        train_one(spec, data, folds, cfg, out_root, p1["partitions"]["seeds"], score_test, a.epochs,
                  log=lambda m: print(m, flush=True))
        extra = f" | peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if device.type == "cuda" else ""
        print(f"[{i + 1}/{len(todo)}] {spec.run_id} done in {time.time() - t:.0f}s{extra}", flush=True)
    print(status(*pending_runs(a.mode, cfg, p1, out_root, a.archs, a.partitions), out_root))


if __name__ == "__main__":
    main()
