"""Phase 4 pilot report (development data only: inner OOF of partition 0, outer fold 0).

    python scripts/pilot_report.py [--runs runs_pilot]

Prints pooled-OOF AUROC / PR-AUC per architecture x epoch (mean over seeds), runtime and the
suggested fixed epoch count (smallest epoch within 0.005 AUROC of each architecture's best, max over archs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from murmur.paths import derived_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_pilot")
    a = ap.parse_args()
    root = REPO / a.runs
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str})
    lab = enc.set_index("patient_id").label
    pooled, walls, vram = {}, [], []
    for d in sorted(root.glob("p0_o0_i*_*_s*")):
        if ".tmp" in d.name or not (d / "manifest.json").exists():
            continue
        m = json.loads((d / "manifest.json").read_text())
        assert not m.get("scored_test", False), f"{d.name} scored the outer test in a pilot!"
        z = np.load(d / "preds.npz")
        _, _, _, *arch, s = d.name.split("_")
        key = ("_".join(arch), int(s[1:]))
        pooled.setdefault(key, []).append((z["oof_ids"].astype(str), z["oof_prob"]))
        walls.append(m.get("wall_s", np.nan))
    rows = []
    for (arch, s), parts in pooled.items():
        if len(parts) < 5:
            print(f"warning: {arch} seed {s} has {len(parts)}/5 inner models")
        ids = np.concatenate([p[0] for p in parts])
        P = np.concatenate([p[1] for p in parts], axis=1)
        y = lab[ids].to_numpy()
        for e in range(P.shape[0]):
            rows.append({"arch": arch, "seed": s, "epoch": e + 1, "auroc": roc_auc_score(y, P[e]), "prauc": average_precision_score(y, P[e])})
    df = pd.DataFrame(rows)
    out = REPO / a.runs / "pilot_curves.csv"
    df.to_csv(out, index=False, lineterminator="\n")
    curve = df.groupby(["arch", "epoch"])[["auroc", "prauc"]].mean().reset_index()
    wide = curve.pivot(index="epoch", columns="arch", values="auroc")
    print("Pooled inner-OOF AUROC (mean over seeds):")
    print(wide.round(3).to_string())
    pick = {}
    for arch, g in curve.groupby("arch"):
        best = g.auroc.max()
        pick[arch] = int(g[g.auroc >= best - 0.005].epoch.min())
        print(f"{arch}: best {best:.3f} at epoch {int(g.loc[g.auroc.idxmax(), 'epoch'])}; within 0.005 from epoch {pick[arch]}")
    print(f"\nSuggested fixed epoch count: {max(pick.values())}")
    print(f"runs: {len(walls)} | median {np.nanmedian(walls):.0f}s/run -> full 1,125 runs ~{np.nanmedian(walls) * 1125 / 3600:.1f} GPU-hours with 1 worker")
    print("curves saved to", out)


if __name__ == "__main__":
    main()
