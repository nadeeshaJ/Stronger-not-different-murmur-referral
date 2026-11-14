"""Simulated run outputs in the exact runs/<run_id>/preds.npz format, for testing the analysis end to end.

Latent model: encounter "visibility" v ~ N(0,1). For a murmur-present encounter, an architecture's logit is
    strength[a] * (y + shared*v + own*u_a) + noise,   u_a ~ N(0,1) arch-specific
`extra` > 0 gives the non-primary architectures access to a component the weak primary misses
(so a secondary can genuinely find the primary's misses); extra = 0 means no added information.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .train import RunSpec


def simulate_runs(enc: pd.DataFrame, folds: pd.DataFrame, out_root: Path, n_part=5, n_outer=5, n_inner=5,
                  archs=("wave_cnn", "spec_cnn", "crnn"), seeds=(0, 1, 2), epochs=3,
                  strength=None, extra=1.0, seed=0, score_test=True):
    strength = strength or {"wave_cnn": 1.2, "spec_cnn": 2.0, "crnn": 2.6}
    rng = np.random.default_rng(seed)
    ids = enc.patient_id.astype(str).to_numpy()
    y = enc.label.to_numpy()
    n = len(ids)
    hidden = rng.standard_normal(n)              # component the weak primary cannot see
    arch_u = {a: rng.standard_normal(n) for a in archs}
    pos = {pid: i for i, pid in enumerate(ids)}
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    def logit(a, s, idx, ep, key):
        kp, ko, kj, kind = key
        r = np.random.default_rng(np.random.SeedSequence([seed, kp, ko, kj, int(kind == "test"), list(archs).index(a), s, ep]))
        sees_hidden = 0.0 if a == "wave_cnn" else extra
        sig = y[idx] * (1.5 + sees_hidden * hidden[idx] + 0.5 * arch_u[a][idx]) - 1.3
        ep_gain = 0.6 + 0.4 * (ep + 1) / epochs
        return strength[a] * ep_gain * sig + r.standard_normal(len(idx)) * 1.0

    for p in range(n_part):
        for o in range(n_outer):
            f = folds[(folds.partition == p) & (folds.outer_fold == o)]
            test_ids = f.loc[f.role == "test", "patient_id"].astype(str).to_numpy()
            for j in range(n_inner):
                oof_ids = f.loc[(f.role == "dev") & (f.inner_fold == j), "patient_id"].astype(str).to_numpy()
                oi = np.array([pos[t] for t in oof_ids])
                ti = np.array([pos[t] for t in test_ids])
                for a in archs:
                    for s in seeds:
                        spec = RunSpec(p, o, (j,), a, s)
                        d = out_root / spec.run_id
                        d.mkdir(exist_ok=True)
                        oof = np.stack([1 / (1 + np.exp(-logit(a, s, oi, e, (p, o, j, "oof")))) for e in range(epochs)])
                        tst = np.stack([1 / (1 + np.exp(-logit(a, s, ti, e, (p, o, j, "test")))) for e in range(epochs)]) if score_test else np.zeros((epochs, 0))
                        np.savez_compressed(d / "preds.npz", oof_ids=oof_ids.astype(str), oof_heldout_fold=np.full(len(oof_ids), j),
                                            oof_prob=oof, test_ids=(test_ids if score_test else np.array([], str)).astype(str),
                                            test_prob=tst, train_loss=np.zeros(epochs))
                        (d / "manifest.json").write_text(json.dumps({"run_id": spec.run_id, "simulated": True}))
    return out_root


def simulate_corrupt(run_root: Path, out: Path, enc: pd.DataFrame, conditions=None, seed=0):
    """Degrade simulated clean test probabilities in the run_corrupt.py output format."""
    from .corrupt import CONDITIONS_DEFAULT
    conditions = conditions or [c["name"] for c in CONDITIONS_DEFAULT]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in Path(run_root).glob("p*_o*_i*_*_s*") if ".tmp" not in d.name and (d / "manifest.json").exists())
    for ci, cn in enumerate(conditions):
        sev = 0.0 if cn == "identity" else min(0.9, 0.08 * ci)
        rng = np.random.default_rng([seed, ci])
        names, ids, probs = [], [], []
        for d in dirs:
            z = np.load(d / "preds.npz")
            p = np.clip(z["test_prob"][-1], 1e-6, 1 - 1e-6)
            lg = np.log(p) - np.log1p(-p)
            lg = lg * (1 - sev) - 1.5 * sev + rng.standard_normal(len(lg)) * sev
            names.append(d.name); ids.append(z["test_ids"].astype(str)); probs.append(1 / (1 + np.exp(-lg)) if sev else z["test_prob"][-1])
        from .corrupt_infer import pack
        np.savez_compressed(out / f"{cn}.npz", **pack(names, ids, probs))
        snr = np.inf if cn == "identity" else 30 - 2.5 * ci
        pd.DataFrame({"recording_id": ["x"], "condition": [cn], "effective_snr_db_filtered": [snr]}).to_csv(out / f"{cn}_recordings.csv", index=False)
        pd.DataFrame([{"run_id": "x", "arch": a, "partition": 0, "outer": 0, "recording_id": "x", "effective_snr_db_input": snr - k}
                      for k, a in enumerate(["wave_cnn", "spec_cnn", "crnn"])]).to_csv(out / f"{cn}_input_snr.csv", index=False)
    return out
