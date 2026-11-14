"""Study B inference: frozen final checkpoints re-score the outer test on corrupted audio.

For each condition the whole corrupted waveform cache is built once on the CPU (corruption on RAW audio,
then the standard preprocessing), so every model sees identical corrupted inputs. Recording lengths are
unchanged, so the clean cache index is reused.
Outputs per condition: <out>/<condition>.npz with run_ids, test_ids (per run) and probabilities, plus
<out>/<condition>_recordings.csv with realised corruption statistics and effective SNR at
(a) the filtered waveform and (b) each architecture's actual input, measured with representative models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .corrupt import apply, condition_seed, effective_snr_db
from .models import build_model
from .preprocess import preprocess_signal
from .train import score_encounters


def corrupted_cache(rec_index: pd.DataFrame, raw_loader, cond: dict, pp_cfg: dict, clean_wave=None):
    """raw_loader(recording_id) -> (raw float signal, fs). Returns (wave float32, per-recording info df)."""
    total = int(rec_index.offset.iloc[-1] + rec_index.length.iloc[-1])
    out = np.empty(total, np.float32)
    rows = []
    for r in rec_index.itertuples(index=False):
        x, fs = raw_loader(r.recording_id)
        y, info = apply(x, fs, cond, condition_seed(r.recording_id, cond["name"]))
        proc = preprocess_signal(y, fs, pp_cfg)
        assert len(proc) == r.length, (r.recording_id, len(proc), r.length)
        out[r.offset : r.offset + r.length] = proc
        if clean_wave is not None:
            clean = np.asarray(clean_wave[r.offset : r.offset + r.length])
            info["effective_snr_db_filtered"] = effective_snr_db(clean, proc)
            if cond.get("type") == "clip":
                info["clip_fraction_processed"] = float(np.mean(np.abs(proc) >= 0.999 * np.max(np.abs(proc)))) if len(proc) else 0.0
            if cond.get("type") == "dropout":
                info["dropout_fraction_processed"] = float(np.mean(np.abs(proc) < 1e-7))
        rows.append({"recording_id": r.recording_id, **info})
    return out, pd.DataFrame(rows)


def load_member(run_dir: Path, arch: str, cfg: dict, device):
    m = build_model(arch, cfg).to(device)
    m.load_state_dict(torch.load(run_dir / "final.pt", map_location=device, weights_only=True))
    m.eval()
    return m


@torch.no_grad()
def input_effective_snr(model, data_clean, data_corrupt, rows, batch=256):
    """Per recording: clean-input power / power of (corrupt input - clean input), at the model's input."""
    out = {}
    starts, owner = data_clean.fixed_segments(rows)
    pc, pd_ = {}, {}
    for i in range(0, len(starts), batch):
        s, o = starts[i : i + batch], owner[i : i + batch]
        a = model.input_transform(data_clean.gather(s, o)).double()
        b = model.input_transform(data_corrupt.gather(s, o)).double()
        ea = (a ** 2).flatten(1).sum(1).cpu().numpy()
        ed = ((b - a) ** 2).flatten(1).sum(1).cpu().numpy()
        for r_, x, y in zip(o, ea, ed):
            pc[r_] = pc.get(r_, 0.0) + x
            pd_[r_] = pd_.get(r_, 0.0) + y
    for r_ in pc:
        out[r_] = float("inf") if pd_[r_] == 0 else 10 * np.log10(pc[r_] / pd_[r_])
    return out


def infer_condition(cond_name, run_dirs: list[Path], data_clean, data_corrupt, folds, cfg, device, rep_runs=()):
    """Score every run's outer-test encounters on the corrupted data. rep_runs: run_ids used for input-level SNR."""
    probs, ids, names = [], [], []
    snr_rows = []
    for d in run_dirs:
        man = json.loads((d / "manifest.json").read_text())
        p, o = man["partition"], man["outer"]
        f = folds[(folds.partition == p) & (folds.outer_fold == o) & (folds.role == "test")]
        test_ids = f.patient_id.astype(str).to_numpy()
        model = load_member(d, man["arch"], cfg, device)
        probs.append(score_encounters(model, data_corrupt, test_ids, cfg))
        ids.append(test_ids)
        names.append(d.name)
        if d.name in rep_runs:
            rows = data_clean.rec_rows(test_ids)
            for r_, v in input_effective_snr(model, data_clean, data_corrupt, rows).items():
                snr_rows.append({"run_id": d.name, "arch": man["arch"], "partition": p, "outer": o,
                                 "recording_id": data_clean.rec.recording_id[r_], "effective_snr_db_input": v})
    return pack(names, ids, probs), pd.DataFrame(snr_rows)


def pack(names, ids, probs) -> dict:
    """Test folds differ in size (159-161): pad to the longest, keep lengths."""
    n = np.array([len(i) for i in ids])
    L = int(n.max())
    tid = np.full((len(ids), L), "", dtype=f"<U{max(len(x) for i in ids for x in i)}")
    pr = np.full((len(ids), L), np.nan)
    for k, (i, q) in enumerate(zip(ids, probs)):
        tid[k, : len(i)] = i
        pr[k, : len(q)] = q
    return {"run_ids": np.asarray(names, str), "test_ids": tid, "prob": pr, "n": n}


def unpack(z) -> dict:
    """{run_id: (test_ids, prob)} from a packed condition file."""
    return {r: (z["test_ids"][k, : z["n"][k]].astype(str), z["prob"][k, : z["n"][k]]) for k, r in enumerate(z["run_ids"].astype(str))}
