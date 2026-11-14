"""Training one member and writing its per-epoch encounter scores.

A run is (partition p, outer fold o, held-out inner set H, architecture a, member seed s).
  single-model path : H = {j}          trains on dev inner folds != j, scores inner fold j (OOF) [+ outer test]
  inner-inner path  : H = {j, k}       trains on dev inner folds not in {j,k}, scores folds j and k (OOF only)

Sealing: outer-test encounters are scored ONLY when protocol/FREEZE.json exists and its code hash equals
the current code hash. Pilot runs (Phase 4) never touch the outer test. Test labels are never loaded here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .models import build_model
from .paths import REPO

def code_files(repo: Path = REPO) -> list[str]:
    """Everything frozen at protocol freeze: all source, scripts, tests and configs."""
    pats = ["src/murmur/*.py", "scripts/*.py", "tests/*.py", "configs/*.json"]
    return sorted(str(p.relative_to(repo)).replace("\\", "/") for pat in pats for p in repo.glob(pat))


# ---------------------------------------------------------------- hashing / sealing

def code_hash(repo: Path = REPO) -> str:
    h = hashlib.sha256()
    for f in code_files(repo):
        p = repo / f
        h.update(f.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def freeze_status(repo: Path = REPO) -> tuple[bool, str]:
    f = repo / "protocol" / "FREEZE.json"
    if not f.exists():
        return False, "protocol/FREEZE.json not found (protocol not frozen)"
    frozen = json.loads(f.read_text())
    cur = code_hash(repo)
    if frozen.get("code_hash") != cur:
        return False, f"code hash {cur[:12]} != frozen {frozen.get('code_hash', '')[:12]}"
    return True, "ok"


# ---------------------------------------------------------------- run specs

ARCH_IDX = {"wave_cnn": 0, "spec_cnn": 1, "crnn": 2}


@dataclass(frozen=True)
class RunSpec:
    partition: int
    outer: int
    held_out: tuple          # inner folds held out: (j,) or (j, k)
    arch: str
    seed: int

    @property
    def run_id(self) -> str:
        h = "i" + str(self.held_out[0]) if len(self.held_out) == 1 else "x" + "".join(map(str, self.held_out))
        return f"p{self.partition}_o{self.outer}_{h}_{self.arch}_s{self.seed}"

    @property
    def single(self) -> bool:
        return len(self.held_out) == 1

    def torch_seed(self, partition_seeds) -> int:
        ss = np.random.SeedSequence([partition_seeds[self.partition], self.outer, *self.held_out, 99, ARCH_IDX[self.arch], self.seed])
        return int(ss.generate_state(1)[0])


def enumerate_runs(n_partitions, n_outer, n_inner, archs, seeds, inner_inner_partitions=()) -> list[RunSpec]:
    runs = []
    for p in range(n_partitions):
        for o in range(n_outer):
            for j in range(n_inner):
                for a in archs:
                    for s in seeds:
                        runs.append(RunSpec(p, o, (j,), a, s))
            if p in inner_inner_partitions:
                for j in range(n_inner):
                    for k in range(j + 1, n_inner):
                        for a in archs:
                            for s in seeds:
                                runs.append(RunSpec(p, o, (j, k), a, s))
    return runs


# ---------------------------------------------------------------- data on device

class DeviceData:
    """Whole waveform cache on the device + recording table. Labels: only what training needs."""

    def __init__(self, wave: np.ndarray, rec_index: pd.DataFrame, enc: pd.DataFrame, cfg: dict, device):
        self.device = device
        self.fs = 4000
        self.L = int(round(cfg["segment_s"] * self.fs))
        self.H = int(round(cfg["hop_s"] * self.fs))
        pad = np.zeros(self.L, np.float32)
        self.wave = torch.from_numpy(np.concatenate([np.asarray(wave, np.float32), pad])).to(device)
        rec = rec_index.copy()
        rec["patient_id"] = rec["patient_id"].astype(str)
        e = enc.set_index(enc["patient_id"].astype(str))
        locs = e["murmur_locations"].fillna("").astype(str) if "murmur_locations" in e else pd.Series("", index=e.index)
        enc_lab = e["label"].astype(int)
        if cfg["train_label"] == "location":
            rec["rec_label"] = [int(enc_lab[p] == 1 and l in locs[p].split("+")) for p, l in zip(rec.patient_id, rec.location)]
        elif cfg["train_label"] == "encounter":
            rec["rec_label"] = [int(enc_lab[p]) for p in rec.patient_id]
        else:
            raise ValueError(cfg["train_label"])
        self.rec = rec.reset_index(drop=True)
        self.by_patient = {p: g.index.to_numpy() for p, g in self.rec.groupby("patient_id")}

    # --- index helpers
    def rec_rows(self, patient_ids) -> np.ndarray:
        return np.concatenate([self.by_patient[str(p)] for p in patient_ids])

    def fixed_segments(self, rows) -> tuple[np.ndarray, np.ndarray]:
        """Starts (absolute offsets) and owning row for the deterministic evaluation segmentation
        (same rule as preprocess.segment)."""
        starts, owner = [], []
        for r in rows:
            off, n = int(self.rec.offset[r]), int(self.rec.length[r])
            if n <= self.L:
                ss = [0]
            else:
                ss = list(range(0, n - self.L + 1, self.H))
                if ss[-1] + self.L < n:
                    ss.append(n - self.L)
            starts += [off + s for s in ss]
            owner += [r] * len(ss)
        return np.asarray(starts, np.int64), np.asarray(owner, np.int64)

    def gather(self, starts: np.ndarray, rows: np.ndarray) -> torch.Tensor:
        st = torch.as_tensor(starts, device=self.device)
        idx = st[:, None] + torch.arange(self.L, device=self.device)[None, :]
        x = self.wave[idx]
        # zero samples past the end of each recording (short recordings are zero-padded)
        end = torch.as_tensor((self.rec.offset.to_numpy()[rows] + self.rec.length.to_numpy()[rows]), device=self.device)
        x = x * (idx < end[:, None])
        return x

    def random_crops(self, rows, per_rec, gen: np.random.Generator):
        rows = np.repeat(rows, per_rec)
        off = self.rec.offset.to_numpy()[rows]
        n = self.rec.length.to_numpy()[rows]
        span = np.maximum(n - self.L, 0)
        st = off + (gen.random(len(rows)) * (span + 1)).astype(np.int64)
        return st, rows


# ---------------------------------------------------------------- training

def _set_determinism(seed: int, deterministic: bool):
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def score_encounters(model, data: DeviceData, patient_ids, cfg, batch=512) -> np.ndarray:
    """Encounter-level probability: segment prob -> recording (mean) -> encounter (max)."""
    model.eval()
    rows = data.rec_rows(patient_ids)
    starts, owner = data.fixed_segments(rows)
    probs = np.empty(len(starts), np.float64)
    use_amp = cfg["amp"] and data.device.type == "cuda"
    for i in range(0, len(starts), batch):
        x = data.gather(starts[i : i + batch], owner[i : i + batch])
        with torch.autocast(data.device.type, dtype=torch.float16, enabled=use_amp):
            z = model(x)
        probs[i : i + batch] = torch.sigmoid(z.float()).cpu().numpy()
    seg = pd.DataFrame({"row": owner, "p": probs})
    if cfg["segment_to_recording"] == "mean_prob":
        rec_p = seg.groupby("row").p.mean()
    elif cfg["segment_to_recording"] == "max_prob":
        rec_p = seg.groupby("row").p.max()
    else:
        raise ValueError(cfg["segment_to_recording"])
    rp = pd.DataFrame({"patient_id": data.rec.patient_id.to_numpy()[rec_p.index], "p": rec_p.to_numpy()})
    agg = {"max": "max", "mean": "mean"}[cfg["recording_to_encounter"]]
    enc_p = rp.groupby("patient_id").p.agg(agg)
    return enc_p.reindex([str(p) for p in patient_ids]).to_numpy()


def train_one(spec: RunSpec, data: DeviceData, folds: pd.DataFrame, cfg: dict, out_root: Path,
              partition_seeds, score_test: bool, epochs: int | None = None, log=print) -> Path:
    """Train one member; write <out_root>/<run_id>/{preds.npz, final.pt, manifest.json} atomically."""
    t0 = time.time()
    final_dir = out_root / spec.run_id
    # unique scratch dir per attempt: leftovers from interrupted attempts are never deleted or reused
    # (deletion can fail on Windows while another process holds a handle on the folder)
    tmp_dir = out_root / f"{spec.run_id}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
    tmp_dir.mkdir(parents=True)
    seed = spec.torch_seed(partition_seeds)
    _set_determinism(seed, cfg["deterministic"])
    gen = np.random.default_rng(seed)
    E = epochs or cfg["epochs"]

    f = folds[(folds.partition == spec.partition) & (folds.outer_fold == spec.outer)]
    dev = f[f.role == "dev"]
    train_ids = dev.loc[~dev.inner_fold.isin(spec.held_out), "patient_id"].astype(str).to_numpy()
    oof = {j: dev.loc[dev.inner_fold == j, "patient_id"].astype(str).to_numpy() for j in spec.held_out}
    test_ids = f.loc[f.role == "test", "patient_id"].astype(str).to_numpy() if (score_test and spec.single) else np.array([], str)
    assert not set(train_ids) & set(np.concatenate(list(oof.values()))), "train/OOF overlap"
    assert not set(train_ids) & set(f.loc[f.role == "test", "patient_id"].astype(str)), "train/test overlap"
    ind = f.set_index(f.patient_id.astype(str)).individual_id.astype(str)
    assert not set(ind[train_ids]) & set(ind[np.concatenate(list(oof.values()))]), "individual leak train/OOF"

    model = build_model(spec.arch, cfg).to(data.device)
    tr_rows = data.rec_rows(train_ids)
    # --- normalisation fitted on this training set only (fixed segmentation), then frozen
    st, ow = data.fixed_segments(tr_rows)
    model.fit_norm(data.gather(st[i : i + 512], ow[i : i + 512]) for i in range(0, len(st), 512))

    y_rec = data.rec.rec_label.to_numpy()
    n_pos = max(int(y_rec[tr_rows].sum()), 1)
    n_neg = max(len(tr_rows) - n_pos, 1)
    pos_weight = torch.tensor(n_neg / n_pos if cfg["class_weighting"] == "pos_weight" else 1.0, device=data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    use_amp = cfg["amp"] and data.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    oof_ids = np.concatenate([oof[j] for j in spec.held_out])
    oof_prob = np.zeros((E, len(oof_ids)))
    test_prob = np.zeros((E, len(test_ids)))
    losses = []
    for ep in range(E):
        model.train()
        st, rows = data.random_crops(tr_rows, cfg["crops_per_recording"], gen)
        perm = gen.permutation(len(st))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i : i + cfg["batch_size"]]
            if len(b) < 2:
                continue
            x = data.gather(st[b], rows[b])
            y = torch.as_tensor(y_rec[rows[b]], dtype=torch.float32, device=data.device)
            with torch.autocast(data.device.type, dtype=torch.float16, enabled=use_amp):
                z = model(x)
            loss = F.binary_cross_entropy_with_logits(z.float(), y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            nb += 1
        losses.append(tot / max(nb, 1))
        oof_prob[ep] = score_encounters(model, data, oof_ids, cfg)
        if len(test_ids):
            test_prob[ep] = score_encounters(model, data, test_ids, cfg)
        log(f"{spec.run_id} ep {ep + 1}/{E} loss {losses[-1]:.4f} ({time.time() - t0:.0f}s)")

    np.savez_compressed(tmp_dir / "preds.npz", oof_ids=np.asarray(oof_ids, dtype=str), oof_heldout_fold=np.concatenate([[j] * len(oof[j]) for j in spec.held_out]),
                        oof_prob=oof_prob, test_ids=np.asarray(test_ids, dtype=str), test_prob=test_prob, train_loss=np.asarray(losses))
    torch.save(model.state_dict(), tmp_dir / "final.pt")
    manifest = {
        "run_id": spec.run_id, "partition": spec.partition, "outer": spec.outer, "held_out": list(spec.held_out),
        "arch": spec.arch, "member_seed": spec.seed, "torch_seed": seed, "epochs": E,
        "n_train_encounters": len(train_ids), "n_train_recordings": len(tr_rows), "pos_weight": float(pos_weight),
        "scored_test": bool(len(test_ids)), "code_hash": code_hash(), "wall_s": round(time.time() - t0, 1),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if data.device.type == "cuda" else "cpu",
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if (final_dir / "manifest.json").exists():
        return final_dir                      # finished elsewhere meanwhile; keep the first result
    if final_dir.exists():                    # an empty/partial final dir: move it aside, never delete
        os.replace(final_dir, out_root / f"{spec.run_id}.stale-{int(time.time() * 1000)}")
    os.replace(tmp_dir, final_dir)
    return final_dir
