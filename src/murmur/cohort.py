"""Cohort construction.

Eligibility (stated explicitly so the protocol can cite it):
  1. Murmur label 'Unknown'  -> excluded.
  2. Missing age (Age == nan; mostly pregnant participants) -> excluded.
Unit of decision = encounter (one CirCor Patient ID row).
Unit of grouping/inference = individual: Patient IDs linked through 'Additional ID'
(the same child seen in both screening campaigns), joined transitively by union-find.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import dataset_dir, load_config

AGE_ORDER = ["Neonate", "Infant", "Child", "Adolescent"]


class _UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        root = a
        while self.p[root] != root:
            root = self.p[root]
        while self.p[a] != root:  # path compression
            self.p[a], a = root, self.p[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller id becomes root
            lo, hi = sorted([ra, rb])
            self.p[hi] = lo


def _clean_id(v) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s in ("", "nan"):
        return None
    return str(int(float(s)))


def load_raw(ds: Path | None = None) -> pd.DataFrame:
    ds = ds or dataset_dir()
    df = pd.read_csv(ds / "training_data.csv", dtype={"Patient ID": str})
    df["Patient ID"] = df["Patient ID"].astype(str).str.strip()
    return df


def parse_recordings(ds: Path, patient_id: str) -> list[dict]:
    """Read the per-patient header .txt: line 1 = '<pid> <n> <fs>', then n lines '<loc> <hea> <wav> <tsv>'."""
    lines = (ds / "training_data" / f"{patient_id}.txt").read_text(encoding="utf-8").splitlines()
    pid, n, fs = lines[0].split()
    assert pid == patient_id, (pid, patient_id)
    out = []
    for ln in lines[1 : 1 + int(n)]:
        loc, hea, wav, tsv = ln.split()
        out.append(
            {
                "patient_id": patient_id,
                "recording_id": Path(wav).stem,
                "location": loc,
                "wav": f"training_data/{wav}",
                "tsv": f"training_data/{tsv}",
                "fs": int(fs),
            }
        )
    return out


def build_cohort(ds: Path | None = None, cfg: dict | None = None):
    """Returns (encounters_df, recordings_df, exclusion_log)."""
    cfg = cfg or load_config()
    ds = ds or dataset_dir(cfg)
    raw = load_raw(ds)
    log = {"raw_encounters": len(raw)}

    elig = cfg["eligibility"]
    m_unknown = raw["Murmur"].isin(elig["exclude_murmur_labels"])
    log["excluded_murmur_unknown"] = int(m_unknown.sum())
    df = raw[~m_unknown]
    if elig["exclude_missing_age"]:
        m_age = df["Age"].isna() | (df["Age"].astype(str) == "nan")
        log["excluded_missing_age"] = int(m_age.sum())
        log["excluded_missing_age_pregnant"] = int((df.loc[m_age, "Pregnancy status"] == True).sum())  # noqa: E712
        df = df[~m_age]
    df = df.copy()

    # individuals via union-find over ALL raw rows (links may pass through excluded rows)
    ids = set(raw["Patient ID"])
    uf = _UnionFind(sorted(ids, key=int))
    for pid, add in zip(raw["Patient ID"], raw["Additional ID"]):
        a = _clean_id(add)
        if a is not None and a in ids:
            uf.union(pid, a)
    df["individual_id"] = [uf.find(p) for p in df["Patient ID"]]

    enc = pd.DataFrame(
        {
            "patient_id": df["Patient ID"].values,
            "individual_id": df["individual_id"].values,
            "murmur": df["Murmur"].values,
            "label": (df["Murmur"] == "Present").astype(int).values,
            "age_group": df["Age"].values,
            "sex": df["Sex"].values,
            "campaign": df["Campaign"].values,
            "outcome": df["Outcome"].values,
            "murmur_locations": df["Murmur locations"].fillna("").values,
        }
    ).sort_values("patient_id", key=lambda s: s.astype(int)).reset_index(drop=True)

    recs = []
    for pid in enc["patient_id"]:
        recs.extend(parse_recordings(ds, pid))
    rec = pd.DataFrame(recs)
    enc["n_recordings"] = enc["patient_id"].map(rec.groupby("patient_id").size())

    log.update(
        {
            "encounters": len(enc),
            "individuals": int(enc["individual_id"].nunique()),
            "recordings": len(rec),
            "murmur_present": int(enc["label"].sum()),
            "murmur_absent": int((1 - enc["label"]).sum()),
            "age_groups": {a: int((enc["age_group"] == a).sum()) for a in AGE_ORDER},
            "age_groups_present": {a: int(((enc["age_group"] == a) & (enc["label"] == 1)).sum()) for a in AGE_ORDER},
            "individuals_with_2plus_encounters": int((enc.groupby("individual_id").size() > 1).sum()),
        }
    )
    return enc, rec, log
