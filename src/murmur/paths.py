"""Path resolution. Set MURMUR_DATA_ROOT to the folder that contains the extracted CirCor directory;
otherwise <repo>/data/raw is used. Derived artefacts go to <repo>/data/derived unless MURMUR_DERIVED is set."""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_config(name: str = "phase1") -> dict:
    with open(REPO / "configs" / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def data_root() -> Path:
    return Path(os.environ.get("MURMUR_DATA_ROOT", REPO / "data" / "raw"))


def dataset_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return data_root() / cfg["dataset_dirname"]


def derived_dir() -> Path:
    d = Path(os.environ.get("MURMUR_DERIVED", REPO / "data" / "derived"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = Path(os.environ.get("MURMUR_CACHE", REPO / "data" / "cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d
