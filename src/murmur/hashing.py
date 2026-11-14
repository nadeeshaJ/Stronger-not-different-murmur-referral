from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_sha256sums(dataset_dir: Path) -> tuple[int, list[str]]:
    """Verify every entry in CirCor's SHA256SUMS.txt. Returns (n_ok, failures)."""
    ok, bad = 0, []
    with open(dataset_dir / "SHA256SUMS.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(maxsplit=1)
            p = dataset_dir / name
            if p.exists() and sha256_file(p) == digest:
                ok += 1
            else:
                bad.append(name)
    return ok, bad
