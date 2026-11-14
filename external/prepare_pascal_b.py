"""Build the PASCAL CHSC2011 dataset B manifest (no labels) and labels file for addendum E1.

Source: the Kaggle copy 'kinguistics/heartbeat-sounds' (the official challenge site was unreachable on 2026-11-12).
Expected after unzipping to data/external/pascal/raw/: set_b.csv (dataset, fname, label, sublabel) and set_b/*.wav.
Labelled files only (unlabelled test files are excluded). Unit = recording (no patient IDs).
Label: murmur = 1; normal or extrastole = 0. Sensitivity analysis excludes extrastole.

    python external/prepare_pascal_b.py
-> external/pascal_b_manifest.csv (unit_id, recording_id, path) ; external/pascal_b_labels.csv (unit_id, label, class, noisy)
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "external" / "pascal" / "raw"


def main():
    meta = pd.read_csv(RAW / "set_b.csv")
    meta = meta[meta.label.notna()].copy()
    rows = []
    for r in meta.itertuples(index=False):
        rel = str(r.fname).strip()
        name = Path(rel).name
        # Kaggle's CSV and file names differ in a few known ways (e.g. 'Btraining_' prefix, 'noisy' sub-folders)
        cands = [RAW / rel, RAW / "set_b" / name, RAW / "set_b" / name.replace("Btraining_", "")]
        cands += list((RAW / "set_b").glob(f"*{Path(name).stem.split('_', 2)[-1]}*")) if (RAW / "set_b").exists() else []
        path = next((c for c in cands if c.exists()), None)
        cls = str(r.label).strip().lower()
        rows.append({"unit_id": Path(name).stem, "recording_id": Path(name).stem,
                     "path": path.relative_to(REPO).as_posix() if path else "", "found": path is not None,
                     "label": int(cls == "murmur"), "class": cls,
                     "noisy": int(str(getattr(r, "sublabel", "")).lower().startswith("noisy"))})
    df = pd.DataFrame(rows).drop_duplicates("unit_id")
    out = REPO / "external"
    df[df.found][["unit_id", "recording_id", "path"]].to_csv(out / "pascal_b_manifest.csv", index=False, lineterminator="\n")
    df[["unit_id", "label", "class", "noisy"]].to_csv(out / "pascal_b_labels.csv", index=False, lineterminator="\n")
    print(f"{len(df)} labelled recordings; {int((~df.found).sum())} not found on disk; classes {df['class'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
