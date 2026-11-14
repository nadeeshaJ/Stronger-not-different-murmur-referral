"""Build the ZCHSound manifest (no labels) and labels file for addendum E1.

    python external/prepare_zchsound.py
-> external/zchsound_manifest.csv  (unit_id, recording_id, path)        used by the scoring stage
-> external/zchsound_labels.csv    (unit_id, label, subset, diagnosis, age_days, age_group, sex)   analysis stage only
Label: CHD (ASD, VSD, PDA, PFO) = 1, NORMAL = 0. One recording per participant (unit = participant).
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "external" / "zchsound" / "raw" / "ZCHSound"
SUBSETS = {"clean": ("Clean Heartsound Data Details.csv", "clean Heartsound Data", ","),
           "noisy": ("Noise Heartsound Data Details.csv", "Noise Heartsound Data Details", ";")}


def age_group(days):
    # CirCor definitions: neonate < 28 days; infant 28 days to < 1 year; child 1 to < 12 years; adolescent 12 to < 18
    if pd.isna(days):
        return "Unknown"
    d = float(days)
    return "Neonate" if d < 28 else "Infant" if d < 365 else "Child" if d < 12 * 365 else "Adolescent"


def main():
    rows = []
    for subset, (csv, folder, sep) in SUBSETS.items():
        d = pd.read_csv(ROOT / csv, sep=sep, encoding="utf-8-sig")
        d.columns = [c.strip() for c in d.columns]
        for r in d.itertuples(index=False):
            f = str(getattr(r, "fileName")).strip()
            path = ROOT / folder / f
            dx = str(getattr(r, "diagnosis")).strip().upper()
            age = pd.to_numeric(getattr(r, "_2"), errors="coerce")       # 'age(days)'
            rows.append({"unit_id": Path(f).stem, "recording_id": Path(f).stem, "path": path.relative_to(REPO).as_posix(), "exists": path.exists(),
                         "label": int(dx != "NORMAL"), "subset": subset, "diagnosis": dx, "age_days": age,
                         "age_group": age_group(age), "sex": str(getattr(r, "gender")).strip()})
    df = pd.DataFrame(rows)
    missing = df[~df.exists]
    assert df.unit_id.is_unique
    out = REPO / "external"
    df[df.exists][["unit_id", "recording_id", "path"]].to_csv(out / "zchsound_manifest.csv", index=False, lineterminator="\n")
    df[["unit_id", "label", "subset", "diagnosis", "age_days", "age_group", "sex"]].to_csv(out / "zchsound_labels.csv", index=False, lineterminator="\n")
    print(f"{len(df)} units ({df.subset.value_counts().to_dict()}); {len(missing)} listed files missing")


if __name__ == "__main__":
    main()
