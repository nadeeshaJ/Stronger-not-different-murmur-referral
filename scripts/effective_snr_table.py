"""Label-free: nominal vs effective corruption at the filtered-waveform input, on a sample of recordings.
Usage: python scripts/effective_snr_table.py [n_recordings]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from murmur.corrupt import CONDITIONS_DEFAULT, apply, condition_seed, effective_snr_db  # noqa: E402
from murmur.paths import dataset_dir, derived_dir, load_config  # noqa: E402
from murmur.preprocess import preprocess_signal, read_wav  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
cfg = load_config()
pp = cfg["preprocess"]
rec = pd.read_csv(derived_dir() / "cohort_recordings.csv", dtype={"patient_id": str})
rec = rec.sample(n=min(n, len(rec)), random_state=1)
rows = []
for r in rec.itertuples():
    x, fs = read_wav(dataset_dir() / r.wav)
    clean = preprocess_signal(x, fs, pp)
    for c in CONDITIONS_DEFAULT:
        y, info = apply(x, fs, c, condition_seed(r.recording_id, c["name"]))
        proc = preprocess_signal(y, fs, pp)
        row = {"recording_id": r.recording_id, **info, "effective_snr_db_waveform": effective_snr_db(clean, proc)}
        if c.get("type") == "clip":
            row["clip_fraction_processed"] = float(np.mean(np.abs(proc) >= 0.999 * np.max(np.abs(proc))))
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(derived_dir() / "effective_snr_sample.csv", index=False, lineterminator="\n")
s = df.replace([np.inf], np.nan).groupby("condition", sort=False)["effective_snr_db_waveform"].describe(percentiles=[0.1, 0.5, 0.9])
print(s[["count", "10%", "50%", "90%"]].round(1).to_string())
