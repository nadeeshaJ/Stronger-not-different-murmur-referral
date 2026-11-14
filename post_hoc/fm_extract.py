"""POST-HOC, EXPLORATORY (deviation 8): frozen AST embeddings for every fixed 4-s segment of every
CirCor recording. Label-free. Writes data/cache/fm_ast_emb.npy (float16, n_segments x 768),
data/cache/fm_ast_segments.csv and data/cache/fm_ast_meta.json."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np, pandas as pd, torch  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from murmur.paths import cache_dir, load_config  # noqa: E402
from murmur.hashing import sha256_file  # noqa: E402

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
FS, FS_AST = 4000, 16000


def fixed_segments(offsets, lengths, L, H):
    """Same rule as murmur.train.DeviceData.fixed_segments."""
    starts, owner = [], []
    for r, (off, n) in enumerate(zip(offsets, lengths)):
        off, n = int(off), int(n)
        if n <= L:
            ss = [0]
        else:
            ss = list(range(0, n - L + 1, H))
            if ss[-1] + L < n:
                ss.append(n - L)
        starts += [off + s for s in ss]
        owner += [r] * len(ss)
    return np.asarray(starts, np.int64), np.asarray(owner, np.int64)


def gather(wave, starts, owner, offsets, lengths, L):
    x = np.zeros((len(starts), L), np.float32)
    for i, (s, r) in enumerate(zip(starts, owner)):
        end = int(offsets[r]) + int(lengths[r])
        n = max(0, min(L, end - int(s)))
        x[i, :n] = wave[s:s + n]
    return x


def main(model_id=MODEL_ID, batch=32, limit=None, model=None):
    from transformers import ASTFeatureExtractor, ASTModel
    import transformers
    t0 = time.time()
    tcfg = load_config("train")
    L, H = int(round(tcfg["segment_s"] * FS)), int(round(tcfg["hop_s"] * FS))
    cd = cache_dir()
    index = pd.read_csv(cd / "index.csv", dtype={"patient_id": str})
    wave = np.load(cd / "waveforms.npy", mmap_mode="r")
    off, ln = index.offset.to_numpy(), index.length.to_numpy()
    starts, owner = fixed_segments(off, ln, L, H)
    if limit:
        starts, owner = starts[:limit], owner[:limit]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = ASTFeatureExtractor.from_pretrained(model_id) if model is None else ASTFeatureExtractor()
    if model is None:
        model = ASTModel.from_pretrained(model_id)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    model = model.to(dev, dtype=dtype).eval()
    emb = np.zeros((len(starts), model.config.hidden_size), np.float16)
    print(f"{len(starts)} segments from {len(index)} recordings on {dev}", flush=True)
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            x = gather(wave, starts[i:i + batch], owner[i:i + batch], off, ln, L)
            x16 = resample_poly(x, FS_AST // FS, 1, axis=1).astype(np.float32)
            feats = fe(list(x16), sampling_rate=FS_AST, return_tensors="pt")["input_values"]
            out = model(input_values=feats.to(dev, dtype=dtype))
            emb[i:i + batch] = out.pooler_output.float().cpu().numpy()
            if (i // batch) % 100 == 0:
                print(f"  {i + len(x)}/{len(starts)} ({time.time() - t0:.0f}s)", flush=True)
    assert np.isfinite(emb.astype(np.float32)).all(), "non-finite embeddings"
    np.save(cd / "fm_ast_emb.npy", emb)
    pd.DataFrame({"start": starts, "owner": owner, "recording_id": index.recording_id.to_numpy()[owner]}).to_csv(
        cd / "fm_ast_segments.csv", index=False, lineterminator="\n")
    meta = {"post_hoc": True, "deviation": 8, "model_id": model_id, "transformers": transformers.__version__,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "dtype": str(dtype), "n_segments": int(len(starts)), "segment_s": tcfg["segment_s"], "hop_s": tcfg["hop_s"],
            "resample": "scipy.signal.resample_poly x4 (4 kHz -> 16 kHz)", "embedding": "ASTModel.pooler_output",
            "waveforms_sha256_expected": json.loads((cd / "meta.json").read_text())["waveforms_sha256"],
            "wall_s": round(time.time() - t0, 1),
            "outputs": {f: sha256_file(cd / f) for f in ("fm_ast_emb.npy", "fm_ast_segments.csv")}}
    (cd / "fm_ast_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
