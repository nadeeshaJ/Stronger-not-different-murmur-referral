"""POST-HOC, EXPLORATORY (deviation 9): fine-tuned AST, reduced design (partition 0, seed 0, 25 models).
Usage:  python post_hoc/ast_finetune.py features | train | analyse
Plan: protocol/deviation9_ast_finetune.md. Outer-test labels are only read by `analyse`."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "post_hoc"))
import numpy as np, pandas as pd  # noqa: E402
from murmur.paths import cache_dir, derived_dir, load_config  # noqa: E402

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
FRAMES = 400
ARCH = "astft"
PARTITION, SEED = 0, 0
CFG = {"epochs": 5, "batch": 16, "lr_backbone": 2e-5, "lr_head": 1e-3, "weight_decay": 1e-4, "crops_per_recording": 4}
FEAT = "ast400_fbank.npy"


# ---------------------------------------------------------------- 1. features (label-free)
def features(model_id=MODEL_ID, limit=None, fe=None):
    from scipy.signal import resample_poly
    from transformers import ASTFeatureExtractor
    from fm_extract import fixed_segments, gather
    t0 = time.time()
    tcfg = load_config("train"); L, H = int(tcfg["segment_s"] * 4000), int(tcfg["hop_s"] * 4000)
    cd = cache_dir()
    index = pd.read_csv(cd / "index.csv", dtype={"patient_id": str})
    wave = np.load(cd / "waveforms.npy", mmap_mode="r")
    off, ln = index.offset.to_numpy(), index.length.to_numpy()
    starts, owner = fixed_segments(off, ln, L, H)
    if limit:
        starts, owner = starts[:limit], owner[:limit]
    fe = fe or ASTFeatureExtractor.from_pretrained(model_id, max_length=FRAMES)
    fe.max_length = FRAMES
    out = np.lib.format.open_memmap(cd / FEAT, mode="w+", dtype=np.float16, shape=(len(starts), FRAMES, 128))
    for i in range(0, len(starts), 256):
        x = gather(wave, starts[i:i + 256], owner[i:i + 256], off, ln, L)
        x16 = resample_poly(x, 4, 1, axis=1).astype(np.float32)
        out[i:i + len(x)] = fe(list(x16), sampling_rate=16000, return_tensors="np")["input_values"].astype(np.float16)
        if (i // 256) % 20 == 0:
            print(f"  features {i + len(x)}/{len(starts)} ({time.time() - t0:.0f}s)", flush=True)
    out.flush(); del out
    pd.DataFrame({"start": starts, "owner": owner}).to_csv(cd / "ast400_segments.csv", index=False, lineterminator="\n")
    print(f"features done: {len(starts)} segments in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 2. model
def build_model(model_id=MODEL_ID, base=None):
    """AST with the time axis of its position embeddings centre-cropped to FRAMES input frames."""
    import torch, torch.nn as nn
    from transformers import ASTConfig, ASTModel
    base = base or ASTModel.from_pretrained(model_id)
    c0 = base.config
    cfg = ASTConfig(**{**c0.to_dict(), "max_length": FRAMES})
    m = ASTModel(cfg)
    f_dim = (c0.num_mel_bins - c0.patch_size) // c0.frequency_stride + 1
    t_old = (c0.max_length - c0.patch_size) // c0.time_stride + 1
    t_new = (FRAMES - c0.patch_size) // c0.time_stride + 1
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    pe = sd["embeddings.position_embeddings"]                          # (1, 2 + f*t, d)
    grid = pe[:, 2:].reshape(1, f_dim, t_old, -1)
    s = t_old // 2 - t_new // 2
    sd["embeddings.position_embeddings"] = torch.cat([pe[:, :2], grid[:, :, s:s + t_new].reshape(1, f_dim * t_new, -1)], 1)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, missing
    head = nn.Sequential(nn.LayerNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, 1))
    return m, head


# ---------------------------------------------------------------- 3. train (25 runs)
def train(out_root=None, only=None, base=None, epochs=None, max_rows=None):
    import torch, torch.nn.functional as F
    t0 = time.time()
    pc = load_config("phase1")["partitions"]
    cd = cache_dir(); out_root = Path(out_root or REPO / "runs_ft"); out_root.mkdir(exist_ok=True)
    folds = pd.read_csv(derived_dir() / "folds.csv", dtype={"patient_id": str, "individual_id": str})
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    index = pd.read_csv(cd / "index.csv", dtype={"patient_id": str})
    seg = pd.read_csv(cd / "ast400_segments.csv")
    X = np.load(cd / FEAT, mmap_mode="r")
    e = enc.set_index("patient_id"); locs, lab = e.murmur_locations.fillna("").astype(str), e.label.astype(int)
    rec_label = np.array([int(lab[p] == 1 and l in locs[p].split("+")) for p, l in zip(index.patient_id, index.location)])
    owner = seg.owner.to_numpy()
    segs_of_row = pd.Series(np.arange(len(seg))).groupby(owner).indices
    rows_of_patient = pd.Series(np.arange(len(index))).groupby(index.patient_id.to_numpy()).indices
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"
    if base is None:
        from transformers import ASTModel
        base = ASTModel.from_pretrained(MODEL_ID)
    E = epochs or CFG["epochs"]

    def rows(pids):
        return np.concatenate([rows_of_patient[str(p)] for p in pids])

    @torch.no_grad()
    def score(m, head, pids):
        m.eval(); head.eval()
        r = rows(pids); si = np.concatenate([segs_of_row[x] for x in r])
        p = np.empty(len(si))
        for i in range(0, len(si), 64):
            xb = torch.from_numpy(np.asarray(X[si[i:i + 64]])).to(dev).float()
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                z = head(m(input_values=xb).pooler_output).squeeze(-1)
            p[i:i + 64] = torch.sigmoid(z.float()).cpu().numpy()
        d = pd.DataFrame({"row": owner[si], "patient": index.patient_id.to_numpy()[owner[si]], "p": p})
        encp = d.groupby(["patient", "row"]).p.mean().reset_index().groupby("patient").p.max()
        return encp.reindex([str(x) for x in pids]).to_numpy()

    f0 = folds[folds.partition == PARTITION]
    for o in range(pc["n_outer"]):
        f = f0[f0.outer_fold == o]; dv = f[f.role == "dev"]
        test_ids = f.loc[f.role == "test", "patient_id"].astype(str).to_numpy()
        for j in range(pc["n_inner"]):
            run_id = f"p{PARTITION}_o{o}_i{j}_{ARCH}_s{SEED}"
            if (only and run_id not in only) or (out_root / run_id / "manifest.json").exists():
                continue
            t1 = time.time()
            train_ids = dv.loc[dv.inner_fold != j, "patient_id"].astype(str).to_numpy()
            oof_ids = dv.loc[dv.inner_fold == j, "patient_id"].astype(str).to_numpy()
            ind = f.set_index(f.patient_id.astype(str)).individual_id.astype(str)
            assert not set(ind[train_ids]) & set(ind[oof_ids]) and not set(train_ids) & set(test_ids)
            tr = rows(train_ids)
            if max_rows:
                tr = tr[:max_rows]
            n_pos = max(int(rec_label[tr].sum()), 1)
            pos_weight = torch.tensor((len(tr) - n_pos) / n_pos, device=dev)
            seed = int(np.random.SeedSequence([20261114, PARTITION, o, j, SEED]).generate_state(1)[0])
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            m, head = build_model(base=base); m.to(dev); head.to(dev)
            opt = torch.optim.AdamW([{"params": m.parameters(), "lr": CFG["lr_backbone"]},
                                     {"params": head.parameters(), "lr": CFG["lr_head"]}], weight_decay=CFG["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            losses = []
            for ep in range(E):
                m.train(); head.train()
                pick = np.concatenate([rng.choice(segs_of_row[r], CFG["crops_per_recording"]) for r in tr])
                pick = pick[rng.permutation(len(pick))]
                tot, nb = 0.0, 0
                for i in range(0, len(pick), CFG["batch"]):
                    b = pick[i:i + CFG["batch"]]
                    if len(b) < 2:
                        continue
                    xb = torch.from_numpy(np.stack([X[k] for k in b])).to(dev).float()
                    yb = torch.as_tensor(rec_label[owner[b]], dtype=torch.float32, device=dev)
                    with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                        z = head(m(input_values=xb).pooler_output).squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(z.float(), yb, pos_weight=pos_weight)
                    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    tot += float(loss.detach()); nb += 1
                losses.append(tot / max(nb, 1))
                print(f"{run_id} epoch {ep + 1}/{E} loss {losses[-1]:.4f} ({time.time() - t1:.0f}s; total {time.time() - t0:.0f}s)", flush=True)
            oof_p, test_p = score(m, head, oof_ids), score(m, head, test_ids)
            assert np.isfinite(oof_p).all() and np.isfinite(test_p).all()
            tmp = out_root / f"{run_id}.tmp"; tmp.mkdir(exist_ok=True)
            np.savez_compressed(tmp / "preds.npz", oof_ids=np.asarray(oof_ids, str), oof_heldout_fold=np.full(len(oof_ids), j),
                                oof_prob=oof_p[None, :], test_ids=np.asarray(test_ids, str), test_prob=test_p[None, :],
                                train_loss=np.asarray(losses))
            (tmp / "manifest.json").write_text(json.dumps({
                "run_id": run_id, "post_hoc": True, "deviation": 9, "arch": ARCH, "partition": PARTITION, "outer": o,
                "held_out": [j], "member_seed": SEED, "torch_seed": seed, "cfg": CFG, "frames": FRAMES, "model_id": MODEL_ID,
                "n_train_recordings": int(len(tr)), "pos_weight": float(pos_weight), "losses": losses,
                "wall_s": round(time.time() - t1, 1),
                "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}, indent=2))
            tmp.rename(out_root / run_id)
            print(f"{run_id} done in {time.time() - t1:.0f}s", flush=True)
            del m, head, opt
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    print(f"training finished in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 4. analyse (reads outer-test labels)
def analyse(ft_root=None, out_dir=None):
    from sklearn.metrics import roc_auc_score
    from murmur import analysis as an, stats as st
    from murmur.referral import detection_vector, refer, standalone_top_m
    from murmur.hashing import sha256_file
    acfg, tcfg, pc = load_config("analysis"), load_config("train"), load_config("phase1")["partitions"]
    seeds, n_inner, n_outer = tcfg["member_seeds"], pc["n_inner"], pc["n_outer"]
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    label_of = enc.set_index("patient_id").label.astype(int)
    order = enc.patient_id.astype(str).to_numpy(); pos_of = {p: i for i, p in enumerate(order)}
    runs = an.load_runs(REPO / "runs", acfg["epoch"]); runs.update(an.load_runs(Path(ft_root or REPO / "runs_ft"), -1))
    sel = pd.read_csv(REPO / "results" / "studyC_selection.csv"); b = acfg["budget_primary"]
    sel = sel[np.isclose(sel.budget, b)].set_index(["partition", "outer", "seed", "primary"])
    p, s = PARTITION, SEED
    reps = [(p, s, j) for j in range(n_inner)]; R = {r: k for k, r in enumerate(reps)}
    arms, rows, auc_rows = {}, [], []

    def rec(pl, arm, k, idx, y, flagged, o, j):
        arms.setdefault((pl, arm), np.full((len(reps), len(order)), np.nan))[k, idx] = detection_vector(flagged, y)
        rows.append({"outer": o, "inner": j, "primary": pl, "arm": arm, "flags": int(flagged.sum()),
                     "detected": int((flagged & (y == 1)).sum()), "fp": int((flagged & (y == 0)).sum()),
                     "n_present": int(y.sum()), "n_absent": int((y == 0).sum())})
    primaries = {**acfg["primaries"], "ft": ARCH}
    for o in range(n_outer):
        c = an.build_context(runs, p, o, n_inner, an.ARCHS, seeds, label_of, acfg["sensitivity_target"])
        cf = an.build_context(runs, p, o, n_inner, [ARCH], [s], label_of, acfg["sensitivity_target"])
        for d in ("cal_test", "cal_oof", "thr", "dev_auc"):
            getattr(c, d).update(getattr(cf, d))
        idx = np.array([pos_of[t] for t in c.test_ids]); y = label_of[c.test_ids].to_numpy()
        for a in an.ARCHS + [ARCH]:
            for j in range(n_inner):
                auc_rows.append({"outer": o, "inner": j, "arch": a, "test_auroc": roc_auc_score(y, c.cal_test[(a, s, j)]),
                                 "dev_auroc": c.dev_auc[(a, s)]})
        for pl, prim in primaries.items():
            thr = c.thr[(prim, s)]
            for j in range(n_inner):
                k = R[(p, s, j)]; pp = c.cal_test[(prim, s, j)]
                crnn_ens = np.mean([c.cal_test[("crnn", s2, j)] for s2 in seeds], 0)
                ft = c.cal_test[(ARCH, s, j)]
                rec(pl, "own_threshold", k, idx, y, refer(pp, thr, pp, b, acfg["budget_base"], seed=k).flagged, o, j)
                if pl == "ft":
                    wf = refer(pp, thr, crnn_ens, b, acfg["budget_base"], seed=k)
                    rec(pl, "crnn_secondary", k, idx, y, wf.flagged, o, j)
                else:
                    chosen = sel.loc[(p, o, s, pl), "chosen"]
                    sc = an.candidate_scores(c, prim, s, j, "test", an.ARCHS, seeds)[chosen]
                    rec(pl, "frozen_secondary", k, idx, y, refer(pp, thr, sc, b, acfg["budget_base"], seed=k).flagged, o, j)
                    wf = refer(pp, thr, ft, b, acfg["budget_base"], seed=k)
                    rec(pl, "astft_secondary", k, idx, y, wf.flagged, o, j)
                mm = int(wf.flagged.sum())
                rec(pl, "crnn_ensemble_topM", k, idx, y, standalone_top_m(crnn_ens, mm, seed=k), o, j)
                rec(pl, "astft_topM", k, idx, y, standalone_top_m(ft, mm, seed=k), o, j)
    present = enc.label.to_numpy() == 1
    idx_b = st.bootstrap_indices(enc[present].individual_id.nunique(), acfg["n_boot"], acfg["boot_seed"])
    contrasts = [(pl, "astft_secondary", c_) for pl in ("weak", "strong") for c_ in ("own_threshold", "frozen_secondary", "crnn_ensemble_topM")]
    contrasts += [("ft", "crnn_secondary", c_) for c_ in ("own_threshold", "crnn_ensemble_topM", "astft_topM")]
    contrasts += [("ft", "astft_topM", "crnn_ensemble_topM")]
    out = []
    for pl, a_, b_ in contrasts:
        r = an.compare(arms, (pl, a_), (pl, b_), enc, idx_b, prior=acfg["beta_prior"], test=False)
        r.update({"primary": pl, "arm": a_, "comparator": b_}); out.append(r)
    out = pd.DataFrame(out)
    d = pd.DataFrame(rows)
    per = d.groupby(["primary", "arm", "inner"])[["flags", "detected", "fp", "n_present", "n_absent"]].sum()
    desc = per.groupby(["primary", "arm"]).mean().reset_index()
    desc["sensitivity"] = desc.detected / desc.n_present; desc["specificity"] = 1 - desc.fp / desc.n_absent
    auc = pd.DataFrame(auc_rows)
    auc_sum = auc.groupby("arch").agg(test_auroc_mean=("test_auroc", "mean"), test_auroc_sd=("test_auroc", "std"),
                                      dev_auroc_mean=("dev_auroc", "mean")).reset_index()
    piv = auc.pivot_table(index=["outer", "inner"], columns="arch", values="test_auroc")
    for a in an.ARCHS:
        dd = piv[ARCH] - piv[a]
        auc_sum.loc[len(auc_sum)] = {"arch": f"{ARCH}_minus_{a}", "test_auroc_mean": dd.mean(), "test_auroc_sd": dd.std(), "dev_auroc_mean": np.nan}
    od = Path(out_dir or REPO / "results" / "post_hoc"); od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / "ft_pilot_contrasts.csv", index=False, lineterminator="\n")
    desc.to_csv(od / "ft_pilot_descriptive.csv", index=False, lineterminator="\n")
    auc_sum.to_csv(od / "ft_pilot_auroc.csv", index=False, lineterminator="\n")
    (od / "ft_pilot_manifest.json").write_text(json.dumps({"post_hoc": True, "deviation": 9, "budget": b,
        "design": "partition 0, seed 0, 5 replicates",
        "plan_sha256": sha256_file(REPO / "protocol" / "deviation9_ast_finetune.md"),
        "outputs": {f.name: sha256_file(f) for f in sorted(od.glob("ft_pilot_*.csv"))}}, indent=2))
    pd.set_option("display.width", 200)
    print(auc_sum.round(4).to_string(index=False))
    print(out[["primary", "arm", "comparator", "estimate_murmurs", "ci95_lo", "ci95_hi"]].assign(
        lo=lambda x: x.ci95_lo * 174, hi=lambda x: x.ci95_hi * 174).round(3).to_string(index=False))
    print(desc[["primary", "arm", "flags", "detected", "sensitivity", "specificity"]].round(3).to_string(index=False))


if __name__ == "__main__":
    {"features": features, "train": train, "analyse": analyse}[sys.argv[1]]()
