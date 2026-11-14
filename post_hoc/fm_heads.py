"""POST-HOC, EXPLORATORY (deviation 8): MLP heads on frozen AST embeddings, trained on exactly the
frozen nested training sets (375 = 5 partitions x 5 outer x 5 inner x 3 seeds). Writes
runs_fm/p{p}_o{o}_i{j}_ast_s{s}/{preds.npz, manifest.json} in the frozen preds format. Outer-test
labels are never read: test encounters are only scored."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F  # noqa: E402
from murmur.paths import cache_dir, derived_dir, load_config  # noqa: E402

HEAD = {"hidden": 128, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "batch": 256, "epochs": 20}
ARCH = "ast"


class Head(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main(out_root=None, only=None):
    t0 = time.time()
    tcfg, pc = load_config("train"), load_config("phase1")["partitions"]
    seeds = tcfg["member_seeds"]
    cd = cache_dir()
    out_root = Path(out_root or REPO / "runs_fm"); out_root.mkdir(exist_ok=True)
    folds = pd.read_csv(derived_dir() / "folds.csv", dtype={"patient_id": str, "individual_id": str})
    enc = pd.read_csv(derived_dir() / "cohort_encounters.csv", dtype={"patient_id": str, "individual_id": str})
    index = pd.read_csv(cd / "index.csv", dtype={"patient_id": str})
    seg = pd.read_csv(cd / "fm_ast_segments.csv")
    E = np.load(cd / "fm_ast_emb.npy").astype(np.float32)
    assert len(seg) == len(E)
    # location-level recording labels (frozen rule, murmur.train.DeviceData)
    e = enc.set_index("patient_id")
    locs, lab = e.murmur_locations.fillna("").astype(str), e.label.astype(int)
    rec_label = np.array([int(lab[p] == 1 and l in locs[p].split("+")) for p, l in zip(index.patient_id, index.location)])
    owner = seg.owner.to_numpy()
    seg_label = rec_label[owner].astype(np.float32)
    seg_patient = index.patient_id.to_numpy()[owner]
    segs_of_patient = pd.Series(np.arange(len(seg))).groupby(seg_patient).indices
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Et = torch.from_numpy(E).to(dev)

    def seg_idx(pids):
        return np.concatenate([segs_of_patient[str(p)] for p in pids])

    @torch.no_grad()
    def score(model, mu, sd, pids):
        si = seg_idx(pids)
        model.eval()
        p = torch.sigmoid(model((Et[si] - mu) / sd)).cpu().numpy().astype(np.float64)
        d = pd.DataFrame({"row": owner[si], "patient": seg_patient[si], "p": p})
        rec = d.groupby(["patient", "row"]).p.mean().reset_index()          # segment -> recording: mean
        encp = rec.groupby("patient").p.max()                                # recording -> encounter: max
        return encp.reindex([str(x) for x in pids]).to_numpy()

    n_done = 0
    for p in range(pc["n_partitions"]):
        for o in range(pc["n_outer"]):
            f = folds[(folds.partition == p) & (folds.outer_fold == o)]
            dv = f[f.role == "dev"]
            test_ids = f.loc[f.role == "test", "patient_id"].astype(str).to_numpy()
            for j in range(pc["n_inner"]):
                train_ids = dv.loc[dv.inner_fold != j, "patient_id"].astype(str).to_numpy()
                oof_ids = dv.loc[dv.inner_fold == j, "patient_id"].astype(str).to_numpy()
                ind = f.set_index(f.patient_id.astype(str)).individual_id.astype(str)
                assert not set(ind[train_ids]) & set(ind[oof_ids]), "individual leak train/OOF"
                assert not set(train_ids) & set(test_ids), "train/test overlap"
                tr = seg_idx(train_ids)
                Xtr = Et[tr]
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                Xtr = (Xtr - mu) / sd
                ytr = torch.from_numpy(seg_label[tr]).to(dev)
                pos_weight = torch.tensor(float((ytr == 0).sum() / max(float(ytr.sum()), 1.0)), device=dev)
                for s in seeds:
                    run_id = f"p{p}_o{o}_i{j}_{ARCH}_s{s}"
                    if only and run_id not in only:
                        continue
                    if (out_root / run_id / "manifest.json").exists():
                        n_done += 1
                        continue
                    seed = int(np.random.SeedSequence([20261113, p, o, j, s]).generate_state(1)[0])
                    torch.manual_seed(seed)
                    g = torch.Generator(device="cpu").manual_seed(seed)
                    model = Head(E.shape[1], HEAD["hidden"], HEAD["dropout"]).to(dev)
                    opt = torch.optim.AdamW(model.parameters(), lr=HEAD["lr"], weight_decay=HEAD["weight_decay"])
                    losses = []
                    for ep in range(HEAD["epochs"]):
                        model.train()
                        perm = torch.randperm(len(tr), generator=g).to(dev)
                        tot, nb = 0.0, 0
                        for i in range(0, len(perm), HEAD["batch"]):
                            b = perm[i:i + HEAD["batch"]]
                            if len(b) < 2:
                                continue
                            loss = F.binary_cross_entropy_with_logits(model(Xtr[b]), ytr[b], pos_weight=pos_weight)
                            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                            tot += float(loss.detach()); nb += 1
                        losses.append(tot / max(nb, 1))
                    oof_p = score(model, mu, sd, oof_ids)
                    test_p = score(model, mu, sd, test_ids)
                    assert np.isfinite(oof_p).all() and np.isfinite(test_p).all()
                    d = out_root / run_id; tmp = out_root / f"{run_id}.tmp"
                    tmp.mkdir(exist_ok=True)
                    np.savez_compressed(tmp / "preds.npz", oof_ids=np.asarray(oof_ids, str),
                                        oof_heldout_fold=np.full(len(oof_ids), j), oof_prob=oof_p[None, :],
                                        test_ids=np.asarray(test_ids, str), test_prob=test_p[None, :],
                                        train_loss=np.asarray(losses))
                    (tmp / "manifest.json").write_text(json.dumps({
                        "run_id": run_id, "post_hoc": True, "deviation": 8, "partition": p, "outer": o, "held_out": [j],
                        "arch": ARCH, "member_seed": s, "torch_seed": seed, "head": HEAD,
                        "n_train_encounters": len(train_ids), "n_train_segments": int(len(tr)),
                        "pos_weight": float(pos_weight), "final_loss": losses[-1],
                        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}, indent=2))
                    tmp.rename(d)
                    n_done += 1
            print(f"p{p} o{o} done: {n_done} heads ({time.time() - t0:.0f}s)", flush=True)
    print(f"finished {n_done} heads in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
