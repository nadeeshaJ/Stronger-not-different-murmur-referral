"""H3 (exploratory): does mixing architectures help beyond member strength?

Prespecified primary contrast, per (partition p, outer o, seed s, primary P):
  heterogeneous pair  = the two non-primary architectures at seed s
  same-arch pair      = two seeds (s, (s+1) % 3) of the stronger of those two architectures (dev strength)
  strength            = pooled inner-OOF AUROC (development data only), per member and per epoch
  matching            = the weaker pair stays at the final epoch; the stronger pair's members move to earlier
                        checkpoints (e1, e2) chosen to satisfy |mean strength difference| <= margin, preferring the
                        latest checkpoints, then the smallest difference. route = "natural" if final epochs already match.
Ranking score = mean of the pair's calibrated probabilities (Platt per member and epoch on pooled OOF).
Capture of the primary's false negatives at the primary budget is compared on the same encounters.

Adjusted analysis: every pair and triple of the 8 non-primary members (final epoch) is used as a ranker; capture is
regressed on mean member strength, development error correlation and a mixed-architecture indicator.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .analysis import ARCHS, compare
from .calibration import Platt, sensitivity_threshold
from .referral import detection_vector, refer, summarise


def load_runs_all_epochs(root: Path) -> dict:
    out = {}
    for d in sorted(Path(root).glob("p*_o*_i*_*_s*")):
        if ".tmp" in d.name or ".stale" in d.name or not (d / "manifest.json").exists():
            continue
        p, o, i, *arch, s = d.name.split("_")
        z = np.load(d / "preds.npz")
        out[(int(p[1:]), int(o[1:]), int(i[1:]), "_".join(arch), int(s[1:]))] = {
            "oof_ids": z["oof_ids"].astype(str), "oof": z["oof_prob"], "test_ids": z["test_ids"].astype(str), "test": z["test_prob"]}
    return out


class EpochContext:
    """Per (p, o): pooled-OOF strength per member x epoch, and lazily calibrated scores."""

    def __init__(self, runs, p, o, n_inner, label_of, archs, seeds):
        self.runs, self.p, self.o, self.n_inner, self.archs, self.seeds = runs, p, o, n_inner, archs, seeds
        self.label_of = label_of
        self.oof_ids = np.concatenate([runs[(p, o, j, archs[0], seeds[0])]["oof_ids"] for j in range(n_inner)])
        self.y_oof = label_of[self.oof_ids].to_numpy()
        self.test_ids = runs[(p, o, 0, archs[0], seeds[0])]["test_ids"]
        self.strength, self._pl = {}, {}
        for a in archs:
            for s in seeds:
                P = np.concatenate([runs[(p, o, j, a, s)]["oof"] for j in range(n_inner)], axis=1)   # (E, n_oof)
                self.strength[(a, s)] = np.array([roc_auc_score(self.y_oof, P[e]) for e in range(P.shape[0])])

    def platt(self, a, s, e):
        if (a, s, e) not in self._pl:
            raw = np.concatenate([self.runs[(self.p, self.o, j, a, s)]["oof"][e] for j in range(self.n_inner)])
            self._pl[(a, s, e)] = Platt().fit(raw, self.y_oof)
        return self._pl[(a, s, e)]

    def cal(self, a, s, e, j, where):
        r = self.runs[(self.p, self.o, j, a, s)]
        return self.platt(a, s, e).transform(r["oof"][e] if where == "oof" else r["test"][e])

    def cal_oof_pooled(self, a, s, e):
        return np.concatenate([self.cal(a, s, e, j, "oof") for j in range(self.n_inner)])


def match_pairs(strength, het, hom, margin):
    """het/hom: lists of two (arch, seed) members. Returns epochs dict and route."""
    E = len(next(iter(strength.values())))
    last = E - 1
    m = lambda pair, e1, e2: (strength[pair[0]][e1] + strength[pair[1]][e2]) / 2
    m_het, m_hom = m(het, last, last), m(hom, last, last)
    if abs(m_het - m_hom) <= margin:
        return {het[0]: last, het[1]: last, hom[0]: last, hom[1]: last}, "natural", m_het, m_hom
    strong, weak = (het, hom) if m_het > m_hom else (hom, het)
    target = m(weak, last, last)
    S1, S2 = strength[strong[0]][:, None], strength[strong[1]][None, :]
    diff = np.abs((S1 + S2) / 2 - target)
    ok = diff <= margin
    if not ok.any():
        return None, "unmatched", m_het, m_hom
    e1g, e2g = np.meshgrid(np.arange(E), np.arange(E), indexing="ij")
    score = np.where(ok, np.minimum(e1g, e2g) * 1e6 + (e1g + e2g) * 1e3 - diff, -np.inf)
    e1, e2 = np.unravel_index(np.argmax(score), score.shape)
    ep = {weak[0]: last, weak[1]: last, strong[0]: int(e1), strong[1]: int(e2)}
    return ep, "checkpoint", m(het, ep[het[0]], ep[het[1]]), m(hom, ep[hom[0]], ep[hom[1]])


def h3_primary(runs, enc, n_part, n_outer, n_inner, seeds, acfg, margin=0.01, archs=ARCHS):
    label_of = enc.set_index("patient_id").label.astype(int)
    order = enc.patient_id.astype(str).to_numpy()
    pos_of = {pid: i for i, pid in enumerate(order)}
    reps = [(p, s, j) for p in range(n_part) for s in seeds for j in range(n_inner)]
    R = {r: k for k, r in enumerate(reps)}
    b, base = acfg["budget_primary"], acfg["budget_base"]
    arms = {}
    log = []
    E_last = None
    for p in range(n_part):
        for o in range(n_outer):
            c = EpochContext(runs, p, o, n_inner, label_of, archs, seeds)
            E_last = len(next(iter(c.strength.values()))) - 1
            idx = np.array([pos_of[t] for t in c.test_ids])
            y_test = label_of[c.test_ids].to_numpy()
            for plabel, P in acfg["primaries"].items():
                others = [a for a in archs if a != P]
                for s in seeds:
                    thr = sensitivity_threshold(c.cal_oof_pooled(P, s, E_last), c.y_oof, acfg["sensitivity_target"])
                    het = [(others[0], s), (others[1], s)]
                    hom_arch = max(others, key=lambda a: c.strength[(a, s)][-1])
                    hom = [(hom_arch, s), (hom_arch, (s + 1) % len(seeds))]
                    ep, route, m_het, m_hom = match_pairs(c.strength, het, hom, margin)
                    log.append({"partition": p, "outer": o, "seed": s, "primary": plabel, "route": route,
                                "het": f"{het[0][0]}+{het[1][0]}", "hom": f"{hom_arch}x2", "strength_het": m_het, "strength_hom": m_hom,
                                **({f"epoch:{a}_s{t}": e for (a, t), e in ep.items()} if ep else {})})
                    if ep is None:
                        continue
                    for j in range(n_inner):
                        k = R[(p, s, j)]
                        pp = c.cal(P, s, E_last, j, "test")
                        for name, pair in (("het", het), ("hom", hom)):
                            sc = np.mean([c.cal(a, t, ep[(a, t)], j, "test") for a, t in pair], 0)
                            res = refer(pp, thr, sc, b, base, seed=k)
                            key = (plabel, name)
                            if key not in arms:
                                arms[key] = np.full((len(reps), len(order)), np.nan)
                            arms[key][k, idx] = detection_vector(res.flagged, y_test)
    lg = pd.DataFrame(log)
    rows = []
    present = enc.label.to_numpy() == 1
    for plabel in acfg["primaries"]:
        if (plabel, "het") not in arms:
            continue
        A, B = arms[(plabel, "het")], arms[(plabel, "hom")]
        # unmatched (p, o, s) cells are NaN in both arms; keep encounters defined in at least one rep
        defined = ~np.isnan(A).all(0) | ~present
        e2 = enc[defined].reset_index(drop=True)
        sub = {(plabel, "het"): A[:, defined], (plabel, "hom"): B[:, defined]}
        r = compare(sub, (plabel, "het"), (plabel, "hom"), e2, n_boot=acfg["n_boot"], boot_seed=acfg["boot_seed"], prior=acfg["beta_prior"], test=False)   # exploratory: interval only
        g = lg[lg.primary == plabel]
        r.update({"analysis": f"H3-{plabel}", "matched_share": float((g.route != "unmatched").mean()),
                  "natural_share": float((g.route == "natural").mean()),
                  "mean_strength_het": g.strength_het.mean(), "mean_strength_hom": g.strength_hom.mean()})
        rows.append(r)
    return pd.DataFrame(rows), lg


def h3_adjusted(runs, enc, n_part, n_outer, n_inner, seeds, acfg, archs=ARCHS, n_boot=2000, seed=20261109):
    """Every pair/triple of non-primary members as ranker (final epoch). OLS of capture on
    mean strength, dev error correlation and mixed-architecture indicator; cluster bootstrap over (partition, seed)."""
    label_of = enc.set_index("patient_id").label.astype(int)
    b, base = acfg["budget_primary"], acfg["budget_base"]
    rows = []
    for p in range(n_part):
        for o in range(n_outer):
            c = EpochContext(runs, p, o, n_inner, label_of, archs, seeds)
            L = len(next(iter(c.strength.values()))) - 1
            y_test = label_of[c.test_ids].to_numpy()
            dev_err = {}
            for a in archs:
                for s in seeds:
                    q = c.cal_oof_pooled(a, s, L)
                    t = sensitivity_threshold(q, c.y_oof, acfg["sensitivity_target"])
                    dev_err[(a, s)] = ((q >= t) != (c.y_oof == 1)).astype(float)
            for plabel, P in acfg["primaries"].items():
                for s in seeds:
                    thr = sensitivity_threshold(c.cal_oof_pooled(P, s, L), c.y_oof, acfg["sensitivity_target"])
                    pool = [(a, t) for a in archs for t in seeds if (a, t) != (P, s)]
                    test_cal = {m: [c.cal(m[0], m[1], L, j, "test") for j in range(n_inner)] for m in pool}
                    prim = [c.cal(P, s, L, j, "test") for j in range(n_inner)]
                    for size in (2, 3):
                        for combo in combinations(pool, size):
                            corr = np.nanmean([np.corrcoef(dev_err[x], dev_err[y])[0, 1] for x, y in combinations(combo, 2)])
                            cap = np.mean([summarise(refer(prim[j], thr, np.mean([test_cal[m][j] for m in combo], 0), b, base, seed=j), y_test)["captured"]
                                           for j in range(n_inner)])
                            rows.append({"partition": p, "outer": o, "seed": s, "primary": plabel, "size": size,
                                         "members": "+".join(f"{a}_s{t}" for a, t in combo),
                                         "mixed": int(len({a for a, _ in combo}) > 1),
                                         "mean_strength": np.mean([c.strength[m][L] for m in combo]),
                                         "dev_error_corr": corr, "captured": cap})
    df = pd.DataFrame(rows)
    # sum capture over outer folds -> one row per (partition, seed, primary, combo)
    agg = df.groupby(["partition", "seed", "primary", "size", "members", "mixed"]).agg(
        captured=("captured", "sum"), mean_strength=("mean_strength", "mean"), dev_error_corr=("dev_error_corr", "mean")).reset_index()
    fits = []
    rng = np.random.default_rng(seed)
    for plabel, g in agg.groupby("primary"):
        def ols(d):
            X = np.column_stack([np.ones(len(d)), d.mean_strength, d.dev_error_corr, d.mixed, d["size"] == 3])
            beta, *_ = np.linalg.lstsq(X, d.captured.to_numpy(), rcond=None)
            return beta
        beta = ols(g)
        units = g[["partition", "seed"]].drop_duplicates().to_numpy()
        groups = {tuple(u): g[(g.partition == u[0]) & (g.seed == u[1])] for u in units}
        bs = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(units), len(units))
            bs.append(ols(pd.concat([groups[tuple(units[i])] for i in pick])))
        bs = np.array(bs)
        for i, name in enumerate(["intercept", "mean_strength", "dev_error_corr", "mixed_architecture", "triple"]):
            fits.append({"primary": plabel, "term": name, "coef": beta[i],
                         "ci95_lo": np.quantile(bs[:, i], 0.025), "ci95_hi": np.quantile(bs[:, i], 0.975)})
    return pd.DataFrame(fits), agg
