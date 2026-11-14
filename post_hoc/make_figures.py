"""Publication figures from the frozen results (results/*.csv) and the post-hoc budget curve
(results/post_hoc/). Reads CSVs only; no outer-test labels are touched here.

    python post_hoc/make_figures.py            -> results/figures/*.png, *.pdf
"""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
OUT = R / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette: validated categorical slots 1-3 (all-pairs pass, light mode) + neutral greys.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
ARCH_C = {"wave_cnn": BLUE, "spec_cnn": ORANGE, "crnn": AQUA}
ARCH_M = {"wave_cnn": "o", "spec_cnn": "s", "crnn": "^"}
ARCH_L = {"wave_cnn": "Waveform CNN (weak primary)", "spec_cnn": "Spectrogram CNN", "crnn": "CRNN (strong primary)"}
ARCH_S = {"wave_cnn": "Waveform CNN", "spec_cnn": "Spectrogram CNN", "crnn": "CRNN"}
PRIM_L = {"weak": "Weak primary (waveform CNN)", "strong": "Strong primary (CRNN)"}
N_PRESENT, N_ENC = 174, 802

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
    "lines.markersize": 4.5, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def ci(s):
    """Parse '(lo, hi)' strings written by the analysis (handles np.float64(...))."""
    if isinstance(s, str):
        s = s.replace("np.float64(", "").replace(")", "").replace("(", "")
        a, b = s.split(",")
        return float(a), float(b)
    return np.nan, np.nan


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def panel_label(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="left")


# ------------------------------------------------------------------ Fig 1 workflow schematic
def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(0, 100); ax.set_ylim(-9.5, 38); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc="#f3f2ef", ec=MUTED, bold=False, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.8,
                fontweight="normal", color=color, linespacing=1.2)

    def arrow(x0, y0, x1, y1, text=None, dy=1.2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6.5, color=INK2)

    box(0.5, 13, 15, 10, "Encounter\n(up to 4\nrecordings)")
    box(20, 12, 18, 12, "Primary model\n(calibrated;\nthreshold at 90%\ndev sensitivity)", fc="#e8f0fb", ec=BLUE, bold=True)
    arrow(15.8, 18, 19.6, 18)
    box(44, 25.5, 17, 7.5, "Predicted positive\n→ flagged", fc="#f3f2ef")
    box(44, 3, 17, 7.5, "Predicted negative\n(holds the misses)", fc="#f3f2ef")
    arrow(38.4, 21, 43.6, 29)
    arrow(38.4, 15, 43.6, 7)
    box(64.5, 0.5, 18.5, 12.5, "Second model ranks\nthe negatives;\nrefer the top\nK = ⌈b × neg.⌉", fc="#fdeee7", ec=ORANGE, bold=True)
    arrow(61.4, 6.75, 64.1, 6.75)
    box(85.5, 13, 14, 10, "Human review\n(total flags =\npositives + K)", fc="#e7f6f0", ec=AQUA, bold=True)
    arrow(61.4, 29, 85.1, 21.5)
    arrow(83.4, 8, 85.1, 14.5)
    ax.text(0.5, 37.8, "Two-stage AI second-opinion workflow (budget b = 10% confirmatory; 5% and 20% secondary)",
            ha="left", va="top", fontsize=7.8, fontweight="bold")
    ax.text(0.5, -9.0, "Comparators at the same workload.  H1: refer the primary's own next-most-likely negatives (threshold lowering).\n"
            "H2: the strongest standalone model flags its top-M encounters, M = the workflow's total flags.", fontsize=6.6, color=INK2, va="bottom")
    save(fig, "fig1_workflow")


# ------------------------------------------------------------------ Fig 2 Study A
def fig_study_a():
    perf = pd.read_csv(R / "studyA_performance.csv")
    rel = pd.read_csv(R / "studyA_reliability.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.05, 1, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for a in ["wave_cnn", "spec_cnn", "crnn"]:
        d = rel[rel.arch == a]
        ax.plot(d.mean_pred, d.observed, color=ARCH_C[a], marker=ARCH_M[a], ms=3.6, lw=1.3, label=ARCH_S[a],
                mec="white", mew=0.5)
    ax.set_xlabel("Mean predicted probability (calibrated)"); ax.set_ylabel("Observed murmur rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title("Calibration (outer test)")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7, handlelength=1.8)
    panel_label(ax, "a")

    for k, (metric, cicol, title) in enumerate([("auroc", "auroc_ci95", "AUROC"),
                                                  ("specificity", "spec_ci95", "Specificity at 90% dev-sensitivity threshold")]):
        ax = axes[k + 1]
        for i, a in enumerate(["wave_cnn", "spec_cnn", "crnn"]):
            r = perf[perf.arch == a].iloc[0]
            lo, hi = ci(r[cicol])
            m = r[f"{metric}_mean"]
            ax.errorbar(m, 2 - i, xerr=[[m - lo], [hi - m]], fmt=ARCH_M[a], color=ARCH_C[a], ms=5.5, capsize=2, lw=1.1,
                        mec="white", mew=0.6)
            ax.text(hi + (0.004 if metric == "auroc" else 0.012), 2 - i, f"{m:.3f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks([2, 1, 0]); ax.set_yticklabels([ARCH_S[a] for a in ["wave_cnn", "spec_cnn", "crnn"]] if k == 0 else [])
        ax.set_ylim(-0.6, 2.6); ax.set_title(title, loc="left")
        ax.set_xlim((0.87, 0.97) if metric == "auroc" else (0.58, 0.84))
        if metric == "auroc":
            ax.set_xticks([0.88, 0.92, 0.96])
        ax.grid(axis="y", visible=False)
        panel_label(ax, "bc"[k])
    fig.text(0.5, -0.04, "Means over 75 replicates; 95% CIs by individual-level resampling. Sensitivity at the transferred threshold: "
             "0.903 / 0.901 / 0.899.", ha="center", fontsize=6.6, color=INK2)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig2_studyA_baselines")


# ------------------------------------------------------------------ Fig 3 budget curve
def fig_budget():
    desc = pd.read_csv(R / "post_hoc" / "budget_curve_descriptive.csv")
    con = pd.read_csv(R / "post_hoc" / "budget_curve_contrasts.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.6), sharex=True)
    for j, pl in enumerate(["weak", "strong"]):
        d = desc[desc.primary == pl]
        ax = axes[0, j]
        piv = d.pivot(index="budget", columns="arm", values="captured")
        orc = d[d.arm == "secondary"].set_index("budget").oracle_ceiling
        x = piv.index * 100
        ax.fill_between(x, 0, orc.values, color="#f1f0ec", lw=0, zorder=0)
        ax.plot(x, orc.values, color=MUTED, lw=1.0, ls=(0, (3, 2)), label="Oracle ceiling (all eligible misses)")
        ax.plot(x, piv["secondary"], color=BLUE, marker="o", ms=3.5, label="Second-model ranking (selected on dev)", mec="white", mew=0.5)
        ax.plot(x, piv["own_threshold"], color=ORANGE, marker="s", ms=3.3, label="Primary's own next-most-likely (H1 comparator)", mec="white", mew=0.5)
        ax.plot(x, piv["random"], color=INK2, marker="D", ms=2.8, lw=1.0, label="Random referral", mec="white", mew=0.5)
        ax.axvline(10, color=INK2, lw=0.6, zorder=0)
        ax.text(10.5, 17.9, "registered b = 10%", fontsize=6.2, color=INK2, va="bottom")
        ax.set_title(PRIM_L[pl], loc="left", fontweight="bold")
        ax.set_ylabel("Primary's missed murmurs recovered\n(mean per replicate, of 174)" if j == 0 else "")
        ax.set_ylim(0, 19.5)
        panel_label(ax, "ab"[j])
        if j == 0:
            fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                       fontsize=6.6, handlelength=1.8)

        ax = axes[1, j]
        for name, col, mk, lab in (("H1", ORANGE, "s", "vs own-threshold referral (H1 contrast)"),
                                    ("H2", AQUA, "^", "vs strongest standalone at equal total flags (H2 contrast)")):
            c = con[(con.primary == pl) & (con.contrast == name)].sort_values("budget")
            xb = c.budget * 100
            ax.fill_between(xb, c.ci95_lo * N_PRESENT, c.ci95_hi * N_PRESENT, color=col, alpha=0.16, lw=0)
            ax.plot(xb, c.estimate_murmurs, color=col, marker=mk, ms=3.5, label=lab, mec="white", mew=0.5)
        ax.axhline(0, color=INK, lw=0.7)
        ax.axvline(10, color=INK2, lw=0.6, zorder=0)
        ax.set_xlabel("Review budget b (% of predicted negatives)")
        ax.set_ylabel("Extra murmurs detected by the\nsecond-opinion workflow (95% CI)" if j == 0 else "")
        ax.set_ylim(min(-4.0, con.ci95_lo.min() * N_PRESENT - 0.3), 6.6)
        panel_label(ax, "cd"[j])
        if j == 1:
            fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.005), ncol=1,
                       fontsize=6.6, handlelength=1.8)
    fig.text(0.5, -0.115, "Budget curve registered in the protocol (§6), computed after the frozen analysis with the frozen Study C code\n"
             "(deviation 3). Ribbons: individual-clustered bootstrap 95% CIs; descriptive, no tests.", ha="center", va="top", fontsize=6.5, color=INK2)
    fig.tight_layout(h_pad=1.4, w_pad=2.0)
    save(fig, "fig3_budget_curve")


# ------------------------------------------------------------------ Fig 4 workload frontier
def fig_frontier():
    desc = pd.read_csv(R / "post_hoc" / "budget_curve_descriptive.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    styles = {("weak", "secondary"): (BLUE, "o", "-", "Weak primary + second opinion"),
              ("weak", "own_threshold"): (BLUE, "o", (0, (3, 2)), "Weak primary, threshold lowered"),
              ("strong", "secondary"): (AQUA, "^", "-", "Strong primary + second opinion"),
              ("strong", "own_threshold"): (AQUA, "^", (0, (3, 2)), "Strong primary, threshold lowered")}
    for pl, mk in (("weak", "o"), ("strong", "^")):
        sa = desc[(desc.primary == pl) & (desc.arm == "standalone_topM")].sort_values("total_flags")
        ax.plot(sa.total_flags / N_ENC * 100, sa.sensitivity_pooled * 100, color=ORANGE, marker="s", ms=3.0, lw=1.1,
                label="Strongest standalone, top-M at the same flags" if pl == "weak" else None, mec="white", mew=0.5, zorder=2)
    for (pl, arm), (col, mk, ls, lab) in styles.items():
        d = desc[(desc.primary == pl) & (desc.arm == arm)].sort_values("total_flags")
        ax.plot(d.total_flags / N_ENC * 100, d.sensitivity_pooled * 100, color=col, marker=mk, ls=ls, ms=3.4 if ls == "-" else 2.6,
                lw=1.4 if ls == "-" else 1.0, label=lab, mec="white", mew=0.5, zorder=3)
        if arm == "secondary":
            r = d[np.isclose(d.budget, 0.10)].iloc[0]
            ax.scatter([r.total_flags / N_ENC * 100], [r.sensitivity_pooled * 100], s=70, facecolor="none", edgecolor=INK, lw=0.9, zorder=4)
    ax.text(41.2, 94.0, "b = 10%", fontsize=6.3, color=INK2)
    ax.annotate("b = 10%", xy=(54.0, 94.3), xytext=(56.2, 93.95), fontsize=6.3, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.set_xlabel("Total encounters flagged for review (% of 802)")
    ax.set_ylabel("Murmur-present encounters detected (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.3, handlelength=2.2)
    ax.set_ylim(89.5, 97); ax.set_xlim(36, 65)
    ax.set_title("Detection vs total review workload (budgets 1–30%)", loc="left")
    save(fig, "fig4_workload_frontier")


# ------------------------------------------------------------------ Fig 5 forest
def fig_forest():
    conf = pd.read_csv(R / "studyC_confirmatory.csv")
    sec = pd.read_csv(R / "studyC_secondary.csv")
    rows = []
    for _, r in conf.iterrows():
        rows.append((f"{r.hypothesis}  (b = 10%)", r.estimate_murmurs, r.ci95_lo * N_PRESENT, r.ci95_hi * N_PRESENT, True, r.p_holm))
    for _, r in sec.iterrows():
        rows.append((r.analysis.replace(" budget ", "  b = ").replace("0.05", "5%").replace("0.2", "20%")
                     .replace(" reviewer ", "  reviewer sens. "), r.estimate_murmurs, r.ci95_lo * N_PRESENT, r.ci95_hi * N_PRESENT, False, np.nan))
    order = ["H1-weak  (b = 10%)", "H1-weak  b = 5%", "H1-weak  b = 20%", "H1-weak  reviewer sens. 0.8", "H1-weak  reviewer sens. 0.9",
             "H2-weak  (b = 10%)", "H2-weak  b = 5%", "H2-weak  b = 20%",
             "H1-strong  (b = 10%)", "H1-strong  b = 5%", "H1-strong  b = 20%", "H1-strong  reviewer sens. 0.8", "H1-strong  reviewer sens. 0.9",
             "H2-strong  (b = 10%)", "H2-strong  b = 5%", "H2-strong  b = 20%"]
    rd = {r[0]: r for r in rows}
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    y = len(order)
    yt, yl = [], []
    for i, k in enumerate(order):
        lab, est, lo, hi, confirm, ph = rd[k]
        yy = y - i - (0.6 if i >= 8 else 0)
        col = BLUE if "weak" in k else AQUA
        ax.errorbar(est, yy, xerr=[[est - lo], [hi - est]], fmt="D" if confirm else "o", color=col if confirm else INK2,
                    ms=6 if confirm else 3.6, capsize=0, lw=1.6 if confirm else 0.9, mec="white", mew=0.6, zorder=3)
        txt = f"{est:+.1f} ({lo:+.1f} to {hi:+.1f})" + (f"   Holm p {'< 0.001' if ph < 0.001 else f'= {ph:.3f}'}" if confirm else "")
        ax.text(6.9, yy, txt, va="center", fontsize=6.6, color=INK if confirm else INK2, fontweight="bold" if confirm else "normal")
        yt.append(yy); yl.append(lab)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.8)
    for t, k in zip(ax.get_yticklabels(), order):
        if "(b = 10%)" in k:
            t.set_fontweight("bold")
    ax.set_xlim(-2, 6.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Extra murmur-present encounters detected (of 174), 95% CI")
    ax.set_title("Study C: confirmatory (bold, Holm-adjusted) and secondary (unadjusted) contrasts", loc="left", fontsize=7.8)
    fig.text(0.02, -0.02, "H1: second-model ranking vs primary's own ranking, same K.  H2: two-stage workflow vs strongest standalone, identical total flags.",
            fontsize=6.0, color=INK2, transform=fig.transFigure)
    ax.set_ylim(-0.6, y + 0.8)
    save(fig, "fig5_studyC_forest")


# ------------------------------------------------------------------ Fig 6 H3
def fig_h3():
    comb = pd.read_csv(R / "H3_combinations.csv")
    h3p = pd.read_csv(R / "H3_primary.csv")
    adj = pd.read_csv(R / "H3_adjusted.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=False)
    for j, pl in enumerate(["weak", "strong"]):
        ax = axes[j]
        d = comb[(comb.primary == pl) & (comb["size"] == 2)]
        for mixed, col, mk, lab in ((0, ORANGE, "s", "Same-architecture pair"), (1, BLUE, "o", "Mixed-architecture pair")):
            dd = d[d.mixed == mixed]
            ax.scatter(dd.mean_strength, dd.captured, s=9, color=col, marker=mk, alpha=0.55, lw=0, label=lab)
        r = h3p[h3p.analysis == f"H3-{pl}"].iloc[0]
        a = adj[adj.primary == pl].set_index("term")
        ax.set_title(PRIM_L[pl], loc="left", fontweight="bold")
        ax.set_xlabel("Mean member strength (pooled inner-OOF AUROC)")
        ax.set_ylabel("Primary's misses recovered at b = 10%\n(per partition × seed)" if j == 0 else "")
        ms, mx = a.loc["mean_strength"], a.loc["mixed_architecture"]
        txt = (f"Strength-matched het − hom: {r.estimate_murmurs:+.2f} ({r.ci95_lo*N_PRESENT:+.2f} to {r.ci95_hi*N_PRESENT:+.2f})\n"
               f"Adjusted: +0.01 AUROC → {ms.coef*0.01:+.2f} ({ms.ci95_lo*0.01:+.2f} to {ms.ci95_hi*0.01:+.2f});\n"
               f"mixed architecture {mx.coef:+.2f} ({mx.ci95_lo:+.2f} to {mx.ci95_hi:+.2f})")
        ax.text(0.0, -0.24, txt, transform=ax.transAxes, va="top", fontsize=6.3, color=INK2)
        panel_label(ax, "ab"[j])
        if j == 1:
            fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                       markerscale=1.6, fontsize=6.8)
    fig.text(0.5, 0.0, "Each point: one two-member secondary (pairs only shown; triples in the adjusted model). Units: murmurs of 174.",
             ha="center", fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_H3_diversity_vs_strength")


# ------------------------------------------------------------------ Fig 7 Study B
COND_ORDER = ["identity", "white_20dB", "white_10dB", "white_0dB", "pink_20dB", "pink_10dB", "pink_0dB",
              "burst_2per10s", "burst_5per10s", "clip_p99.9", "clip_p99.5", "clip_p99", "dropout_5pct", "dropout_10pct", "dropout_20pct"]
COND_L = {"identity": "Clean (identity)", "white_20dB": "White noise 20 dB", "white_10dB": "White noise 10 dB", "white_0dB": "White noise 0 dB",
          "pink_20dB": "Pink noise 20 dB", "pink_10dB": "Pink noise 10 dB", "pink_0dB": "Pink noise 0 dB",
          "burst_2per10s": "Bursts 2 / 10 s", "burst_5per10s": "Bursts 5 / 10 s", "clip_p99.9": "Clipping p99.9",
          "clip_p99.5": "Clipping p99.5", "clip_p99": "Clipping p99", "dropout_5pct": "Dropout 5%",
          "dropout_10pct": "Dropout 10%", "dropout_20pct": "Dropout 20%"}


def fig_study_b():
    rob = pd.read_csv(R / "studyB_robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=True)
    ys = {c: len(COND_ORDER) - i - (0.5 if i >= 1 else 0) - (0.5 if i >= 7 else 0) - (0.5 if i >= 9 else 0) - (0.5 if i >= 12 else 0)
          for i, c in enumerate(COND_ORDER)}
    off = {"wave_cnn": 0.22, "spec_cnn": 0.0, "crnn": -0.22}
    for k, (metric, title) in enumerate([("sensitivity", "Sensitivity"), ("specificity", "Specificity")]):
        ax = axes[k]
        for a in ["wave_cnn", "spec_cnn", "crnn"]:
            d = rob[rob.arch == a]
            clean = d[d.condition == "identity"][metric].iloc[0]
            ax.axvline(clean, color=ARCH_C[a], lw=0.6, alpha=0.6, zorder=0)
            yy = [ys[c] + off[a] for c in d.condition]
            ax.scatter(d[metric], yy, color=ARCH_C[a], marker=ARCH_M[a], s=16, label=ARCH_S[a], edgecolor="white", lw=0.4, zorder=3)
        ax.set_title(title + " at the frozen clean threshold", loc="left")
        ax.set_xlim(0 if metric == "specificity" else 0.7, 1.0)
        ax.grid(axis="y", visible=False)
        panel_label(ax, "ab"[k])
    axes[0].set_yticks([ys[c] for c in COND_ORDER]); axes[0].set_yticklabels([COND_L[c] for c in COND_ORDER])
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.55, 1.05))
    fig.text(0.5, -0.03, "Frozen models, calibrators and thresholds; corruption applied to raw audio. Vertical lines: clean values. "
             "Means over 75 replicates.", ha="center", fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig7_studyB_operating_point")


def fig_study_b_snr():
    rob = pd.read_csv(R / "studyB_robustness.csv")
    rob = rob[rob.condition.str.startswith(("white", "pink"))]
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for a in ["wave_cnn", "spec_cnn", "crnn"]:
        d = rob[rob.arch == a]
        ax.scatter(d.snr_filtered_median, d.snr_input_median, color=ARCH_C[a], marker=ARCH_M[a], s=20, label=ARCH_S[a] + " input",
                   edgecolor="white", lw=0.4, zorder=3)
    lim = (-7, 27)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(18, 19.7, "input = band-passed", fontsize=6.2, color=INK2, rotation=33)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Effective SNR after 20–1000 Hz band-pass (dB, median)")
    ax.set_ylabel("Effective SNR or perturbation ratio\nat model input (dB, median)")
    ax.set_title("White and pink noise (nominal 0 / 10 / 20 dB)", loc="left")
    ax.legend(loc="upper left", fontsize=6.4)
    save(fig, "figS1_effective_snr")


def fig_study_b_referral():
    ref = pd.read_csv(R / "studyB_referral.csv")
    conds = [c for c in COND_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.9), sharey=True)
    ys = {c: len(conds) - i for i, c in enumerate(conds)}
    for j, pl in enumerate(["weak", "strong"]):
        ax = axes[j]
        d = ref[ref.primary == pl].set_index("condition")
        for c in conds:
            r = d.loc[c]
            lo1, hi1 = ci(r.H1_ci95); lo2, hi2 = ci(r.H2_ci95)
            ax.errorbar(r.H1_estimate_murmurs, ys[c] + 0.15, xerr=[[r.H1_estimate_murmurs - lo1], [hi1 - r.H1_estimate_murmurs]],
                        fmt="s", color=ORANGE, ms=3.6, lw=0.9, mec="white", mew=0.4, label="H1 contrast" if c == "identity" else None)
            ax.errorbar(r.H2_estimate_murmurs, ys[c] - 0.15, xerr=[[r.H2_estimate_murmurs - lo2], [hi2 - r.H2_estimate_murmurs]],
                        fmt="^", color=AQUA, ms=3.8, lw=0.9, mec="white", mew=0.4, label="H2 contrast" if c == "identity" else None)
            ax.text(8.3, ys[c], f"{r.eligible_fn:.0f}", fontsize=6.3, va="center", ha="right", color=INK2)
        ax.text(8.3, len(conds) + 1.05, "eligible misses", fontsize=6.0, va="center", ha="right", color=INK2)
        ax.set_ylim(0.4, len(conds) + 1.5)
        ax.axvline(0, color=INK, lw=0.7)
        ax.set_xlim(-2.6, 8.5); ax.grid(axis="y", visible=False)
        ax.set_title(PRIM_L[pl], loc="left", fontweight="bold")
        ax.set_xlabel("Extra murmurs detected (of 174), 95% CI")
        panel_label(ax, "ab"[j])
    axes[0].set_yticks([ys[c] for c in conds]); axes[0].set_yticklabels([COND_L[c] for c in conds])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.55, 1.05))
    fig.text(0.5, -0.03, "Exploratory, unadjusted. Frozen secondary choices and thresholds reused under each corruption; b = 10%.",
             ha="center", fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig8_studyB_referral")


# ------------------------------------------------------------------ Fig 9 age
def fig_age():
    age = pd.read_csv(R / "age_strata.csv")
    groups = ["Infant", "Child", "Adolescent"]
    arms = [("model:wave_cnn", "Waveform CNN alone", BLUE, "o", "none"),
            ("C:weak:secondary", "Waveform CNN + second opinion", BLUE, "o", "full"),
            ("model:crnn", "CRNN alone", AQUA, "^", "none"),
            ("C:strong:secondary", "CRNN + second opinion", AQUA, "^", "full")]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for gi, g in enumerate(groups):
        for ai, (arm, lab, col, mk, fill) in enumerate(arms):
            r = age[(age.arm == arm) & (age.age_group == g)].iloc[0]
            lo, hi = ci(r.sens_ci95)
            x = gi + (ai - 1.5) * 0.17
            ax.errorbar(x, r.sensitivity * 100, yerr=[[(r.sensitivity - lo) * 100], [(hi - r.sensitivity) * 100]], fmt=mk,
                        color=col, mfc=col if fill == "full" else "white", mec=col, ms=5, lw=1.0, capsize=0,
                        label=lab if gi == 0 else None)
    n = age[age.arm == "model:crnn"].set_index("age_group")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{g}\n{int(n.loc[g].n_present)} present / {int(n.loc[g].n_absent)} absent" for g in groups], fontsize=6.8)
    ax.set_ylabel("Sensitivity (%), 95% CI"); ax.set_ylim(60, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.3, ncol=2)
    ax.grid(axis="x", visible=False)
    ax.set_title("Detection by age group (neonates: n = 1 present, not shown)", loc="left")
    save(fig, "fig9_age")


if __name__ == "__main__":
    fig_workflow(); fig_study_a(); fig_budget(); fig_frontier(); fig_forest(); fig_h3()
    fig_study_b(); fig_study_b_snr(); fig_study_b_referral(); fig_age()
