"""Generate all Phase-3 paper figures + table data from REAL artifacts.

Reads results/exp_11_analysis.json (verdicts + probe), the paired matrices via
experiments/exp_11_analysis.load_log_regret(), and the Part-1 artifacts exp_05_tier1a.json /
exp_07_tier1b.json. Produces Figures 1-4 (PDF) and prints every table's numbers so the LaTeX
cites only real values. NOTHING is fabricated; every number traces to an artifact.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import exp_11_analysis as A  # noqa: E402

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "pdf.fonttype": 42})

R = json.load(open(ROOT / "results" / "exp_11_analysis.json"))
L, probe, ncells, nclip = A.load_log_regret()
ARMS = ["EI", "LogEI", "UCB", "Exploit", "eps-PF"]
SH = A.H2_SHARED  # 10 De Ath problems
HI = A.H2_HIGHD   # 4 d>=6
LO = [p for p in SH if A.PROBLEMS[p][1] <= 2]
PROBS11 = list(A.PROBLEMS)


def Mpa(p, a, o):
    return float(np.mean([L[(p, a, o, s)] for s in range(30) if (p, a, o, s) in L]))


def Ma(a, o, probs):
    return float(np.mean([Mpa(p, a, o) for p in probs]))


def Se(a, o, probs):
    """Problem-level SEM of M_pa (the cross-problem spread behind each bar)."""
    v = np.array([Mpa(p, a, o) for p in probs])
    return float(v.std(ddof=1) / np.sqrt(len(v)))


# ============================== TABLE DATA (printed) ==============================
print("=" * 78)
print("TABLE 3 — mean log-regret M_a per arm x optimizer x subset (lower=better)")
print(f"{'arm':8s} | {'grad_all':>9s} {'nsga_all':>9s} | {'grad_hi':>8s} {'nsga_hi':>8s} | {'grad_lo':>8s} {'nsga_lo':>8s}")
for a in ARMS:
    print(f"{a:8s} | {Ma(a,'grad',SH):9.2f} {Ma(a,'nsga',SH):9.2f} | "
          f"{Ma(a,'grad',HI):8.2f} {Ma(a,'nsga',HI):8.2f} | {Ma(a,'grad',LO):8.2f} {Ma(a,'nsga',LO):8.2f}")

print("\n" + "=" * 78)
print("TABLE 4 — pre-registered verdicts")
h1 = R["H1"]["per_optimizer"]; h3 = R["H3"]; h2 = R["H2"]
print(f"H1 grad: slope {h1['grad']['primary_ols_hc3']['slope']:+.3f} HC3p {h1['grad']['primary_ols_hc3']['p']:.3f} "
      f"rho {h1['grad']['primary_ols_hc3']['spearman_rho']:+.2f} (p {h1['grad']['primary_ols_hc3']['spearman_p']:.3f})")
print(f"H1 nsga: slope {h1['nsga']['primary_ols_hc3']['slope']:+.3f} HC3p {h1['nsga']['primary_ols_hc3']['p']:.3f} "
      f"rho {h1['nsga']['primary_ols_hc3']['spearman_rho']:+.2f} (p {h1['nsga']['primary_ols_hc3']['spearman_p']:.3f}) "
      f"=> H1_CONFIRMED {R['H1']['H1_CONFIRMED']}")
print(f"H2a: order {h2['our_order_best_to_worst']} rho {h2['criterion1_spearman']['rho_vs_DeAth_ref']:+.2f} "
      f"d>=6 {h2['criterion2_highd_majority']['n_exploit_wins']}/{h2['criterion2_highd_majority']['n_total']} "
      f"=> PASS {h2['H2a_PASS']}")
print(f"H3: D_p mean {h3['mean_D_p']:+.4f} Wilcoxon W {h3['primary_wilcoxon_one_sided_greater']['statistic']:.0f} "
      f"p {h3['primary_wilcoxon_one_sided_greater']['p']:.3f} | t-p {h3['sensitivity_ttest_one_sided']['p']:.3f} "
      f"sign-p {h3['sensitivity_sign_test']['p']:.3f} | MixedLM 1side-p "
      f"{h3['secondary_seedlevel_mixedlm']['p_one_sided_in_direction']:.3f} => {h3['H3_VERDICT']}")
print(f"LogEI advantage mean: grad {h3['descriptive_components']['mean_logei_adv_grad']:+.4f} "
      f"nsga {h3['descriptive_components']['mean_logei_adv_nsga']:+.4f} | clips {nclip}/{ncells}")

print("\n" + "=" * 78)
print("TABLE 1 — Part-1 De Ath Table-2 reproduction (representative cells, exp_05 validation gate)")
e5 = json.load(open(ROOT / "results" / "exp_05_tier1a.json"))
vg = e5["validation_gate"]
print(f"validation gate: {vg['n_median_pass']}/{vg['n_median_cells']} median cells reproduced; passed={vg['passed']}")
cells = vg["cells"]
shown = [c for c in cells if c.get("problem") in ("Branin", "logGSobol", "logHartmann6")][:9]
if shown:
    k0 = shown[0]
    print("  cell keys:", list(k0.keys()))
    for c in shown:
        print("   ", {k: c[k] for k in list(c.keys())})

print("\n" + "=" * 78)
print("TIER-1B (exp_07) reproduction tests — sign + Mann-Whitney per (problem,method) cell")
e7 = json.load(open(ROOT / "results" / "exp_07_tier1b.json"))
sign_ps, mw_ps, mw51_ps, dlogs = [], [], [], []
for cell, d in e7["comparisons"].items():
    ours = np.asarray(d["final_regret_ours"], float)
    theirs = np.asarray(d["final_regret_theirs_paired"], float)
    # sign test: paired ours<theirs
    npos = int(np.sum(ours < theirs))
    sp = stats.binomtest(npos, len(ours), 0.5).pvalue
    mw = stats.mannwhitneyu(ours, theirs, alternative="two-sided").pvalue
    mw51 = d["mannwhitney_11_vs_51"]["p_value"]  # the paper's comparison (Section 3.1)
    sign_ps.append(sp); mw_ps.append(mw); mw51_ps.append(mw51); dlogs.append(d["paired_dlog_median"])
    print(f"  {cell:18s} med_ours {d['median_ours']:.3e} med_theirs51 {d['median_theirs_all51']:.3e} "
          f"dlog {d['paired_dlog_median']:+.2f} | sign-p {sp:.3f} MW11v51-p {mw51:.3f} (MW-paired-p {mw:.3f})")
print(f"  => min sign-p {min(sign_ps):.4f}; paper's MW (ours-11 vs all-51): min {min(mw51_ps):.3f}, "
      f"Holm x{len(mw51_ps)} = {min(mw51_ps)*len(mw51_ps):.2f}; paired-11 MW min {min(mw_ps):.3f} "
      f"(diagnostic only, not the paper's test); max|dlog| {max(abs(x) for x in dlogs):.2f}, n_cells {len(sign_ps)}")


# ============================== FIGURES ==============================
# Okabe-Ito colourblind-safe palette, used across all figures.
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9"}
COL = {"eps-PF": OI["blue"], "UCB": OI["green"], "EI": OI["orange"],
       "LogEI": OI["purple"], "Exploit": OI["vermillion"]}

# ---- Fig 1: M_a per arm x optimiser, all vs high-d (the central result) ----
order = sorted(ARMS, key=lambda a: Ma(a, "nsga", SH))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)  # independent y (scales differ)
for ax, (subset, title) in zip(axes, [(SH, "All 10 De Ath problems"), (HI, "High-dim (d$\\geq$6, 4 problems), zoomed")]):
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, [Ma(a, "grad", subset) for a in order], w, label="gradient-based", color=OI["blue"],
           yerr=[Se(a, "grad", subset) for a in order], capsize=2, error_kw={"lw": 0.8})
    ax.bar(x + w/2, [Ma(a, "nsga", subset) for a in order], w, label="NSGA-II (gradient-free)", color=OI["orange"],
           yerr=[Se(a, "nsga", subset) for a in order], capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_title(title, fontsize=9); ax.axhline(0, color="k", lw=0.6)
axes[0].set_ylabel("mean log precise regret\n(lower is better)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, frameon=False, fontsize=8, loc="upper center",
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(FIG / "fig1_main_Ma.pdf"); plt.close(fig)

# ---- Fig 1 appendix variant: shared vertical axis across the two panels ----
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, (subset, title) in zip(axes, [(SH, "All 10 De Ath problems"), (HI, "High-dim (d$\\geq$6, 4 problems)")]):
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, [Ma(a, "grad", subset) for a in order], w, label="gradient-based", color=OI["blue"],
           yerr=[Se(a, "grad", subset) for a in order], capsize=2, error_kw={"lw": 0.8})
    ax.bar(x + w/2, [Ma(a, "nsga", subset) for a in order], w, label="NSGA-II (gradient-free)", color=OI["orange"],
           yerr=[Se(a, "nsga", subset) for a in order], capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_title(title, fontsize=9); ax.axhline(0, color="k", lw=0.6)
axes[0].set_ylabel("mean log precise regret\n(lower is better)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, frameon=False, fontsize=8, loc="upper center",
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(FIG / "fig1_shared_axis.pdf"); plt.close(fig)

# ---- Fig 2: H3 D_p per problem ----
Dp = [R["H3"]["D_p"][p] for p in PROBS11]
dims = [A.PROBLEMS[p][1] for p in PROBS11]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
xc = np.arange(len(PROBS11))
ax.bar(xc, Dp, color=[OI["blue"] if d > 0 else OI["vermillion"] for d in Dp])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xc); ax.set_xticklabels([f"{p}\n(d{dims[i]})" for i, p in enumerate(PROBS11)], rotation=40, ha="right", fontsize=7)
ax.set_ylabel("$D_p$  (LogEI advantage:\ngrad $-$ nsga)")
ax.set_title(f"H3: $D_p\\approx0$ (mean {R['H3']['mean_D_p']:+.3f}, one-sided Wilcoxon $p={R['H3']['primary_wilcoxon_one_sided_greater']['p']:.2f}$)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig2_h3_Dp.pdf"); plt.close(fig)

# ---- Fig 3: probe value-underflow vs gradient-vanishing by dimension (EI) ----
att = {(r["acq"], r["optimizer"], int(r["dim"])): r for r in R["attribution"]["per_acq_optimizer_dim"]}
dimlevels = [1, 2, 6, 8, 10]
val_nsga = [att[("EI", "nsga", d)]["raw_ei_underflow_frac"] for d in dimlevels]
grad_van_grad = [att[("EI", "grad", d)]["own_grad_frac_below_tiny"] for d in dimlevels]
grad_van_nsga = [att[("EI", "nsga", d)]["own_grad_frac_below_tiny"] for d in dimlevels]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
xx = np.arange(len(dimlevels))
ax.plot(xx, val_nsga, "o-", color=OI["vermillion"], label="EI value-underflow (raw EI $=0$), NSGA-II probe")
ax.plot(xx, grad_van_grad, "s--", color=OI["blue"], label="EI gradient-vanishing, gradient-based probe ($K{=}64$)")
ax.plot(xx, grad_van_nsga, "^:", color=OI["green"], label="EI gradient-vanishing, NSGA-II probe ($K{=}4096$)")
ax.set_xticks(xx); ax.set_xticklabels([f"d={d}" for d in dimlevels])
ax.set_ylabel("fraction of probe points"); ax.set_ylim(-0.02, 1.0)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.set_title("Mechanism probe: value-underflow is low-dimensional; gradients are unused by NSGA-II", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig3_probe.pdf"); plt.close(fig)

# ---- Fig 4: optimizer invariance scatter (grad vs nsga M_pa, arm x problem) ----
fig, ax = plt.subplots(figsize=(3.6, 3.4))
xs, ys = [], []
for a in ARMS:
    gx = [Mpa(p, a, "grad") for p in SH]; ny = [Mpa(p, a, "nsga") for p in SH]
    ax.scatter(gx, ny, s=18, color=COL[a], label=a, alpha=0.8, edgecolor="none")
    xs += gx; ys += ny
lo, hi = min(xs + ys) - 0.5, max(xs + ys) + 0.5
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
r = np.corrcoef(xs, ys)[0, 1]
ax.set_xlabel("gradient-based mean log-regret"); ax.set_ylabel("NSGA-II mean log-regret")
spear_in = float(np.mean([stats.spearmanr([Mpa(p, a, "grad") for a in ARMS],
                                          [Mpa(p, a, "nsga") for a in ARMS]).statistic
                          for p in SH]))
ax.set_title(f"Across-optimiser agreement ($r={r:.2f}$)", fontsize=9)
ax.annotate(f"within-problem rank agreement {spear_in:.2f}", xy=(0.97, 0.05),
            xycoords="axes fraction", ha="right", fontsize=7.5)
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig4_invariance.pdf"); plt.close(fig)
# ============================== APPENDIX TABLES (.tex, generated) ==============================
import exp_09_prereg_analysis as e9x  # noqa: E402  (frozen; read-only, for y_opt table)

merged_opt = {**e9x._problem_optimal_values(), **e9x._ext_optimal_values()}
DIMS = {p: A.PROBLEMS[p][1] for p in PROBS11}
# De Ath stems from the frozen exp_09 registries (exp_11_analysis's tuple holds the
# source matrix, not the stem); Ackley8 has no De Ath counterpart.
STEMS = {p: (e9x.EXT_PROBLEMS[p][0] if p == "Ackley8" else e9x.TIER2_PROBLEMS[p][0]) or "---"
         for p in PROBS11}


def _seed_vals(p, a, o):
    return np.array([L[(p, a, o, s)] for s in range(30) if (p, a, o, s) in L])


def _se(p, a, o):
    v = _seed_vals(p, a, o)
    return float(v.std(ddof=1) / np.sqrt(len(v)))


with open(FIG / "appd_mpa_tables.tex", "w") as f:
    for o, label in (("grad", "gradient-based"), ("nsga", "NSGA-II")):
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{Per-problem mean log precise regret $M_{{p,a}}$ under the "
                f"{label} optimiser, as mean (SE) over the 30 seeds per cell; "
                f"lower is better.}}\n")
        f.write(f"\\label{{tab:appd-mpa-{o}}}\n\\footnotesize\n")
        f.write("\\begin{tabular}{lr" + "r" * len(ARMS) + "}\n\\toprule\n")
        f.write("Problem & $d$ & " + " & ".join(ARMS) + " \\\\\n\\midrule\n")
        for p in PROBS11:
            cells = " & ".join(f"${Mpa(p, a, o):+.2f}$ ({_se(p, a, o):.2f})" for a in ARMS)
            f.write(f"{p} & {DIMS[p]} & {cells} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

with open(FIG / "appd_dp_table.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n")
    f.write("\\caption{Per-problem H3 contrast $D_p$ (LogEI advantage, gradient-based minus "
            "gradient-free), the seed-level standard error of $D_{p,s}$, and the per-problem "
            "LogEI-versus-EI advantage under each optimiser (positive = LogEI better).}\n")
    f.write("\\label{tab:appd-dp}\n\\small\n")
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
    f.write("Problem & $d$ & $D_p$ & SE$(D_{p,s})$ & adv$_{\\mathrm{grad}}$ & "
            "adv$_{\\mathrm{nsga}}$ \\\\\n\\midrule\n")
    for p in PROBS11:
        ag = _seed_vals(p, "EI", "grad") - _seed_vals(p, "LogEI", "grad")
        an = _seed_vals(p, "EI", "nsga") - _seed_vals(p, "LogEI", "nsga")
        dps = ag - an
        f.write(f"{p} & {DIMS[p]} & ${dps.mean():+.2f}$ & ${dps.std(ddof=1) / np.sqrt(len(dps)):.2f}$"
                f" & ${ag.mean():+.2f}$ & ${an.mean():+.2f}$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

with open(FIG / "app_problems_table.tex", "w") as f:
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{The eleven problems: dimension, the De Ath problem it maps to "
            "(Ackley has none; it was added at pre-registration), and the reference "
            "optimum $f^*$ used for precise regret (this repo maximises; log-prefixed "
            "De Ath problems are his log-transformed variants).}\n")
    f.write("\\label{tab:app-problems}\n\\small\n\\begin{tabular}{lrlr}\n\\toprule\n")
    f.write("Problem & $d$ & De Ath problem & $f^*$ \\\\\n\\midrule\n")
    for p in PROBS11:
        f.write(f"{p} & {DIMS[p]} & {STEMS[p]} & ${merged_opt[p]:+.6g}$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---- Table 1 (tab:repro): generated from the committed exp_05 validation gate ----
_T1_METHODS = ["LHS", "Explore", "EI", "PI", "UCB", "PFRandom", "eRandom", "eFront", "Exploit"]
_t1 = {c["method"]: c for c in e5["validation_gate"]["cells"] if c["problem"] == "Branin"}
_compact = lambda s: s.replace("e-0", "e-").replace("e+0", "e+")
with open(FIG / "tab_repro.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n")
    f.write("\\caption{Part-1 reproduction, representative of the $90/90$ reproduced "
            "Table-2 cells. Median absolute regret at $T=250$ over De Ath's $51$ published "
            "runs on Branin; our recomputation matches his published value to three "
            "significant figures for every method. In De Ath's labels, eFront and eRandom "
            "are the eps-PF and eps-RS eps-greedy methods ($\\epsilon=0.1$) and Exploit is "
            "pure greedy, matching the arm names used in Part~2.}\n")
    f.write("\\label{tab:repro}\n\\small\n\\begin{tabular}{lcc}\n\\toprule\n")
    f.write("Method & De Ath (published) & Ours (recomputed) \\\\\n\\midrule\n")
    for m in _T1_METHODS:
        f.write(f"{m:8s} & {_compact(_t1[m]['expected_3sf'])} & {_compact(_t1[m]['computed_3sf'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---- exp_12 post-hoc arms: App D per-problem table (from exp_12_analysis.json) ----
E12_PATH = ROOT / "results" / "exp_12_analysis.json"
if E12_PATH.exists():
    e12 = json.load(open(E12_PATH))
    if "arms" in e12:
        pp = e12["arms"]["per_problem_M_pa"]
        ppse = e12["arms"]["per_problem_SE"]
        arms12 = ["eps-PF-exact", "EI-RS", "LogEI-RS"]
        heads = {"eps-PF-exact": "eps-PF (exact front)", "EI-RS": "EI (RS)", "LogEI-RS": "LogEI (RS)"}
        with open(FIG / "appd_exp12_table.tex", "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n")
            f.write("\\caption{Per-problem mean log precise regret for the post-hoc arms "
                    "(Section~4.7), as mean (SE) over the 30 seeds per cell, beside the "
                    "pre-registered eps-PF variant under NSGA-II.}\n")
            f.write("\\label{tab:appd-exp12}\n\\footnotesize\n\\begin{tabular}{lrrrrr}\n\\toprule\n")
            f.write("Problem & $d$ & " + " & ".join(heads[a] for a in arms12)
                    + " & eps-PF (variant) \\\\\n\\midrule\n")
            for p in PROBS11:
                cells = " & ".join(f"${pp[a][p]:+.2f}$ ({ppse[a][p]:.2f})" for a in arms12)
                f.write(f"{p} & {DIMS[p]} & {cells} & "
                        f"${Mpa(p, 'eps-PF', 'nsga'):+.2f}$ ({_se(p, 'eps-PF', 'nsga'):.2f}) \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print("\nFIGURES written:", sorted(p.name for p in FIG.glob("*.pdf")))
print("TABLES written:", sorted(p.name for p in FIG.glob("*.tex")))
print(f"Fig4 invariance Pearson r(grad M_pa, nsga M_pa) over 5 arms x 10 problems = {r:.4f}")
