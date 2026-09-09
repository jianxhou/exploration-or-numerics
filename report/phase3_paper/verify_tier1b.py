"""Reproduce the Part-1 Tier-1B reproduction statistics from committed + released artifacts.

Reads results/exp_07_tier1b.json (committed: our 11 Docker re-runs per cell) and De Ath's
released 51-run results (~/projects/egreedy/results_paper/, his published data). For each of the
8 (problem, method) cells it computes:
  - the paired sign test (ours vs De Ath's paired run, shared initial design),
  - the Mann-Whitney test of our 11 runs against De Ath's full 51-run distribution,
  - Holm-Bonferroni correction across the 8 cells.
This is the test reported in Part 1: ours-11 vs theirs-ALL-51 (not paired-11-vs-paired-11).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).parent.parent.parent
RP = Path.home() / "projects" / "egreedy" / "results_paper"
e7 = json.load(open(ROOT / "results" / "exp_07_tier1b.json"))
fopt = e7["config"]["f_opt"]
MFILE = {"EI": "EI", "eRandom": "eRandom_eps0.1", "eFront": "eFront_eps0.1", "Exploit": "Exploit"}


def theirs_final_regret(problem, method_file, run):
    z = np.load(RP / f"{problem}_{run}_250_{method_file}.npz")
    y = np.asarray(z["Ytr"]).ravel()[:250]
    return float(np.min(np.abs(y - fopt[problem])))  # cummin(|y - yopt|) at T=250


print(f"{'cell':18s} {'sign_p':>7s} {'MW(all-51)':>11s} {'|dlog|':>7s}")
sign_ps, mw_ps = [], []
for cell, d in e7["comparisons"].items():
    prob, meth = cell.split("_", 1)
    ours = np.asarray(d["final_regret_ours"], float)
    paired = np.asarray(d["final_regret_theirs_paired"], float)
    t51 = np.asarray([theirs_final_regret(prob, MFILE[meth], r) for r in range(1, 52)])
    sp = stats.binomtest(int(np.sum(ours < paired)), len(ours), 0.5).pvalue
    mw = stats.mannwhitneyu(ours, t51, alternative="two-sided").pvalue
    sign_ps.append(sp); mw_ps.append(mw)
    print(f"{cell:18s} {sp:7.3f} {mw:11.3f} {abs(d['paired_dlog_median']):7.2f}")
mw_holm = multipletests(mw_ps, method="holm")[1]
print(f"\nmin sign-p = {min(sign_ps):.3f}  |  min MW(all-51) = {min(mw_ps):.3f}  ->  Holm-corrected {min(mw_holm):.3f}")
print(f"max |dlog median| = {max(abs(d['paired_dlog_median']) for d in e7['comparisons'].values()):.2f}")
