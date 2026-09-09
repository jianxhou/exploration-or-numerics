"""
Experiment 05: Tier 1A re-analysis of De Ath et al. (2021) published raw results.

Re-analyzes the per-run optimization trajectories published with "Greed is
Good" (ACM TELO 2021, github.com/georgedeath/egreedy, results_paper/*.npz).
A validation gate first reproduces the paper's Table 2 median-regret matrix
at T=250 (90 cells, 3 significant figures) plus MAD spot checks; only then
are the new cross-problem analyses run:

  Analysis A   Demsar-strict Friedman + Nemenyi on the 10 x 9 matrix
               (51 runs per problem/method aggregated by median).
  Analysis B   the paper's per-problem procedure: paired Wilcoxon vs the
               best-median method, Holm-corrected, alpha = 0.05; run both
               one-sided (paper text) and two-sided (their code's default).
  Analysis C   Analysis A repeated at budget slices T = 20, 50, 150, 250.
               At T=20 the d=10 problems contain only the initial design
               (M = 2d = 20), so T=20 is reported with and without them.
  Analysis D   MixedLM: log(regret + floor) ~ strategy (ref LHS), random
               intercept per problem, run-within-problem variance component;
               fit at T=250 and T=50, floors 1e-6 and 1e-8.
  Analysis E   paired effect sizes at T=250 for eRandom/eFront/Exploit vs EI
               and eRandom vs UCB: per-problem median paired difference of
               log regret, 95% bootstrap CI (paired by run), rank-biserial.

Requires a clone of github.com/georgedeath/egreedy at DATA_DIR. Their data
stays outside this repository; only derived statistics are written here.

Usage:
    python experiments/exp_05_tier1a_reanalysis.py

Outputs:
    results/exp_05_tier1a.json
    figures/cd_diagram_deathdata_n10.png
    figures/budget_slices_deathdata.png
"""
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import studentized_range
from statsmodels.stats.multitest import multipletests

DATA_DIR = Path.home() / "projects" / "egreedy"

# Table 2 synthetic problems, in the paper's row order, with dimensionality.
PROBLEMS = {
    "WangFreitas": 1,
    "BraninForrester": 2,
    "Branin": 2,
    "Cosines": 2,
    "logGoldsteinPrice": 2,
    "logSixHumpCamel": 2,
    "logHartmann6": 6,
    "logGSobol": 10,
    "logRosenbrock": 10,
    "logStyblinskiTang": 10,
}

# Table 2 column order. File names use eRandom_eps0.1 / eFront_eps0.1 for the
# headline eps = 0.1 arms.
METHODS = ["LHS", "Explore", "EI", "PI", "UCB", "PFRandom", "eRandom", "eFront", "Exploit"]
METHOD_FILE = {m: m for m in METHODS}
METHOD_FILE["eRandom"] = "eRandom_eps0.1"
METHOD_FILE["eFront"] = "eFront_eps0.1"

RUNS = range(1, 52)
T_FULL = 250
CHECKPOINTS = [20, 50, 150, 250]
ALPHA = 0.05

# Published Table 2 medians at T=250 (3 significant figures), row order as
# PROBLEMS, column order as METHODS.
EXPECTED_T2 = {
    "WangFreitas": [1.27e-2, 1.04e-2, 2.00, 2.06, 2.00, 2.00e-4, 1.04e-6, 2.00, 2.00],
    "BraninForrester": [4.59e-1, 4.58e-1, 2.47e-6, 3.73e-4, 4.96e-6, 2.70e-3, 2.00e-6, 2.31e-6,
                        4.61e-6],
    "Branin": [1.31e-1, 1.66e-1, 4.15e-6, 2.26e-5, 4.42e-6, 1.67e-3, 3.17e-6, 3.57e-6, 3.08e-6],
    "Cosines": [4.79e-1, 4.56e-1, 6.31e-6, 2.50e-3, 7.12e-6, 8.82e-3, 8.66e-6, 2.02e-6, 4.13e-1],
    "logGoldsteinPrice": [1.08, 1.01, 2.73e-6, 2.92e-3, 6.15e-6, 2.54e-3, 2.33e-6, 8.76e-7,
                          2.26e-6],
    "logSixHumpCamel": [6.52, 6.53, 7.42e-5, 1.46e-1, 3.84, 1.52e-1, 3.81e-5, 4.06e-5, 4.21e-5],
    "logHartmann6": [3.37e-1, 3.07e-1, 1.06e-3, 6.15e-4, 2.04e-1, 6.57e-2, 5.09e-4, 7.71e-4,
                     6.37e-4],
    "logGSobol": [1.51e1, 1.75e1, 7.15, 6.29, 1.45e1, 5.60, 5.13, 5.06, 5.27],
    "logRosenbrock": [1.16e1, 1.28e1, 6.62, 6.89, 8.31, 5.23, 4.75, 4.64, 4.54],
    "logStyblinskiTang": [2.85, 3.19, 2.34, 2.29, 3.19, 2.70, 1.61, 1.53, 1.82],
}

# MAD spot checks. The paper's MADs are computed with the (deprecated)
# scipy.stats.median_absolute_deviation, whose default scale is 1.4826
# (egreedy/util/plotting.py:10,731) — i.e. the normal-consistency-scaled MAD,
# not the unscaled one.
MAD_SCALE = 1.4826
EXPECTED_MAD = [
    ("Branin", "EI", 3.76e-6),
    ("Branin", "eRandom", 2.46e-6),
    ("logGSobol", "EI", 1.58),
    ("logGSobol", "eFront", 1.37),
]

EFFECT_PAIRS = [("eRandom", "EI"), ("eFront", "EI"), ("Exploit", "EI"), ("eRandom", "UCB")]

K = len(METHODS)
N_PROBLEMS = len(PROBLEMS)


def sig3(x: float) -> str:
    """3-significant-figure scientific representation used for gate matching."""
    return f"{float(x):.2e}"


def upstream_commit() -> str:
    out = subprocess.run(
        ["git", "-C", str(DATA_DIR), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def load_fopt() -> tuple[dict, dict]:
    """Instantiate each problem class from their synthetic_problems.py.

    Returns (fopt values, source records). The source record carries the file
    and the line of the `self.yopt` assignment inside the class body.
    """
    src_file = DATA_DIR / "egreedy" / "test_problems" / "synthetic_problems.py"
    spec = importlib.util.spec_from_file_location("egreedy_synthetic", src_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    src_lines = src_file.read_text().splitlines()
    fopt, sources = {}, {}
    for name in PROBLEMS:
        cls = getattr(module, name)
        f = cls()
        fopt[name] = float(np.squeeze(f.yopt))
        cls_line = next(
            i for i, ln in enumerate(src_lines) if ln.startswith(f"class {name}")
        )
        yopt_line = 1 + next(
            i for i, ln in enumerate(src_lines[cls_line:], start=cls_line)
            if "self.yopt" in ln
        )
        sources[name] = {
            "class": f"{cls.__name__}",
            "file": str(src_file.relative_to(DATA_DIR)),
            "yopt_assignment_line": yopt_line,
            "value": fopt[name],
        }
    return fopt, sources


def load_best_so_far(fopt: dict) -> dict:
    """best[problem][method]: (51, 250) cumulative-best regret per run.

    Mirrors their pipeline (egreedy/util/plotting.py:74-78): the per-evaluation
    absolute distance |y - yopt| is taken FIRST, then the cumulative min. This
    differs from |cummin(y) - yopt| exactly when evaluations land below the
    recorded yopt (observed on logSixHumpCamel, whose yopt comes from a coarse
    xopt). Rows 1..M of Ytr are the initial design (M = 2d), included in the
    250-evaluation budget.
    """
    best = {}
    for prob in PROBLEMS:
        best[prob] = {}
        for meth in METHODS:
            curves = np.empty((len(list(RUNS)), T_FULL))
            for i, run in enumerate(RUNS):
                path = DATA_DIR / "results_paper" / f"{prob}_{run}_{T_FULL}_{METHOD_FILE[meth]}.npz"
                y = np.load(path)["Ytr"].ravel()
                if y.shape[0] != T_FULL:
                    raise ValueError(f"{path.name}: expected {T_FULL} rows, got {y.shape[0]}")
                curves[i] = np.minimum.accumulate(np.abs(y - fopt[prob]))
            best[prob][meth] = curves
    return best


def regret_at(best: dict, fopt: dict, prob: str, meth: str, t: int) -> np.ndarray:
    """(51,) cumulative-best regret after t evaluations."""
    return best[prob][meth][:, t - 1]


def check_sign_convention(fopt: dict) -> dict:
    """Reproduce the Branin/EI anchor (median 4.15e-6) before the full gate."""
    anchor = sig3(4.15e-6)
    out = {"anchor": "Branin/EI median regret at T=250", "expected": anchor}
    for label, acc in [("cummin", np.minimum.accumulate), ("cummax", np.maximum.accumulate)]:
        vals = []
        for run in RUNS:
            path = DATA_DIR / "results_paper" / f"Branin_{run}_{T_FULL}_EI.npz"
            y = np.load(path)["Ytr"].ravel()
            vals.append(abs(acc(y)[T_FULL - 1] - fopt["Branin"]))
        med = float(np.median(vals))
        out[label] = med
        if sig3(med) == anchor:
            out["convention"] = label
            return out
    print("Sign-convention check FAILED: neither cummin nor cummax reproduces "
          f"the anchor {anchor} (got cummin={sig3(out['cummin'])}, "
          f"cummax={sig3(out['cummax'])}).")
    sys.exit(1)


def validation_gate(best: dict, fopt: dict) -> dict:
    """Reproduce Table 2 (90 medians) and the 4 MAD spot checks at 3 sig figs."""
    cells, n_pass = [], 0
    for prob in PROBLEMS:
        for j, meth in enumerate(METHODS):
            reg = regret_at(best, fopt, prob, meth, T_FULL)
            computed = float(np.median(reg))
            expected = EXPECTED_T2[prob][j]
            ok = sig3(computed) == sig3(expected)
            n_pass += ok
            cells.append({
                "problem": prob, "method": meth,
                "computed": computed, "computed_3sf": sig3(computed),
                "expected_3sf": sig3(expected), "pass": bool(ok),
            })

    mads = []
    for prob, meth, expected in EXPECTED_MAD:
        reg = regret_at(best, fopt, prob, meth, T_FULL)
        mad = float(MAD_SCALE * np.median(np.abs(reg - np.median(reg))))
        mads.append({
            "problem": prob, "method": meth,
            "computed": mad, "computed_3sf": sig3(mad),
            "expected_3sf": sig3(expected), "pass": sig3(mad) == sig3(expected),
        })

    passed = n_pass == len(cells) and all(m["pass"] for m in mads)
    if not passed:
        print("\n!!! VALIDATION GATE FAILED — per-cell diagnostic !!!")
        for c in cells:
            if not c["pass"]:
                print(f"  median {c['problem']}/{c['method']}: "
                      f"computed {c['computed_3sf']}  expected {c['expected_3sf']}")
        for m in mads:
            if not m["pass"]:
                print(f"  MAD {m['problem']}/{m['method']}: "
                      f"computed {m['computed_3sf']}  expected {m['expected_3sf']}")
        print(f"  medians passed: {n_pass}/{len(cells)}")
        sys.exit(1)

    return {
        "n_median_cells": len(cells), "n_median_pass": int(n_pass),
        "mad_spot_checks": mads, "passed": True, "cells": cells,
    }


def median_matrix(best: dict, fopt: dict, t: int, problems=None) -> np.ndarray:
    probs = problems if problems is not None else list(PROBLEMS)
    return np.array([
        [float(np.median(regret_at(best, fopt, p, m, t))) for m in METHODS]
        for p in probs
    ])


def friedman_nemenyi(matrix: np.ndarray) -> dict:
    chi2, p = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    ranks = pd.DataFrame(matrix, columns=METHODS).rank(axis=1).mean(axis=0)
    nemenyi = sp.posthoc_nemenyi_friedman(matrix)
    nemenyi.columns = METHODS
    nemenyi.index = METHODS
    n, k = matrix.shape
    q = studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {
        "n_blocks": int(n), "k": int(k),
        "friedman": {
            "chi_square": float(chi2), "p_value": float(p),
            "df": k - 1, "rejected_H0": bool(p < ALPHA),
        },
        "average_ranks": {m: round(float(r), 4) for m, r in ranks.items()},
        "nemenyi_pvalues": nemenyi.round(6).to_dict(),
        "q_alpha": round(float(q), 4), "CD": round(float(cd), 4),
    }


def equivalence_sets(best: dict, fopt: dict, t: int, alternative: str = "greater") -> dict:
    """Per-problem: paired Wilcoxon vs best-median method, Holm correction.

    p >= alpha after Holm -> the method is statistically equivalent to the
    best on that problem (the paper's Table 2 procedure). alternative
    'greater' is the one-sided test the paper text describes; 'two-sided' is
    what their create_table_data actually computes (scipy wilcoxon default,
    egreedy/util/plotting.py:748).
    """
    out = {}
    for prob in PROBLEMS:
        reg = {m: regret_at(best, fopt, prob, m, t) for m in METHODS}
        medians = {m: float(np.median(r)) for m, r in reg.items()}
        best_meth = min(medians, key=medians.get)
        others = [m for m in METHODS if m != best_meth]
        pvals = []
        for m in others:
            diff = reg[m] - reg[best_meth]
            if np.all(diff == 0):
                pvals.append(1.0)
            else:
                # 'greater' H1: the method's regret exceeds the best method's.
                _, p = stats.wilcoxon(reg[m], reg[best_meth], alternative=alternative)
                pvals.append(float(p))
        rejected, p_holm, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
        equivalent = [best_meth] + [
            m for m, rej in zip(others, rejected, strict=True) if not rej
        ]
        out[prob] = {
            "best_method": best_meth,
            "best_median": medians[best_meth],
            "holm_pvalues": {
                m: round(float(p), 6) for m, p in zip(others, p_holm, strict=True)
            },
            "equivalent_to_best": sorted(equivalent, key=METHODS.index),
        }
    return out


def mixedlm_fit(best: dict, fopt: dict, t: int, floor: float) -> dict:
    """log(regret + floor) ~ strategy (ref LHS), random intercept per problem,
    run-within-problem variance component (runs share initial designs).
    """
    records = []
    for prob in PROBLEMS:
        for meth in METHODS:
            reg = regret_at(best, fopt, prob, meth, t)
            for i, run in enumerate(RUNS):
                records.append({
                    "problem": prob, "strategy": meth, "run": run,
                    "y": float(np.log(reg[i] + floor)),
                })
    df = pd.DataFrame.from_records(records)

    formula = "y ~ C(strategy, Treatment(reference='LHS'))"
    md = smf.mixedlm(
        formula, df, groups=df["problem"], re_formula="1",
        vc_formula={"run": "0 + C(run)"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mdf = md.fit()
    warn_msgs = sorted({f"{w.category.__name__}: {w.message}" for w in caught})

    conf = mdf.conf_int()
    coefficients = {}
    for meth in METHODS:
        if meth == "LHS":
            continue
        term = f"C(strategy, Treatment(reference='LHS'))[T.{meth}]"
        coefficients[meth] = {
            "coef": float(mdf.params[term]),
            "std_err": float(mdf.bse[term]),
            "ci_low": float(conf.loc[term, 0]),
            "ci_high": float(conf.loc[term, 1]),
            "p_value": float(mdf.pvalues[term]),
        }
    return {
        "T": t, "floor": floor, "formula": formula,
        "reference": "LHS", "n_obs": int(len(df)),
        "group_var_problem": float(mdf.cov_re.values[0, 0]) if mdf.cov_re.size else None,
        "vc_run_within_problem": float(mdf.vcomp[0]) if len(mdf.vcomp) else None,
        "coefficients": coefficients,
        "converged": bool(mdf.converged),
        "warnings": warn_msgs,
    }


def effect_sizes(best: dict, fopt: dict, t: int, n_boot: int = 10_000) -> dict:
    """Per-problem paired log-regret differences (A - B) with bootstrap CIs.

    Paired by run number (shared initial designs). Negative median difference
    means method A reaches lower regret than method B.
    """
    rng = np.random.default_rng(20260611)
    tiny = 1e-300  # log() guard only; no regret in this data is exactly 0
    out = {}
    for a, b in EFFECT_PAIRS:
        pair_key = f"{a}_vs_{b}"
        out[pair_key] = {}
        for prob in PROBLEMS:
            la = np.log(np.maximum(regret_at(best, fopt, prob, a, t), tiny))
            lb = np.log(np.maximum(regret_at(best, fopt, prob, b, t), tiny))
            d = la - lb
            n = len(d)
            idx = rng.integers(0, n, size=(n_boot, n))
            boot_medians = np.median(d[idx], axis=1)
            lo, hi = np.percentile(boot_medians, [2.5, 97.5])

            # Rank-biserial from the signed ranks of the nonzero differences.
            nz = d[d != 0]
            r_abs = stats.rankdata(np.abs(nz))
            w_pos = float(r_abs[nz > 0].sum())
            w_neg = float(r_abs[nz < 0].sum())
            rb = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) > 0 else 0.0

            out[pair_key][prob] = {
                "median_log_diff": float(np.median(d)),
                "ci95_low": float(lo), "ci95_high": float(hi),
                "rank_biserial": round(float(rb), 4),
            }
    return out


def plot_cd_diagram(avg_ranks: dict, cd: float, output_path: Path, subtitle: str):
    """Demsar-style CD diagram (exp_04 visual style, maximal-clique bars)."""
    ranks_sorted = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    names = [n for n, _ in ranks_sorted]
    vals = [v for _, v in ranks_sorted]

    lo = float(np.floor(min(vals)))
    hi = float(np.ceil(max(vals)))
    mid = (lo + hi) / 2

    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_axis = 0.0
    ax.plot([lo, hi], [y_axis, y_axis], color="black", linewidth=1.6, zorder=1)
    for t in np.arange(lo, hi + 0.01, 1.0):
        ax.plot([t, t], [y_axis, y_axis + 0.05], color="black", linewidth=1.2)
        ax.text(t, y_axis + 0.13, f"{t:.0f}", ha="center", va="bottom", fontsize=11)

    right = [(n, v) for n, v in zip(names, vals, strict=True) if v <= mid]
    left = [(n, v) for n, v in zip(names, vals, strict=True) if v > mid]
    col_r = lo - 0.40
    col_l = hi + 0.40

    def _draw(rank, name, side, level):
        y = -0.22 - 0.24 * level
        ax.plot([rank, rank], [y_axis, y], color="0.45", linewidth=1.4)
        if side == "right":
            ax.plot([rank, col_r], [y, y], color="0.45", linewidth=1.4)
            ax.text(col_r, y, f"  {name} ({rank:.2f})", ha="left",
                    va="center", fontsize=11)
        else:
            ax.plot([rank, col_l], [y, y], color="0.45", linewidth=1.4)
            ax.text(col_l, y, f"({rank:.2f}) {name}  ", ha="right",
                    va="center", fontsize=11)

    for lvl, (n, v) in enumerate(right):
        _draw(v, n, "right", lvl)
    for lvl, (n, v) in enumerate(left):
        _draw(v, n, "left", lvl)

    # Maximal clique bars: widest runs [i, j] with vals[j] - vals[i] < CD,
    # dropping any bar contained in another.
    n = len(vals)
    spans = []
    for i in range(n):
        j = i
        while j + 1 < n and (vals[j + 1] - vals[i]) < cd:
            j += 1
        if j > i:
            spans.append((i, j))
    maximal = [
        (a, b) for (a, b) in spans
        if not any(a2 <= a and b <= b2 and (a2, b2) != (a, b) for (a2, b2) in spans)
    ]
    for level, (i, j) in enumerate(maximal):
        y_clique = -0.07 - 0.06 * level
        ax.plot([vals[i], vals[j]], [y_clique, y_clique],
                color="#c0392b", linewidth=5, solid_capstyle="round", zorder=3)

    y_cd = y_axis + 0.46
    cd_x0 = hi - 0.05
    ax.plot([cd_x0, cd_x0 - cd], [y_cd, y_cd], color="black", linewidth=1.8)
    for xx in (cd_x0, cd_x0 - cd):
        ax.plot([xx, xx], [y_cd - 0.04, y_cd + 0.04], color="black", linewidth=1.8)
    ax.text(cd_x0 - cd / 2, y_cd + 0.07, f"CD = {cd:.3f}", ha="center",
            va="bottom", fontsize=11, fontweight="bold")

    ax.text(mid, y_axis + 0.84,
            f"Critical Difference Diagram (Nemenyi, $\\alpha$={ALPHA})",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(mid, y_axis + 0.70, subtitle, ha="center", fontsize=10, color="0.3")

    ax.set_xlim(hi + 3.2, lo - 3.2)  # inverted: best rank on the right
    ax.set_ylim(-0.22 - 0.24 * max(len(right), len(left)) - 0.12, 1.06)
    ax.axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.04)
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="white")
    plt.close()
    print(f"  CD diagram saved to {output_path}  (CD={cd:.4f})")


def plot_budget_slices(slices: dict, output_path: Path):
    """Average rank per method across budget checkpoints; right panel drops
    the d=10 problems (whose T=20 slice is initial design only)."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, sharey=True)
    panels = [
        (axes[0], "all", f"All {N_PROBLEMS} problems"),
        (axes[1], "no_d10", "Excluding d=10 problems (n=7)"),
    ]
    for ax, key, title in panels:
        for meth, color in zip(METHODS, colors, strict=True):
            xs = [t for t in CHECKPOINTS if key in slices[str(t)]]
            ys = [slices[str(t)][key]["average_ranks"][meth] for t in xs]
            ax.plot(xs, ys, "o-", color=color, label=meth, linewidth=1.6, markersize=5)
        ax.set_xlabel("Evaluations T")
        ax.set_xticks(CHECKPOINTS)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # rank 1 (best) on top
    axes[0].set_ylabel("Average rank (lower = better)")
    axes[1].legend(fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("Average ranks across budget checkpoints "
                 "(medians over 51 runs, De Ath et al. data)", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close()
    print(f"  Budget-slice figure saved to {output_path}")


def main():
    if not DATA_DIR.is_dir() or not (DATA_DIR / "results_paper").is_dir():
        sys.exit(
            f"Data not found at {DATA_DIR}. Clone the published results first:\n"
            f"  git clone https://github.com/georgedeath/egreedy {DATA_DIR}"
        )

    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    commit = upstream_commit()
    print(f"Data: {DATA_DIR}  (upstream commit {commit[:12]})")

    fopt, fopt_sources = load_fopt()
    print("\nf_opt per problem (from their synthetic_problems.py):")
    for name, rec in fopt_sources.items():
        print(f"  {name:<18} {rec['value']:.10g}   "
              f"({rec['file']}:{rec['yopt_assignment_line']})")

    print("\nSign-convention check (Branin/EI anchor)...")
    convention = check_sign_convention(fopt)
    print(f"  convention = {convention['convention']}  "
          f"(median {sig3(convention[convention['convention']])} "
          f"vs expected {convention['expected']})")

    print("\nLoading all trajectories "
          f"({N_PROBLEMS} problems x {K} methods x 51 runs)...")
    best = load_best_so_far(fopt)

    print("\n=== VALIDATION GATE: Table 2 medians at T=250 (3 sig figs) ===")
    gate = validation_gate(best, fopt)
    print(f"  medians: {gate['n_median_pass']}/{gate['n_median_cells']} cells match")
    for m in gate["mad_spot_checks"]:
        print(f"  MAD {m['problem']}/{m['method']}: {m['computed_3sf']} "
              f"(expected {m['expected_3sf']}) {'ok' if m['pass'] else 'FAIL'}")
    print("  PASSED — proceeding to new analyses.\n")

    print("=== Analysis A: Demsar-strict Friedman + Nemenyi (N=10, T=250) ===")
    a = friedman_nemenyi(median_matrix(best, fopt, T_FULL))
    print(f"  Friedman chi2({a['friedman']['df']}) = "
          f"{a['friedman']['chi_square']:.4f}  p = {a['friedman']['p_value']:.4e}")
    print("  average ranks: " + ", ".join(
        f"{m}={a['average_ranks'][m]:.2f}" for m in METHODS))
    print(f"  CD (k={K}, N={N_PROBLEMS}) = {a['CD']:.3f}")
    plot_cd_diagram(
        a["average_ranks"], a["CD"],
        figures_dir / "cd_diagram_deathdata_n10.png",
        f"N = {N_PROBLEMS} blocks (problems), 51 runs aggregated by median, "
        f"k = {K} methods, T = 250",
    )
    print()

    print("=== Analysis B: per-problem equivalent-to-best sets (Wilcoxon + Holm) ===")
    b = equivalence_sets(best, fopt, T_FULL, alternative="greater")
    b2 = equivalence_sets(best, fopt, T_FULL, alternative="two-sided")
    print("  one-sided (paper text):")
    for prob, rec in b.items():
        print(f"    {prob:<18} best={rec['best_method']:<8} "
              f"set: {', '.join(rec['equivalent_to_best'])}")
    print("  two-sided (their code's wilcoxon default):")
    for prob, rec in b2.items():
        print(f"    {prob:<18} best={rec['best_method']:<8} "
              f"set: {', '.join(rec['equivalent_to_best'])}")
    disagreements = {
        prob: {
            "one_sided": b[prob]["equivalent_to_best"],
            "two_sided": b2[prob]["equivalent_to_best"],
        }
        for prob in PROBLEMS
        if b[prob]["equivalent_to_best"] != b2[prob]["equivalent_to_best"]
    }
    if disagreements:
        print("  variants disagree on: " + ", ".join(disagreements))
    else:
        print("  variants agree on all problems.")
    cosines_set = set(b["Cosines"]["equivalent_to_best"])
    soft_ok = {"EI", "UCB", "PI"} <= cosines_set
    print(f"  soft check (paper: EI, UCB, PI equivalent on Cosines): "
          f"{'consistent' if soft_ok else 'NOT consistent'}\n")

    print("=== Analysis C: budget slices ===")
    slices = {}
    d10 = [p for p, d in PROBLEMS.items() if d == 10]
    non_d10 = [p for p in PROBLEMS if p not in d10]
    for t in CHECKPOINTS:
        slices[str(t)] = {"all": friedman_nemenyi(median_matrix(best, fopt, t))}
        slices[str(t)]["no_d10"] = friedman_nemenyi(
            median_matrix(best, fopt, t, problems=non_d10))
        fa = slices[str(t)]["all"]["friedman"]
        print(f"  T={t:<4} chi2={fa['chi_square']:8.3f}  p={fa['p_value']:.3e}  "
              "ranks: " + ", ".join(
                  f"{m}={slices[str(t)]['all']['average_ranks'][m]:.2f}"
                  for m in METHODS))
    print(f"  note: at T=20 the d=10 problems ({', '.join(d10)}) contain only "
          "the initial design (M = 2d = 20); see the no_d10 variant.")
    plot_budget_slices(slices, figures_dir / "budget_slices_deathdata.png")
    print()

    print("=== Analysis D: mixed-effects models ===")
    mixed = []
    for t in (250, 50):
        for floor in (1e-6, 1e-8):
            print(f"  fitting T={t}, floor={floor:g}...")
            fit = mixedlm_fit(best, fopt, t, floor)
            mixed.append(fit)
            for meth, co in fit["coefficients"].items():
                print(f"    {meth:<9} coef={co['coef']:+.4f}  se={co['std_err']:.4f}  "
                      f"95% CI=[{co['ci_low']:+.4f}, {co['ci_high']:+.4f}]")
            if fit["warnings"]:
                for w in fit["warnings"]:
                    print(f"    warning: {w}")
            else:
                print("    no warnings emitted.")
    print()

    print("=== Analysis E: paired effect sizes at T=250 ===")
    e = effect_sizes(best, fopt, T_FULL)
    for pair, recs in e.items():
        print(f"  {pair}:")
        for prob, r in recs.items():
            print(f"    {prob:<18} median dlog={r['median_log_diff']:+8.3f}  "
                  f"CI=[{r['ci95_low']:+8.3f}, {r['ci95_high']:+8.3f}]  "
                  f"rank-biserial={r['rank_biserial']:+.3f}")
    print()

    summary = {
        "provenance": {
            "source": "https://github.com/georgedeath/egreedy",
            "commit": commit,
            "data_dir": str(DATA_DIR),
            "paper": "De Ath et al., Greed is Good (ACM TELO 2021), Table 2",
            "file_pattern": "{Problem}_{run}_250_{Method}.npz, runs 1..51, "
                            "eps arms eRandom_eps0.1 / eFront_eps0.1",
            "sign_convention": convention,
            "pipeline_adaptations": [
                "regret trajectory = cummin(|y - yopt|), abs before cummin, "
                "matching egreedy/util/plotting.py:74-78 (differs from "
                "|cummin(y) - yopt| on logSixHumpCamel)",
                "Table 2 MADs use scipy.stats.median_absolute_deviation with "
                "its default scale 1.4826 (egreedy/util/plotting.py:10,731), "
                "i.e. scaled MAD, not unscaled",
                "their create_table_data calls two-sided wilcoxon() "
                "(egreedy/util/plotting.py:748) although the paper text says "
                "one-sided; Analysis B implements the one-sided test as "
                "specified for this experiment",
            ],
        },
        "f_opt_sources": fopt_sources,
        "validation_gate": gate,
        "analysis_A_friedman_n10": a,
        "analysis_B_equivalence_sets": b,
        "analysis_B_equivalence_sets_two_sided": b2,
        "analysis_B_variant_disagreements": disagreements,
        "analysis_B_soft_check_cosines": {
            "expected_superset": ["EI", "UCB", "PI"], "consistent": bool(soft_ok),
        },
        "analysis_C_budget_slices": slices,
        "analysis_D_mixedlm": mixed,
        "analysis_E_effect_sizes": e,
    }
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "exp_05_tier1a.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results written to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
