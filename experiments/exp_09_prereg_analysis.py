r"""
Experiment 09 — Tier 2 core-matrix analysis.

================================================================================
PRE-REGISTERED ANALYSIS PLAN
================================================================================
This script is committed BEFORE the Tier 2 core-matrix data (exp_08) exists. The
hypotheses, directions, test statistics, and constants stated here are frozen.
`--gate-only` runs the validation gates (G1-G3) on the already-published Tier 1A
data tonight; the full mode fail-fasts cleanly until results/exp_08_matrix/ is
present.

FROZEN CONSTANTS
  floors                = {1e-6, 1e-8}            (log-regret offset; both reported)
  reference arm         = Random
  dim groups            = low (d<=2), mid (d=6), high (d=10)
  stagnation window     W = 10                    (trailing iters with no improvement)
  post-degeneracy K     = 10                      (improvement look-ahead window)
  alpha                 = 0.05
  CD (k=9, N=10)        = 3.799   (q = 3.1017)    Tier 1A reference, 9 De Ath arms
  CD (k=8, N=10)        = 3.320   (q = 3.031)     Tier 2, our 8 arms
  checkpoints           = {20, 50, 150, 250}      (function evaluations)
  T=20 caveat           = at T=20 the d=10 problems contain only the initial
                          design (M = 2d = 20); ALL T=20 analyses exclude the
                          high dim group.

REGRET TRACKS (both computed for every run)
  PRECISE  = best-so-far minus the class optimal_value (this stack's precise
             optimum). PRIMARY track for ALL within-our-stack inference.
  DEATH    = De Ath-style abs-then-cummin against DEATH_YOPT (their coarse code
             yopt). Used ONLY for cross-stack comparisons with their data,
             matching the exp_05 / exp_07 conventions.

VALIDATION GATES
  G1  Through the same De Ath npz loader the cross-stack analysis uses, recompute
      the per-(problem, method) T=250 medians on their data; assert exact
      agreement with results/exp_05_tier1a.json.
  G2  Assert the analysis utilities reproduce exp_05's N=10 Friedman
      chi2 = 50.66 (tol 0.01) and its rank vector from their data.
  G3  Recompute CD(k=9, N=10) = 3.799 and CD(k=8, N=10) = 3.320 from the q table;
      assert both.
  G4 (full mode only) exp_08 manifest: 2400 files, no NaN final regrets, and rank
      sums equal k(k+1)/2 in every Friedman call.

ANALYSIS A — cross-problem inference on OUR stack (PRECISE track), per checkpoint
  - Friedman over N=10 problem blocks, k=8 arms, 30 seeds median-aggregated per
    cell; Nemenyi; CD diagram at T=250 (maximal-clique bars, exp_04 style).
  - MixedLM: log(regret + floor) ~ strategy, random intercept problem,
    run-within-problem variance component; both floors; coefficients with 95%
    CIs; convergence reported honestly.
  - Per-problem one-sided paired Wilcoxon vs the best-median arm, Holm-corrected.
  - Generalization model: log(regret + floor) ~ strategy*budget_cat +
    strategy*dim_group + (1|problem) + run-within-problem VC, on the stacked
    checkpoint data. On non-convergence, fall back to the pre-specified
    decomposition (per-checkpoint and per-dim-group models) and state which ran.

ANALYSIS B — claim verdicts on OUR stack
  C1  contrasts (EpsRS-EI, EpsPF-EI, EpsRS-UCB, EpsPF-UCB) as mixed-model
      contrasts with CIs at each checkpoint, alongside the rank-gap-vs-CD reading.
  C2  via the strategy x dim_group interaction.
  C4  via Exploit's high-group ranks and the Exploit-EI contrast.

ANALYSIS C — mechanism (pre-registered predictions)
  P1a (clean E3 identification, within our stack): paired per (problem, seed)
      Delta = log regret(EI) - log regret(LogEI), PRECISE track.
      HYPOTHESIS: Delta > 0, increasing with dim group.
      TESTS: per-problem one-sided Wilcoxon signed-rank; pooled MixedLM
      Delta ~ 1 + dim_group with (1|problem); forest plot per problem.
  P1b (optimizer mediation, cross-stack difference-in-differences, DEATH track,
      seeds/runs 1..30 paired by shared initial designs):
        D(problem, s) = [logR_ours(EI) - logR_ours(EpsRS)]
                      - [logR_theirs(EI) - logR_theirs(eRandom)].
      HYPOTHESIS: D > 0 (EI's standing relative to greedy is worse under the
      gradient stack). TESTS: per-problem one-sided Wilcoxon; pooled mixed model
      with (1|problem). The output labels P1b
      "consistent-with evidence; stack differences beyond the acquisition
      optimizer are not held constant".
  P2  (underflow-performance coupling, EI and LogEI arms, probe logs):
      degenerate iteration := candidate zero_fraction == 1.0 OR
      all_restarts_degenerate. stagnation := no improvement in best-so-far over
      the trailing W=10 iterations. PRIMARY statistic: within each EI run,
      P(improvement within next K=10 | degenerate iteration) vs
      P(improvement within next K=10 | non-degenerate), paired; sign test across
      the 30 seeds per problem; LogEI arms reported identically as control.
      HEADLINE descriptive: % of EI runs with >=1 degenerate iteration, by
      problem and dim group; same for LogEI (expected ~0); NumericsWarning counts.
  P3  (scaling): median candidate zero-fraction and below-tiny fraction vs
      iteration index, per problem; Spearman trend within problem; compare levels
      across dim groups. Performance side: the P1a gap by dim group (from the P1a
      model) read against the probe trend.

OUTPUTS
  results/exp_09_analysis.json   (all numbers, gates first)
  figures/cd_diagram_tier2_n10.png, rank_vs_budget_tier2.png,
          p1a_forest_by_problem.png, p1b_did_by_problem.png,
          probe_zero_fraction_heatmap.png

USAGE
  python experiments/exp_09_prereg_analysis.py --gate-only   # tonight
  python experiments/exp_09_prereg_analysis.py               # after exp_08 exists

EXP_08 DATA CONTRACT (frozen here, before the data exists)
  results/exp_08_matrix/ holds 2400 JSON files, one per (arm, problem, seed) run
  for 8 arms x 10 problems x 30 seeds. Each file's CONTENT (not its name) must
  carry at least:
    arm     : one of ARMS (canonical labels below)
    problem : one of TIER2_PROBLEMS keys (this project's names)
    seed    : int in 0..29; seed s used De Ath initial-design run (s+1) so the
              cross-stack P1b pairing by shared initial design holds
    y       : list of 250 per-evaluation raw (maximization) objective values
              (the run's train_y, M initial-design evals first)
    probe   : EI/LogEI arms only — the run's per-iteration AcqProbe log
              (list length n_iter); absent/None for other arms
  optimal_value and DEATH_YOPT are taken from this project's classes, not the
  files. Both regret tracks are recomputed here from `y`.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# The De Ath data loader and CD-diagram drawer are reused verbatim from Tier 1A,
# so the cross-stack analysis uses "the same loader" the gates validate.
import exp_05_tier1a_reanalysis as t1a  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scikit_posthocs as sp  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import studentized_range  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

# ----------------------------------------------------------------------------
# Frozen constants
# ----------------------------------------------------------------------------
ALPHA = 0.05
FLOORS = (1e-6, 1e-8)
REFERENCE_ARM = "Random"
W_STAGNATION = 10
K_LOOKAHEAD = 10
CHECKPOINTS = (20, 50, 150, 250)
T_FULL = 250
LOG_GUARD = 1e-300
N_SEEDS = 30
N_FILES_EXPECTED = 2400  # 8 arms x 10 problems x 30 seeds

# k=8 arms, in a fixed display order; reference arm last is fine (factor handles it).
ARMS = ["EI", "LogEI", "UCB", "eps-RS", "eps-PF", "Exploit", "Explore", "Random"]

# our problem name -> (De Ath stem, dim, dim_group)
TIER2_PROBLEMS = {
    "WangFreitas": ("WangFreitas", 1, "low"),
    "BraninForrester": ("BraninForrester", 2, "low"),
    "Branin": ("Branin", 2, "low"),
    "Cosines": ("Cosines", 2, "low"),
    "GoldsteinPriceLog": ("logGoldsteinPrice", 2, "low"),
    "SixHumpCamelLog": ("logSixHumpCamel", 2, "low"),
    "Hartmann6Log": ("logHartmann6", 6, "mid"),
    "GSobolLog": ("logGSobol", 10, "high"),
    "RosenbrockLog": ("logRosenbrock", 10, "high"),
    "StyblinskiTangLog": ("logStyblinskiTang", 10, "high"),
}
HIGH_PROBLEMS = [p for p, (_, _, g) in TIER2_PROBLEMS.items() if g == "high"]

# cross-stack arm correspondence (our label -> De Ath file label)
CROSS_ARM = {"EI": "EI", "eps-RS": "eRandom"}

PROJECT_ROOT = Path(__file__).parent.parent
MATRIX_DIR = PROJECT_ROOT / "results" / "exp_08_matrix"
RESULTS_JSON = PROJECT_ROOT / "results" / "exp_09_analysis.json"
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================================
# POST-HOC EXTENSION (exp_10 extension matrix) -- added 2026-06-17.
# NOT part of the exp09-prereg pre-registration. The frozen objects above
# (ARMS, TIER2_PROBLEMS, N_FILES_EXPECTED, T_FULL, N_SEEDS) ARE the pre-
# registered contract and are left BYTE-FOR-BYTE UNTOUCHED: their scientific
# value is that they were frozen before the core data was seen and have never
# been modified since -- the same standard applied to the De Ath F1 audit and
# the engineering-buffer disclosure. These ADDITIVE names extend the loader for
# a separate, clearly-labelled post-hoc extension matrix only (raw GSobol /
# Rosenbrock for the C5 raw-vs-log axis; the Ament Ackley-d8 anchor for D7).
# run_exp08_core.py never reads them; its contract gate still sees the frozen
# core (10 problems, 2400 files) and stays green.
# ============================================================================
# our name -> (De Ath stem or None for the Sobol fallback, dim, dim_group)
EXT_PROBLEMS = {
    "GSobol": ("GSobol", 10, "high"),          # raw gSobol (C5); De Ath injection; no DEATH_YOPT
    "Rosenbrock": ("Rosenbrock", 10, "high"),  # raw Rosenbrock (C5); De Ath injection; no DEATH_YOPT
    "Ackley8": (None, 8, "mid"),               # Ament anchor (D7); Sobol fallback; no shared De Ath design
}
EXT_N_FILES = len(ARMS) * len(EXT_PROBLEMS) * N_SEEDS   # 8 x 3 x 30 = 720
ALL_PROBLEMS = {**TIER2_PROBLEMS, **EXT_PROBLEMS}        # core (frozen) + extension; exp_10 loader only
EXT_MATRIX_DIR = PROJECT_ROOT / "results" / "exp_10_matrix"


def _ext_optimal_values() -> dict[str, float]:
    """optimal_value per extension problem, from this project's classes (PRECISE
    track only; none of these carry a DEATH_YOPT). Additive -- does NOT touch the
    frozen _problem_optimal_values()."""
    from al_benchmark.problems import death as D
    from al_benchmark.problems.synthetic import Ackley
    classes = {"GSobol": D.GSobol, "Rosenbrock": D.Rosenbrock,
               "Ackley8": lambda: Ackley(dim=8)}
    return {name: float(cls().optimal_value) for name, cls in classes.items()}


# ----------------------------------------------------------------------------
# Generic statistics utilities (validated against exp_05 by G2)
# ----------------------------------------------------------------------------
def cd_value(k: int, n: int, alpha: float = ALPHA) -> float:
    """Nemenyi critical difference from the studentized-range q table."""
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def friedman_nemenyi(matrix: np.ndarray, labels: list[str]) -> dict:
    """Friedman + Nemenyi over an (N_blocks, k) matrix with named columns.

    Asserts the average ranks sum to k(k+1)/2 (G4 invariant), which holds for any
    valid ranking and so guards against block/label mis-wiring.
    """
    n, k = matrix.shape
    chi2, p = stats.friedmanchisquare(*[matrix[:, i] for i in range(k)])
    ranks = pd.DataFrame(matrix, columns=labels).rank(axis=1).mean(axis=0)
    rank_sum = float(ranks.sum())
    assert abs(rank_sum - k * (k + 1) / 2) < 1e-6, (
        f"rank sum {rank_sum} != k(k+1)/2 = {k * (k + 1) / 2} (k={k})"
    )
    nemenyi = sp.posthoc_nemenyi_friedman(matrix)
    nemenyi.columns = labels
    nemenyi.index = labels
    return {
        "n_blocks": int(n),
        "k": int(k),
        "friedman": {
            "chi_square": float(chi2),
            "p_value": float(p),
            "df": k - 1,
            "rejected_H0": bool(p < ALPHA),
        },
        "average_ranks": {m: float(r) for m, r in ranks.items()},
        "rank_sum": rank_sum,
        "nemenyi_pvalues": nemenyi.round(6).to_dict(),
        "CD": round(cd_value(k, n), 4),
    }


def holm_wilcoxon_vs_best(cell: dict[str, np.ndarray], arms: list[str]) -> dict:
    """Per-block one-sided paired Wilcoxon of every arm vs the best-median arm,
    Holm-corrected. `cell[arm]` is the paired vector (seeds) of regrets.
    """
    medians = {m: float(np.median(cell[m])) for m in arms}
    best = min(medians, key=medians.get)
    others = [m for m in arms if m != best]
    pvals = []
    for m in others:
        diff = cell[m] - cell[best]
        if np.all(diff == 0):
            pvals.append(1.0)
        else:
            _, p = stats.wilcoxon(cell[m], cell[best], alternative="greater")
            pvals.append(float(p))
    rejected, p_holm, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
    equivalent = [best] + [m for m, rej in zip(others, rejected, strict=True) if not rej]
    return {
        "best_arm": best,
        "best_median": medians[best],
        "holm_pvalues": {m: round(float(p), 6) for m, p in zip(others, p_holm, strict=True)},
        "equivalent_to_best": sorted(equivalent, key=ARMS.index),
    }


# ----------------------------------------------------------------------------
# Validation gates
# ----------------------------------------------------------------------------
def _load_exp05_json() -> dict:
    path = PROJECT_ROOT / "results" / "exp_05_tier1a.json"
    if not path.is_file():
        sys.exit(f"G1/G2 need {path}; run exp_05 first.")
    with open(path) as f:
        return json.load(f)


def gate_g1_g2(exp05: dict) -> dict:
    """G1 (median reproduction) and G2 (Friedman chi2 + ranks) on De Ath data.

    Uses the Tier 1A loader (t1a) — the same loader the cross-stack analysis uses.
    Returns the gate record and the loaded `best` structure for reuse by P1b.
    """
    if not t1a.DATA_DIR.is_dir() or not (t1a.DATA_DIR / "results_paper").is_dir():
        sys.exit(
            f"De Ath data not found at {t1a.DATA_DIR}; G1/G2 cannot run. "
            "Clone github.com/georgedeath/egreedy there first."
        )
    fopt, _ = t1a.load_fopt()
    best = t1a.load_best_so_far(fopt)

    # G1: per-(problem, method) T=250 medians vs the frozen exp_05 cells.
    expected_cells = {
        (c["problem"], c["method"]): c["computed"] for c in exp05["validation_gate"]["cells"]
    }
    g1_cells, max_abs_err = [], 0.0
    for prob in t1a.PROBLEMS:
        for meth in t1a.METHODS:
            computed = float(np.median(t1a.regret_at(best, fopt, prob, meth, T_FULL)))
            expected = expected_cells[(prob, meth)]
            err = abs(computed - expected)
            max_abs_err = max(max_abs_err, err)
            g1_cells.append(
                {"problem": prob, "method": meth, "computed": computed,
                 "expected": expected, "abs_err": err}
            )
    g1_pass = max_abs_err <= 1e-9

    # G2: Friedman chi2 + ranks on the N=10, k=9 median matrix.
    matrix = t1a.median_matrix(best, fopt, T_FULL)
    fn = friedman_nemenyi(matrix, t1a.METHODS)
    chi2 = fn["friedman"]["chi_square"]
    exp_a = exp05["analysis_A_friedman_n10"]
    chi2_pass = abs(chi2 - exp_a["friedman"]["chi_square"]) <= 1e-6
    chi2_target_pass = abs(chi2 - 50.66) <= 0.01
    rank_err = max(
        abs(fn["average_ranks"][m] - exp_a["average_ranks"][m]) for m in t1a.METHODS
    )
    ranks_pass = rank_err <= 1e-6

    return {
        "G1_median_reproduction": {
            "n_cells": len(g1_cells),
            "max_abs_err": max_abs_err,
            "tolerance": 1e-9,
            "passed": bool(g1_pass),
            "cells": g1_cells,
        },
        "G2_friedman_reproduction": {
            "chi_square": chi2,
            "exp05_chi_square": exp_a["friedman"]["chi_square"],
            "target_chi_square": 50.66,
            "chi2_matches_exp05": bool(chi2_pass),
            "chi2_matches_target_0.01": bool(chi2_target_pass),
            "average_ranks": fn["average_ranks"],
            "rank_max_abs_err_vs_exp05": rank_err,
            "ranks_passed": bool(ranks_pass),
            "passed": bool(chi2_pass and chi2_target_pass and ranks_pass),
        },
    }, best, fopt


def gate_g3() -> dict:
    """G3: CD(k=9, N=10) = 3.799 and CD(k=8, N=10) = 3.320 from the q table."""
    cd9 = cd_value(9, 10)
    cd8 = cd_value(8, 10)
    g3 = {
        "CD_k9_N10": {"computed": cd9, "target": 3.799, "passed": abs(cd9 - 3.799) <= 0.01},
        "CD_k8_N10": {"computed": cd8, "target": 3.320, "passed": abs(cd8 - 3.320) <= 0.01},
    }
    g3["passed"] = bool(g3["CD_k9_N10"]["passed"] and g3["CD_k8_N10"]["passed"])
    return g3


def gate_g4_manifest(runs: dict) -> dict:
    """G4 (full mode): exp_08 manifest sanity. `runs` keyed by (problem, arm, seed)."""
    n_files = len(runs)
    nan_finals = [
        f"{p}/{a}/seed{s}"
        for (p, a, s), r in runs.items()
        if not np.isfinite(r["precise_final"])
    ]
    g4 = {
        "n_files": n_files,
        "n_files_expected": N_FILES_EXPECTED,
        "files_ok": n_files == N_FILES_EXPECTED,
        "n_nan_final_regrets": len(nan_finals),
        "nan_runs": nan_finals[:50],
        "passed": n_files == N_FILES_EXPECTED and not nan_finals,
    }
    return g4


# ----------------------------------------------------------------------------
# exp_08 loader and regret tracks
# ----------------------------------------------------------------------------
def _problem_optimal_values() -> dict[str, float]:
    """optimal_value per Tier 2 problem, from this project's problem classes."""
    from al_benchmark.problems import death as D
    from al_benchmark.problems.synthetic import Branin

    classes = {
        "WangFreitas": D.WangFreitas, "BraninForrester": D.BraninForrester,
        "Branin": Branin, "Cosines": D.Cosines, "GoldsteinPriceLog": D.GoldsteinPriceLog,
        "SixHumpCamelLog": D.SixHumpCamelLog, "Hartmann6Log": D.Hartmann6Log,
        "GSobolLog": D.GSobolLog, "RosenbrockLog": D.RosenbrockLog,
        "StyblinskiTangLog": D.StyblinskiTangLog,
    }
    return {name: float(cls().optimal_value) for name, cls in classes.items()}


def _death_yopt() -> dict[str, float]:
    from death_suite import DEATH_YOPT

    return DEATH_YOPT


def _tracks_from_y(y: np.ndarray, optimal_value: float, death_yopt: float | None) -> dict:
    """Per-evaluation PRECISE and DEATH regret trajectories (length T_FULL).

    PRECISE: optimal_value - cummax(y)  (maximization; this project's optimum).
    DEATH:   cummin(|(-y) - death_yopt|)  (their minimization convention; -y = g).
    """
    best_max = np.maximum.accumulate(y)
    precise = optimal_value - best_max
    out = {"precise": precise, "precise_final": float(precise[-1])}
    if death_yopt is not None:
        g = -y
        death = np.minimum.accumulate(np.abs(g - death_yopt))
        out["death"] = death
        out["death_final"] = float(death[-1])
    return out


def load_matrix(matrix_dir: Path) -> dict:
    """Read results/exp_08_matrix/*.json into {(problem, arm, seed): record}.

    Each record carries the per-eval `y`, both regret tracks, and the probe log.
    Indexing is by file CONTENT, robust to filename conventions.
    """
    optvals = _problem_optimal_values()
    yopts = _death_yopt()
    runs = {}
    files = sorted(matrix_dir.glob("*.json"))
    if not files:
        sys.exit(f"{matrix_dir} contains no *.json run files.")
    for path in files:
        with open(path) as f:
            rec = json.load(f)
        prob, arm, seed = rec["problem"], rec["arm"], int(rec["seed"])
        if prob not in TIER2_PROBLEMS:
            sys.exit(f"{path.name}: unknown problem {prob!r}")
        if arm not in ARMS:
            sys.exit(f"{path.name}: unknown arm {arm!r}")
        y = np.asarray(rec["y"], dtype=np.float64)
        if y.shape != (T_FULL,):
            sys.exit(f"{path.name}: y must have length {T_FULL}, got {y.shape}")
        tracks = _tracks_from_y(y, optvals[prob], yopts.get(prob))
        runs[(prob, arm, seed)] = {
            "y": y,
            "precise": tracks["precise"],
            "precise_final": tracks["precise_final"],
            "death": tracks.get("death"),
            "death_final": tracks.get("death_final"),
            "probe": rec.get("probe"),
            "dim_group": TIER2_PROBLEMS[prob][2],
            "dim": TIER2_PROBLEMS[prob][1],
        }
    return runs


def _checkpoint_index(t: int) -> int:
    """Per-evaluation trajectory index for checkpoint t (1-based eval count)."""
    return t - 1


def precise_cell(runs: dict, problem: str, arm: str, t: int) -> np.ndarray:
    """(N_SEEDS,) PRECISE regret at checkpoint t for one (problem, arm)."""
    idx = _checkpoint_index(t)
    return np.array(
        [runs[(problem, arm, s)]["precise"][idx] for s in range(N_SEEDS)]
    )


def problems_at_checkpoint(t: int) -> list[str]:
    """Problems analysable at checkpoint t (drop the high group at T=20)."""
    if t <= 20:
        return [p for p in TIER2_PROBLEMS if p not in HIGH_PROBLEMS]
    return list(TIER2_PROBLEMS)


# ----------------------------------------------------------------------------
# Analysis A
# ----------------------------------------------------------------------------
def _precise_long(runs: dict, t: int, floor: float) -> pd.DataFrame:
    probs = problems_at_checkpoint(t)
    idx = _checkpoint_index(t)
    rows = []
    for p in probs:
        for a in ARMS:
            for s in range(N_SEEDS):
                rows.append({
                    "problem": p, "strategy": a, "seed": s,
                    "dim_group": TIER2_PROBLEMS[p][2],
                    "budget_cat": str(t),
                    "regret": float(runs[(p, a, s)]["precise"][idx]),
                    "y": float(np.log(max(runs[(p, a, s)]["precise"][idx], 0.0) + floor)),
                })
    return pd.DataFrame.from_records(rows)


def _mixedlm(df: pd.DataFrame, formula: str) -> dict:
    md = smf.mixedlm(
        formula, df, groups=df["problem"], re_formula="1",
        vc_formula={"seed": "0 + C(seed)"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            mdf = md.fit()
        except Exception as exc:  # noqa: BLE001 — report fit failure, do not crash
            return {"formula": formula, "converged": False, "error": str(exc),
                    "coefficients": {}, "warnings": []}
    warn_msgs = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    conf = mdf.conf_int()
    coefficients = {
        term: {
            "coef": float(mdf.params[term]),
            "ci_low": float(conf.loc[term, 0]),
            "ci_high": float(conf.loc[term, 1]),
            "p_value": float(mdf.pvalues[term]),
        }
        for term in mdf.fe_params.index  # fixed effects only (exclude variance terms)
    }
    return {
        "formula": formula,
        "converged": bool(mdf.converged),
        "n_obs": int(df.shape[0]),
        "coefficients": coefficients,
        "warnings": warn_msgs,
        "_fit": mdf,  # retained in-process for contrasts; stripped before JSON
    }


def _contrast(mdf, term_a: str, term_b: str) -> dict | None:
    """CI for (coef_a - coef_b) via a linear hypothesis on the fixed effects."""
    names = list(mdf.fe_params.index)
    if term_a not in names or term_b not in names:
        return None
    vec = np.zeros((1, len(names)))  # (1, k_fe) row vector for t_test
    vec[0, names.index(term_a)] = 1.0
    vec[0, names.index(term_b)] = -1.0
    tt = mdf.t_test(vec)
    return {
        "estimate": float(np.ravel(tt.effect)[0]),
        "ci_low": float(np.ravel(tt.conf_int())[0]),
        "ci_high": float(np.ravel(tt.conf_int())[1]),
        "p_value": float(np.ravel(tt.pvalue)[0]),
    }


def _safe_contrast(mdf, converged: bool, term_a: str, term_b: str) -> dict | None:
    """_contrast guarded against a None / non-converged fit."""
    if mdf is None or not converged:
        return None
    return _contrast(mdf, term_a, term_b)


def _term(arm: str) -> str:
    return f"C(strategy, Treatment(reference='{REFERENCE_ARM}'))[T.{arm}]"


def analysis_A(runs: dict) -> dict:
    """Friedman/Nemenyi/CD, MixedLM (both floors), Wilcoxon, generalization."""
    out = {"per_checkpoint": {}, "mixedlm": {}, "wilcoxon_vs_best": {}}

    for t in CHECKPOINTS:
        probs = problems_at_checkpoint(t)
        matrix = np.array([
            [float(np.median(precise_cell(runs, p, a, t))) for a in ARMS] for p in probs
        ])
        out["per_checkpoint"][str(t)] = {
            "problems": probs,
            "friedman_nemenyi": friedman_nemenyi(matrix, ARMS),
        }
        # Per-problem one-sided Wilcoxon vs best, Holm.
        out["wilcoxon_vs_best"][str(t)] = {
            p: holm_wilcoxon_vs_best({a: precise_cell(runs, p, a, t) for a in ARMS}, ARMS)
            for p in probs
        }

    # MixedLM at each checkpoint, both floors.
    mform = f"y ~ C(strategy, Treatment(reference='{REFERENCE_ARM}'))"
    for t in CHECKPOINTS:
        out["mixedlm"][str(t)] = {}
        for floor in FLOORS:
            fit = _mixedlm(_precise_long(runs, t, floor), mform)
            fit.pop("_fit", None)
            out["mixedlm"][str(t)][f"floor_{floor:g}"] = fit

    # Generalization model on stacked checkpoints (drop high group at T=20 rows).
    stacked = pd.concat([_precise_long(runs, t, FLOORS[0]) for t in CHECKPOINTS], ignore_index=True)
    gen_form = (
        "y ~ C(strategy, Treatment(reference='%s')) * C(budget_cat)"
        " + C(strategy, Treatment(reference='%s')) * C(dim_group)" % (REFERENCE_ARM, REFERENCE_ARM)
    )
    gen = _mixedlm(stacked, gen_form)
    gen.pop("_fit", None)
    if gen["converged"]:
        out["generalization"] = {"path": "joint", "model": gen}
    else:
        # Pre-specified fallback: per-checkpoint and per-dim-group simpler models.
        per_cp = {str(t): out["mixedlm"][str(t)][f"floor_{FLOORS[0]:g}"] for t in CHECKPOINTS}
        per_dg = {}
        for g in ("low", "mid", "high"):
            sub = stacked[stacked["dim_group"] == g]
            if sub["problem"].nunique() >= 2 and not sub.empty:
                fit = _mixedlm(sub, mform)
                fit.pop("_fit", None)
                per_dg[g] = fit
        out["generalization"] = {
            "path": "fallback_decomposition",
            "reason": "joint generalization model did not converge",
            "per_checkpoint": per_cp,
            "per_dim_group": per_dg,
        }
    return out


# ----------------------------------------------------------------------------
# Analysis B — claim verdicts (C1, C2, C4)
# ----------------------------------------------------------------------------
def analysis_B(runs: dict) -> dict:
    out = {"C1_contrasts": {}, "C2_dim_interaction": {}, "C4_exploit": {}}
    mform = f"y ~ C(strategy, Treatment(reference='{REFERENCE_ARM}'))"

    c1_pairs = [("eps-RS", "EI"), ("eps-PF", "EI"), ("eps-RS", "UCB"), ("eps-PF", "UCB")]
    for t in CHECKPOINTS:
        fit = _mixedlm(_precise_long(runs, t, FLOORS[0]), mform)
        mdf = fit.pop("_fit", None)
        fn = friedman_nemenyi(
            np.array([[float(np.median(precise_cell(runs, p, a, t))) for a in ARMS]
                      for p in problems_at_checkpoint(t)]),
            ARMS,
        )
        ranks, cd = fn["average_ranks"], fn["CD"]
        contrasts = {}
        for a, b in c1_pairs:
            key = f"{a}-{b}"
            cc = _safe_contrast(mdf, fit["converged"], _term(a), _term(b))
            contrasts[key] = {
                "mixedlm_contrast": cc,
                "rank_gap": round(ranks[a] - ranks[b], 4),
                "exceeds_CD": bool(abs(ranks[a] - ranks[b]) > cd),
            }
        out["C1_contrasts"][str(t)] = {"CD": cd, "contrasts": contrasts}

    # C2: strategy x dim_group interaction (joint or per-dim ranks).
    gen_form = (
        "y ~ C(strategy, Treatment(reference='%s')) * C(dim_group)" % REFERENCE_ARM
    )
    stacked = _precise_long(runs, T_FULL, FLOORS[0])
    c2 = _mixedlm(stacked, gen_form)
    c2.pop("_fit", None)
    out["C2_dim_interaction"] = {
        "model_T250": c2,
        "ranks_by_dim_group": _ranks_by_dim_group(runs, T_FULL),
    }

    # C4: Exploit high-group ranks + Exploit-EI contrast at each checkpoint.
    for t in CHECKPOINTS:
        high = [p for p in problems_at_checkpoint(t) if TIER2_PROBLEMS[p][2] == "high"]
        if high:
            matrix = np.array([
                [float(np.median(precise_cell(runs, p, a, t))) for a in ARMS] for p in high
            ])
            if len(high) >= 2:
                high_ranks = friedman_nemenyi(matrix, ARMS)["average_ranks"]
            else:
                ranks = pd.DataFrame(matrix, columns=ARMS).rank(axis=1).mean(axis=0)
                high_ranks = {a: float(ranks[a]) for a in ARMS}
        else:
            high_ranks = None
        fit = _mixedlm(_precise_long(runs, t, FLOORS[0]), mform)
        mdf = fit.pop("_fit", None)
        cc = _safe_contrast(mdf, fit["converged"], _term("Exploit"), _term("EI"))
        out["C4_exploit"][str(t)] = {
            "high_group_ranks": high_ranks,
            "exploit_minus_ei_contrast": cc,
        }
    return out


def _ranks_by_dim_group(runs: dict, t: int) -> dict:
    out = {}
    for g in ("low", "mid", "high"):
        probs = [p for p in problems_at_checkpoint(t) if TIER2_PROBLEMS[p][2] == g]
        if not probs:
            continue
        matrix = np.array([
            [float(np.median(precise_cell(runs, p, a, t))) for a in ARMS] for p in probs
        ])
        ranks = pd.DataFrame(matrix, columns=ARMS).rank(axis=1).mean(axis=0)
        out[g] = {a: float(ranks[a]) for a in ARMS}
    return out


# ----------------------------------------------------------------------------
# Analysis C — mechanism (P1a, P1b, P2, P3)
# ----------------------------------------------------------------------------
def analysis_P1a(runs: dict) -> dict:
    """Paired Delta = log regret(EI) - log regret(LogEI), PRECISE, at T=250."""
    idx = _checkpoint_index(T_FULL)
    floor = FLOORS[0]
    per_problem, rows = {}, []
    for p in TIER2_PROBLEMS:
        ei = np.array([np.log(max(runs[(p, "EI", s)]["precise"][idx], 0.0) + floor)
                       for s in range(N_SEEDS)])
        lg = np.array([np.log(max(runs[(p, "LogEI", s)]["precise"][idx], 0.0) + floor)
                       for s in range(N_SEEDS)])
        delta = ei - lg
        if np.all(delta == 0):
            wstat, wp = float("nan"), 1.0
        else:
            wstat, wp = stats.wilcoxon(ei, lg, alternative="greater")
        per_problem[p] = {
            "dim_group": TIER2_PROBLEMS[p][2],
            "median_delta": float(np.median(delta)),
            "delta_ci": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
            "wilcoxon_greater_p": float(wp),
        }
        for s in range(N_SEEDS):
            rows.append({"problem": p, "dim_group": TIER2_PROBLEMS[p][2], "delta": float(delta[s])})
    df = pd.DataFrame.from_records(rows)
    pooled = _mixedlm_simple(df, "delta ~ C(dim_group)", group="problem")
    return {"hypothesis": "Delta > 0, increasing with dim group",
            "per_problem": per_problem, "pooled_mixedlm": pooled}


def analysis_P1b(runs: dict, best_death: dict, fopt: dict) -> dict:
    """Cross-stack difference-in-differences on the DEATH track (runs/seeds 1..30)."""
    idx = _checkpoint_index(T_FULL)
    per_problem, rows = {}, []
    for p, (death_name, _, dg) in TIER2_PROBLEMS.items():
        ours_ei = np.array([np.log(runs[(p, "EI", s)]["death_final"] + LOG_GUARD)
                            for s in range(N_SEEDS)])
        ours_rs = np.array([np.log(runs[(p, "eps-RS", s)]["death_final"] + LOG_GUARD)
                            for s in range(N_SEEDS)])
        theirs_ei = np.log(best_death[death_name][CROSS_ARM["EI"]][:N_SEEDS, idx] + LOG_GUARD)
        theirs_rs = np.log(best_death[death_name][CROSS_ARM["eps-RS"]][:N_SEEDS, idx] + LOG_GUARD)
        d = (ours_ei - ours_rs) - (theirs_ei - theirs_rs)
        if np.all(d == 0):
            wp = 1.0
        else:
            _, wp = stats.wilcoxon(d, np.zeros_like(d), alternative="greater")
        per_problem[p] = {
            "dim_group": dg, "median_D": float(np.median(d)),
            "D_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "wilcoxon_greater_p": float(wp),
        }
        for s in range(N_SEEDS):
            rows.append({"problem": p, "dim_group": dg, "D": float(d[s])})
    df = pd.DataFrame.from_records(rows)
    pooled = _mixedlm_simple(df, "D ~ 1", group="problem")
    return {
        "hypothesis": "D > 0 (EI's standing relative to greedy is worse under the gradient stack)",
        "interpretation_label": (
            "consistent-with evidence; stack differences beyond the acquisition "
            "optimizer are not held constant"
        ),
        "per_problem": per_problem,
        "pooled_mixedlm": pooled,
    }


def _degenerate_flags(probe: list) -> np.ndarray:
    """Per-iteration degenerate flag from a run's probe log."""
    flags = []
    for entry in probe:
        zero_frac = entry["candidate_pool"]["frac_acqf_zero"]
        all_deg = entry["optimizer"]["all_restarts_degenerate"]
        flags.append(bool(zero_frac == 1.0 or all_deg))
    return np.array(flags, dtype=bool)


def _improvement_flags(y: np.ndarray, n_init: int, n_iter: int, k: int) -> np.ndarray:
    """Per-iteration: did best-so-far improve within the next k iterations?"""
    best = np.maximum.accumulate(y)
    # best-so-far after iteration i is at eval index n_init + i (0-based eval idx).
    after = np.array([best[n_init + i] for i in range(n_iter)])
    improved = np.zeros(n_iter, dtype=bool)
    for i in range(n_iter):
        j = min(i + k, n_iter - 1)
        improved[i] = after[j] > after[i] + 0.0
    return improved


def analysis_P2(runs: dict) -> dict:
    """Underflow-performance coupling for EI (LogEI as control)."""
    out = {"arms": {}, "headline": {}}
    for arm in ("EI", "LogEI"):
        per_problem, descr = {}, {}
        for p in TIER2_PROBLEMS:
            n_init = 2 * TIER2_PROBLEMS[p][1]
            n_iter = T_FULL - n_init
            diffs, n_runs_with_deg, warn_counts = [], 0, []
            for s in range(N_SEEDS):
                rec = runs[(p, arm, s)]
                probe = rec["probe"]
                if not probe:
                    continue
                deg = _degenerate_flags(probe)
                imp = _improvement_flags(rec["y"], n_init, n_iter, K_LOOKAHEAD)
                m = min(len(deg), len(imp))
                deg, imp = deg[:m], imp[:m]
                n_runs_with_deg += int(deg.any())
                warn_counts.append(int(sum(e["numerics_warnings"] for e in probe)))
                if deg.any() and (~deg).any():
                    p_deg = float(imp[deg].mean())
                    p_non = float(imp[~deg].mean())
                    diffs.append(p_deg - p_non)
            diffs = np.array(diffs)
            if diffs.size:
                n_pos, n_neg = int((diffs > 0).sum()), int((diffs < 0).sum())
                sign_p = (stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue
                          if (n_pos + n_neg) > 0 else 1.0)
            else:
                n_pos = n_neg = 0
                sign_p = 1.0
            per_problem[p] = {
                "dim_group": TIER2_PROBLEMS[p][2],
                "n_runs_paired": int(diffs.size),
                "median_p_improve_diff": float(np.median(diffs)) if diffs.size else None,
                "sign_test": {"n_pos": n_pos, "n_neg": n_neg, "p_value": float(sign_p)},
            }
            descr[p] = {
                "dim_group": TIER2_PROBLEMS[p][2],
                "pct_runs_with_degenerate_iter": round(100.0 * n_runs_with_deg / N_SEEDS, 2),
                "mean_numerics_warnings": float(np.mean(warn_counts)) if warn_counts else 0.0,
            }
        out["arms"][arm] = per_problem
        out["headline"][arm] = descr
    return out


def analysis_P3(runs: dict) -> dict:
    """Scaling of candidate zero / below-tiny fractions vs iteration, per problem."""
    out = {}
    for p in TIER2_PROBLEMS:
        n_init = 2 * TIER2_PROBLEMS[p][1]
        n_iter = T_FULL - n_init
        zero_by_iter = np.full((N_SEEDS, n_iter), np.nan)
        tiny_by_iter = np.full((N_SEEDS, n_iter), np.nan)
        for s in range(N_SEEDS):
            probe = runs[(p, "EI", s)]["probe"]
            if not probe:
                continue
            for i, entry in enumerate(probe[:n_iter]):
                zero_by_iter[s, i] = entry["candidate_pool"]["frac_acqf_zero"]
                tiny_by_iter[s, i] = entry["candidate_pool"]["frac_below_tiny"]
        med_zero = np.nanmedian(zero_by_iter, axis=0)
        med_tiny = np.nanmedian(tiny_by_iter, axis=0)
        valid = np.isfinite(med_zero)
        iters = np.arange(n_iter)[valid]
        if iters.size >= 3:
            rho_z, pz = stats.spearmanr(iters, med_zero[valid])
            rho_t, pt = stats.spearmanr(iters, med_tiny[valid])
        else:
            rho_z = pz = rho_t = pt = float("nan")
        out[p] = {
            "dim_group": TIER2_PROBLEMS[p][2],
            "median_zero_fraction_final": float(med_zero[valid][-1]) if iters.size else None,
            "median_tiny_fraction_final": float(med_tiny[valid][-1]) if iters.size else None,
            "spearman_zero_vs_iter": {"rho": float(rho_z), "p_value": float(pz)},
            "spearman_tiny_vs_iter": {"rho": float(rho_t), "p_value": float(pt)},
        }
    return out


def _mixedlm_simple(df: pd.DataFrame, formula: str, group: str) -> dict:
    md = smf.mixedlm(formula, df, groups=df[group])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            mdf = md.fit()
        except Exception as exc:  # noqa: BLE001
            return {"formula": formula, "converged": False, "error": str(exc), "coefficients": {}}
    conf = mdf.conf_int()
    coeffs = {
        term: {"coef": float(mdf.params[term]), "ci_low": float(conf.loc[term, 0]),
               "ci_high": float(conf.loc[term, 1]), "p_value": float(mdf.pvalues[term])}
        for term in mdf.params.index if term != "Group Var"
    }
    return {"formula": formula, "converged": bool(mdf.converged), "n_obs": int(df.shape[0]),
            "coefficients": coeffs, "warnings": sorted({f"{w.category.__name__}" for w in caught})}


# ----------------------------------------------------------------------------
# Figures (full mode only)
# ----------------------------------------------------------------------------
def fig_cd_tier2(runs: dict) -> None:
    matrix = np.array([
        [float(np.median(precise_cell(runs, p, a, T_FULL))) for a in ARMS]
        for p in TIER2_PROBLEMS
    ])
    fn = friedman_nemenyi(matrix, ARMS)
    t1a.plot_cd_diagram(
        fn["average_ranks"], fn["CD"], FIGURES_DIR / "cd_diagram_tier2_n10.png",
        f"N = {len(TIER2_PROBLEMS)} problem blocks, 30 seeds aggregated by median, "
        f"k = {len(ARMS)} arms, T = 250 (PRECISE track)",
    )


def fig_rank_vs_budget(runs: dict) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(ARMS)))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for a, c in zip(ARMS, colors, strict=True):
        xs, ys = [], []
        for t in CHECKPOINTS:
            probs = problems_at_checkpoint(t)
            matrix = np.array([
                [float(np.median(precise_cell(runs, p, aa, t))) for aa in ARMS] for p in probs
            ])
            ranks = pd.DataFrame(matrix, columns=ARMS).rank(axis=1).mean(axis=0)
            xs.append(t)
            ys.append(float(ranks[a]))
        ax.plot(xs, ys, "o-", color=c, label=a, linewidth=1.7, markersize=5)
    ax.set_xlabel("Evaluations T")
    ax.set_ylabel("Average rank (lower = better)")
    ax.set_xticks(CHECKPOINTS)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Tier 2 average ranks across budget checkpoints (PRECISE)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rank_vs_budget_tier2.png", dpi=150, facecolor="white")
    plt.close()


def _forest(per_problem: dict, value_key: str, ci_key: str, title: str, fname: str) -> None:
    probs = list(per_problem)
    vals = [per_problem[p][value_key] for p in probs]
    cis = [per_problem[p][ci_key] for p in probs]
    ys = np.arange(len(probs))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for y, v, (lo, hi) in zip(ys, vals, cis, strict=True):
        ax.plot([lo, hi], [y, y], color="0.4", linewidth=1.6)
        ax.plot(v, y, "o", color="#c0392b", markersize=6)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(probs, fontsize=9)
    ax.set_xlabel("median (95% CI)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=150, facecolor="white")
    plt.close()


def fig_probe_heatmap(runs: dict) -> None:
    probs = list(TIER2_PROBLEMS)
    bins = np.linspace(0, T_FULL, 11)
    grid = np.full((len(probs), len(bins) - 1), np.nan)
    for r, p in enumerate(probs):
        n_init = 2 * TIER2_PROBLEMS[p][1]
        per_iter = []
        for s in range(N_SEEDS):
            probe = runs[(p, "EI", s)]["probe"]
            if probe:
                per_iter.append([e["candidate_pool"]["frac_acqf_zero"] for e in probe])
        if not per_iter:
            continue
        med = np.nanmedian(np.array([np.pad(x, (0, max(0, (T_FULL - n_init) - len(x))),
                                            constant_values=np.nan)[: T_FULL - n_init]
                                     for x in per_iter], dtype=float), axis=0)
        evals = n_init + np.arange(len(med))
        for b in range(len(bins) - 1):
            mask = (evals >= bins[b]) & (evals < bins[b + 1])
            if mask.any():
                grid[r, b] = np.nanmedian(med[mask])
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(bins) - 1))
    ax.set_xticklabels([f"{int(bins[b])}-{int(bins[b + 1])}" for b in range(len(bins) - 1)],
                       rotation=45, fontsize=8)
    ax.set_yticks(np.arange(len(probs)))
    ax.set_yticklabels(probs, fontsize=9)
    ax.set_xlabel("Evaluation bin")
    ax.set_title("EI candidate zero-fraction (median over seeds)")
    fig.colorbar(im, ax=ax, label="frac_acqf_zero")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "probe_zero_fraction_heatmap.png", dpi=150, facecolor="white")
    plt.close()


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_gates(full: bool) -> tuple[dict, dict, dict]:
    """Run G1-G3 (always); returns (gates, best_death, fopt) for reuse by P1b."""
    exp05 = _load_exp05_json()
    print("Running gates G1-G3 on Tier 1A data ...")
    g12, best_death, fopt = gate_g1_g2(exp05)
    g3 = gate_g3()
    gates = {**g12, "G3_critical_differences": g3}
    for key in ("G1_median_reproduction", "G2_friedman_reproduction", "G3_critical_differences"):
        status = "PASS" if gates[key]["passed"] else "FAIL"
        print(f"  {key}: {status}")
    print(f"    G1 max abs err = {gates['G1_median_reproduction']['max_abs_err']:.2e}")
    print(f"    G2 chi2 = {gates['G2_friedman_reproduction']['chi_square']:.4f} "
          f"(exp05 {gates['G2_friedman_reproduction']['exp05_chi_square']:.4f}, target 50.66)")
    print(f"    G3 CD(k=9)={g3['CD_k9_N10']['computed']:.4f} (3.799), "
          f"CD(k=8)={g3['CD_k8_N10']['computed']:.4f} (3.320)")
    gates["all_core_gates_passed"] = bool(
        gates["G1_median_reproduction"]["passed"]
        and gates["G2_friedman_reproduction"]["passed"]
        and g3["passed"]
    )
    return gates, best_death, fopt


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 2 pre-registered analysis (exp_09).")
    parser.add_argument("--gate-only", action="store_true",
                        help="Run only the validation gates on existing Tier 1A data.")
    args = parser.parse_args()

    gates, best_death, fopt = run_gates(full=not args.gate_only)

    if args.gate_only:
        RESULTS_JSON.parent.mkdir(exist_ok=True)
        with open(RESULTS_JSON, "w") as f:
            json.dump({"mode": "gate-only", "gates": gates}, f, indent=2)
        print(f"\nGate-only results written to {RESULTS_JSON}")
        if not gates["all_core_gates_passed"]:
            sys.exit("Validation gates FAILED.")
        print("All core gates (G1-G3) PASSED.")
        return

    # Full mode: require the exp_08 matrix; fail fast and clean if absent.
    if not MATRIX_DIR.is_dir():
        sys.exit(
            f"Full analysis needs {MATRIX_DIR}, which is absent. "
            "Run the Tier 2 core matrix (exp_08) first, or use --gate-only."
        )
    print(f"\nLoading exp_08 matrix from {MATRIX_DIR} ...")
    runs = load_matrix(MATRIX_DIR)
    g4 = gate_g4_manifest(runs)
    gates["G4_manifest"] = g4
    print(f"  G4 manifest: {'PASS' if g4['passed'] else 'FAIL'} "
          f"({g4['n_files']}/{g4['n_files_expected']} files, "
          f"{g4['n_nan_final_regrets']} NaN finals)")
    if not (gates["all_core_gates_passed"] and g4["passed"]):
        sys.exit("Gates FAILED; aborting before analysis.")

    FIGURES_DIR.mkdir(exist_ok=True)
    print("Analysis A (cross-problem inference) ...")
    a = analysis_A(runs)
    print("Analysis B (claim verdicts C1/C2/C4) ...")
    b = analysis_B(runs)
    print("Analysis C (mechanism P1a/P1b/P2/P3) ...")
    c = {
        "P1a_e3_identification": analysis_P1a(runs),
        "P1b_optimizer_mediation": analysis_P1b(runs, best_death, fopt),
        "P2_underflow_coupling": analysis_P2(runs),
        "P3_scaling": analysis_P3(runs),
    }

    print("Figures ...")
    fig_cd_tier2(runs)
    fig_rank_vs_budget(runs)
    _forest(c["P1a_e3_identification"]["per_problem"], "median_delta", "delta_ci",
            "P1a: log regret(EI) - log regret(LogEI) at T=250", "p1a_forest_by_problem.png")
    _forest(c["P1b_optimizer_mediation"]["per_problem"], "median_D", "D_ci",
            "P1b: difference-in-differences (DEATH track) at T=250", "p1b_did_by_problem.png")
    fig_probe_heatmap(runs)

    summary = {
        "mode": "full",
        "gates": gates,
        "analysis_A_cross_problem": a,
        "analysis_B_claim_verdicts": b,
        "analysis_C_mechanism": c,
    }
    RESULTS_JSON.parent.mkdir(exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {RESULTS_JSON}")
    print("Done.")


if __name__ == "__main__":
    main()
