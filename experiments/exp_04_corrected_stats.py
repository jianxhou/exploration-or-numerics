"""
Experiment 04: corrected cross-problem statistical analysis.

The published exp_03 analysis treats the N = 60 (problem x seed) pairs as
independent Friedman blocks. That is pseudo-replication: the 10 seeds within a
problem are repeated measures of the *same* benchmark, not 10 independent
datasets. Demsar (2006) requires the blocks to be independent datasets, so the
correct unit of analysis is the problem (N = 6), not the problem-seed pair.

This script keeps exp_03 untouched and adds the statistically defensible
analyses alongside it:

  Validation gate  reproduce the published N = 60 Friedman result from the same
                   six exp_02 files, as a guard that the loader is correct.
  Analysis A       Demsar-strict Friedman + Nemenyi on the 6 x 4 matrix obtained
                   by aggregating the 10 seeds per (problem, strategy) by median.
  Analysis B       per-problem Friedman across the 10 seeds (k = 4 strategies).
  Analysis C       linear mixed-effects model on log final regret with a random
                   intercept per problem (the seeds are the within-problem
                   replicates).

Usage:
    python experiments/exp_04_corrected_stats.py

Outputs:
    results/exp_04_corrected_stats.json
    figures/cd_diagram_n6.png
"""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import statsmodels.formula.api as smf
from scipy import stats

# Problem order is irrelevant to the rank statistics (blocks are exchangeable);
# this is just the canonical order used throughout the project.
PROBLEMS = ["branin", "sixhumpcamel", "hartmann6", "piston", "borehole", "ackley"]
STRATEGIES = ["EI", "UCB", "Uncertainty", "Random"]
ALPHA = 0.05

# Nemenyi critical-difference constants for Analysis A (k = 4, N = 6, alpha = .05).
# q_alpha is the studentized-range quantile divided by sqrt(2) for k = 4 groups.
K = 4
N_PROBLEMS = 6
Q_ALPHA = 2.569
CD_N6 = Q_ALPHA * np.sqrt(K * (K + 1) / (6.0 * N_PROBLEMS))  # = 1.915

# Published N = 60 reference values (from results/exp_03_stats_summary.json).
N60_EXPECTED = {
    "chi_square": 128.815699658703,
    "p_value": 9.733829173297537e-28,
    "ranks": {"EI": 1.7583, "UCB": 1.3917, "Uncertainty": 3.3750, "Random": 3.4750},
}


def load_final_regrets(results_dir: Path) -> pd.DataFrame:
    """Load final regret per (problem, strategy, seed) from the six exp_02 files.

    The final regret is the LAST element of each seed's regret trajectory. Only
    the exp_02_<problem>_strategies.json files are read; exp_01_* and the older
    01_branin_ei_baseline.json are deliberately ignored.
    """
    records = []
    for problem in PROBLEMS:
        path = results_dir / f"exp_02_{problem}_strategies.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run exp_02_strategies_per_problem.py first."
            )
        with open(path) as f:
            raw = json.load(f)
        for strat in STRATEGIES:
            traj = np.asarray(raw["results"][strat], dtype=float)
            n_seeds = traj.shape[0]
            for seed_idx in range(n_seeds):
                records.append(
                    {
                        "problem": problem,
                        "strategy": strat,
                        "seed": seed_idx,
                        "final_regret": float(traj[seed_idx, -1]),
                    }
                )
        n_seeds = np.asarray(raw["results"]["EI"]).shape[0]
        print(f"  Loaded {path.name}  (n_seeds={n_seeds}, strategies={len(STRATEGIES)})")
    df = pd.DataFrame.from_records(records)
    print(
        f"  Long-format table: {len(df)} rows "
        f"({df['problem'].nunique()} problems x {df['strategy'].nunique()} "
        f"strategies x {df['seed'].nunique()} seeds)"
    )
    return df


def _avg_ranks(matrix: np.ndarray) -> dict:
    """Average rank per strategy across the rows of `matrix` (rank 1 = best)."""
    ranks = pd.DataFrame(matrix, columns=STRATEGIES).rank(axis=1)
    return {s: float(r) for s, r in ranks.mean(axis=0).items()}


def _friedman(matrix: np.ndarray) -> dict:
    chi2, p = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    return {
        "chi_square": float(chi2),
        "p_value": float(p),
        "df": matrix.shape[1] - 1,
        "rejected_H0": bool(p < ALPHA),
    }


def validation_gate(df: pd.DataFrame) -> dict:
    """Reproduce the published N = 60 Friedman analysis as a correctness guard.

    Returns the reproduction dict. Raises RuntimeError (after printing a
    diagnostic) if the result deviates from the published values beyond rounding.
    """
    # Build the 60 x 4 matrix: one block per (problem, seed).
    pivot = df.pivot_table(
        index=["problem", "seed"], columns="strategy", values="final_regret"
    )[STRATEGIES]
    matrix = pivot.to_numpy()

    friedman = _friedman(matrix)
    ranks = _avg_ranks(matrix)

    chi2_ok = np.isclose(friedman["chi_square"], N60_EXPECTED["chi_square"], atol=1e-2)
    p_ok = np.isclose(friedman["p_value"], N60_EXPECTED["p_value"], rtol=1e-3, atol=0.0)
    ranks_ok = all(
        np.isclose(ranks[s], N60_EXPECTED["ranks"][s], atol=1e-3) for s in STRATEGIES
    )
    passed = bool(matrix.shape == (60, 4) and chi2_ok and p_ok and ranks_ok)

    if not passed:
        print("\n!!! VALIDATION GATE FAILED — diagnostic dump !!!")
        print(f"  Loaded matrix shape: {matrix.shape} (expected (60, 4))")
        print(f"  Friedman chi2 = {friedman['chi_square']:.6f} "
              f"(expected {N60_EXPECTED['chi_square']:.6f})")
        print(f"  Friedman p    = {friedman['p_value']:.6e} "
              f"(expected {N60_EXPECTED['p_value']:.6e})")
        print("  Average ranks (loaded vs expected):")
        for s in STRATEGIES:
            print(f"    {s:<12} {ranks[s]:.4f}  vs  {N60_EXPECTED['ranks'][s]:.4f}")
        print("  Block index (first rows):")
        print(pivot.head(8).to_string())
        raise RuntimeError(
            "N=60 reproduction does not match published exp_03 values; aborting "
            "before the new analyses. See diagnostic above."
        )

    return {
        "description": "Reproduction of published exp_03 N=60 (problem x seed) Friedman.",
        "n_blocks": int(matrix.shape[0]),
        "friedman": friedman,
        "average_ranks": {s: round(ranks[s], 4) for s in STRATEGIES},
        "validation_passed": passed,
    }


def analysis_a(df: pd.DataFrame) -> dict:
    """Demsar-strict Friedman + Nemenyi on the 6 x 4 median-aggregated matrix."""
    median = (
        df.groupby(["problem", "strategy"])["final_regret"].median().unstack("strategy")
    )
    median = median.loc[PROBLEMS, STRATEGIES]
    matrix = median.to_numpy()

    friedman = _friedman(matrix)
    ranks = _avg_ranks(matrix)

    nemenyi = sp.posthoc_nemenyi_friedman(matrix)
    nemenyi.columns = STRATEGIES
    nemenyi.index = STRATEGIES

    return {
        "description": "Friedman + Nemenyi on 6 problems, 10 seeds aggregated by median.",
        "aggregation": "median",
        "n_blocks": int(matrix.shape[0]),
        "k": int(matrix.shape[1]),
        "median_matrix": {
            p: {s: float(median.loc[p, s]) for s in STRATEGIES} for p in PROBLEMS
        },
        "friedman": friedman,
        "average_ranks": {s: round(ranks[s], 4) for s in STRATEGIES},
        "nemenyi_pvalues": nemenyi.round(6).to_dict(),
        "q_alpha": Q_ALPHA,
        "CD": round(float(CD_N6), 4),
    }


def analysis_b(df: pd.DataFrame) -> dict:
    """Per-problem Friedman across the 10 seeds (k = 4 strategies)."""
    out = {}
    for problem in PROBLEMS:
        sub = df[df["problem"] == problem]
        pivot = sub.pivot_table(index="seed", columns="strategy", values="final_regret")[
            STRATEGIES
        ]
        matrix = pivot.to_numpy()
        friedman = _friedman(matrix)
        ranks = _avg_ranks(matrix)
        out[problem] = {
            "n_seeds": int(matrix.shape[0]),
            "friedman": friedman,
            "mean_ranks": {s: round(ranks[s], 4) for s in STRATEGIES},
        }
    return out


def analysis_c(df: pd.DataFrame) -> dict:
    """Linear mixed-effects model: log final regret ~ strategy, random intercept/problem.

    Treatment coding with reference 'Random'; the 10 seeds are the within-problem
    replicates. With only 6 groups a ConvergenceWarning from statsmodels is
    expected — it is captured and reported rather than silently suppressed.
    """
    work = df.copy()
    work["y"] = np.log(work["final_regret"] + 1e-6)

    formula = "y ~ C(strategy, Treatment(reference='Random'))"
    md = smf.mixedlm(formula, work, groups=work["problem"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mdf = md.fit()
    warn_msgs = [f"{w.category.__name__}: {w.message}" for w in caught]

    conf = mdf.conf_int()
    label_map = {
        "EI": "C(strategy, Treatment(reference='Random'))[T.EI]",
        "UCB": "C(strategy, Treatment(reference='Random'))[T.UCB]",
        "Uncertainty": "C(strategy, Treatment(reference='Random'))[T.Uncertainty]",
    }
    coefficients = {}
    for strat, term in label_map.items():
        coefficients[strat] = {
            "coef": float(mdf.params[term]),
            "std_err": float(mdf.bse[term]),
            "ci_low": float(conf.loc[term, 0]),
            "ci_high": float(conf.loc[term, 1]),
            "p_value": float(mdf.pvalues[term]),
            "ci_excludes_zero": bool(conf.loc[term, 0] > 0 or conf.loc[term, 1] < 0),
        }

    return {
        "description": (
            "MixedLM on log(final_regret + 1e-6); fixed effect = strategy "
            "(Treatment, ref=Random); random intercept grouped by problem."
        ),
        "formula": formula,
        "reference": "Random",
        "response": "log(final_regret + 1e-6)",
        "n_obs": int(len(work)),
        "n_groups": int(work["problem"].nunique()),
        "intercept_random_var": float(mdf.cov_re.iloc[0, 0]),
        "coefficients": coefficients,
        "convergence_warnings": warn_msgs,
    }


def plot_cd_diagram_n6(avg_ranks: dict, cd: float, output_path: Path):
    """Demsar-style CD diagram for the N = 6 analysis, in the exp_03 visual style.

    Clique bars connect maximal groups of strategies whose pairwise average-rank
    gap stays below the critical difference (i.e. not significantly different
    under Nemenyi). Maximal cliques are used rather than the adjacent-pair
    shortcut, so a wide chain of overlapping-but-not-all-within-CD strategies is
    rendered as separate bars instead of one continuous bar.
    """
    ranks_sorted = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    names = [n for n, _ in ranks_sorted]
    vals = [v for _, v in ranks_sorted]

    lo = float(np.floor(min(vals)))
    hi = float(np.ceil(max(vals)))
    mid = (lo + hi) / 2

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_axis = 0.0
    ax.plot([lo, hi], [y_axis, y_axis], color="black", linewidth=1.6, zorder=1)
    for t in np.arange(lo, hi + 0.01, 0.5):
        ax.plot([t, t], [y_axis, y_axis + 0.05], color="black", linewidth=1.2)
        ax.text(t, y_axis + 0.13, f"{t:.1f}", ha="center", va="bottom", fontsize=12)

    right = [(n, v) for n, v in zip(names, vals, strict=True) if v <= mid]
    left = [(n, v) for n, v in zip(names, vals, strict=True) if v > mid]
    col_r = lo - 0.30
    col_l = hi + 0.30

    def _draw(rank, name, side, level):
        y = -0.18 - 0.22 * level
        ax.plot([rank, rank], [y_axis, y], color="0.45", linewidth=1.4)
        if side == "right":
            ax.plot([rank, col_r], [y, y], color="0.45", linewidth=1.4)
            ax.text(col_r, y, f"  {name} ({rank:.2f})", ha="left",
                    va="center", fontsize=12)
        else:
            ax.plot([rank, col_l], [y, y], color="0.45", linewidth=1.4)
            ax.text(col_l, y, f"({rank:.2f}) {name}  ", ha="right",
                    va="center", fontsize=12)

    for lvl, (n, v) in enumerate(right):
        _draw(v, n, "right", lvl)
    for lvl, (n, v) in enumerate(left):
        _draw(v, n, "left", lvl)

    # Maximal clique bars: each bar spans the widest run [i, j] whose endpoints
    # differ by less than CD, with any bar contained in another dropped.
    n = len(vals)
    spans = []
    for i in range(n):
        j = i
        while j + 1 < n and (vals[j + 1] - vals[i]) < cd:
            j += 1
        if j > i:
            spans.append((i, j))
    maximal = [
        (a, b)
        for (a, b) in spans
        if not any(a2 <= a and b <= b2 and (a2, b2) != (a, b) for (a2, b2) in spans)
    ]

    for level, (i, j) in enumerate(maximal):
        y_clique = -0.07 - 0.06 * level
        ax.plot([vals[i], vals[j]], [y_clique, y_clique],
                color="#c0392b", linewidth=5, solid_capstyle="round", zorder=3)

    # CD ruler (top).
    y_cd = y_axis + 0.42
    cd_x0 = hi - 0.05
    ax.plot([cd_x0, cd_x0 - cd], [y_cd, y_cd], color="black", linewidth=1.8)
    for xx in (cd_x0, cd_x0 - cd):
        ax.plot([xx, xx], [y_cd - 0.04, y_cd + 0.04], color="black", linewidth=1.8)
    ax.text(cd_x0 - cd / 2, y_cd + 0.07, f"CD = {cd:.3f}", ha="center",
            va="bottom", fontsize=12, fontweight="bold")

    ax.text(mid, y_axis + 0.74,
            f"Critical Difference Diagram (Nemenyi, $\\alpha$={ALPHA})",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(mid, y_axis + 0.62,
            f"N = {N_PROBLEMS} blocks (problems), seeds aggregated by median, "
            f"$\\alpha$ = {ALPHA}",
            ha="center", fontsize=10, color="0.3")

    ax.set_xlim(hi + 2.4, lo - 2.4)  # inverted: best rank on the right
    ax.set_ylim(-0.78, 0.97)
    ax.axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.05)
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="white")
    plt.close()
    print(f"  CD diagram saved to {output_path}  (CD={cd:.4f})")


def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    print(f"Loading final regrets from {results_dir}/ (exp_02 files only)")
    df = load_final_regrets(results_dir)

    print("\n=== VALIDATION GATE: reproduce published N=60 analysis ===")
    n60 = validation_gate(df)
    print(f"  chi2(3) = {n60['friedman']['chi_square']:.4f}  "
          f"p = {n60['friedman']['p_value']:.6e}")
    print("  average ranks: " + ", ".join(
        f"{s}={n60['average_ranks'][s]:.4f}" for s in STRATEGIES))
    print("  PASSED — loader matches published exp_03; proceeding.\n")

    print("=== Analysis A: Demsar-strict (N=6, seeds aggregated by median) ===")
    a = analysis_a(df)
    print(f"  Friedman chi2({a['friedman']['df']}) = "
          f"{a['friedman']['chi_square']:.4f}  p = {a['friedman']['p_value']:.4f}")
    print("  average ranks: " + ", ".join(
        f"{s}={a['average_ranks'][s]:.4f}" for s in STRATEGIES))
    print(f"  CD (q_alpha={Q_ALPHA}, k={K}, N={N_PROBLEMS}) = {a['CD']:.3f}")
    print("  Nemenyi post-hoc p-values:")
    print(pd.DataFrame(a["nemenyi_pvalues"]).loc[STRATEGIES, STRATEGIES]
          .round(4).to_string().replace("\n", "\n    ").rjust(0))
    print()

    print("=== Analysis B: per-problem Friedman (k=4, 10 seeds) ===")
    b = analysis_b(df)
    for problem in PROBLEMS:
        r = b[problem]
        ranks_str = ", ".join(f"{s}={r['mean_ranks'][s]:.2f}" for s in STRATEGIES)
        print(f"  {problem:<13} chi2={r['friedman']['chi_square']:.3f}  "
              f"p={r['friedman']['p_value']:.2e}  | ranks: {ranks_str}")
    print()

    print("=== Analysis C: linear mixed-effects model (random intercept/problem) ===")
    c = analysis_c(df)
    print(f"  Response: {c['response']}   reference: {c['reference']}   "
          f"n_obs={c['n_obs']}, n_groups={c['n_groups']}")
    for strat in ("EI", "UCB", "Uncertainty"):
        co = c["coefficients"][strat]
        print(f"  {strat:<12} coef={co['coef']:+.4f}  se={co['std_err']:.4f}  "
              f"95% CI=[{co['ci_low']:+.4f}, {co['ci_high']:+.4f}]  "
              f"{'(excludes 0)' if co['ci_excludes_zero'] else '(includes 0)'}")
    if c["convergence_warnings"]:
        print("  NOTE: statsmodels emitted the following warning(s) during fit "
              "(expected with only 6 groups; reported, not suppressed):")
        for w in c["convergence_warnings"]:
            print(f"    - {w}")
    else:
        print("  NOTE: no convergence warning emitted.")
    print()

    print("Generating N=6 CD diagram...")
    cd_path = figures_dir / "cd_diagram_n6.png"
    plot_cd_diagram_n6(a["average_ranks"], CD_N6, cd_path)

    summary = {
        "config": {
            "problems": PROBLEMS,
            "strategies": STRATEGIES,
            "alpha": ALPHA,
            "note": (
                "Corrected analyses treating the problem (not problem x seed) as "
                "the independent block, per Demsar (2006). exp_03 is left intact "
                "for comparison."
            ),
        },
        "n60_reproduction": n60,
        "analysis_A_demsar_n6": a,
        "analysis_B_per_problem": b,
        "analysis_C_mixedlm": c,
        "critical_difference_n6": round(float(CD_N6), 4),
    }
    summary_path = results_dir / "exp_04_corrected_stats.json"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results written to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
