"""
Experiment 03: Friedman + Nemenyi statistical analysis across all benchmarks.

Loads the JSON result files produced by exp_02_strategies_per_problem.py for
each problem in PROBLEMS, extracts final regret per (problem, strategy, seed),
runs Friedman + Nemenyi post-hoc tests, and draws a critical difference diagram.

Usage:
    python experiments/exp_03_friedman_nemenyi.py

Outputs:
    figures/exp_03_cd_diagram.png
    results/exp_03_stats_summary.json
"""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import studentized_range

warnings.filterwarnings("ignore")

PROBLEMS = ["branin", "hartmann6", "ackley", "sixhumpcamel", "borehole", "piston"]
STRATEGIES = ["EI", "UCB", "Uncertainty", "Random"]
ALPHA = 0.05


def load_results(results_dir: Path) -> dict:
    """Load all exp_02_*.json files into a dict[problem][strategy] = trajectory array."""
    data = {}
    for problem in PROBLEMS:
        path = results_dir / f"exp_02_{problem}_strategies.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run exp_02_strategies_per_problem.py first."
            )
        with open(path) as f:
            raw = json.load(f)
        data[problem] = {
            strat: np.array(raw["results"][strat]) for strat in STRATEGIES
        }
        n_seeds, n_iter_plus_1 = data[problem]["EI"].shape
        print(f"  Loaded {path.name}  (n_seeds={n_seeds}, n_iter+1={n_iter_plus_1})")
    return data


def build_friedman_matrix(data: dict):
    """Build the (n_blocks, n_strategies) Friedman input matrix."""
    rows = []
    labels = []
    for problem in PROBLEMS:
        n_seeds = data[problem]["EI"].shape[0]
        for seed_idx in range(n_seeds):
            row = [data[problem][strat][seed_idx, -1] for strat in STRATEGIES]
            rows.append(row)
            labels.append(f"{problem}_seed{seed_idx}")
    matrix = np.array(rows)
    print(f"  Friedman input shape: {matrix.shape}")
    return matrix, labels


def run_friedman(matrix: np.ndarray) -> dict:
    """Run Friedman omnibus test."""
    chi2, p = stats.friedmanchisquare(
        *[matrix[:, i] for i in range(matrix.shape[1])]
    )
    return {
        "chi_square": float(chi2),
        "p_value": float(p),
        "df": matrix.shape[1] - 1,
        "rejected_H0": bool(p < ALPHA),
    }


def run_nemenyi(matrix: np.ndarray) -> pd.DataFrame:
    """Run Nemenyi post-hoc, return strategy x strategy p-value matrix."""
    pvalues = sp.posthoc_nemenyi_friedman(matrix)
    pvalues.columns = STRATEGIES
    pvalues.index = STRATEGIES
    return pvalues


def compute_avg_ranks(matrix: np.ndarray) -> pd.Series:
    """Average rank per strategy across all blocks (lower = better)."""
    ranks = pd.DataFrame(matrix, columns=STRATEGIES).rank(axis=1)
    return ranks.mean(axis=0)


def plot_cd_diagram(avg_ranks, nemenyi_pvalues, n_blocks, output_path,
                    n_problems, k, alpha=ALPHA):
    """Generate and save a Demsar-style CD diagram.

    Draws the rank axis, a critical-difference (CD) ruler, and clique bars
    connecting strategies whose average-rank gap is below the CD (i.e. not
    significantly different under Nemenyi). The CD is computed analytically
    from the studentized range statistic rather than read off the post-hoc
    matrix, so it stays correct if N or k change.
    """
    # Nemenyi critical difference: CD = q_alpha * sqrt(k(k+1) / 6N)
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    CD = q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))

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

    # Clique bars: connect adjacent strategies whose rank gap < CD.
    y_clique = -0.10
    for i in range(len(vals) - 1):
        if abs(vals[i] - vals[i + 1]) < CD:
            ax.plot([vals[i], vals[i + 1]], [y_clique, y_clique],
                    color="#c0392b", linewidth=5, solid_capstyle="round", zorder=3)

    # CD ruler (top).
    y_cd = y_axis + 0.42
    cd_x0 = hi - 0.05
    ax.plot([cd_x0, cd_x0 - CD], [y_cd, y_cd], color="black", linewidth=1.8)
    for xx in (cd_x0, cd_x0 - CD):
        ax.plot([xx, xx], [y_cd - 0.04, y_cd + 0.04], color="black", linewidth=1.8)
    ax.text(cd_x0 - CD / 2, y_cd + 0.07, f"CD = {CD:.2f}", ha="center",
            va="bottom", fontsize=12, fontweight="bold")

    ax.text(mid, y_axis + 0.74,
            f"Critical Difference Diagram (Nemenyi, $\\alpha$={alpha})",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(mid, y_axis + 0.62,
            f"N = {n_blocks} blocks ({n_problems} problems "
            f"$\\times$ {n_blocks // n_problems} seeds), k = {k} strategies",
            ha="center", fontsize=10, color="0.3")

    ax.set_xlim(hi + 2.4, lo - 2.4)  # inverted: best rank on the right
    ax.set_ylim(-0.78, 0.97)
    ax.axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.05)
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="white")
    plt.close()
    print(f"  CD diagram saved to {output_path}  (CD={CD:.4f})")


def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    print(f"Loading results from {results_dir}/")
    data = load_results(results_dir)

    print("\nBuilding Friedman matrix...")
    matrix, labels = build_friedman_matrix(data)

    print("\nRunning Friedman omnibus test...")
    friedman_result = run_friedman(matrix)
    print(f"  Chi-square: {friedman_result['chi_square']:.4f}")
    print(f"  df:         {friedman_result['df']}")
    print(f"  p-value:    {friedman_result['p_value']:.6e}")
    print(f"  Reject H0:  {friedman_result['rejected_H0']}  (alpha={ALPHA})")

    print("\nRunning Nemenyi post-hoc...")
    nemenyi_pvalues = run_nemenyi(matrix)
    print(nemenyi_pvalues.round(6).to_string())

    avg_ranks = compute_avg_ranks(matrix)
    print("\nAverage ranks (lower = better):")
    for strat, rank in avg_ranks.items():
        print(f"  {strat:<10} {rank:.4f}")

    print("\nGenerating CD diagram...")
    cd_path = figures_dir / "exp_03_cd_diagram.png"
    plot_cd_diagram(
        avg_ranks, nemenyi_pvalues,
        n_blocks=matrix.shape[0],
        output_path=cd_path,
        n_problems=len(PROBLEMS),
        k=matrix.shape[1],
    )

    summary = {
        "config": {
            "problems": PROBLEMS,
            "strategies": STRATEGIES,
            "alpha": ALPHA,
            "n_blocks": int(matrix.shape[0]),
        },
        "friedman": friedman_result,
        "nemenyi_pvalues": nemenyi_pvalues.round(6).to_dict(),
        "average_ranks": avg_ranks.round(4).to_dict(),
    }
    summary_path = project_root / "results" / "exp_03_stats_summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Statistical summary saved to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
