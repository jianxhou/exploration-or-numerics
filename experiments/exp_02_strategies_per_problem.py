"""
Experiment 02: compare acquisition strategies on a given problem.

Same protocol as exp_01, with the problem taken as a command-line argument.

Usage:
    python experiments/exp_02_strategies_per_problem.py --problem Branin
    python experiments/exp_02_strategies_per_problem.py --problem Hartmann6 --n-iter 30
    python experiments/exp_02_strategies_per_problem.py --problem Ackley --n-seeds 5
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from death_suite import DEATH_PROBLEMS

from al_benchmark.core.bo_loop import run_bo
from al_benchmark.problems.engineering import Borehole, Piston
from al_benchmark.problems.synthetic import Ackley, Branin, Hartmann6, SixHumpCamel
from al_benchmark.strategies.ei import EI
from al_benchmark.strategies.random_strategy import Random
from al_benchmark.strategies.ucb import UCB
from al_benchmark.strategies.uncertainty import Uncertainty

warnings.filterwarnings("ignore")

PROBLEMS = {
    "Branin": lambda: Branin(),
    "Hartmann6": lambda: Hartmann6(),
    "Ackley": lambda: Ackley(),
    "SixHumpCamel": lambda: SixHumpCamel(),
    "Borehole": lambda: Borehole(),
    "Piston": lambda: Piston(),
    # Ported De Ath synthetic suite (Step 3b); classes are callable factories.
    **DEATH_PROBLEMS,
}

STRATEGIES = {
    "EI": lambda: EI(),
    "UCB": lambda: UCB(beta=2.0),
    "Uncertainty": lambda: Uncertainty(),
    "Random": lambda: Random(),
}

COLORS = {"EI": "#1f77b4", "UCB": "#2ca02c", "Uncertainty": "#9467bd", "Random": "#d62728"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        choices=list(PROBLEMS.keys()),
        help="Benchmark problem name",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of random seeds (default: 10)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of BO iterations after initial design (default: 20)",
    )
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))
    print(f"Running {len(STRATEGIES)} strategies x {args.n_seeds} seeds on {args.problem}")
    print(f"  Iterations per run: {args.n_iter}")
    print(f"  Seeds: {seeds}\n")

    results = {}
    for strat_name, strat_factory in STRATEGIES.items():
        print(f"  {strat_name}:")
        regret_traj = []
        for seed in seeds:
            result = run_bo(
                problem=PROBLEMS[args.problem](),
                strategy=strat_factory(),
                seed=seed,
                n_iter=args.n_iter,
                verbose=False,
            )
            regret_traj.append(result.regret_history)
            print(f"    seed {seed:2d}: final regret = {result.final_regret:.4f}")
        results[strat_name] = np.array(regret_traj)

    stats = {}
    for strat_name, traj in results.items():
        stats[strat_name] = {
            "mean": traj.mean(axis=0),
            "std": traj.std(axis=0),
            "median": np.median(traj, axis=0),
            "iqr_lo": np.percentile(traj, 25, axis=0),
            "iqr_hi": np.percentile(traj, 75, axis=0),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, scale in zip(axes, ["linear", "log"], strict=True):
        for strat_name in results:
            s = stats[strat_name]
            x = np.arange(len(s["mean"]))
            ax.plot(x, s["mean"], label=strat_name, color=COLORS[strat_name], linewidth=1.8)
            ax.fill_between(
                x, s["mean"] - s["std"], s["mean"] + s["std"],
                alpha=0.2, color=COLORS[strat_name],
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Simple regret")
        ax.set_title(f"Mean +/- 1 std ({scale} scale)")
        ax.grid(True, alpha=0.3)
        if scale == "log":
            ax.set_yscale("log")
        ax.legend(loc="upper right" if scale == "linear" else "lower left", fontsize=10)

    plt.suptitle(
        f"BO strategy comparison on {args.problem} ({args.n_seeds} seeds, {args.n_iter} iters)",
        y=1.02,
        fontsize=12,
    )
    plt.tight_layout()

    project_root = Path(__file__).parent.parent
    tag = args.problem.lower()
    fig_path = project_root / "figures" / f"exp_02_{tag}_strategies.png"
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")

    results_path = project_root / "results" / f"exp_02_{tag}_strategies.json"
    results_path.parent.mkdir(exist_ok=True)
    serialized = {
        "config": {
            "problem": args.problem,
            "seeds": seeds,
            "n_iter": args.n_iter,
        },
        "results": {name: traj.tolist() for name, traj in results.items()},
    }
    with open(results_path, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"Results JSON saved to {results_path}")

    print(f"\n=== Summary on {args.problem} (final regret) ===")
    print(f"{'Strategy':<12} {'Mean':>10} {'Std':>10} {'Median':>10}")
    print("-" * 44)
    for strat_name in results:
        finals = results[strat_name][:, -1]
        print(
            f"{strat_name:<12} {finals.mean():>10.4f} {finals.std():>10.4f} {np.median(finals):>10.4f}" # noqa: E501
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
