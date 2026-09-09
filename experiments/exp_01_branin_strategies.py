"""
Experiment 01: compare EI, UCB, Random on Branin across multiple seeds.

Outputs a regret-curve figure (mean + std band per strategy) and a JSON
file with the raw trajectories.
"""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from al_benchmark.core.bo_loop import run_bo
from al_benchmark.problems.synthetic import Branin
from al_benchmark.strategies.ei import EI
from al_benchmark.strategies.random_strategy import Random
from al_benchmark.strategies.ucb import UCB

warnings.filterwarnings("ignore")

SEEDS = list(range(10))
N_ITER = 20
STRATEGIES = [
    ("EI", lambda: EI()),
    ("UCB", lambda: UCB(beta=2.0)),
    ("Random", lambda: Random()),
]
PROBLEM_FACTORY = lambda: Branin()

# results[strategy_name]: array of shape (n_seeds, n_init + n_iter + 1)
print(f"Running {len(STRATEGIES)} strategies x {len(SEEDS)} seeds on Branin...")
results = {}

for strat_name, strat_factory in STRATEGIES:
    print(f"\n  {strat_name}:")
    regret_traj = []
    for seed in SEEDS:
        result = run_bo(
            problem=PROBLEM_FACTORY(),
            strategy=strat_factory(),
            seed=seed,
            n_iter=N_ITER,
            verbose=False,
        )
        regret_traj.append(result.regret_history)
        print(f"    seed {seed:2d}: final regret = {result.final_regret:.4f}")
    results[strat_name] = np.array(regret_traj)

print("\nComputing statistics across seeds...")
stats = {}
for strat_name, traj in results.items():
    stats[strat_name] = {
        "mean": traj.mean(axis=0),
        "std": traj.std(axis=0),
        "median": np.median(traj, axis=0),
        "iqr_lo": np.percentile(traj, 25, axis=0),
        "iqr_hi": np.percentile(traj, 75, axis=0),
    }

print("\nPlotting...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = {"EI": "#1f77b4", "UCB": "#2ca02c", "Random": "#d62728"}

for ax, scale in zip(axes, ["linear", "log"]):
    for strat_name in results:
        s = stats[strat_name]
        x = np.arange(len(s["mean"]))
        ax.plot(x, s["mean"], label=strat_name, color=colors[strat_name], linewidth=1.8)
        ax.fill_between(
            x, s["mean"] - s["std"], s["mean"] + s["std"],
            alpha=0.2, color=colors[strat_name],
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Simple regret")
    ax.set_title(f"Mean +/- 1 std ({scale} scale)")
    ax.grid(True, alpha=0.3)
    if scale == "log":
        ax.set_yscale("log")
    ax.legend(loc="upper right" if scale == "linear" else "lower left", fontsize=10)

plt.suptitle(
    f"BO strategy comparison on Branin ({len(SEEDS)} seeds, {N_ITER} iters)",
    y=1.02,
    fontsize=12,
)
plt.tight_layout()

project_root = Path(__file__).parent.parent
fig_path = project_root / "figures" / "exp_01_branin_strategies.png"
fig_path.parent.mkdir(exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved to {fig_path}")

results_path = project_root / "results" / "exp_01_branin_strategies.json"
results_path.parent.mkdir(exist_ok=True)
serialized = {
    "config": {"seeds": SEEDS, "n_iter": N_ITER, "problem": "Branin"},
    "results": {name: traj.tolist() for name, traj in results.items()},
}
with open(results_path, "w") as f:
    json.dump(serialized, f, indent=2)
print(f"Results JSON saved to {results_path}")

print("\n=== Summary (final regret) ===")
print(f"{'Strategy':<12} {'Mean':>10} {'Std':>10} {'Median':>10}")
print("-" * 44)
for strat_name in results:
    finals = results[strat_name][:, -1]
    print(f"{strat_name:<12} {finals.mean():>10.4f} {finals.std():>10.4f} {np.median(finals):>10.4f}")

print("\nDone.")
