"""
Experiment 02b: normalization-OFF ablation on Piston (7D).

Purpose: archive the diagnostic run behind Table `tab:normalization`'s
"Before norm." column. The original first-pass diagnostic (lab_notebook
line 164: EI 0.41 / UCB 0.39 / Random 0.40) was never saved to results/.
This reruns it under the exact main-experiment protocol, removing only the
GP input normalization.

Protocol (identical to exp_02 except input normalization):
  - 10 seeds (0..9), 20 BO iterations, n_init = 2d = 14, Sobol initial design.
  - GP surrogate = SingleTaskGP with Standardize(m=1) outputs but NO
    input_transform=Normalize  (the single ablated knob; cf. Phase 2 probe).
  - Strategies EI, UCB(beta=2.0), Random; optimize_acqf(num_restarts=10,
    raw_samples=64) via the repo's own strategy classes.

Writes results/exp_02b_piston_no_norm.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from al_benchmark.problems.engineering import Piston  # noqa: E402
from al_benchmark.strategies.ei import EI  # noqa: E402
from al_benchmark.strategies.random_strategy import Random  # noqa: E402
from al_benchmark.strategies.ucb import UCB  # noqa: E402


def fit_gp_no_norm(train_x, train_y):
    """SingleTaskGP with output Standardize but NO input Normalize."""
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def run_bo_no_norm(problem, strategy, seed, n_iter=20):
    """Mirror of al_benchmark.core.bo_loop.run_bo, GP without input Normalize."""
    torch.manual_seed(seed)
    n_init = 2 * problem.dim
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n_init, q=1).squeeze(1)
    train_y = problem(train_x).unsqueeze(-1)
    regret_history = [problem.regret(train_y.max().item())]
    printed = False
    for _ in range(n_iter):
        model = fit_gp_no_norm(train_x, train_y)
        if not printed:
            # one-time hyperparameter dump to check for an outputscale param
            names = [n for n, _ in model.named_parameters()]
            globals()["_PARAM_NAMES"] = names
            printed = True
        candidate = strategy.select_next(
            model=model, bounds=problem.bounds, train_x=train_x, train_y=train_y
        )
        new_y = problem(candidate).unsqueeze(-1)
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])
        regret_history.append(problem.regret(train_y.max().item()))
    return regret_history


def main():
    problem = Piston()
    seeds = list(range(10))
    strat_factories = {
        "EI": lambda: EI(),
        "UCB": lambda: UCB(beta=2.0),
        "Random": lambda: Random(),
    }

    results = {}
    print("Normalization-OFF Piston (7D), 10 seeds x 20 iter, n_init=14\n")
    for name, factory in strat_factories.items():
        trajs = []
        for seed in seeds:
            trajs.append(run_bo_no_norm(problem, factory(), seed=seed, n_iter=20))
        results[name] = np.array(trajs)
        finals = results[name][:, -1]
        print(f"  {name:<8} mean_final={finals.mean():.4f}  median_final={np.median(finals):.4f}")

    print(f"\n  GP parameter names (check for outputscale): {globals().get('_PARAM_NAMES')}")

    out = {
        "config": {
            "problem": "Piston",
            "ablation": "input normalization removed (Normalize off, Standardize kept)",
            "seeds": seeds,
            "n_iter": 20,
            "n_init": 2 * problem.dim,
            "botorch": __import__("botorch").__version__,
        },
        "results": {k: v.tolist() for k, v in results.items()},
    }
    out_path = ROOT / "results" / "exp_02b_piston_no_norm.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved {out_path}")

    print("\n  vs report tab:normalization 'Before norm.':  EI 0.41 / UCB 0.39 / Random 0.40")


if __name__ == "__main__":
    main()
