"""Sequential Bayesian optimization loop (`run_bo`)."""
from dataclasses import dataclass, field

import torch
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from al_benchmark.problems.base import BaseProblem
from al_benchmark.strategies.base import BaseStrategy
from al_benchmark.surrogates.gp import GPSurrogate


@dataclass
class BOResult:
    """The trajectory and final state of a BO run."""

    problem_name: str
    strategy_name: str
    seed: int
    n_init: int
    n_iter: int
    train_x: Tensor                    # shape (n_init + n_iter, dim)
    train_y: Tensor                    # shape (n_init + n_iter, 1)
    best_observed: list[float] = field(default_factory=list)  # length n_init + n_iter
    regret_history: list[float] = field(default_factory=list) # length n_init + n_iter
    final_regret: float = 0.0
    probe: list | None = None          # acquisition diagnostics, one entry per iter


def run_bo(
    problem: BaseProblem,
    strategy: BaseStrategy,
    seed: int = 42,
    n_init: int | None = None,
    n_iter: int = 20,
    verbose: bool = False,
    initial_design: Tensor | None = None,
) -> BOResult:
    """Run a Sobol initial design, then `n_iter` acquisition steps (maximization).

    `seed` controls both the initial design and strategy randomness.
    `n_init` defaults to 2 * problem.dim.

    `initial_design` (D8): an (M, dim) array used verbatim INSTEAD of generating a
    fresh Sobol design (e.g. De Ath's published designs). `seed` still seeds the
    acquisition/strategy RNG; only the initial X is replaced. Its y values are
    re-evaluated through this pipeline (the problems are deterministic), so the
    injected points appear verbatim in `result.train_x[:M]`.
    """
    torch.manual_seed(seed)

    if initial_design is not None:
        train_x = torch.as_tensor(initial_design, dtype=problem.bounds.dtype)
        if train_x.ndim != 2 or train_x.shape[-1] != problem.dim:
            raise ValueError(
                f"initial_design must be (M, {problem.dim}); got {tuple(train_x.shape)}"
            )
        n_init = train_x.shape[0]
    else:
        if n_init is None:
            n_init = 2 * problem.dim
        train_x = draw_sobol_samples(bounds=problem.bounds, n=n_init, q=1).squeeze(1)
    train_y = problem(train_x).unsqueeze(-1)

    best_observed: list[float] = [train_y.max().item()]
    regret_history: list[float] = [problem.regret(best_observed[-1])]

    if verbose:
        print(f"Initial best: {best_observed[-1]:.4f}  "
              f"(regret {regret_history[-1]:.4f})")

    surrogate = GPSurrogate()
    for i in range(n_iter):
        model = surrogate.fit(train_x, train_y, problem.bounds)

        candidate = strategy.select_next(
            model=model,
            bounds=problem.bounds,
            train_x=train_x,
            train_y=train_y,
        )

        new_y = problem(candidate).unsqueeze(-1)

        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])
        best_observed.append(train_y.max().item())
        regret_history.append(problem.regret(best_observed[-1]))

        if verbose:
            print(f"Iter {i+1:3d}: new y = {new_y.item():.4f}  "
                  f"best = {best_observed[-1]:.4f}  "
                  f"regret = {regret_history[-1]:.4f}")

    return BOResult(
        problem_name=problem.name,
        strategy_name=strategy.name,
        seed=seed,
        n_init=n_init,
        n_iter=n_iter,
        train_x=train_x,
        train_y=train_y,
        best_observed=best_observed,
        regret_history=regret_history,
        final_regret=regret_history[-1],
        # Strategies running an AcqProbe expose its per-iteration log here;
        # absent for all other strategies (and probe-off EI/LogEI), so the
        # result is unchanged for existing experiments.
        probe=getattr(strategy, "probe_log", None),
    )
