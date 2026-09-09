"""Initial-design injection (D8) and the De Ath training_data loader.

Injecting a fixed (M, d) design replaces the Sobol initial design verbatim; with
the same seed the resulting trajectory is identical, and the design appears
unchanged in the run result. The loader is exercised against De Ath's published
npz when present, and skipped otherwise so the suite is self-contained.
"""
import os
import warnings
from pathlib import Path

import torch

from al_benchmark.core.bo_loop import run_bo
from al_benchmark.problems.death import load_death_initial_design
from al_benchmark.problems.synthetic import Branin
from al_benchmark.strategies.ei import EI

# A fixed, deterministic 4-point design in Branin's domain (lb=[-5,0], ub=[10,15]).
FIXED_DESIGN = torch.tensor(
    [[0.0, 5.0], [3.0, 2.5], [-2.0, 12.0], [7.5, 9.0]], dtype=torch.double
)


def test_injected_design_gives_identical_trajectories():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = run_bo(problem=Branin(), strategy=EI(), seed=7, n_iter=6, initial_design=FIXED_DESIGN)
        b = run_bo(problem=Branin(), strategy=EI(), seed=7, n_iter=6, initial_design=FIXED_DESIGN)
    assert torch.equal(a.train_x, b.train_x)
    assert a.final_regret == b.final_regret


def test_injected_design_appears_verbatim():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_bo(
            problem=Branin(), strategy=EI(), seed=7, n_iter=6, initial_design=FIXED_DESIGN
        )
    m = FIXED_DESIGN.shape[0]
    assert result.n_init == m
    assert torch.equal(result.train_x[:m], FIXED_DESIGN)


def test_injection_overrides_sobol_design():
    # A fresh Sobol run and an injected run with the same seed must differ in the
    # initial design (proving the injection replaced it, not appended).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sobol = run_bo(problem=Branin(), strategy=EI(), seed=7, n_iter=0)
        injected = run_bo(problem=Branin(), strategy=EI(), seed=7, n_iter=0,
                          initial_design=FIXED_DESIGN)
    assert not torch.equal(sobol.train_x, injected.train_x)
    assert torch.equal(injected.train_x, FIXED_DESIGN)


def _death_data_present(name="Branin", run_no=1):
    base = Path(os.path.expanduser("~/projects/egreedy/training_data"))
    return (base / f"{name}_{run_no}.npz").exists()


def test_loader_returns_design_and_injects_verbatim():
    if not _death_data_present():
        import pytest

        pytest.skip("De Ath training_data not present")
    x = load_death_initial_design("Branin", 1)
    assert x.ndim == 2 and x.shape[1] == 2 and x.dtype == torch.double
    assert x.shape[0] == 2 * Branin().dim  # M = 2d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_bo(problem=Branin(), strategy=EI(), seed=3, n_iter=0, initial_design=x)
    assert torch.equal(result.train_x[: x.shape[0]], x)
