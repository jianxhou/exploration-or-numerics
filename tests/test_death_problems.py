"""Verification of the ported De Ath synthetic suite.

(a) precise optimum: scipy refinement from De Ath's xopt agrees with each class's
    optimal_value to 1e-9, and evaluating at the refined point is within 1e-6;
(b) bounds/dim match De Ath's code (synthetic_problems.py literals);
(c) GSobolLog/RosenbrockLog equal log(raw counterpart + shift) pointwise;
(d) determinism: same seed/problem/strategy -> identical trajectory.
"""
import warnings

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from al_benchmark.core.bo_loop import run_bo
from al_benchmark.problems.death import (
    BraninForrester,
    Cosines,
    GoldsteinPriceLog,
    GSobol,
    GSobolLog,
    Hartmann6Log,
    Rosenbrock,
    RosenbrockLog,
    SixHumpCamelLog,
    StyblinskiTangLog,
    WangFreitas,
)
from al_benchmark.strategies.ei import EI

# De Ath's xopt per problem (synthetic_problems.py), used to seed local refinement.
DEATH_XOPT = {
    WangFreitas: [0.9],
    BraninForrester: [-3.689, 13.629],
    Cosines: [0.3125, 0.3125],
    GoldsteinPriceLog: [0.0, -1.0],
    SixHumpCamelLog: [0.0898, -0.7126],
    Hartmann6Log: [0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657300],
    GSobolLog: [0.5] * 10,
    RosenbrockLog: [1.0] * 10,
    StyblinskiTangLog: [-2.903534] * 10,
    GSobol: [0.5] * 10,
    Rosenbrock: [1.0] * 10,
}

# De Ath's lb/ub literals (synthetic_problems.py) for the bounds/dim check.
DEATH_BOUNDS = {
    WangFreitas: ([0.0], [1.0]),
    BraninForrester: ([-5.0, 0.0], [10.0, 15.0]),
    Cosines: ([0.0, 0.0], [5.0, 5.0]),
    GoldsteinPriceLog: ([-2.0, -2.0], [2.0, 2.0]),
    SixHumpCamelLog: ([-3.0, -2.0], [3.0, 2.0]),
    Hartmann6Log: ([0.0] * 6, [1.0] * 6),
    GSobolLog: ([-5.0] * 10, [5.0] * 10),
    RosenbrockLog: ([-5.0] * 10, [10.0] * 10),
    StyblinskiTangLog: ([-5.0] * 10, [5.0] * 10),
    GSobol: ([-5.0] * 10, [5.0] * 10),
    Rosenbrock: ([-5.0] * 10, [10.0] * 10),
}

ALL_PROBLEMS = list(DEATH_XOPT.keys())


def _maximize(problem):
    """Refine our maximization optimum from De Ath's xopt; return (x*, f*)."""
    x0 = np.asarray(DEATH_XOPT[type(problem)], dtype=np.float64)
    lb = problem.bounds[0].numpy()
    ub = problem.bounds[1].numpy()

    def neg(xv):  # minimize -f to maximize f
        t = torch.as_tensor(xv, dtype=torch.double).reshape(1, -1)
        return -float(problem._evaluate(t)[0])

    r1 = minimize(neg, x0, method="L-BFGS-B", bounds=list(zip(lb, ub, strict=True)),
                  options=dict(maxiter=20000, ftol=1e-15, gtol=1e-12))
    r2 = minimize(neg, r1.x, method="Nelder-Mead",
                  options=dict(xatol=1e-12, fatol=1e-14, maxiter=40000))
    best = r1 if r1.fun <= r2.fun else r2
    return best.x, -best.fun


@pytest.mark.parametrize("cls", ALL_PROBLEMS, ids=lambda c: c.__name__)
def test_optimum_matches_optimal_value(cls):
    problem = cls()
    x_star, f_star = _maximize(problem)
    # (a) refinement agrees with the analytic optimal_value to 1e-9 ...
    assert abs(f_star - problem.optimal_value) < 1e-9
    # ... and evaluating at the refined location reproduces it to 1e-6.
    val = float(problem._evaluate(torch.as_tensor(x_star, dtype=torch.double).reshape(1, -1))[0])
    assert abs(val - problem.optimal_value) < 1e-6


@pytest.mark.parametrize("cls", ALL_PROBLEMS, ids=lambda c: c.__name__)
def test_bounds_and_dim_match_death(cls):
    problem = cls()
    lb, ub = DEATH_BOUNDS[cls]
    assert problem.dim == len(lb)
    assert torch.equal(problem.bounds[0], torch.tensor(lb, dtype=torch.double))
    assert torch.equal(problem.bounds[1], torch.tensor(ub, dtype=torch.double))


def test_log_equals_log_of_raw_plus_shift():
    # Under the maximization (negate) convention: -log_problem == log(-raw + shift).
    rng = np.random.default_rng(0)
    for LogCls, RawCls, shift in [(GSobolLog, GSobol, 0.0), (RosenbrockLog, Rosenbrock, 0.5)]:
        logp, rawp = LogCls(), RawCls()
        lb, ub = rawp.bounds[0].numpy(), rawp.bounds[1].numpy()
        X = torch.as_tensor(rng.uniform(lb, ub, size=(100, rawp.dim)), dtype=torch.double)
        lhs = -logp._evaluate(X)
        rhs = torch.log(-rawp._evaluate(X) + shift)
        assert torch.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_trajectory_determinism():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = run_bo(problem=GoldsteinPriceLog(), strategy=EI(), seed=1, n_iter=8)
        b = run_bo(problem=GoldsteinPriceLog(), strategy=EI(), seed=1, n_iter=8)
    assert torch.equal(a.train_x, b.train_x)
    assert a.final_regret == b.final_regret
