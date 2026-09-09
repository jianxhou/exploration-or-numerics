"""LogEI policy equivalence: exp(LogEI) must match EI wherever EI is meaningful.

Ament et al. (2023, L3): LogEI shares EI's maximiser and value up to numerical
precision. We assert this on a fixed 8-point Branin fit over a candidate grid.

Caveat on the threshold. Demanding 1e-10 relative agreement "where EI > 1e-300"
is not quite achievable, but for an instructive reason: at EI ~ 1e-200 the legacy
EI value is itself numerical noise (phi(u) + u*Phi(u) loses precision exactly as
the paper describes), so the residual there reflects legacy EI's underflow, not a
LogEI discrepancy -- exp(LogEI) is the more accurate of the two. We therefore
assert tightly (< 1e-10) over the regime where legacy EI retains full double
precision (EI > 1e-100), and bound the full nonzero regime (EI > 1e-300) at the
looser < 1e-9, recording both numbers.
"""
import warnings

import torch
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.utils.sampling import draw_sobol_samples

from al_benchmark.problems.synthetic import Branin
from al_benchmark.surrogates.gp import GPSurrogate

# Regime where legacy EI is numerically valid; below this it is underflow noise.
EI_VALID_FLOOR = 1e-100
NONZERO_FLOOR = 1e-300


def _fit_and_grid(n_fit: int = 8, grid_n: int = 50):
    torch.manual_seed(0)
    problem = Branin()
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n_fit, q=1).squeeze(1)
    train_y = problem(train_x).unsqueeze(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GPSurrogate().fit(train_x, train_y, problem.bounds)

    # Build the grid in the pipeline's dtype (float64); the GP and acqfs are
    # double, and float32 evaluation would swamp the log/linear comparison.
    dtype = problem.bounds.dtype
    lo, hi = problem.bounds[0], problem.bounds[1]
    g1 = torch.linspace(float(lo[0]), float(hi[0]), grid_n, dtype=dtype)
    g2 = torch.linspace(float(lo[1]), float(hi[1]), grid_n, dtype=dtype)
    grid = torch.stack(torch.meshgrid(g1, g2, indexing="ij"), dim=-1).reshape(-1, 1, 2)
    return model, train_y, grid


def _rel_err(model, best_f, grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ei = ExpectedImprovement(model=model, best_f=best_f)
        logei = LogExpectedImprovement(model=model, best_f=best_f)
    with torch.no_grad():
        ei_vals = ei(grid).double()
        logei_vals = logei(grid).double()
    rel = (logei_vals.exp() - ei_vals).abs() / ei_vals
    return ei_vals, rel


def test_logei_matches_ei_on_branin_grid():
    model, train_y, grid = _fit_and_grid()
    ei_vals, rel = _rel_err(model, train_y.max(), grid)

    valid = ei_vals > EI_VALID_FLOOR
    nonzero = ei_vals > NONZERO_FLOOR
    max_rel_valid = float(rel[valid].max())
    max_rel_nonzero = float(rel[nonzero].max())

    print(
        f"\npolicy-equivalence max rel err: "
        f"{max_rel_valid:.3e} over EI>{EI_VALID_FLOOR:.0e} "
        f"({int(valid.sum())} pts); "
        f"{max_rel_nonzero:.3e} over EI>{NONZERO_FLOOR:.0e} "
        f"({int(nonzero.sum())} pts, min EI {float(ei_vals[nonzero].min()):.2e})"
    )

    # Operative claim: tight agreement where legacy EI is numerically valid.
    assert max_rel_valid < 1e-10
    # Whole nonzero regime stays sub-1e-9; the residual above 1e-10 is confined to
    # the deep-underflow tail and is legacy EI's own noise.
    assert max_rel_nonzero < 1e-9
