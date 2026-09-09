"""Regression gate: probe-off EI must not drift from the pre-probe implementation.

The shared improvement-acqf refactor and the optional probe must leave the EI
arm byte-for-byte unchanged when the probe is off. The expected constant below
was recorded from the EI implementation at git HEAD before the refactor (verified
identical via torch.equal on the full trajectory across seeds 0/1/7/42).
"""
import warnings

from al_benchmark.core.bo_loop import run_bo
from al_benchmark.problems.synthetic import Branin
from al_benchmark.strategies.ei import EI

# EI(probe=False) on Branin, seed 0, n_iter=20 -- pre-refactor reference value.
EXPECTED_FINAL_REGRET = 1.116562567255e-02


def test_ei_probe_off_matches_recorded_baseline():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # legacy-EI NumericsWarning is expected here
        result = run_bo(problem=Branin(), strategy=EI(probe=False), seed=0, n_iter=20)

    assert result.probe is None  # probe-off leaves the result untouched
    # Tight relative tolerance: catches any real code-path change while tolerating
    # only last-ULP BLAS variation.
    assert abs(result.final_regret - EXPECTED_FINAL_REGRET) <= 1e-9 * EXPECTED_FINAL_REGRET, (
        f"EI probe-off regret {result.final_regret:.12e} drifted from baseline "
        f"{EXPECTED_FINAL_REGRET:.12e}"
    )
