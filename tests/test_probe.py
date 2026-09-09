"""Acquisition-probe smoke test: EI with probe ON over a short Branin run.

Asserts the probe produces one fully-populated, JSON-serialisable entry per
iteration with all blueprint Section 8 keys, finite fractions, and no errors.
"""
import json
import math
import warnings

from al_benchmark.core.bo_loop import run_bo
from al_benchmark.problems.synthetic import Branin
from al_benchmark.strategies.ei import EI

POOL_KEYS = {
    "n_candidates",
    "frac_acqf_zero",
    "frac_below_tiny",
    "max_acqf",
    "median_acqf",
    "frac_nonzero_grad",
    "max_grad_norm",
    "median_grad_norm",
    "n_nan",
    "n_inf",
}
OPT_KEYS = {
    "n_restarts",
    "init_acqf",
    "final_acqf",
    "final_degenerate",
    "all_restarts_degenerate",
}
FRACTION_KEYS = ("frac_acqf_zero", "frac_below_tiny", "frac_nonzero_grad")


def test_probe_on_branin_produces_full_log():
    n_iter = 10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_bo(problem=Branin(), strategy=EI(probe=True), seed=0, n_iter=n_iter)

    assert result.probe is not None
    assert len(result.probe) == n_iter  # one entry per iteration

    for entry in result.probe:
        assert set(entry) == {"numerics_warnings", "candidate_pool", "optimizer"}
        assert set(entry["candidate_pool"]) == POOL_KEYS
        assert set(entry["optimizer"]) == OPT_KEYS
        # No NaN in the candidate-pool fractions on a well-behaved Branin fit.
        for key in FRACTION_KEYS:
            assert not math.isnan(entry["candidate_pool"][key])
        # legacy-EI construction emits exactly one NumericsWarning, and it is counted.
        assert entry["numerics_warnings"] == 1
        assert entry["optimizer"]["n_restarts"] == len(entry["optimizer"]["final_acqf"])

    # The whole log must be JSON-serialisable (it ships inside the run result).
    json.dumps(result.probe)
