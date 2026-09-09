"""eps-greedy determinism and boundary behaviour (eps in {0, 1}).

Covers the seeded-RNG contract: identical seeds reproduce the exact selections;
eps=0 collapses to Exploit byte-for-byte (the coin draw must not perturb the
global stream Exploit's optimiser consumes); eps=1 always explores, so it never
lands on the Exploit point.
"""
import warnings

import torch
from botorch.utils.sampling import draw_sobol_samples

from al_benchmark.core.bo_loop import run_bo
from al_benchmark.problems.synthetic import Branin
from al_benchmark.strategies.eps_greedy import EpsRS
from al_benchmark.strategies.exploit import Exploit
from al_benchmark.surrogates.gp import GPSurrogate


def _silence():
    warnings.simplefilter("ignore")


def test_same_seed_gives_identical_points():
    with warnings.catch_warnings():
        _silence()
        a = run_bo(problem=Branin(), strategy=EpsRS(eps=0.3), seed=5, n_iter=15)
        b = run_bo(problem=Branin(), strategy=EpsRS(eps=0.3), seed=5, n_iter=15)
    assert torch.equal(a.train_x, b.train_x)
    assert a.final_regret == b.final_regret


def test_eps_zero_reduces_to_exploit_exactly():
    with warnings.catch_warnings():
        _silence()
        exploit = run_bo(problem=Branin(), strategy=Exploit(), seed=3, n_iter=15)
        eps0 = run_bo(problem=Branin(), strategy=EpsRS(eps=0.0), seed=3, n_iter=15)
    assert torch.equal(exploit.train_x, eps0.train_x)


def test_eps_one_never_selects_exploit_point():
    with warnings.catch_warnings():
        _silence()
        torch.manual_seed(0)
        problem = Branin()
        train_x = draw_sobol_samples(bounds=problem.bounds, n=4, q=1).squeeze(1)
        train_y = problem(train_x).unsqueeze(-1)

        strat = EpsRS(eps=1.0, seed=11)
        exploit = Exploit()
        for _ in range(50):
            model = GPSurrogate().fit(train_x, train_y, problem.bounds)
            chosen = strat.select_next(model, problem.bounds, train_x, train_y)
            exploit_pt = exploit.select_next(model, problem.bounds, train_x, train_y)
            assert not torch.allclose(chosen, exploit_pt, atol=1e-6)
            new_y = problem(chosen).unsqueeze(-1)
            train_x = torch.cat([train_x, chosen])
            train_y = torch.cat([train_y, new_y])
