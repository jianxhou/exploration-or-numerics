"""Gradient-free (NSGA-II) acquisition-optimizer arms (pre-registration phase3-prereg).

These are the Phase 3 optimizer-axis counterparts of the gradient-based arms: the
acquisition object is built EXACTLY as in the gradient-based strategy (same EI/LogEI
``best_f``, same UCB ``beta``, same PosteriorMean), and ONLY the optimizer changes --
``optimize_acqf`` (multi-start L-BFGS) is replaced by a pymoo evolutionary search at
De Ath (2021) settings. This isolates the gradient-based-vs-gradient-free axis that
H1/H2/H3 test; nothing else about the acquisition or the BO loop differs.

Frozen NSGA-II settings (pre-reg §4.4, De Ath 2021 §4): population 100*d, 50 generations
(budget 100*d*50 = 5000d acquisition evaluations), SBX crossover prob 0.8 eta 20,
polynomial mutation prob 1/d eta 20. Single-objective acquisitions (EI, LogEI, UCB,
Exploit) use pymoo GA (NSGA-II degenerates to GA for one objective); the bi-objective
eps-PF (mean, sigma) front uses pymoo NSGA2 with a uniform random pick from the Pareto set.

RNG discipline (pre-reg §4.4a -- load-bearing for "optimizer-controlled reproduction").
The per-cell pymoo seed is derived deterministically from the
(problem, acquisition, optimizer, run-seed) tuple and held CONSTANT across all BO
iterations within the cell. pymoo 0.6.x's ``seed=`` drives population init, SBX and PM
through its OWN (numpy) generator and does not touch the global torch/numpy streams, so
the injected initial design (De Ath load / Ackley8 Sobol fallback, both torch-seeded by
``run_bo``) and the per-iteration GP refit are untouched. eps-PF's extra strategy-level
randomness (the eps coin and the random pick from the front) uses a dedicated
``torch.Generator`` seeded from the same tuple -- mirroring the frozen eps-greedy
discipline -- never a global reseed, which would perturb the cross-iteration GP-refit
stream and violate §4.4a's clause that the optimizer RNG governs ONLY its own randomness.
The run seed is taken from the explicit constructor ``seed`` when provided (the production
runner passes it), else read lazily from ``torch.initial_seed()`` (set by ``run_bo``) on
first use, so it reflects the cell seed without being threaded through ``select_next``
(whose contract carries no seed/index argument).
"""
import hashlib

import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
)
from botorch.models.model import Model
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from torch import Tensor

from al_benchmark.strategies.base import BaseStrategy

# Frozen NSGA-II settings -- pre-registration phase3-prereg §4.4 (De Ath 2021 §4).
# Do NOT alter: these define the optimizer arm under test.
_POP_MULT = 100  # population size = 100 * d
_N_GEN = 50  # generations (=> budget 100*d*50 = 5000d acquisition evaluations)
_CX_PROB = 0.8  # SBX crossover probability
_CX_ETA = 20  # SBX distribution index
_MUT_ETA = 20  # polynomial mutation distribution index
# mutation probability = 1 / d, computed per-call from the problem dimension.
_OPTIMIZER = "nsga"  # optimizer tag for the §4.4a (problem, acq, optimizer, seed) tuple


def _stable_seed(problem: str | None, acq: str, optimizer: str, run_seed: int) -> int:
    """Deterministic 31-bit seed from the §4.4a tuple, stable across processes.

    Uses blake2b (NOT Python's per-process-salted ``hash``) so ProcessPoolExecutor
    workers derive identical seeds for the same cell. Range [0, 2**31 - 2], a valid
    pymoo / numpy seed.
    """
    key = f"{problem}|{acq}|{optimizer}|{run_seed}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


class _AcqProblem(Problem):
    """Single-objective pymoo wrapper: minimize ``-acq(x)`` over the box (maximize acq)."""

    def __init__(self, acq, bounds: Tensor) -> None:
        dim = bounds.shape[-1]
        super().__init__(
            n_var=dim,
            n_obj=1,
            n_ieq_constr=0,
            xl=bounds[0].detach().cpu().numpy(),
            xu=bounds[1].detach().cpu().numpy(),
        )
        self._acq = acq

    def _evaluate(self, X, out, *args, **kwargs) -> None:
        xt = torch.as_tensor(X, dtype=torch.double).unsqueeze(1)  # (pop, 1, d)
        with torch.no_grad():
            vals = self._acq(xt)  # (pop,)
        out["F"] = (-vals.detach().cpu().numpy()).reshape(-1, 1)


class _ParetoProblem(Problem):
    """Bi-objective pymoo wrapper for eps-PF: maximize (mean, sigma) of the GP posterior.

    Objectives are returned as negatives because pymoo minimizes, so the front is the
    (maximize mean, maximize sigma) trade-off whose greedy end coincides with the Exploit
    point -- matching the frozen eps-greedy ``_pareto_front_max_max`` convention, including
    the ``variance.clamp_min(0).sqrt()`` guard against tiny negative variances.
    """

    def __init__(self, model: Model, bounds: Tensor) -> None:
        dim = bounds.shape[-1]
        super().__init__(
            n_var=dim,
            n_obj=2,
            n_ieq_constr=0,
            xl=bounds[0].detach().cpu().numpy(),
            xu=bounds[1].detach().cpu().numpy(),
        )
        self._model = model

    def _evaluate(self, X, out, *args, **kwargs) -> None:
        xt = torch.as_tensor(X, dtype=torch.double)  # (pop, d)
        with torch.no_grad():
            post = self._model.posterior(xt)
            mean = post.mean.squeeze(-1)
            sigma = post.variance.clamp_min(0).sqrt().squeeze(-1)
        out["F"] = torch.stack([-mean, -sigma], dim=-1).detach().cpu().numpy()


def _ga(dim: int) -> GA:
    return GA(
        pop_size=_POP_MULT * dim,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=_CX_PROB, eta=_CX_ETA),
        mutation=PM(prob=1.0 / dim, eta=_MUT_ETA),
        eliminate_duplicates=False,
    )


def _nsga2(dim: int) -> NSGA2:
    return NSGA2(
        pop_size=_POP_MULT * dim,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=_CX_PROB, eta=_CX_ETA),
        mutation=PM(prob=1.0 / dim, eta=_MUT_ETA),
        eliminate_duplicates=False,
    )


def nsga2_argmax_acqf(acq, bounds: Tensor, seed: int) -> Tensor:
    """Maximize a single-objective BoTorch acquisition with pymoo GA. Returns (1, d)."""
    dim = bounds.shape[-1]
    res = minimize(
        _AcqProblem(acq, bounds),
        _ga(dim),
        termination=("n_gen", _N_GEN),
        seed=seed,
        verbose=False,
    )
    x = torch.as_tensor(np.atleast_2d(res.X), dtype=torch.double)
    return x.reshape(1, dim)


def nsga2_pareto_set(model: Model, bounds: Tensor, seed: int) -> Tensor:
    """Bi-objective NSGA2 over (maximize mean, maximize sigma). Returns the Pareto-set X (k, d)."""
    dim = bounds.shape[-1]
    res = minimize(
        _ParetoProblem(model, bounds),
        _nsga2(dim),
        termination=("n_gen", _N_GEN),
        seed=seed,
        verbose=False,
    )
    return torch.as_tensor(np.atleast_2d(res.X), dtype=torch.double)  # (k, d)


class _NSGA2Base(BaseStrategy):
    """Shared NSGA-II optimizer-arm machinery: per-cell constant seed derivation (§4.4a).

    Subclasses set ``_acq_label`` (the stable acquisition identity used in the seed tuple,
    so distinct arms draw decorrelated GA searches from the same run seed). The run seed is
    the explicit constructor ``seed`` if given, else ``torch.initial_seed()`` resolved
    lazily on first use (after ``run_bo`` has set it). The derived pymoo seed is a pure
    function of the tuple, hence constant across all BO iterations within the cell.
    """

    _acq_label: str = ""

    def __init__(self, seed: int | None = None, problem: str | None = None) -> None:
        self._run_seed = seed
        self._problem = problem

    def _pymoo_seed(self, suffix: str = "") -> int:
        if self._run_seed is None:
            self._run_seed = int(torch.initial_seed())
        return _stable_seed(self._problem, self._acq_label + suffix, _OPTIMIZER, self._run_seed)


class _NSGA2Single(_NSGA2Base):
    """Single-objective NSGA-II arm: build the acquisition, then GA-maximize it."""

    def _make_acq(self, model: Model, train_y: Tensor):
        raise NotImplementedError

    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,
        train_y: Tensor,
    ) -> Tensor:
        acq = self._make_acq(model, train_y)
        return nsga2_argmax_acqf(acq, bounds, self._pymoo_seed())


class EINSGA2(_NSGA2Single):
    """Expected Improvement, optimized by NSGA-II (pymoo GA). H3 EI/nsga arm."""

    _acq_label = "EI"

    def __init__(self, seed: int | None = None, problem: str | None = None) -> None:
        super().__init__(seed, problem)
        self.name = "EINSGA2"

    def _make_acq(self, model: Model, train_y: Tensor):
        return ExpectedImprovement(model=model, best_f=train_y.max())


class LogEINSGA2(_NSGA2Single):
    """Log Expected Improvement, optimized by NSGA-II (pymoo GA). H3 LogEI/nsga arm."""

    _acq_label = "LogEI"

    def __init__(self, seed: int | None = None, problem: str | None = None) -> None:
        super().__init__(seed, problem)
        self.name = "LogEINSGA2"

    def _make_acq(self, model: Model, train_y: Tensor):
        return LogExpectedImprovement(model=model, best_f=train_y.max())


class UCBNSGA2(_NSGA2Single):
    """Upper Confidence Bound (beta=2.0), optimized by NSGA-II (pymoo GA)."""

    _acq_label = "UCB"

    def __init__(
        self, beta: float = 2.0, seed: int | None = None, problem: str | None = None
    ) -> None:
        super().__init__(seed, problem)
        self.beta = beta
        self.name = "UCBNSGA2"

    def _make_acq(self, model: Model, train_y: Tensor):
        return UpperConfidenceBound(model=model, beta=self.beta)


class ExploitNSGA2(_NSGA2Single):
    """Greedy posterior-mean exploitation, optimized by NSGA-II (pymoo GA)."""

    _acq_label = "Exploit"

    def __init__(self, seed: int | None = None, problem: str | None = None) -> None:
        super().__init__(seed, problem)
        self.name = "ExploitNSGA2"

    def _make_acq(self, model: Model, train_y: Tensor):
        return PosteriorMean(model=model)


class EpsPFNSGA2(_NSGA2Base):
    """eps-PF / eFront under NSGA-II (De Ath 2021): greedy with probability 1 - eps, else a
    uniform random pick from the (mean, sigma) Pareto front approximated by pymoo NSGA2.

    Faithful to De Ath's eps-PF (front approximated by NSGA-II over the GP), and the
    pre-registered NSGA-II counterpart of the frozen gradient-based eps-PF. The exploit
    branch is the greedy posterior-mean point found by GA (ExploitNSGA2); the explore branch
    evolves the bi-objective front by NSGA2 and picks a member uniformly. The eps coin and
    the random pick use a dedicated ``torch.Generator`` seeded from the §4.4a tuple
    (suffix ``/coin``); the two pymoo searches use constant per-cell seeds (suffixes
    ``/exploit`` and ``/front``). Reported descriptively only (pre-reg H2b); the
    front-approximation divergence from De Ath's exact eFront is disclosed (§6.3).
    """

    _acq_label = "eps-PF"

    def __init__(
        self, eps: float = 0.1, seed: int | None = None, problem: str | None = None
    ) -> None:
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        super().__init__(seed, problem)
        self.eps = eps
        self.name = "EpsPFNSGA2"
        self._rng: torch.Generator | None = None

    def _generator(self) -> torch.Generator:
        """Dedicated coin/pick generator, seeded from the §4.4a tuple; built lazily so the
        run seed is read after ``run_bo`` sets it. Disjoint from the global torch stream."""
        if self._rng is None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self._pymoo_seed("/coin"))
        return self._rng

    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,
        train_y: Tensor,
    ) -> Tensor:
        gen = self._generator()
        coin = torch.rand(1, generator=gen).item()
        if coin < self.eps:
            front = nsga2_pareto_set(model, bounds, self._pymoo_seed("/front"))  # (k, d)
            pick = int(torch.randint(front.shape[0], (1,), generator=gen))
            return front[pick : pick + 1]  # (1, d)
        acq = PosteriorMean(model=model)
        return nsga2_argmax_acqf(acq, bounds, self._pymoo_seed("/exploit"))
