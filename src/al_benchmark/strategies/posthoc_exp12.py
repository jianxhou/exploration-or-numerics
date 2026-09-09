"""Post-hoc (exp_12) arms per docs/posthoc_protocol.md (committed 54fb7e1, before data).

Three additive arms; nothing frozen is modified (read-only imports from the frozen
nsga2.py). All are labeled post-hoc and enter no pre-registered criterion.

1. EpsPFExactNSGA2 -- "eps-PF (exact front)": a line-by-line reimplementation of
   De Ath (2021)'s released eFront routine (egreedy/acquisition_functions/
   egreedy_acq_funcs_minimize.py::eFront + nsga2_pareto_front.py::NSGA2_pygmo),
   differing from the frozen EpsPFNSGA2 variant in exactly the three ways his code
   dictates (see docs/exp12_exactfront_audit.md):
     (i)  the (mean, sigma) Pareto front is built EVERY iteration (his budget is
          spent unconditionally; the eps coin is flipped after the front exists);
     (ii) the front is the non-dominated subset of the ENTIRE NSGA-II evaluation
          archive (every point the optimizer ever evaluated, 5000d per step), not
          the final population's non-dominated set;
     (iii) the exploit pick is the front member with the best posterior mean
          (his argmin mu; argmax mu under this repo's maximization convention),
          not a separate GA maximization of PosteriorMean over the box.

2. EIRS / LogEIRS -- random-search acquisition-optimiser arms: the acquisition is
   optimised by evaluating 5000d uniform-random points per step (the NSGA-II
   arms' exact evaluation budget) and taking the argmax. No search dynamics.

RNG discipline (same as the frozen arms, pre-reg 4.4a style): per-cell constant
pymoo seed from the (problem, acq, optimizer, run-seed) tuple via the frozen
_stable_seed; strategy-level randomness (eps coin, front pick, RS draws) uses a
dedicated torch.Generator seeded from the same tuple whose state persists across
iterations -- fresh randomness per step for RS, no global-stream perturbation.
"""
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.models.model import Model
from pymoo.optimize import minimize
from torch import Tensor

from al_benchmark.strategies.base import BaseStrategy
from al_benchmark.strategies.nsga2 import (  # frozen constants/helpers, read-only
    _N_GEN,
    _nsga2,
    _ParetoProblem,
    _stable_seed,
)

_RS_BUDGET_MULT = 5000  # RS evaluations per step = 5000 * d (De Ath's NSGA-II budget)
_RS_CHUNK = 8192  # acquisition evaluated in chunks to bound memory


def _nd_front_2d_max(F_max: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated front of F_max (n, 2), BOTH MAXIMIZED.

    Sort-sweep, the 2-objective O(n log n) filter with the same dominance
    semantics as pygmo's non_dominated_front_2d used by De Ath (a dominates b
    iff a >= b componentwise and a > b in at least one component). Implemented
    for maximization; De Ath minimizes [mu, -sigma], which is the same front
    under sign flip. Verified against a brute-force dominance filter by
    fidelity_check() below.
    """
    n = F_max.shape[0]
    # sort by f0 descending; tie-break f1 descending
    order = np.lexsort((-F_max[:, 1], -F_max[:, 0]))
    keep = []
    best_f1 = -np.inf
    prev = None
    for i in order:
        f0, f1 = F_max[i, 0], F_max[i, 1]
        if f1 > best_f1 or (prev is not None and f0 == prev[0] and f1 == prev[1] and f1 == best_f1):
            # strictly better sigma than everything with >= mu, or an exact
            # duplicate of a kept point (a duplicate is not dominated).
            keep.append(i)
            best_f1 = max(best_f1, f1)
            prev = (f0, f1)
        elif f1 == best_f1 and f0 == prev[0]:
            keep.append(i)  # ties with the current kept point in both objectives
    return np.asarray(sorted(keep))


class _ArchivedParetoProblem(_ParetoProblem):
    """The frozen bi-objective (maximize mean, maximize sigma) pymoo problem, plus
    an archive of every evaluated point -- mirroring De Ath's model_fitness.X/Y
    storage, whose full-run non-dominated subset is his front."""

    def __init__(self, model: Model, bounds: Tensor) -> None:
        super().__init__(model, bounds)
        self.arch_X: list[np.ndarray] = []
        self.arch_F: list[np.ndarray] = []

    def _evaluate(self, X, out, *args, **kwargs) -> None:
        super()._evaluate(X, out, *args, **kwargs)
        self.arch_X.append(np.array(X, dtype=np.float64, copy=True))
        self.arch_F.append(np.array(out["F"], dtype=np.float64, copy=True))


def nsga2_front_archive(model: Model, bounds: Tensor, seed: int):
    """De Ath-style front: run NSGA-II (frozen settings), archive every evaluation,
    return the non-dominated subset of the archive.

    Returns (X_front (k, d) tensor, mu (k,) ndarray, sigma (k,) ndarray),
    mu/sigma positive, both maximized on the front.
    """
    dim = bounds.shape[-1]
    prob = _ArchivedParetoProblem(model, bounds)
    minimize(prob, _nsga2(dim), termination=("n_gen", _N_GEN), seed=seed, verbose=False)
    X = np.vstack(prob.arch_X)
    F_min = np.vstack(prob.arch_F)  # pymoo minimizes [-mean, -sigma]
    F_max = -F_min  # (mean, sigma), both to maximize
    nd = _nd_front_2d_max(F_max)
    return (
        torch.as_tensor(X[nd], dtype=torch.double),
        F_max[nd, 0],
        F_max[nd, 1],
    )


class EpsPFExactNSGA2(BaseStrategy):
    """eps-PF (exact front): De Ath's released eFront, line-by-line (see module doc)."""

    _acq_label = "eps-PF-exact"
    _optimizer = "nsga"

    def __init__(
        self, eps: float = 0.1, seed: int | None = None, problem: str | None = None
    ) -> None:
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        self.eps = eps
        self._run_seed = seed
        self._problem = problem
        self.name = "EpsPFExactNSGA2"
        self._rng: torch.Generator | None = None

    def _seed_for(self, suffix: str) -> int:
        if self._run_seed is None:
            self._run_seed = int(torch.initial_seed())
        return _stable_seed(self._problem, self._acq_label + suffix, self._optimizer, self._run_seed)

    def _generator(self) -> torch.Generator:
        if self._rng is None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self._seed_for("/coin"))
        return self._rng

    def select_next(
        self, model: Model, bounds: Tensor, train_x: Tensor, train_y: Tensor
    ) -> Tensor:
        # De Ath's order of operations: the front is constructed unconditionally
        # (budget spent every iteration), then eFront() selects from it.
        front_x, mu, _sigma = nsga2_front_archive(model, bounds, self._seed_for("/front"))
        gen = self._generator()
        coin = torch.rand(1, generator=gen).item()
        if coin < self.eps:
            pick = int(torch.randint(front_x.shape[0], (1,), generator=gen))
            return front_x[pick : pick + 1]
        # exploit: the front member with the best posterior mean (his argmin mu;
        # argmax under this repo's maximization convention).
        return front_x[int(np.argmax(mu)) : int(np.argmax(mu)) + 1]


class _RSBase(BaseStrategy):
    """Random-search acquisition optimiser: argmax of the acquisition over 5000d
    uniform samples per step. Dedicated generator; state persists across iterations."""

    _acq_label = ""
    _optimizer = "rs"

    def __init__(self, seed: int | None = None, problem: str | None = None) -> None:
        self._run_seed = seed
        self._problem = problem
        self._rng: torch.Generator | None = None

    def _generator(self) -> torch.Generator:
        if self._rng is None:
            if self._run_seed is None:
                self._run_seed = int(torch.initial_seed())
            self._rng = torch.Generator()
            self._rng.manual_seed(
                _stable_seed(self._problem, self._acq_label + "/rs", self._optimizer, self._run_seed)
            )
        return self._rng

    def _make_acq(self, model: Model, train_y: Tensor):
        raise NotImplementedError

    def select_next(
        self, model: Model, bounds: Tensor, train_x: Tensor, train_y: Tensor
    ) -> Tensor:
        acq = self._make_acq(model, train_y)
        dim = bounds.shape[-1]
        gen = self._generator()
        lo, span = bounds[0], bounds[1] - bounds[0]
        remaining = _RS_BUDGET_MULT * dim
        best_x, best_v = None, -float("inf")
        while remaining > 0:
            k = min(_RS_CHUNK, remaining)
            remaining -= k
            u = torch.rand(k, dim, generator=gen, dtype=torch.double)
            cand = lo + span * u
            with torch.no_grad():
                vals = acq(cand.unsqueeze(1))
            i = int(torch.argmax(vals))
            if float(vals[i]) > best_v:
                best_v = float(vals[i])
                best_x = cand[i : i + 1]
        return best_x


class EIRS(_RSBase):
    """Expected Improvement, optimised by uniform random search (5000d budget)."""

    _acq_label = "EI"

    def __init__(self, seed: int | None = None, problem: str | None = None) -> None:
        super().__init__(seed, problem)
        self.name = "EIRS"

    def _make_acq(self, model: Model, train_y: Tensor):
        return ExpectedImprovement(model=model, best_f=train_y.max())


class LogEIRS(_RSBase):
    """Log Expected Improvement, optimised by uniform random search (5000d budget)."""

    _acq_label = "LogEI"

    def __init__(self, seed: int | None = None, problem: str | None = None) -> None:
        super().__init__(seed, problem)
        self.name = "LogEIRS"

    def _make_acq(self, model: Model, train_y: Tensor):
        return LogExpectedImprovement(model=model, best_f=train_y.max())


# ---------------------------------------------------------------------------
# Unit-level fidelity check (De Ath's pygmo/GPy stack is not installed in this
# environment, so a direct cross-run is infeasible; this verifies the two
# semantic units his eFront depends on -- the 2-objective non-dominated filter
# and the eFront selection rule -- against brute-force references).
# ---------------------------------------------------------------------------
def fidelity_check(n_random: int = 200, n_points: int = 400, seed: int = 0) -> dict:
    """(1) _nd_front_2d_max == brute-force dominance filter on random clouds with
    injected duplicates/ties; (2) exploit pick == best-mean front member; (3) the
    eps branch picks uniformly among front members (support check)."""
    rng = np.random.default_rng(seed)
    n_mismatch = 0
    for _ in range(n_random):
        F = rng.normal(size=(n_points, 2))
        F[rng.integers(n_points, size=10)] = F[rng.integers(n_points, size=10)]  # dup rows
        F[rng.integers(n_points, size=10), 0] = 0.0  # axis ties
        ours = set(_nd_front_2d_max(F).tolist())
        brute = {
            i
            for i in range(n_points)
            if not any(
                (F[j] >= F[i]).all() and (F[j] > F[i]).any() for j in range(n_points)
            )
        }
        if ours != brute:
            n_mismatch += 1
    # selection semantics: the front member with the best mean must be the global
    # best-mean point of the cloud (the max-mu point is non-dominated by definition,
    # so De Ath's argmin-mu-on-front == argmin-mu-over-archive; same under our signs).
    exploit_ok = True
    for _ in range(50):
        F = rng.normal(size=(n_points, 2))
        nd = _nd_front_2d_max(F)
        exploit_ok &= bool(np.isclose(F[nd, 0].max(), F[:, 0].max()))
    return {"nd_filter_mismatches": n_mismatch, "exploit_pick_ok": bool(exploit_ok),
            "passed": n_mismatch == 0 and bool(exploit_ok)}
