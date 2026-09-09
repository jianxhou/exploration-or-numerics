"""De Ath et al. (2021) "Greed is Good" synthetic suite, ported to this project.

Authoritative source for every formula and domain is their code,
``egreedy/test_problems/synthetic_problems.py`` (commit 4ab1a99); line numbers in
the comments below refer to that file. The supplementary PDF is secondary and
contains at least one typo (the gSobol formula; see GSobolLog).

Convention. De Ath *minimize*; this project *maximizes* and negates naturally
minimized functions (problems/base.py). So every ``_evaluate`` here returns
``-g(x)`` where ``g`` is their ``__call__``, and ``optimal_value`` is ``-g*``
(the negation of their true minimum). The log problems implement exactly their
``log(g(x) + shift)`` inner formula with the same shift constants, then negate.

``optimal_value`` is the *precise* optimum, derived from the true g* and verified
to 1e-9 by local refinement (tests/test_death_problems.py). It is deliberately
distinct from De Ath's coarse code ``yopt`` (= g at a rounded xopt), which lives
in the experiments layer as ``DEATH_YOPT`` for De Ath-comparable regret only.
``death_name`` records their file/class stem so initial designs can be loaded.
"""
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from al_benchmark.problems.base import BaseProblem


def _bounds(lb: list[float], ub: list[float]) -> Tensor:
    """(2, dim) float64 bounds tensor from lower/upper lists."""
    return torch.tensor([lb, ub], dtype=torch.double)


class WangFreitas(BaseProblem):
    """WangFreitas 1-d deceptive function (synthetic_problems.py:39, dom L65-66).

    f = -(2 e^{-0.5 (a-x)^2/t1^2} + 4 e^{-0.5 (b-x)^2/t2^2}); a=0.1, b=0.9,
    t1=0.1, t2=0.01. The narrow deep peak at b=0.9 is the global optimum and is
    hard to find -- the diagnostic case where only global random exploration
    succeeds. optimal_value = 4 + 2 e^{-32} (the x=0.9 peak; argmax is 0.9 to ~1e-16).
    """

    death_name = "WangFreitas"

    def __init__(self) -> None:
        self.name = "WangFreitas"
        self.dim = 1
        self.bounds = _bounds([0.0], [1.0])
        self.optimal_value = 4.0 + 2.0 * math.exp(-32.0)  # = 4.000000000000026

    def _evaluate(self, x: Tensor) -> Tensor:
        x1 = x[:, 0]
        return (
            2.0 * torch.exp(-0.5 * (0.1 - x1) ** 2 / 0.1**2)
            + 4.0 * torch.exp(-0.5 * (0.9 - x1) ** 2 / 0.01**2)
        )


class BraninForrester(BaseProblem):
    """Branin with Forrester's +5*x1 term, single global min (s_p.py:136, dom L163-164).

    optimal_value refined from their xopt (-3.689, 13.629); their coarse yopt is
    -16.64402, the precise minimum is -16.64402157084319.
    """

    death_name = "BraninForrester"

    def __init__(self) -> None:
        self.name = "BraninForrester"
        self.dim = 2
        self.bounds = _bounds([-5.0, 0.0], [10.0, 15.0])
        self.optimal_value = 16.64402157084319

    def _evaluate(self, x: Tensor) -> Tensor:
        b = 5.1 / (4 * math.pi**2)
        c = 5.0 / math.pi
        t = 1.0 / (8 * math.pi)
        x1, x2 = x[:, 0], x[:, 1]
        g = (x2 - b * x1**2 + c * x1 - 6.0) ** 2 + 10.0 * (1 - t) * torch.cos(x1) + 10.0 + 5.0 * x1
        return -g


class Cosines(BaseProblem):
    """Cosines (Gonzalez et al.) 2-d (synthetic_problems.py:287, dom L310-311).

    f = -(1 - sum_i (g_i - r_i)); g=(1.6x-0.5)^2, r=0.3 cos(3 pi (1.6x-0.5)).
    Exact analytic optimum -1.6 at (0.3125, 0.3125) -> optimal_value 1.6.
    """

    death_name = "Cosines"

    def __init__(self) -> None:
        self.name = "Cosines"
        self.dim = 2
        self.bounds = _bounds([0.0, 0.0], [5.0, 5.0])
        self.optimal_value = 1.6

    def _evaluate(self, x: Tensor) -> Tensor:
        u = 1.6 * x - 0.5
        g = u**2
        r = 0.3 * torch.cos(3 * math.pi * u)
        return 1.0 - (g - r).sum(dim=1)  # = -their f


class GoldsteinPriceLog(BaseProblem):
    """log(Goldstein-Price) 2-d (synthetic_problems.py:190 logGoldsteinPrice, dom L202-203).

    Inner Goldstein-Price has exact global min 3 at (0, -1); no extra shift, so
    f* = log(3) and optimal_value = -log(3).
    """

    death_name = "logGoldsteinPrice"

    def __init__(self) -> None:
        self.name = "GoldsteinPriceLog"
        self.dim = 2
        self.bounds = _bounds([-2.0, -2.0], [2.0, 2.0])
        self.optimal_value = -math.log(3.0)

    def _evaluate(self, x: Tensor) -> Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        term1 = 1 + (x1 + x2 + 1) ** 2 * (
            19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        )
        term2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        )
        return -torch.log(term1 * term2)


class SixHumpCamelLog(BaseProblem):
    """log(Six-Hump Camel + 1.0316 + 1e-4) 2-d (synthetic_problems.py:326, dom L338-339).

    shift a+b = 1.0316 + 1e-4 (their L353-354). The true camel min is
    g* = -1.031628453489877, so the precise f* = log(g* + 1.0317) = -9.54516282851,
    distinct from De Ath's coarse yopt -9.54473575989 (the visible SixHumpCamel
    gap); the sensitivity comes from g* + shift being ~7e-5.
    """

    death_name = "logSixHumpCamel"
    _shift = 1.0316 + 1e-4

    def __init__(self) -> None:
        self.name = "SixHumpCamelLog"
        self.dim = 2
        self.bounds = _bounds([-3.0, -2.0], [3.0, 2.0])
        # g* = -1.031628453489877 (precise camel minimum)
        self.optimal_value = -math.log(-1.031628453489877 + self._shift)

    def _evaluate(self, x: Tensor) -> Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        g = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
        return -torch.log(g + self._shift)


class Hartmann6Log(BaseProblem):
    """-(-log(sum_j a_j exp(-sum_k A_jk (x_k-P_jk)^2))) 6-d (s_p.py:393, dom L400-401).

    Their logHartmann6 returns -log(S(x)); S* = 3.32236801141551 (the Hartmann-6
    optimum), so f* = -log(S*) and optimal_value = log(S*) = 1.200677785132358.
    """

    death_name = "logHartmann6"

    def __init__(self) -> None:
        self.name = "Hartmann6Log"
        self.dim = 6
        self.bounds = _bounds([0.0] * 6, [1.0] * 6)
        self.optimal_value = math.log(3.32236801141551)
        self._alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.double)
        self._A = torch.tensor(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ],
            dtype=torch.double,
        )
        self._P = torch.tensor(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ],
            dtype=torch.double,
        )

    def _evaluate(self, x: Tensor) -> Tensor:
        diff = x[:, None, :] - self._P[None, :, :]  # (n, 4, 6)
        inner = (self._A[None, :, :] * diff**2).sum(dim=2)  # (n, 4)
        s = (self._alpha[None, :] * torch.exp(-inner)).sum(dim=1)  # (n,)
        return torch.log(s)  # = -their f (which is -log(s))


class GSobolLog(BaseProblem):
    """log(gSobol) 10-d (synthetic_problems.py:497 logGSobol, dom L518-519).

    From THEIR CODE: prod_i (|4 x_i - 2| + a_i) / (1 + a_i), a_i = 1, no shift.
    (The supplementary prints prod (4 x_i - a_i)/2, which can be <= 0 and make the
    log undefined -- a typesetting error; the code form is the standard gSobol.)
    At x=0.5 each factor is 0.5, so g* = 0.5^10 and optimal_value = -log(0.5^10).
    """

    death_name = "logGSobol"

    def __init__(self) -> None:
        self.name = "GSobolLog"
        self.dim = 10
        self.bounds = _bounds([-5.0] * 10, [5.0] * 10)
        self.optimal_value = -math.log(0.5**10)

    def _evaluate(self, x: Tensor) -> Tensor:
        factors = (torch.abs(4 * x - 2) + 1.0) / 2.0
        return -torch.log(factors.prod(dim=1))


class GSobol(BaseProblem):
    """Raw gSobol 10-d (synthetic_problems.py:534, dom L555-556); C5 transform axis.

    prod_i (|4 x_i - 2| + 1)/2 (their code form; see GSobolLog on the supp typo).
    g* = 0.5^10 at x=0.5 -> optimal_value = -0.5^10.
    """

    death_name = "GSobol"

    def __init__(self) -> None:
        self.name = "GSobol"
        self.dim = 10
        self.bounds = _bounds([-5.0] * 10, [5.0] * 10)
        self.optimal_value = -(0.5**10)

    def _evaluate(self, x: Tensor) -> Tensor:
        factors = (torch.abs(4 * x - 2) + 1.0) / 2.0
        return -factors.prod(dim=1)


class RosenbrockLog(BaseProblem):
    """log(Rosenbrock + 0.5) 10-d (synthetic_problems.py:571 logRosenbrock, dom L582-583).

    sum_i 100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2, +0.5 shift. Raw min is exactly 0
    at x=1, so f* = log(0.5) and optimal_value = -log(0.5) = log(2).
    """

    death_name = "logRosenbrock"

    def __init__(self) -> None:
        self.name = "RosenbrockLog"
        self.dim = 10
        self.bounds = _bounds([-5.0] * 10, [10.0] * 10)
        self.optimal_value = -math.log(0.5)

    def _evaluate(self, x: Tensor) -> Tensor:
        sqr = x**2
        last = (x - 1) ** 2
        s = (100 * (x[:, 1:] - sqr[:, :-1]) ** 2 + last[:, :-1]).sum(dim=1) + 0.5
        return -torch.log(s)


class Rosenbrock(BaseProblem):
    """Raw Rosenbrock 10-d (synthetic_problems.py:600, dom L611-612); C5 transform axis.

    sum_i 100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2; exact analytic min 0 at x=1 ->
    optimal_value = 0.
    """

    death_name = "Rosenbrock"

    def __init__(self) -> None:
        self.name = "Rosenbrock"
        self.dim = 10
        self.bounds = _bounds([-5.0] * 10, [10.0] * 10)
        self.optimal_value = 0.0

    def _evaluate(self, x: Tensor) -> Tensor:
        s = (100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2).sum(dim=1)
        return -s


class StyblinskiTangLog(BaseProblem):
    """log(0.5*sum(x^4-16x^2+5x) + 400) 10-d (s_p.py:633 logStyblinskiTang, dom L643-644).

    shift = 40*D = 400. Per-dim min at x0 = -2.903534027771177 (root of 4x^3-32x+5);
    optimal_value = -log(5*h(x0) + 400) = -2.1208645110528286.
    """

    death_name = "logStyblinskiTang"

    def __init__(self) -> None:
        self.name = "StyblinskiTangLog"
        self.dim = 10
        self.bounds = _bounds([-5.0] * 10, [5.0] * 10)
        self.optimal_value = -2.1208645110528286

    def _evaluate(self, x: Tensor) -> Tensor:
        s = 0.5 * (x**4 - 16 * x**2 + 5 * x).sum(dim=1) + 40.0 * self.dim
        return -torch.log(s)


# --- De Ath initial-design loader (D8) -------------------------------------

_DEFAULT_TRAINING_DATA = os.path.expanduser("~/projects/egreedy/training_data")


def load_death_initial_design(
    death_name: str, run_no: int, data_dir: str | os.PathLike | None = None
) -> Tensor:
    """Return De Ath's published initial design X for (problem, run_no), shape (M, d).

    Reads ``{data_dir}/{death_name}_{run_no}.npz`` (key ``arr_0`` = X, ``arr_1`` =
    Y), where ``death_name`` is the De Ath file stem (e.g. "logSixHumpCamel"; see
    each problem's ``death_name``). Y is ignored: this project re-evaluates it
    through its own deterministic pipeline. Returns a float64 tensor for injection
    into ``run_bo(initial_design=...)``.
    """
    base = Path(data_dir) if data_dir is not None else Path(_DEFAULT_TRAINING_DATA)
    path = base / f"{death_name}_{run_no}.npz"
    with np.load(path) as npz:
        x = np.asarray(npz["arr_0"], dtype=np.float64)
    return torch.as_tensor(x, dtype=torch.double)
