"""Engineering benchmark problems with multi-scale physical inputs."""
import numpy as np
import torch
from torch import Tensor

from al_benchmark.problems.base import BaseProblem


class Borehole(BaseProblem):
    """Borehole water-flow function (8D), maximized.

    The 8 inputs are physical parameters spanning ~6 orders of magnitude.
    SMT exposes this under the name `WaterFlow` (not `Borehole`) and returns
    the raw flow rate.

    Monotone in each input, so the global maximum lies at a box vertex;
    evaluating all 2^8 = 256 corners gives 309.5756. optimal_value = 310.0
    includes a small buffer so simple regret stays non-negative. (An earlier
    random-sampling estimate of ~285 never reached the corner and produced
    negative regret once input normalization let BO find the optimum.)
    """

    def __init__(self) -> None:
        from smt.problems import WaterFlow

        self.name = "Borehole"
        self.dim = 8
        self._fn = WaterFlow()
        # SMT stores bounds as xlimits with shape (dim, 2); BaseProblem wants (2, dim)
        xlimits = self._fn.xlimits  # (8, 2)
        self.bounds = torch.tensor(xlimits.T, dtype=torch.float64)  # (2, 8)
        self.optimal_value = 310.0  # corner max 309.5756 + buffer (see docstring)

    def _evaluate(self, x: Tensor) -> Tensor:
        # SMT expects a (n, dim) numpy float array and returns (n, 1) numpy
        x_np = x.detach().cpu().numpy().astype(np.float64)
        y_np = self._fn(x_np)  # (n, 1), raw flow rate (maximization)
        y = torch.tensor(y_np.ravel(), dtype=x.dtype, device=x.device)  # (n,)
        return y

class Piston(BaseProblem):
    """Piston simulation function (7D): cycle time in seconds, maximized.

    Implemented from the standard formula (Surjanovic & Bingham, Virtual
    Library of Simulation Experiments) — not available in this SMT version.
    Input order: [M, S, V0, k, P0, Ta, T0].

    Near-monotone in each input; the global max over the box is ~1.199 s at a
    vertex (200k random points never exceeded the corner value).
    optimal_value = 1.20 includes a buffer so simple regret stays
    non-negative. Midpoint value ~0.4644.
    """

    def __init__(self) -> None:
        self.name = "Piston"
        self.dim = 7
        # rows: [lower; upper]; columns: M, S, V0, k, P0, Ta, T0
        self.bounds = torch.tensor(
            [
                [30.0, 0.005, 0.002, 1000.0, 90000.0, 290.0, 340.0],
                [60.0, 0.020, 0.010, 5000.0, 110000.0, 296.0, 360.0],
            ],
            dtype=torch.double,
        )
        self.optimal_value = 1.20

    def _evaluate(self, x: Tensor) -> Tensor:
        x = x.to(dtype=torch.double)
        M, S, V0, k, P0, Ta, T0 = (x[..., i] for i in range(7))
        A = P0 * S + 19.62 * M - k * V0 / S
        PV = P0 * V0 / T0
        V = (S / (2.0 * k)) * (torch.sqrt(A**2 + 4.0 * k * PV * Ta) - A)
        C = 2.0 * torch.pi * torch.sqrt(M / (k + S**2 * PV * (Ta / V**2)))
        return C
