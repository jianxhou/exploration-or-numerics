"""Numerically stable log Expected Improvement acquisition strategy.

LogEI (Ament et al. 2023) has the same maximiser as EI but is computed in log
space, avoiding the value/gradient underflow that cripples EI's acquisition
optimisation. To make the two arms comparable, this strategy is wired through
the *identical* code path as :class:`~al_benchmark.strategies.ei.EI`
(:func:`optimize_improvement_acqf`); the only difference is the acquisition
class passed in -- ``LogExpectedImprovement`` instead of ``ExpectedImprovement``.
"""
from botorch.acquisition import LogExpectedImprovement
from botorch.models.model import Model
from torch import Tensor

from al_benchmark.probe import AcqProbe
from al_benchmark.strategies._improvement import optimize_improvement_acqf
from al_benchmark.strategies.base import BaseStrategy


class LogEI(BaseStrategy):
    """Analytic LogExpectedImprovement (maximization).

    Constructor mirrors :class:`~al_benchmark.strategies.ei.EI` exactly so the EI
    and LogEI arms share model construction, fit, ``best_f`` convention, and
    optimiser settings; ``probe=True`` attaches the same diagnostics.
    """

    def __init__(
        self,
        num_restarts: int = 10,
        raw_samples: int = 64,
        probe: bool = False,
    ) -> None:
        self.name = "LogEI"
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self._probe = AcqProbe() if probe else None
        self.probe_log = self._probe.log if self._probe is not None else None

    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,
        train_y: Tensor,
    ) -> Tensor:
        return optimize_improvement_acqf(
            LogExpectedImprovement,
            model,
            bounds,
            train_y.max(),
            self.num_restarts,
            self.raw_samples,
            self._probe,
        )
