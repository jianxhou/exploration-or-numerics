"""Expected Improvement acquisition strategy."""
from botorch.acquisition import ExpectedImprovement
from botorch.models.model import Model
from torch import Tensor

from al_benchmark.probe import AcqProbe
from al_benchmark.strategies._improvement import optimize_improvement_acqf
from al_benchmark.strategies.base import BaseStrategy


class EI(BaseStrategy):
    """Expected Improvement over the best observed value (maximization).

    num_restarts / raw_samples configure the inner acquisition optimizer.
    ``probe=True`` attaches an :class:`~al_benchmark.probe.AcqProbe`; its
    per-iteration log is exposed as ``probe_log`` and collected by ``run_bo``.
    With ``probe=False`` (default) the path is identical to the pre-probe
    implementation, so existing experiments are unaffected.
    """

    def __init__(
        self,
        num_restarts: int = 10,
        raw_samples: int = 64,
        probe: bool = False,
    ) -> None:
        self.name = "EI"
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
            ExpectedImprovement,
            model,
            bounds,
            train_y.max(),
            self.num_restarts,
            self.raw_samples,
            self._probe,
        )
