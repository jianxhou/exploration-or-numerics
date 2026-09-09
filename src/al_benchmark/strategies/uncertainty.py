"""Pure-exploration acquisition strategy (maximum posterior std)."""
from botorch.acquisition import PosteriorStandardDeviation
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from torch import Tensor

from al_benchmark.strategies.base import BaseStrategy


class Uncertainty(BaseStrategy):
    """Maximize posterior standard deviation, ignoring the mean.

    Baseline that isolates the value of the GP's uncertainty estimate.
    num_restarts / raw_samples configure the inner optimizer.
    """

    def __init__(
        self,
        num_restarts: int = 10,
        raw_samples: int = 64,
    ) -> None:
        self.name = "Uncertainty"
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,   # unused
        train_y: Tensor,   # unused
    ) -> Tensor:
        acq = PosteriorStandardDeviation(model=model)

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return candidate
