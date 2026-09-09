"""Upper Confidence Bound acquisition strategy."""
from botorch.acquisition import UpperConfidenceBound
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from torch import Tensor

from al_benchmark.strategies.base import BaseStrategy


class UCB(BaseStrategy):
    """Upper Confidence Bound: mu(x) + sqrt(beta) * sigma(x).

    Larger beta = more exploration; beta=2.0 is roughly the 97.5%-quantile
    under a Gaussian posterior. BoTorch multiplies sigma by sqrt(beta), not
    beta. num_restarts / raw_samples configure the inner optimizer.
    """

    def __init__(
        self,
        beta: float = 2.0,
        num_restarts: int = 10,
        raw_samples: int = 64,
    ) -> None:
        self.name = f"UCB(beta={beta})"
        self.beta = beta
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,   # unused
        train_y: Tensor,   # unused
    ) -> Tensor:
        acq = UpperConfidenceBound(model=model, beta=self.beta)

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return candidate
