"""GP surrogate wrapping BoTorch's SingleTaskGP."""
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


class GPSurrogate:
    """SingleTaskGP with fixed-bounds input normalization to [0,1]^d and
    output standardization, so the GP stays calibrated on multi-scale inputs.
    """

    name = "GP"

    def __init__(self) -> None:
        self.model: SingleTaskGP | None = None

    def fit(self, train_x: Tensor, train_y: Tensor, bounds: Tensor) -> SingleTaskGP:
        """Fit on raw-scale train_x (n, dim), train_y (n, 1); bounds (2, dim).

        Transforms are baked into the returned model: callers pass raw-scale
        points and get raw-scale predictions.
        """
        dim = train_x.shape[-1]
        self.model = SingleTaskGP(
            train_x,
            train_y,
            input_transform=Normalize(d=dim, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        return self.model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

