"""Acquisition strategy interface."""
from abc import ABC, abstractmethod

from botorch.models.model import Model
from torch import Tensor


class BaseStrategy(ABC):
    """Picks the next point to evaluate given the fitted surrogate and data."""

    name: str

    @abstractmethod
    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,
        train_y: Tensor,
    ) -> Tensor:
        """Return the next point to evaluate, shape (1, dim).

        bounds: (2, dim); train_x: (n, dim); train_y: (n, 1).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
