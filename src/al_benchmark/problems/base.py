"""Benchmark problem interface.

Convention: all problems are framed as maximization (BoTorch standard);
naturally minimized functions are negated.
"""
from abc import ABC, abstractmethod

from torch import Tensor


class BaseProblem(ABC):
    """Subclasses implement `_evaluate` and set name, dim, bounds, optimal_value."""

    name: str
    dim: int
    bounds: Tensor          # shape (2, dim): row 0 = lower, row 1 = upper
    optimal_value: float    # global max of the (possibly negated) function

    @abstractmethod
    def _evaluate(self, x: Tensor) -> Tensor:
        """x: (n, dim) query points -> (n,) values to be maximized."""
        ...

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate with shape checking; accepts (n, dim) or a single (dim,) point."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"{self.name}: expected input with {self.dim} dimensions, "
                f"got shape {tuple(x.shape)}"
            )
        return self._evaluate(x)

    def regret(self, y_best: float) -> float:
        """Simple regret: optimal_value - y_best (>= 0; smaller is better)."""
        return self.optimal_value - y_best

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"
