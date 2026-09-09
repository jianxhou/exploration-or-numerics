"""Section 5.2 mechanism probe (Phase 3, pre-registration phase3-prereg) -- FROZEN protocol.

A PASSIVE per-iteration diagnostic for the NSGA-II production matrix. At every BO
iteration, on a fixed K=4096 Sobol probe set over the problem domain, it records:
  - raw-EI value underflow (for EVERY arm): fraction of probe points with raw EI == 0.0,
    plus value max/median -- the cross-optimizer-comparable underflow signal;
  - raw-LogEI value max/median;
  - the arm's OWN acquisition-surface gradient health (||grad||_2 via autograd): the
    fraction below 1e-12 and the median log10 norm (EI arm -> EI gradient, LogEI arm ->
    LogEI gradient, UCB -> UCB, Exploit -> PosteriorMean; eps-PF has no single scalar
    surface -> null);
  - optimizer progress: the own-acquisition value at the returned candidate vs the best
    own-acquisition over the probe set, and the rank of the returned candidate among the
    K probe points.

Descriptive ONLY (pre-reg Claim C / H3 attribution); it NEVER changes selection. It is
attached by wrapping a strategy in :class:`MechanismProbedStrategy`, whose ``select_next``
delegates verbatim to the inner arm and exposes the per-iteration log as ``probe_log`` so
``run_bo`` collects it unchanged.

Separability (pre-reg §4.4a). The K=4096 probe set is drawn once at construction with an
explicit, fixed probe seed distinct from the BO seed (botorch's seeded SobolEngine, a local
generator), so it does NOT advance the global torch RNG; the injected initial design and the
per-iteration GP refit are untouched. The probe set is a pure function of the problem bounds,
hence identical across all cells of the same problem (the frozen §5.2 requirement).
"""
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.models.model import Model
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from al_benchmark.strategies.base import BaseStrategy

K_PROBE = 4096  # frozen §5.2 probe-set size
_TINY = float(np.finfo(np.float64).tiny)  # smallest positive normal float64
_GRAD_TINY = 1e-12  # frozen §5.2 vanishing-gradient threshold
# Fixed probe seed, distinct from any BO/cell seed; same probe set per problem bounds.
_PROBE_SEED = int.from_bytes(b"PROBE", "big") % (2**31 - 1)


def _probe_set(bounds: Tensor) -> Tensor:
    """K=4096 Sobol points over the problem bounds, identical across all cells of the
    problem (deterministic in the bounds and the fixed probe seed; local SobolEngine,
    so the global torch RNG is untouched)."""
    return draw_sobol_samples(bounds=bounds, n=K_PROBE, q=1, seed=_PROBE_SEED).squeeze(1)  # (K, d)


def _value_stats(acq, x1: Tensor) -> dict:
    """Value stats of an analytic acquisition on the probe set (x1: (K, 1, d))."""
    with torch.no_grad():
        v = acq(x1)  # (K,)
    finite = v[torch.isfinite(v)]
    return {
        "frac_zero": float((v == 0).double().mean()),
        "frac_below_tiny": float((v < _TINY).double().mean()),
        "max": float(finite.max()) if finite.numel() else float("nan"),
        "median": float(finite.median()) if finite.numel() else float("nan"),
    }


def _grad_stats(acq, x1: Tensor) -> dict:
    """Own-surface gradient health: ||grad_x acq||_2 on the probe set via one batched
    backward (each output depends only on its own row, so no cross-terms)."""
    x = x1.clone().requires_grad_(True)
    v = acq(x)
    v.sum().backward()
    g = x.grad.reshape(x.shape[0], -1).norm(dim=-1)
    g = g[torch.isfinite(g)]
    pos = g[g > 0]
    return {
        "frac_below_tiny": float((g < _GRAD_TINY).double().mean()) if g.numel() else float("nan"),
        "median_log10_norm": float(pos.log10().median()) if pos.numel() else float("nan"),
    }


class MechanismProbe:
    """Accumulates one §5.2 diagnostics dict per BO iteration in ``log``."""

    def __init__(self, bounds: Tensor) -> None:
        self._x1 = _probe_set(bounds).unsqueeze(1)  # (K, 1, d)
        self.log: list[dict] = []
        self._pending: dict | None = None
        self._own_vals: Tensor | None = None

    def observe(self, model: Model, best_f: Tensor, own_acq, own_label) -> None:
        """Record value/underflow/gradient stats BEFORE the optimizer selects a point."""
        ei = ExpectedImprovement(model=model, best_f=best_f)
        logei = LogExpectedImprovement(model=model, best_f=best_f)
        ei_val = _value_stats(ei, self._x1)
        logei_val = _value_stats(logei, self._x1)
        entry: dict = {
            "raw_ei_frac_zero": ei_val["frac_zero"],
            "raw_ei_frac_below_tiny": ei_val["frac_below_tiny"],
            "raw_ei_max": ei_val["max"],
            "raw_ei_median": ei_val["median"],
            "raw_logei_max": logei_val["max"],
            "raw_logei_median": logei_val["median"],
            "own_acq": own_label,
        }
        if own_acq is not None:
            entry["own_grad"] = _grad_stats(own_acq, self._x1)
            with torch.no_grad():
                self._own_vals = own_acq(self._x1)  # (K,)
        else:
            entry["own_grad"] = None
            self._own_vals = None
        self._pending = entry

    def record_candidate(self, own_acq, candidate: Tensor) -> None:
        """Record optimizer progress AFTER the optimizer returns its candidate, then
        commit the iteration's entry to the log."""
        entry = self._pending
        if own_acq is not None and self._own_vals is not None:
            with torch.no_grad():
                cand_val = float(own_acq(candidate.unsqueeze(1)))  # (1,d) -> (1,1,d) -> scalar
            probe_vals = self._own_vals
            entry["progress"] = {
                "cand_own_acq": cand_val,
                "probe_own_acq_max": float(probe_vals.max()),
                "rank": int((probe_vals > cand_val).sum()),  # 0 = beats every probe point
                "n_probe": int(probe_vals.numel()),
            }
        else:
            entry["progress"] = None
        self.log.append(entry)
        self._pending = None
        self._own_vals = None


class MechanismProbedStrategy(BaseStrategy):
    """Wraps an arm with the §5.2 probe. Selection is delegated verbatim to ``inner``;
    the probe observes the GP/incumbent before selection and the candidate after, then
    exposes its log as ``probe_log`` for ``run_bo`` to collect."""

    def __init__(self, inner: BaseStrategy, bounds: Tensor) -> None:
        self.inner = inner
        self.name = inner.name
        self._probe = MechanismProbe(bounds)
        self.probe_log = self._probe.log

    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,
        train_y: Tensor,
    ) -> Tensor:
        best_f = train_y.max()
        # The arm's own scalar acquisition (None for the bi-objective eps-PF).
        own_acq = self.inner._make_acq(model, train_y) if hasattr(self.inner, "_make_acq") else None
        own_label = getattr(self.inner, "_acq_label", None)
        self._probe.observe(model, best_f, own_acq, own_label)
        candidate = self.inner.select_next(model, bounds, train_x, train_y)
        self._probe.record_candidate(own_acq, candidate)
        return candidate
