"""Shared acquisition-optimisation path for the improvement-based strategies.

EI and LogEI must differ in *exactly one* thing: the acquisition class. The
blueprint's fairness hard rule (Section 7) requires identical model, fit, best_f
convention, optimiser call, ``num_restarts``, ``raw_samples`` and bounds across
the two arms, so this single helper is the only acquisition-optimisation path
both use. An optional :class:`~al_benchmark.probe.AcqProbe` instruments it; when
absent the call is byte-identical to the pre-probe ``optimize_acqf`` invocation.
"""
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from torch import Tensor

from al_benchmark.probe import AcqProbe


def optimize_improvement_acqf(
    acq_cls: type,
    model: Model,
    bounds: Tensor,
    best_f: Tensor,
    num_restarts: int,
    raw_samples: int,
    probe: AcqProbe | None = None,
) -> Tensor:
    """Optimise an improvement acqf and return the next point, shape (1, dim).

    ``acq_cls`` is constructed as ``acq_cls(model=model, best_f=best_f)`` -- the
    sole difference between EI (``ExpectedImprovement``) and LogEI
    (``LogExpectedImprovement``). ``best_f=train_y.max()`` follows the project's
    maximization convention. When ``probe`` is given, the same logical path runs
    under instrumentation; otherwise it is the plain ``optimize_acqf`` call.
    """
    if probe is not None:
        return probe.run(acq_cls, model, bounds, best_f, num_restarts, raw_samples)

    acq = acq_cls(model=model, best_f=best_f)
    candidate, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidate
