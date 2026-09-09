"""Acquisition diagnostics probe (Phase 2 blueprint, Section 8).

Instruments improvement-based acquisition optimisation to expose the numerical
pathology Ament et al. (2023) attribute to legacy EI: candidate pools whose
acquisition values and gradients vanish over large regions, and optimiser
restarts that begin and end in degenerate (near-zero) acquisition surfaces.

Attached to EI; applied with *identical* logging to LogEI as a numerical
control. The value-based metrics (zero / below-tiny fractions, degeneracy flags)
are defined on the raw acqf output: for EI they measure underflow; for LogEI,
whose output is a log value (large-magnitude negative), the same thresholds are
trivially crossed and those fields serve only as a control marker -- the
*comparable* signals across the two arms are the gradient-availability and
gradient-norm statistics. Interpretation differs; the computation is the same.

Overhead is kept low: the raw Sobol candidate pool is drawn once and reused both
for the candidate-level diagnostics and to seed the optimiser restarts; no
additional candidate draws are made beyond what a probe-off run would perform.
"""
import warnings

import numpy as np
import torch
from botorch.exceptions.warnings import NumericsWarning
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.optim.initializers import initialize_q_batch
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

# Smallest positive normal float64; values at or below it have underflowed.
_TINY = float(np.finfo(np.float64).tiny)


class AcqProbe:
    """Per-iteration acquisition diagnostics; one JSON-serialisable entry per call.

    ``log`` accumulates one dict per BO iteration. Strategies hold an instance
    when constructed with ``probe=True`` and expose ``log`` as ``probe_log``,
    which ``run_bo`` attaches to the run result under the ``probe`` field.
    """

    def __init__(self) -> None:
        self.log: list[dict] = []

    def run(
        self,
        acq_cls: type,
        model: Model,
        bounds: Tensor,
        best_f: Tensor,
        num_restarts: int,
        raw_samples: int,
    ) -> Tensor:
        """Probe one acquisition step and return the chosen point, shape (1, dim).

        Builds the acqf while counting botorch NumericsWarnings, draws one Sobol
        candidate pool, computes candidate-level diagnostics on it, seeds the
        optimiser restarts from that same pool (botorch's Boltzmann default), then
        records per-restart diagnostics. The optimiser path mirrors the probe-off
        helper; only the initial conditions are materialised explicitly so the
        pool can be reused.
        """
        # Construct the acqf, counting (never suppressing) botorch NumericsWarnings.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            acq = acq_cls(model=model, best_f=best_f)
        n_numerics = sum(issubclass(w.category, NumericsWarning) for w in caught)

        # One Sobol candidate pool, reused for diagnostics and optimiser seeding.
        cand = draw_sobol_samples(bounds=bounds, n=raw_samples, q=1)  # (raw, 1, d)
        pool, pool_vals = self._candidate_pool(acq, cand)

        # Seed restarts from the same pool via botorch's Boltzmann selection.
        ics, _ = initialize_q_batch(cand, pool_vals, n=num_restarts)  # (nr, 1, d)
        with torch.no_grad():
            init_vals = acq(ics)
        candidates, final_vals = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            batch_initial_conditions=ics,
            return_best_only=False,
        )
        optimizer = self._optimizer_level(init_vals, final_vals)

        best = candidates[int(final_vals.argmax())].detach()  # (1, d)

        self.log.append(
            {
                "numerics_warnings": int(n_numerics),
                "candidate_pool": pool,
                "optimizer": optimizer,
            }
        )
        return best

    @staticmethod
    def _candidate_pool(acq, cand: Tensor) -> tuple[dict, Tensor]:
        """Candidate-pool diagnostics from a single batched, grad-enabled pass.

        Returns the stats dict and the detached acqf values (reused to seed the
        optimiser, so the pool is evaluated only once).
        """
        x = cand.clone().requires_grad_(True)
        vals = acq(x)  # (raw,)
        # One batched backward: each output depends only on its own candidate row,
        # so summing gives each candidate its own gradient with no cross-terms.
        vals.sum().backward()
        grad_norm = x.grad.reshape(x.shape[0], -1).norm(dim=-1)  # (raw,)

        v = vals.detach()
        n = int(v.numel())
        finite = v[torch.isfinite(v)]
        g = grad_norm[torch.isfinite(grad_norm)]

        stats = {
            "n_candidates": n,
            "frac_acqf_zero": float((v == 0).double().mean()),
            "frac_below_tiny": float((v < _TINY).double().mean()),
            "max_acqf": float(finite.max()) if finite.numel() else float("nan"),
            "median_acqf": float(finite.median()) if finite.numel() else float("nan"),
            "frac_nonzero_grad": float((g > 0).double().mean()) if g.numel() else float("nan"),
            "max_grad_norm": float(g.max()) if g.numel() else float("nan"),
            "median_grad_norm": float(g.median()) if g.numel() else float("nan"),
            "n_nan": int(torch.isnan(v).sum()),
            "n_inf": int(torch.isinf(v).sum()),
        }
        return stats, v

    @staticmethod
    def _optimizer_level(init_vals: Tensor, final_vals: Tensor) -> dict:
        """Per-restart diagnostics: initial/final acqf value and degeneracy flags."""
        iv = init_vals.detach()
        fv = final_vals.detach()
        degenerate = (fv == 0) | (fv < _TINY)
        return {
            "n_restarts": int(fv.numel()),
            "init_acqf": [float(x) for x in iv],
            "final_acqf": [float(x) for x in fv],
            "final_degenerate": [bool(x) for x in degenerate],
            "all_restarts_degenerate": bool(degenerate.all()),
        }
