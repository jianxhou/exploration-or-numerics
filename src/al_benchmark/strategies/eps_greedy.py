"""Epsilon-greedy strategies (De Ath et al. 2021): eps-RS and eps-PF.

Both spend a 1 - eps fraction of selections on the greedy
:class:`~al_benchmark.strategies.exploit.Exploit` point and an eps fraction on an
exploratory move. They differ only in how the exploratory point is drawn:
eps-RS (file label ``eRandom``) takes a uniform random point in the domain;
eps-PF (``eFront``) takes a uniform random point from the surrogate's
mean/sigma trade-off front.

RNG discipline. Strategy-level randomness (the eps coin, the random/front draws)
uses a dedicated ``torch.Generator`` seeded deterministically from the run seed
(``torch.initial_seed()``, set by ``run_bo``), never the global numpy RNG. This
generator is kept disjoint from the global torch RNG that ``optimize_acqf`` and
Sobol candidate generation consume, which is what makes ``eps=0`` reduce to
Exploit *exactly*: the coin draw does not perturb the global stream that the
Exploit optimisation depends on. Same seed -> identical coins -> identical
selections; the run seed couples the exploration to the rest of the pipeline.
"""
import torch
from botorch.models.model import Model
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from al_benchmark.strategies.base import BaseStrategy
from al_benchmark.strategies.exploit import Exploit


def _pareto_front_max_max(a: Tensor, b: Tensor) -> Tensor:
    """Boolean mask of the non-dominated set maximizing both ``a`` and ``b``.

    Point i is dominated iff some j has a_j >= a_i and b_j >= b_i with at least
    one strict inequality. Ties (equal in both objectives) are mutually
    non-dominated and both retained. O(n^2) with an inner vectorised pass; the
    candidate pool is small (raw_samples), so this is negligible.
    """
    n = a.shape[0]
    dominated = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        ge = (a >= a[i]) & (b >= b[i])
        strict = (a > a[i]) | (b > b[i])
        if bool((ge & strict).any()):
            dominated[i] = True
    return ~dominated


class _EpsGreedy(BaseStrategy):
    """Shared eps-greedy machinery: the coin flip and the dedicated RNG.

    Subclasses set ``_seed_offset`` (so different eps strategies draw distinct,
    decorrelated coin streams from the same run seed) and implement ``_explore``.
    """

    _seed_offset: int = 0

    def __init__(
        self,
        eps: float = 0.1,
        num_restarts: int = 10,
        raw_samples: int = 64,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        self.eps = eps
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self._exploit = Exploit(num_restarts, raw_samples)
        self._seed = seed
        self._rng: torch.Generator | None = None

    def _generator(self) -> torch.Generator:
        """Lazily build the dedicated generator, seeded from the run seed.

        Lazy so the seed is read *after* ``run_bo`` has set the global seed
        (strategies are constructed before the run starts). An explicit
        constructor ``seed`` overrides the run seed for direct unit testing.
        Reading ``torch.initial_seed()`` does not advance the global stream.
        """
        if self._rng is None:
            base = self._seed if self._seed is not None else torch.initial_seed()
            self._rng = torch.Generator()
            self._rng.manual_seed(int(base) + self._seed_offset)
        return self._rng

    def _explore(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,
        train_y: Tensor,
        gen: torch.Generator,
    ) -> Tensor:
        raise NotImplementedError

    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,
        train_y: Tensor,
    ) -> Tensor:
        gen = self._generator()
        # Coin drawn from the dedicated generator: it never touches the global
        # RNG, so the Exploit branch sees the same stream a bare Exploit would.
        coin = torch.rand(1, generator=gen).item()
        if coin < self.eps:
            return self._explore(model, bounds, train_x, train_y, gen)
        return self._exploit.select_next(model, bounds, train_x, train_y)


class EpsRS(_EpsGreedy):
    """eps-RS / eRandom: greedy with probability 1 - eps, else a uniform random
    point in the domain (De Ath et al. 2021). eps defaults to 0.1.
    """

    _seed_offset = 0x5253  # 'RS'

    def __init__(
        self,
        eps: float = 0.1,
        num_restarts: int = 10,
        raw_samples: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(eps, num_restarts, raw_samples, seed)
        self.name = f"eps-RS(eps={eps})"

    def _explore(self, model, bounds, train_x, train_y, gen):
        lower, upper = bounds[0], bounds[1]
        u = torch.rand(1, bounds.shape[-1], generator=gen, dtype=bounds.dtype)
        return lower + (upper - lower) * u  # (1, d)


class EpsPF(_EpsGreedy):
    """eps-PF / eFront: greedy with probability 1 - eps, else a uniform random
    point from the surrogate's approximate Pareto front (De Ath et al. 2021).

    Over the Sobol candidate pool the non-dominated set of (mean, sigma) is taken
    and a member is chosen uniformly. Convention: De Ath minimize, so their front
    trades off (minimize mean, maximize sigma); this project negates objectives
    to maximize, so the equivalent front is (maximize mean, maximize sigma) -- its
    greedy end coincides with the Exploit point. Framework difference: De Ath
    approximate the front with NSGA-II over the GP; we approximate it over the
    acquisition candidate pool. eps defaults to 0.1.
    """

    _seed_offset = 0x5046  # 'PF'

    def __init__(
        self,
        eps: float = 0.1,
        num_restarts: int = 10,
        raw_samples: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(eps, num_restarts, raw_samples, seed)
        self.name = f"eps-PF(eps={eps})"

    def _explore(self, model, bounds, train_x, train_y, gen):
        # Sobol pool seeded from the dedicated generator (explicit seed -> no
        # global-RNG draw), keeping the global stream reserved for Exploit.
        pool_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
        cand = draw_sobol_samples(
            bounds=bounds, n=self.raw_samples, q=1, seed=pool_seed
        ).squeeze(1)  # (raw, d)

        post = model.posterior(cand)
        mean = post.mean.squeeze(-1)
        sigma = post.variance.clamp_min(0).sqrt().squeeze(-1)

        front = _pareto_front_max_max(mean, sigma)
        idx = front.nonzero(as_tuple=False).squeeze(-1)
        pick = int(idx[int(torch.randint(len(idx), (1,), generator=gen))])
        return cand[pick : pick + 1]  # (1, d)
