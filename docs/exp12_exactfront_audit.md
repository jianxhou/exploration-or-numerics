# Code Audit — `EpsPFExactNSGA2` vs De Ath's released eFront

Fidelity source: De Ath's released code at `~/projects/egreedy` (github.com/georgedeath/egreedy,
commit 4ab1a99), files `egreedy/acquisition_functions/egreedy_acq_funcs_minimize.py`,
`egreedy/acquisition_functions/nsga2_pareto_front.py`,
`egreedy/acquisition_functions/acq_func_optimisers.py`, `egreedy/optimizer.py`.
Ours: `src/al_benchmark/strategies/posthoc_exp12.py::EpsPFExactNSGA2`.

## Line-by-line mapping

| # | De Ath (released code) | Ours | Match |
|---|---|---|---|
| 1 | `optimizer.py:174`: `acq_budget = 5000 * f_dim` | Frozen `_POP_MULT=100`, `_N_GEN=50`; pymoo `("n_gen", 50)` evaluates the initial population (gen 1) plus 49 offspring generations = `100d * 50 = 5000d` evaluations per step | Yes (same budget) |
| 2 | `nsga2_pareto_front.py:107-108`: `POPSIZE = D*100`, `N_GENS = ceil(fevals/POPSIZE)` = 50 | population `100d`, 50 generations | Yes |
| 3 | `nsga2_pareto_front.py:111-119`: pygmo NSGA-II, `cr=0.8, eta_c=20, m=1/D, eta_m=20` | pymoo NSGA2, SBX `prob=0.8, eta=20`, PM `prob=1/d, eta=20` (frozen `_nsga2`) | Yes (settings); see divergence D1 (implementation) |
| 4 | `nsga2_pareto_front.py:95`: objectives `[mu, -sigma]` minimized (his problems are minimization) | objectives `[-mean, -sigma]` minimized by pymoo, i.e. `(mean, sigma)` maximized (this repo maximizes) | Yes (sign-flipped equivalently; the front is the same trade-off) |
| 5 | `nsga2_pareto_front.py:97-99, 122-123`: every evaluated point is archived (`model_fitness.X/Y` preallocated for the full run) | `_ArchivedParetoProblem` appends every `_evaluate` batch to `arch_X/arch_F` | Yes |
| 6 | `nsga2_pareto_front.py:137`: front = `pg.non_dominated_front_2d(model_fitness.Y)` over the archive | `_nd_front_2d_max(-F_min)` over the archive: O(n log n) sort-sweep with dominance "a dominates b iff a >= b componentwise and a > b somewhere" | Yes (semantics verified against a brute-force dominance filter, `fidelity_check()`: 0 mismatches on 200 random clouds with injected duplicates and axis ties) |
| 7 | `acq_func_optimisers.py:129-132` (`eFront.__call__`): the front is constructed unconditionally each iteration, then selection happens | `select_next` builds the front first, then flips the coin | Yes (order of operations preserved; budget spent every iteration) |
| 8 | `egreedy_acq_funcs_minimize.py:45-46`: `if np.random.rand() < epsilon: return X[np.random.choice(mu.size), :]` | dedicated `torch.Generator` coin; uniform `torch.randint` over front members | Yes (semantics); see divergence D2 (RNG stream) |
| 9 | `egreedy_acq_funcs_minimize.py:50`: exploit = `X[np.argmin(mu.ravel()), :]` (front member with best mean; minimization) | `front_x[argmax(mu)]` (front member with best mean; maximization) | Yes (sign-flipped equivalently) |
| 10 | `optimizer.py`: epsilon = 0.1 (`eFront_eps0.1`) | `eps=0.1` default | Yes |

## Residual divergences (explicit)

- **D1 — NSGA-II implementation:** his pygmo 2.x `pg.nsga2` vs our pymoo 0.6 `NSGA2`.
  Same algorithm and identical hyperparameters (population, generations, SBX/PM
  probabilities and distribution indices), but different codebases: internal
  tournament/crowding tie-breaking and RNG streams differ, so the evolved archives are
  not bitwise-identical for the same seed. This is the same pygmo-to-pymoo substitution
  already present in the frozen pre-registered arms.
- **D2 — RNG discipline:** his `np.random.rand()` on the global numpy stream vs our
  dedicated per-cell `torch.Generator` (the repo-wide 4.4a-style discipline that keeps
  strategy randomness off the global GP-fit streams). Distributionally identical
  (uniform coin, uniform front pick).
- **D3 — surrogate stack:** his GPy Matern-5/2 GP vs our BoTorch/GPyTorch Matern-5/2
  (the Phase-3 stack shared by every arm in the paper). The exact-front arm isolates
  the front-construction and selection mechanism; it does not, and cannot, also swap
  the surrogate.
- **D4 — duplicate handling at the front filter:** exact duplicate (mu, sigma) pairs
  are mutually non-dominating; ours keeps all copies. pygmo's
  `non_dominated_front_2d` behavior on exact duplicates is not documented; for
  continuous GP posteriors exact ties have measure zero.

## Fidelity verification performed

- `python experiments/run_exp12_posthoc.py --fidelity`: PASS
  (`nd_filter_mismatches: 0` over 200 random 400-point clouds including injected
  duplicate rows and axis ties; exploit pick equals the global best-mean point on
  every trial, which is De Ath's `argmin mu` under sign flip).
- A direct cross-run against his binary (same posterior in, same front out) is not
  feasible in this environment: his stack requires `pygmo` and `GPy`, neither of
  which is installed (and pygmo's RNG would preclude bitwise equality regardless,
  per D1). The fidelity anchor is therefore the line-by-line mapping above plus the
  semantic unit test, with D1-D4 as the complete list of residual divergences.

## Addendum — cell-file access before protocol amendment A1

Added after amendment A1 (`a58f2cb`), while the exp_12 production run was in
progress and before any exp_12 performance result had been read or computed.

For precision about what "before any result was read" means: prior to `a58f2cb`,
exp_12 cell JSON files WERE opened programmatically twice -- by the 6-cell smoke
validator and by the two d=6 timing runs. What was read was schema and metadata
only: the length of the `y` array (not its values), the arm/seed/problem labels,
the design-source string, and wall-clock time. No objective value, no regret, and
no aggregate quantity was read, computed, or displayed from any exp_12 cell before
`a58f2cb`; the exp_12 analysis output at that commit contained no arm aggregation
(no `arms` key). The paper states the provable form; this note records the full
factual form.

### Addendum 2 — exact disk state at amendment A1 (precision note, Day 38)

Measured from cell-file mtimes against commit timestamps during the pre-submit
verification (P3): at the moment A1 (`a58f2cb`) was committed, 101 of the 990 exp_12
cell files already existed on disk -- the 8 smoke/timing cells described above (the
only files ever opened before A1, schema/metadata reads only) plus 93 cells written
by the then-running production batch and never opened by any reader before A1. Zero
cells predate the protocol commit (`54fb7e1`). This makes the first addendum's
account precise; it changes no claim: the committed statements assert only that no
performance result had been read or computed before A1 (the analysis output carried
no arm aggregation at that commit), which the measured state confirms.
