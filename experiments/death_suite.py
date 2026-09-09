"""De Ath suite integration for the experiments layer: registry + coarse yopt.

``DEATH_PROBLEMS`` maps ``--problem`` names to the ported De Ath synthetic
classes. ``DEATH_YOPT`` holds De Ath's own *coarse* code ``yopt`` (their
minimization f* = g evaluated at a rounded xopt), sourced verbatim from
``results/exp_05_tier1a.json`` (``f_opt_sources``).

Two optima tracks, kept deliberately distinct (blueprint Step 3b, item 2):

  * ``problem.optimal_value`` -- the *precise* optimum in this project's
    maximization convention; what ``problem.regret`` uses.
  * ``DEATH_YOPT[name]``      -- De Ath's *coarse* f* in their minimization
    convention; for reproducing their Table 2 regret only.

To compute De Ath-comparable regret from one of our maximization runs (best =
max observed): ``regret_death = (-DEATH_YOPT[name]) - best``. The two tracks
agree for the analytically exact problems and differ where De Ath rounded
(BraninForrester, Hartmann6Log, and notably SixHumpCamelLog, gap ~4.3e-4).

Raw GSobol/Rosenbrock are not in De Ath's Table 2, so they carry no DEATH_YOPT.
"""
from al_benchmark.problems.death import (
    BraninForrester,
    Cosines,
    GoldsteinPriceLog,
    GSobol,
    GSobolLog,
    Hartmann6Log,
    Rosenbrock,
    RosenbrockLog,
    SixHumpCamelLog,
    StyblinskiTangLog,
    WangFreitas,
)

# --problem name -> factory (classes are callable factories).
DEATH_PROBLEMS = {
    "WangFreitas": WangFreitas,
    "BraninForrester": BraninForrester,
    "Cosines": Cosines,
    "GoldsteinPriceLog": GoldsteinPriceLog,
    "SixHumpCamelLog": SixHumpCamelLog,
    "Hartmann6Log": Hartmann6Log,
    "GSobolLog": GSobolLog,
    "RosenbrockLog": RosenbrockLog,
    "StyblinskiTangLog": StyblinskiTangLog,
    "GSobol": GSobol,
    "Rosenbrock": Rosenbrock,
}

# De Ath's coarse code yopt (minimization f*), from results/exp_05_tier1a.json
# f_opt_sources; Branin keyed under its existing name. Do NOT use these for this
# project's regret -- use problem.optimal_value. See module docstring.
DEATH_YOPT = {
    "WangFreitas": -4.000000000000026,
    "BraninForrester": -16.64402,
    "Branin": 0.397887,
    "Cosines": -1.6,
    "GoldsteinPriceLog": 1.0986122886681098,
    "SixHumpCamelLog": -9.54473575988675,
    "Hartmann6Log": -1.20067779,
    "GSobolLog": -6.931471805599453,
    "RosenbrockLog": -0.6931471805599453,
    "StyblinskiTangLog": 2.120864511052842,
}
