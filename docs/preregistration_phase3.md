# Pre-Registration: Optimizer-Controlled Reproduction of De Ath (2021) and Ament (2023)

**Status:** FROZEN pre-registration. To be committed and tagged BEFORE the full
experiment matrix is run. No outcome data exists at the time of this freeze.

**Project:** al-benchmark, Phase 3 (optimizer axis)
**Author:** Jianxiu Hou
**Freeze metadata (fill at commit time):** absolute UTC date = 2026-06-23; git commit hash
of this file = ea25adb23fc6b55836c226bf8f6a83449e7f47ac (this hash identifies the content
freeze — the commit of this file before the hash was inscribed; the amend commit that inscribes
it is byte-identical except for this line, and the tag `phase3-prereg` points at that amend
commit); git tag = `phase3-prereg`; SHA-256 of the analysis script(s) that will
execute this plan = analysis script TBD — to be hashed and recorded as an amendment when first
committed (the Phase 3 H1/H2/H3 analysis script does not yet exist; it will reuse the frozen
`exp_09_prereg_analysis.py` regret/optimal-value helpers, exp_09 being frozen under tag
`exp09-prereg`). The tag and commit hash, recorded before any
production cell is run, are the physical evidence that the plan preceded the data.

---

## 0. Why this pre-registration exists

This document fixes the claims, hypotheses, experimental design, and analysis plan
BEFORE any Phase 3 outcome data is generated. It exists so that the H3 result (below)
cannot be characterized as post-hoc. The analysis plan (H1/H3 estimands, probe protocol)
is specified here, in advance, exactly as it will be reported. This freeze is the
methodological backbone of the contribution.

**On the pilot cell.** A single throwaway pilot cell (EI-via-NSGA-II on GSobol d=10, seed 0)
was run earlier solely to measure per-cell wall-clock/memory and to sanity-check that NSGA-II
finds the same acquisition maximum as L-BFGS. That pilot is NOT a production result: it does
NOT enter any endpoint analysis, any hypothesis test, or any reported BO outcome, and its
script is throwaway/uncommitted. "No outcome data exists" refers to the production matrix that
the hypotheses below are tested on.

The existing gradient-based matrix (exp_08, exp_10) and its pre-registered analysis
(exp_09, tag `exp09-prereg`) are NOT modified by Phase 3. Phase 3 is purely additive:
it adds NSGA-II optimizer arms and compares them, under the analysis plan below, to the
already-frozen gradient-based arms.

---

## 1. Positioning relative to prior work (READ FIRST — this is the honesty core)

This contribution is a **controlled reproduction + mechanism-confirmation study**, NOT a
claim of a novel phenomenon. The conceptual space is partially occupied, and we state
exactly by whom, up front:

- **De Ath et al. (2021), "Greed is Good" (arXiv:1911.12809, ACM TELO).** Established
  greedy/ε-greedy exploitation is competitive in high dimensions, using gradient-free
  NSGA-II (budget 5000d) to optimize all acquisitions. We reproduce this with a modern
  gradient-based BoTorch stack and (separately) with a faithful NSGA-II arm.

- **Rehbach et al. (2020), "Expected Improvement versus Predicted Value" (arXiv:2001.02957,
  GECCO).** Independently confirmed that pure exploitation (predicted value) beats EI on
  most problems of d>=5, using a surrogate-based (gradient-free) optimizer. So
  "greedy beats EI in high-dim" is established under gradient-free optimization by two
  independent groups. Our H1 therefore tests ROBUSTNESS of this finding across the
  optimizer-type axis, not the existence of the phenomenon.

- **Ament et al. (2023), "Unexpected Improvements to EI" (arXiv:2310.20708, NeurIPS).**
  Identifies TWO numerical pathologies in EI — numerically-zero acquisition VALUES and
  vanishing GRADIENTS — both arising under gradient-based optimization; §6 states the LogEI
  advantage is "fundamentally due to concentration of high objective values, not the
  dimensionality itself," and that under poor surrogates "better acquisition values do not
  necessarily lead to better BO performance." H2/H3 build directly on this, testing WHICH of
  the two pathologies drives the LogEI advantage by introducing a gradient-free optimizer
  (immune to vanishing gradients, not to value-underflow).

- **Xie et al. (2024), "Pandora's Box Gittins Index" (arXiv:2406.20062, NeurIPS).** In an
  appendix, disentangles LogEI's benefit over EI into (a) numerical stability and (b)
  gradient/log-scaling — but conducts this disentanglement ENTIRELY WITHIN gradient-based
  optimization. It does NOT separate EI's two distinct pathologies — numerically-zero VALUES
  vs vanishing GRADIENTS — which a gradient-free optimizer can disambiguate (gradient-free
  optimization is immune to vanishing gradients but NOT to value-underflow). **H3 performs
  exactly this value-vs-gradient disambiguation on De Ath's suite, which Xie et al. do not.**
  Because value-underflow can still degrade a gradient-free optimizer, H3's result is not a
  foregone corollary.

- **Georgiou, Mitsos et al. (2025), "Deterministic Global Optimization of the Acquisition
  Function: To Do or Not To Do?" (arXiv:2503.03625).** A controlled comparison of
  acquisition-optimizer choice (deterministic-global MAiNGO vs local L-BFGS-B vs
  multi-start) and its effect on BO, finding "suboptimal optimization of poorly chosen
  acquisition functions can be preferable to their optimal solution." **This occupies the
  general "optimizer choice decouples acquisition quality from BO performance" insight.**
  We differ on a specific, complementary axis: they use LCB and exact-vs-local; we use
  EI/LogEI and gradient-free(NSGA-II)-vs-gradient-based(L-BFGS), directly engaging Ament's
  vanishing-gradient pathology, which their LCB study does not touch.

**Honest contribution statement.** What is NOT done by any of the above, and is therefore
our contribution: (i) a faithful gradient-based reproduction of De Ath's findings; and
(ii) the specific, never-run controlled experiment isolating whether the EI->LogEI
advantage is gradient-based-specific (H3), conducted on De Ath's exact problem suite, with
a pre-registered analysis. We claim correctness and usefulness to the BO practitioner
community, NOT novelty of the underlying mechanism.

---

## 2. Claims

**Claim A (reproduction).** Under a faithful NSGA-II acquisition optimizer matching De Ath
(2021), our pipeline reproduces the qualitative ordering of strategies De Ath reported on
the shared problems (greedy/exploitation competitive, increasingly so with dimension).

**Claim B (optimizer-axis, the core — stated as a hypothesis to be tested, not asserted).**
We test whether the performance advantage of LogEI over EI is specific to gradient-based
acquisition optimization. A gradient-free optimizer (NSGA-II) is immune to EI's
vanishing-GRADIENT pathology but NOT to its numerically-zero-VALUE pathology. A LogEI
advantage that shrinks substantially under NSGA-II is consistent with gradient-vanishing
being the dominant operational limitation (which gradient-free search removes); one that
persists is consistent with value-underflow remaining operationally limiting even under
gradient-free search. The result is reported either way (see H3).

**Claim C (mechanism, descriptive).** The instrumentation (probe: acquisition value
underflow fraction, gradient-norm distribution, optimizer progress; protocol frozen in §5.2)
is reported to interpret whether the EI/LogEI gap under gradient-based optimization co-occurs
with the measured gradient pathology, and how the value-underflow compares across optimizers.
Because optimizer type changes more than gradient availability (search dynamics, budget),
this is an operational interpretation, not a causal proof (see H3).

---

## 3. Hypotheses (FALSIFIABLE — with pre-specified decision criteria)

Endpoint throughout: **precise final regret** = (known optimal value) - (best observed
value) at the end of the T=250 budget, per De Ath. Lower is better. **All H1/H3 analyses are
on LOG precise regret** (the authoritative scale and numerics — epsilon, negative-regret
clipping — are fixed in §5.1); where the hypotheses write L(a,o,p,s) below, L denotes log
precise regret. Comparisons are within-problem, paired by seed and problem.

### H1 — The greedy advantage grows with dimension, under both optimizers.
De Ath (2021) and Rehbach (2020) do not merely claim "greedy is not worse"; their claim is
directional: exploration LOSES value AS DIMENSION INCREASES. We test that directional claim
(a stronger, non-null-confirming form than "Exploit is not significantly worse than EI").
- **Statement:** The advantage of Exploit over EI increases with problem dimension, and this
  dimensional trend holds under BOTH optimizers.
- **Primary test (problem-level, to avoid dimension being a seed-level pseudo-replicate).**
  Dimension is a PROBLEM-level variable; the independent units supporting a dimension trend
  are the problems (~11), not the 330 cells. Therefore, within each optimizer o, aggregate to
  problem level: A_p = mean_s [ L(EI, o, p, s) - L(Exploit, o, p, s) ] (positive A_p = Exploit
  better). Then regress `A_p ~ log(dimension)` across problems (OLS). **Because this
  regression has only ~11 problem-level observations, the reported p-value uses HC3
  heteroskedasticity-robust standard errors, and the Spearman correlation between A_p and
  log(dimension) is reported as a nonparametric sensitivity.** **H1 confirmed iff the
  log(dimension) slope is positive at alpha=0.05 (HC3) under BOTH optimizers** (Exploit's
  advantage over EI grows with dimension regardless of optimizer type), with the Spearman
  sensitivity in the same direction.
- **Secondary (descriptive, NOT the verdict):** the seed-level mixed model
  `L ~ C(acq) * log(dimension)` with random intercept for problem, restricted to
  {EI, Exploit}, reported for completeness; and per-problem paired Wilcoxon Exploit-vs-EI with
  effect sizes, Holm-Bonferroni across problems. A non-significant Wilcoxon is NOT treated as
  confirmation (absence of evidence is not evidence of absence).
- **Falsification:** If the `A_p ~ log(dimension)` slope is absent or negative under the
  gradient-based optimizer (greedy does NOT increasingly beat EI as d grows), the
  De Ath/Rehbach dimensional trend does not transfer to the gradient-based stack — a real,
  reportable outcome.

### H2 — Faithful NSGA-II reproduces De Ath's qualitative ordering (reproduction check).
This is a reproduction check, not an inferential hypothesis. Split to resolve the eps-PF
divergence cleanly:

**H2a (anchor strategies EI, UCB, Exploit only).**
- **Statement:** Under the NSGA-II arm matching De Ath's settings, the rank ordering of
  {EI, UCB, Exploit} by aggregate precise log-regret, on the shared problems, agrees with
  De Ath (2021), AND the high-dimensional greedy-competitive pattern holds.
- **Aggregation (frozen, equal problem weight).** For each shared problem p and strategy a,
  compute M_{p,a} = mean_s L(a, nsga, p, s) (seed mean within problem). For the aggregate
  ordering, compute M_a = mean_p M_{p,a} (equal weight per problem, NOT a pooled mean over all
  cells), consistent with the anti-pseudo-replication logic used throughout.
- **Reproduction criteria (both required):** (1) Spearman rho >= 0.5 between the {EI, UCB,
  Exploit} ordering by M_a and De Ath's reported ordering; AND (2) on shared problems with
  d >= 6, Exploit ranks at or above EI by M_{p,a} in a majority of those problems.
- **Reproduction failure:** if either criterion fails (rho < 0.5, or EI beats Exploit on a
  majority of d>=6 shared problems where De Ath found the reverse), this is reported as a
  FAILED reproduction with investigation of the cause (e.g., the optimizer substitution, the
  Ackley8 fallback, or implementation divergence).

**H2b (eps-PF — descriptive only, NOT part of any pass/fail criterion).**
Our eps-PF reimplements De Ath's NSGA-II eFront as a different procedure (documented in
eps_greedy.py). eps-PF results are reported descriptively for completeness and are EXCLUDED
from the H2a reproduction verdict; we do not claim to reproduce De Ath's eps-PF.

### H3 — Which EI pathology drives the LogEI advantage: vanishing GRADIENTS or vanishing VALUES? **(CORE — THE BET)**

**Why this is the sharp question, not merely "is the advantage gradient-specific."**
Ament (2023) identifies TWO numerical pathologies in EI: (1) acquisition VALUES that are
numerically *exactly zero* over large regions, and (2) vanishing GRADIENTS. Gradient-based
L-BFGS is hurt by BOTH (zero value => zero gradient => stuck). A gradient-free optimizer
(NSGA-II) does not use gradients, so it is immune to (2) — but NOT automatically immune to
(1): if EI's value is numerically flat-zero over most of the domain, NSGA-II optimizes a
flat function and also degrades toward random selection. Therefore "EI ~= LogEI under
NSGA-II" is NOT a foregone conclusion, and H3 is not a trivial corollary. H3 isolates which
pathology is operative on De Ath's suite:
  - If EI recovers to ~= LogEI under NSGA-II => this is consistent with vanishing GRADIENTS
    (2) being the dominant operational limitation, which gradient-free optimization removes.
  - If EI stays worse than LogEI under NSGA-II => this is consistent with vanishing VALUES
    (1) remaining operationally limiting, which gradient-free optimization does NOT remove.
  (Pilot motivation, not a result: on GSobol d=10 s0, NSGA-II found EI = 1.55e7, the same
  maximum L-BFGS found — single-cell hint that value-underflow did not cripple NSGA-II
  there.) This value-vs-gradient decomposition via a gradient-free optimizer is NOT done by
  Xie et al. (2024), whose stability-vs-log-scaling disentanglement stays entirely within
  gradient-based optimization.

- **Statement:** The EI-vs-LogEI advantage (on log precise-regret) is significantly smaller
  under the NSGA-II (gradient-free) optimizer than under the gradient-based optimizer.
- **Primary estimand (explicit seed-level paired interaction contrast).** Let
  L(a, o, p, s) = log precise final regret for acquisition a in {EI, LogEI}, optimizer o in
  {grad, nsga}, problem p, seed s (all four share the same initial design per (p, s), per the
  §4.0 pairing gate). Define the per-(problem, seed) contrast:
  ```
  D_{p,s} = [ L(EI, grad, p, s) - L(LogEI, grad, p, s) ]
          - [ L(EI, nsga, p, s) - L(LogEI, nsga, p, s) ].
  ```
  D > 0 means the EI->LogEI improvement is LARGER under gradient-based than under NSGA-II
  (i.e., the LogEI advantage shrinks under the gradient-free optimizer). Sign convention fixed
  here (EI minus LogEI, grad minus nsga); H3 direction is D > 0.
- **PRIMARY test (problem-level, maximally robust to pseudo-replication).** Aggregate each
  problem to D_p = mean_s D_{p,s} (equal weight per problem), then test D_p > 0 across the
  ~11 problems with a one-sided Wilcoxon signed-rank test against zero; report a paired
  t-test and sign test as sensitivity. The independent unit is the problem, consistent with
  H1 and with the Phase-1 anti-pseudo-replication discipline. **H3 confirmed iff this
  problem-level test is significant at alpha=0.05 in the direction D > 0.**
- **SECONDARY test (seed-level precision / power source — NOT a tie-breaker chosen post-hoc,
  specified here).** Fit `D_{p,s} ~ 1` with a random intercept for problem (statsmodels
  MixedLM, REML). This uses the within-problem seed replication for precision. **It is
  reported alongside the primary regardless of outcome.** Pre-specified interpretation of the
  two together (this is the explicit Phase-1 lesson, stated in advance so it is not a
  post-hoc rescue): the ~11-problem Wilcoxon has LOW POWER; if the problem-level primary is
  non-significant but the seed-level model is significant in the predicted direction, this is
  reported transparently as "directionally supported but underpowered at the problem level"
  — exactly the omnibus-underpowered / structured-model-detects pattern documented in Phase 1
  (N=6 Friedman non-significant while the mixed-effects model detected the effect). We do NOT
  claim H3 confirmed on the seed-level model alone; we report both and let the gap between
  them be visible.
- **Attribution (Claim C, descriptive only — see §5.2 for the frozen probe protocol AND its
  cross-optimizer comparability caveat):** the probe's per-(acq, optimizer, dimension)
  value-underflow fraction and gradient-norm distribution are used to interpret the result: a
  confirmed H3 is **consistent with vanishing GRADIENTS being the dominant operational
  limitation** (underflow broadly similar across optimizers, gradients pathological only under
  gradient-based); a rejected H3 is **consistent with value-underflow remaining operationally
  limiting even under gradient-free search** (underflow high under both). Per §5.2, the
  gradient-based probe is the existing K=64/own-acqf probe and the NSGA-II probe is K=4096/raw-EI,
  so any cross-optimizer probe statement is a caveated DESCRIPTIVE comparison, never a tested
  claim. Because optimizer type also changes search dynamics (global vs local) and evaluation
  budget, H3 is interpreted as an OPERATIONAL disambiguation, NOT a proof that gradients are
  the sole causal factor.
- **Falsification (REAL and must be honored):** if the problem-level primary is not
  significant in the direction D > 0 (and the seed-level secondary does not show a clear
  predicted-direction effect either), H3 is REJECTED: the LogEI advantage is NOT smaller
  under gradient-free optimization, consistent with value-underflow rather than
  gradient-vanishing being operationally dominant. **This outcome is reported as the headline
  if it occurs**, correcting the corollary implied by Xie et al. (2024) that LogEI's benefit
  is purely about gradient-optimization mechanics.

**Pre-commitment:** H3 is reported in whichever direction it resolves; the probe attribution
is reported regardless. Framing adapts to the result, never the reverse. Both resolutions are
useful, reportable contributions.

---

## 4. Experimental design (FROZEN)

### 4.0 Pre-run verification gate (MUST pass before launching the matrix)
H1/H3 depend on PAIRING NSGA-II cells with already-existing gradient-based cells on the
SAME (problem, seed, initial design). Before any NSGA-II production run, CC must verify and
report:
1. **Existence:** the gradient-based matrix already contains EI, LogEI, UCB, Exploit, and
   eps-PF cells for every (problem, seed) in scope (all listed problems x 30 seeds). If
   LogEI-gradient-based cells are missing for any in-scope cell, H3 cannot be paired there —
   either those gradient-based cells are generated first under the identical harness, or the
   affected problems are dropped from H3 (decision recorded as an amendment).
2. **Pairing key:** confirm the seed->design mapping (seed s => De Ath design run s+1) is
   byte-identical between the existing gradient-based arms and the new NSGA-II arms, so the
   initial design is shared per (problem, seed). Verify on at least one cell that the
   injected initial X matches between the two optimizers' runs (the sanity gate carried over
   from the gradient-based matrix's design-injection check).
3. **Endpoint availability:** confirm precise final regret is recoverable for every existing
   gradient-based cell using the SAME merged optimal-value table the NSGA-II analysis will use.
   **The merged table is { **_problem_optimal_values(), **_ext_optimal_values() } (exp_09).**
   `_ext_optimal_values()` alone covers only {GSobol, Rosenbrock, Ackley8} (i.e., only Ackley8
   of the 11 in-scope problems); the other 10 in-scope problems' optima live in
   `_problem_optimal_values()`. The analysis MUST use the merged dict so all 11 problems
   resolve; calling `_ext_optimal_values` alone would KeyError on 10/11. All 11 optima are
   exact/known constants (no stochastic estimate); BraninForrester's optimum
   (16.64402157084319) is a refined numerical optimum and is the most likely to trigger the
   §5.1 negative-regret clip, which is handled there.
This gate is a hard precondition; the ~2-4 day run is not launched until it passes.

### 4.1 Strategy arms
NSGA-II versions of: **EI, LogEI, UCB, Exploit, eps-PF** (5 arms). Each is compared to its
already-frozen gradient-based counterpart (exp_08/exp_10). Random and eps-RS are not given
new NSGA-II arms (Random has no acquisition optimizer; eps-RS's exploit branch is covered
by Exploit). LogEI-NSGA2 is included specifically for H3.

### 4.2 Problems
All De Ath problems available in the existing matrix, spanning dimensions d in {1, 2, 6, 8,
10}: WangFreitas (d1); Branin, BraninForrester, Cosines, logGoldsteinPrice, logSixHumpCamel
(d2); logHartman6 (d6); Ackley8 (d8, Sobol-fallback initial design — see below);
logGSobol, logRosenbrock, logStyblinskiTang (d10). This spread of dimensions is what powers
the H1 dimension-trend test (H1 aggregates to problem level and regresses on log(dimension);
see §3 H1).

**Name map (pre-reg / De Ath stem ↔ result-file & optimal-value class key — the analysis
script keys on the result-file names):**
```
WangFreitas        -> WangFreitas        (d1, exp_08)
Branin             -> Branin             (d2, exp_08)
BraninForrester    -> BraninForrester    (d2, exp_08)
Cosines            -> Cosines            (d2, exp_08)
logGoldsteinPrice  -> GoldsteinPriceLog  (d2, exp_08)
logSixHumpCamel    -> SixHumpCamelLog    (d2, exp_08)
logHartman6        -> Hartmann6Log       (d6, exp_08)
Ackley8            -> Ackley8            (d8, exp_10, Sobol fallback)
logGSobol          -> GSobolLog          (d10, exp_08)
logRosenbrock      -> RosenbrockLog      (d10, exp_08)
logStyblinskiTang  -> StyblinskiTangLog  (d10, exp_08)
```
NOTE: the in-scope d10 problems are the LOG variants from exp_08 (GSobolLog etc.), NOT
exp_10's separate raw GSobol/Rosenbrock trio (which are out of scope).
**Disclosed limitation:** Ackley8 has no De Ath initial-design file and uses a Sobol
fallback; it is included for the dimension trend but flagged, and H2 (De Ath reproduction)
does not rely on it.

### 4.3 Seeds
30 seeds per cell, seed s using De Ath design run s+1 (matching the existing matrix's
seed->design mapping, so NSGA-II and gradient-based arms share initial designs per seed and
are paired).

### 4.4 NSGA-II settings (FROZEN — matching De Ath 2021 Section 4)
- Library: pymoo 0.6.1.6 (to be added to requirements.txt / environment.yml).
- Population size: 100*d.
- Generations: 50 (=> budget 100*d*50 = 5000d acquisition evaluations).
- Crossover: SBX, prob 0.8, eta (distribution index) 20.
- Mutation: polynomial (PM), prob 1/d, eta 20.
- Single-objective acquisitions (EI, LogEI, UCB, Exploit): pymoo GA (NSGA-II degenerates to
  GA for single objective).
- Bi-objective eps-PF (mu, sigma) front: pymoo NSGA2; select randomly from the resulting
  Pareto set, matching De Ath's eps-PF random-from-front rule.
- pymoo evaluator kept serial (no pymoo-internal parallelism); one matrix cell per
  ProcessPoolExecutor worker, matching the gradient-based arms' per-cell threading.

### 4.4a Stochastic-source seeding (FROZEN — required for "optimizer-controlled reproduction")
NSGA-II has stochastic population initialization, crossover, and mutation; left unseeded these
are researcher degrees of freedom that would undermine the reproduction claim. Therefore, for
each cell, ALL stochastic sources are seeded deterministically from the tuple
(problem, acquisition, optimizer, seed): pymoo population initialization, pymoo
crossover/mutation RNG, PyTorch, NumPy, and Python `random`. **The pymoo GA seed is held
CONSTANT across all 250 BO iterations within a cell** (one frozen seed per cell, derived from
the 4-tuple; not offset by iteration index). Re-running a completed cell with the same code
and seed must reproduce the candidate sequence up to documented hardware nondeterminism
(e.g., BLAS float reassociation).
**Consistency with the existing gradient-based arms (so §4.0 pairing holds).** The initial
design for each (problem, seed) is the SAME De Ath design (run s+1) already used by the
gradient-based arms; the NSGA-II RNG seeding governs ONLY the acquisition-optimizer's internal
randomness and MUST NOT alter the injected initial design. The §4.0 gate (c) — injected
initial X byte-identical across the two optimizers for at least one verified cell — is the
check that this separation held.

### 4.5 BO loop
T=250 total budget; initial design = De Ath's per-problem LHS design (M=2d) via the
existing seed->design mapping; GP refit (fit_gpytorch_mll, hyperparameters re-optimized)
every iteration, identical to the gradient-based harness. Matern-5/2 kernel, ARD,
input normalized to unit cube, output standardized — unchanged from existing pipeline.

### 4.6 Batch / reliability harness
Reuse the existing chunk-40 / watchdog-900s(FIRST_COMPLETED) / atomic-write /
skip-existing-resume harness unchanged. 2 workers.

---

## 5. Analysis plan (FROZEN — exactly as it will be reported)

### 5.1 Shared numerics (frozen) — model specifications live in §3 H1/H3
- **Endpoint and scale.** Per-cell endpoint: precise final regret = (known optimal value) -
  (best observed value). ALL primary and fallback analyses (H1, H3) are on **log precise
  regret**; no raw-regret contrast is used anywhere. A fixed epsilon = 1e-12 is added before
  the log.
- **Negative-regret handling (frozen).** If a cell's precise regret is < 0 due to numerical
  tolerance (best observed slightly exceeds the tabulated optimum), set
  regret_clipped = max(regret, 0) before adding epsilon, and report the frequency of clipping.
- **Estimation engine.** statsmodels `MixedLM`, REML, for the random-intercept models named
  in H1/H3; OLS for the H1 problem-level `A_p ~ log(dimension)` regression. Significance at
  alpha=0.05; CIs reported and required to exclude zero in the pre-specified direction.
- **Reference levels / sign conventions (frozen).** acq reference = EI; optimizer reference =
  grad. The H3 contrast D is defined (in §3 H3) as (EI - LogEI) under grad minus (EI - LogEI)
  under nsga, with H3 direction D > 0. The H1 quantity A_p is (EI - Exploit), with H1
  direction = positive log(dimension) slope. These are fixed so coefficient signs are
  unambiguous regardless of dummy-coding.
- **The H1 and H3 primary tests and their secondary models are specified in full in §3**
  (H3 PRIMARY: one-sided Wilcoxon on problem-level D_p; H3 SECONDARY/precision:
  D_{p,s} ~ 1 + (1|problem); H1 PRIMARY: A_p ~ log(dimension) OLS with HC3 per optimizer,
  Spearman sensitivity). They are not restated here to avoid divergence; §3 is authoritative.

### 5.2 Mechanism probe protocol (Claim C) — FROZEN, descriptive only
The probe is specified in advance so attribution cannot be chosen post-hoc. At a fixed
cadence (every BO iteration), using the GP fitted at that iteration:
- **Probe set.** A fixed Sobol set of K = 4096 points over the (normalized) domain, drawn
  with a fixed probe seed (distinct from the BO seed), regenerated identically across all
  cells of the same problem.
- **Value-underflow fraction.** Underflow is a property of the RAW EI value (LogEI does not
  itself underflow to zero, so LogEI == -inf is not the operative definition). For EVERY arm
  (including LogEI arms), value-underflow is measured by the raw EI value induced by the SAME
  posterior and incumbent at that iteration: evaluate raw EI on the probe set in float64 and
  record the fraction of probe points with raw EI exactly == 0.0. Reported per (acq,
  optimizer, dimension, iteration), aggregated to per-(acq, optimizer, dimension) medians.
  (This makes the EI and LogEI arms comparable on one underflow definition: how much of the
  domain the underlying improvement signal has collapsed to numerical zero.)
- **Gradient health.** Computed on EACH arm's OWN acquisition surface: for EI arms,
  ||grad_x EI||_2; for LogEI arms, ||grad_x LogEI||_2 — via autograd, on the same probe
  points; record the fraction with norm < 1e-12 and the median log10 norm. (Gradient health
  is a property of the acquisition surface the optimizer would traverse; under NSGA-II the
  optimizer does not USE these gradients, which is exactly the contrast H3 exploits. We do
  NOT compare EI-surface gradients to LogEI-surface gradients as if interchangeable; each is
  reported on its own surface and the EI-vs-LogEI gradient difference is the phenomenon, not
  an artifact of mismatched surfaces.)
- **Optimizer progress.** For each acquisition-optimization call, record the best acquisition
  value at initialization vs at the returned candidate (progress = final - initial on the raw
  acquisition scale) and the rank of the returned candidate among the K probe points.
- **Role.** Descriptive ONLY; no significance test is claimed on the probe. It is used solely
  to interpret the H3 verdict per the "consistent with ..." language in §3 H3 (operational
  disambiguation, not causal proof).
- **Cross-optimizer comparability caveat (F3 — frozen decision: option a, the cheap honest
  one).** The §5.2 probe above (K=4096, raw-EI-for-every-arm) is computed for the NSGA-II
  arms. The EXISTING gradient-based arms already carry a different probe: K=64 (probe.py),
  logging each arm's OWN acquisition value (so for LogEI it records the LogEI value, not raw
  EI). We DO NOT re-run the gradient-based matrix to harmonize this. Therefore any
  cross-optimizer probe comparison in §3 H3's attribution (e.g., "underflow comparable across
  optimizers") is explicitly flagged as comparing a K=4096 raw-EI probe (nsga) against a K=64,
  own-acqf probe (grad) — a descriptive, caveated comparison, NEVER a tested claim. The probe
  is interpretive context for the regret-based H3 verdict, which does not depend on the probe
  at all. (Option b — passively re-running gradient-based EI+LogEI with the K=4096 probe,
  ~+0.27 day — is available but declined as unnecessary given the descriptive-only role.)

### 5.3 Multiplicity & reporting
- All p-values Holm-Bonferroni corrected within each hypothesis family.
- Effect sizes and CIs reported alongside every p-value.
- **Pseudo-replication is explicitly guarded against** (the central Phase 1 lesson, applied
  here): H3's PRIMARY test aggregates the paired seed-level contrast D_{p,s} to a
  problem-level D_p and tests across the ~11 problems (the paired seed-level contrast removes
  cell-level pairing concerns; the seed-level mixed model D_{p,s} ~ 1 + (1|problem) is
  reported as a precision/power model, with the gap between the two made visible per §3 H3).
  H1's primary test likewise aggregates seeds to a problem-level A_p before regressing on
  log(dimension), so the ~11 problems — not the 330 cells — are the independent units. Seeds
  are never treated as independent blocks in any primary verdict.

---

## 6. Outcome-independent commitments

1. H3 is reported in whichever direction it resolves; the paper title/abstract adapt to the
   result, not vice versa.
2. Georgiou (2503.03625) and Xie/Pandora's Box (2406.20062) are cited prominently in the
   intro and related work, with our delta stated explicitly. We do not claim the general
   "optimizer-decoupling" insight as novel.
3. The eps-PF reimplementation divergence from De Ath's exact NSGA-II eFront (already
   documented in eps_greedy.py) is disclosed in the paper as a faithfulness limitation.
4. The Ackley8 Sobol-fallback is disclosed.
5. Any deviation from this pre-registration discovered necessary during execution is
   recorded as a dated amendment appended below, never by silent edit of the above.

---

## 7. Amendments (append-only)

### Amendment 1 (post-data) — analysis script hash + execution record
Date (UTC): 2026-06-25
This amendment fills the "analysis script TBD" placeholder in the freeze metadata and records
the confirmatory run. It does NOT alter any pre-registered hypothesis, test, criterion, or the
§3/§5 analysis plan, all of which were frozen before the production matrix existed.

- Analysis script: experiments/exp_11_analysis.py
- Script SHA-256: 871de0a7ccb7c5cef65d21272bd4c2526bc3283472b1f8aaf1775bf7178586cf
- Script commit: 555a7d21eb48203bdae5fb4fd9a6923ba1b8a663 (author Jianxiu Hou, no AI trailer)
- Production matrix: results/exp_11_matrix/, 1650 NSGA-II cells (5 arms x 11 problems x
  30 seeds), backup archive SHA-256 d2832fabf382772fce3b2c79ccbd7e786b9f5a421a50f2253376868393ad6fa1
- Paired against frozen gradient-based matrices exp_08 + exp_10 (Ackley8); 3300 cells total;
  0/3300 negative-regret clips.

Pre-registered verdicts (reported in whichever direction they resolved, per §6):
- H1 (greedy advantage grows with dimension, both optimizers): NOT CONFIRMED. Directional
  trend present (OLS slope on A_p ~ log(dim) positive under both optimizers: grad +1.81,
  nsga +1.52; Spearman positive under both, significant under grad rho=0.71 p=0.014) but the
  frozen HC3-OLS criterion (p<0.05 under BOTH) was not met (grad p=0.072, nsga p=0.128). The
  ~11-problem, 5-distinct-dimension regression is underpowered, as anticipated in §3 H1.
- H2a (faithful NSGA-II reproduction of De Ath ordering): FAILED. Anchor ordering
  UCB < EI < Exploit (Exploit worst), Spearman vs De Ath reference = -0.50 (< 0.5 threshold);
  Exploit >= EI on 2/4 d>=6 shared problems. Descriptive note: the gradient-based stack shows
  the IDENTICAL ordering, so the failed reproduction is not attributable to the NSGA-II
  optimizer substitution.
- H3 (LogEI advantage shrinks under NSGA-II; THE CORE): REJECTED. Problem-level one-sided
  Wilcoxon on D_p > 0 non-significant (W=29, p=0.65); seed-level secondary MixedLM also not
  in-direction significant (one-sided p=0.46); not upgraded to confirmed on the secondary.
  Descriptive components: the EI-vs-LogEI gap is itself small under both optimizers (mean
  LogEI advantage grad -0.084, nsga -0.113), so H3's premise of a clear gradient-based LogEI
  advantage is only weakly present (and only on the d=10 subset); the binary
  gradient-vanishing-vs-value-underflow disambiguation does not cleanly resolve, and the
  attribution probe shows EI value-underflow ~0 at d>=6, reported faithfully per §5.2
  (descriptive only, with the K=4096-vs-K=64 cross-optimizer caveat).
