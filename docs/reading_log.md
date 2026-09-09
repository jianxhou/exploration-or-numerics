# Reading Log

A summary of papers read for this project. Each entry includes:
- One-paragraph summary in my own words
- 3 takeaways relevant to my project
- 2-3 specific quotes or claims to potentially cite

## Planned Reading (in order)

1. Frazier (2018) — Tutorial on Bayesian Optimization [arXiv:1807.02811]
2. De Ath et al. (2019) — Greed is Good [arXiv:1911.12809]
3. Bliek et al. (2021) — EXPObench [arXiv:2106.04618]
4. Shahriari et al. (2016) — Taking the Human Out of the Loop
5. Demšar (2006) — Statistical Comparisons of Classifiers

---

## 1. Frazier (2018) — A Tutorial on Bayesian Optimization

**arXiv**: 1807.02811
**Sections read**: 1 (Introduction), 2 (Gaussian Process Regression), 3 (Acquisition Functions)
**Status**: Core concepts understood; will revisit Section 4 (algorithm) and Section 5 (extensions) when needed for Phase 2 work on qEI.

### Summary

Frazier provides a self-contained tutorial on Bayesian optimization (BO) for expensive black-box functions. The framework has two ingredients: a Gaussian Process (GP) surrogate that models the unknown objective with calibrated uncertainty, and an acquisition function that scores candidate points by their value for the next evaluation. The tutorial derives closed-form expressions for the GP posterior under Gaussian assumptions, then walks through the three classical acquisition functions — Expected Improvement, Probability of Improvement, and Upper Confidence Bound — showing how each balances exploration of uncertain regions against exploitation of currently promising areas. Frazier emphasizes BO is most useful when function evaluations are expensive (minutes to hours per call), dimensionality is modest (typically d ≤ 20), and gradients are unavailable.

### Section 2: Gaussian Process Regression

- A GP is fully specified by a mean function μ(x) and a covariance function (kernel) k(x, x'). The kernel encodes the smoothness assumption — nearby points have correlated outputs.
- Frazier recommends the Matérn 5/2 kernel as the practical default in BO rather than RBF, because it produces twice-differentiable sample paths that match real engineering responses better than the infinitely-smooth RBF.
- The GP posterior after observing n points has closed-form mean and variance, but requires inverting an n × n kernel matrix, which is O(n³). This is why BO is typically restricted to n < 1000.
- Lengthscale hyperparameters in the kernel control how quickly correlations decay across input dimensions. With ARD (one lengthscale per dimension), the GP learns which inputs matter most — this is what BoTorch's default `SingleTaskGP` does, and what the two-element lengthscale tensor in my GP smoke test was showing.

### Section 3: Acquisition Functions

- All three classical acquisition functions have closed forms when the surrogate is a GP, which is why the BO inner loop is fast despite the GP fitting cost.
- **Expected Improvement (EI)**: α(x) = E[max(f(x) − f*, 0)], where f* is the current best. Closed form involves the Gaussian CDF Φ and PDF φ. Balances exploration (high σ → larger expected upside) and exploitation (high μ relative to f* → larger expected improvement).
- **Probability of Improvement (PI)**: α(x) = P(f(x) > f*). Simpler than EI but tends to be overly exploitative — it doesn't differentiate between "barely beats f*" and "dramatically beats f*".
- **Upper Confidence Bound (UCB)**: α(x) = μ(x) + β^(1/2) · σ(x). The β parameter directly controls the exploration-exploitation tradeoff. GP-UCB has theoretical no-regret guarantees (Srinivas et al. 2010), unlike EI.
- Acquisition optimization itself is non-trivial: the acquisition surface is multimodal, so BO implementations use multi-start L-BFGS-B (this is what BoTorch's `optimize_acqf` does internally with `num_restarts` and `raw_samples`).

### Three takeaways relevant to my project

1. **EI's closed-form derivation explains its exploration phase**. EI is non-zero even when μ(x) < f*, as long as σ(x) is large enough — the formula gives positive weight to the tail of the posterior that exceeds f*. This explains why my Day 1 notebook's EI on Branin stayed at best = -1.42 for 11 iterations before breaking through: it was systematically probing high-σ regions whose posterior mean was unattractive but whose tails justified the cost. The "exploration phase" is not a quirk — it's mandated by the math.

2. **Kernel choice is not innocuous**. Frazier recommends Matérn 5/2 as the BO default, but BoTorch's `SingleTaskGP` defaults to RBF (my smoke test confirmed this). This is a methodological design choice I should note explicitly in my report, and possibly run an ablation on (RBF vs Matérn 5/2) if time allows in Phase 2.

3. **Closed-form acquisitions justify sequential BO, but qEI loses this advantage**. Section 3 ends by noting batch acquisitions like qEI require Monte Carlo estimation because no closed form exists. This is exactly why qEI is computationally heavier than EI — a structural reason supporting my decision to frame qEI as Phase 2 rather than Phase 1.

### Possible citations for my paper

- "BO is most useful when function evaluations are expensive, the dimensionality is modest, and gradients are unavailable" (Frazier 2018, Section 1) — to motivate why I focus on d ≤ 10 engineering problems.
- "The Matérn 5/2 kernel produces sample paths that are twice-differentiable, a property that matches many physical responses better than the infinitely-smooth RBF kernel" (Section 2) — for an ablation discussion or design choice justification.
- "Expected Improvement has no theoretical finite-time regret guarantee, unlike GP-UCB which has logarithmic-regret bounds under appropriate β scaling" (Section 3) — when discussing EI's instability vs UCB's worst-case behavior, especially in light of my Day 2 experiment where UCB had one catastrophic seed (regret 1.55) but better median performance overall.

### Open questions / to revisit

- [ ] What's the exact Matérn 5/2 formula and how do its hyperparameters interact with BO inner-loop optimization?
- [ ] How does GP-UCB's theoretical β schedule (β_t = c · log(t)) compare with the fixed β = 2 that BoTorch uses by default? Is this a practical compromise worth discussing in my report?
- [ ] Frazier's discussion is for noise-free observations. For Phase 2's UCI Concrete problem (real measured data), what changes in the EI/UCB formulas under heteroscedastic noise?

---

## De Ath, G., Everson, R. M., Rahat, A. A. M., & Fieldsend, J. E. (2019).
### "Greed is Good? On the Choice of Exploitation Versus Exploration in Bayesian Optimization"
### arXiv:1911.12809

**Read**: Abstract, Introduction, Section 2 (acquisition function as Pareto problem), Section 3 (ε-greedy methods), skim of experiments.

### Core thesis

The paper challenges the dominant narrative that BO requires carefully designed acquisition functions to balance exploration and exploitation. The authors argue that on many practical BO problems, a simple **ε-greedy strategy** — selecting the GP posterior mean maximizer 90% of the time and exploring randomly 10% of the time — performs at least as well as, and often better than, EI and UCB. The advantage is most pronounced in higher dimensions and with limited budgets.

### Three main contributions

1. **Pareto-front reinterpretation of acquisition functions.**
   They cast BO as a bi-objective problem: maximize the GP posterior mean (exploitation) and maximize the GP posterior variance (exploration). They prove EI and UCB always select points on the exploration-exploitation Pareto front; PI does not always; weighted EI only does under certain weight ranges. This is the conceptual hook of the paper.

2. **Two ε-greedy methods.**
   - **ε-PF**: with probability 1-ε, pick the greedy (mean-maximizing) point; otherwise sample randomly from the Pareto front.
   - **ε-RS**: with probability 1-ε, pick the greedy point; otherwise sample uniformly from the entire search space.
   They use ε=0.1 throughout — a deliberate choice, not optimized per-problem.

3. **Large-scale empirical study.**
   - 10 synthetic benchmarks, 1-10D
   - 2 real-world tasks (CFD, robot active learning)
   - Result: ε-greedy is at least competitive with EI/UCB, and frequently better in higher dimensions.

### Authors' explanation for why ε-greedy works

In higher dimensions, the GP surrogate is inherently inaccurate. So even a purely greedy strategy (which trusts the model fully) ends up doing some implicit exploration because the model's "best guess" is often wrong. Adding 10% random exploration on top of this provides enough additional coverage without the over-cautious behavior that EI/UCB exhibit.

### Why this matters for my project

**Direct connection to my Unit 3.4 finding**: I observed EI vs UCB with Nemenyi p=0.89 (statistically indistinguishable) and EI/UCB only marginally better than Random on Ackley-10D. De Ath et al. give a **principled explanation** for both:
- EI ≈ UCB because both are Pareto-front methods with similar exploration weights
- BO collapses toward Random in high dimensions because GP uncertainty estimates degrade

**Implication for my paper**: I should consider adding ε-greedy as a fourth acquisition strategy in Phase 2. It's cheap to implement (basically `if uniform() < epsilon: random else: max posterior mean`), and would strengthen the methodological story.

### Critical observations / limitations

- **ε=0.1 is empirically chosen, not theoretically justified.** No convergence guarantees. Different problems likely have different optimal ε.
- The paper does not directly compare against Thompson Sampling, which is another "implicit exploration" baseline that could be relevant.
- Their CFD task is OpenFOAM-based; not the same as OpenAeroStruct (my planned Phase 2 task), but methodologically similar (expensive black-box function).
- ε-PF requires solving the bi-objective Pareto front, which is non-trivial in high dimensions. ε-RS is much simpler. The paper shows both work; ε-RS is probably what I'd implement first.

### Quotes worth remembering (paraphrased — direct quotes not copied)

- The Pareto-front reinterpretation: any "reasonable" acquisition function should be Pareto-optimal in the (mean, variance) plane.
- The greedy intuition: trusting the model fully isn't necessarily bad if the model is bad in a "uniform" way across the search space.

### Open questions for further investigation

1. **Does the ε-greedy advantage hold for engineering surrogate models (e.g., OpenAeroStruct airfoil design)?** De Ath et al. test on CFD but not on aerodynamic design. This is exactly the gap my Phase 2 could fill.
2. **How does the optimal ε scale with dimensionality?** Their ε=0.1 is fixed; would ε=0.05 work in 2D and ε=0.2 in 10D? An ε-schedule could be its own ablation.
3. **Random Forest vs GP**: if the model is RF (less smooth uncertainty estimates), does the ε-greedy story still hold? This connects to my Phase 2 plan.
4. **Connection to Thompson Sampling**: TS samples from the posterior; ε-greedy is a kind of degenerate TS where most samples are at the mean. Is there a unifying framework?

### Action items

- [ ] If Phase 2 time permits, add `EpsilonGreedy(epsilon=0.1)` as a fourth acquisition strategy. Easy to implement.
- [ ] In my paper Discussion, cite De Ath et al. 2019 as supporting evidence for the "EI ≈ UCB" observation.
- [ ] Look up the references they cite for "GP uncertainty is poorly calibrated in high D" — this is methodologically important for my Ackley result.

---

## Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016).
### "Taking the Human Out of the Loop: A Review of Bayesian Optimization"
### Proceedings of the IEEE, 104(1)

**Read**: Acquisition functions (PI/EI/UCB/Thompson/ES), surrogate models (GP-focused, RF), high-dimensional BO discussion.

### Core thesis

A broad survey framing BO as a *system* of two interacting components rather than a single algorithm: a **surrogate model** that encodes our beliefs about the unknown function and its uncertainty, and an **acquisition function** that uses those beliefs to decide where to sample next. The paper's most practically important claim is that the choice of surrogate model often matters more than the fine-grained choice of acquisition function — get the model, kernel, noise, and dimensional structure right first, then worry about EI vs UCB vs Thompson. BO is positioned as best-suited to black-box, expensive, gradient-free, non-convex/multimodal problems of modest dimensionality.

### Acquisition functions — the comparison

The Bayes-optimal sequential policy is generally intractable, so BO relies on **myopic heuristics** that each pick the locally most valuable point. All trade off exploitation (high posterior mean) against exploration (high posterior variance):

- **PI (Probability of Improvement)** — probability of exceeding a target τ. Simple but ignores *how much* it exceeds, so it turns greedy/over-exploitative once τ is set to the current best (its picks sit close to the incumbent).
- **EI (Expected Improvement)** — expected *magnitude* of improvement over τ, not just the probability. More robust than PI: visits promising regions but also weights high-uncertainty regions for their upside. Setting τ to the current best is reasonable for EI even though it's too greedy for PI.
- **UCB / GP-UCB** — optimism under uncertainty: α(x) = μ_n(x) + β_n·σ_n(x). β tunes exploration. Has theoretical regret guarantees, but performance depends on the β schedule.
- **Thompson Sampling** — sample a function from the posterior and optimize it. Conceptually simple, naturally balances explore/exploit, good for batch/delayed feedback; but continuous-GP sampling needs approximation (e.g. spectral) and the paper notes it can over-explore in high dimensions.
- **ES / PES (Entropy Search)** — information-based: maximize the information gained about the *location of the optimizer* x*, rather than the value at x. Most directly aligned with "find the optimum," but computationally heavy and approximation-dependent.

The paper's stance: acquisition functions matter, but don't fetishize any single one — surrogate adequacy usually dominates.

### Surrogate models

GP is the standard because it natively provides the posterior mean and *principled* uncertainty that acquisition functions need. A GP is determined by a (usually constant) prior mean and a kernel controlling smoothness/length-scales. Key points:

- **Kernel choice**: Matérn (ν controls smoothness) vs squared-exponential (extremely smooth). **ARD kernels** give each dimension its own length-scale, which doubles as a relevance signal for which inputs matter.
- **GP strength**: even far from data it falls back to the prior with *large* uncertainty — better for exploration than an over-confident model.
- **GP weakness**: exact inference is O(n³) (covariance inversion / Cholesky), re-incurred when hyperparameters update; sparse GPs (SPGP, SSGP) scale better but introduce uncertainty artifacts (variance pinching near pseudo-inputs; spurious oscillation far from data).
- **Random forest** (SMAC): scales well, handles categorical/conditional variables, but uncertainty is less principled, can be confidently wrong far from data, and the response surface is non-differentiable (no gradient-based acquisition optimization).

### High-dimensional BO

The survey is cautious: BO suits *modest* dimensionality; high-D is the bottleneck because covering the space needs exponentially many samples (curse of dimensionality). Three coping strategies discussed:
1. **Sparsity / variable selection** — identify the few relevant dimensions (e.g. Chen et al.'s sequential likelihood-ratio tests), but assumes axis-aligned relevance.
2. **RF implicit feature selection** — SMAC's trees subsample features; strong on some high-D algorithm-configuration tasks, weak theory.
3. **Low effective dimensionality** — many high-D problems have few truly-active dimensions; random embeddings (REMBO-style) optimize in a low-D subspace that still contains the optimum.

### Three takeaways relevant to my project

1. **"Surrogate > acquisition" is the theoretical backing for today's normalization fix.** My Borehole/Piston results collapsed BO toward Random *not* because EI/UCB were inadequate, but because the un-normalized GP couldn't learn sensible per-dimension length-scales across inputs spanning ~10⁵. Adding input Normalize + output Standardize recovered 1-2 orders of magnitude. This is exactly Shahriari's point: I had a surrogate problem masquerading as an acquisition problem. Worth foregrounding in the report's Methods/Discussion.

2. **The survey explains my EI ≈ UCB result (Nemenyi p=0.31, N=60).** Both EI and UCB are well-behaved explore/exploit heuristics on a GP; the paper's framing ("don't over-index on the specific acquisition") predicts they should be hard to separate on smooth benchmarks — which is precisely what my 6-problem Friedman/Nemenyi shows. Combined with De Ath (2019)'s Pareto-front argument, I now have two independent literature supports for the same empirical finding.

3. **The high-D section frames my Ackley-10D degradation correctly.** My EI-over-Random advantage shrinks sharply in 10D. Shahriari attributes this to coverage cost + GP uncertainty quality in high-D, and points to effective-dimensionality / embedding methods as the escape — a natural "future work" hook (and a contrast: Ackley has *no* low effective dimensionality, so it's a worst case by construction).

### Possible citations for my paper (paraphrased — no direct quotes copied)

- BO suits black-box, expensive, gradient-free, modest-dimensional problems (Sec. 1) — to scope my d ≤ 10 problem suite.
- The choice of surrogate/statistical model often matters more than the fine choice of acquisition function — to motivate why I treat the GP normalization fix as a first-class methodological result, not a footnote.
- GP's fallback to a high-variance prior far from data is what makes its uncertainty useful for exploration (surrogate section) — to justify GP over RF for my smooth synthetic + engineering functions.
- Exact GP inference is O(n³), bounding practical n — to justify my 20-iteration budget and small-n regime.

### Open questions / to revisit

1. **Kernel ablation (ties back to Frazier's open question).** Both Frazier and Shahriari flag Matérn vs RBF as consequential. BoTorch's SingleTaskGP defaults to RBF. With normalization now in place, an RBF-vs-Matérn-5/2 ablation would be cleaner to interpret — Phase 2 candidate.
2. **β schedule for UCB.** Shahriari (like Frazier) notes UCB's performance depends on β; I use fixed β=2. Does a theory-motivated β_t schedule change the EI≈UCB picture?
3. **Does "surrogate > acquisition" extend to RF?** My Phase 2 plan includes an RF surrogate. Shahriari warns RF uncertainty is less principled and the surface is non-differentiable — so RF + EI/UCB may behave very differently. Worth testing whether the normalization lesson even applies to RF (it shouldn't need input scaling the same way).

---

## Demšar (2006) — Statistical Comparisons of Classifiers over Multiple Data Sets

*Journal of Machine Learning Research, 7:1–30.*

### Summary (own words)

Demšar examines how to correctly compare multiple algorithms across multiple
datasets — a setting that breaks the assumptions of the naive approaches people
often default to. He argues that comparing mean performance with repeated paired
t-tests is statistically unsound here: performance scores across heterogeneous
datasets are not commensurable, are usually non-Gaussian, and repeated pairwise
testing inflates the family-wise type-I error. Instead he recommends
non-parametric, rank-based procedures. For comparing several algorithms over many
datasets, the recommended workflow is the Friedman test (a non-parametric omnibus
test on ranks) followed, on rejection, by a post-hoc test — Nemenyi for all-pairs
comparisons, or Bonferroni–Dunn when comparing everything against a single control.
He also introduces the Critical Difference (CD) diagram as a compact visualization
of which algorithms are statistically separable.

### Key takeaways (tied to this project)

1. **Rank-based, not mean-based.** Regret values across my six problems live on
   wildly different scales (Branin ~0.03 vs Borehole ~100s); averaging or
   t-testing raw regret across them is meaningless. Ranking within each
   (problem, seed) block and testing the ranks is exactly Demšar's prescription —
   this is *why* the project uses Friedman + Nemenyi rather than mean ± std.

2. **Omnibus before post-hoc.** Demšar's two-stage protocol (Friedman first, only
   run pairwise Nemenyi if the omnibus null is rejected) is the structure of
   exp_03 — it controls the type-I error that 6 independent pairwise tests would
   inflate.

3. **CD diagram as the reporting standard.** The CD diagram is Demšar's own
   recommended visualization; using it (rather than a table of p-values alone)
   makes the two-clique result immediately legible and follows community
   convention for this kind of comparison.

### Quotable points (paraphrased)

- Comparing classifiers over multiple datasets needs non-parametric rank tests,
  because per-dataset scores are not drawn from a common commensurable
  distribution and parametric assumptions (normality, homoscedasticity) generally
  fail.
- The Friedman test with a suitable post-hoc (Nemenyi for all-pairs) is thes
  recommended general procedure; the critical difference quantifies the minimum
  average-rank gap that counts as significant.

### Open questions

- Demšar focuses on classifier accuracy; my "datasets" are (problem, seed) blocks
  on a single optimization metric. The rank machinery transfers cleanly, but the
  independence assumption across blocks (10 seeds of the same problem are not
  fully independent of each other) is worth thinking about — does within-problem
  seed correlation weaken the Friedman assumptions? (Likely minor at N=60, but
  noted.)
- For comparing my best strategy (UCB) against all others specifically, would
  Bonferroni–Dunn against UCB-as-control be more powerful than all-pairs Nemenyi?