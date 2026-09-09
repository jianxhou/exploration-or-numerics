# Ament et al. (2023), "Unexpected Improvements to Expected Improvement for Bayesian Optimization" — protocol extraction and BoTorch implementation notes

Source: arXiv 2310.20708 **v3** (source tarball dated 2025-01-07; abs page lists v1–v3, /src/ serves latest). Single TeX file `main.tex` (plus `main.bbl`, `main.bib`, `fig/`, `neurips_2023.sty`). Line numbers below refer to v3 `main.tex`.

## 1. Paper protocol (verbatim numbers, condensed prose)

### Section 5 preamble (lines 541–556)

- All experiments implemented in BoTorch; multi-start acquisition optimization with **scipy's L-BFGS-B**.
- "In order to avoid conflating the effect of BoTorch's default initialization strategy with those of our contributions, we use **16 initial points chosen uniformly at random** from which to start the L-BFGS-B optimization."
- "We run multiple replicates and report mean and error bars of **±2 standard errors of the mean**."

### Section 5, paragraph "Single-objective sequential BO" (lines 562–586)

- Problems:
  - **Sum-of-Squares (SoS), 10-dim**, convex: f(x) = sum_{i=1}^{10} (x_i − 0.5)^2, "using **20 restarts seeded from 1024 pseudo-random samples through BoTorch's default initialization heuristic**".
  - **Ackley** and **Michalewicz** (Surjanovic & Bingham test suite), "for varying numbers of input dimensions". The dimensions are shown only in figure `fig/so_best_obj_q1.pdf`, not stated in the TeX. Budget (iteration count) likewise appears only on figure axes.
- EI stalls on SoS after ~75 observations from vanishing gradients; LogEI gap on Ackley widens with dimension.

**Internal inconsistency — finding F4 (main text, flagged):** the Section 5 preamble specifies 16 uniform-random L-BFGS-B starts precisely to avoid the default heuristic, but the SoS paragraph in the same section uses 20 restarts from 1024 samples via the default (Boltzmann) heuristic. The settings are per-experiment; neither is wrong but they differ within Section 5.

### Appendix D.1 "Experimental details" (lines 1533–1539)

- Analytic EI, qEI, cEI: standard BoTorch implementations; JES/GIBBON/MO-JES: original authors' implementations from the main BoTorch repo.
- "All simulations are ran with **32 replicates** and error bars represent **±2 times the standard error of the mean**."
- GP: **Matern-5/2 kernel with ARD**, "**top-hat prior on the length-scales in [0.01, 100]**".
- "Input spaces are normalized to the unit hyper-cube and the objective values are **standardized during each optimization iteration**."
- Constrained problems (Section 5 figure caption, line 602): "after an initial **2d pseudo-random evaluations**". No initial-design size is stated for the unconstrained single-objective runs.

Consistency: Section 5 "multiple replicates, ±2 SEM" agrees with D.1's 32 replicates, ±2 SEM. The 16-vs-20 restart discrepancy above is the only disagreement found.

### Appendix D.2 "Additional Empirical Results on Vanishing Values and Gradients" (lines 1541–1570)

Definitions behind the vanishing-gradient measurements (intro Figure 1, left):

- Measured quantity: "fraction of points sampled from the domain for which the **magnitude of the gradient of EI vanishes to < 1e-10**, as a function of the number of randomly generated data points n for different dimensions d on the Ackley function" (caption, line 214).
- Data-generating process: **80%** of training points uniform on the domain, **20%** from a multivariate Gaussian centered at the function maximum with standard deviation **25% of the domain length** (mimics data seen during a BO loop without running BO).
- Illustration replicate: **60 training points, 2000 test points** (50 shown); z(x) is the argument to h in the analytic EI formula EI = sigma * h(z); thresholds drawn at z values below which h(z) is under each threshold. sigma(x) ≈ 1 for most test points (**mean 0.87, std 0.07**), so thresholds on h(z) are approximately thresholds on EI.

### Appendix B.1 "Common initialization heuristics for multi-start gradient-descent" (lines 1340–1356)

- Problem: gradient-based acquisition optimization sticks in local optima; addressed by restarts from initial conditions across the domain.
- Package strategies: scikit-optimize uniform-random restarts; GPyOpt augments random restarts with best observed points or Thompson samples; Spearmint Gaussian perturbations of the current best; **BoTorch performs Boltzmann sampling on a set of random points according to their acquisition value** — biased toward high-acquisition regions but asymptotically space-filling; Trieste similar but deterministic top-k instead of soft randomization; Gramacy et al. Delaunay-triangulation candidates (poor scaling in d and n).
- Appendix D ("Effect of the initialization strategy", lines 1740–1744) adds: BoTorch's default evaluates the acquisition on points from a **scrambled Sobol sequence** and selects n restarts by Boltzmann sampling (softmax over acquisition values); the comparison there uses **1024 initial candidates**.

## 2. BoTorch implementation (local probe, conda env oas-test)

Versions: **botorch 0.17.2, gpytorch 1.15.2, torch 2.12.0**.

### Analytic LogEI path (`botorch/acquisition/analytic.py`)

`LogExpectedImprovement.forward`:

```python
mean, sigma = self._mean_and_sigma(X)
u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
return (_log_ei_helper(u) + sigma.log()).squeeze(-1)
```

`_scaled_improvement` (same module): `u = (mean - best_f) / sigma`, negated if minimizing.

`_log_ei_helper` (same module), verbatim:

```python
def _log_ei_helper(u: Tensor) -> Tensor:
    """Accurately computes log(phi(u) + u * Phi(u)) in a differentiable manner for u in
    [-10^100, 10^100] in double precision, and [-10^20, 10^20] in single precision.
    Beyond these intervals, a basic squaring of u can lead to floating point overflow.
    In contrast, the implementation in _ei_helper only yields usable gradients down to
    u ~ -10. As a consequence, _log_ei_helper improves the range of inputs for which a
    backward pass yields usable gradients by many orders of magnitude.
    """
    if not (u.dtype == torch.float32 or u.dtype == torch.float64):
        raise TypeError(...)
    # The function has two branching decisions. The first is u < bound, and in this
    # case, just taking the logarithm of the naive _ei_helper implementation works.
    bound = -1
    u_upper = u.masked_fill(u < bound, bound)  # mask u to avoid NaNs in gradients
    log_ei_upper = _ei_helper(u_upper).log()

    # When u <= bound, we need to be more careful and rearrange the EI formula as
    # log(phi(u)) + log(1 - exp(w)), where w = log(abs(u) * Phi(u) / phi(u)).
    neg_inv_sqrt_eps = -1e6 if u.dtype == torch.float64 else -1e3

    u_lower = u.masked_fill(u > bound, bound)
    u_eps = u_lower.masked_fill(u < neg_inv_sqrt_eps, neg_inv_sqrt_eps)
    w = _log_abs_u_Phi_div_phi(u_eps)

    log_ei_lower = log_phi(u) + (
        torch.where(
            u > neg_inv_sqrt_eps,
            log1mexp(w),
            # captures the leading order of the log1mexp term for very negative u
            -2 * u_lower.abs().log(),
        )
    )
    return torch.where(u > bound, log_ei_upper, log_ei_lower)
```

Supporting helpers (same module): `_ei_helper(u) = phi(u) + u * Phi(u)`;
`_log_abs_u_Phi_div_phi(u) = log(erfcx(-u/sqrt(2)) * |u|) + log(sqrt(pi/2))`, valid for u < 0, built on the scaled complementary error function `torch.special.erfcx`.

### Temperature/tau finding

**No temperature or tau parameter enters the analytic LogEI path.** The string "tau"/"temperature" does not occur in `botorch.acquisition.analytic` (0 occurrences). The branch structure of `_log_ei_helper` is exact (case analysis on u with fixed thresholds −1 and −1/sqrt(eps)), not a smoothed approximation. `tau_relu` (default 1e-6) and `tau_max` (default 1e-2) exist only in `botorch.acquisition.logei` for the Monte Carlo batch variants (`qLogExpectedImprovement` etc.), where they smooth max(·, 0) and the q-max.

### optimize_acqf and the default initializer

`botorch.optim.optimize_acqf` signature defaults: `raw_samples=None`, `options=None`, `batch_initial_conditions=None`, `return_best_only=True`, `gen_candidates=None` (resolves to `gen_candidates_scipy`, i.e. scipy L-BFGS-B for box constraints), `sequential=False`, `ic_generator=None` (resolves to `gen_batch_initial_conditions`), `retry_on_optimization_warning=True`. `num_restarts` is required; `raw_samples` must be supplied unless initial conditions are passed explicitly.

Default initializer chain, `gen_batch_initial_conditions` -> `initialize_q_batch`:

1. Draw `raw_samples` points from a **scrambled Sobol sequence** over the bounds (`draw_sobol_samples`; falls back to iid uniform above Sobol's max dimension); evaluate the acquisition on all of them.
2. `initialize_q_batch` selects `num_restarts` of them **without replacement with probability proportional to exp(eta * Z)**, where `Z = (acq_vals - mean(acq_vals)) / std(acq_vals)` and `eta=1.0` by default (`boltzmann_sample`; eta is halved on inf weights). The argmax of the raw acquisition values is forced into the selected set. If all raw acquisition values are equal (std 0), selection falls back to uniform random with a `BadInitialCandidatesWarning`.
3. Variants: `initialize_q_batch_nonneg` (for `options={"nonnegative": True}`, suited to acquisitions that are exactly zero over large regions, e.g. qEI) and `initialize_q_batch_topn` (`options={"topn": True}`, Trieste-style top-k).

This matches the paper's B.1 description of BoTorch's strategy (Boltzmann sampling over acquisition values, asymptotically space-filling).

## 3. Sanity check

SingleTaskGP (default Matern-5/2 ARD) on 5 uniform-random points of the 2d quadratic f(x) = -sum((x - 0.5)^2), double precision, seed 0; best_f = -0.051460. EI and LogEI evaluated at 3 test points (evaluation only):

| x | EI | LogEI | exp(LogEI) | rel err |
|---|---|---|---|---|
| (0.50, 0.50) | 2.113148e-03 | -6.159576 | 2.113148e-03 | 2.1e-16 |
| (0.05, 0.95) | 3.884172e-03 | -5.550845 | 3.884172e-03 | 5.6e-16 |
| (0.90, 0.10) | 1.004344e-03 | -6.903421 | 1.004344e-03 | 1.1e-15 |

exp(LogEI) matches EI at machine precision wherever EI > 0. Note: instantiating `ExpectedImprovement` in botorch 0.17.2 emits a `NumericsWarning` recommending `LogExpectedImprovement`, citing this paper.

## 4. Benchmark dimensions and budgets recovered from figure text

The TeX never states the Ackley/Michalewicz dimensions or budgets; they are embedded in the figure PDFs. Extracted with pdfminer.six (pdftotext unavailable) from `/fig/` of the v3 source; panel titles and tick labels are machine-readable, rotated axis labels render letter-by-letter but are decodable. The verifiable datum for each budget is the last x-axis tick label; whether the axis extends past the last tick cannot be established from text extraction, so budgets below are "ticks end at N".

| Problem | Dim | Budget (last x tick) | q | Source figure |
|---|---|---|---|---|
| Ackley | 2, 8, 16 | 250 (Function evaluations) | 1 | `so_best_obj_q1.pdf` (Sec. 5 figure; panel titles `d = 2/8/16`, both problem rows) |
| Michalewicz | 2, 8, 16 | 250 (Function evaluations) | 1 | `so_best_obj_q1.pdf` |
| Sum-of-Squares | 10 (TeX line 563) | 200 (Number of evaluations) | 1 | `logei_intro_fig_small.pdf` |
| Ackley | 16 | 200 (Function evaluations) | 4, 16, 32 | `qso_best_obj_ackley.pdf` (Sec. 5 parallel figure, panel titles `q = 4/16/32`) |
| Ackley, Levy | 16 | 250 (Function evaluations) | 4, 8, 16, 32 | `qso_best_obj.pdf` (appendix; Levy confirmed in rotated label) |
| Hartmann | 6 | 200 (Function evaluations) | 1 | `nso_best_obj_q1.pdf` (noisy; panels Noise = 1.0/2.0/5.0% * Range(f)) |
| Ackley | 8, 16 | 200 (Function evaluations) | 1 | `nso_best_obj_q1.pdf` |
| Ackley, Michalewicz | 2, 8, 16 | 200 (Function Evaluations) | 1 | `so_boltzmann_random.pdf` (init-strategy comparison) |
| Ackley, Levy | 16 (caption, line 1761) | x-axis = number of restart points (ticks 4-16) | 1, 4, 16 | `num_restarts_sensitivity.pdf` (panel titles `ackley/levy q=1/4/16`) |
| Ackley, SoS | 2, 16 | 250 | 1 | `qlogei_tau_softplus_ablation.pdf` (tau ablation, `tau_softplus` in {1e-1..1e-6}) |
| Ackley | 50 | 10000 (Number of evaluations) | 50 | `turbo_ackley.pdf` (panel title `50D Ackley, q=50`) |
| Tension-Compression String | 3 (+4 constraints) | 100 (Iterations) | 1 | `cso_best_obj_q1.pdf` (panel titles state dims and constraint counts) |
| Pressure Vessel Design | 4 (+4 constraints) | 100 (Iterations) | 1 | `cso_best_obj_q1.pdf` |
| Welded Beam Design | 4 (+5 constraints) | 100 (Iterations) | 1 | `cso_best_obj_q1.pdf` |
| Speed Reducer Design | 7 (+11 constraints) | 100 (Iterations) | 1 | `cso_best_obj_q1.pdf` |
| Embedded Hartmann6 | 100 | 100 (Iterations) | 4 | `high_dim.pdf` (panel title `100D Embedded Hartmann6 (q = 4)`) |
| SVM | 103 | 100 (Iterations) | 4 | `high_dim.pdf` (panel title `103D SVM (q = 4)`) |
| Rover | 100 (main text, "100-dimensional rover trajectory planning") | 100 (Iterations) | 4 (by analogy with sibling panels; q not in extracted text for this panel) | `high_dim.pdf` (rotated "Rover" label confirmed) |

Not establishable from text extraction: the exact axis endpoints beyond the last tick; the SoS budget past 200; the q for the Rover panel (no `(q = ...)` string extracted); `VanishingGradientPlot.pdf` contains no extractable text at all (the intro vanishing-gradient figure's n and d values are therefore unrecoverable this way; its TeX caption gives only "for different dimensions d").
