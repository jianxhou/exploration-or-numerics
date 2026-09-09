# De Ath et al. supplementary material vs exp_05 findings — documentation audit

Audited sources:
- arXiv 1911.12809 **v2** source (tarball dated 2021-04-29, latest version): `main.tex` + `supp.tex` (the 27-page supplementary). Line numbers refer to these files.
- Repository clone at ~/projects/egreedy (commit 4ab1a991e461): README.md, all markdown cells of the four notebooks, docstrings in `egreedy/util/plotting.py` and `egreedy/test_problems/`.

Search terms used per finding are listed under each verdict. Read-only audit; nothing in their materials was modified.

**Author-confirmed (2026):** F1, F2, F3, and F5 were put to George De Ath by email and confirmed as real paper-text-vs-released-code discrepancies (not deliberate undocumented choices); he granted permission to cite the clarification. Substance in `death_author_correspondence.md`.

## F1 — regret = cummin of per-evaluation |y − yopt|, with coarse-grid yopt

**Verdict: PARTIALLY DOCUMENTED.**

Documented:
- The metric is absolute distance to the optimum. Main text Table 2 caption (main.tex:1034): "Median absolute distance (left) and median absolute deviation from the median (MAD, right) from the optimum after 250 function evaluations, across the 51 runs"; main.tex:1042-1044: "the median regret, i.e. the median difference between the estimated optimum f* and the true optimum". Notebook `Process_results_...ipynb` cell 5 (markdown): "each element of a run is the distance to the optimum of the best (lowest) function evaluation seen so far."
- The log-shift constants that create the situation are documented. supp.tex:208-216 (logSixHumpCamel): "f(x) = log(g(x) + a + b), where a = 1.0316 and b = 1e-4. g(x) has a minimum value of −1.0316 and, therefore, we add a plus a small constant b."

Not documented:
- The order of operations: the code takes |y − yopt| per evaluation and then the cumulative min (`plotting.py:74-78`). The notebook's own phrasing ("distance to the optimum of the best function evaluation seen so far") reads as the other order, |cummin(y) − yopt|, and the `process_results` docstring (plotting.py:36-38) describes the return value as "the minimum seen expensive function evaluations", which is not what is returned (cumulative-min distances are).
- That `yopt` in the code is f(xopt) at a rounded xopt = (0.0898, −0.7126), so observed evaluations can fall below `yopt`, which is the only reason the order of operations is material (4 of 90 Table 2 cells on logSixHumpCamel).

Author clarification (second reply, adopted): the two orderings are equivalent when `yopt` is the true optimum — taking |y − yopt| per evaluation then the cumulative minimum equals the distance from the best (lowest) value seen to yopt. F1's materiality on logSixHumpCamel is therefore attributable to the rounded-coordinate `yopt` (which is not quite the true optimum), not to the ordering per se; recomputing logSixHumpCamel regret against a more accurate reference changes the exact regret values and one equivalence annotation, but not the best-median method or the median-performance conclusion.

Attribution note: the logSixHumpCamel equivalence-annotation change traces to the regret/`yopt` issue (F1); the WangFreitas and logRosenbrock annotation changes trace to the Wilcoxon-sidedness issue (F3) — two distinct root causes affecting different problems.

Search terms checked: "distance", "regret", "estimated optimum", "true optimum", "yopt", "absolute distance" in main.tex, supp.tex, README, notebook markdown, plotting.py, test_problems docstrings.

## F2 — Table "MAD" is scipy's median_absolute_deviation with default scale 1.4826

**Verdict: UNDOCUMENTED** — and the text affirms the unscaled definition the code does not compute.

- Main text (main.tex:1034-1035, 1044-1045) and supplementary (supp.tex:309-310, 319-320, 349-350, 471, 520-521) consistently define the quantity as "median absolute deviation from the median (MAD)". No qualifier about scaling appears anywhere.
- Zero occurrences of "1.4826", "scaled", "consistent estimator", "normal consistency" in main.tex, supp.tex, README.md, notebook markdown cells, or package docstrings. The factor enters only through `scipy.stats.median_absolute_deviation`'s default `scale=1.4826` (plotting.py:10, 731).

Search terms checked: "median absolute deviation", "MAD", "1.4826", "scaled", "consistent estimator" in all audited sources.

## F3 — paper says one-sided Wilcoxon; code calls scipy's two-sided default

**Verdict: UNDOCUMENTED** — the supplementary repeats the one-sided claim.

- main.tex:1047-1048: "statistically equivalent to the best method according to a one-sided paired Wilcoxon signed-rank test [Knowles et al.] with Holm-Bonferroni correction (p >= 0.05)". The identical sentence appears three times in the supplementary (supp.tex:322, 364, 473-474).
- The notebook (cell 23 markdown) says only "using the Wilcoxon signed rank test with Holm-Bonferonni correct." — no sidedness. The code calls `wilcoxon(best, other)` with scipy's two-sided default (plotting.py:748). Zero occurrences of "two-sided"/"two-tailed" anywhere in the audited sources.

Search terms checked: "one-sided", "two-sided", "one-tailed", "two-tailed", "Wilcoxon", "signed-rank", "alternative" in all audited sources.

## Optima: supplementary (Section "Synthetic function details", supp.tex:154-264) vs code yopt (results/exp_05_tier1a.json)

The supplementary gives formulae and shift constants but states no numeric f* for any problem; implied optima are derived here from its stated minima/constants.

| Problem | Supplementary statement | Implied f* | Code yopt | Match |
|---|---|---|---|---|
| WangFreitas | formula, f = −g, peak b = 0.9 | ≈ −4 | −4 | yes |
| Branin | formula only, no optimum stated | — | 0.397887 | n/a (standard value) |
| BraninForrester | formula only | — | −16.64402 | n/a |
| Cosines | formula only | — | −1.6 | n/a (consistent with formula at (0.3125, 0.3125)) |
| logGoldsteinPrice | formula only | — | 1.098612 = log(3) | n/a (standard g min 3) |
| logSixHumpCamel | "g has a minimum value of −1.0316", shift a+b = 1.0316 + 1e-4 | log(1e-4) = **−9.21034** | **−9.54474** | **MISMATCH (0.334)** |
| logHartmann6 | formula + constants only | — | −1.20068 = −log(3.32237) | n/a (standard) |
| logGSobol | g(x) = prod (4x_i − a_i)/2 | undefined (g can be ≤ 0; log min −inf) | −6.931472 = log(0.5^10) | **formula mistyped** |
| logRosenbrock | "g has a minimum value of 0", +0.5 | log(0.5) = −0.693147 | −0.6931472 | yes |
| logStyblinskiTang | "g has a minimum value of −39.16599·D", +40D | log(8.3401) = 2.121075 | 2.120865 | near (2.1e-4; rounded constant) |

Flags:
1. **logSixHumpCamel**: the supplementary's rounded g-min (−1.0316; true −1.0316285) implies f* = log(1e-4) = −9.2103, but the code's yopt = f(rounded xopt) = −9.5447, and the true f* ≈ log(7.15e-5) ≈ −9.55. Neither equals the supplementary's implied value; this imprecision is the root cause of F1's materiality.
2. **logGSobol (finding F5)**: the supplementary formula `prod (4x_i − a_i)/2` differs from the code (`prod (|4x_i − 2| + a_i)/(1 + a_i)`, synthetic_problems.py); as written it can be zero or negative, making the log undefined. Apparent typesetting error; the code's version is the standard gSobol.

## Summary

F2 and F3 are undocumented, and in both cases the supplementary affirms the definition that the code does not implement (unscaled MAD; one-sided test). F1 is partially documented: the absolute-distance metric and the log-shift constants are stated, but the abs-before-cummin order and the coarse yopt are not, and the repo's own docstring/notebook descriptions match the un-implemented order.

## Tier 1B environment notes (Docker re-execution)

- `docker pull georgedeath/egreedy`: 6.6 GB image (bundles OpenFOAM v5 for PitzDaily), 19m19s pull on Apple Silicon with `--platform linux/amd64` (qemu emulation; ~2 s per BO iteration on Branin, fast enough for re-execution).
- The README's documented invocation is misleading for non-interactive use: the image entrypoint is `/bin/sh -c "/entry.sh"`, which prints the OpenFOAM welcome banner and exits 0, silently ignoring any command passed after the image name. `docker run --rm georgedeath/egreedy python -m egreedy.optimizer` therefore "succeeds" having run nothing.
- Working invocation: `docker run --rm --platform linux/amd64 --entrypoint bash georgedeath/egreedy -lc '<command>'` with the repo at `/egreedy` (image WorkingDir) and Python on the login-shell PATH.
- Single-run CLI (README "Reproduction of experiments", and run_experiment.py): `python run_experiment.py -p PROBLEM -b BUDGET -r RUN_NO -a ACQUISITION [-aa epsilon:0.1]`, e.g. `python run_experiment.py -p Branin -b 250 -r 1 -a eFront -aa epsilon:0.1`. Paths are CWD-relative: initial designs from `training_data/{p}_{r}.npz` (bundled in the image), outputs to `results/{p}_{r}_{b}_{a}[_eps{eps:g}].npz`, written every iteration with `continue_runs=True`, so interrupted runs resume. For Tier 1B we symlink `results -> /out` (a host mount) inside the ephemeral container, keeping the published `results_paper/` untouched.
