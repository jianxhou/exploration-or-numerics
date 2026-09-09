# De Ath et al. — Reproducibility Audit Notes for the Author

## Preamble

I ran an independent reproducibility audit of the released code and supplementary material against the paper text, working from the arXiv v2 source (`main.tex` + `supp.tex`, the 27-page supplementary) and the egreedy repository (notebooks, `egreedy/util/plotting.py`, and the test-problem definitions). The notes below record a small set of places where the code and the text describe slightly different things: in each case the code computes one thing while the paper or supplementary text describes a slightly different thing. These are real but limited in scope, mostly definitional or order-of-operations details, and none of them change the median-performance conclusions of the paper. They are shared here collegially in case they are useful for a future revision or for anyone reproducing the results.

## F1 — regret order of operations and `yopt` evaluated at a rounded `xopt`

The metric is documented as absolute distance to the optimum: the Table 2 caption (main.tex:1034) reads "Median absolute distance (left) and median absolute deviation from the median (MAD, right) from the optimum after 250 function evaluations, across the 51 runs," and main.tex:1042-1044 describes "the median regret, i.e. the median difference between the estimated optimum f* and the true optimum." The notebook (`Process_results_...ipynb`, cell 5, markdown) phrases it as "each element of a run is the distance to the optimum of the best (lowest) function evaluation seen so far."

Two details of the implementation differ from that wording:

- **Order of operations.** The code computes the per-evaluation absolute distance `|y − yopt|` first and then takes the cumulative minimum (`plotting.py:74-78`). The notebook's own phrasing ("distance to the optimum of the best function evaluation seen so far") reads as the other order, `|cummin(y) − yopt|`, and the `process_results` docstring (plotting.py:36-38) describes the return value as "the minimum seen expensive function evaluations," whereas what is returned is the cumulative-min distances.
- **`yopt` at a rounded `xopt`.** In the code `yopt = f(xopt)` is evaluated at `xopt = (0.0898, −0.7126)`; because this `xopt` is specified to four decimals, some observed evaluations are slightly smaller than `yopt`. This is the only reason the order of operations is material here: it affects 4 of the 90 Table 2 cells, all on logSixHumpCamel. The relevant log-shift constants are documented (supp.tex:208-216, logSixHumpCamel: "f(x) = log(g(x) + a + b), where a = 1.0316 and b = 1e-4. g(x) has a minimum value of −1.0316 and, therefore, we add a plus a small constant b.").

## F2 — MAD scaling

The reported MAD quantity is computed via `scipy.stats.median_absolute_deviation`, which applies its default `scale=1.4826` (plotting.py:10, 731). The text defines the quantity without that factor: main.tex:1034-1035, 1044-1045 and supp.tex:309-310, 319-320, 349-350, 471, 520-521 all describe it as the (unscaled) "median absolute deviation from the median (MAD)," with no qualifier about scaling. So the table reports the scaled (normal-consistent) estimator while the text defines the unscaled deviation.

## F3 — Wilcoxon sidedness

The equivalence test is described as one-sided in the text: main.tex:1047-1048 reads "statistically equivalent to the best method according to a one-sided paired Wilcoxon signed-rank test [Knowles et al.] with Holm-Bonferroni correction (p >= 0.05)," and the same sentence appears three times in the supplementary (supp.tex:322, 364, 473-474). The code calls `wilcoxon(best, other)` with scipy's two-sided default (plotting.py:748); the notebook (cell 23, markdown) mentions the Wilcoxon test without specifying sidedness. This affects the equivalence annotations on WangFreitas and logRosenbrock.

## F5 — gSobol expression in the supplementary

The supplementary prints the gSobol component as `prod((4x_i − a_i)/2)` (supp.tex:154-264, "Synthetic function details"). As printed, the argument of the subsequent log can be ≤ 0, whereas the implemented expression is strictly positive. The implemented function is the standard Gonzalez gSobol, `prod((|4x_i − 2| + a_i)/(1 + a_i))` with `a_i = 1` (`synthetic_problems.py`). The printed expression and the implemented expression differ; the implemented version is the standard Gonzalez gSobol form. A future revision may wish to reconcile the two.

## Optima reconciliation

The supplementary (Section "Synthetic function details," supp.tex:154-264) gives formulae and shift constants but states no numeric f* for any problem; the implied optima below are derived from its stated minima and constants, alongside the values the implemented code uses.

| Problem | Supplementary statement | Implied optimum | Implemented optimum | Note |
|---|---|---|---|---|
| WangFreitas | formula, f = −g, peak b = 0.9 | ≈ −4 | −4 | match |
| Branin | formula only, no optimum stated | — | 0.397887 | standard value |
| BraninForrester | formula only | — | −16.64402 | — |
| Cosines | formula only | — | −1.6 | consistent with formula at (0.3125, 0.3125) |
| logGoldsteinPrice | formula only | — | 1.098612 = log(3) | standard g min 3 |
| logSixHumpCamel | "g has a minimum value of −1.0316", shift a+b = 1.0316 + 1e-4 | log(1e-4) = −9.21034 | −9.54474 | implemented yopt = f(rounded xopt); true f* ≈ log(7.15e-5) ≈ −9.55 |
| logHartmann6 | formula + constants only | — | −1.20068 = −log(3.32237) | standard |
| logGSobol | g(x) = prod (4x_i − a_i)/2 | undefined (g can be ≤ 0; log min −inf) | −6.931472 = log(0.5^10) | formula as printed; see F5 |
| logRosenbrock | "g has a minimum value of 0", +0.5 | log(0.5) = −0.693147 | −0.6931472 | match |
| logStyblinskiTang | "g has a minimum value of −39.16599·D", +40D | log(8.3401) = 2.121075 | 2.120865 | near match (differs by ~2.1e-4) |

Two entries connect to the findings above: for logSixHumpCamel the supplementary's rounded g-min (−1.0316; exact −1.0316285) implies f* = log(1e-4) = −9.2103, while the implemented yopt = f(rounded xopt) = −9.5447 and the true f* ≈ −9.55 — this difference between the rounded g-min and the exact value is what makes the order of operations material in F1. For logGSobol the printed expression `prod (4x_i − a_i)/2` differs from the implemented `prod (|4x_i − 2| + a_i)/(1 + a_i)` (see F5); as printed the argument of the log can be ≤ 0, whereas the implemented version is strictly positive and is the standard gSobol form.
