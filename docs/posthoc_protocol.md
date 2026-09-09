# Post-hoc Protocol — Revision-Round Additions (exp_12)

**Status: declared before any exp_12 data exists.** This note extends the Phase 3
integrity chain to the revision round. It is append-only in spirit: nothing in the
frozen pre-registration (`phase3-prereg` @ `fccc28a`, Amendment 1 @ `aa53ba5`), the
frozen analysis scripts, or the three confirmatory verdicts (H1 not confirmed, H2a
failed, H3 rejected) is modified by anything below. Everything below is post-hoc and
will be labeled as such wherever it is reported.

## Motivation

External blind-review-style feedback on the paper committed at
`4a7b51a` raised three addressable evidence gaps: (1) the eps-PF arm is a documented
variant of De Ath's eFront (candidate-pool front approximation rather than an NSGA-II
front over the GP posterior), so the corroboration of De Ath's eps-greedy design
rests on a variant; (2) the NSGA-II arms run at De Ath's 5000d acquisition-evaluation
budget while the gradient-based arms run at standard BoTorch settings, leaving a
budget-vs-search-dynamics confound unprobed; (3) the H3 bounded-null claim reports
point estimates without interval statements.

## Additions

All under experiment number **exp_12** (next free number in the repository).

1. **Exact-front eps-PF arm** (`eps-PF (exact front)`): eps-greedy with
   epsilon = 0.1 whose exploration step draws from a Pareto front of the GP
   posterior (mu, sigma) constructed by NSGA-II at De Ath's settings, implemented
   line-by-line against his released `eFront` routine
   (`~/projects/egreedy`, `egreedy/acquisition_functions/`); the exploit step
   follows his routine. A code-audit note maps our implementation to his line by
   line and lists any residual divergence explicitly. 11 problems x 30 seeds =
   **330 cells**, NSGA-II regime only.
2. **Random-search acquisition-optimiser arms** (`EI (RS)`, `LogEI (RS)`): the
   acquisition is optimised by evaluating 5000d uniform-random samples per step
   (seeded per cell) and taking the argmax — the same evaluation budget as the
   NSGA-II arms with no search dynamics. 2 arms x 11 problems x 30 seeds =
   **660 cells**.
3. **Problem-level bootstrap CI analysis**: percentile bootstrap over problems
   (B = 10,000, 95%) for (i) the LogEI-over-EI advantage per optimiser, on both
   the 11-problem and the 10-De-Ath-problem aggregations, and (ii) the mean H3
   contrast D_p. Descriptive only; no decision criterion attaches to it.

## Pairing contract (identical to the main matrix)

- Same byte-identical initial designs as exp_08/exp_10/exp_11: seed *s* maps to
  De Ath design run *s + 1*; T = 250 evaluations.
- Same sanity gate before production: assert the injected initial X equals the
  MAPPED De Ath design run (byte-identical); assert final regret is finite;
  assert 0 negative-regret clips.
- Same atomic-write, skip-existing batch harness as exp_11.

## Scope control

The new arms carry **no mechanism probe**. The probe protocol is frozen with the
pre-registered arms; adding probes to post-hoc arms would extend the mechanism
analysis beyond its declared scope.

## Reporting rules

- Results are reported in whichever direction they resolve, under the same
  aggregations as the main matrix (All-10 De Ath / high-d / low-d / 11-problem)
  plus full per-problem tables.
- The three pre-registered verdicts are unaffected: no exp_12 quantity enters any
  H1/H2a/H3 criterion or is used to reinterpret a verdict.
- Every exp_12 number in the paper is labeled post-hoc and cited to the exp_12
  analysis output; nothing is transcribed from working notes.

## Analysis discipline

The exp_12 analysis script reuses the existing regret helpers read-only (the same
faithful-executor discipline as `exp_11_analysis.py`) and modifies no frozen file.

## Amendment A1 — result-framing pre-commitment (recorded before any exp_12 result was read)

Recorded while the exp_12 matrix run is in progress and before any exp_12 result has
been read or aggregated. Purpose: fix the Section-4.7 framing rules in advance so the
prose cannot be steered by the direction of the results.

**Comparison protocol.** Purely descriptive; no new hypothesis test. Two qualitative
markers, computed from the exp_12 analysis JSON only:
(i) whether exact-front eps-PF remains the best arm on the NSGA-II side (reference:
UCB at mean log-regret -3.24 over De Ath's ten problems);
(ii) the size of the exact-vs-variant gap relative to the variant-vs-UCB gap (0.26).

**Pre-committed framings.**
- Outcome A (exact ~ variant; both best or near-best): report as "the corroboration is
  robust to the front-construction mechanics"; the eps-PF Limitations entry is
  downgraded to resolved post hoc.
- Outcome B (exact better than variant): report as such, adding that the
  pre-registered variant understated eps-PF.
- Outcome C (exact worse than the variant, or loses best-arm status): report first,
  explain after; the Section-4.4 corroboration wording must be weakened to state that
  the pre-registered corroboration rests on the variant and the exact mechanism did
  not reproduce that advantage on our stack; the Limitations entry is upgraded, not
  resolved. Not to be buried.
- RS arms: if the RS ordering matches both main regimes, one sentence is added to
  Section 4.6 closing the budget objection; if RS ordering deviates, report it as
  such -- the Section-4.6 conclusion (about the two main regimes) stands, but the
  two-regime defence in Section 4.2 is correspondingly softened.

**Naming discipline.** The new arm is called "exact-front eps-PF" throughout; the
definition sentence states that it reproduces De Ath's front-construction and
selection mechanics line-by-line from his released code, within our surrogate stack,
with residual divergences documented in the released audit note
(`docs/exp12_exactfront_audit.md`). The phrases "exact De Ath method" and
"De Ath's exact eps-PF" are prohibited.
