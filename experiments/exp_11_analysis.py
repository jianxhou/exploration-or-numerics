"""Phase 3 confirmatory analysis (pre-registration phase3-prereg, §3 + §5) -- FAITHFUL EXECUTOR.

Implements the FROZEN H1/H2a/H3 tests verbatim, pairing the NSGA-II matrix
(results/exp_11_matrix/) against the frozen gradient-based matrices (exp_08 + exp_10) by
(problem, acquisition, seed). It computes log precise regret via the MERGED optimal-value
table {**_problem_optimal_values(), **_ext_optimal_values()} (epsilon=1e-12, negative-regret
clip max(regret,0) per §5.1) and runs the pre-registered tests with their pre-registered
decision criteria. NO test, threshold, aggregation, or primary/secondary ordering is chosen
from the data; verdicts are reported in whichever direction they resolve (§6).

Frozen artifacts (exp_09_prereg_analysis.py, exp_08/exp_10 matrices, nsga2.py) are READ-ONLY
here: exp_09's regret helpers are imported, never modified.

Usage:  python experiments/exp_11_analysis.py            # run all, write JSON + print summary
        python experiments/exp_11_analysis.py --json-only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).parent.parent
GRAD = {"exp_08": PROJECT_ROOT / "results" / "exp_08_matrix",
        "exp_10": PROJECT_ROOT / "results" / "exp_10_matrix"}
NSGA = PROJECT_ROOT / "results" / "exp_11_matrix"
OUT_JSON = PROJECT_ROOT / "results" / "exp_11_analysis.json"

EPS = 1e-12
ALPHA = 0.05
N_SEEDS = 30

# 11 in-scope problems -> (gradient-based matrix key, dimension). Ackley8 from exp_10.
PROBLEMS = {
    "WangFreitas": ("exp_08", 1), "BraninForrester": ("exp_08", 2), "Branin": ("exp_08", 2),
    "Cosines": ("exp_08", 2), "GoldsteinPriceLog": ("exp_08", 2), "SixHumpCamelLog": ("exp_08", 2),
    "Hartmann6Log": ("exp_08", 6), "Ackley8": ("exp_10", 8),
    "GSobolLog": ("exp_08", 10), "RosenbrockLog": ("exp_08", 10),
    "StyblinskiTangLog": ("exp_08", 10),
}
H2_SHARED = [p for p in PROBLEMS if p != "Ackley8"]  # Ackley8 excluded from De Ath repro (§4.2)
H2_HIGHD = [p for p in H2_SHARED if PROBLEMS[p][1] >= 6]  # d>=6 shared: Hartmann6Log + 3 d10

# De Ath (2021) reference ordering of the 3 anchors, best->worst, for H2a criterion (1).
# From the pre-reg framing (§1/§2 Claim A): exploitation competitive/best, EI weakest in
# aggregate. The EI-vs-UCB sub-order is NOT crisply specified by De Ath -> FLAGGED; a sensitivity
# ordering with EI/UCB swapped is also reported. We do NOT invent a per-problem reference.
DEATH_REF_ORDER = ["Exploit", "UCB", "EI"]
DEATH_REF_ALT = ["Exploit", "EI", "UCB"]


def _e9():
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import exp_09_prereg_analysis as e9
    return e9


def load_log_regret() -> tuple[dict, dict, int, int]:
    """Return (L, probe, n_cells, n_clipped). L[(problem, acq, optimizer, seed)] = log precise
    regret via the merged table; clip max(regret,0) + EPS before log (§5.1)."""
    e9 = _e9()
    merged = {**e9._problem_optimal_values(), **e9._ext_optimal_values()}  # F1 merged table
    L, probe = {}, {}
    n_cells = n_clipped = 0
    acqs = ("EI", "LogEI", "UCB", "Exploit", "eps-PF")

    def add(rec, optimizer):
        nonlocal n_cells, n_clipped
        p = rec["problem"]
        acq = rec.get("acq", rec["arm"])  # nsga records carry acq; grad records: arm == acq
        if p not in PROBLEMS or acq not in acqs:
            return
        seed = int(rec["seed"])
        tracks = e9._tracks_from_y(np.asarray(rec["y"], dtype=np.float64), merged[p], None)
        regret = tracks["precise_final"]
        if regret < 0:
            n_clipped += 1
        L[(p, acq, optimizer, seed)] = float(np.log(max(regret, 0.0) + EPS))
        n_cells += 1
        if rec.get("probe") and acq in ("EI", "LogEI"):  # attribution uses EI/LogEI only
            probe[(p, acq, optimizer, seed)] = rec["probe"]

    # gradient-based: exp_08 (10 problems) + exp_10 (Ackley8 only, in scope)
    for p, (mkey, _dim) in PROBLEMS.items():
        for acq in acqs:
            for s in range(N_SEEDS):
                f = GRAD[mkey] / f"{p}__{acq}__s{s:02d}.json"
                if f.exists():
                    add(json.load(open(f)), "grad")
    # NSGA-II: exp_11 (read by content -> acq/optimizer fields)
    for f in sorted(NSGA.glob("*.json")):
        rec = json.load(open(f))
        add(rec, rec.get("optimizer", "nsga"))
    return L, probe, n_cells, n_clipped


def _ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return [float("nan"), float("nan")]
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return [float(x.mean() - t * se), float(x.mean() + t * se)]


# ---------------------------------------------------------------------------
# H1 -- greedy advantage grows with dimension, under BOTH optimizers
# ---------------------------------------------------------------------------
def analyze_h1(L) -> dict:
    out = {"per_optimizer": {}, "dimension_levels": sorted({d for _, d in PROBLEMS.values()})}
    probs = list(PROBLEMS)
    logdim = np.array([np.log(PROBLEMS[p][1]) for p in probs])
    for o in ("grad", "nsga"):
        A_p, missing = [], []
        for p in probs:
            diffs = [L[(p, "EI", o, s)] - L[(p, "Exploit", o, s)]
                     for s in range(N_SEEDS)
                     if (p, "EI", o, s) in L and (p, "Exploit", o, s) in L]
            if len(diffs) != N_SEEDS:
                missing.append((p, len(diffs)))
            A_p.append(float(np.mean(diffs)))
        A_p = np.asarray(A_p)
        # PRIMARY: A_p ~ log(dim), OLS with HC3
        X = sm.add_constant(logdim)
        res = sm.OLS(A_p, X).fit(cov_type="HC3")
        slope = float(res.params[1])
        pval = float(res.pvalues[1])
        ci = [float(c) for c in res.conf_int()[1]]
        rho, rho_p = stats.spearmanr(A_p, logdim)
        confirmed = bool(slope > 0 and pval < ALPHA and rho > 0)
        # SECONDARY: seed-level mixed model L ~ C(acq)*log(dim) + (1|problem), {EI, Exploit}
        rows = [{"L": L[(p, a, o, s)], "acq": a, "logdim": np.log(PROBLEMS[p][1]), "problem": p}
                for p in probs for a in ("EI", "Exploit") for s in range(N_SEEDS)
                if (p, a, o, s) in L]
        df = pd.DataFrame(rows)
        sec = {}
        try:
            m = smf.mixedlm("L ~ C(acq, Treatment('EI')) * logdim", df,
                            groups="problem").fit(reml=True)
            key = [k for k in m.params.index if "logdim" in k and "acq" in k]
            if key:
                k = key[0]
                sec = {"interaction_term": k, "coef": float(m.params[k]),
                       "p": float(m.pvalues[k]), "ci": [float(c) for c in m.conf_int().loc[k]]}
        except Exception as e:  # noqa: BLE001
            sec = {"error": str(e)}
        # per-problem paired Wilcoxon Exploit-vs-EI (Holm across problems)
        wil = []
        for p in probs:
            d = np.array([L[(p, "EI", o, s)] - L[(p, "Exploit", o, s)] for s in range(N_SEEDS)])
            try:
                w = stats.wilcoxon(d, alternative="greater")  # Exploit better => L(EI)-L(Exploit)>0
                pv = float(w.pvalue)
            except ValueError:
                pv = float("nan")
            wil.append({"problem": p, "dim": PROBLEMS[p][1],
                        "median_logregret_diff": float(np.median(d)),
                        "frac_seeds_exploit_better": float(np.mean(d > 0)), "wilcoxon_p_raw": pv})
        pvs = [w["wilcoxon_p_raw"] for w in wil]
        finite = np.isfinite(pvs)
        holm = np.full(len(pvs), np.nan)
        if finite.any():
            holm[finite] = multipletests(np.array(pvs)[finite], method="holm")[1]
        for w, hp in zip(wil, holm, strict=False):
            w["wilcoxon_p_holm"] = float(hp) if np.isfinite(hp) else None
        out["per_optimizer"][o] = {
            "A_p": {p: float(a) for p, a in zip(probs, A_p, strict=False)},
            "primary_ols_hc3": {"slope": slope, "p": pval, "ci95": ci,
                                "spearman_rho": float(rho), "spearman_p": float(rho_p),
                                "confirmed_this_optimizer": confirmed},
            "secondary_mixedlm_interaction": sec,
            "secondary_perproblem_wilcoxon_holm": wil,
            "missing_cells": missing,
        }
    g, n = out["per_optimizer"]["grad"], out["per_optimizer"]["nsga"]
    out["H1_CONFIRMED"] = bool(g["primary_ols_hc3"]["confirmed_this_optimizer"]
                              and n["primary_ols_hc3"]["confirmed_this_optimizer"])
    out["note_dimension_ties"] = ("5 distinct dims {1,2,6,8,10}; ties at d=2 (n=5), d=10 (n=3) "
                                  "=> slope + Spearman driven by few effective dimension points")
    return out


# ---------------------------------------------------------------------------
# H2a -- De Ath reproduction (NSGA-II arm), anchors {EI, UCB, Exploit}
# ---------------------------------------------------------------------------
def analyze_h2(L) -> dict:
    anchors = ["EI", "UCB", "Exploit"]

    def m_pa(optimizer):
        return {a: {p: float(np.mean([L[(p, a, optimizer, s)] for s in range(N_SEEDS)]))
                    for p in H2_SHARED} for a in anchors}

    M_pa = m_pa("nsga")
    M_a = {a: float(np.mean([M_pa[a][p] for p in H2_SHARED])) for a in anchors}  # equal weight
    our_order = sorted(anchors, key=lambda a: M_a[a])  # ascending log-regret = best -> worst

    def spearman_vs(ref):
        rank_ref = {a: i for i, a in enumerate(ref)}
        rank_our = {a: i for i, a in enumerate(our_order)}
        x = [rank_ref[a] for a in anchors]
        y = [rank_our[a] for a in anchors]
        return float(stats.spearmanr(x, y).statistic)

    rho_primary = spearman_vs(DEATH_REF_ORDER)
    rho_alt = spearman_vs(DEATH_REF_ALT)
    # criterion (2): on d>=6 shared, Exploit at-or-above EI (M_Exploit <= M_EI) in a majority
    exploit_geq_ei = {p: bool(M_pa["Exploit"][p] <= M_pa["EI"][p]) for p in H2_HIGHD}
    n_exploit = sum(exploit_geq_ei.values())
    crit2 = bool(n_exploit > len(H2_HIGHD) / 2)
    crit1 = bool(rho_primary >= 0.5)
    epspf = {p: float(np.mean([L[(p, "eps-PF", "nsga", s)] for s in range(N_SEEDS)]))
             for p in H2_SHARED}
    # descriptive context (NOT the verdict): the same ordering under the gradient-based stack
    # tells whether a failed reproduction is specific to the NSGA-II substitution or general.
    g_Mpa = m_pa("grad")
    g_Ma = {a: float(np.mean([g_Mpa[a][p] for p in H2_SHARED])) for a in anchors}
    g_highd = sum(g_Mpa["Exploit"][p] <= g_Mpa["EI"][p] for p in H2_HIGHD)
    return {
        "M_pa_nsga": M_pa, "M_a_nsga": M_a, "our_order_best_to_worst": our_order,
        "criterion1_spearman": {
            "rho_vs_DeAth_ref": rho_primary, "ref_order": DEATH_REF_ORDER,
            "rho_vs_alt_EIUCB_swapped": rho_alt, "alt_order": DEATH_REF_ALT,
            "threshold": 0.5, "passed": crit1,
            "ambiguity_flag": "De Ath's exact EI-vs-UCB sub-order is not crisply specified; "
                              "reference uses Exploit-best (paper-supported) with EI/UCB flagged; "
                              "alt rho reported as sensitivity."},
        "criterion2_highd_majority": {
            "problems": H2_HIGHD, "exploit_at_or_above_ei": exploit_geq_ei,
            "n_exploit_wins": n_exploit, "n_total": len(H2_HIGHD), "passed": crit2},
        "H2a_PASS": bool(crit1 and crit2),
        "H2b_epsPF_descriptive": {"M_pa_nsga": epspf,
                                  "M_a_nsga": float(np.mean(list(epspf.values()))),
                                  "note": "descriptive only; excluded from any pass/fail (§3 H2b)"},
        "gradient_context_descriptive": {
            "M_a_grad": g_Ma, "order_best_to_worst": sorted(anchors, key=lambda a: g_Ma[a]),
            "exploit_at_or_above_ei_highd": f"{g_highd}/{len(H2_HIGHD)}",
            "note": "NOT the verdict; for cause attribution. A matching grad ordering means the "
                    "failed reproduction is NOT due to the NSGA-II substitution."},
    }


# ---------------------------------------------------------------------------
# H3 -- value-vs-gradient (THE CORE), reported either way
# ---------------------------------------------------------------------------
def analyze_h3(L) -> dict:
    probs = list(PROBLEMS)
    D_ps, rows = {}, []
    for p in probs:
        ds = []
        for s in range(N_SEEDS):
            keys = [(p, "EI", "grad", s), (p, "LogEI", "grad", s),
                    (p, "EI", "nsga", s), (p, "LogEI", "nsga", s)]
            if not all(k in L for k in keys):
                continue
            d = (L[keys[0]] - L[keys[1]]) - (L[keys[2]] - L[keys[3]])
            ds.append(d)
            rows.append({"D": d, "problem": p})
        D_ps[p] = ds
    D_p = np.array([float(np.mean(D_ps[p])) for p in probs])

    # descriptive components of D_p (NOT a test): LogEI advantage = mean_s[L(EI)-L(LogEI)] under
    # each optimizer (positive = LogEI better). Shows whether there is a LogEI edge to shrink.
    comp = {}
    for p in probs:
        ga = float(np.mean([L[(p, "EI", "grad", s)] - L[(p, "LogEI", "grad", s)]
                            for s in range(N_SEEDS)]))
        na = float(np.mean([L[(p, "EI", "nsga", s)] - L[(p, "LogEI", "nsga", s)]
                            for s in range(N_SEEDS)]))
        comp[p] = {"logei_adv_grad": ga, "logei_adv_nsga": na}
    mean_adv_grad = float(np.mean([comp[p]["logei_adv_grad"] for p in probs]))
    mean_adv_nsga = float(np.mean([comp[p]["logei_adv_nsga"] for p in probs]))

    # PRIMARY: one-sided Wilcoxon signed-rank on D_p > 0 across the 11 problems
    w = stats.wilcoxon(D_p, alternative="greater")
    wilcoxon_p = float(w.pvalue)
    # sensitivity: paired (one-sample) t-test + sign test, direction D>0
    t = stats.ttest_1samp(D_p, 0.0, alternative="greater")
    sign = stats.binomtest(int(np.sum(D_p > 0)), len(D_p), 0.5, alternative="greater")
    confirmed = bool(wilcoxon_p < ALPHA)  # one-sided 'greater' => p<alpha means D>0 significant

    # SECONDARY: seed-level D_{p,s} ~ 1 + (1|problem), MixedLM REML
    df = pd.DataFrame(rows)
    sec = {}
    try:
        m = smf.mixedlm("D ~ 1", df, groups="problem").fit(reml=True)
        coef = float(m.params["Intercept"])
        pp = float(m.pvalues["Intercept"])
        ci = [float(c) for c in m.conf_int().loc["Intercept"]]
        # two-sided p from statsmodels; one-sided in-direction (>0):
        one_sided = pp / 2 if coef > 0 else 1 - pp / 2
        sec = {"intercept": coef, "p_two_sided": pp, "p_one_sided_in_direction": float(one_sided),
               "ci95": ci, "significant_in_direction": bool(coef > 0 and one_sided < ALPHA)}
    except Exception as e:  # noqa: BLE001
        sec = {"error": str(e)}

    # combined interpretation per §3 H3 (do NOT upgrade to confirmed on seed-level alone)
    if confirmed:
        interp = "H3 CONFIRMED at the problem level (D>0): LogEI advantage shrinks under NSGA-II."
    elif sec.get("significant_in_direction"):
        interp = ("H3 NOT confirmed at the problem level, but the seed-level model is significant "
                  "in-direction => 'directionally supported but underpowered at the problem level' "
                  "(Phase-1 omnibus-underpowered pattern). NOT upgraded to confirmed.")
    else:
        interp = ("H3 REJECTED: the LogEI advantage is NOT smaller under gradient-free "
                  "optimization (problem-level primary not sig in direction D>0, seed-level "
                  "not in-direction either) => consistent with value-underflow, not "
                  "gradient-vanishing, being operationally dominant (§3 H3 falsification).")
    return {
        "D_p": {p: float(d) for p, d in zip(probs, D_p, strict=False)},
        "mean_D_p": float(D_p.mean()), "median_D_p": float(np.median(D_p)),
        "ci95_mean_D_p": _ci95(D_p), "n_problems_D_p_positive": int(np.sum(D_p > 0)),
        "primary_wilcoxon_one_sided_greater": {"statistic": float(w.statistic), "p": wilcoxon_p,
                                               "alpha": ALPHA, "H3_CONFIRMED": confirmed},
        "sensitivity_ttest_one_sided": {"t": float(t.statistic), "p": float(t.pvalue)},
        "sensitivity_sign_test": {"n_positive": int(np.sum(D_p > 0)), "n": len(D_p),
                                  "p": float(sign.pvalue)},
        "secondary_seedlevel_mixedlm": sec,
        "descriptive_components": {"per_problem_logei_advantage": comp,
                                   "mean_logei_adv_grad": mean_adv_grad,
                                   "mean_logei_adv_nsga": mean_adv_nsga,
                                   "note": "both means near zero => little robust LogEI "
                                           "edge over EI under either optimizer (descriptive)"},
        "H3_VERDICT": ("CONFIRMED" if confirmed else "REJECTED"),
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Attribution (Claim C / §5.2 probe) -- DESCRIPTIVE ONLY, no significance test
# ---------------------------------------------------------------------------
def analyze_attribution(probe) -> dict:
    """Per (acq in {EI,LogEI}, optimizer, dimension): raw-EI underflow fraction + own-surface
    gradient health. NSGA probe = K=4096 raw-EI/own-grad; grad probe = K=64 own-acqf (caveat)."""
    rows = []
    for (p, acq, o, _seed), series in probe.items():
        if acq not in ("EI", "LogEI"):
            continue
        dim = PROBLEMS[p][1]
        if o == "nsga":  # §5.2 K=4096 probe
            uf = np.mean([e["raw_ei_frac_zero"] for e in series])
            gv = [e["own_grad"]["frac_below_tiny"] for e in series if e.get("own_grad")]
            gn = [e["own_grad"]["median_log10_norm"] for e in series
                  if e.get("own_grad") and np.isfinite(e["own_grad"]["median_log10_norm"])]
            rows.append({"acq": acq, "optimizer": o, "dim": dim,
                         "raw_ei_underflow_frac": float(uf),
                         "own_grad_frac_below_tiny": float(np.mean(gv)) if gv else float("nan"),
                         "own_grad_median_log10_norm":
                             float(np.median(gn)) if gn else float("nan")})
        else:  # grad: existing K=64 AcqProbe (own-acqf value + gradient)
            cp = [e["candidate_pool"] for e in series if "candidate_pool" in e]
            if not cp:
                continue
            uf = np.mean([c["frac_acqf_zero"] for c in cp])  # own-acqf (EI=raw EI; LogEI=LogEI val)
            gf = np.mean([1.0 - c["frac_nonzero_grad"] for c in cp
                          if np.isfinite(c.get("frac_nonzero_grad", np.nan))])
            gn = [np.log10(c["median_grad_norm"]) for c in cp
                  if c.get("median_grad_norm", 0) and np.isfinite(c["median_grad_norm"])
                  and c["median_grad_norm"] > 0]
            rows.append({"acq": acq, "optimizer": o, "dim": dim,
                         "own_acqf_underflow_frac": float(uf),
                         "own_grad_frac_below_tiny": float(gf),
                         "own_grad_median_log10_norm":
                             float(np.median(gn)) if gn else float("nan")})
    df = pd.DataFrame(rows)
    summary = {}
    if not df.empty:
        agg = df.groupby(["acq", "optimizer", "dim"]).mean(numeric_only=True).reset_index()
        summary = agg.to_dict(orient="records")
    return {
        "per_acq_optimizer_dim": summary,
        "CAVEAT": ("NSGA probe = K=4096 raw-EI underflow + own-surface gradient; grad probe = "
                   "existing K=64 AcqProbe (own-acqf value -> for LogEI that is the LogEI value, "
                   "NOT raw EI). Cross-optimizer comparison is DESCRIPTIVE, never a tested claim."),
    }


def _fmt_p(p):
    return "n/a" if p is None or not np.isfinite(p) else f"{p:.4g}"


def print_summary(R) -> None:
    print("=" * 84)
    print("PHASE 3 CONFIRMATORY ANALYSIS (pre-reg phase3-prereg §3/§5) -- PRE-REGISTERED VERDICTS")
    print("=" * 84)
    print(f"cells loaded: {R['data']['n_cells']} | negative-regret clips: {R['data']['n_clipped']} "
          f"({100*R['data']['n_clipped']/max(R['data']['n_cells'],1):.2f}%)")

    print("\n--- H1: greedy advantage grows with dimension, under BOTH optimizers ---")
    for o in ("grad", "nsga"):
        pr = R["H1"]["per_optimizer"][o]["primary_ols_hc3"]
        verdict = "pass" if pr["confirmed_this_optimizer"] else "FAIL"
        print(f"  [{o}] OLS slope(A_p~log dim) = {pr['slope']:+.4f}  HC3 p = {_fmt_p(pr['p'])}  "
              f"Spearman = {pr['spearman_rho']:+.3f} (p {_fmt_p(pr['spearman_p'])}) -> {verdict}")
    print(f"  H1 CONFIRMED (both optimizers): {R['H1']['H1_CONFIRMED']}")

    print("\n--- H2a: De Ath reproduction (NSGA-II), anchors {EI,UCB,Exploit} ---")
    h2 = R["H2"]
    print("  M_a (mean log-regret, lower=better): " +
          "  ".join(f"{a}={v:.3f}" for a, v in h2["M_a_nsga"].items()))
    print(f"  our order best->worst: {h2['our_order_best_to_worst']}")
    c1 = h2["criterion1_spearman"]
    c2 = h2["criterion2_highd_majority"]
    print(f"  crit1 Spearman vs {c1['ref_order']}: rho={c1['rho_vs_DeAth_ref']:+.2f} "
          f"(alt: {c1['rho_vs_alt_EIUCB_swapped']:+.2f}) >=0.5? {c1['passed']}  [FLAG EI/UCB]")
    print(f"  crit2 Exploit>=EI on d>=6 ({c2['n_exploit_wins']}/{c2['n_total']}): "
          f"majority? {c2['passed']}")
    print(f"  H2a PASS (both criteria): {h2['H2a_PASS']}")
    print(f"  H2b eps-PF (descriptive): M_a={h2['H2b_epsPF_descriptive']['M_a_nsga']:.3f}")
    gc = h2["gradient_context_descriptive"]
    print(f"  [context] grad order: {gc['order_best_to_worst']}  "
          f"Exploit>=EI d>=6: {gc['exploit_at_or_above_ei_highd']}")

    print("\n--- H3: value-vs-gradient (THE CORE) -- reported either way ---")
    h3 = R["H3"]
    print(f"  D_p (mean={h3['mean_D_p']:+.4f}, median={h3['median_D_p']:+.4f}, "
          f"{h3['n_problems_D_p_positive']}/11 positive)")
    pw = h3["primary_wilcoxon_one_sided_greater"]
    state = "CONFIRMED" if pw["H3_CONFIRMED"] else "NOT confirmed"
    print(f"  PRIMARY one-sided Wilcoxon (D_p>0): W={pw['statistic']:.1f}  p={_fmt_p(pw['p'])}  "
          f"-> H3 {state}")
    print(f"  sensitivity: t-test p={_fmt_p(h3['sensitivity_ttest_one_sided']['p'])}  "
          f"sign-test p={_fmt_p(h3['sensitivity_sign_test']['p'])}")
    sec = h3["secondary_seedlevel_mixedlm"]
    if "intercept" in sec:
        print(f"  SECONDARY seed-level MixedLM: intercept={sec['intercept']:+.4f}  "
              f"p1sided={_fmt_p(sec['p_one_sided_in_direction'])}  "
              f"in-dir sig? {sec['significant_in_direction']}")
    dc = h3["descriptive_components"]
    print(f"  [context] mean LogEI advantage: grad={dc['mean_logei_adv_grad']:+.4f}  "
          f"nsga={dc['mean_logei_adv_nsga']:+.4f}")
    print(f"  >>> H3 VERDICT: {h3['H3_VERDICT']}")
    print(f"  {h3['interpretation']}")
    print("\n  per-problem D_p:")
    for p in PROBLEMS:
        print(f"    {p:18s} d={PROBLEMS[p][1]:2d}  D_p={h3['D_p'][p]:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args()
    L, probe, n_cells, n_clipped = load_log_regret()
    R = {
        "meta": {"prereg_tag": "phase3-prereg", "alpha": ALPHA, "epsilon": EPS,
                 "merged_optimal_table": "{**_problem_optimal_values(), **_ext_optimal_values()}",
                 "n_problems": len(PROBLEMS), "n_seeds": N_SEEDS},
        "data": {"n_cells": n_cells, "n_clipped": n_clipped},
        "H1": analyze_h1(L), "H2": analyze_h2(L), "H3": analyze_h3(L),
        "attribution": analyze_attribution(probe),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(R, f, indent=1, default=float)
    if not args.json_only:
        print_summary(R)
    print(f"\nResults JSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
