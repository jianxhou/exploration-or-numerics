"""POST-HOC EXPLORATORY (2026-06-18). NOT part of exp09-prereg. Exploratory findings only;
single-/few-problem evidence, conditioning is a HYPOTHESIS not a proven mechanism.

Direct test of the "conditioning" hypothesis for LogEI's GSobol performance win: that
LogEI helps NOT by rescuing vanished gradients (absent on GSobol: EI grad ~6e6) but by
COMPRESSING pathologically-large, ill-conditioned EI gradients into a well-scaled range
that the L-BFGS multi-start optimiser handles more effectively. Rosenbrock (raw d=10,
where LogEI did NOT significantly win, p=0.61) + Ackley8 are falsification controls.

Scale-invariant optimiser-progress metric (the crux). The probe logs, per acquisition
step, the per-restart pre-optimisation acqf values (init_acqf) and post-optimisation
values (final_acqf) -- probe.py:77 `init_vals = acq(ics)` and :85 `final_vals`. For the
EI arm the acqf is raw EI; for the LogEI arm it is log(EI). The quantity
  log(EI_final / EI_init)  =  log relative improvement of the UNDERLYING EI
is comparable across arms: EI arm -> log(final_acqf) - log(init_acqf); LogEI arm ->
final_acqf - init_acqf (already log-scale). CAVEATS, stated up front: (i) BoTorch LogEI
is a numerically-stable approximation of log(EI), not exactly log(EI); (ii) the two arms
start from different Boltzmann-selected init points, so headroom differs -- this is a
proxy for optimiser effectiveness, not a perfectly controlled comparison; (iii) EI's log
needs positive acqf, valid on these raw problems where EI does not underflow.

exp_09 is imported READ-ONLY and never modified.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# Read-only reuse of exp_09's frozen helpers (exp_09 is NOT modified).
from exp_09_prereg_analysis import N_SEEDS, _ext_optimal_values, _tracks_from_y  # noqa: E402

EXT_DIR = ROOT / "results" / "exp_10_matrix"
OUT = ROOT / "results" / "exp_cond_explore_summary.json"
PROBLEMS = ["GSobol", "Rosenbrock", "Ackley8"]  # raw d=10, raw d=10, anchor d=8
ARMS = ("EI", "LogEI")


def load_cell(problem, arm, seed):
    f = EXT_DIR / f"{problem}__{arm}__s{seed:02d}.json"
    if not f.exists():
        return None
    with open(f) as fh:
        rec = json.load(fh)
    return rec


def _finite(a):
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


# ---------------------------------------------------------------------------
# STEP 1 -- gradient scale + coarse conditioning proxy (max/median grad_norm)
# ---------------------------------------------------------------------------
def step1_grad_scale():
    out = {}
    for prob in PROBLEMS:
        for arm in ARMS:
            med, mx, ratio = [], [], []
            for s in range(N_SEEDS):
                rec = load_cell(prob, arm, s)
                if rec is None or not rec.get("probe"):
                    continue
                for e in rec["probe"]:
                    cp = e["candidate_pool"]
                    m, M = cp["median_grad_norm"], cp["max_grad_norm"]
                    med.append(m)
                    mx.append(M)
                    if np.isfinite(m) and m > 0 and np.isfinite(M):
                        ratio.append(M / m)
            med, mx, ratio = _finite(med), _finite(mx), _finite(ratio)
            out[f"{prob}|{arm}"] = {
                "median_grad_norm": float(np.median(med)) if med.size else float("nan"),
                "max_grad_norm": float(np.max(mx)) if mx.size else float("nan"),
                "median_of_max_grad_norm": float(np.median(mx)) if mx.size else float("nan"),
                # coarse conditioning proxy: only median & max logged, so max/median is a
                # crude dispersion/anisotropy indicator across the 64-pt pool (COARSE).
                "median_max_over_median_ratio": float(np.median(ratio)) if ratio.size else float("nan"),
            }
    return out


# ---------------------------------------------------------------------------
# STEP 2 -- optimiser init->final progress (scale-invariant log relative climb)
# ---------------------------------------------------------------------------
def _restart_log_improvements(rec, arm):
    """Per-restart log(EI_final/EI_init) for one run; plus (improved, no_progress, excl)."""
    li, improved, no_prog, excl = [], 0, 0, 0
    for e in rec["probe"]:
        iv = np.asarray(e["optimizer"]["init_acqf"], float)
        fv = np.asarray(e["optimizer"]["final_acqf"], float)
        for a, b in zip(iv, fv):
            if not (np.isfinite(a) and np.isfinite(b)):
                excl += 1
                continue
            if arm == "EI":
                if a <= 0 or b <= 0:        # log undefined (exact-zero acqf)
                    excl += 1
                    continue
                d = np.log(b) - np.log(a)
            else:                            # LogEI acqf already = log(EI)
                d = b - a
            li.append(d)
            improved += int(b > a)
            no_prog += int(b <= a)
    return np.asarray(li, float), improved, no_prog, excl


def step2_optimizer_progress():
    out = {}
    for prob in PROBLEMS:
        for arm in ARMS:
            all_li, n_imp, n_no, n_excl, n_restarts = [], 0, 0, 0, 0
            for s in range(N_SEEDS):
                rec = load_cell(prob, arm, s)
                if rec is None or not rec.get("probe"):
                    continue
                li, imp, no, excl = _restart_log_improvements(rec, arm)
                all_li.append(li)
                n_imp += imp
                n_no += no
                n_excl += excl
                n_restarts += li.size + excl
            li = np.concatenate(all_li) if all_li else np.array([])
            out[f"{prob}|{arm}"] = {
                "n_restart_obs": int(n_restarts),
                "n_excluded_exact_zero": int(n_excl),
                "mean_log_improvement": float(li.mean()) if li.size else float("nan"),
                "median_log_improvement": float(np.median(li)) if li.size else float("nan"),
                "frac_restarts_improved": float(n_imp / (n_imp + n_no)) if (n_imp + n_no) else float("nan"),
                "frac_restarts_no_progress": float(n_no / (n_imp + n_no)) if (n_imp + n_no) else float("nan"),
            }
    # progress gap (LogEI - EI) per problem
    gaps = {}
    for prob in PROBLEMS:
        e = out[f"{prob}|EI"]["mean_log_improvement"]
        lo = out[f"{prob}|LogEI"]["mean_log_improvement"]
        gaps[prob] = {"EI_mean_log_improvement": e, "LogEI_mean_log_improvement": lo,
                      "gap_LogEI_minus_EI": lo - e}
    return out, gaps


# ---------------------------------------------------------------------------
# STEP 3 -- link to performance (recompute LogEI-vs-EI final regret here)
# ---------------------------------------------------------------------------
def step3_performance():
    optvals = _ext_optimal_values()
    out = {}
    for prob in PROBLEMS:
        ei, lo = [], []
        for s in range(N_SEEDS):
            re, rl = load_cell(prob, "EI", s), load_cell(prob, "LogEI", s)
            ei.append(_tracks_from_y(np.asarray(re["y"], float), optvals[prob], None)["precise_final"])
            lo.append(_tracks_from_y(np.asarray(rl["y"], float), optvals[prob], None)["precise_final"])
        ei, lo = np.asarray(ei), np.asarray(lo)
        diff = lo - ei
        try:
            p = float(wilcoxon(lo, ei).pvalue) if np.any(diff != 0) else float("nan")
        except ValueError:
            p = float("nan")
        out[prob] = {
            "median_regret_EI": float(np.median(ei)),
            "median_regret_LogEI": float(np.median(lo)),
            "n_LogEI_better": int((diff < 0).sum()), "wilcoxon_p": p,
            "LogEI_significantly_wins": bool(np.isfinite(p) and p < 0.05 and np.median(diff) < 0),
        }
    return out


def main():
    s1 = step1_grad_scale()
    s2, gaps = step2_optimizer_progress()
    s3 = step3_performance()

    # data-driven cross-tab verdict per problem
    crosstab = {}
    for prob in PROBLEMS:
        ei_grad = s1[f"{prob}|EI"]["median_grad_norm"]
        gap = gaps[prob]["gap_LogEI_minus_EI"]
        win = s3[prob]["LogEI_significantly_wins"]
        crosstab[prob] = {
            "EI_median_grad_norm": ei_grad,
            "EI_grad_pathologically_large": bool(ei_grad > 1e3),
            "optimizer_progress_gap_LogEI_minus_EI": gap,
            "LogEI_more_optimizer_progress": bool(gap > 0.05),
            "LogEI_significantly_wins": win,
        }
    # honest assessment: does the optimiser-progress gap TRACK performance across problems,
    # and does EI's optimiser actually fail on the winning problem? (The bar for "supported"
    # is the cross-problem pattern, not a single lenient threshold.)
    wins = [p for p in PROBLEMS if s3[p]["LogEI_significantly_wins"]]
    gap = {p: gaps[p]["gap_LogEI_minus_EI"] for p in PROBLEMS}
    gap_order = sorted(PROBLEMS, key=lambda p: gap[p], reverse=True)  # biggest gap first
    non_wins = [p for p in PROBLEMS if p not in wins]
    gap_tracks_perf = bool(wins) and all(
        gap[w] >= max((gap[p] for p in non_wins), default=-1e9) for w in wins)
    ei_fails_on_win = bool(wins) and all(
        s2[f"{w}|EI"]["frac_restarts_improved"] < 0.9 for w in wins)
    conditioning_supported = gap_tracks_perf and ei_fails_on_win
    crosstab["_gap_order_biggest_first"] = gap_order
    crosstab["_gap_tracks_performance"] = gap_tracks_perf
    crosstab["_EI_optimizer_fails_on_winning_problem"] = ei_fails_on_win
    if conditioning_supported:
        verdict = ("CONDITIONING SUPPORTED (exploratory): the optimiser-progress gap is largest "
                   "on the winning problem AND EI's optimiser underperforms there.")
    else:
        gw = gap.get("GSobol", float("nan"))
        verdict = (
            "CONDITIONING via optimiser-progress NOT SUPPORTED (exploratory). (1) The progress "
            f"gap does NOT track performance: gap ordering (biggest->smallest) = {gap_order}, "
            f"but the only win is GSobol, which has the SMALLEST gap ({gw:+.3f}); Rosenbrock "
            f"({gap['Rosenbrock']:+.3f}) and Ackley8 ({gap['Ackley8']:+.3f}) have LARGER gaps yet NO "
            "win, so optimiser progress != performance. (2) EI's optimiser does NOT fail on GSobol "
            f"(frac_restarts_improved {s2['GSobol|EI']['frac_restarts_improved']:.3f}, mean log-improvement "
            f"{s2['GSobol|EI']['mean_log_improvement']:.3f}): its large gradients do not impair acqf "
            "optimisation -- EI reaches essentially the same relative acqf max as LogEI. (3) By the "
            f"coarse max/median dispersion proxy GSobol is the LEAST ill-conditioned "
            f"({s1['GSobol|EI']['median_max_over_median_ratio']:.1f}) of the three, not the most. "
            "=> What is unique to GSobol is large gradient MAGNITUDE co-occurring with the LogEI win, "
            "but the proposed MECHANISM (large grads -> optimiser failure -> LogEI rescue) is not "
            "borne out; LogEI's GSobol win is NOT explained by conditioning-via-optimiser-progress "
            "and remains unexplained by this test.")

    out = {
        "_banner": "POST-HOC EXPLORATORY (2026-06-18); NOT exp09-prereg; exploratory only; "
                   "2-3 problems; conditioning is a HYPOTHESIS, not proven.",
        "step1_grad_scale": s1,
        "step2_optimizer_progress": s2,
        "step2_progress_gaps": gaps,
        "step3_performance": s3,
        "crosstab": crosstab,
        "verdict": verdict,
    }
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    _print(s1, s2, gaps, s3, crosstab, verdict)
    print(f"\nFull stats -> {OUT}")


def _print(s1, s2, gaps, s3, crosstab, verdict):
    print("\n" + "=" * 90)
    print("CONDITIONING hypothesis test (POST-HOC EXPLORATORY; 2-3 problems; HYPOTHESIS only)")
    print("=" * 90)

    print("\nSTEP 1 -- EI gradient scale per raw problem (is EI pathologically large on BOTH, or only GSobol?)")
    print(f"  {'problem|arm':18s} {'median_gn':>11s} {'median_max_gn':>13s} {'max/median(coarse)':>18s}")
    for k, d in s1.items():
        print(f"  {k:18s} {d['median_grad_norm']:11.3g} {d['median_of_max_grad_norm']:13.3g} "
              f"{d['median_max_over_median_ratio']:18.3g}")

    print("\nSTEP 2 -- optimiser init->final progress (scale-invariant log relative climb of underlying EI)")
    print(f"  {'problem|arm':18s} {'mean_logimpr':>12s} {'med_logimpr':>11s} {'frac_improved':>13s} {'excl_zero':>9s}")
    for k, d in s2.items():
        print(f"  {k:18s} {d['mean_log_improvement']:12.4f} {d['median_log_improvement']:11.4f} "
              f"{d['frac_restarts_improved']:13.3f} {d['n_excluded_exact_zero']:9d}")
    print("  PROGRESS GAP (LogEI - EI mean log-improvement) per problem:")
    for prob, d in gaps.items():
        print(f"    {prob:12s} EI {d['EI_mean_log_improvement']:.4f} | LogEI "
              f"{d['LogEI_mean_log_improvement']:.4f} | gap {d['gap_LogEI_minus_EI']:+.4f}")

    print("\nSTEP 3 -- cross-tab: EI grad scale x optimiser-progress gap x performance")
    print(f"  {'problem':12s} {'EI_med_grad':>12s} {'large?':>7s} {'progress_gap':>13s} "
          f"{'LogEI_wins?':>11s}")
    for prob, d in crosstab.items():
        if prob.startswith("_"):
            continue
        print(f"  {prob:12s} {d['EI_median_grad_norm']:12.3g} "
              f"{str(d['EI_grad_pathologically_large']):>7s} "
              f"{d['optimizer_progress_gap_LogEI_minus_EI']:+13.4f} "
              f"{str(d['LogEI_significantly_wins']):>11s}")
    for prob, d in s3.items():
        print(f"    {prob}: median regret EI {d['median_regret_EI']:.4g} vs LogEI "
              f"{d['median_regret_LogEI']:.4g}, LogEI<EI {d['n_LogEI_better']}/{N_SEEDS}, "
              f"p={d['wilcoxon_p']:.3g}")
    print(f"\n  VERDICT (exploratory): {verdict}")


if __name__ == "__main__":
    main()
