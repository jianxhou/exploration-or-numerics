"""POST-HOC EXPLORATORY ANALYSIS (created 2026-06-18). NOT part of exp09-prereg.
exp_09 is the frozen confirmatory analysis; ALL findings here are exploratory
and must be reported as exploratory in any write-up.

Analyses of the exp_10 extension matrix (results/exp_10_matrix/, 8 arms x 3
problems {GSobol, Rosenbrock, Ackley8} x 30 seeds x T=250):
  (1) per-(problem, arm, checkpoint) PRECISE-regret summary (median+IQR, mean+se);
  (2) EI vs LogEI paired head-to-head at T=250 (Wilcoxon signed-rank, paired by
      shared per-seed initial design);
  (3) underflow probe analysis on the 180 EI/LogEI probe logs -- does vanishing
      acquisition appear in these high-dim RAW problems, and WHEN (early/late/never);
  (4) per-problem 8-arm ranking by final regret, locating Exploit (greedy).

The regret arithmetic is delegated to exp_09's frozen helpers, imported READ-ONLY
(exp_09 is never edited): _tracks_from_y, _ext_optimal_values, _checkpoint_index,
plus the frozen constants CHECKPOINTS / T_FULL / N_SEEDS / ARMS / EXT_PROBLEMS.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# Read-only reuse of exp_09's frozen analysis (exp_09 is NOT modified).
from exp_09_prereg_analysis import (  # noqa: E402
    ARMS,
    CHECKPOINTS,
    EXT_PROBLEMS,
    N_SEEDS,
    T_FULL,
    _checkpoint_index,
    _ext_optimal_values,
    _tracks_from_y,
)

MATRIX_DIR = PROJECT_ROOT / "results" / "exp_10_matrix"
OUT_JSON = PROJECT_ROOT / "results" / "exp_10_explore_summary.json"
PROBE_ARMS = ("EI", "LogEI")
EXT_PROBLEM_NAMES = list(EXT_PROBLEMS)  # {"GSobol", "Rosenbrock", "Ackley8"}


def load_ext_matrix(matrix_dir: Path) -> dict:
    """Faithful read-only copy of the file-reading half of exp_09.load_matrix --
    exp_09 is NOT edited. (exp_09.load_matrix itself cannot be reused: it validates
    `prob not in TIER2_PROBLEMS` and would sys.exit on these extension problems,
    which live in exp_09's additive EXT_PROBLEMS, not the frozen TIER2_PROBLEMS.)
    Regret is still computed by exp_09._tracks_from_y, imported read-only."""
    runs = {}
    files = sorted(matrix_dir.glob("*.json"))
    if not files:
        sys.exit(f"{matrix_dir} contains no *.json run files.")
    for path in files:
        with open(path) as f:
            rec = json.load(f)
        prob, arm, seed = rec["problem"], rec["arm"], int(rec["seed"])
        if prob not in EXT_PROBLEMS:
            sys.exit(f"{path.name}: unknown extension problem {prob!r}")
        if arm not in ARMS:
            sys.exit(f"{path.name}: unknown arm {arm!r}")
        y = np.asarray(rec["y"], dtype=np.float64)
        if y.shape != (T_FULL,):
            sys.exit(f"{path.name}: y must have length {T_FULL}, got {y.shape}")
        runs[(prob, arm, seed)] = rec
    return runs


def regret_at(runs: dict, optvals: dict, prob: str, arm: str, t: int) -> np.ndarray:
    """(N_SEEDS,) PRECISE simple regret at checkpoint t for one (problem, arm)."""
    idx = _checkpoint_index(t)
    out = []
    for s in range(N_SEEDS):
        y = np.asarray(runs[(prob, arm, s)]["y"], dtype=np.float64)
        out.append(_tracks_from_y(y, optvals[prob], None)["precise"][idx])
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# (1) per-(problem, arm, checkpoint) regret summary
# ---------------------------------------------------------------------------
def analysis_1_regret_summary(runs, optvals):
    out = {}
    for prob in EXT_PROBLEM_NAMES:
        out[prob] = {}
        for arm in ARMS:
            out[prob][arm] = {}
            for t in CHECKPOINTS:
                r = regret_at(runs, optvals, prob, arm, t)
                q25, med, q75 = np.percentile(r, [25, 50, 75])
                out[prob][arm][str(t)] = {
                    "median": float(med), "q25": float(q25), "q75": float(q75),
                    "iqr": float(q75 - q25), "mean": float(r.mean()),
                    "se": float(r.std(ddof=1) / np.sqrt(len(r))),
                }
    return out


# ---------------------------------------------------------------------------
# (2) EI vs LogEI paired head-to-head at T=250
# ---------------------------------------------------------------------------
def analysis_2_ei_vs_logei(runs, optvals, t=T_FULL):
    out = {}
    for prob in EXT_PROBLEM_NAMES:
        ei = regret_at(runs, optvals, prob, "EI", t)
        logei = regret_at(runs, optvals, prob, "LogEI", t)
        diff = logei - ei  # > 0 => LogEI has higher regret => LogEI worse
        n_logei_better = int((diff < 0).sum())
        n_ei_better = int((diff > 0).sum())
        n_tie = int((diff == 0).sum())
        try:
            w = wilcoxon(logei, ei, zero_method="wilcox")
            stat, pval = float(w.statistic), float(w.pvalue)
        except ValueError as e:  # e.g. all differences zero
            stat, pval = float("nan"), float("nan")
            print(f"  [wilcoxon {prob}] {e}", flush=True)
        better = ("LogEI" if np.median(diff) < 0 else
                  "EI" if np.median(diff) > 0 else "tie")
        out[prob] = {
            "median_regret_EI": float(np.median(ei)),
            "median_regret_LogEI": float(np.median(logei)),
            "median_diff_LogEI_minus_EI": float(np.median(diff)),
            "mean_diff_LogEI_minus_EI": float(diff.mean()),
            "n_LogEI_better": n_logei_better, "n_EI_better": n_ei_better, "n_tie": n_tie,
            "wilcoxon_stat": stat, "wilcoxon_p": pval,
            "lower_median_regret": better,
            "significant_0.05": bool(pval < 0.05) if np.isfinite(pval) else None,
        }
    return out


# ---------------------------------------------------------------------------
# (3) underflow probe analysis (EI + LogEI x 3 problems x 30 seeds = 180)
# ---------------------------------------------------------------------------
def _underflow_first_iter(probe):
    """First BO iteration (1-based) with candidate-pool acqf-zero underflow, plus
    first all-restarts-degenerate iteration; None if never. frac_acqf_zero is the
    raw-acqf zero fraction -- a genuine underflow signal for EI; for LogEI (log-
    scale outputs) it is confounded with normal late-stage small values (see
    exp_09's low-dim finding), so LogEI is reported separately and flagged."""
    n = len(probe)
    first_zero = None
    first_degen = None
    max_below_tiny = 0.0
    n_iters_zero = 0
    for i, e in enumerate(probe):
        fz = e["candidate_pool"]["frac_acqf_zero"]
        if fz > 0:
            n_iters_zero += 1
            if first_zero is None:
                first_zero = i + 1
        max_below_tiny = max(max_below_tiny, e["candidate_pool"].get("frac_below_tiny", 0.0))
        if e["optimizer"]["all_restarts_degenerate"] and first_degen is None:
            first_degen = i + 1
    return {
        "n_iter": n, "underflow_ever": first_zero is not None,
        "first_underflow_iter": first_zero,
        "first_underflow_frac_of_run": (first_zero / n) if first_zero else None,
        "n_iters_with_zero": n_iters_zero,
        "first_all_restarts_degenerate_iter": first_degen,
        "max_frac_below_tiny": max_below_tiny,
    }


def analysis_3_underflow(runs):
    per_run = {}
    summary = {}
    for prob in EXT_PROBLEM_NAMES:
        for arm in PROBE_ARMS:
            firsts, fracs, ever = [], [], []
            degen_firsts = []
            below_tiny = []
            for s in range(N_SEEDS):
                rec = runs[(prob, arm, s)]
                u = _underflow_first_iter(rec["probe"])
                per_run[f"{prob}|{arm}|s{s}"] = u
                ever.append(u["underflow_ever"])
                below_tiny.append(u["max_frac_below_tiny"])
                if u["underflow_ever"]:
                    firsts.append(u["first_underflow_iter"])
                    fracs.append(u["first_underflow_frac_of_run"])
                if u["first_all_restarts_degenerate_iter"] is not None:
                    degen_firsts.append(u["first_all_restarts_degenerate_iter"])
            frac_runs_underflow = float(np.mean(ever))
            when = "n/a"
            if fracs:
                mf = float(np.median(fracs))
                when = "early (<0.33)" if mf < 0.33 else \
                       "late/near-convergence (>0.66)" if mf > 0.66 else "mid (0.33-0.66)"
            summary[f"{prob}|{arm}"] = {
                "frac_runs_with_underflow": frac_runs_underflow,
                "n_runs_with_underflow": int(np.sum(ever)),
                "median_first_underflow_iter": (float(np.median(firsts)) if firsts else None),
                "median_first_underflow_frac_of_run": (float(np.median(fracs)) if fracs else None),
                "timing_verdict": when,
                "frac_runs_all_restarts_degenerate": float(len(degen_firsts) / N_SEEDS),
                "frac_runs_any_below_tiny": float(np.mean(np.asarray(below_tiny) > 0)),
                "mean_max_frac_below_tiny": float(np.mean(below_tiny)),
                "note": ("frac_acqf_zero/below_tiny/degenerate are genuine underflow "
                         "signals for EI (defined on the raw acqf output)"
                         if arm == "EI" else
                         "LogEI: below_tiny / all_restarts_degenerate are defined on the "
                         "raw acqf = LOG-scale output, so they trip on normal late-stage "
                         "small log-values and are CONFOUNDED (cf. exp_09 low-dim finding); "
                         "NOT a clean underflow signal"),
            }
    return {"per_problem_arm": summary, "per_run": per_run}


# ---------------------------------------------------------------------------
# (4) per-problem 8-arm ranking by final-checkpoint regret
# ---------------------------------------------------------------------------
def analysis_4_ranking(runs, optvals, t=T_FULL):
    per_problem = {}
    rank_of = {a: [] for a in ARMS}  # arm -> list of ranks across problems
    for prob in EXT_PROBLEM_NAMES:
        med = {a: float(np.median(regret_at(runs, optvals, prob, a, t))) for a in ARMS}
        order = sorted(ARMS, key=lambda a: med[a])  # lower regret first
        ranks = {a: i + 1 for i, a in enumerate(order)}
        for a in ARMS:
            rank_of[a].append(ranks[a])
        per_problem[prob] = {
            "median_final_regret": med,
            "ranking_best_to_worst": order,
            "rank": ranks,
            "exploit_rank": ranks["Exploit"],
            "exploit_vs": {x: ("Exploit better" if ranks["Exploit"] < ranks[x]
                               else "Exploit worse" if ranks["Exploit"] > ranks[x]
                               else "tie")
                           for x in ("EI", "LogEI", "UCB", "Random")},
        }
    mean_rank = {a: float(np.mean(rank_of[a])) for a in ARMS}
    overall_order = sorted(ARMS, key=lambda a: mean_rank[a])
    return {
        "per_problem": per_problem,
        "overall_mean_rank": mean_rank,
        "overall_ranking_best_to_worst": overall_order,
        "exploit_overall_mean_rank": mean_rank["Exploit"],
    }


# ---------------------------------------------------------------------------
def main():
    runs = load_ext_matrix(MATRIX_DIR)
    optvals = _ext_optimal_values()
    assert len(runs) == len(ARMS) * len(EXT_PROBLEM_NAMES) * N_SEEDS, \
        f"expected {len(ARMS) * len(EXT_PROBLEM_NAMES) * N_SEEDS} runs, got {len(runs)}"

    a1 = analysis_1_regret_summary(runs, optvals)
    a2 = analysis_2_ei_vs_logei(runs, optvals)
    a3 = analysis_3_underflow(runs)
    a4 = analysis_4_ranking(runs, optvals)

    out = {
        "_banner": "POST-HOC EXPLORATORY, created 2026-06-18; NOT exp09-prereg; "
                   "all findings exploratory.",
        "n_runs": len(runs), "problems": EXT_PROBLEM_NAMES, "arms": ARMS,
        "checkpoints": list(CHECKPOINTS), "optimal_values": optvals,
        "analysis_1_regret_summary": a1,
        "analysis_2_ei_vs_logei_T250": a2,
        "analysis_3_underflow": a3,
        "analysis_4_ranking_T250": a4,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)

    _print_summary(a1, a2, a3, a4)
    print(f"\nFull stats -> {OUT_JSON}")


def _print_summary(a1, a2, a3, a4):
    print("=" * 78)
    print("exp_10 EXPLORATORY analysis (POST-HOC, NOT exp09-prereg; findings exploratory)")
    print("=" * 78)

    print("\n(1) PRECISE simple-regret summary at final checkpoint T=250 "
          "[median (IQR) | mean +/- se]")
    for prob in EXT_PROBLEM_NAMES:
        print(f"  {prob}:")
        for arm in ARMS:
            d = a1[prob][arm]["250"]
            print(f"    {arm:8s}  median {d['median']:11.4g}  IQR {d['iqr']:11.4g}  "
                  f"| mean {d['mean']:11.4g} +/- {d['se']:.3g}")

    print("\n(2) EI vs LogEI at T=250 (paired Wilcoxon; diff = LogEI - EI; "
          "lower regret is better)")
    for prob, d in a2.items():
        sig = "sig" if d["significant_0.05"] else "ns"
        print(f"  {prob:11s}  EI med {d['median_regret_EI']:11.4g} | "
              f"LogEI med {d['median_regret_LogEI']:11.4g} | "
              f"median diff {d['median_diff_LogEI_minus_EI']:+11.4g} | "
              f"LogEI<EI {d['n_LogEI_better']:2d}/{N_SEEDS} | "
              f"W={d['wilcoxon_stat']:.0f} p={d['wilcoxon_p']:.3g} ({sig}) -> "
              f"lower: {d['lower_median_regret']}")

    print("\n(3) Underflow probe: does vanishing acquisition appear in high-dim RAW, "
          "and WHEN?  [pool-zero | below-tiny | all-restarts-degenerate]")
    for key, d in a3["per_problem_arm"].items():
        flag = "  <- CONFOUNDED for LogEI (log-scale acqf)" if key.endswith("LogEI") else ""
        print(f"  {key:20s}  pool-zero {d['n_runs_with_underflow']:2d}/{N_SEEDS} | "
              f"any-below-tiny {int(round(d['frac_runs_any_below_tiny'] * N_SEEDS)):2d}/{N_SEEDS} "
              f"(mean max {d['mean_max_frac_below_tiny']:.3f}) | "
              f"all-restarts-degen {int(round(d['frac_runs_all_restarts_degenerate'] * N_SEEDS)):2d}"
              f"/{N_SEEDS}{flag}")

    print("\n(4) 8-arm ranking by median final regret (T=250); where greedy (Exploit) lands")
    for prob in EXT_PROBLEM_NAMES:
        d = a4["per_problem"][prob]
        print(f"  {prob:11s} best->worst: {' > '.join(d['ranking_best_to_worst'])}")
        print(f"              Exploit rank {d['exploit_rank']}/8  "
              f"(vs EI/LogEI/UCB/Random: "
              f"{', '.join(f'{k}:{v.split()[1]}' for k, v in d['exploit_vs'].items())})")
    print(f"  OVERALL mean-rank best->worst: {' > '.join(a4['overall_ranking_best_to_worst'])}")
    print(f"  Exploit overall mean rank: {a4['exploit_overall_mean_rank']:.2f}/8")


if __name__ == "__main__":
    main()
