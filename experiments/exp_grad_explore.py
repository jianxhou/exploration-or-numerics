"""POST-HOC EXPLORATORY (2026-06-18). NOT part of exp09-prereg. Exploratory findings only.

Gradient-probe analysis: tests Ament et al. (2023)'s ACTUAL mechanism -- region-wise
VANISHING GRADIENTS in legacy EI that starve the multi-start gradient optimiser --
using the already-logged but never-analysed gradient probe fields (frac_nonzero_grad,
median_grad_norm, max_grad_norm), EI vs LogEI, across dimension and across the core
(exp_08, what exp_09 reads) + extension (exp_10) matrices.

Gradients are the confound-free EI-vs-LogEI signal: the value thresholds (frac_acqf_zero,
frac_below_tiny, all_restarts_degenerate) are log-sign-confounded for LogEI (probe.py
docstring lines 9-14), whereas gradient availability/magnitude is computed identically
and is comparable across the two arms.

vanishing_grad_frac := 1 - frac_nonzero_grad = fraction of the 64-point candidate pool
with EXACTLY zero gradient norm (frac_nonzero_grad counts grad_norm > 0). Magnitudes
(median/max grad_norm) are reported alongside to catch "near-vanishing but nonzero".

exp_09 is imported READ-ONLY and is never modified.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# Read-only reuse of exp_09's frozen registries (exp_09 is NOT modified).
from exp_09_prereg_analysis import EXT_PROBLEMS, N_SEEDS, TIER2_PROBLEMS  # noqa: E402

ARMS = ("EI", "LogEI")
CORE_DIR = ROOT / "results" / "exp_08_matrix"
EXT_DIR = ROOT / "results" / "exp_10_matrix"
OUT = ROOT / "results" / "exp_grad_explore_summary.json"

# problem -> (dim, source, matrix_dir). Names are distinct (core GSobolLog/RosenbrockLog
# vs ext GSobol/Rosenbrock; Ackley8 unique), so a flat catalog has no clashes.
CATALOG = {}
for _p, (_stem, _dim, _g) in TIER2_PROBLEMS.items():
    CATALOG[_p] = (_dim, "core", CORE_DIR)
for _p, (_stem, _dim, _g) in EXT_PROBLEMS.items():
    CATALOG[_p] = (_dim, "ext", EXT_DIR)


def load_grad(matrix_dir, problem, arm, seed):
    """Read-only: per-iteration gradient stats from one probe log.
    Returns (vanish_frac, median_gn, max_gn) float arrays, or None if absent."""
    f = matrix_dir / f"{problem}__{arm}__s{seed:02d}.json"
    if not f.exists():
        return None
    with open(f) as fh:
        rec = json.load(fh)
    probe = rec.get("probe")
    if not probe:
        return None
    vanish, med, mx = [], [], []
    for e in probe:
        cp = e["candidate_pool"]
        vanish.append(1.0 - cp["frac_nonzero_grad"])
        med.append(cp["median_grad_norm"])
        mx.append(cp["max_grad_norm"])
    return np.asarray(vanish, float), np.asarray(med, float), np.asarray(mx, float)


def _finite(a):
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


def cell_stats(matrix_dir, problem, arm):
    """Aggregate one (problem, arm) over its 30 seeds x all iterations."""
    all_vanish, all_med, all_max = [], [], []
    per_seed_mean_vanish = []
    n_cells = 0
    for s in range(N_SEEDS):
        g = load_grad(matrix_dir, problem, arm, s)
        if g is None:
            continue
        n_cells += 1
        vanish, med, mx = g
        all_vanish.append(vanish)
        all_med.append(med)
        all_max.append(mx)
        per_seed_mean_vanish.append(float(np.mean(vanish)))
    if n_cells == 0:
        return None
    vanish = np.concatenate(all_vanish)
    med = _finite(np.concatenate(all_med))
    mx = _finite(np.concatenate(all_max))
    return {
        "n_cells": n_cells, "n_iter_obs": int(vanish.size),
        # exact-zero-gradient signal (Ament's region-wise vanishing)
        "mean_vanishing_grad_frac": float(vanish.mean()),
        "frac_iters_any_vanishing": float(np.mean(vanish > 0)),
        "max_vanishing_grad_frac": float(vanish.max()),
        "per_seed_mean_vanish": per_seed_mean_vanish,
        # magnitude signal (near-vanishing but nonzero) -- scale is problem-dependent,
        # so compare EI vs LogEI WITHIN a problem only.
        "median_grad_norm": float(np.median(med)) if med.size else float("nan"),
        "q10_median_grad_norm": float(np.percentile(med, 10)) if med.size else float("nan"),
        "min_median_grad_norm": float(med.min()) if med.size else float("nan"),
        "median_max_grad_norm": float(np.median(mx)) if mx.size else float("nan"),
    }


def main():
    stats = {}
    print("loading probe gradient logs (EI + LogEI, core + ext) ...", flush=True)
    for prob, (dim, src, mdir) in CATALOG.items():
        for arm in ARMS:
            st = cell_stats(mdir, prob, arm)
            if st is not None:
                st.update(dim=dim, source=src)
                stats[(prob, arm)] = st

    # ---- STEP 1: dimension story (EI vanishing_grad_frac ordered by dim) ----
    ei_rows = sorted(
        [(p, s["dim"], s["source"], s) for (p, a), s in stats.items() if a == "EI"],
        key=lambda r: (r[1], r[2], r[0]),
    )
    step1 = [
        {"problem": p, "dim": d, "source": src,
         "mean_vanishing_grad_frac": s["mean_vanishing_grad_frac"],
         "frac_iters_any_vanishing": s["frac_iters_any_vanishing"],
         "max_vanishing_grad_frac": s["max_vanishing_grad_frac"],
         "median_grad_norm": s["median_grad_norm"],
         "min_median_grad_norm": s["min_median_grad_norm"]}
        for (p, d, src, s) in ei_rows
    ]
    # dim-group aggregate for EI (does it grow with dim?)
    groups = {"low (d<=2)": [], "mid (d=6)": [], "high-core (d=10)": [], "ext (d=8-10)": []}
    for (p, d, src, s) in ei_rows:
        key = ("ext (d=8-10)" if src == "ext" else
               "low (d<=2)" if d <= 2 else "mid (d=6)" if d == 6 else "high-core (d=10)")
        groups[key].append(s["mean_vanishing_grad_frac"])
    step1_groups = {k: (float(np.mean(v)) if v else None) for k, v in groups.items()}

    # ---- STEP 2: EI vs LogEI head-to-head per problem (paired) ----
    step2 = {}
    for prob in CATALOG:
        if (prob, "EI") not in stats or (prob, "LogEI") not in stats:
            continue
        e, lo = stats[(prob, "EI")], stats[(prob, "LogEI")]
        ev = np.asarray(e["per_seed_mean_vanish"])
        lv = np.asarray(lo["per_seed_mean_vanish"])
        paired = None
        if ev.shape == lv.shape and np.any(ev - lv != 0):
            try:
                w = wilcoxon(ev, lv)
                paired = {"wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue)}
            except ValueError:
                paired = None
        step2[prob] = {
            "dim": e["dim"], "source": e["source"],
            "EI_vanish_frac": e["mean_vanishing_grad_frac"],
            "LogEI_vanish_frac": lo["mean_vanishing_grad_frac"],
            "EI_minus_LogEI_vanish": e["mean_vanishing_grad_frac"] - lo["mean_vanishing_grad_frac"],
            "EI_median_grad_norm": e["median_grad_norm"],
            "LogEI_median_grad_norm": lo["median_grad_norm"],
            "EI_min_median_grad_norm": e["min_median_grad_norm"],
            "LogEI_min_median_grad_norm": lo["min_median_grad_norm"],
            "paired_vanish": paired,
            "EI_grad_healthier_than_LogEI": (
                e["mean_vanishing_grad_frac"] < lo["mean_vanishing_grad_frac"]),
        }

    # ---- STEP 3: GSobol puzzle ----
    step3 = None
    if ("GSobol", "EI") in stats:
        e, lo = stats[("GSobol", "EI")], stats[("GSobol", "LogEI")]
        ei_healthy = e["mean_vanishing_grad_frac"] < 0.01 and e["min_median_grad_norm"] > 0
        step3 = {
            "EI_vanish_frac": e["mean_vanishing_grad_frac"],
            "EI_frac_iters_any_vanishing": e["frac_iters_any_vanishing"],
            "EI_median_grad_norm": e["median_grad_norm"],
            "EI_min_median_grad_norm": e["min_median_grad_norm"],
            "LogEI_vanish_frac": lo["mean_vanishing_grad_frac"],
            "LogEI_median_grad_norm": lo["median_grad_norm"],
            "verdict": ("EI gradients HEALTHY on GSobol (mechanism ABSENT) -> LogEI's win "
                        "is a GENUINE PUZZLE beyond Ament's mechanism"
                        if ei_healthy else
                        "EI gradients VANISH on GSobol (mechanism PRESENT) -> LogEI's win "
                        "is via Ament's mechanism (confirmation)"),
        }

    out = {
        "_banner": "POST-HOC EXPLORATORY (2026-06-18); NOT exp09-prereg; exploratory only.",
        "definition": "vanishing_grad_frac = 1 - frac_nonzero_grad (frac of 64-pt pool "
                      "with EXACTLY zero gradient norm); magnitudes are problem-scale-dependent.",
        "step0_fields_present": True,
        "step1_dimension_story": {"per_problem_EI": step1, "EI_by_dim_group": step1_groups},
        "step2_ei_vs_logei": step2,
        "step3_gsobol_puzzle": step3,
        "all_stats": {f"{p}|{a}": {k: v for k, v in s.items() if k != "per_seed_mean_vanish"}
                      for (p, a), s in stats.items()},
    }
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    _print(step1, step1_groups, step2, step3)
    print(f"\nFull stats -> {OUT}")


def _print(step1, step1_groups, step2, step3):
    print("\n" + "=" * 86)
    print("GRADIENT probe analysis (POST-HOC EXPLORATORY) -- Ament's mechanism: vanishing grads")
    print("=" * 86)

    print("\nSTEP 1 -- EI vanishing_grad_frac by dimension (does it GROW with dim? Ament predicts yes)")
    print(f"  {'problem':20s} {'dim':>3s} {'src':4s} {'mean_vanish':>11s} {'any_vanish':>10s} "
          f"{'med_grad':>10s} {'min_med_grad':>12s}")
    for r in step1:
        print(f"  {r['problem']:20s} {r['dim']:3d} {r['source']:4s} "
              f"{r['mean_vanishing_grad_frac']:11.4f} {r['frac_iters_any_vanishing']:10.3f} "
              f"{r['median_grad_norm']:10.3g} {r['min_median_grad_norm']:12.3g}")
    print("  EI mean vanishing_grad_frac BY DIM GROUP:")
    for k, v in step1_groups.items():
        print(f"    {k:18s} {v:.4f}" if v is not None else f"    {k:18s} n/a")

    print("\nSTEP 2 -- EI vs LogEI gradients per problem (confound-free; lower vanish + higher norm = healthier)")
    print(f"  {'problem':20s} {'dim':>3s} {'EI_vanish':>9s} {'LogEI_van':>9s} {'EI-Log':>8s} | "
          f"{'EI_medGN':>10s} {'LogEI_medGN':>11s} | {'paired p':>9s}  healthier")
    for prob, d in sorted(step2.items(), key=lambda kv: (kv[1]["dim"], kv[1]["source"], kv[0])):
        p = d["paired_vanish"]
        ps = f"{p['wilcoxon_p']:.2g}" if p else "  n/a"
        healthier = "EI" if d["EI_grad_healthier_than_LogEI"] else "LogEI"
        print(f"  {prob:20s} {d['dim']:3d} {d['EI_vanish_frac']:9.4f} {d['LogEI_vanish_frac']:9.4f} "
              f"{d['EI_minus_LogEI_vanish']:+8.4f} | {d['EI_median_grad_norm']:10.3g} "
              f"{d['LogEI_median_grad_norm']:11.3g} | {ps:>9s}  {healthier}")

    if step3:
        print("\nSTEP 3 -- GSobol puzzle (LogEI beat EI; EI showed 0/30 VALUE-underflow). EI gradients?")
        print(f"  EI:    vanish_frac {step3['EI_vanish_frac']:.4f} | any-vanishing iters "
              f"{step3['EI_frac_iters_any_vanishing']:.3f} | median_grad_norm "
              f"{step3['EI_median_grad_norm']:.3g} | min_median_grad_norm "
              f"{step3['EI_min_median_grad_norm']:.3g}")
        print(f"  LogEI: vanish_frac {step3['LogEI_vanish_frac']:.4f} | median_grad_norm "
              f"{step3['LogEI_median_grad_norm']:.3g}")
        print(f"  VERDICT: {step3['verdict']}")


if __name__ == "__main__":
    main()
