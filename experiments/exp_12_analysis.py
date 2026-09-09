"""exp_12 post-hoc analysis (docs/posthoc_protocol.md @ 54fb7e1). Two parts:

  --bootstrap  Problem-level percentile bootstrap CIs (B=10000, 95%) for
               (i) the LogEI-over-EI advantage per optimiser, on the 11-problem
               and 10-De-Ath-problem aggregations, and (ii) the mean H3 contrast
               D_p (11 problems). Uses ONLY the existing frozen matrices via the
               frozen exp_11 loader (read-only). Descriptive, post-hoc.

  --arms       Aggregates the exp_12 post-hoc arms (eps-PF exact front, EI/LogEI
               random search) on the SAME log-precise-regret scale and the SAME
               aggregations as the main matrix (All-10 / high-d / low-d / 11) +
               full per-problem tables. Requires results/exp_12_matrix complete.

Faithful-executor discipline: imports exp_09/exp_11 helpers read-only; no frozen
file is modified; no exp_12 quantity feeds any pre-registered criterion.

Output: results/exp_12_analysis.json (both parts merge into one document).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
import exp_09_prereg_analysis as e9  # noqa: E402  (frozen; read-only)
import exp_11_analysis as A  # noqa: E402  (frozen; read-only)

OUT_PATH = PROJECT_ROOT / "results" / "exp_12_analysis.json"
MATRIX_DIR = PROJECT_ROOT / "results" / "exp_12_matrix"

B = 10_000
CI_LEVEL = 95
BOOT_SEED = 20260701  # fixed, documented; problem-resampling only
N_SEEDS = 30
ARMS_NEW = {"EpsPFExactNSGA2": "eps-PF-exact", "EIRS": "EI-RS", "LogEIRS": "LogEI-RS"}

ALL11 = list(A.PROBLEMS)
DEATH10 = A.H2_SHARED
HIGHD = A.H2_HIGHD  # d>=6 among the De Ath 10
LOWD = [p for p in DEATH10 if A.PROBLEMS[p][1] <= 2]


def _mpa(L: dict) -> dict:
    """Per-(problem, acq, optimizer) mean log precise regret over seeds."""
    out = {}
    for p in ALL11:
        for a in ("EI", "LogEI", "UCB", "Exploit", "eps-PF"):
            for o in ("grad", "nsga"):
                vals = [L[(p, a, o, s)] for s in range(N_SEEDS) if (p, a, o, s) in L]
                if vals:
                    out[(p, a, o)] = float(np.mean(vals))
    return out


def _pctile_ci(stats: np.ndarray) -> list:
    lo = (100 - CI_LEVEL) / 2
    return [float(np.percentile(stats, lo)), float(np.percentile(stats, 100 - lo))]


def analyze_bootstrap() -> dict:
    L, _, _, _ = A.load_log_regret()
    mpa = _mpa(L)
    rng = np.random.default_rng(BOOT_SEED)

    def adv(p, o):  # LogEI advantage over EI: positive = LogEI better (lower regret)
        return mpa[(p, "EI", o)] - mpa[(p, "LogEI", o)]

    out = {"spec": {"B": B, "ci_level": CI_LEVEL, "seed": BOOT_SEED,
                    "method": "percentile bootstrap over problems (resample problems, "
                              "problem-level means fixed)"},
           "logei_advantage": {}, "mean_D_p": {}}

    for label, probs in (("problems11", ALL11), ("death10", DEATH10)):
        n = len(probs)
        for o in ("grad", "nsga"):
            vals = np.array([adv(p, o) for p in probs])
            boots = np.array([vals[rng.integers(n, size=n)].mean() for _ in range(B)])
            out["logei_advantage"][f"{label}_{o}"] = {
                "point": float(vals.mean()), "ci95": _pctile_ci(boots),
                "per_problem": {p: float(adv(p, o)) for p in probs}}

    dvals = np.array([adv(p, "grad") - adv(p, "nsga") for p in ALL11])
    boots = np.array([dvals[rng.integers(len(ALL11), size=len(ALL11))].mean() for _ in range(B)])
    out["mean_D_p"] = {"point": float(dvals.mean()), "ci95": _pctile_ci(boots),
                       "n_problems": len(ALL11)}

    # context: the eps-PF gap the CI is compared against in Section 4.5 (death10, nsga)
    gap = float(np.mean([mpa[(p, "LogEI", "nsga")] - mpa[(p, "eps-PF", "nsga")]
                         for p in DEATH10]))
    out["context_logei_minus_epspf_death10_nsga"] = gap
    return out


def _load_exp12_log_regret() -> tuple[dict, int, int]:
    """Log precise regret for the exp_12 arms, exp_11 conventions (EPS=1e-12, clip at 0)."""
    merged = {**e9._problem_optimal_values(), **e9._ext_optimal_values()}
    L, n_clip = {}, 0
    files = sorted(MATRIX_DIR.glob("*.json"))
    for path in files:
        with open(path) as f:
            r = json.load(f)
        y = np.asarray(r["y"], dtype=np.float64)
        reg = float(merged[r["problem"]] - y.max())
        if reg < 0:
            n_clip += 1
        L[(r["problem"], ARMS_NEW[r["arm"]], r["seed"])] = float(np.log(max(reg, 0.0) + A.EPS))
    return L, len(files), n_clip


def analyze_arms() -> dict:
    L12, n_files, n_clip = _load_exp12_log_regret()
    expected = len(ARMS_NEW) * len(ALL11) * N_SEEDS
    if n_files != expected:
        raise SystemExit(f"exp_12 matrix incomplete: {n_files}/{expected} files")

    Lmain, _, _, _ = A.load_log_regret()
    mpa_main = _mpa(Lmain)

    def mpa12(p, a):
        return float(np.mean([L12[(p, a, s)] for s in range(N_SEEDS)]))

    arms12 = list(ARMS_NEW.values())
    per_problem = {a: {p: mpa12(p, a) for p in ALL11} for a in arms12}

    def se12(p, a):
        v = np.array([L12[(p, a, s)] for s in range(N_SEEDS)])
        return float(v.std(ddof=1) / np.sqrt(len(v)))

    per_problem_se = {a: {p: se12(p, a) for p in ALL11} for a in arms12}
    aggregates = {}
    for label, probs in (("death10", DEATH10), ("highd", HIGHD), ("lowd", LOWD),
                         ("problems11", ALL11)):
        aggregates[label] = {a: float(np.mean([per_problem[a][p] for p in probs]))
                             for a in arms12}
        # main-matrix reference arms on the same aggregation, for side-by-side reporting
        aggregates[label]["ref"] = {
            f"{a}/{o}": float(np.mean([mpa_main[(p, a, o)] for p in probs]))
            for a in ("EI", "LogEI", "UCB", "Exploit", "eps-PF") for o in ("grad", "nsga")}

    # per-problem win/loss: exact-front eps-PF vs the pool-variant eps-PF (nsga)
    exact_vs_pool = {p: {"exact": per_problem["eps-PF-exact"][p],
                         "pool": mpa_main[(p, "eps-PF", "nsga")],
                         "exact_better": bool(per_problem["eps-PF-exact"][p]
                                              < mpa_main[(p, "eps-PF", "nsga")])}
                     for p in ALL11}
    # RS-vs-regime orderings: does EI/LogEI/optimizer ordering replicate under RS?
    rs_order = {}
    for label, probs in (("death10", DEATH10), ("problems11", ALL11)):
        rs_order[label] = {
            "EI_RS": aggregates[label]["EI-RS"], "LogEI_RS": aggregates[label]["LogEI-RS"],
            "logei_adv_rs": float(np.mean([mpa12(p, "EI-RS") - mpa12(p, "LogEI-RS")
                                           for p in probs]))}

    return {"n_files": n_files, "n_negative_clips": n_clip,
            "per_problem_M_pa": per_problem, "per_problem_SE": per_problem_se,
            "aggregates": aggregates,
            "exactfront_vs_pool_per_problem": exact_vs_pool,
            "rs_summary": rs_order}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--arms", action="store_true")
    args = ap.parse_args()
    if not (args.bootstrap or args.arms):
        ap.error("specify --bootstrap and/or --arms")

    doc = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            doc = json.load(f)
    doc.setdefault("protocol", "docs/posthoc_protocol.md@54fb7e1")
    if args.bootstrap:
        doc["bootstrap"] = analyze_bootstrap()
        print(json.dumps(doc["bootstrap"]["logei_advantage"], indent=1))
        print(json.dumps(doc["bootstrap"]["mean_D_p"], indent=1))
    if args.arms:
        doc["arms"] = analyze_arms()
        print(json.dumps(doc["arms"]["aggregates"], indent=1))
    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"written: {OUT_PATH}")


if __name__ == "__main__":
    main()
