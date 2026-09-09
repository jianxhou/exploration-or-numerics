"""
Experiment 06: timing pilot for Tier 2 matrix sizing.

Runs the existing run_bo loop with EI on Branin (d=2) and Ackley (d=10),
2 seeds each, T = 250 total evaluations, and records per-iteration GP-fit
and acquisition-optimization wall times via runtime wrappers (src/ is not
modified). Extrapolates single-process and 6-worker hours for the core
Tier 2 matrix (8 arms x 10 problems x 30 seeds x T=250).

Usage:
    python experiments/exp_06_timing_pilot.py

Outputs:
    results/exp_06_timing_pilot.json
"""
import json
import time
import warnings
from pathlib import Path

from al_benchmark.core.bo_loop import run_bo
from al_benchmark.problems.synthetic import Ackley, Branin
from al_benchmark.strategies.ei import EI
from al_benchmark.surrogates.gp import GPSurrogate

warnings.filterwarnings("ignore")

T_TOTAL = 250
SEEDS = [0, 1]
CHECKPOINTS = [20, 50, 100, 150, 200, 250]  # training-set sizes n
CHECKPOINT_WINDOW = 2                       # average iterations with |n - cp| <= 2

# Core Tier 2 matrix dimensions for the extrapolation.
N_ARMS = 8
N_SEEDS_MATRIX = 30
# De Ath synthetic suite dimensionality split: six problems with d <= 2,
# one with d = 6 (logHartmann6), three with d = 10.
N_D2_PROBLEMS = 6
N_D6_PROBLEMS = 1
N_D10_PROBLEMS = 3

FIT_TIMES: list[float] = []
ACQ_TIMES: list[float] = []

_orig_fit = GPSurrogate.fit
_orig_select = EI.select_next


def _timed_fit(self, *args, **kwargs):
    t0 = time.perf_counter()
    out = _orig_fit(self, *args, **kwargs)
    FIT_TIMES.append(time.perf_counter() - t0)
    return out


def _timed_select(self, *args, **kwargs):
    t0 = time.perf_counter()
    out = _orig_select(self, *args, **kwargs)
    ACQ_TIMES.append(time.perf_counter() - t0)
    return out


GPSurrogate.fit = _timed_fit
EI.select_next = _timed_select


def run_one(problem_factory, seed: int) -> dict:
    FIT_TIMES.clear()
    ACQ_TIMES.clear()
    problem = problem_factory()
    n_init = 2 * problem.dim
    n_iter = T_TOTAL - n_init

    t0 = time.perf_counter()
    result = run_bo(problem=problem, strategy=EI(), seed=seed, n_iter=n_iter)
    total_s = time.perf_counter() - t0

    fit = list(FIT_TIMES)
    acq = list(ACQ_TIMES)
    assert len(fit) == n_iter and len(acq) == n_iter

    # Iteration i (0-based) fits a GP on n_init + i points.
    checkpoints = {}
    for cp in CHECKPOINTS:
        idx = [i for i in range(n_iter)
               if abs((n_init + i) - cp) <= CHECKPOINT_WINDOW]
        if not idx:
            continue
        checkpoints[str(cp)] = {
            "fit_s": sum(fit[i] for i in idx) / len(idx),
            "acq_s": sum(acq[i] for i in idx) / len(idx),
            "iter_s": sum(fit[i] + acq[i] for i in idx) / len(idx),
            "n_averaged": len(idx),
        }

    return {
        "problem": problem.name,
        "dim": problem.dim,
        "seed": seed,
        "n_init": n_init,
        "n_iter": n_iter,
        "total_s": total_s,
        "fit_total_s": sum(fit),
        "acq_total_s": sum(acq),
        "final_regret": result.final_regret,
        "checkpoints": checkpoints,
        "fit_times_s": [round(t, 4) for t in fit],
        "acq_times_s": [round(t, 4) for t in acq],
    }


def main():
    project_root = Path(__file__).parent.parent
    runs = []
    for factory, label in [(Branin, "Branin d=2"), (lambda: Ackley(dim=10), "Ackley d=10")]:
        for seed in SEEDS:
            print(f"Running {label}, seed {seed} (T={T_TOTAL})...", flush=True)
            rec = run_one(factory, seed)
            runs.append(rec)
            print(f"  total {rec['total_s']:.1f}s  "
                  f"(fit {rec['fit_total_s']:.1f}s, acq {rec['acq_total_s']:.1f}s, "
                  f"final regret {rec['final_regret']:.4g})", flush=True)

    print(f"\n{'run':<22} {'total_s':>8} | per-iteration s (fit+acq) at n=")
    header = "".join(f"{cp:>10}" for cp in CHECKPOINTS)
    print(f"{'':<22} {'':>8} | {header}")
    for r in runs:
        cells = "".join(
            f"{r['checkpoints'][str(cp)]['iter_s']:>10.2f}"
            if str(cp) in r["checkpoints"] else f"{'-':>10}"
            for cp in CHECKPOINTS
        )
        print(f"{r['problem'] + ' s' + str(r['seed']):<22} {r['total_s']:>8.1f} | {cells}")

    # Mean per-run wall time per dimensionality.
    t_d2 = sum(r["total_s"] for r in runs if r["dim"] == 2) / len(SEEDS)
    t_d10 = sum(r["total_s"] for r in runs if r["dim"] == 10) / len(SEEDS)
    t_d6 = t_d2 + (6 - 2) / (10 - 2) * (t_d10 - t_d2)  # linear in d

    # Assumptions: every arm costs as much as EI per run (upper bound: the
    # random/greedy arms are cheaper); per-run times include the initial
    # design; 6 workers scale linearly with no contention.
    per_seed_block = (
        N_D2_PROBLEMS * t_d2 + N_D6_PROBLEMS * t_d6 + N_D10_PROBLEMS * t_d10
    )
    single_h = N_ARMS * N_SEEDS_MATRIX * per_seed_block / 3600.0
    six_worker_h = single_h / 6.0
    # Variant with the blueprint's stated 7 d<=2 problems (11 problems total).
    single_h_7d2 = N_ARMS * N_SEEDS_MATRIX * (per_seed_block + t_d2) / 3600.0

    print(f"\nMean per-run wall time: d=2 {t_d2:.1f}s, "
          f"d=6 (interpolated) {t_d6:.1f}s, d=10 {t_d10:.1f}s")
    print(f"Core matrix ({N_ARMS} arms x 10 problems x {N_SEEDS_MATRIX} seeds x "
          f"T={T_TOTAL}, problem split {N_D2_PROBLEMS}/{N_D6_PROBLEMS}/{N_D10_PROBLEMS}):")
    print(f"  single process: {single_h:.1f} h")
    print(f"  6 workers (linear scaling assumed): {six_worker_h:.1f} h")
    print(f"  (with 7 d<=2 problems, 11 total, as in the step-2 spec: "
          f"{single_h_7d2:.1f} h single / {single_h_7d2 / 6:.1f} h on 6 workers)")

    summary = {
        "config": {
            "strategy": "EI",
            "T_total": T_TOTAL,
            "seeds": SEEDS,
            "checkpoints_n": CHECKPOINTS,
            "checkpoint_window": CHECKPOINT_WINDOW,
        },
        "runs": runs,
        "extrapolation": {
            "assumptions": [
                "all 8 arms cost the same per run as EI (upper bound; "
                "random/greedy arms are cheaper)",
                "per-run wall time includes the initial design",
                "d=6 per-run time linearly interpolated in d between "
                "measured d=2 and d=10",
                "problem split 6 x d<=2, 1 x d=6, 3 x d=10 (De Ath synthetic "
                "suite); 7/1/3 variant also reported",
                "6 workers scale linearly with no contention",
            ],
            "mean_run_s": {"d2": t_d2, "d6_interpolated": t_d6, "d10": t_d10},
            "matrix": {"arms": N_ARMS, "problems": 10, "seeds": N_SEEDS_MATRIX},
            "single_process_hours": round(single_h, 2),
            "six_worker_hours": round(six_worker_h, 2),
            "single_process_hours_7d2_variant": round(single_h_7d2, 2),
        },
    }
    out_path = project_root / "results" / "exp_06_timing_pilot.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
