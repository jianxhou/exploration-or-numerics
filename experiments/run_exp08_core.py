"""
Experiment 08: Tier 2 core matrix runner.

Executes the frozen core matrix (blueprint Sections 6-10): 8 arms x 10 problems
x 30 seeds, T = 250 total evaluations including the initial design. Initial
designs are De Ath's published training_data, injected via the D8 loader and
shared across all 8 arms per (problem, seed); seed s maps to De Ath design run
s+1 (exp_09 contract). Full trajectories are saved; checkpoints {20, 50, 150}
are extracted at analysis time.

DATA CONTRACT: experiments/exp_09_prereg_analysis.py (tag exp09-prereg) is
frozen and defines the schema. This runner conforms to it -- never the reverse.
Every mode asserts its arm labels, problem keys, and T against exp_09 at
startup. Note the arm label for the max-posterior-std baseline is "Explore"
(exp_09 ARMS), implemented by the Uncertainty strategy class.

Modes:
    python experiments/run_exp08_core.py                # run the matrix (resumable)
    python experiments/run_exp08_core.py --workers 6    # explicit pool size
    python experiments/run_exp08_core.py --gates        # sanity gates (a)-(d)
    python experiments/run_exp08_core.py --summarize    # gates + summary JSON

Outputs:
    results/exp_08_matrix/{problem}__{arm}__s{seed:02d}.json   (local, gitignored)
    results/exp_08_core_summary.json                           (committed)
"""
import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
MATRIX_DIR = PROJECT_ROOT / "results" / "exp_08_matrix"
SUMMARY_PATH = PROJECT_ROOT / "results" / "exp_08_core_summary.json"
TRAINING_DATA = Path(os.path.expanduser("~/projects/egreedy/training_data"))

T_FULL = 250
N_SEEDS = 30
CHECKPOINTS = (20, 50, 150, 250)

# Frozen labels; asserted against exp_09 in every mode before anything runs.
ARMS = ["EI", "LogEI", "UCB", "eps-RS", "eps-PF", "Exploit", "Explore", "Random"]
PROBLEMS = {  # our name -> De Ath training_data stem
    "WangFreitas": "WangFreitas",
    "BraninForrester": "BraninForrester",
    "Branin": "Branin",
    "Cosines": "Cosines",
    "GoldsteinPriceLog": "logGoldsteinPrice",
    "SixHumpCamelLog": "logSixHumpCamel",
    "Hartmann6Log": "logHartmann6",
    "GSobolLog": "logGSobol",
    "RosenbrockLog": "logRosenbrock",
    "StyblinskiTangLog": "logStyblinskiTang",
}

PROBE_ARMS = ("EI", "LogEI")
GATE_SAMPLE_CELLS = 30
GATE_RNG_SEED = 20260612


def _assert_exp09_contract() -> None:
    """Fail closed if this runner drifts from the frozen analysis contract."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import exp_09_prereg_analysis as e9

    assert ARMS == e9.ARMS, f"arm labels diverge from exp_09: {ARMS} vs {e9.ARMS}"
    assert set(PROBLEMS) == set(e9.TIER2_PROBLEMS), "problem keys diverge from exp_09"
    for p, stem in PROBLEMS.items():
        assert e9.TIER2_PROBLEMS[p][0] == stem, f"De Ath stem diverges for {p}"
    assert T_FULL == e9.T_FULL and N_SEEDS == e9.N_SEEDS
    assert len(ARMS) * len(PROBLEMS) * N_SEEDS == e9.N_FILES_EXPECTED


def _out_path(problem: str, arm: str, seed: int) -> Path:
    return MATRIX_DIR / f"{problem}__{arm}__s{seed:02d}.json"


# ----------------------------------------------------------------------------
# Worker (executed in subprocesses; keep imports inside or module-light)
# ----------------------------------------------------------------------------
def _init_worker(n_threads: int) -> None:
    import torch

    warnings.filterwarnings("ignore")
    torch.set_num_threads(n_threads)


def _make_problem(name: str):
    from al_benchmark.problems import death as D
    from al_benchmark.problems.synthetic import Branin

    classes = {
        "WangFreitas": D.WangFreitas, "BraninForrester": D.BraninForrester,
        "Branin": Branin, "Cosines": D.Cosines, "GoldsteinPriceLog": D.GoldsteinPriceLog,
        "SixHumpCamelLog": D.SixHumpCamelLog, "Hartmann6Log": D.Hartmann6Log,
        "GSobolLog": D.GSobolLog, "RosenbrockLog": D.RosenbrockLog,
        "StyblinskiTangLog": D.StyblinskiTangLog,
    }
    return classes[name]()


def _make_strategy(arm: str):
    from al_benchmark.strategies import EI, UCB, EpsPF, EpsRS, Exploit, LogEI, Random, Uncertainty

    factories = {
        "EI": lambda: EI(probe=True),
        "LogEI": lambda: LogEI(probe=True),
        "UCB": lambda: UCB(beta=2.0),
        "eps-RS": lambda: EpsRS(eps=0.1),
        "eps-PF": lambda: EpsPF(eps=0.1),
        "Exploit": lambda: Exploit(),
        "Explore": lambda: Uncertainty(),  # exp_09 label for max posterior std
        "Random": lambda: Random(),
    }
    return factories[arm]()


def _versions() -> dict:
    import botorch
    import gpytorch
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "gpytorch": gpytorch.__version__,
        "numpy": np.__version__,
    }


def run_one(task: tuple[str, str, int]) -> dict:
    """Execute one (problem, arm, seed) cell and write its JSON atomically."""
    problem_name, arm, seed = task
    try:
        from al_benchmark.core.bo_loop import run_bo
        from al_benchmark.problems.death import load_death_initial_design

        problem = _make_problem(problem_name)
        strategy = _make_strategy(arm)
        # exp_09 contract: seed s uses De Ath initial-design run s+1.
        design = load_death_initial_design(PROBLEMS[problem_name], seed + 1)
        m = design.shape[0]
        n_iter = T_FULL - m

        t0 = time.perf_counter()
        result = run_bo(
            problem=problem, strategy=strategy, seed=seed,
            n_iter=n_iter, initial_design=design,
        )
        wall = time.perf_counter() - t0

        y = result.train_y.squeeze(-1).tolist()
        assert len(y) == T_FULL, f"y length {len(y)} != {T_FULL}"
        record = {
            "arm": arm,
            "problem": problem_name,
            "seed": seed,
            "death_design_run": seed + 1,
            "n_init": int(m),
            "n_iter": int(n_iter),
            "x": result.train_x.tolist(),
            "y": y,
            "best_so_far": np.maximum.accumulate(np.asarray(y)).tolist(),
            "wall_time_s": round(wall, 3),
            "versions": _versions(),
            "probe": result.probe,  # list for EI/LogEI, None otherwise
        }

        MATRIX_DIR.mkdir(parents=True, exist_ok=True)
        final = _out_path(problem_name, arm, seed)
        with tempfile.NamedTemporaryFile(
            "w", dir=MATRIX_DIR, suffix=".tmp", delete=False
        ) as f:
            json.dump(record, f)
            tmp = f.name
        os.replace(tmp, final)  # atomic on POSIX
        return {"ok": True, "task": task, "wall_s": wall}
    except Exception:
        return {"ok": False, "task": task, "error": traceback.format_exc()}


# ----------------------------------------------------------------------------
# Run mode
# ----------------------------------------------------------------------------
MAX_ROUNDS = 6           # bounded crash-recovery / retry rounds
CHUNK_SIZE = 40          # fresh pool per chunk: self-managed worker recycling
STALL_TIMEOUT_S = 900    # watchdog: no completion within this window -> rebuild


def _kill_pool(pool: ProcessPoolExecutor) -> None:
    """Hard-teardown a possibly deadlocked pool without joining its threads."""
    for proc in list(getattr(pool, "_processes", {}).values()):
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 -- already-dead workers are fine
            pass
    pool.shutdown(wait=False, cancel_futures=True)


def cmd_run(n_workers: int) -> None:
    """Run the matrix, surviving worker OOM kills and executor deadlocks.

    Worker recycling is self-managed: each ~CHUNK_SIZE-cell chunk gets a fresh
    ProcessPoolExecutor, torn down between chunks. max_tasks_per_child is
    deliberately NOT used -- its mid-stream respawn races a worker death and can
    deadlock the executor without raising BrokenProcessPool (observed on this
    host, CPython 3.11). A stall watchdog (wait(FIRST_COMPLETED), so the window
    resets after every completion) kills the pool if no cell completes within
    STALL_TIMEOUT_S and falls through to the recovery loop, which recomputes
    pending from disk (outputs are atomic; skip-existing) up to MAX_ROUNDS.
    """
    tasks = [
        (p, a, s) for p in PROBLEMS for a in ARMS for s in range(N_SEEDS)
    ]
    total = len(tasks)
    n_threads = max(1, (os.cpu_count() or 8) // n_workers)
    t_start = time.perf_counter()
    rounds = 0
    while True:
        pending = [t for t in tasks if not _out_path(*t).exists()]
        if not pending:
            break
        # Interleave by arm so the heavy probe-carrying EI/LogEI cells never
        # occupy all workers at once (their per-run memory is ~13x the light
        # arms'); peak RSS stays near the steady-state arm mix.
        by_arm = [[t for t in pending if t[1] == a] for a in ARMS]
        pending = [
            t for batch in itertools.zip_longest(*by_arm) for t in batch if t is not None
        ]
        rounds += 1
        if rounds > MAX_ROUNDS:
            print(f"Giving up after {MAX_ROUNDS} rounds; "
                  f"{len(pending)} cells still missing:", flush=True)
            for t in pending[:30]:
                print(f"  {t}")
            sys.exit(1)
        done0 = total - len(pending)
        print(f"exp_08 core matrix round {rounds}: {total} cells total, "
              f"{done0} complete, {len(pending)} to run, {n_workers} workers "
              f"(x{n_threads} threads, fresh pool per {CHUNK_SIZE}-cell chunk, "
              f"stall watchdog {STALL_TIMEOUT_S}s).", flush=True)
        done = 0
        failures = []
        interrupted = None
        t_round = time.perf_counter()
        last_print = t_round
        for start in range(0, len(pending), CHUNK_SIZE):
            chunk = pending[start:start + CHUNK_SIZE]
            pool = ProcessPoolExecutor(
                max_workers=n_workers, initializer=_init_worker, initargs=(n_threads,)
            )
            try:
                not_done = {pool.submit(run_one, t) for t in chunk}
                while not_done:
                    finished, not_done = wait(
                        not_done, timeout=STALL_TIMEOUT_S, return_when=FIRST_COMPLETED
                    )
                    if not finished:
                        interrupted = (f"stall watchdog: no completion in "
                                       f"{STALL_TIMEOUT_S}s")
                        break
                    for fut in finished:
                        res = fut.result()
                        done += 1
                        if not res["ok"]:
                            failures.append(res)
                            print(f"[FAIL] {res['task']}: "
                                  f"{res['error'].splitlines()[-1]}", flush=True)
                    now = time.perf_counter()
                    if done % 25 < len(finished) or (now - last_print) > 120:
                        rate = done / (now - t_round)
                        eta_s = (len(pending) - done) / rate if rate > 0 else float("inf")
                        print(f"  {done0 + done}/{total} done "
                              f"({100 * (done0 + done) / total:.1f}%) | "
                              f"elapsed {(now - t_start) / 60:.1f} min | "
                              f"{rate:.2f} runs/s | ETA {eta_s / 60:.0f} min",
                              flush=True)
                        last_print = now
            except BrokenProcessPool:
                interrupted = "pool broke"
            if interrupted:
                print(f"  [{interrupted} after {done} completions this round; "
                      f"killing pool, recomputing pending from disk]", flush=True)
                _kill_pool(pool)
                break
            pool.shutdown(wait=True)
        if interrupted or failures:
            if failures:
                print(f"  round {rounds}: {len(failures)} task failures "
                      f"(no file written; next round retries them).", flush=True)
            continue
        break

    wall = time.perf_counter() - t_start
    n_files = len(list(MATRIX_DIR.glob("*.json")))
    print(f"\nBATCH COMPLETE: {n_files}/{total} cells on disk "
          f"in {wall / 3600:.2f} h ({wall:.0f} s) over {rounds} round(s).", flush=True)
    if n_files != total:
        sys.exit(1)


# ----------------------------------------------------------------------------
# Gates mode
# ----------------------------------------------------------------------------
def _load_all() -> dict:
    recs = {}
    for path in sorted(MATRIX_DIR.glob("*.json")):
        with open(path) as f:
            r = json.load(f)
        recs[(r["problem"], r["arm"], int(r["seed"]))] = r
    return recs


def cmd_gates(recs: dict | None = None) -> dict:
    """Sanity gates (a)-(d). Returns the gate record; prints PASS/FAIL."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import exp_09_prereg_analysis as e9

    if recs is None:
        recs = _load_all()
    optvals = e9._problem_optimal_values()

    # (a) exactly 2400 files
    n_expected = len(ARMS) * len(PROBLEMS) * N_SEEDS
    ga = {"n_files": len(recs), "expected": n_expected, "passed": len(recs) == n_expected}

    # (b) no NaN final regrets (both tracks, via the frozen track definitions)
    yopts = e9._death_yopt()
    bad = []
    for key, r in recs.items():
        tracks = e9._tracks_from_y(
            np.asarray(r["y"], dtype=np.float64), optvals[r["problem"]],
            yopts.get(r["problem"]),
        )
        if not np.isfinite(tracks["precise_final"]) or not np.isfinite(
            tracks.get("death_final", 0.0)
        ):
            bad.append(key)
    gb = {"n_nonfinite_final": len(bad), "bad_runs": [list(k) for k in bad[:20]],
          "passed": not bad}

    # (c) sampled cells: all 8 arms share identical injected X[:M], byte-identical
    # to De Ath's training_data npz for run seed+1 (exp_09 mapping).
    rng = np.random.default_rng(GATE_RNG_SEED)
    cells = [(p, s) for p in PROBLEMS for s in range(N_SEEDS)]
    sample_idx = rng.choice(len(cells), size=min(GATE_SAMPLE_CELLS, len(cells)),
                            replace=False)
    mismatches = []
    for i in sample_idx:
        p, s = cells[int(i)]
        npz = np.load(TRAINING_DATA / f"{PROBLEMS[p]}_{s + 1}.npz")["arr_0"].astype(np.float64)
        m = npz.shape[0]
        ref = None
        for a in ARMS:
            r = recs.get((p, a, s))
            if r is None:
                mismatches.append((p, s, a, "missing run"))
                continue
            x0 = np.asarray(r["x"], dtype=np.float64)[:m]
            if not np.array_equal(x0, npz):
                mismatches.append((p, s, a, "differs from De Ath npz"))
            if ref is None:
                ref = x0
            elif not np.array_equal(x0, ref):
                mismatches.append((p, s, a, "differs across arms"))
    gc = {"n_cells_sampled": len(sample_idx), "mismatches": mismatches[:20],
          "passed": not mismatches}

    # (d) EI/LogEI probe logs exist with one entry per iteration
    missing_probe = []
    for (p, a, s), r in recs.items():
        if a in PROBE_ARMS:
            probe = r.get("probe")
            if not probe or len(probe) != r["n_iter"]:
                missing_probe.append((p, a, s))
    gd = {"n_probe_runs": sum(1 for k in recs if k[1] in PROBE_ARMS),
          "n_missing_or_short": len(missing_probe),
          "bad": [list(k) for k in missing_probe[:20]], "passed": not missing_probe}

    gates = {"a_file_count": ga, "b_no_nan_finals": gb,
             "c_shared_designs_byte_identical": gc, "d_probe_logs_present": gd}
    gates["all_passed"] = all(g["passed"] for g in (ga, gb, gc, gd))
    for name, g in gates.items():
        if name != "all_passed":
            print(f"  gate {name}: {'PASS' if g['passed'] else 'FAIL'}")
    return gates


# ----------------------------------------------------------------------------
# Summary mode
# ----------------------------------------------------------------------------
def _probe_aggregates(probe: list | None) -> dict | None:
    if not probe:
        return None
    zero = [e["candidate_pool"]["frac_acqf_zero"] for e in probe]
    return {
        "n_iters_zero_fraction_pos": int(sum(z > 0 for z in zero)),
        "max_zero_fraction": float(max(zero)),
        "numerics_warnings_total": int(sum(e["numerics_warnings"] for e in probe)),
        "n_all_restarts_degenerate": int(
            sum(e["optimizer"]["all_restarts_degenerate"] for e in probe)
        ),
    }


def cmd_summarize() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import exp_09_prereg_analysis as e9

    recs = _load_all()
    print(f"Loaded {len(recs)} run files from {MATRIX_DIR}")
    print("Sanity gates:")
    gates = cmd_gates(recs)
    if not gates["all_passed"]:
        sys.exit("Gates FAILED; summary not written.")

    optvals = e9._problem_optimal_values()
    yopts = e9._death_yopt()

    runs_out = []
    for (p, a, s), r in sorted(recs.items()):
        y = np.asarray(r["y"], dtype=np.float64)
        tracks = e9._tracks_from_y(y, optvals[p], yopts.get(p))
        row = {
            "problem": p, "arm": a, "seed": s,
            "wall_time_s": r["wall_time_s"],
            "final_regret_precise": tracks["precise_final"],
            "final_regret_death": tracks.get("death_final"),
        }
        for t in CHECKPOINTS:  # {20, 50, 150, 250}; T250 also exposed as final_*
            row[f"regret_precise_T{t}"] = float(tracks["precise"][t - 1])
            row[f"regret_death_T{t}"] = (
                float(tracks["death"][t - 1]) if tracks.get("death") is not None else None
            )
        row["probe"] = _probe_aggregates(r.get("probe"))
        runs_out.append(row)

    # Per-(problem, arm) zero-fraction aggregates for the probe arms.
    zero_cells = {}
    for row in runs_out:
        if row["probe"] is not None:
            key = (row["problem"], row["arm"])
            zero_cells.setdefault(key, []).append(row["probe"]["max_zero_fraction"])
    probe_cells = [
        {
            "problem": p, "arm": a,
            "median_max_zero_fraction": float(np.median(v)),
            "max_max_zero_fraction": float(np.max(v)),
            "n_runs": len(v),
        }
        for (p, a), v in sorted(zero_cells.items())
    ]
    probe_cells.sort(key=lambda d: d["median_max_zero_fraction"], reverse=True)

    # Manifest: file count + sha256 of the sorted (name, size) listing.
    listing = sorted(
        f"{f.name} {f.stat().st_size}" for f in MATRIX_DIR.glob("*.json")
    )
    manifest = {
        "n_files": len(listing),
        "total_bytes": int(sum(int(line.split()[-1]) for line in listing)),
        "sha256_of_listing": hashlib.sha256("\n".join(listing).encode()).hexdigest(),
    }

    wall_total = float(sum(r["wall_time_s"] for r in recs.values()))
    summary = {
        "config": {
            "arms": ARMS, "problems": list(PROBLEMS), "n_seeds": N_SEEDS,
            "T_total": T_FULL, "checkpoints": list(CHECKPOINTS),
            "initial_designs": "De Ath training_data, seed s -> run s+1 (exp_09 contract)",
            "design_tag": "tier2-design-frozen", "analysis_tag": "exp09-prereg",
            "versions": _versions(),
        },
        "gates": gates,
        "manifest": manifest,
        "wall_time": {
            "sum_of_run_wall_s": wall_total,
            "sum_of_run_wall_h": round(wall_total / 3600, 2),
        },
        "probe_zero_fraction_cells": probe_cells,
        "runs": runs_out,
    }
    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nSummary written to {SUMMARY_PATH} "
          f"({len(runs_out)} runs, manifest sha256 {manifest['sha256_of_listing'][:16]}...)")
    print("Top probe zero-fraction cells (median over seeds of per-run max):")
    for c in probe_cells[:5]:
        print(f"  {c['problem']:18s} {c['arm']:6s} median={c['median_max_zero_fraction']:.4f} "
              f"max={c['max_max_zero_fraction']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gates", action="store_true", help="run sanity gates only")
    parser.add_argument("--summarize", action="store_true",
                        help="run gates and write the committed summary JSON")
    args = parser.parse_args()

    _assert_exp09_contract()
    if args.gates:
        gates = cmd_gates()
        sys.exit(0 if gates["all_passed"] else 1)
    elif args.summarize:
        cmd_summarize()
    else:
        cmd_run(args.workers)


if __name__ == "__main__":
    main()
