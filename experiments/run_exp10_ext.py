"""exp_10 extension matrix runner (POST-HOC; NOT pre-registered).

Independent runner for the Phase-2 extension matrix, kept strictly separate from
the frozen exp_08 core (tag exp09-prereg): raw GSobol + raw Rosenbrock (the C5
raw-vs-log axis) via De Ath initial-design injection, and the Ament anchor
Ackley-d8 (D7) via the Sobol initial-design fallback. 8 arms x 3 problems x 30
seeds x T=250 = 720 cells, written to results/exp_10_matrix/.

CONTRACT. This runner asserts its config against exp_09's ADDITIVE EXT_PROBLEMS /
EXT_N_FILES (added post-hoc 2026-06-17) and additionally guards that the frozen
pre-registered core (TIER2_PROBLEMS=10, N_FILES_EXPECTED=2400) is still untouched.
It never reads or mutates the frozen objects; run_exp08_core's contract gate stays
green (set(PROBLEMS)==set(e9.TIER2_PROBLEMS) is still 10==10).

BATCH MACHINERY (canonical, proven on the core matrix). Fresh ProcessPoolExecutor
per CHUNK_SIZE-cell chunk (self-managed worker recycling); stall watchdog
wait(FIRST_COMPLETED, STALL_TIMEOUT_S) rebuilds the pool on a stall; max_tasks_per_child
deliberately NOT used (mid-stream respawn deadlock race, CPython 3.11); atomic
temp-file writes + skip-existing resume; bounded crash-recovery rounds.

INTERLEAVE (problem x arm) round-robin -- generalises the core matrix's by-arm
zip_longest. The per-arm timing sweep showed slow/fast is problem-dependent, not a
fixed arm property (log problems: acq-opt-bound, exploit slow; raw d=10: GP-fit-bound,
all acq arms ~100-225s; Explore/Random cheaper). Rotating through all (problem,arm)
groups mixes the cheap Ackley-d8 fast-arm cells with the expensive raw-d10 cells so
per-chunk wall variance stays low and the 900s watchdog sits well above the slowest
observed cell (~225s).

Usage:
  python experiments/run_exp10_ext.py --gates             # contract + post-run sanity gates
  python experiments/run_exp10_ext.py --smoke 2           # run the first 2 interleaved cells
  python experiments/run_exp10_ext.py --run --workers 2   # full 720-cell run (resumable)
  python experiments/run_exp10_ext.py --summarize         # gates + summary JSON
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
MATRIX_DIR = PROJECT_ROOT / "results" / "exp_10_matrix"
SUMMARY_PATH = PROJECT_ROOT / "results" / "exp_10_ext_summary.json"
TRAINING_DATA = Path(os.path.expanduser("~/projects/egreedy/training_data"))

T_FULL = 250
N_SEEDS = 30
CHECKPOINTS = (20, 50, 150, 250)

# Frozen 8-arm labels (must equal exp_09.ARMS; asserted before anything runs).
ARMS = ["EI", "LogEI", "UCB", "eps-RS", "eps-PF", "Exploit", "Explore", "Random"]
# our name -> (De Ath training_data stem or None for the Sobol fallback, dim)
PROBLEMS = {
    "GSobol": ("GSobol", 10),          # raw gSobol (C5): De Ath injection (seed s -> run s+1)
    "Rosenbrock": ("Rosenbrock", 10),  # raw Rosenbrock (C5): De Ath injection
    "Ackley8": (None, 8),              # Ament anchor (D7): Sobol fallback (n_init=2*dim)
}

PROBE_ARMS = ("EI", "LogEI")
GATE_SAMPLE_CELLS = 24
GATE_RNG_SEED = 20261017


def _assert_ext_contract() -> None:
    """Fail closed if this runner drifts from exp_09's ADDITIVE extension registry,
    and assert the frozen pre-registered core is still untouched (defense-in-depth
    for exp09-prereg purity)."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import exp_09_prereg_analysis as e9

    assert ARMS == e9.ARMS, f"arm labels diverge from exp_09: {ARMS} vs {e9.ARMS}"
    assert set(PROBLEMS) == set(e9.EXT_PROBLEMS), "problem keys diverge from exp_09.EXT_PROBLEMS"
    for p, (stem, dim) in PROBLEMS.items():
        assert e9.EXT_PROBLEMS[p][0] == stem, f"De Ath stem diverges for {p}"
        assert e9.EXT_PROBLEMS[p][1] == dim, f"dim diverges for {p}"
    assert T_FULL == e9.T_FULL and N_SEEDS == e9.N_SEEDS
    assert len(ARMS) * len(PROBLEMS) * N_SEEDS == e9.EXT_N_FILES
    # frozen-core purity guard: exp_10 must NOT have perturbed the pre-registration.
    assert len(e9.TIER2_PROBLEMS) == 10 and e9.N_FILES_EXPECTED == 2400, \
        "FROZEN CORE MUTATED -- exp09-prereg purity violated"


def _out_path(problem: str, arm: str, seed: int) -> Path:
    return MATRIX_DIR / f"{problem}__{arm}__s{seed:02d}.json"


def _versions() -> dict:
    import botorch
    import gpytorch
    import torch

    return {"python": platform.python_version(), "torch": torch.__version__,
            "botorch": botorch.__version__, "gpytorch": gpytorch.__version__,
            "numpy": np.__version__}


# ----------------------------------------------------------------------------
# Worker (executed in subprocesses; keep imports inside or module-light)
# ----------------------------------------------------------------------------
def _init_worker(n_threads: int) -> None:
    import torch

    warnings.filterwarnings("ignore")
    torch.set_num_threads(n_threads)


def _make_problem(name: str):
    from al_benchmark.problems import death as D
    from al_benchmark.problems.synthetic import Ackley

    classes = {"GSobol": D.GSobol, "Rosenbrock": D.Rosenbrock,
               "Ackley8": lambda: Ackley(dim=8)}
    return classes[name]()


def _make_strategy(arm: str):
    from al_benchmark.strategies import EI, UCB, EpsPF, EpsRS, Exploit, LogEI, Random, Uncertainty

    factories = {
        "EI": lambda: EI(probe=True), "LogEI": lambda: LogEI(probe=True),
        "UCB": lambda: UCB(beta=2.0), "eps-RS": lambda: EpsRS(eps=0.1),
        "eps-PF": lambda: EpsPF(eps=0.1), "Exploit": lambda: Exploit(),
        "Explore": lambda: Uncertainty(), "Random": lambda: Random(),
    }
    return factories[arm]()


def run_one(task: tuple[str, str, int]) -> dict:
    """Execute one (problem, arm, seed) cell and write its JSON atomically."""
    problem_name, arm, seed = task
    try:
        from al_benchmark.core.bo_loop import run_bo
        from al_benchmark.problems.death import load_death_initial_design

        problem = _make_problem(problem_name)
        strategy = _make_strategy(arm)
        stem, dim = PROBLEMS[problem_name]
        t0 = time.perf_counter()
        if stem is not None:
            # De Ath injection (= core path): seed s uses initial-design run s+1.
            design = load_death_initial_design(stem, seed + 1)
            m = int(design.shape[0])
            design_src = f"deAth:{stem}_{seed + 1}"
            death_design_run = seed + 1
            result = run_bo(problem=problem, strategy=strategy, seed=seed,
                            n_iter=T_FULL - m, initial_design=design)
        else:
            # Sobol fallback (D7 Ackley-d8): n_init=2*dim. run_bo seeds the Sobol
            # draw before the strategy, so all 8 arms share one design per seed
            # (verified bitwise); no shared De Ath design exists for this anchor.
            m = 2 * dim
            design_src = "sobol-fallback"
            death_design_run = None
            result = run_bo(problem=problem, strategy=strategy, seed=seed,
                            n_iter=T_FULL - m, initial_design=None)
        wall = time.perf_counter() - t0

        y = result.train_y.squeeze(-1).tolist()
        assert len(y) == T_FULL, f"y length {len(y)} != {T_FULL}"
        record = {
            "arm": arm, "problem": problem_name, "seed": seed,
            "design_src": design_src, "death_design_run": death_design_run,
            "n_init": int(m), "n_iter": int(T_FULL - m),
            "x": result.train_x.tolist(), "y": y,
            "best_so_far": np.maximum.accumulate(np.asarray(y)).tolist(),
            "wall_time_s": round(wall, 3), "versions": _versions(),
            "probe": result.probe,  # list for EI/LogEI, None otherwise
        }
        MATRIX_DIR.mkdir(parents=True, exist_ok=True)
        final = _out_path(problem_name, arm, seed)
        with tempfile.NamedTemporaryFile("w", dir=MATRIX_DIR, suffix=".tmp", delete=False) as f:
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


def _interleave(pending: list) -> list:
    """(problem x arm) round-robin: rotate through all 24 (problem,arm) groups so
    each chunk mixes cheap cells (Ackley-d8 fast arms; Explore/Random) with the
    expensive raw-d10 cells, keeping per-chunk wall variance low."""
    by_pa = [[t for t in pending if t[0] == p and t[1] == a]
             for p in PROBLEMS for a in ARMS]
    return [t for batch in itertools.zip_longest(*by_pa) for t in batch if t is not None]


def cmd_run(n_workers: int) -> None:
    """Run the 720-cell extension matrix, surviving worker OOM kills and executor
    deadlocks (same recovery machinery as the core matrix)."""
    tasks = [(p, a, s) for p in PROBLEMS for a in ARMS for s in range(N_SEEDS)]
    total = len(tasks)
    n_threads = max(1, (os.cpu_count() or 8) // n_workers)
    t_start = time.perf_counter()
    rounds = 0
    while True:
        pending = [t for t in tasks if not _out_path(*t).exists()]
        if not pending:
            break
        pending = _interleave(pending)
        rounds += 1
        if rounds > MAX_ROUNDS:
            print(f"Giving up after {MAX_ROUNDS} rounds; "
                  f"{len(pending)} cells still missing:", flush=True)
            for t in pending[:30]:
                print(f"  {t}")
            sys.exit(1)
        done0 = total - len(pending)
        print(f"exp_10 ext matrix round {rounds}: {total} cells total, "
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


def cmd_smoke(n: int, n_workers: int = 2) -> None:
    """Run the first `n` interleaved cells end-to-end to validate the path before
    locking the full batch. Writes real records to MATRIX_DIR (atomic, skip-existing)."""
    tasks = _interleave([(p, a, s) for p in PROBLEMS for a in ARMS for s in range(N_SEEDS)])[:n]
    n_threads = max(1, (os.cpu_count() or 8) // n_workers)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"exp_10 SMOKE: {len(tasks)} interleaved cells -> {MATRIX_DIR} "
          f"({n_workers} workers x{n_threads} threads)", flush=True)
    ok = True
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(n_threads,)) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for fut in futs:
            res = fut.result()
            t = res["task"]
            if not res["ok"]:
                ok = False
                print(f"  [FAIL] {t}: {res['error'].splitlines()[-1]}", flush=True)
                continue
            with open(_out_path(*t)) as f:
                rec = json.load(f)
            valid = (len(rec["y"]) == T_FULL and rec["problem"] == t[0]
                     and rec["arm"] == t[1] and int(rec["seed"]) == t[2])
            ok &= valid
            print(f"  [ok] {t[0]:10s} {t[1]:8s} s{t[2]} | wall {res['wall_s']:.1f}s | "
                  f"n_init {rec['n_init']} n_iter {rec['n_iter']} | y_len {len(rec['y'])} | "
                  f"{rec['design_src']} | probe={'list' if rec['probe'] else None} | "
                  f"valid={valid}", flush=True)
    print(f"SMOKE {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------------------
# Gates mode (post-run sanity; reuses exp_09 frozen track definitions)
# ----------------------------------------------------------------------------
def _load_all() -> dict:
    recs = {}
    for path in sorted(MATRIX_DIR.glob("*.json")):
        with open(path) as f:
            r = json.load(f)
        recs[(r["problem"], r["arm"], int(r["seed"]))] = r
    return recs


def cmd_gates(recs: dict | None = None) -> dict:
    """Extension sanity gates (a)-(d). Mirrors the core gates but: (b) uses the
    additive _ext_optimal_values (PRECISE only; no DEATH_YOPT for any extension
    problem), and (c) checks byte-identical injected designs for the De Ath-injected
    problems while checking only shared-across-arms for the Sobol-fallback anchor."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import exp_09_prereg_analysis as e9

    if recs is None:
        recs = _load_all()
    optvals = e9._ext_optimal_values()

    # (a) exactly 720 files
    n_expected = len(ARMS) * len(PROBLEMS) * N_SEEDS
    ga = {"n_files": len(recs), "expected": n_expected, "passed": len(recs) == n_expected}

    # (b) no non-finite PRECISE final regrets (extension carries no DEATH track)
    bad = []
    for key, r in recs.items():
        tracks = e9._tracks_from_y(np.asarray(r["y"], dtype=np.float64),
                                   optvals[r["problem"]], None)
        if not np.isfinite(tracks["precise_final"]):
            bad.append(key)
    gb = {"n_nonfinite_final": len(bad), "bad_runs": [list(k) for k in bad[:20]],
          "passed": not bad}

    # (c) sampled cells: all 8 arms share identical injected X[:M]; for the De Ath-
    # injected problems also byte-identical to the training_data npz (run seed+1).
    rng = np.random.default_rng(GATE_RNG_SEED)
    cells = [(p, s) for p in PROBLEMS for s in range(N_SEEDS)]
    sample_idx = rng.choice(len(cells), size=min(GATE_SAMPLE_CELLS, len(cells)), replace=False)
    mismatches = []
    for i in sample_idx:
        p, s = cells[int(i)]
        stem, dim = PROBLEMS[p]
        npz = None
        if stem is not None:
            npz = np.load(TRAINING_DATA / f"{stem}_{s + 1}.npz")["arr_0"].astype(np.float64)
            m = npz.shape[0]
        else:
            m = 2 * dim
        ref = None
        for a in ARMS:
            r = recs.get((p, a, s))
            if r is None:
                mismatches.append((p, s, a, "missing run"))
                continue
            x0 = np.asarray(r["x"], dtype=np.float64)[:m]
            if npz is not None and not np.array_equal(x0, npz):
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
             "c_shared_designs": gc, "d_probe_logs_present": gd}
    gates["all_passed"] = all(g["passed"] for g in (ga, gb, gc, gd))
    for name, g in gates.items():
        if name != "all_passed":
            print(f"  gate {name}: {'PASS' if g['passed'] else 'FAIL'}")
    return gates


def cmd_summarize() -> None:
    recs = _load_all()
    print(f"Loaded {len(recs)} run files from {MATRIX_DIR}")
    gates = cmd_gates(recs)
    if not gates["all_passed"]:
        sys.exit("Gates FAILED; summary not written.")
    listing = sorted(f"{f.name} {f.stat().st_size}" for f in MATRIX_DIR.glob("*.json"))
    manifest = {"n_files": len(listing),
                "sha256_of_listing": hashlib.sha256("\n".join(listing).encode()).hexdigest()}
    summary = {
        "config": {"arms": ARMS, "problems": list(PROBLEMS), "n_seeds": N_SEEDS,
                   "T_total": T_FULL, "checkpoints": list(CHECKPOINTS),
                   "initial_designs": "GSobol/Rosenbrock: De Ath seed s -> run s+1; "
                                      "Ackley8: Sobol fallback n_init=2*dim",
                   "design_tag": "exp10-ext-postdoc", "versions": _versions()},
        "gates": gates, "manifest": manifest,
    }
    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"Summary written to {SUMMARY_PATH} "
          f"(manifest sha256 {manifest['sha256_of_listing'][:16]}...)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run", action="store_true", help="run the full 720-cell matrix")
    parser.add_argument("--smoke", type=int, metavar="N",
                        help="run the first N interleaved cells to validate the path")
    parser.add_argument("--gates", action="store_true", help="run sanity gates only")
    parser.add_argument("--summarize", action="store_true",
                        help="run gates and write the summary JSON")
    args = parser.parse_args()

    _assert_ext_contract()  # also guards that the frozen core is untouched
    if args.smoke is not None:
        cmd_smoke(args.smoke, args.workers)
    elif args.gates:
        gates = cmd_gates()
        sys.exit(0 if gates["all_passed"] else 1)
    elif args.summarize:
        cmd_summarize()
    elif args.run:
        cmd_run(args.workers)
    else:
        parser.error("specify one of --run / --smoke N / --gates / --summarize")


if __name__ == "__main__":
    main()
