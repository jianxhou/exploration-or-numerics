"""exp_11 Phase-3 NSGA-II optimizer-axis matrix runner (pre-registration phase3-prereg).

Independent runner for the FROZEN Phase-3 pre-registration (tag phase3-prereg), kept
strictly separate from the frozen gradient-based matrices (exp_08 core / tag exp09-prereg
and the exp_10 extension): it adds the 5 NSGA-II optimizer arms (EINSGA2, LogEINSGA2,
UCBNSGA2, ExploitNSGA2, EpsPFNSGA2; nsga2.py, committed b3ecf2d) on the pre-reg's 11
in-scope problems x 30 seeds x T=250 = 1650 cells, written to results/exp_11_matrix/.
Each cell pairs by (problem, seed, initial design) with its already-frozen gradient-based
counterpart for the H1/H3 analysis (a separate, post-run step -- NOT performed here).

CONTRACT. This runner asserts its problem stems/dims against exp_09's frozen
TIER2_PROBLEMS (10) and EXT_PROBLEMS (Ackley8) so the De Ath initial designs match the
gradient-based cells byte-for-byte, and guards (defense-in-depth) that the frozen core
(N_FILES_EXPECTED=2400) and extension (EXT_N_FILES=720) are untouched. It never reads or
mutates any frozen object; it writes only to its own exp_11_matrix/.

SEEDING (pre-reg §4.4a -- load-bearing). Every arm is constructed with problem=<name> AND
seed=<cell seed>; the runner ASSERTS both reached the arm (fail-closed) so the per-cell
pymoo seed is the full (problem, acq, optimizer, seed) tuple and the GA streams are
decorrelated across problems -- which the problem-level independence of H1/H3 depends on.

PROBE (pre-reg §5.2 -- FROZEN). Each cell attaches the K=4096 mechanism probe
(al_benchmark.mechanism_probe), logged every BO iteration for every arm: raw-EI value
underflow, raw-LogEI value, own-surface gradient health, optimizer progress. Descriptive
only; it does not change selection and (seeded SobolEngine) does not perturb the design.

BATCH MACHINERY (canonical, proven on the gradient-based matrices, reused unchanged).
Fresh ProcessPoolExecutor per CHUNK_SIZE-cell chunk; stall watchdog
wait(FIRST_COMPLETED, STALL_TIMEOUT_S) rebuilds the pool on a stall; max_tasks_per_child
deliberately NOT used; atomic temp-file writes + skip-existing resume; bounded recovery.

ORDERING (problem-major, seed-major, arm-minor). Whole problems complete one at a time
(dimension ascending: cheap problems first for early validation + early complete problems),
and within a problem each (problem, seed) completes all 5 arms together -- a full pairing
unit. An early stop therefore yields complete problems for the problem-level H1/H3 tests
rather than a scatter of partial ones.

Usage:
  python experiments/run_exp11_nsga2.py --gates             # contract + post-run sanity gates
  python experiments/run_exp11_nsga2.py --smoke 5           # run the first 5 ordered cells
  python experiments/run_exp11_nsga2.py --run --workers 2   # full 1650-cell run (resumable)
  python experiments/run_exp11_nsga2.py --summarize         # gates + summary JSON
"""
import argparse
import hashlib
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
MATRIX_DIR = PROJECT_ROOT / "results" / "exp_11_matrix"
SUMMARY_PATH = PROJECT_ROOT / "results" / "exp_11_nsga2_summary.json"
TRAINING_DATA = Path(os.path.expanduser("~/projects/egreedy/training_data"))

T_FULL = 250
N_SEEDS = 30
CHECKPOINTS = (20, 50, 150, 250)

# 5 NSGA-II arms (Phase 3, pre-reg §4.1); arm string -> gradient-based pairing acq label.
ARMS = ["EINSGA2", "LogEINSGA2", "UCBNSGA2", "ExploitNSGA2", "EpsPFNSGA2"]
ARM_ACQ = {"EINSGA2": "EI", "LogEINSGA2": "LogEI", "UCBNSGA2": "UCB",
           "ExploitNSGA2": "Exploit", "EpsPFNSGA2": "eps-PF"}
OPTIMIZER = "nsga"

# 11 in-scope problems (pre-reg §4.2 name map): our name -> (De Ath stem or None, dim).
# Stems/dims are asserted equal to the frozen registries so designs pair byte-for-byte.
PROBLEMS = {
    "WangFreitas": ("WangFreitas", 1),
    "BraninForrester": ("BraninForrester", 2),
    "Branin": ("Branin", 2),
    "Cosines": ("Cosines", 2),
    "GoldsteinPriceLog": ("logGoldsteinPrice", 2),
    "SixHumpCamelLog": ("logSixHumpCamel", 2),
    "Hartmann6Log": ("logHartmann6", 6),
    "Ackley8": (None, 8),  # Sobol fallback (no De Ath design)
    "GSobolLog": ("logGSobol", 10),
    "RosenbrockLog": ("logRosenbrock", 10),
    "StyblinskiTangLog": ("logStyblinskiTang", 10),
}
# Problem-major order: dimension ascending (cheap first), insertion order within a dim.
PROBLEM_ORDER = list(PROBLEMS)
# Gradient-based matrix each problem pairs against (for the post-run pairing gate).
GRAD_MATRIX = {p: ("exp_10_matrix" if p == "Ackley8" else "exp_08_matrix") for p in PROBLEMS}

GATE_SAMPLE_CELLS = 22
GATE_RNG_SEED = 20260619


def _assert_contract() -> None:
    """Fail closed if exp_11 drifts from the frozen registries it pairs against, and
    guard (defense-in-depth) that the frozen core/extension contracts are untouched."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import exp_09_prereg_analysis as e9

    for p, (stem, dim) in PROBLEMS.items():
        ref = e9.EXT_PROBLEMS[p] if p == "Ackley8" else e9.TIER2_PROBLEMS[p]
        assert (ref[0], ref[1]) == (stem, dim), f"{p} stem/dim drift vs frozen registry: {ref[:2]}"
    assert T_FULL == e9.T_FULL and N_SEEDS == e9.N_SEEDS
    assert len(ARMS) == 5 and len(PROBLEMS) == 11
    assert len(ARMS) * len(PROBLEMS) * N_SEEDS == 1650
    # frozen-purity guard: exp_08/exp_10 contracts must NOT have been perturbed.
    assert len(e9.TIER2_PROBLEMS) == 10 and e9.N_FILES_EXPECTED == 2400, "FROZEN CORE MUTATED"
    assert len(e9.EXT_PROBLEMS) == 3 and e9.EXT_N_FILES == 720, "FROZEN EXTENSION MUTATED"


def _out_path(problem: str, arm: str, seed: int) -> Path:
    return MATRIX_DIR / f"{problem}__{arm}__s{seed:02d}.json"


def _versions() -> dict:
    import botorch
    import gpytorch
    import pymoo
    import torch

    return {"python": platform.python_version(), "torch": torch.__version__,
            "botorch": botorch.__version__, "gpytorch": gpytorch.__version__,
            "numpy": np.__version__, "pymoo": pymoo.__version__}


# ----------------------------------------------------------------------------
# Worker (executed in subprocesses)
# ----------------------------------------------------------------------------
def _init_worker(n_threads: int) -> None:
    import torch

    warnings.filterwarnings("ignore")
    torch.set_num_threads(n_threads)


def _make_problem(name: str):
    from al_benchmark.problems import death as D
    from al_benchmark.problems.synthetic import Ackley, Branin

    classes = {
        "WangFreitas": D.WangFreitas, "BraninForrester": D.BraninForrester,
        "Branin": Branin, "Cosines": D.Cosines, "GoldsteinPriceLog": D.GoldsteinPriceLog,
        "SixHumpCamelLog": D.SixHumpCamelLog, "Hartmann6Log": D.Hartmann6Log,
        "Ackley8": lambda: Ackley(dim=8),
        "GSobolLog": D.GSobolLog, "RosenbrockLog": D.RosenbrockLog,
        "StyblinskiTangLog": D.StyblinskiTangLog,
    }
    return classes[name]()


def _make_strategy(arm: str, problem_name: str, seed: int):
    """Construct the NSGA-II arm with problem= and seed= per §4.4a, and ASSERT both
    reached the arm (fail-closed: a problem=None production cell would silently
    correlate GA streams across problems and is forbidden)."""
    from al_benchmark.strategies import EINSGA2, UCBNSGA2, EpsPFNSGA2, ExploitNSGA2, LogEINSGA2

    assert problem_name is not None, "problem_name MUST be set (cross-problem decorrelation, §4.4a)"
    factories = {
        "EINSGA2": lambda: EINSGA2(seed=seed, problem=problem_name),
        "LogEINSGA2": lambda: LogEINSGA2(seed=seed, problem=problem_name),
        "UCBNSGA2": lambda: UCBNSGA2(seed=seed, problem=problem_name),
        "ExploitNSGA2": lambda: ExploitNSGA2(seed=seed, problem=problem_name),
        "EpsPFNSGA2": lambda: EpsPFNSGA2(seed=seed, problem=problem_name),
    }
    inner = factories[arm]()
    assert getattr(inner, "_problem", None) == problem_name, \
        f"4.4a: problem= not threaded into {arm}"
    assert getattr(inner, "_run_seed", None) == seed, \
        f"4.4a: seed= not threaded into {arm}"
    return inner


def run_one(task: tuple[str, str, int]) -> dict:
    """Execute one (problem, arm, seed) NSGA-II cell and write its JSON atomically."""
    problem_name, arm, seed = task
    try:
        from al_benchmark.core.bo_loop import run_bo
        from al_benchmark.mechanism_probe import MechanismProbedStrategy
        from al_benchmark.problems.death import load_death_initial_design

        problem = _make_problem(problem_name)
        inner = _make_strategy(arm, problem_name, seed)  # asserts problem= and seed= threaded
        strategy = MechanismProbedStrategy(inner, problem.bounds)  # §5.2 K=4096 probe
        stem, dim = PROBLEMS[problem_name]
        t0 = time.perf_counter()
        if stem is not None:
            # De Ath injection (= gradient-based path): seed s uses initial-design run s+1.
            design = load_death_initial_design(stem, seed + 1)
            m = int(design.shape[0])
            design_src = f"deAth:{stem}_{seed + 1}"
            death_design_run = seed + 1
            result = run_bo(problem=problem, strategy=strategy, seed=seed,
                            n_iter=T_FULL - m, initial_design=design)
        else:
            # Sobol fallback (Ackley8): n_init=2*dim, run_bo seeds the draw before the
            # strategy, so the design is byte-identical to the gradient-based cell's.
            m = 2 * dim
            design_src = "sobol-fallback"
            death_design_run = None
            result = run_bo(problem=problem, strategy=strategy, seed=seed,
                            n_iter=T_FULL - m, initial_design=None)
        wall = time.perf_counter() - t0

        y = result.train_y.squeeze(-1).tolist()
        assert len(y) == T_FULL, f"y length {len(y)} != {T_FULL}"
        record = {
            "arm": arm, "acq": ARM_ACQ[arm], "optimizer": OPTIMIZER,
            "problem": problem_name, "seed": seed,
            "design_src": design_src, "death_design_run": death_design_run,
            "n_init": int(m), "n_iter": int(T_FULL - m),
            "x": result.train_x.tolist(), "y": y,
            "best_so_far": np.maximum.accumulate(np.asarray(y)).tolist(),
            "wall_time_s": round(wall, 3), "versions": _versions(),
            "probe": result.probe,  # §5.2 K=4096 series, one entry per iteration (every arm)
            "probe_protocol": "phase3-prereg-5.2-K4096",
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


def _order(pending: list) -> list:
    """Problem-major, seed-major, arm-minor: complete whole problems (dimension
    ascending) and whole (problem, seed) 5-arm pairing units before moving on, so an
    early stop yields complete problems for the problem-level H1/H3 tests."""
    prank = {p: i for i, p in enumerate(PROBLEM_ORDER)}
    arank = {a: i for i, a in enumerate(ARMS)}
    return sorted(pending, key=lambda t: (prank[t[0]], t[2], arank[t[1]]))


def cmd_run(n_workers: int) -> None:
    """Run the 1650-cell NSGA-II matrix, surviving worker OOM kills and executor
    deadlocks (same recovery machinery as the gradient-based matrices)."""
    tasks = [(p, a, s) for p in PROBLEMS for a in ARMS for s in range(N_SEEDS)]
    total = len(tasks)
    n_threads = max(1, (os.cpu_count() or 8) // n_workers)
    t_start = time.perf_counter()
    rounds = 0
    while True:
        pending = [t for t in tasks if not _out_path(*t).exists()]
        if not pending:
            break
        pending = _order(pending)
        rounds += 1
        if rounds > MAX_ROUNDS:
            print(f"Giving up after {MAX_ROUNDS} rounds; "
                  f"{len(pending)} cells still missing:", flush=True)
            for t in pending[:30]:
                print(f"  {t}")
            sys.exit(1)
        done0 = total - len(pending)
        print(f"exp_11 NSGA-II matrix round {rounds}: {total} cells total, "
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
                        interrupted = f"stall watchdog: no completion in {STALL_TIMEOUT_S}s"
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
    """Run the first `n` ordered cells end-to-end to validate the path before the full
    batch. Writes real records to MATRIX_DIR (atomic, skip-existing)."""
    tasks = _order([(p, a, s) for p in PROBLEMS for a in ARMS for s in range(N_SEEDS)])[:n]
    n_threads = max(1, (os.cpu_count() or 8) // n_workers)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"exp_11 SMOKE: {len(tasks)} ordered cells -> {MATRIX_DIR} "
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
            probe = rec["probe"]
            valid = (len(rec["y"]) == T_FULL and rec["problem"] == t[0]
                     and rec["arm"] == t[1] and int(rec["seed"]) == t[2]
                     and rec["optimizer"] == OPTIMIZER and rec["acq"] == ARM_ACQ[t[1]]
                     and isinstance(probe, list) and len(probe) == rec["n_iter"])
            ok &= valid
            print(f"  [ok] {t[0]:18s} {t[1]:13s} s{t[2]} | wall {res['wall_s']:.1f}s | "
                  f"n_init {rec['n_init']} | y_len {len(rec['y'])} | {rec['design_src']} | "
                  f"acq={rec['acq']}/{rec['optimizer']} | probe_len={len(probe)} | "
                  f"valid={valid}", flush=True)
    print(f"SMOKE {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------------------
# Gates mode (post-run sanity; merged optimal-value table for precise regret)
# ----------------------------------------------------------------------------
def _load_all() -> dict:
    recs = {}
    for path in sorted(MATRIX_DIR.glob("*.json")):
        with open(path) as f:
            r = json.load(f)
        recs[(r["problem"], r["arm"], int(r["seed"]))] = r
    return recs


def cmd_gates(recs: dict | None = None) -> dict:
    """Sanity gates (a)-(d): file count; finite PRECISE final regret via the merged
    optimal-value table; injected designs byte-identical to the paired gradient-based
    cells; §5.2 probe present (one entry per iteration, every arm)."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import exp_09_prereg_analysis as e9

    if recs is None:
        recs = _load_all()
    merged = {**e9._problem_optimal_values(), **e9._ext_optimal_values()}  # F1 merged table

    # (a) exactly 1650 files
    n_expected = len(ARMS) * len(PROBLEMS) * N_SEEDS
    ga = {"n_files": len(recs), "expected": n_expected, "passed": len(recs) == n_expected}

    # (b) no non-finite PRECISE final regrets via the merged table
    bad = []
    for key, r in recs.items():
        tracks = e9._tracks_from_y(np.asarray(r["y"], dtype=np.float64), merged[r["problem"]], None)
        if not np.isfinite(tracks["precise_final"]):
            bad.append(key)
    gb = {"n_nonfinite_final": len(bad), "bad_runs": [list(k) for k in bad[:20]], "passed": not bad}

    # (c) sampled cells: injected X[:M] byte-identical across the 5 arms, to the De Ath
    # npz (stem problems), AND to the paired gradient-based cell (the pairing guarantee).
    rng = np.random.default_rng(GATE_RNG_SEED)
    cells = [(p, s) for p in PROBLEMS for s in range(N_SEEDS)]
    sample_idx = rng.choice(len(cells), size=min(GATE_SAMPLE_CELLS, len(cells)), replace=False)
    mismatches = []
    for i in sample_idx:
        p, s = cells[int(i)]
        stem, dim = PROBLEMS[p]
        if stem is not None:
            ref = np.load(TRAINING_DATA / f"{stem}_{s + 1}.npz")["arr_0"].astype(np.float64)
            m = ref.shape[0]
        else:
            ref, m = None, 2 * dim
        # paired gradient-based design (EI arm of the same problem/seed)
        gpath = PROJECT_ROOT / "results" / GRAD_MATRIX[p] / f"{p}__EI__s{s:02d}.json"
        gref = None
        if gpath.exists():
            with open(gpath) as f:
                gref = np.asarray(json.load(f)["x"], dtype=np.float64)[:m]
        first = None
        for a in ARMS:
            r = recs.get((p, a, s))
            if r is None:
                mismatches.append((p, s, a, "missing run"))
                continue
            x0 = np.asarray(r["x"], dtype=np.float64)[:m]
            if ref is not None and not np.array_equal(x0, ref):
                mismatches.append((p, s, a, "differs from De Ath npz"))
            if gref is not None and not np.array_equal(x0, gref):
                mismatches.append((p, s, a, "differs from gradient-based pair"))
            if first is None:
                first = x0
            elif not np.array_equal(x0, first):
                mismatches.append((p, s, a, "differs across arms"))
    gc = {"n_cells_sampled": len(sample_idx), "mismatches": mismatches[:20],
          "passed": not mismatches}

    # (d) §5.2 probe present for every arm: one entry per iteration with the frozen fields
    req = {"raw_ei_frac_zero", "raw_logei_max", "own_acq", "own_grad", "progress"}
    bad_probe = []
    for (p, a, s), r in recs.items():
        probe = r.get("probe")
        if not isinstance(probe, list) or len(probe) != r["n_iter"]:
            bad_probe.append((p, a, s, "missing/short"))
        elif probe and not req.issubset(probe[0]):
            bad_probe.append((p, a, s, "fields"))
    gd = {"n_runs": len(recs), "n_bad_probe": len(bad_probe),
          "bad": [list(k) for k in bad_probe[:20]], "passed": not bad_probe}

    gates = {"a_file_count": ga, "b_finite_finals": gb,
             "c_paired_designs": gc, "d_probe_present": gd}
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
        "config": {"arms": ARMS, "arm_acq": ARM_ACQ, "optimizer": OPTIMIZER,
                   "problems": list(PROBLEMS), "n_seeds": N_SEEDS, "T_total": T_FULL,
                   "checkpoints": list(CHECKPOINTS), "probe_protocol": "phase3-prereg-5.2-K4096",
                   "prereg_tag": "phase3-prereg", "versions": _versions()},
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
    parser.add_argument("--run", action="store_true", help="run the full 1650-cell matrix")
    parser.add_argument("--smoke", type=int, metavar="N",
                        help="run the first N ordered cells to validate the path")
    parser.add_argument("--gates", action="store_true", help="run sanity gates only")
    parser.add_argument("--summarize", action="store_true", help="run gates, write summary JSON")
    args = parser.parse_args()

    _assert_contract()  # ties stems/dims to the frozen registries + guards their purity
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
