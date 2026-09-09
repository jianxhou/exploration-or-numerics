"""exp_12 post-hoc matrix runner (docs/posthoc_protocol.md, committed 54fb7e1 before data).

Adds the three post-hoc arms declared in the protocol -- EpsPFExactNSGA2 (De Ath's exact
eFront, 330 cells) and EIRS/LogEIRS (random-search acquisition optimiser at the NSGA-II
5000d budget, 660 cells) -- on the same 11 problems x 30 seeds x T=250 as exp_11, written
to results/exp_12_matrix/. 990 cells total. NO mechanism probe (protocol scope control).

Pairing contract (identical to exp_11): De Ath initial-design injection with seed s ->
design run s+1 (Ackley8 Sobol fallback); asserts stems/dims against the frozen exp_09
registries; gate (c) asserts injected designs byte-identical to the De Ath npz AND to the
paired gradient-based cell. Batch machinery reused unchanged (atomic writes,
skip-existing resume, fresh pool per chunk, stall watchdog).

Usage:
  python experiments/run_exp12_posthoc.py --fidelity          # unit fidelity check (no data)
  python experiments/run_exp12_posthoc.py --smoke 6           # first 6 ordered cells
  python experiments/run_exp12_posthoc.py --run --workers 2   # full 990-cell run (resumable)
  python experiments/run_exp12_posthoc.py --gates             # post-run sanity gates
  python experiments/run_exp12_posthoc.py --summarize         # gates + summary JSON
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
MATRIX_DIR = PROJECT_ROOT / "results" / "exp_12_matrix"
SUMMARY_PATH = PROJECT_ROOT / "results" / "exp_12_posthoc_summary.json"
TRAINING_DATA = Path(os.path.expanduser("~/projects/egreedy/training_data"))

T_FULL = 250
N_SEEDS = 30

ARMS = ["EpsPFExactNSGA2", "EIRS", "LogEIRS"]
ARM_ACQ = {"EpsPFExactNSGA2": "eps-PF-exact", "EIRS": "EI", "LogEIRS": "LogEI"}
ARM_OPTIMIZER = {"EpsPFExactNSGA2": "nsga", "EIRS": "rs", "LogEIRS": "rs"}
PROTOCOL = "docs/posthoc_protocol.md@54fb7e1"

# Same 11 problems as exp_11 (asserted against the frozen registries below).
PROBLEMS = {
    "WangFreitas": ("WangFreitas", 1),
    "BraninForrester": ("BraninForrester", 2),
    "Branin": ("Branin", 2),
    "Cosines": ("Cosines", 2),
    "GoldsteinPriceLog": ("logGoldsteinPrice", 2),
    "SixHumpCamelLog": ("logSixHumpCamel", 2),
    "Hartmann6Log": ("logHartmann6", 6),
    "Ackley8": (None, 8),
    "GSobolLog": ("logGSobol", 10),
    "RosenbrockLog": ("logRosenbrock", 10),
    "StyblinskiTangLog": ("logStyblinskiTang", 10),
}
PROBLEM_ORDER = list(PROBLEMS)
GRAD_MATRIX = {p: ("exp_10_matrix" if p == "Ackley8" else "exp_08_matrix") for p in PROBLEMS}

GATE_SAMPLE_CELLS = 22
GATE_RNG_SEED = 20260701


def _assert_contract() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import exp_09_prereg_analysis as e9

    for p, (stem, dim) in PROBLEMS.items():
        ref = e9.EXT_PROBLEMS[p] if p == "Ackley8" else e9.TIER2_PROBLEMS[p]
        assert (ref[0], ref[1]) == (stem, dim), f"{p} stem/dim drift vs frozen registry"
    assert T_FULL == e9.T_FULL and N_SEEDS == e9.N_SEEDS
    assert len(ARMS) * len(PROBLEMS) * N_SEEDS == 990
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
    from al_benchmark.strategies.posthoc_exp12 import EIRS, EpsPFExactNSGA2, LogEIRS

    assert problem_name is not None, "problem_name MUST be set (cross-problem decorrelation)"
    factories = {
        "EpsPFExactNSGA2": lambda: EpsPFExactNSGA2(seed=seed, problem=problem_name),
        "EIRS": lambda: EIRS(seed=seed, problem=problem_name),
        "LogEIRS": lambda: LogEIRS(seed=seed, problem=problem_name),
    }
    inner = factories[arm]()
    assert getattr(inner, "_problem", None) == problem_name, f"problem= not threaded into {arm}"
    assert getattr(inner, "_run_seed", None) == seed, f"seed= not threaded into {arm}"
    return inner


def run_one(task: tuple[str, str, int]) -> dict:
    problem_name, arm, seed = task
    try:
        from al_benchmark.core.bo_loop import run_bo
        from al_benchmark.problems.death import load_death_initial_design

        problem = _make_problem(problem_name)
        strategy = _make_strategy(arm, problem_name, seed)  # NO probe (protocol scope control)
        stem, dim = PROBLEMS[problem_name]
        t0 = time.perf_counter()
        if stem is not None:
            design = load_death_initial_design(stem, seed + 1)
            m = int(design.shape[0])
            design_src = f"deAth:{stem}_{seed + 1}"
            death_design_run = seed + 1
            result = run_bo(problem=problem, strategy=strategy, seed=seed,
                            n_iter=T_FULL - m, initial_design=design)
        else:
            m = 2 * dim
            design_src = "sobol-fallback"
            death_design_run = None
            result = run_bo(problem=problem, strategy=strategy, seed=seed,
                            n_iter=T_FULL - m, initial_design=None)
        wall = time.perf_counter() - t0

        y = result.train_y.squeeze(-1).tolist()
        assert len(y) == T_FULL, f"y length {len(y)} != {T_FULL}"
        record = {
            "arm": arm, "acq": ARM_ACQ[arm], "optimizer": ARM_OPTIMIZER[arm],
            "problem": problem_name, "seed": seed,
            "design_src": design_src, "death_design_run": death_design_run,
            "n_init": int(m), "n_iter": int(T_FULL - m),
            "x": result.train_x.tolist(), "y": y,
            "best_so_far": np.maximum.accumulate(np.asarray(y)).tolist(),
            "wall_time_s": round(wall, 3), "versions": _versions(),
            "posthoc_protocol": PROTOCOL,
        }
        MATRIX_DIR.mkdir(parents=True, exist_ok=True)
        final = _out_path(problem_name, arm, seed)
        with tempfile.NamedTemporaryFile("w", dir=MATRIX_DIR, suffix=".tmp", delete=False) as f:
            json.dump(record, f)
            tmp = f.name
        os.replace(tmp, final)
        return {"ok": True, "task": task, "wall_s": wall}
    except Exception:
        return {"ok": False, "task": task, "error": traceback.format_exc()}


MAX_ROUNDS = 6
CHUNK_SIZE = 40
STALL_TIMEOUT_S = 1800  # exact-front d=10 cells build a 50k-point front per iteration


def _kill_pool(pool: ProcessPoolExecutor) -> None:
    for proc in list(getattr(pool, "_processes", {}).values()):
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    pool.shutdown(wait=False, cancel_futures=True)


def _order(pending: list) -> list:
    prank = {p: i for i, p in enumerate(PROBLEM_ORDER)}
    arank = {a: i for i, a in enumerate(ARMS)}
    return sorted(pending, key=lambda t: (prank[t[0]], t[2], arank[t[1]]))


def cmd_run(n_workers: int) -> None:
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
            print(f"Giving up after {MAX_ROUNDS} rounds; {len(pending)} cells missing:", flush=True)
            for t in pending[:30]:
                print(f"  {t}")
            sys.exit(1)
        done0 = total - len(pending)
        print(f"exp_12 post-hoc matrix round {rounds}: {total} cells total, "
              f"{done0} complete, {len(pending)} to run, {n_workers} workers "
              f"(x{n_threads} threads).", flush=True)
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
                              f"{rate:.2f} runs/s | ETA {eta_s / 60:.0f} min", flush=True)
                        last_print = now
            except BrokenProcessPool:
                interrupted = "pool broke"
            if interrupted:
                print(f"  [{interrupted}; killing pool, recomputing pending]", flush=True)
                _kill_pool(pool)
                break
            pool.shutdown(wait=True)
        if interrupted or failures:
            if failures:
                print(f"  round {rounds}: {len(failures)} failures (retried next round).",
                      flush=True)
            continue
        break

    wall = time.perf_counter() - t_start
    n_files = len(list(MATRIX_DIR.glob("*.json")))
    print(f"\nBATCH COMPLETE: {n_files}/{total} cells in {wall / 3600:.2f} h "
          f"over {rounds} round(s).", flush=True)
    if n_files != total:
        sys.exit(1)


def cmd_smoke(n: int, n_workers: int = 2) -> None:
    tasks = _order([(p, a, s) for p in PROBLEMS for a in ARMS for s in range(N_SEEDS)])[:n]
    n_threads = max(1, (os.cpu_count() or 8) // n_workers)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"exp_12 SMOKE: {len(tasks)} ordered cells -> {MATRIX_DIR}", flush=True)
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
                     and rec["arm"] == t[1] and int(rec["seed"]) == t[2]
                     and rec["acq"] == ARM_ACQ[t[1]]
                     and rec["optimizer"] == ARM_OPTIMIZER[t[1]]
                     and rec["posthoc_protocol"] == PROTOCOL)
            ok &= valid
            print(f"  [ok] {t[0]:18s} {t[1]:16s} s{t[2]} | wall {res['wall_s']:.1f}s | "
                  f"{rec['design_src']} | valid={valid}", flush=True)
    print(f"SMOKE {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        sys.exit(1)


def _load_all() -> dict:
    recs = {}
    for path in sorted(MATRIX_DIR.glob("*.json")):
        with open(path) as f:
            r = json.load(f)
        recs[(r["problem"], r["arm"], int(r["seed"]))] = r
    return recs


def cmd_gates(recs: dict | None = None) -> dict:
    """Protocol gates: (a) 990 files; (b) finite precise final regret + 0 negative
    clips via the merged optimal-value table; (c) injected designs byte-identical to
    the MAPPED De Ath run and the paired gradient-based cell."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import exp_09_prereg_analysis as e9

    if recs is None:
        recs = _load_all()
    merged = {**e9._problem_optimal_values(), **e9._ext_optimal_values()}

    n_expected = len(ARMS) * len(PROBLEMS) * N_SEEDS
    ga = {"n_files": len(recs), "expected": n_expected, "passed": len(recs) == n_expected}

    bad, n_neg = [], 0
    for key, r in recs.items():
        y = np.asarray(r["y"], dtype=np.float64)
        tracks = e9._tracks_from_y(y, merged[r["problem"]], None)
        if not np.isfinite(tracks["precise_final"]):
            bad.append(key)
        n_neg += int(float(y.max()) > float(merged[r["problem"]]))  # raw regret < 0 => clip
    gb = {"n_nonfinite_final": len(bad), "n_negative_clips": n_neg,
          "bad_runs": [list(k) for k in bad[:20]], "passed": not bad}

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
        gpath = PROJECT_ROOT / "results" / GRAD_MATRIX[p] / f"{p}__EI__s{s:02d}.json"
        gref = None
        if gpath.exists():
            with open(gpath) as f:
                gref = np.asarray(json.load(f)["x"], dtype=np.float64)[:m]
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
    gc = {"n_cells_sampled": len(sample_idx), "mismatches": mismatches[:20],
          "passed": not mismatches}

    gates = {"a_file_count": ga, "b_finite_finals": gb, "c_paired_designs": gc}
    gates["all_passed"] = all(g["passed"] for g in (ga, gb, gc))
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
        "config": {"arms": ARMS, "arm_acq": ARM_ACQ, "arm_optimizer": ARM_OPTIMIZER,
                   "problems": list(PROBLEMS), "n_seeds": N_SEEDS, "T_total": T_FULL,
                   "posthoc_protocol": PROTOCOL, "versions": _versions()},
        "gates": gates, "manifest": manifest,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"Summary written to {SUMMARY_PATH} "
          f"(manifest sha256 {manifest['sha256_of_listing'][:16]}...)")


def cmd_fidelity() -> None:
    from al_benchmark.strategies.posthoc_exp12 import fidelity_check

    res = fidelity_check()
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["passed"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", type=int, metavar="N")
    parser.add_argument("--gates", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--fidelity", action="store_true",
                        help="unit fidelity check for the exact-front machinery")
    args = parser.parse_args()

    _assert_contract()
    if args.fidelity:
        cmd_fidelity()
    elif args.smoke is not None:
        cmd_smoke(args.smoke, args.workers)
    elif args.gates:
        gates = cmd_gates()
        sys.exit(0 if gates["all_passed"] else 1)
    elif args.summarize:
        cmd_summarize()
    elif args.run:
        cmd_run(args.workers)
    else:
        parser.error("specify one of --run / --smoke N / --gates / --summarize / --fidelity")


if __name__ == "__main__":
    main()
