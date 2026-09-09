"""exp_10 first-wave timing pilot (scratch; not committed pending review).

Blueprint Sec 91 mandatory pilot: {Branin, logGSobol} x {EI, Exploit} x 3 seeds,
full T=250, plus the Ament anchor Ackley-d8 (D7) x {EI, Exploit} x 2 seeds.
Branin/logGSobol use De Ath injection (= core matrix path); Ackley-d8 uses the
Sobol fallback (initial_design=None -> n_init=2*dim=16). Reports per-cell wall
time and peak RSS so the first-wave worker/chunk/total-time can be sized.

Each cell runs in a fresh spawned process (max_tasks_per_child=1) so ru_maxrss
is a clean per-cell peak; 2 workers = canonical batch config. NOTE: the pilot
sets max_tasks_per_child=1 only to get clean per-cell RSS -- the production
orchestrator deliberately omits it (deadlock race on long runs).
"""
import argparse
import os
import platform
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

T_FULL = 250

# pilot problem name -> (De Ath stem or None for Sobol fallback, dim)
PILOT = {
    "Branin": ("Branin", 2),          # De Ath injection (core path)
    "logGSobol": ("logGSobol", 10),   # De Ath injection (core path)
    "GSobol": ("GSobol", 10),         # raw GSobol (D3): De Ath injection, no DEATH_YOPT
    "Rosenbrock": ("Rosenbrock", 10),  # raw Rosenbrock (D3): De Ath injection, no DEATH_YOPT
    "Ackley8": (None, 8),             # D7 anchor: Sobol fallback
}

ARMS = ["EI", "LogEI", "UCB", "eps-RS", "eps-PF", "Exploit", "Explore", "Random"]


def _make_problem(name):
    from al_benchmark.problems import death as D
    from al_benchmark.problems.synthetic import Ackley, Branin
    if name == "Branin":
        return Branin()
    if name == "logGSobol":
        return D.GSobolLog()
    if name == "GSobol":
        return D.GSobol()
    if name == "Rosenbrock":
        return D.Rosenbrock()
    if name == "Ackley8":
        return Ackley(dim=8)
    raise KeyError(name)


def _make_strategy(arm):
    from al_benchmark.strategies import EI, UCB, EpsPF, EpsRS, Exploit, LogEI, Random, Uncertainty
    return {
        "EI": lambda: EI(probe=True), "LogEI": lambda: LogEI(probe=True),
        "UCB": lambda: UCB(beta=2.0), "eps-RS": lambda: EpsRS(eps=0.1),
        "eps-PF": lambda: EpsPF(eps=0.1), "Exploit": lambda: Exploit(),
        "Explore": lambda: Uncertainty(), "Random": lambda: Random(),
    }[arm]()


def run_cell(task):
    name, arm, seed = task
    import warnings

    import torch
    warnings.filterwarnings("ignore")
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    from al_benchmark.core.bo_loop import run_bo
    from al_benchmark.problems.death import load_death_initial_design

    stem, dim = PILOT[name]
    prob = _make_problem(name)
    strat = _make_strategy(arm)
    t0 = time.perf_counter()
    if stem is not None:
        design = load_death_initial_design(stem, seed + 1)
        m = int(design.shape[0])
        r = run_bo(prob, strat, seed=seed, n_iter=T_FULL - m, initial_design=design)
        design_src = f"deAth:{stem}_{seed + 1}"
    else:
        m = 2 * dim
        r = run_bo(prob, strat, seed=seed, n_iter=T_FULL - m, initial_design=None)
        design_src = "sobol-fallback"
    wall = time.perf_counter() - t0

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024
    n_y = len(r.train_y)
    probe_len = len(r.probe) if r.probe is not None else 0
    return dict(
        name=name, arm=arm, seed=seed, n_init=m, n_iter=T_FULL - m, n_y=n_y,
        design_src=design_src, wall_s=round(wall, 2), peak_mb=round(peak_mb, 1),
        probe_len=probe_len, final_regret=float(r.final_regret),
    )


def run_cell_capture(task):
    """Stability variant of run_cell: CAPTURE warnings (not suppress), check
    NaN/Inf, report GP-fit numerical diagnostics. For the raw-Rosenbrock
    numerical-stability spot-check before locking the matrix."""
    import traceback
    import warnings as _w

    import numpy as np
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    from al_benchmark.core.bo_loop import run_bo
    from al_benchmark.problems.death import load_death_initial_design

    name, arm, seed = task
    stem, dim = PILOT[name]
    prob = _make_problem(name)
    strat = _make_strategy(arm)
    err = None
    r = None
    t0 = time.perf_counter()
    with _w.catch_warnings(record=True) as wlist:
        _w.simplefilter("always")
        try:
            if stem is not None:
                design = load_death_initial_design(stem, seed + 1)
                m = int(design.shape[0])
                r = run_bo(prob, strat, seed=seed, n_iter=T_FULL - m, initial_design=design)
                src = f"deAth:{stem}_{seed + 1}"
            else:
                m = 2 * dim
                r = run_bo(prob, strat, seed=seed, n_iter=T_FULL - m, initial_design=None)
                src = "sobol-fallback"
        except Exception:
            err = traceback.format_exc()
            src = stem or "sobol-fallback"
            m = int(design.shape[0]) if stem is not None else 2 * dim
        captured = [(w.category.__name__, str(w.message)) for w in wlist]
    wall = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024

    wsum = {}
    for cat, msg in captured:
        key = (cat, msg[:90])
        wsum[key] = wsum.get(key, 0) + 1

    out = dict(name=name, arm=arm, seed=seed, n_init=m, design_src=src,
               wall_s=round(wall, 2), peak_mb=round(peak_mb, 1),
               n_warnings=len(captured), warn_summary=sorted(wsum.items()), error=err)
    if r is not None:
        y = np.asarray(r.train_y.squeeze(-1).tolist(), dtype=np.float64)
        out.update(n_y=len(y), y_has_nan=bool(np.isnan(y).any()),
                   y_has_inf=bool(np.isinf(y).any()), y_min=float(y.min()),
                   y_max=float(y.max()), optimal_value=float(prob.optimal_value),
                   final_regret=float(r.final_regret), best_final=float(y.max()))
    return out


_FLAGGED = ("cholesky", "jitter", "not p.d", "singular", "nan", "inf", "converg",
            "numerical", "optimization", "psd", "ill-conditioned", "linalg")


def cmd_stability():
    # raw Rosenbrock GP-fit numerical-stability spot-check (+ Ackley-d8 EI/Exploit fill).
    tasks = [(p, a, 0) for p in ("Rosenbrock", "Ackley8") for a in ("EI", "Exploit")]
    print(f"exp_10 stability spot-check: {len(tasks)} cells | fresh proc/cell | T={T_FULL} | "
          "WARNINGS CAPTURED (not suppressed)", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_cell_capture, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = "ERROR" if r.get("error") else "ok"
            print(f"\n--- {r['name']} {r['arm']} s{r['seed']} [{tag}] | wall {r['wall_s']}s | "
                  f"peakRSS {r['peak_mb']}MB | {r['design_src']} ---", flush=True)
            if r.get("error"):
                print("  EXCEPTION:\n   " + r["error"].replace("\n", "\n   "), flush=True)
            else:
                print(f"  y: len {r['n_y']} | NaN={r['y_has_nan']} Inf={r['y_has_inf']} | "
                      f"min {r['y_min']:.4g} max {r['y_max']:.4g}", flush=True)
                print(f"  regret: optimal_value {r['optimal_value']:.4g} | "
                      f"best_final {r['best_final']:.4g} | final_regret {r['final_regret']:.4g}",
                      flush=True)
            print(f"  warnings: {r['n_warnings']} total, "
                  f"{len(r['warn_summary'])} distinct", flush=True)
            for (cat, msg), cnt in r["warn_summary"]:
                hit = any(f in (cat + " " + msg).lower() for f in _FLAGGED)
                mark = "   <-- NUMERIC FLAG" if hit else ""
                print(f"    [{cnt:4d}x] {cat}: {msg}{mark}", flush=True)

    rosen = [r for r in results if r["name"] == "Rosenbrock"]
    bad = [r for r in rosen if r.get("error") or r.get("y_has_nan") or r.get("y_has_inf")]
    flagged = [r for r in rosen if any(
        any(f in (cat + " " + msg).lower() for f in _FLAGGED)
        for (cat, msg), _ in r["warn_summary"])]
    print("\n=== ROSENBROCK GP-FIT STABILITY VERDICT ===", flush=True)
    print(f"  cells with error/NaN/Inf: {len(bad)} | cells with flagged numeric warnings: "
          f"{len(flagged)}", flush=True)
    verdict = ("UNSTABLE -- investigate (jitter/numeric params/optimal_value) before orchestrator"
               if (bad or flagged) else
               "STABLE -- no errors, no NaN/Inf, no flagged numeric warnings")
    print(f"  -> {verdict}", flush=True)


def run_tasks(tasks, label):
    print(f"{label}: {len(tasks)} cells | 2 workers | fresh proc/cell | "
          f"T={T_FULL} | host {platform.system()} cpu={os.cpu_count()}", flush=True)
    t_start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_cell, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            assert r["n_y"] == T_FULL, f"{r}: y len != {T_FULL}"
            results.append(r)
            print(f"  [{len(results):2d}/{len(tasks)}] {r['name']:10s} {r['arm']:8s} "
                  f"s{r['seed']} | wall {r['wall_s']:7.2f}s | peakRSS {r['peak_mb']:7.1f}MB | "
                  f"n_init {r['n_init']:2d} n_iter {r['n_iter']:3d} | {r['design_src']:16s} | "
                  f"regret {r['final_regret']:.4g}", flush=True)
    wall_total = time.perf_counter() - t_start

    print("\n=== per (problem, arm): mean wall / mean peak RSS ===", flush=True)
    agg = {}
    for r in results:
        agg.setdefault((r["name"], r["arm"]), []).append(r)
    for (name, arm), rs in sorted(agg.items()):
        ws = [r["wall_s"] for r in rs]
        ps = [r["peak_mb"] for r in rs]
        print(f"  {name:10s} {arm:8s} | n={len(rs)} | wall mean {sum(ws)/len(ws):7.2f}s "
              f"(min {min(ws):.1f}/max {max(ws):.1f}) | peakRSS mean {sum(ps)/len(ps):7.1f}MB "
              f"(max {max(ps):.1f})", flush=True)
    peak_max = max(r["peak_mb"] for r in results)
    print(f"\nmax per-cell peak RSS {peak_max:.1f}MB | wall {wall_total/60:.1f} min "
          f"({len(tasks)} cells, 2 workers)", flush=True)
    return results


def cmd_pilot():
    tasks = []
    for s in (0, 1, 2):
        for arm in ("EI", "Exploit"):
            tasks += [("Branin", arm, s), ("logGSobol", arm, s)]
    for s in (0, 1):
        for arm in ("EI", "Exploit"):
            tasks += [("Ackley8", arm, s)]
    results = run_tasks(tasks, "exp_10 pilot")
    ei = [r["wall_s"] for r in results if r["arm"] == "EI"]
    ex_ = [r["wall_s"] for r in results if r["arm"] == "Exploit"]
    print(f"EI mean {sum(ei)/len(ei):.2f}s | Exploit mean {sum(ex_)/len(ex_):.2f}s | "
          f"EI/Exploit wall ratio {(sum(ei)/len(ei))/(sum(ex_)/len(ex_)):.2f}x", flush=True)


def cmd_sweep():
    # 8 arms x raw GSobol (d=10, De Ath injection) x 2 seeds -> per-arm wall profile
    # for the arm-interleave redesign; also exercises the injection path on all 8 arms.
    tasks = [("GSobol", arm, s) for s in (0, 1) for arm in ARMS]
    results = run_tasks(tasks, "exp_10 per-arm sweep (raw GSobol d=10, De Ath injection)")
    by_arm = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r["wall_s"])
    rows = sorted(((a, sum(w) / len(w)) for a, w in by_arm.items()), key=lambda x: -x[1])
    fastest = min(mw for _, mw in rows)
    print("\n=== per-arm mean wall on raw GSobol d=10 (slow -> fast; drives arm-interleave) ===",
          flush=True)
    for a, mw in rows:
        tier = "SLOW" if mw > 1.5 * fastest else "fast"
        print(f"  {a:8s} {mw:7.2f}s  ({mw / fastest:4.1f}x fastest)  [{tier}]", flush=True)


def main():
    p = argparse.ArgumentParser(description="exp_10 first-wave pilot / per-arm sweep")
    p.add_argument("--sweep", action="store_true",
                   help="8-arm per-arm wall sweep on raw GSobol")
    p.add_argument("--stability", action="store_true",
                   help="raw-Rosenbrock GP-fit numerical-stability spot-check (warnings captured)")
    args = p.parse_args()
    if args.stability:
        cmd_stability()
    elif args.sweep:
        cmd_sweep()
    else:
        cmd_pilot()


if __name__ == "__main__":
    main()
