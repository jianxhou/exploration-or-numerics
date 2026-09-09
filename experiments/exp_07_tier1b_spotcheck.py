"""
Experiment 07: Tier 1B spot-check — our Docker re-execution vs published runs.

Compares the re-executed De Ath et al. runs (experiments/run_tier1b_docker.sh,
outputs in ~/projects/egreedy_tier1b) against the published results_paper npz
files. Run numbers 1..11 use the same published initial designs, so per-run
differences are paired. Regret is computed exactly as in exp_05: cumulative
min of the per-evaluation |y - yopt|, yopt from their synthetic_problems.py.

Per (problem, method):
  (a) paired log-regret differences at T=250 for runs 1..11 (ours - theirs),
      median delta and an exact sign test;
  (b) two-sample Mann-Whitney U: our 11 final regrets vs their full 51;
  (c) summary table.

Usage:
    python experiments/exp_07_tier1b_spotcheck.py

Outputs:
    results/exp_07_tier1b.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

DATA_DIR = Path.home() / "projects" / "egreedy"
OURS_DIR = Path.home() / "projects" / "egreedy_tier1b"

PROBLEMS = ["Branin", "logGSobol"]
METHODS = ["EI", "eRandom", "eFront", "Exploit"]
METHOD_FILE = {"EI": "EI", "eRandom": "eRandom_eps0.1",
               "eFront": "eFront_eps0.1", "Exploit": "Exploit"}
RUNS_OURS = range(1, 12)
RUNS_THEIRS = range(1, 52)
T = 250
LOG_GUARD = 1e-300


def load_fopt() -> dict:
    src = DATA_DIR / "egreedy" / "test_problems" / "synthetic_problems.py"
    spec = importlib.util.spec_from_file_location("egreedy_synthetic", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {p: float(np.squeeze(getattr(module, p)().yopt)) for p in PROBLEMS}


def final_regret(path: Path, fopt: float) -> float:
    y = np.load(path)["Ytr"].ravel()
    if y.shape[0] != T:
        raise ValueError(f"{path.name}: expected {T} rows, got {y.shape[0]}")
    return float(np.minimum.accumulate(np.abs(y - fopt))[T - 1])


def main():
    if not OURS_DIR.is_dir():
        raise SystemExit(f"Missing {OURS_DIR}; run experiments/run_tier1b_docker.sh first.")
    fopt = load_fopt()

    comparisons = {}
    print(f"{'problem':<10} {'method':<8} {'med ours':>10} {'med theirs':>10} "
          f"{'med dlog':>9} {'sign p':>8} {'MW p':>8}")
    for prob in PROBLEMS:
        for meth in METHODS:
            ours = np.array([
                final_regret(OURS_DIR / f"{prob}_{r}_{T}_{METHOD_FILE[meth]}.npz",
                             fopt[prob])
                for r in RUNS_OURS
            ])
            theirs_all = np.array([
                final_regret(
                    DATA_DIR / "results_paper" / f"{prob}_{r}_{T}_{METHOD_FILE[meth]}.npz",
                    fopt[prob])
                for r in RUNS_THEIRS
            ])
            theirs_paired = theirs_all[:len(ours)]  # runs 1..11, same initial designs

            dlog = (np.log(np.maximum(ours, LOG_GUARD))
                    - np.log(np.maximum(theirs_paired, LOG_GUARD)))
            n_pos = int((dlog > 0).sum())
            n_neg = int((dlog < 0).sum())
            sign_p = (stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue
                      if (n_pos + n_neg) > 0 else 1.0)
            mw = stats.mannwhitneyu(ours, theirs_all, alternative="two-sided")

            rec = {
                "n_ours": len(ours),
                "n_theirs": len(theirs_all),
                "final_regret_ours": [float(x) for x in ours],
                "final_regret_theirs_paired": [float(x) for x in theirs_paired],
                "median_ours": float(np.median(ours)),
                "median_theirs_runs1_11": float(np.median(theirs_paired)),
                "median_theirs_all51": float(np.median(theirs_all)),
                "paired_dlog_median": float(np.median(dlog)),
                "paired_dlog": [float(x) for x in dlog],
                "sign_test": {"n_pos": n_pos, "n_neg": n_neg,
                              "n_zero": int((dlog == 0).sum()),
                              "p_value": float(sign_p)},
                "mannwhitney_11_vs_51": {"U": float(mw.statistic),
                                         "p_value": float(mw.pvalue)},
            }
            comparisons[f"{prob}_{meth}"] = rec
            print(f"{prob:<10} {meth:<8} {rec['median_ours']:>10.3e} "
                  f"{rec['median_theirs_all51']:>10.3e} "
                  f"{rec['paired_dlog_median']:>9.3f} {sign_p:>8.3f} "
                  f"{mw.pvalue:>8.3f}")

    out = {
        "config": {
            "problems": PROBLEMS,
            "methods": METHODS,
            "runs_ours": list(RUNS_OURS),
            "T": T,
            "f_opt": fopt,
            "regret_definition": "cummin(|y - yopt|), as exp_05 / their plotting.py",
            "ours_dir": str(OURS_DIR),
            "environment": "georgedeath/egreedy Docker image, linux/amd64 under "
                           "qemu on Apple Silicon, entrypoint overridden (see "
                           "literature/death_supplementary_audit.md)",
        },
        "comparisons": comparisons,
    }
    out_path = Path(__file__).parent.parent / "results" / "exp_07_tier1b.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
