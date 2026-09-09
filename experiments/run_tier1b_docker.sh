#!/bin/bash
# Tier 1B spot-check orchestrator: re-execute De Ath et al. runs in their
# Docker image. Outputs go to ~/projects/egreedy_tier1b (never into the
# clone's results_paper). Per-run resume is native to their CLI
# (continue_runs=True; the npz is saved every iteration), so a stopped
# container leaves a resumable partial file. Complete outputs (250 rows)
# are skipped host-side.
#
# Usage:
#   run_tier1b_docker.sh gate            # the two timing-gate runs
#   run_tier1b_docker.sh batch [NPAR]    # remaining matrix, NPAR workers (default 2)
set -u

IMAGE=georgedeath/egreedy
OUT="$HOME/projects/egreedy_tier1b"
PY=/opt/anaconda3/envs/oas-test/bin/python
RUN_LIMIT_S=7200   # hard per-run cap, enforced via docker stop
BUDGET=250

PROBLEMS=(Branin logGSobol)
# method:acquisition-args (empty = none); file suffix handled by their code
METHODS=("EI:" "eRandom:epsilon:0.1" "eFront:epsilon:0.1" "Exploit:")
RUNS=$(seq 1 11)

mkdir -p "$OUT"

outfile() {  # problem run method -> expected npz name
    local p=$1 r=$2 m=$3
    case "$m" in
        eRandom|eFront) echo "${p}_${r}_${BUDGET}_${m}_eps0.1.npz" ;;
        *)              echo "${p}_${r}_${BUDGET}_${m}.npz" ;;
    esac
}

is_complete() {  # path -> 0 if npz exists with 250 Ytr rows
    [ -f "$1" ] || return 1
    "$PY" -c "
import sys, numpy as np
try:
    d = np.load(sys.argv[1])
    sys.exit(0 if d['Ytr'].shape[0] == $BUDGET else 1)
except Exception:
    sys.exit(1)
" "$1"
}

run_one() {  # problem run method args
    local p=$1 r=$2 m=$3 aa=$4
    local of; of=$(outfile "$p" "$r" "$m")
    if is_complete "$OUT/$of"; then
        echo "[skip] $of complete"
        return 0
    fi
    # An unreadable npz (kill mid-savez) would crash their in-container resume;
    # delete it so the run restarts cleanly.
    if [ -f "$OUT/$of" ] && ! "$PY" -c "
import sys, numpy as np
np.load(sys.argv[1])['Ytr']
" "$OUT/$of" 2>/dev/null; then
        echo "[reset] $of unreadable, deleting partial"
        rm -f "$OUT/$of"
    fi
    local cmd="cd /egreedy && rm -rf results && ln -s /out results && python run_experiment.py -p $p -b $BUDGET -r $r -a $m"
    [ -n "$aa" ] && cmd="$cmd -aa $aa"
    local name="tier1b_${p}_${r}_${m}"
    local t0=$SECONDS
    docker rm -f "$name" >/dev/null 2>&1
    docker run -d --platform linux/amd64 --name "$name" \
        -v "$OUT:/out" --entrypoint bash "$IMAGE" -lc "$cmd" >/dev/null
    # Watchdog targets the container, not a host process.
    ( sleep "$RUN_LIMIT_S"; docker stop -t 10 "$name" >/dev/null 2>&1 ) &
    local wpid=$!
    local code; code=$(docker wait "$name")
    kill "$wpid" >/dev/null 2>&1; wait "$wpid" 2>/dev/null
    docker rm "$name" >/dev/null 2>&1
    echo "[done] $of exit=$code wall=$((SECONDS - t0))s"
}

worker() {  # worker_idx n_workers
    local idx=0
    for p in "${PROBLEMS[@]}"; do
        for ms in "${METHODS[@]}"; do
            local m=${ms%%:*} aa=${ms#*:}
            for r in $RUNS; do
                if [ $((idx % $2)) -eq "$1" ]; then
                    run_one "$p" "$r" "$m" "$aa"
                fi
                idx=$((idx + 1))
            done
        done
    done
}

case "${1:-}" in
    gate)
        run_one Branin 1 EI ""
        run_one logGSobol 1 EI ""
        ;;
    batch)
        npar=${2:-2}
        for w in $(seq 0 $((npar - 1))); do
            worker "$w" "$npar" &
        done
        wait
        n=$(ls "$OUT"/*.npz 2>/dev/null | wc -l | tr -d ' ')
        echo "BATCH DONE: $n npz files in $OUT"
        ;;
    *)
        echo "usage: $0 gate | batch [NPAR]" >&2
        exit 1
        ;;
esac
