#!/bin/bash
# NicheTrans training orchestrator
# Runs all training notebooks with exactly 2 concurrent jobs.
# Launch with: nohup bash train_all.sh > training_logs/orchestrator.log 2>&1 &

WORKDIR="/mnt/datadisk0/vic/NicheTrans-experiment"
LOGDIR="$WORKDIR/training_logs"
EXEC_DIR="$LOGDIR/executed_notebooks"
JUPYTER="/home/ubuntu/miniconda3/envs/NicheTrans_CellCluster/bin/jupyter"
STATUS_FILE="$LOGDIR/training_status.txt"
MAX_JOBS=2

mkdir -p "$LOGDIR" "$EXEC_DIR"

NOTEBOOKS=(
    "Tutorial_3.1__Train_NicheTrans_on_SMA_data.ipynb"
    "Tutorial_4.1__Train_NicheTrans_STAR_on_SMA_data.ipynb"
    "Tutorial_5.1__Train_NicheTrans_on_STARmap_PLUS_data.ipynb"
    "Tutorial_6.1__Train_NicheTrans_STAR__on_STARmap_PLUS_data.ipynb"
    "Tutorial_7.1__Train_NicheTrans_on_10x_Xenium_data.ipynb"
    "Tutorial_8.1__Train_NicheTrans_on_human_lymph_node_data.ipynb"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS_FILE"
}

run_notebook() {
    local nb="$1"
    local stem="${nb%.ipynb}"
    local logfile="$LOGDIR/${stem}.log"

    log "STARTED   | $nb"

    "$JUPYTER" nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output-dir "$EXEC_DIR" \
        "$WORKDIR/$nb" \
        > "$logfile" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        log "SUCCESS   | $nb"
    else
        log "FAILED($rc)| $nb  -- see $logfile"
    fi
    return $rc
}

# ── main ──────────────────────────────────────────────────────────────────────
log "=== Training started | ${#NOTEBOOKS[@]} notebooks queued | MAX_JOBS=$MAX_JOBS ==="

declare -a pids=()

for nb in "${NOTEBOOKS[@]}"; do

    # Remove finished PIDs from the tracking array
    while true; do
        alive=()
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive+=("$pid")
        done
        pids=("${alive[@]}")
        [ ${#pids[@]} -lt $MAX_JOBS ] && break
        sleep 5
    done

    run_notebook "$nb" &
    new_pid=$!
    pids+=("$new_pid")
    log "LAUNCHED  | $nb  (PID=$new_pid  running=${#pids[@]}/$MAX_JOBS)"
done

# Wait for last batch
for pid in "${pids[@]}"; do
    wait "$pid"
done

log "=== All training jobs finished ==="
