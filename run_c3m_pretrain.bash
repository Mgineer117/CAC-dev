#!/bin/bash
#
# C3M + CMG pretraining sweep — one GPU, sequential.
# Runs 10 jobs per environment, one env at a time.
# Flag: --c3m-pretrain-cmg

AGENTS=5
GPU=0
SCRIPT=search_c3m_pretrain.py

echo "=================================================="
echo "   C3M Pretrain Sweep  (GPU $GPU, $AGENTS jobs/env)"
echo "=================================================="

run_env() {
    local TASK=$1
    local PROJECT="C3M-PRETRAIN-$(echo $TASK | tr '[:lower:]' '[:upper:]')"

    echo ""
    echo "--------------------------------------------------"
    echo "  ENV: $TASK   project: $PROJECT"
    echo "--------------------------------------------------"

    # Create sweep (count=0 → exit after registering, no agents)
    local OUT
    OUT=$(python3 $SCRIPT --count 0 --project "$PROJECT" --task "$TASK" --c3m-pretrain-cmg 2>&1)
    local SWEEP_ID
    SWEEP_ID=$(echo "$OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

    if [ -z "$SWEEP_ID" ]; then
        echo "  [ERROR] Failed to create sweep for $TASK. Output:"
        echo "$OUT"
        return 1
    fi
    echo "  Sweep created: $SWEEP_ID"

    # Run agents sequentially (no &)
    for ((i=1; i<=AGENTS; i++)); do
        echo "  Running agent $i / $AGENTS ..."
        CUDA_VISIBLE_DEVICES=$GPU python3 $SCRIPT \
            --sweep_id "$SWEEP_ID" \
            --project  "$PROJECT" \
            --task     "$TASK"    \
            --c3m-pretrain-cmg \
            > "log_c3m_pretrain_${TASK}_${i}.txt" 2>&1
        echo "  Agent $i done."
    done

    echo "  All $AGENTS agents finished for $TASK."
}

# ── Environments — runs sequentially ──────────────────────────────────────
run_env cartpole
run_env segway
run_env car
run_env turtlebot

echo ""
echo "=================================================="
echo "  All environments complete."
echo "  Logs: log_c3m_pretrain_<env>_<n>.txt"
echo "=================================================="
