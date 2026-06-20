#!/bin/bash

AGENTS=5
GPU=0

echo "=========================================="
echo "    C3M Sweep — Turtlebot (GPU $GPU)      "
echo "=========================================="

TURTLE_OUT=$(python3 search_c3m.py --count 0 --project C3M-SWEEP-TURTLEBOT 2>&1)
SWEEP_ID=$(echo "$TURTLE_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to create sweep. Output:"
    echo "$TURTLE_OUT"
    exit 1
fi
echo "Sweep created: $SWEEP_ID"

echo "Launching $AGENTS agents on GPU $GPU..."
for ((i=1; i<=AGENTS; i++)); do
    CUDA_VISIBLE_DEVICES=$GPU python3 search_c3m.py \
        --sweep_id $SWEEP_ID \
        --project C3M-SWEEP-TURTLEBOT \
        --task turtlebot \
        > log_sweep_c3m_turtlebot_$i.txt 2>&1 &
done

echo "Agents running in background. Logs: log_sweep_c3m_turtlebot_X.txt"
