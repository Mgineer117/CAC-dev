#!/bin/bash

# Number of agents to run per GPU
AGENTS_PER_GPU=10

echo "=========================================================="
echo "  Launching C3M Hyperparameter Sweeps: Segway & Turtlebot "
echo "=========================================================="

# ----------------------------------------------------------------------
# 1. SEGWAY (GPU 0)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'segway'..."
SEGWAY_OUT=$(python3 search_c3m.py --count 0 --project C3M-SWEEP-SEGWAY --task segway 2>&1)
SWEEP_ID_SEGWAY=$(echo "$SEGWAY_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_SEGWAY" ]; then
    echo "Failed to create sweep for SEGWAY. Output:"
    echo "$SEGWAY_OUT"
    exit 1
fi
echo "SEGWAY Sweep created with ID: $SWEEP_ID_SEGWAY"

echo "Launching $AGENTS_PER_GPU agents for SEGWAY on GPU 0..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=0 python3 search_c3m.py \
        --sweep_id $SWEEP_ID_SEGWAY \
        --project C3M-SWEEP-SEGWAY \
        --task segway \
        > log_sweep_c3m_segway_gpu0_$i.txt 2>&1 &
done
echo ""

# ----------------------------------------------------------------------
# 2. TURTLEBOT (GPU 1)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'turtlebot'..."
TURTLEBOT_OUT=$(python3 search_c3m.py --count 0 --project C3M-SWEEP-TURTLEBOT --task turtlebot 2>&1)
SWEEP_ID_TURTLEBOT=$(echo "$TURTLEBOT_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_TURTLEBOT" ]; then
    echo "Failed to create sweep for TURTLEBOT. Output:"
    echo "$TURTLEBOT_OUT"
    exit 1
fi
echo "TURTLEBOT Sweep created with ID: $SWEEP_ID_TURTLEBOT"

echo "Launching $AGENTS_PER_GPU agents for TURTLEBOT on GPU 1..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=1 python3 search_c3m.py \
        --sweep_id $SWEEP_ID_TURTLEBOT \
        --project C3M-SWEEP-TURTLEBOT \
        --task turtlebot \
        > log_sweep_c3m_turtlebot_gpu1_$i.txt 2>&1 &
done
echo ""

echo "=========================================================="
echo "All agents deployed."
echo "  SEGWAY logs   : log_sweep_c3m_segway_gpu0_X.txt"
echo "  TURTLEBOT logs: log_sweep_c3m_turtlebot_gpu1_X.txt"
echo "=========================================================="
