#!/bin/bash

# Number of agents to run per GPU (5 agents * 1 GPU = 5 agents per task)
AGENTS_PER_GPU=10

echo "=========================================================="
echo "  Launching C3M Hyperparameter Sweeps for Segway in Background  "
echo "=========================================================="

# ----------------------------------------------------------------------
# 1. SEGWAY (GPU 0)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'segway'..."
# Create the sweep and capture the ID
SEGWAY_OUT=$(python3 search_c3m.py --count 0 --project C3M-SWEEP-SEGWAY 2>&1)
SWEEP_ID_SEGWAY=$(echo "$SEGWAY_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_SEGWAY" ]; then
    echo "❌ Failed to create sweep for SEGWAY. Output:"
    echo "$SEGWAY_OUT"
    exit 1
fi
echo "✅ SEGWAY Sweep created with ID: $SWEEP_ID_SEGWAY"

echo "🚀 Launching $AGENTS_PER_GPU agents for SEGWAY on GPU 0..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=0 python3 search_c3m.py --sweep_id $SWEEP_ID_SEGWAY --project C3M-SWEEP-SEGWAY --task segway > log_sweep_segway_gpu0_$i.txt 2>&1 &
done

echo ""
echo "=========================================================="
echo "🎉 All $AGENTS_PER_GPU agents successfully deployed to the background! "
echo "   - SEGWAY logs   : log_sweep_segway_gpuX_Y.txt"
echo "=========================================================="
