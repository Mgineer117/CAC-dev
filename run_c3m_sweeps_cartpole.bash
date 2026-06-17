#!/bin/bash

# Number of agents to run per GPU (5 agents * 1 GPU = 5 agents per task)
AGENTS_PER_GPU=10

echo "=========================================================="
echo " Launching C3M Hyperparameter Sweeps for Cartpole in Background "
echo "=========================================================="

# ----------------------------------------------------------------------
# 1. CARTPOLE (GPU 0)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'cartpole'..."
# Create the sweep and capture the ID
CARTPOLE_OUT=$(python3 search_c3m.py --count 0 --project C3M-SWEEP-CARTPOLE 2>&1)
SWEEP_ID_CARTPOLE=$(echo "$CARTPOLE_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_CARTPOLE" ]; then
    echo "❌ Failed to create sweep for CARTPOLE. Output:"
    echo "$CARTPOLE_OUT"
    exit 1
fi
echo "✅ CARTPOLE Sweep created with ID: $SWEEP_ID_CARTPOLE"

echo "🚀 Launching $AGENTS_PER_GPU agents for CARTPOLE on GPU 0..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=0 python3 search_c3m.py --sweep_id $SWEEP_ID_CARTPOLE --project C3M-SWEEP-CARTPOLE --task cartpole > log_sweep_cartpole_gpu0_$i.txt 2>&1 &
done

echo ""
echo "=========================================================="
echo "🎉 All $AGENTS_PER_GPU agents successfully deployed to the background! "
echo "   - CARTPOLE logs   : log_sweep_cartpole_gpuX_Y.txt"
echo "=========================================================="
