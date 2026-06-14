#!/bin/bash

# Number of agents to run per GPU (5 agents * 2 GPUs = 10 agents per task)
AGENTS_PER_GPU=5

echo "=========================================================="
echo "    Launching C3M Hyperparameter Sweeps in Background     "
echo "=========================================================="

# ----------------------------------------------------------------------
# 1. CAR (GPUs 0, 1)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'car'..."
# Create the sweep and capture the ID
CAR_OUT=$(python3 search_c3m.py --count 0 --project C3M-SWEEP-CAR 2>&1)
SWEEP_ID_CAR=$(echo "$CAR_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_CAR" ]; then
    echo "❌ Failed to create sweep for CAR. Output:"
    echo "$CAR_OUT"
    exit 1
fi
echo "✅ CAR Sweep created with ID: $SWEEP_ID_CAR"

echo "🚀 Launching 10 agents for CAR (5 on GPU 0, 5 on GPU 1)..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=0 python3 search_c3m.py --sweep_id $SWEEP_ID_CAR --project C3M-SWEEP-CAR --task car > log_sweep_car_gpu0_$i.txt 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python3 search_c3m.py --sweep_id $SWEEP_ID_CAR --project C3M-SWEEP-CAR --task car > log_sweep_car_gpu1_$i.txt 2>&1 &
done


# ----------------------------------------------------------------------
# 2. PVTOL (GPUs 2, 3)
# ----------------------------------------------------------------------
echo ""
echo "Initializing WandB Sweep for 'pvtol'..."
# Create the sweep and capture the ID
PVTOL_OUT=$(python3 search_c3m.py --count 0 --project C3M-SWEEP-PVTOL 2>&1)
SWEEP_ID_PVTOL=$(echo "$PVTOL_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_PVTOL" ]; then
    echo "❌ Failed to create sweep for PVTOL. Output:"
    echo "$PVTOL_OUT"
    exit 1
fi
echo "✅ PVTOL Sweep created with ID: $SWEEP_ID_PVTOL"

echo "🚀 Launching 10 agents for PVTOL (5 on GPU 2, 5 on GPU 3)..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=2 python3 search_c3m.py --sweep_id $SWEEP_ID_PVTOL --project C3M-SWEEP-PVTOL --task pvtol > log_sweep_pvtol_gpu2_$i.txt 2>&1 &
    CUDA_VISIBLE_DEVICES=3 python3 search_c3m.py --sweep_id $SWEEP_ID_PVTOL --project C3M-SWEEP-PVTOL --task pvtol > log_sweep_pvtol_gpu3_$i.txt 2>&1 &
done

echo ""
echo "=========================================================="
echo "🎉 All 20 agents successfully deployed to the background! "
echo "   - CAR logs   : log_sweep_car_gpuX_Y.txt"
echo "   - PVTOL logs : log_sweep_pvtol_gpuX_Y.txt"
echo "=========================================================="
