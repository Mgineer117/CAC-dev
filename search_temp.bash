#!/bin/bash

# Number of agents to run per GPU
AGENTS_PER_GPU=5

echo "=========================================================="
echo "    Launching TEMP Hyperparameter Sweeps in Background    "
echo "=========================================================="

# ----------------------------------------------------------------------
# 1. CARTPOLE (GPU 0)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'cartpole'..."
CARTPOLE_OUT=$(python3 search_temp.py --count 0 --project TEMP-SWEEP-CARTPOLE --task cartpole --policy sac 2>&1)
SWEEP_ID_CARTPOLE=$(echo "$CARTPOLE_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_CARTPOLE" ]; then
    echo "❌ Failed to create sweep for CARTPOLE. Output:"
    echo "$CARTPOLE_OUT"
    exit 1
fi
echo "✅ CARTPOLE Sweep created with ID: $SWEEP_ID_CARTPOLE"

echo "🚀 Launching $AGENTS_PER_GPU agents for CARTPOLE on GPU 0..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=0 python3 search_temp.py --sweep_id $SWEEP_ID_CARTPOLE --project TEMP-SWEEP-CARTPOLE --task cartpole --policy sac > log_sweep_temp_cartpole_gpu0_$i.txt 2>&1 &
done
echo ""

# ----------------------------------------------------------------------
# 2. SEGWAY (GPU 1)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'segway'..."
SEGWAY_OUT=$(python3 search_temp.py --count 0 --project TEMP-SWEEP-SEGWAY --task segway --policy sac 2>&1)
SWEEP_ID_SEGWAY=$(echo "$SEGWAY_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_SEGWAY" ]; then
    echo "❌ Failed to create sweep for SEGWAY. Output:"
    echo "$SEGWAY_OUT"
    exit 1
fi
echo "✅ SEGWAY Sweep created with ID: $SWEEP_ID_SEGWAY"

echo "🚀 Launching $AGENTS_PER_GPU agents for SEGWAY on GPU 1..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=1 python3 search_temp.py --sweep_id $SWEEP_ID_SEGWAY --project TEMP-SWEEP-SEGWAY --task segway --policy sac > log_sweep_temp_segway_gpu1_$i.txt 2>&1 &
done
echo ""

# ----------------------------------------------------------------------
# 3. CAR (GPU 2)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'car'..."
CAR_OUT=$(python3 search_temp.py --count 0 --project TEMP-SWEEP-CAR --task car --policy sac 2>&1)
SWEEP_ID_CAR=$(echo "$CAR_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_CAR" ]; then
    echo "❌ Failed to create sweep for CAR. Output:"
    echo "$CAR_OUT"
    exit 1
fi
echo "✅ CAR Sweep created with ID: $SWEEP_ID_CAR"

echo "🚀 Launching $AGENTS_PER_GPU agents for CAR on GPU 2..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=2 python3 search_temp.py --sweep_id $SWEEP_ID_CAR --project TEMP-SWEEP-CAR --task car --policy sac > log_sweep_temp_car_gpu2_$i.txt 2>&1 &
done
echo ""

echo "=========================================================="
echo "🎉 All agents successfully deployed to the background! "
echo "   - CARTPOLE logs : log_sweep_temp_cartpole_gpu0_X.txt"
echo "   - SEGWAY logs   : log_sweep_temp_segway_gpu1_X.txt"
echo "   - CAR logs      : log_sweep_temp_car_gpu2_X.txt"
echo "=========================================================="
