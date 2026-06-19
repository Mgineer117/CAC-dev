#!/bin/bash

# Number of agents to run per GPU
AGENTS_PER_GPU=10

echo "=========================================================="
echo "    Launching CARL Hyperparameter Sweeps in Background    "
echo "=========================================================="

# ----------------------------------------------------------------------
# 1. CARTPOLE (GPU 0)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'cartpole'..."
CARTPOLE_OUT=$(python3 search_carl.py --count 0 --project CARL-SWEEP-CARTPOLE 2>&1)
SWEEP_ID_CARTPOLE=$(echo "$CARTPOLE_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_CARTPOLE" ]; then
    echo "❌ Failed to create sweep for CARTPOLE. Output:"
    echo "$CARTPOLE_OUT"
    exit 1
fi
echo "✅ CARTPOLE Sweep created with ID: $SWEEP_ID_CARTPOLE"

echo "🚀 Launching $AGENTS_PER_GPU agents for CARTPOLE on GPU 0..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=0 python3 search_carl.py --sweep_id $SWEEP_ID_CARTPOLE --project CARL-SWEEP-CARTPOLE --task cartpole > log_sweep_carl_cartpole_gpu0_$i.txt 2>&1 &
done
echo ""

# ----------------------------------------------------------------------
# 2. SEGWAY (GPU 1)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'segway'..."
SEGWAY_OUT=$(python3 search_carl.py --count 0 --project CARL-SWEEP-SEGWAY 2>&1)
SWEEP_ID_SEGWAY=$(echo "$SEGWAY_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_SEGWAY" ]; then
    echo "❌ Failed to create sweep for SEGWAY. Output:"
    echo "$SEGWAY_OUT"
    exit 1
fi
echo "✅ SEGWAY Sweep created with ID: $SWEEP_ID_SEGWAY"

echo "🚀 Launching $AGENTS_PER_GPU agents for SEGWAY on GPU 1..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=1 python3 search_carl.py --sweep_id $SWEEP_ID_SEGWAY --project CARL-SWEEP-SEGWAY --task segway > log_sweep_carl_segway_gpu1_$i.txt 2>&1 &
done
echo ""

# ----------------------------------------------------------------------
# 3. CAR (GPU 2)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'car'..."
CAR_OUT=$(python3 search_carl.py --count 0 --project CARL-SWEEP-CAR 2>&1)
SWEEP_ID_CAR=$(echo "$CAR_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_CAR" ]; then
    echo "❌ Failed to create sweep for CAR. Output:"
    echo "$CAR_OUT"
    exit 1
fi
echo "✅ CAR Sweep created with ID: $SWEEP_ID_CAR"

echo "🚀 Launching $AGENTS_PER_GPU agents for CAR on GPU 2..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=2 python3 search_carl.py --sweep_id $SWEEP_ID_CAR --project CARL-SWEEP-CAR --task car > log_sweep_carl_car_gpu2_$i.txt 2>&1 &
done
echo ""

# ----------------------------------------------------------------------
# 4. TURTLEBOT (GPU 3)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'turtlebot'..."
TURTLEBOT_OUT=$(python3 search_carl.py --count 0 --project CARL-SWEEP-TURTLEBOT 2>&1)
SWEEP_ID_TURTLEBOT=$(echo "$TURTLEBOT_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_TURTLEBOT" ]; then
    echo "❌ Failed to create sweep for TURTLEBOT. Output:"
    echo "$TURTLEBOT_OUT"
    exit 1
fi
echo "✅ TURTLEBOT Sweep created with ID: $SWEEP_ID_TURTLEBOT"

echo "🚀 Launching $AGENTS_PER_GPU agents for TURTLEBOT on GPU 3..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=3 python3 search_carl.py --sweep_id $SWEEP_ID_TURTLEBOT --project CARL-SWEEP-TURTLEBOT --task turtlebot > log_sweep_carl_turtlebot_gpu3_$i.txt 2>&1 &
done
echo ""

echo "=========================================================="
echo "🎉 All agents successfully deployed to the background! "
echo "   - CARTPOLE logs  : log_sweep_carl_cartpole_gpu0_X.txt"
echo "   - SEGWAY logs    : log_sweep_carl_segway_gpu1_X.txt"
echo "   - CAR logs       : log_sweep_carl_car_gpu2_X.txt"
echo "   - TURTLEBOT logs : log_sweep_carl_turtlebot_gpu3_X.txt"
echo "=========================================================="
