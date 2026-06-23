#!/bin/bash

# Number of agents to run
AGENTS=8

echo "=========================================================="
echo "   Launching CARL_M Hyperparameter Sweep in Background    "
echo "=========================================================="

# ----------------------------------------------------------------------
# 1. Create the sweep
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for CARL_M..."
SWEEP_OUT=$(python3 search_carl_m.py --count 0 --project CARL-M-SWEEP 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Failed to create sweep. Output:"
    echo "$SWEEP_OUT"
    exit 1
fi
echo "✅ CARL_M Sweep created with ID: $SWEEP_ID"

# ----------------------------------------------------------------------
# 2. Launch agents in the background
# ----------------------------------------------------------------------
echo "🚀 Launching $AGENTS agents..."
for ((i=1; i<=$AGENTS; i++)); do
    python3 search_carl_m.py --sweep_id $SWEEP_ID --project CARL-M-SWEEP --task cartpole > log_sweep_carl_m_$i.txt 2>&1 &
done
echo ""

echo "=========================================================="
echo "🎉 All agents successfully deployed to the background!"
echo "   - Logs: log_sweep_carl_m_X.txt"
echo "=========================================================="
