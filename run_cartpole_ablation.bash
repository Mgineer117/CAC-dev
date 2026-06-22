#!/bin/bash
# =============================================================
# Cartpole ablation: C3M (joint, tanh) + CARL_M (-||e||^2_M reward)
# All 5 conditions launched simultaneously on one GPU.
# =============================================================

GPU=0
NUM_RUNS=5
PROJECT="CARTPOLE-ABLATION"

echo "=========================================================="
echo "   Cartpole Ablation  |  GPU $GPU  |  $NUM_RUNS seeds/cond"
echo "   Launching all 5 conditions in parallel..."
echo "=========================================================="
echo ""

# 1. C3M — joint training, tanh/tanh, no pretrain
CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
    --task cartpole \
    --algo-name c3m \
    --project "$PROJECT" \
    --group cartpole-c3m-joint \
    --cmg-activation tanh \
    --actor-activation tanh \
    --num-runs $NUM_RUNS \
    --gpu-idx $GPU \
    > log_cartpole_c3m_joint.txt 2>&1 &
echo "  [1/5] C3M joint         -> log_cartpole_c3m_joint.txt  (PID $!)"

# 2. CARL_M — γ=0.3
CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
    --task cartpole \
    --algo-name carl_m \
    --project "$PROJECT" \
    --group cartpole-carl_m \
    --cmg-activation tanh \
    --actor-activation tanh \
    --gamma 0.3 \
    --num-runs $NUM_RUNS \
    --gpu-idx $GPU \
    > log_cartpole_carl_m_g0.3.txt 2>&1 &
echo "  [2/5] CARL_M γ=0.30     -> log_cartpole_carl_m_g0.3.txt  (PID $!)"

# 3. CARL_M — γ=0.6
CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
    --task cartpole \
    --algo-name carl_m \
    --project "$PROJECT" \
    --group cartpole-carl_m \
    --cmg-activation tanh \
    --actor-activation tanh \
    --gamma 0.6 \
    --num-runs $NUM_RUNS \
    --gpu-idx $GPU \
    > log_cartpole_carl_m_g0.6.txt 2>&1 &
echo "  [3/5] CARL_M γ=0.60     -> log_cartpole_carl_m_g0.6.txt  (PID $!)"

# 4. CARL_M — γ=0.9
CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
    --task cartpole \
    --algo-name carl_m \
    --project "$PROJECT" \
    --group cartpole-carl_m \
    --cmg-activation tanh \
    --actor-activation tanh \
    --gamma 0.9 \
    --num-runs $NUM_RUNS \
    --gpu-idx $GPU \
    > log_cartpole_carl_m_g0.9.txt 2>&1 &
echo "  [4/5] CARL_M γ=0.90     -> log_cartpole_carl_m_g0.9.txt  (PID $!)"

# 5. CARL_M — γ=0.99
CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
    --task cartpole \
    --algo-name carl_m \
    --project "$PROJECT" \
    --group cartpole-carl_m \
    --cmg-activation tanh \
    --actor-activation tanh \
    --gamma 0.99 \
    --num-runs $NUM_RUNS \
    --gpu-idx $GPU \
    > log_cartpole_carl_m_g0.99.txt 2>&1 &
echo "  [5/5] CARL_M γ=0.99     -> log_cartpole_carl_m_g0.99.txt  (PID $!)"

echo ""
echo "All 5 launched. Waiting for completion..."
wait
echo ""
echo "=========================================================="
echo " All 5 conditions finished."
echo "   C3M           : log_cartpole_c3m_joint.txt"
echo "   CARL_M γ=0.30 : log_cartpole_carl_m_g0.3.txt"
echo "   CARL_M γ=0.60 : log_cartpole_carl_m_g0.6.txt"
echo "   CARL_M γ=0.90 : log_cartpole_carl_m_g0.9.txt"
echo "   CARL_M γ=0.99 : log_cartpole_carl_m_g0.99.txt"
echo "=========================================================="
