#!/bin/bash
# =============================================================
# Cartpole ablation: C3M (joint, tanh) + CARL_M (-||e||^2_M reward)
# across multiple discount factors on a single GPU.
#
# C3M:
#   - No pretrain (joint training)
#   - CMG activation: tanh (default)
#   - Actor activation: tanh (default)
#
# CARL_M:
#   - Reward: -||e||^2_M  (Mahalanobis tracking error, hardcoded in class)
#   - CMG activation: tanh (default)
#   - Actor activation: tanh (default)
#   - Gamma: 0.3, 0.6, 0.9, 0.99 (one group each)
#
# All runs are sequential (one GPU).
# =============================================================

GPU=0
NUM_RUNS=5   # seeds per condition
PROJECT="CARTPOLE-ABLATION"

echo "=========================================================="
echo "   Cartpole Ablation  |  GPU $GPU  |  $NUM_RUNS seeds/cond"
echo "=========================================================="
echo ""

# ----------------------------------------------------------
# 1. C3M  —  joint training, tanh CMG, tanh actor, no pretrain
# ----------------------------------------------------------
echo ">>> [1/5] C3M | joint | tanh"
CUDA_VISIBLE_DEVICES=$GPU python3 main.py \
    --task cartpole \
    --algo-name c3m \
    --project "$PROJECT" \
    --group cartpole-c3m-joint \
    --cmg-activation tanh \
    --actor-activation tanh \
    --num-runs $NUM_RUNS \
    --gpu-idx $GPU \
    > log_cartpole_c3m_joint.txt 2>&1
echo "    done. log -> log_cartpole_c3m_joint.txt"
echo ""

# ----------------------------------------------------------
# 2. CARL_M  —  -||e||^2_M reward, γ=0.3
# ----------------------------------------------------------
echo ">>> [2/5] CARL_M | -||e||^2_M | gamma=0.3"
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
    > log_cartpole_carl_m_g0.3.txt 2>&1
echo "    done. log -> log_cartpole_carl_m_g0.3.txt"
echo ""

# ----------------------------------------------------------
# 3. CARL_M  —  -||e||^2_M reward, γ=0.6
# ----------------------------------------------------------
echo ">>> [3/5] CARL_M | -||e||^2_M | gamma=0.6"
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
    > log_cartpole_carl_m_g0.6.txt 2>&1
echo "    done. log -> log_cartpole_carl_m_g0.6.txt"
echo ""

# ----------------------------------------------------------
# 4. CARL_M  —  -||e||^2_M reward, γ=0.9
# ----------------------------------------------------------
echo ">>> [4/5] CARL_M | -||e||^2_M | gamma=0.9"
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
    > log_cartpole_carl_m_g0.9.txt 2>&1
echo "    done. log -> log_cartpole_carl_m_g0.9.txt"
echo ""

# ----------------------------------------------------------
# 5. CARL_M  —  -||e||^2_M reward, γ=0.99
# ----------------------------------------------------------
echo ">>> [5/5] CARL_M | -||e||^2_M | gamma=0.99"
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
    > log_cartpole_carl_m_g0.99.txt 2>&1
echo "    done. log -> log_cartpole_carl_m_g0.99.txt"
echo ""

echo "=========================================================="
echo " All 5 conditions finished."
echo "   C3M         : log_cartpole_c3m_joint.txt"
echo "   CARL_M γ=0.30 : log_cartpole_carl_m_g0.3.txt"
echo "   CARL_M γ=0.60 : log_cartpole_carl_m_g0.6.txt"
echo "   CARL_M γ=0.90 : log_cartpole_carl_m_g0.9.txt"
echo "   CARL_M γ=0.99 : log_cartpole_carl_m_g0.99.txt"
echo "=========================================================="
