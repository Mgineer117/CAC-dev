#!/bin/bash
# TEMP sweeps: one per task, across 3 GPUs. The optimal policy is fixed per
# sweep via --policy (sac or ppo) so the search stays in one parameter space.

SCRIPT="search/files/search_temp.py"
PROJECT_PREFIX="TEMP-SWEEP"
ALGO="temp"
AGENTS_PER_GPU=5
TASKS=(cartpole segway car)
GPUS=(0)
EXTRA_ARGS="--policy sac"

source "$(dirname "$0")/sweep_runner.sh"


# SCRIPT="search/files/search_temp.py"
# PROJECT_PREFIX="TEMP-SWEEP"
# ALGO="temp"
# AGENTS_PER_GPU=10
# TASKS=(cartpole segway car)
# GPUS=(0 1 2)
# EXTRA_ARGS="--policy ppo"

# source "$(dirname "$0")/sweep_runner.sh"
