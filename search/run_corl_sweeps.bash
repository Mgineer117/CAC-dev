#!/bin/bash
# CORL sweeps: one per task, across 4 GPUs.

SCRIPT="search/files/search_corl.py"
PROJECT_PREFIX="CORL-SWEEP"
ALGO="corl"
AGENTS_PER_GPU=10
TASKS=(cartpole segway car turtlebot)
GPUS=(0 1 2 3)

source "$(dirname "$0")/sweep_runner.sh"
