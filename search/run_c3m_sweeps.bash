#!/bin/bash
# C3M sweeps: one per task, across 4 GPUs.

SCRIPT="search/files/search_c3m.py"
PROJECT_PREFIX="C3M-SWEEP"
ALGO="c3m"
AGENTS_PER_GPU=5
TASKS=(cartpole segway car turtlebot)
GPUS=(0 1 2 3)

source "$(dirname "$0")/sweep_runner.sh"
