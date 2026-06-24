#!/bin/bash
# CARL_M sweep: a single sweep on cartpole, agents share the default GPU.

SCRIPT="search/files/search_carl_m.py"
PROJECT_PREFIX="CARL-M-SWEEP"
ALGO="carl_m"
AGENTS_PER_GPU=7
TASKS=(cartpole)
GPUS=()

source "$(dirname "$0")/sweep_runner.sh"
