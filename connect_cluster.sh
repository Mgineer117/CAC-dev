#!/bin/bash
# ---------------------------------------------------------------------------
# Connect to the campus cluster and drop straight into an interactive GPU
# shell (srun --pty bash) on a compute node.
#
#   ./connect_cluster.sh
#
# `ssh -t` forces a pseudo-terminal so the interactive srun shell works; the
# srun command runs on the login node and lands you on the compute node. When
# you exit the compute-node shell, srun + the ssh session close.
#
# Everything is overridable from the environment, e.g.:
#   TIME=08:00:00 MEM=32G GPUS=2 ./connect_cluster.sh
#   PARTITION=ic-express ./connect_cluster.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# --- connection ---
USER_HOST=${USER_HOST:-minjae5@cc-login.campuscluster.illinois.edu}

# --- srun resource request (defaults match the ic-express interactive recipe) ---
PARTITION=${PARTITION:-ic-express}
ACCOUNT=${ACCOUNT:-huytran1-ic}
NODES=${NODES:-1}
GPUS=${GPUS:-1}
# CPUs for the (single) interactive task. Without this srun defaults to 1 CPU,
# which starves the multi-worker sampler. Override with CPUS=8 ./connect...
CPUS=${CPUS:-4}
MEM=${MEM:-16G}
TIME=${TIME:-04:00:00}

# Absolute path to the repo on the cluster to land in on the compute node.
REPO_DIR=${REPO_DIR:-/projects/illinois/eng/aero/huytran1/mjcho/research/asdf}

# Remote script (heredoc): local ${VARS} are substituted now; \$REMOTE vars are
# escaped so they evaluate on the cluster. It prints only the estimated time to
# allocation (via srun --test-only), then launches srun and, on the allocated
# node, a one-line cpu/mem/gpu summary before handing off to interactive bash.
REMOTE_CMD=$(cat <<EOF
# ---- estimated time-to-allocation for THIS request ----
TEST=\$(srun --partition=${PARTITION} --account=${ACCOUNT} --nodes=${NODES} --gres=gpu:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} --test-only 2>&1 || true)
START=\$(printf '%s' "\$TEST" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
NOW_S=\$(date +%s); ST_S=0
[ -n "\$START" ] && ST_S=\$(date -d "\$START" +%s 2>/dev/null || echo 0)
REM=\$(( ST_S - NOW_S ))
if [ "\$ST_S" -gt 0 ] && [ "\$REM" -gt 60 ]; then
  H=\$(( REM/3600 )); M=\$(( (REM%3600)/60 ))
  [ "\$H" -gt 0 ] && ETA="\${H}h \${M}m" || ETA="\${M}m"
  printf '\033[1;33mAllocation in ~%s (queued)\033[0m\n' "\$ETA"
else
  printf '\033[1;32mAllocation available now\033[0m\n'
fi

exec srun --partition=${PARTITION} --account=${ACCOUNT} --nodes=${NODES} --gres=gpu:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} --pty bash -c '
echo
echo "==== allocated machine ===="
echo "cpus   : \${SLURM_CPUS_ON_NODE:-?} on node    mem(GB): \$(( \${SLURM_MEM_PER_NODE:-0} / 1024 ))"
GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader -i "\${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null | head -1)
[ -z "\$GPU_NAME" ] && GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "gpus   : \${GPU_NAME:-?}"
cd ${REPO_DIR} 2>/dev/null || echo "warning: could not cd to ${REPO_DIR}"
echo "cwd    : \$(pwd)"
echo
exec bash'
EOF
)

# -t : force a TTY so the interactive srun shell behaves like a terminal.
exec ssh -t "${USER_HOST}" "${REMOTE_CMD}"
